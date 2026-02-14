from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..event_stream import AssistantMessageEventStream
from ..types import Context, Model, StreamOptions


def _usage_from_openai(usage: dict[str, Any]) -> dict[str, Any]:
	"""Map OpenAI-style usage to our internal usage schema."""
	prompt = usage.get("prompt_tokens")
	completion = usage.get("completion_tokens")
	total = usage.get("total_tokens")
	if not isinstance(prompt, int):
		prompt = 0
	if not isinstance(completion, int):
		completion = 0
	if not isinstance(total, int):
		total = prompt + completion

	out: dict[str, Any] = {
		"input": prompt,
		"output": completion,
		"cache_read": 0,
		"cache_write": 0,
		"total_tokens": total,
	}

	# OpenRouter (and sometimes others) include cost details in the usage object.
	# We map whatever we can without assuming provider-specific semantics.
	cost_total = usage.get("cost")
	cost_details = usage.get("cost_details")

	cost: dict[str, float] = {
		"input": 0.0,
		"output": 0.0,
		"cache_read": 0.0,
		"cache_write": 0.0,
		"total": 0.0,
	}

	if isinstance(cost_details, dict):
		prompt_cost = cost_details.get("upstream_inference_prompt_cost")
		completion_cost = cost_details.get("upstream_inference_completions_cost")
		if isinstance(prompt_cost, (int, float)):
			cost["input"] = float(prompt_cost)
		if isinstance(completion_cost, (int, float)):
			cost["output"] = float(completion_cost)
		cost["total"] = cost["input"] + cost["output"]

	if isinstance(cost_total, (int, float)):
		# Prefer explicit total if present.
		cost["total"] = float(cost_total)
		if cost["input"] == 0.0 and cost["output"] == 0.0:
			# We don't know the split; keep total only.
			pass

	out["cost"] = cost
	return out


def _pick_last_user_text(context: Context) -> str:
	for msg in reversed(context["messages"]):
		if msg.get("role") == "user":
			content = msg.get("content")
			if isinstance(content, str):
				return content
	return ""


def _to_openai_messages(context: Context) -> list[dict[str, Any]]:
	"""Convert minimal Context to OpenAI chat.completions message format.

	For now we only support user/assistant text and tool results.
	"""
	out: list[dict[str, Any]] = []
	if context.get("system_prompt"):
		out.append({"role": "system", "content": context["system_prompt"]})

	for msg in context["messages"]:
		role = msg.get("role")
		if role == "user":
			out.append({"role": "user", "content": msg.get("content", "")})
		elif role == "assistant":
			# Our minimal AssistantMessage stores blocks; join them.
			blocks = msg.get("content", [])
			if not isinstance(blocks, list):
				blocks = []
			text = "".join(
				b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
			)
			out.append({"role": "assistant", "content": text})
		elif role == "toolResult":
			# Map our toolResult message to OpenAI's `tool` role message.
			blocks = msg.get("content", [])
			if not isinstance(blocks, list):
				blocks = []
			text = "".join(
				b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
			)
			out.append(
				{
					"role": "tool",
					"tool_call_id": msg.get("tool_call_id"),
					"content": text,
				}
			)
	return out


class OpenAICompatibleProvider:
	"""OpenAI Chat Completions compatible provider with SSE streaming.

	Works for providers like OpenRouter, DeepSeek, and Qwen when they expose an
	OpenAI-compatible `/v1/chat/completions` endpoint.
	"""

	def __init__(
		self,
		*,
		base_url: str,
		provider_name: str,
		default_headers: dict[str, str] | None = None,
	):
		self._base_url = base_url.rstrip("/")
		self._provider_name = provider_name
		self._default_headers = default_headers or {}

	def stream(self, model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
		opts = options or {}
		api_key = opts.get("api_key")
		temperature = opts.get("temperature")
		max_tokens = opts.get("max_tokens")
		timeout_s = float(opts.get("timeout_s", 60.0))

		headers: dict[str, str] = {**self._default_headers, **(opts.get("headers") or {})}
		if api_key:
			headers["Authorization"] = f"Bearer {api_key}"

		payload: dict[str, Any] = {
			"model": model.model,
			"messages": _to_openai_messages(context),
			"stream": True,
		}
		# Ask providers to include usage in streaming chunks when supported.
		payload["stream_options"] = {"include_usage": True}

		tools = context.get("tools")
		if tools:
			payload["tools"] = [
				{
					"type": "function",
					"function": {
						"name": t["name"],
						"description": t.get("description", ""),
						"parameters": t.get("parameters", {}),
					},
				}
				for t in tools
			]
			payload["tool_choice"] = "auto"
			payload["parallel_tool_calls"] = True
		if temperature is not None:
			payload["temperature"] = temperature
		if max_tokens is not None:
			payload["max_tokens"] = max_tokens

		async def gen() -> AsyncIterator[dict[str, Any]]:
			yield {
				"type": "start",
				"partial": {"api": model.api, "provider": self._provider_name, "model": model.model},
			}

			if self._base_url.endswith("/v1"):
				url = f"{self._base_url}/chat/completions"
			else:
				url = f"{self._base_url}/v1/chat/completions"
			saw_tool_calls = False
			finish_reason_seen: str | None = None
			pricing_in = opts.get("input_cost_per_1m")
			pricing_out = opts.get("output_cost_per_1m")

			try:
				async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
					async with client.stream("POST", url, headers=headers, json=payload) as resp:
						if resp.status_code >= 400:
							body = await resp.aread()
							yield {
								"type": "error",
								"error": f"HTTP {resp.status_code}: {body.decode('utf-8', 'replace')}",
							}
							return

						async for line in resp.aiter_lines():
							if not line:
								continue
							if not line.startswith("data:"):
								continue
							data = line[len("data:") :].strip()
							if data == "[DONE]":
								yield {"type": "done", "reason": ("tool_use" if saw_tool_calls else "stop")}
								return

							try:
								event = json.loads(data)
							except json.JSONDecodeError:
								continue

							usage_block = event.get("usage")
							if isinstance(usage_block, dict) and usage_block:
								mapped = _usage_from_openai(usage_block)
								# If provider didn't include cost split/total, optionally compute from pricing.
								if isinstance(mapped.get("cost"), dict):
									cost = mapped["cost"]
									if cost.get("total", 0.0) == 0.0 and (
										isinstance(pricing_in, (int, float)) or isinstance(pricing_out, (int, float))
									):
										if isinstance(pricing_in, (int, float)):
											cost["input"] = float(mapped["input"]) / 1_000_000.0 * float(pricing_in)
										if isinstance(pricing_out, (int, float)):
											cost["output"] = float(mapped["output"]) / 1_000_000.0 * float(pricing_out)
										cost["total"] = float(cost.get("input", 0.0)) + float(cost.get("output", 0.0))
								yield {"type": "usage", "usage": mapped}

							choices = event.get("choices")
							if not choices:
								continue

							delta = choices[0].get("delta") or {}
							chunk = delta.get("content")
							if isinstance(chunk, str) and chunk:
								await asyncio.sleep(0)
								yield {"type": "text_delta", "delta": chunk}

							tool_calls = delta.get("tool_calls")
							if isinstance(tool_calls, list) and tool_calls:
								saw_tool_calls = True
								for tc in tool_calls:
									if not isinstance(tc, dict):
										continue
									index = tc.get("index")
									if not isinstance(index, int):
										continue
									fn = tc.get("function") or {}
									if not isinstance(fn, dict):
										fn = {}
									name = fn.get("name")
									args_delta = fn.get("arguments")
									yield {
										"type": "toolcall_delta",
										"index": index,
										"id": tc.get("id"),
										"name": name,
										"arguments_delta": args_delta,
									}

							# Some providers include a finish_reason in streamed chunks.
							finish_reason = choices[0].get("finish_reason")
							if isinstance(finish_reason, str) and finish_reason:
								finish_reason_seen = finish_reason
								if finish_reason in {"tool_calls", "tool_call"}:
									saw_tool_calls = True

						# Do not terminate on finish_reason; some providers send usage after.

						# Stream ended without an explicit [DONE]. Best-effort finalize.
						reason = "tool_use" if saw_tool_calls or finish_reason_seen in {"tool_calls", "tool_call"} else "stop"
						yield {"type": "done", "reason": reason}

			except (httpx.HTTPError, OSError) as e:
				yield {"type": "error", "error": str(e)}

		return AssistantMessageEventStream(model=model, context=context, events=gen())
