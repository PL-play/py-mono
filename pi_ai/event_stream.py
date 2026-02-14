from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from .types import AssistantMessage, Context, Model, StopReason, Usage


def _now_ms() -> int:
	return int(time.time() * 1000)


def _zero_usage() -> Usage:
	return {
		"input": 0,
		"output": 0,
		"cache_read": 0,
		"cache_write": 0,
		"total_tokens": 0,
		"cost": {
			"input": 0.0,
			"output": 0.0,
			"cache_read": 0.0,
			"cache_write": 0.0,
			"total": 0.0,
		},
	}


class AssistantMessageEventStream:
	"""Async event stream + a final assembled assistant message.

	This mirrors the ergonomics of pi-ai's AssistantMessageEventStream:
	- iterate over events
	- call .result() to get the final AssistantMessage

	We keep the event schema minimal for the first iteration.
	"""

	def __init__(
		self,
		*,
		model: Model,
		context: Context,
		events: AsyncIterator[dict[str, Any]],
	):
		self._model = model
		self._context = context
		self._events_iter = events.__aiter__()
		self._text = ""
		self._tool_calls: dict[int, dict[str, Any]] = {}
		self._usage: Usage = _zero_usage()
		self._final: AssistantMessage | None = None
		self._stop_reason: StopReason | None = None
		self._error_message: str | None = None

	def __aiter__(self) -> "AssistantMessageEventStream":
		return self

	async def __anext__(self) -> dict[str, Any]:
		event = await self._events_iter.__anext__()

		match event.get("type"):
			case "text_delta":
				delta = event.get("delta")
				if isinstance(delta, str):
					self._text += delta
			case "toolcall_delta":
				index = event.get("index")
				if not isinstance(index, int):
					return event

				call = self._tool_calls.get(index)
				if call is None:
					call = {"id": "", "name": "", "arguments_json": ""}
					self._tool_calls[index] = call

				tool_call_id = event.get("id")
				if isinstance(tool_call_id, str) and tool_call_id:
					call["id"] = tool_call_id
				name = event.get("name")
				if isinstance(name, str) and name:
					call["name"] = name
				args_delta = event.get("arguments_delta")
				if isinstance(args_delta, str) and args_delta:
					call["arguments_json"] += args_delta
			case "done":
				reason = event.get("reason")
				if reason in {"stop", "length", "tool_use", "error", "aborted"}:
					self._stop_reason = reason  # type: ignore[assignment]
				else:
					self._stop_reason = "stop"
				self._final = self._assemble_final()
			case "usage":
				usage = event.get("usage")
				if isinstance(usage, dict):
					# Best-effort merge; providers may send partial fields.
					for k in ("input", "output", "cache_read", "cache_write", "total_tokens"):
						v = usage.get(k)
						if isinstance(v, int):
							self._usage[k] = v  # type: ignore[literal-required]
					cost = usage.get("cost")
					if isinstance(cost, dict):
						for ck in ("input", "output", "cache_read", "cache_write", "total"):
							cv = cost.get(ck)
							if isinstance(cv, (int, float)):
								self._usage["cost"][ck] = float(cv)
			case "error":
				err = event.get("error")
				self._error_message = str(err) if err is not None else "unknown error"
				self._stop_reason = "error"
				self._final = self._assemble_final()

		return event

	async def result(self) -> AssistantMessage:
		"""Consume the stream (if needed) and return the final assistant message."""
		if self._final is not None:
			return self._final

		async for _ in self:
			pass

		if self._final is None:
			# Provider ended without an explicit done/error.
			self._stop_reason = self._stop_reason or "stop"
			self._final = self._assemble_final()
		return self._final

	def _assemble_final(self) -> AssistantMessage:
		stop_reason: StopReason = self._stop_reason or "stop"
		content: list[dict[str, Any]] = []
		if self._text:
			content.append({"type": "text", "text": self._text})

		if self._tool_calls:
			for index in sorted(self._tool_calls.keys()):
				call = self._tool_calls[index]
				args_obj: dict[str, Any] = {}
				args_json = call.get("arguments_json")
				if isinstance(args_json, str) and args_json.strip():
					try:
						parsed = json.loads(args_json)
						if isinstance(parsed, dict):
							args_obj = parsed
					except json.JSONDecodeError:
						args_obj = {}

				content.append(
					{
						"type": "toolCall",
						"id": call.get("id") or f"toolcall:{index}",
						"name": call.get("name") or "unknown",
						"arguments": args_obj,
					}
				)

			# If the provider didn't mark tool use explicitly, infer it.
			if stop_reason == "stop":
				stop_reason = "tool_use"

		message: AssistantMessage = {
			"role": "assistant",
			"content": content,
			"api": self._model.api,
			"provider": self._model.provider,
			"model": self._model.model,
			"usage": self._usage,
			"stop_reason": stop_reason,
			"timestamp": _now_ms(),
		}
		if self._error_message:
			message["error_message"] = self._error_message
		return message
