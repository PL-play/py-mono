import asyncio
import time
import unittest

from pi_ai import stream
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.types import Context, Model


def now_ms() -> int:
	return int(time.time() * 1000)


class LocalToolCallProvider:
	def stream(self, model: Model, context: Context, options=None) -> AssistantMessageEventStream:  # noqa: ANN001
		async def gen():
			yield {"type": "start", "partial": {"api": model.api, "provider": model.provider, "model": model.model}}
			yield {
				"type": "toolcall_delta",
				"index": 0,
				"id": "call_1",
				"name": "add",
				"arguments_delta": "{\"a\": 1,",
			}
			yield {"type": "toolcall_delta", "index": 0, "arguments_delta": " \"b\": 2}"}
			yield {"type": "done", "reason": "tool_use"}

		return AssistantMessageEventStream(model=model, context=context, events=gen())


class TestToolCallAssembly(unittest.TestCase):
	def test_toolcall_is_assembled(self):
		# Import here to avoid importing registry at module import time.
		from pi_ai.registry import register_api_provider

		api = "local-toolcall"
		register_api_provider(api, LocalToolCallProvider())

		model = Model(api=api, provider="local", model="x")
		context: Context = {
			"system_prompt": "",
			"messages": [{"role": "user", "content": "hi", "timestamp": now_ms()}],
		}

		async def run():
			s = stream(model, context)
			msg = await s.result()
			tool_calls = [b for b in msg["content"] if b["type"] == "toolCall"]
			self.assertEqual(len(tool_calls), 1)
			self.assertEqual(tool_calls[0]["id"], "call_1")
			self.assertEqual(tool_calls[0]["name"], "add")
			self.assertEqual(tool_calls[0]["arguments"], {"a": 1, "b": 2})
			self.assertEqual(msg["stop_reason"], "tool_use")

		asyncio.run(run())


if __name__ == "__main__":
	unittest.main()
