import asyncio
import time
import unittest

from pi_ai import stream
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.types import Context, Model


def now_ms() -> int:
	return int(time.time() * 1000)


class LocalUsageProvider:
	def stream(self, model: Model, context: Context, options=None) -> AssistantMessageEventStream:  # noqa: ANN001
		async def gen():
			yield {"type": "start", "partial": {"api": model.api, "provider": model.provider, "model": model.model}}
			yield {
				"type": "usage",
				"usage": {
					"input": 10,
					"output": 5,
					"cache_read": 0,
					"cache_write": 0,
					"total_tokens": 15,
					"cost": {"input": 0.01, "output": 0.02, "cache_read": 0.0, "cache_write": 0.0, "total": 0.03},
				},
			}
			yield {"type": "text_delta", "delta": "ok"}
			yield {"type": "done", "reason": "stop"}

		return AssistantMessageEventStream(model=model, context=context, events=gen())


class TestUsageAssembly(unittest.TestCase):
	def test_usage_is_attached(self):
		from pi_ai.registry import register_api_provider

		api = "local-usage"
		register_api_provider(api, LocalUsageProvider())

		model = Model(api=api, provider="local", model="x")
		context: Context = {
			"system_prompt": "",
			"messages": [{"role": "user", "content": "hi", "timestamp": now_ms()}],
		}

		async def run():
			msg = await stream(model, context).result()
			self.assertEqual(msg["usage"]["input"], 10)
			self.assertEqual(msg["usage"]["output"], 5)
			self.assertEqual(msg["usage"]["total_tokens"], 15)
			self.assertAlmostEqual(msg["usage"]["cost"]["total"], 0.03)

		asyncio.run(run())


if __name__ == "__main__":
	unittest.main()
