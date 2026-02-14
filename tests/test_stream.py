import asyncio
import os
import time
import unittest

from pi_ai import complete, get_model, stream


def now_ms() -> int:
	return int(time.time() * 1000)


class TestStream(unittest.TestCase):
	@unittest.skipUnless(os.getenv("OPENROUTER_API_KEY"), "requires OPENROUTER_API_KEY (can be provided via .env)")
	def test_stream_and_complete(self):
		model = get_model("openrouter", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
		context = {
			"system_prompt": "You are helpful.",
			"messages": [
				{
					"role": "user",
					"content": "Reply with the single word: pong",
					"timestamp": now_ms(),
				},
			],
		}

		async def run():
			s = stream(
				model,
				context,
				{
					"api_key": os.environ["OPENROUTER_API_KEY"],
					"headers": {
						"HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
						"X-Title": os.getenv("OPENROUTER_X_TITLE", "py-mono"),
					},
					"timeout_s": 60.0,
				},
			)
			events = []
			async for e in s:
				events.append(e)

			self.assertEqual(events[0]["type"], "start")
			self.assertTrue(any(e["type"] == "text_delta" for e in events))
			self.assertEqual(events[-1]["type"], "done")

			msg = await s.result()
			self.assertEqual(msg["role"], "assistant")
			self.assertEqual(msg["provider"], "openrouter")
			self.assertEqual(msg["api"], "openrouter-chat")
			text = "".join(block.get("text", "") for block in msg["content"] if block["type"] == "text")
			self.assertTrue(text.strip())
			self.assertGreater(msg.get("usage", {}).get("total_tokens", 0), 0)

			msg2 = await complete(
				model,
				context,
				{
					"api_key": os.environ["OPENROUTER_API_KEY"],
					"headers": {
						"HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
						"X-Title": os.getenv("OPENROUTER_X_TITLE", "py-mono"),
					},
					"timeout_s": 60.0,
				},
			)
			text2 = "".join(block.get("text", "") for block in msg2["content"] if block["type"] == "text")
			self.assertTrue(text2.strip())
			self.assertGreater(msg2.get("usage", {}).get("total_tokens", 0), 0)

		asyncio.run(run())


if __name__ == "__main__":
	unittest.main()
