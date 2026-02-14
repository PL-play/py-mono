import asyncio
import os
import time
import unittest

from pi_ai import complete, get_model

def now_ms() -> int:
	return int(time.time() * 1000)


def mk_context(text: str):
	return {
		"system_prompt": "You are a helpful assistant.",
		"messages": [{"role": "user", "content": text, "timestamp": now_ms()}],
	}


class TestRealProvidersSmoke(unittest.TestCase):
	@unittest.skipUnless(os.getenv("OPENROUTER_API_KEY"), "requires OPENROUTER_API_KEY")
	def test_openrouter_complete(self):
		model = get_model("openrouter", os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
		ctx = mk_context("Say 'pong' and nothing else.")

		async def run():
			msg = await complete(
				model,
				ctx,
				{
					"api_key": os.environ["OPENROUTER_API_KEY"],
					"headers": {
						# OpenRouter recommends setting these (optional)
						"HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
						"X-Title": os.getenv("OPENROUTER_X_TITLE", "py-mono"),
					},
					"timeout_s": 60.0,
				},
			)
			print(msg)
			text = "".join(b["text"] for b in msg["content"])
			self.assertTrue(text.strip())

		asyncio.run(run())

	@unittest.skipUnless(os.getenv("DEEPSEEK_API_KEY"), "requires DEEPSEEK_API_KEY")
	def test_deepseek_complete(self):
		model = get_model("deepseek", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
		ctx = mk_context("Reply with the single word: pong")

		async def run():
			msg = await complete(model, ctx, {"api_key": os.environ["DEEPSEEK_API_KEY"], "timeout_s": 60.0})
			text = "".join(b["text"] for b in msg["content"])
			self.assertTrue(text.strip())

		asyncio.run(run())

	@unittest.skipUnless(os.getenv("QWEN_API_KEY"), "requires QWEN_API_KEY")
	def test_qwen_complete(self):
		model = get_model("qwen", os.getenv("QWEN_MODEL", "qwen-turbo"))
		ctx = mk_context("Reply with the single word: pong")

		async def run():
			msg = await complete(model, ctx, {"api_key": os.environ["QWEN_API_KEY"], "timeout_s": 60.0})
			text = "".join(b["text"] for b in msg["content"])
			self.assertTrue(text.strip())

		asyncio.run(run())


if __name__ == "__main__":
	unittest.main()
