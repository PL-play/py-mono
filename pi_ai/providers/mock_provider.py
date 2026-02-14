from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..event_stream import AssistantMessageEventStream
from ..types import Context, Model, StreamOptions


class MockChatProvider:
	"""A deterministic provider used for learning + tests.

	Behavior:
	- Emits: start -> text_delta* -> done
	- The text is a simple echo of the last user message.
	"""

	def stream(self, model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
		async def gen() -> AsyncIterator[dict[str, Any]]:
			yield {
				"type": "start",
				"partial": {"api": model.api, "provider": model.provider, "model": model.model},
			}

			last_user = None
			for msg in reversed(context["messages"]):
				if msg.get("role") == "user":
					last_user = msg
					break

			text = "(no user message)"
			if last_user and isinstance(last_user.get("content"), str):
				text = f"echo: {last_user['content']}"

			# Stream it in chunks to mimic real providers.
			for chunk in [text[:5], text[5:]]:
				if chunk:
					await asyncio.sleep(0)
					yield {"type": "text_delta", "delta": chunk}

			yield {"type": "done", "reason": "stop"}

		return AssistantMessageEventStream(model=model, context=context, events=gen())
