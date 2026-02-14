from __future__ import annotations

import os

from ..registry import register_api_provider
from .mock_provider import MockChatProvider
from .openai_compat import OpenAICompatibleProvider


_registered = False


def register_builtins() -> None:
	global _registered
	if _registered:
		return

	# OpenAI-compatible real providers (streaming chat.completions)
	register_api_provider(
		"openrouter-chat",
		OpenAICompatibleProvider(
			base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api"),
			provider_name="openrouter",
		),
	)
	register_api_provider(
		"deepseek-chat",
		OpenAICompatibleProvider(
			base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
			provider_name="deepseek",
		),
	)
	register_api_provider(
		"qwen-chat",
		OpenAICompatibleProvider(
			base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode"),
			provider_name="qwen",
		),
	)
	_registered = True
