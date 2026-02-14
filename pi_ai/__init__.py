"""Minimal Python re-implementation of pi-mono core building blocks.

This package starts with the pi-ai layer: types, provider registry, and
stream/complete entrypoints.
"""

from .types import (
	AssistantMessage,
	Context,
	Model,
	Tool,
	ToolResultMessage,
	UserMessage,
)
from .stream import complete, stream


def get_model(provider: str, model: str) -> Model:
	"""Get a model handle.

	We keep this intentionally small and only support a few providers to start.
	"""
	if provider == "mock":
		return Model(api="mock-chat", provider="mock", model=model)
	if provider == "openrouter":
		return Model(api="openrouter-chat", provider="openrouter", model=model)
	if provider == "deepseek":
		return Model(api="deepseek-chat", provider="deepseek", model=model)
	if provider == "qwen":
		return Model(api="qwen-chat", provider="qwen", model=model)
	raise ValueError(
		f"Unknown provider: {provider!r}. Supported: mock, openrouter, deepseek, qwen."
	)


__all__ = [
	"AssistantMessage",
	"Context",
	"Model",
	"Tool",
	"ToolResultMessage",
	"UserMessage",
	"complete",
	"stream",
	"get_model",
]
