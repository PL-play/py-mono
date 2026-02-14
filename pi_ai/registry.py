from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .types import Api, Context, Model, StreamOptions


class ApiProvider(Protocol):
	def stream(self, model: Model, context: Context, options: StreamOptions | None = None):  # noqa: ANN001
		...


_PROVIDERS: dict[Api, ApiProvider] = {}


def register_api_provider(api: Api, provider: ApiProvider) -> None:
	if api in _PROVIDERS:
		raise ValueError(f"Provider already registered for api: {api}")
	_PROVIDERS[api] = provider


def get_api_provider(api: Api) -> ApiProvider | None:
	return _PROVIDERS.get(api)


def clear_registry_for_tests() -> None:
	"""Test helper; not part of the intended public API."""
	_PROVIDERS.clear()
