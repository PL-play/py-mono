from __future__ import annotations

from pathlib import Path

from .dotenv import load_dotenv
from .event_stream import AssistantMessageEventStream
from .providers.register_builtins import register_builtins
from .registry import get_api_provider
from .types import AssistantMessage, Context, Model, StreamOptions

# Load .env in a couple of common locations.
# - CWD: convenient for running examples/tests from py-mono/
# - Package root: makes it work even when CWD is the monorepo root
_pkg_root = Path(__file__).resolve().parent.parent
load_dotenv(Path.cwd() / ".env")
load_dotenv(_pkg_root / ".env")

# Register built-in providers on import, similar to pi-ai's side-effect imports.
register_builtins()


def stream(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
	provider = get_api_provider(model.api)
	if provider is None:
		raise ValueError(f"No API provider registered for api: {model.api!r}")
	return provider.stream(model, context, options)


async def complete(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
	s = stream(model, context, options)
	return await s.result()
