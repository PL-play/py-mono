from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

Api = str
Provider = str


@dataclass(frozen=True)
class Model:
	api: Api
	provider: Provider
	model: str


class StreamOptions(TypedDict, total=False):
	temperature: float
	max_tokens: int
	api_key: str
	session_id: str
	timeout_s: float
	headers: dict[str, str]
	# Optional pricing for cost calculation (USD per 1M tokens)
	input_cost_per_1m: float
	output_cost_per_1m: float


class TextContent(TypedDict):
	type: Literal["text"]
	text: str


class ToolCallContent(TypedDict):
	type: Literal["toolCall"]
	id: str
	name: str
	arguments: dict[str, Any]


class UsageCost(TypedDict):
	input: float
	output: float
	cache_read: float
	cache_write: float
	total: float


class Usage(TypedDict):
	input: int
	output: int
	cache_read: int
	cache_write: int
	total_tokens: int
	cost: UsageCost


StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]


class UserMessage(TypedDict):
	role: Literal["user"]
	content: str
	timestamp: int


class AssistantMessage(TypedDict):
	role: Literal["assistant"]
	content: list[TextContent | ToolCallContent]
	api: Api
	provider: Provider
	model: str
	usage: Usage
	stop_reason: StopReason
	error_message: NotRequired[str]
	timestamp: int


class ToolResultMessage(TypedDict):
	role: Literal["toolResult"]
	tool_call_id: str
	tool_name: str
	content: list[TextContent]
	is_error: bool
	timestamp: int


Message = UserMessage | AssistantMessage | ToolResultMessage


class Tool(TypedDict):
	name: str
	description: str
	parameters: dict[str, Any]


class Context(TypedDict):
	system_prompt: str
	messages: list[Message]
	tools: NotRequired[list[Tool]]
