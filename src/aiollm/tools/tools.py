import asyncio
import json
import typing

from aiollm.messages.tool_message import ToolMessage
from aiollm.tools.tool import Tool
from aiollm.tools.tool_call import ToolCall

Handler = typing.Callable[[dict], typing.Awaitable[dict]]


class Tools:
    def __init__(self):
        self._tools: list[tuple[Tool, Handler | None]] = []

    def __iter__(self):
        return iter(tool for tool, _ in self._tools)

    def register(self, tool: Tool, handler: Handler | None = None):
        self._tools.append((tool, handler))

    def resolve(self, tool_call: ToolCall) -> tuple[Tool, Handler | None]:
        try:
            return next((tool, handler) for tool, handler in self._tools if tool.name == tool_call.name)
        except StopIteration:
            raise ValueError(f"Tool {tool_call.name} not found") from None

    async def call(self, tool_call: ToolCall) -> ToolMessage:
        _, handler = self.resolve(tool_call)

        if handler is None:
            raise ValueError(f"Tool {tool_call.name} has no handler")

        result = await handler(tool_call.arguments)

        return ToolMessage(id=tool_call.id, content=json.dumps(result, default=str))

    async def calls(self, tool_calls: typing.Sequence[ToolCall]) -> list[ToolMessage]:
        return await asyncio.gather(*[self.call(tool_call) for tool_call in tool_calls])
