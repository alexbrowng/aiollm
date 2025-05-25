import asyncio
import functools
import json
import typing

from aiollm.json_schema.object import Object
from aiollm.messages.tool_message import ToolMessage
from aiollm.tools.tool import Tool
from aiollm.tools.tool_call import ToolCall

Handler = typing.Callable[[dict, dict], typing.Awaitable[dict]]


class Tools:
    def __init__(self, instructions: str | None = None, context: dict | None = None):
        self._instructions = instructions
        self._tools: list[Tool] = []
        self._handlers: dict[str, Handler] = {}
        self._context = context or {}

    @property
    def instructions(self) -> str | None:
        return self._instructions

    @property
    def context(self) -> dict:
        return self._context

    def __iter__(self):
        return iter(self._tools)

    def register(
        self,
        parameters: Object,
        name: str | None = None,
        description: str | None = None,
        strict: bool = True,
    ):
        def decorator(handler: Handler) -> Handler:
            tool_name = name or handler.__name__
            tool_description = description or handler.__doc__ or ""
            tool = Tool(name=tool_name, description=tool_description, parameters=parameters, strict=strict)
            self._tools.append(tool)

            def wrapper(*args, **kwargs):
                return handler(*args, **kwargs)

            tool_handler = functools.update_wrapper(wrapper, handler)
            self._handlers[tool.name] = tool_handler
            return tool_handler

        return decorator

    async def __call__(self, tool_call: ToolCall) -> ToolMessage:
        handler = self._handlers[tool_call.name]
        result = await handler(tool_call.arguments, self.context)
        return ToolMessage(id=tool_call.id, content=json.dumps(result, default=str), name=tool_call.name)

    async def call(self, tool_call: ToolCall) -> ToolMessage:
        return await self.__call__(tool_call)

    async def calls(self, tool_calls: typing.Sequence[ToolCall]) -> typing.Sequence[ToolMessage]:
        return await asyncio.gather(*[self.call(tool_call) for tool_call in tool_calls])
