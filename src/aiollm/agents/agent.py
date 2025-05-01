import typing

from aiollm.chat_completion_events.chat_completion_event import ChatCompletionEvent
from aiollm.chat_completion_events.to_chat_completion import ToChatCompletion
from aiollm.chat_completions.chat_completion import ChatCompletion
from aiollm.messages.message import Message
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.providers.provider import Provider
from aiollm.response_formats.response_format import ResponseFormat
from aiollm.thread.history import History
from aiollm.thread.thread import Thread
from aiollm.tools.tool import Tool
from aiollm.tools.tools import Tools
from aiollm.utils.models import load_model_by_name
from aiollm.utils.providers import load_provider_by_name, load_provider_from_model


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        model: str | Model,
        provider: str | Provider | None = None,
        parameters: Parameters | None = None,
        tools: Tools | None = None,
        response_format: ResponseFormat | None = None,
    ):
        self._name = name
        self._description = description
        self._instructions = instructions

        if isinstance(model, str):
            self._model = load_model_by_name(model)
        else:
            self._model = model

        if isinstance(provider, str):
            self._provider = load_provider_by_name(provider)
        elif not provider:
            self._provider = load_provider_from_model(self._model)
        else:
            self._provider = provider

        self._parameters = parameters
        self._tools = tools or Tools()
        self._response_format = response_format

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def tools(self) -> Tools | None:
        return self._tools

    @property
    def response_format(self) -> ResponseFormat | None:
        return self._response_format

    def _prepare_messages(
        self,
        thread: Thread,
        include_tool_calls: bool = False,
        max_history_turns: int | None = None,
    ) -> list[Message]:
        messages: list[Message] = [SystemMessage(content=self.instructions)]
        messages.extend(History(thread, include_tool_calls=include_tool_calls, max_turns=max_history_turns))
        return messages

    async def _handle_tool_calls(self, chat_completion: ChatCompletion, thread: Thread) -> bool:
        if tool_calls := chat_completion.message.tool_calls:
            tool_messages: typing.Sequence[ToolMessage] = await self._tools.calls(tool_calls)
            thread.extend(tool_messages)
            return True
        return False

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        strict: bool = True,
        handler: typing.Callable[[dict], typing.Awaitable[dict]] | None = None,
    ) -> Tool:
        tool = Tool(name=name, description=description, parameters=parameters, strict=strict)
        self._tools.register(tool, handler)
        return tool

    async def run(
        self,
        thread: Thread,
        include_tool_calls: bool = False,
        max_history_turns: int | None = None,
        max_iterations: int = 5,
    ) -> None:
        llm = load_provider_from_model(self._model)

        iteration_count = 0
        while iteration_count < max_iterations:
            iteration_count += 1

            messages = self._prepare_messages(
                thread=thread,
                include_tool_calls=include_tool_calls,
                max_history_turns=max_history_turns,
            )

            chat_completion = await llm.chat_completion(
                model=self._model,
                messages=messages,
                tools=self._tools,
                response_format=self.response_format,
                parameters=self._parameters,
            )

            thread.append(chat_completion.message)

            if not await self._handle_tool_calls(chat_completion, thread):
                break
        else:
            print(f"Warning: Agent reached maximum iterations ({max_iterations}).")

    async def stream(
        self,
        thread: Thread,
        include_tool_calls: bool = False,
        max_history_turns: int | None = None,
        max_iterations: int = 5,
    ) -> typing.AsyncIterator[ChatCompletionEvent]:
        llm = load_provider_from_model(self._model)

        iteration_count = 0
        while iteration_count < max_iterations:
            iteration_count += 1

            messages = self._prepare_messages(
                thread=thread,
                include_tool_calls=include_tool_calls,
                max_history_turns=max_history_turns,
            )

            events: list[ChatCompletionEvent] = []
            async for event in llm.chat_completion_stream(
                model=self._model,
                messages=messages,
                tools=self._tools,
                response_format=self.response_format,
                parameters=self._parameters,
            ):
                events.append(event)
                yield event

            chat_completion = ToChatCompletion.from_chat_completion_events(events)
            thread.append(chat_completion.message)

            if not await self._handle_tool_calls(chat_completion, thread):
                break
        else:
            print(f"Warning: Agent reached maximum iterations ({max_iterations}).")
