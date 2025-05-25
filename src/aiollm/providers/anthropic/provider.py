import os
import typing

from anthropic import APIError, AsyncAnthropic

from aiollm.chat_completion_events.chat_completion_event import ChatCompletionEvent
from aiollm.chat_completions.chat_completion import ChatCompletion
from aiollm.exceptions.llm_error import LLMError
from aiollm.messages.message import Message
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.providers.anthropic.from_args import FromArgs
from aiollm.providers.anthropic.to_chat_completion import ToChatCompletion
from aiollm.providers.anthropic.to_chat_completion_event import ToChatCompletionEvent
from aiollm.providers.provider import Provider
from aiollm.response_formats.response_format import ResponseFormat
from aiollm.tools.tool import Tool
from aiollm.tools.tools import Tools


class AnthropicProvider(Provider):
    def __init__(
        self,
        base_url: str = "https://api.anthropic.com",
        api_key: str | None = os.getenv("ANTHROPIC_API_KEY"),
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = AsyncAnthropic(api_key=self._api_key, base_url=self._base_url)

    async def chat_completion(
        self,
        model: Model,
        messages: list[Message],
        tools: Tools | list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        parameters: Parameters | None = None,
        name: str | None = None,
    ) -> ChatCompletion:
        request = FromArgs.from_args(
            model=model.id,
            messages=messages,
            tools=tools,
            parameters=parameters,
            response_format=response_format,
        )
        try:
            message = await self._client.messages.create(**request)
        except APIError as e:
            raise LLMError(str(e)) from None

        return ToChatCompletion.from_message(message, response_format, name)

    async def chat_completion_stream(
        self,
        model: Model,
        messages: list[Message],
        tools: Tools | list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        parameters: Parameters | None = None,
        name: str | None = None,
    ) -> typing.AsyncIterator[ChatCompletionEvent]:
        request = FromArgs.from_args(
            model=model.id,
            messages=messages,
            tools=tools,
            parameters=parameters,
            response_format=response_format,
        )

        try:
            stream = await self._client.messages.create(**request, stream=True)
        except APIError as e:
            raise LLMError(str(e)) from None

        async for event in ToChatCompletionEvent(stream=stream, model=model, name=name):
            yield event
