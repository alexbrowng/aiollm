import abc
import typing

from aiollm.chat_completion_events.chat_completion_event import ChatCompletionEvent
from aiollm.chat_completions.chat_completion import ChatCompletion
from aiollm.messages.message import Message
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.response_formats.response_format import ResponseFormat
from aiollm.tools.tool import Tool
from aiollm.tools.tools import Tools


class Provider(abc.ABC):
    @abc.abstractmethod
    def chat_completion(
        self,
        model: Model,
        messages: list[Message],
        tools: Tools | list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        parameters: Parameters | None = None,
        name: str | None = None,
    ) -> typing.Awaitable[ChatCompletion]:
        """
        Returns the chat completion.
        """

    @abc.abstractmethod
    def chat_completion_stream(
        self,
        model: Model,
        messages: list[Message],
        tools: Tools | list[Tool] | None = None,
        response_format: ResponseFormat | None = None,
        parameters: Parameters | None = None,
        name: str | None = None,
    ) -> typing.AsyncIterator[ChatCompletionEvent]:
        """
        Streams the chat completion events.
        """
