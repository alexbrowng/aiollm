import typing

from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage


class Turn:
    def __init__(self, messages: typing.Sequence[UserMessage | AssistantMessage | ToolMessage] | None = None):
        self._messages = list(messages) if messages else []

    def __iter__(self):
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def append(self, message: UserMessage | AssistantMessage | ToolMessage) -> None:
        self._messages.append(message)

    def extend(self, messages: typing.Sequence[UserMessage | AssistantMessage | ToolMessage]) -> None:
        for message in messages:
            self.append(message)
