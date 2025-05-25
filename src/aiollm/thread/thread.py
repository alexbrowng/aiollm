import typing

from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage
from aiollm.thread.turn import Turn
from aiollm.thread.turns import Turns


class Thread:
    def __init__(self, messages: typing.Sequence[UserMessage | AssistantMessage | ToolMessage] | None = None):
        self._turns = Turns()

        if messages:
            self.extend(messages)

    @property
    def turns(self) -> Turns:
        return self._turns

    def __iter__(self):
        return iter(message for turn in self._turns for message in turn)

    def append(self, message: UserMessage | AssistantMessage | ToolMessage) -> Turn:
        return self._turns.append(message)

    def extend(self, messages: typing.Sequence[UserMessage | AssistantMessage | ToolMessage]) -> None:
        for message in messages:
            self.append(message)
