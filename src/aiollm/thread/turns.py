import typing

from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage
from aiollm.thread.turn import Turn


class Turns:
    def __init__(self, turns: typing.Sequence[Turn] | None = None):
        self._turns = list(turns) if turns else []

    def __iter__(self):
        return iter(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def append(self, message: UserMessage | AssistantMessage | ToolMessage) -> Turn:
        if isinstance(message, UserMessage) or not self._turns:
            turn = Turn()
            self._turns.append(turn)
        else:
            turn = self._turns[-1]

        turn.append(message)
        return turn

    def extend(self, messages: typing.Sequence[UserMessage | AssistantMessage | ToolMessage]) -> None:
        for message in messages:
            self.append(message)
