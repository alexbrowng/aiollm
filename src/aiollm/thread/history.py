import typing

from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage
from aiollm.thread.thread import Thread


class History:
    def __init__(self, thread: Thread, include_tool_calls: bool = False, max_turns: int | None = None):
        self._thread = thread
        self._include_tool_calls = include_tool_calls
        self._max_turns = max_turns

    def __iter__(self) -> typing.Iterator[UserMessage | AssistantMessage | ToolMessage]:
        messages: list[UserMessage | AssistantMessage | ToolMessage] = []

        turns = list(self._thread.turns)
        if self._max_turns:
            turns = turns[-self._max_turns :]
        elif self._max_turns == 0:
            turns = []

        if self._include_tool_calls:
            messages.extend(message for turn in turns for message in turn)
        elif turns:
            previous_turns = turns[:-1]
            messages.extend(
                message
                for turn in previous_turns
                for message in turn
                if isinstance(message, UserMessage)
                or (isinstance(message, AssistantMessage) and not message.tool_calls)
            )

            last_turn = turns[-1]
            messages.extend(message for message in last_turn)

        return iter(messages)
