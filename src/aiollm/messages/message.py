import typing

from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage

Message = typing.Union[UserMessage, AssistantMessage, ToolMessage, SystemMessage]
