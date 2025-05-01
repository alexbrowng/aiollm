import typing

from aiollm.chat_completion_events.content_delta_event import ContentDeltaEvent
from aiollm.chat_completion_events.finish_event import FinishEvent
from aiollm.chat_completion_events.start_event import StartEvent
from aiollm.chat_completion_events.tool_call_event import ToolCallEvent
from aiollm.chat_completion_events.usage_event import UsageEvent

ChatCompletionEvent = typing.Union[ContentDeltaEvent, FinishEvent, StartEvent, ToolCallEvent, UsageEvent]
