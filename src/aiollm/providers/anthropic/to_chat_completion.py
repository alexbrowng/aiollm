import typing

from anthropic.types import Message as AnthropicMessage

from aiollm.chat_completions.chat_completion import ChatCompletion
from aiollm.contents.json_content import JsonContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.assistant_message import AssistantMessage
from aiollm.response_formats.json_schema_response_format import JsonSchemaResponseFormat
from aiollm.response_formats.response_format import ResponseFormat
from aiollm.tools.tool_call import ToolCall
from aiollm.usage.usage import Usage


class ToChatCompletion:
    @staticmethod
    def finish_reason(message: AnthropicMessage) -> typing.Literal["stop", "tool_calls"]:
        if message.stop_reason == "tool_use":
            return "tool_calls"
        else:
            return "stop"

    @staticmethod
    def usage(message: AnthropicMessage) -> Usage:
        return Usage(
            message.usage.input_tokens,
            message.usage.output_tokens,
            message.usage.input_tokens + message.usage.output_tokens,
        )

    @staticmethod
    def message(
        message: AnthropicMessage,
        response_format: ResponseFormat | None = None,
        name: str | None = None,
    ) -> AssistantMessage:
        content = []
        tool_calls = []

        for block in message.content:
            if block.type == "text":
                if response_format and isinstance(response_format, JsonSchemaResponseFormat):
                    content.append(JsonContent(json=block.text))
                else:
                    content.append(TextContent(text=block.text))
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            name=name,
        )

    @staticmethod
    def from_message(
        message: AnthropicMessage,
        response_format: ResponseFormat | None = None,
        name: str | None = None,
    ) -> ChatCompletion:
        return ChatCompletion(
            finish_reason=ToChatCompletion.finish_reason(message),
            message=ToChatCompletion.message(message, response_format, name),
            usage=ToChatCompletion.usage(message),
        )
