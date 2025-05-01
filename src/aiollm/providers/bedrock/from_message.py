import typing

from types_aiobotocore_bedrock_runtime.type_defs import ContentBlockTypeDef, MessageTypeDef

from aiollm.contents.image_content import ImageContent
from aiollm.contents.json_content import JsonContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage
from aiollm.utils.image_url import ImageURL


class FromMessage:
    @staticmethod
    def from_system_message(message: SystemMessage) -> ContentBlockTypeDef:
        return {"text": message.content}

    @staticmethod
    def from_user_content(content: TextContent | ImageContent) -> ContentBlockTypeDef:
        if isinstance(content, TextContent):
            return {"text": content.text}
        elif isinstance(content, ImageContent):
            image_url = ImageURL.from_base64_url(content.url)
            _type, subtype = image_url.mime_type.split("/")
            return {
                "image": {
                    "format": typing.cast(typing.Literal["gif", "jpeg", "png", "webp"], subtype),
                    "source": {"bytes": image_url.bytes},
                }
            }

    @staticmethod
    def from_user_message(message: UserMessage) -> MessageTypeDef:
        return {
            "role": "user",
            "content": [FromMessage.from_user_content(content) for content in message.content],
        }

    @staticmethod
    def from_assistant_content(content: TextContent | JsonContent) -> ContentBlockTypeDef:
        if isinstance(content, TextContent):
            return {"text": content.text}
        elif isinstance(content, JsonContent):
            return {"text": content.json}

    @staticmethod
    def from_assistant_message(message: AssistantMessage) -> MessageTypeDef:
        content = []

        if message.content:
            content.extend([FromMessage.from_assistant_content(content) for content in message.content])

        if message.tool_calls:
            content.extend(
                [
                    {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            )

        return {
            "role": "assistant",
            "content": content,
        }

    @staticmethod
    def from_tool_message(message: ToolMessage) -> MessageTypeDef:
        return {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": message.id, "content": [{"text": message.content}]}}],
        }

    @staticmethod
    def from_message(message: UserMessage | AssistantMessage | ToolMessage) -> MessageTypeDef:
        match message:
            case UserMessage():
                return FromMessage.from_user_message(message)
            case AssistantMessage():
                return FromMessage.from_assistant_message(message)
            case ToolMessage():
                return FromMessage.from_tool_message(message)
