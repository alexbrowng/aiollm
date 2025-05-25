import typing

from types_aiobotocore_bedrock_runtime.type_defs import ContentBlockTypeDef, MessageTypeDef

from aiollm.contents.audio_content import AudioContent
from aiollm.contents.document_content import DocumentContent
from aiollm.contents.image_content import ImageContent
from aiollm.contents.json_content import JsonContent
from aiollm.contents.refusal_content import RefusalContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage

IMAGE_FORMAT = typing.Literal["gif", "jpeg", "png", "webp"]
FILE_FORMAT = typing.Literal["csv", "doc", "docx", "html", "md", "pdf", "txt", "xls", "xlsx"]


class FromMessage:
    @staticmethod
    def from_system_message(message: SystemMessage) -> ContentBlockTypeDef:
        if isinstance(message.content, str):
            return {"text": message.content}

        return {"text": "\n".join([content.text for content in message.content])}

    @staticmethod
    def from_user_content(
        content: str | TextContent | ImageContent | DocumentContent | AudioContent,
    ) -> ContentBlockTypeDef:
        if isinstance(content, str):
            return {"text": content}

        if isinstance(content, TextContent):
            return {"text": content.text}

        if isinstance(content, ImageContent):
            _type, subtype = content.source.media_type.split("/")
            return {
                "image": {
                    "format": typing.cast(IMAGE_FORMAT, subtype),
                    "source": {"bytes": content.source.bytes},
                }
            }

        if isinstance(content, DocumentContent):
            _type, subtype = content.source.media_type.split("/")
            return {
                "document": {
                    "format": typing.cast(FILE_FORMAT, subtype),
                    "name": content.name,
                    "source": {"bytes": content.source.bytes},
                }
            }

        if isinstance(content, AudioContent):
            raise NotImplementedError("Audio content is not supported")

    @staticmethod
    def from_user_message(message: UserMessage) -> MessageTypeDef:
        content = []

        if isinstance(message.content, str):
            content.append(FromMessage.from_user_content(message.content))
        else:
            content.extend([FromMessage.from_user_content(content) for content in message.content])

        return {"role": "user", "content": content}

    @staticmethod
    def from_assistant_content(content: str | TextContent | JsonContent | RefusalContent) -> ContentBlockTypeDef:
        if isinstance(content, str):
            return {"text": content}

        if isinstance(content, TextContent):
            return {"text": content.text}

        if isinstance(content, JsonContent):
            return {"text": content.json}

        if isinstance(content, RefusalContent):
            return {"text": content.refusal}

    @staticmethod
    def from_tool_content(content: str | TextContent) -> ContentBlockTypeDef:
        if isinstance(content, str):
            return {"text": content}

        if isinstance(content, TextContent):
            return {"text": content.text}

    @staticmethod
    def from_assistant_message(message: AssistantMessage) -> MessageTypeDef:
        content = []

        if message.content:
            if isinstance(message.content, str):
                content.append(FromMessage.from_assistant_content(message.content))
            else:
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

        return {"role": "assistant", "content": content}

    @staticmethod
    def from_tool_message(message: ToolMessage) -> MessageTypeDef:
        content = []

        if isinstance(message.content, str):
            content.append(FromMessage.from_tool_content(message.content))
        else:
            content.extend([FromMessage.from_tool_content(content) for content in message.content])

        return {"role": "user", "content": [{"toolResult": {"toolUseId": message.id, "content": content}}]}

    @staticmethod
    def from_message(message: UserMessage | AssistantMessage | ToolMessage) -> MessageTypeDef:
        match message:
            case UserMessage():
                return FromMessage.from_user_message(message)
            case AssistantMessage():
                return FromMessage.from_assistant_message(message)
            case ToolMessage():
                return FromMessage.from_tool_message(message)
