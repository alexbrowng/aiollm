import json

from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_content_part_image_param import ChatCompletionContentPartImageParam, ImageURL
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import ChatCompletionMessageToolCallParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

from aiollm.contents.image_content import ImageContent
from aiollm.contents.json_content import JsonContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.message import Message
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage


class FromMessage:
    @staticmethod
    def from_user_content(
        content: TextContent | ImageContent,
    ) -> ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam:
        if isinstance(content, TextContent):
            return ChatCompletionContentPartTextParam(text=content.text, type="text")
        elif isinstance(content, ImageContent):
            return ChatCompletionContentPartImageParam(
                image_url=ImageURL(url=content.url, detail=content.detail), type="image_url"
            )

    @staticmethod
    def from_assistant_content(
        content: TextContent | JsonContent,
    ) -> ChatCompletionContentPartTextParam:
        if isinstance(content, TextContent):
            return ChatCompletionContentPartTextParam(text=content.text, type="text")
        elif isinstance(content, JsonContent):
            return ChatCompletionContentPartTextParam(text=content.json, type="text")

    @staticmethod
    def from_user_message(message: UserMessage) -> ChatCompletionUserMessageParam:
        param = ChatCompletionUserMessageParam(
            role="user",
            content=[FromMessage.from_user_content(content) for content in message.content],
        )

        if message.name:
            param["name"] = message.name

        return param

    @staticmethod
    def from_assistant_message(message: AssistantMessage) -> ChatCompletionAssistantMessageParam:
        param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=[FromMessage.from_assistant_content(content) for content in message.content],
        )

        if len(message.tool_calls):
            param["tool_calls"] = [
                ChatCompletionMessageToolCallParam(
                    id=tool_call.id,
                    type="function",
                    function={
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                )
                for tool_call in message.tool_calls
            ]

        if message.name:
            param["name"] = message.name

        return param

    @staticmethod
    def from_tool_message(message: ToolMessage) -> ChatCompletionToolMessageParam:
        return ChatCompletionToolMessageParam(role="tool", tool_call_id=message.id, content=message.content)

    @staticmethod
    def from_system_message(message: SystemMessage) -> ChatCompletionSystemMessageParam:
        param = ChatCompletionSystemMessageParam(role="system", content=message.content)

        if message.name:
            param["name"] = message.name

        return param

    @staticmethod
    def from_message(message: Message) -> ChatCompletionMessageParam:
        match message:
            case UserMessage():
                return FromMessage.from_user_message(message)
            case AssistantMessage():
                return FromMessage.from_assistant_message(message)
            case ToolMessage():
                return FromMessage.from_tool_message(message)
            case SystemMessage():
                return FromMessage.from_system_message(message)
