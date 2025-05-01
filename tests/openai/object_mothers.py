import typing

from aiollm.contents.image_content import ImageContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.user_message import UserMessage
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.tools.tool import Tool
from aiollm.tools.tool_call import ToolCall


def make_model(
    id: str = "gpt-4o-mini",
    name: str = "GPT 4o mini",
    provider: str = "OpenAI",
    input_price: float | None = None,
    output_price: float | None = None,
) -> Model:
    return Model(id=id, name=name, provider=provider, input_price=input_price, output_price=output_price)


def make_text_content(text: str = "Hello!") -> TextContent:
    return TextContent(text=text)


def make_image_content(
    url: str = "https://example.com/image.png",
    detail: typing.Literal["auto", "low", "high"] = "auto",
) -> ImageContent:
    return ImageContent(url=url, detail=detail)


def make_user_message(
    content: typing.Optional[list[TextContent | ImageContent]] = None,
    name: str | None = None,
) -> UserMessage:
    if content is None:
        content = [make_text_content()]
    return UserMessage(content=content, name=name)


def make_system_message(
    content: str = "You are a helpful assistant.",
    name: str | None = None,
) -> SystemMessage:
    return SystemMessage(content=content, name=name)


def make_tool(
    name: str = "get_weather",
    description: str = "Get the weather for a location.",
    parameters: typing.Optional[dict] = None,
    strict: bool = False,
) -> Tool:
    if parameters is None:
        parameters = {"location": {"type": "string", "description": "Location to get weather for."}}
    return Tool(name=name, description=description, parameters=parameters, strict=strict)


def make_tool_call(
    id: str = "call_123",
    name: str = "get_weather",
    arguments: typing.Optional[dict] = None,
) -> ToolCall:
    if arguments is None:
        arguments = {"location": "Tokyo"}
    return ToolCall(id=id, name=name, arguments=arguments)


def make_assistant_message(
    content: typing.Optional[list[TextContent]] = None,
    tool_calls: typing.Optional[list[ToolCall]] = None,
    name: str | None = None,
) -> AssistantMessage:
    if content is None:
        content = [make_text_content("Hello! How can I help?")]
    if tool_calls is None:
        tool_calls = []
    return AssistantMessage(content=content, tool_calls=tool_calls, name=name)


def make_parameters(
    max_tokens: int | None = 16,
    temperature: float | None = 0.7,
    top_p: float | None = 1.0,
    frequency_penalty: float | None = 0.0,
    presence_penalty: float | None = 0.0,
    stop: typing.Optional[list[str]] = None,
) -> Parameters:
    return Parameters(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
    )
