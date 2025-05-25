import asyncio
import os

from aiollm.messages.system_message import SystemMessage
from aiollm.messages.user_message import UserMessage
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.providers.anthropic.provider import AnthropicProvider
from aiollm.response_formats.text_response_format import TextResponseFormat
from examples.tools.tools import tools


async def main():
    llm = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the weather in Tokyo?"),
    ]

    model = Model(id="claude-3-haiku-20240307", name="Claude 3.5 Haiku", provider="Anthropic")
    parameters = Parameters(max_tokens=256, temperature=0.2)
    response_format = TextResponseFormat()

    chat_completion = await llm.chat_completion(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        parameters=parameters,
        name="assistant",
    )

    messages.append(chat_completion.message)

    if chat_completion.finish_reason == "tool_calls":
        tool_messages = await tools.calls(chat_completion.message.tool_calls)
        messages.extend(tool_messages)

    chat_completion = await llm.chat_completion(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        parameters=parameters,
        name="assistant",
    )

    print(chat_completion)


if __name__ == "__main__":
    asyncio.run(main())
