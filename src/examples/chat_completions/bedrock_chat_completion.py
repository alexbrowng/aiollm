import asyncio

from aiollm.chat_completion_events.to_chat_completion import ToChatCompletion
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.user_message import UserMessage
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.providers.bedrock.provider import BedrockProvider
from examples.tools.tools import tools


async def run():
    llm = BedrockProvider()

    thread = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the weather in Tokyo?"),
    ]

    model = Model(id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", name="Claude 3.7 Sonnet", provider="Bedrock")

    parameters = Parameters(max_tokens=256, temperature=0.2)

    events = []
    async for event in llm.chat_completion_stream(
        model=model,
        messages=thread,
        tools=tools,
        parameters=parameters,
        name="assistant",
    ):
        events.append(event)

    chat_completion = ToChatCompletion.from_chat_completion_events(events)

    thread.append(chat_completion.message)

    if chat_completion.finish_reason == "tool_calls":
        tool_messages = await tools.calls(chat_completion.message.tool_calls)
        thread.extend(tool_messages)

    chat_completion = await llm.chat_completion(
        model=model,
        messages=thread,
        tools=tools,
        parameters=parameters,
        name="assistant",
    )

    print(chat_completion)


if __name__ == "__main__":
    asyncio.run(run())
