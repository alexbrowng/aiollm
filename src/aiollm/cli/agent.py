import argparse
import asyncio
import importlib

from aiollm.agents.agent import Agent
from aiollm.chat_completion_events.chat_completion_event import ChatCompletionEvent
from aiollm.contents.text_content import TextContent
from aiollm.messages.user_message import UserMessage
from aiollm.thread.thread import Thread


async def run(agent: Agent):
    thread = Thread()

    while True:
        prompt = input("User: ")

        user_message = UserMessage(content=[TextContent(text=prompt)])

        thread.append(user_message)

        print("Assistant: ", end="", flush=True)
        events: list[ChatCompletionEvent] = []
        async for event in agent.stream(thread):
            events.append(event)

            if event.type == "content_delta":
                print(event.delta, end="", flush=True)

            if event.type == "finish" and event.finish_reason == "stop":
                print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str)

    args = parser.parse_args()

    module_name, agent_name = args.module.rsplit(":", 1)
    module = importlib.import_module(module_name)
    agent = getattr(module, agent_name)

    asyncio.run(run(agent=agent))
