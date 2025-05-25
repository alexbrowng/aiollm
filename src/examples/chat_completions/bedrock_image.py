import asyncio
import pathlib

from aiollm.contents.image_content import ImageContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.user_message import UserMessage
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.providers.bedrock.provider import BedrockProvider
from aiollm.response_formats.text_response_format import TextResponseFormat
from aiollm.sources.base64_source import Base64Source

BASE_PATH = pathlib.Path(__file__).parent.parent
IMAGES_PATH = BASE_PATH / "images"


async def run():
    llm = BedrockProvider()

    image_path = IMAGES_PATH / "tesseract_on_ocrfeeder.png"
    image = Base64Source.from_file_path(image_path)

    thread = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content=[TextContent(text="What is the content of the image?"), ImageContent(source=image)]),
    ]

    model = Model(id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", name="Claude 3.7 Sonnet", provider="Bedrock")

    parameters = Parameters(max_tokens=256, temperature=0.2)

    response_format = TextResponseFormat()
    chat_completion = await llm.chat_completion(
        model=model,
        messages=thread,
        parameters=parameters,
        response_format=response_format,
        name="assistant",
    )
    print(chat_completion)


if __name__ == "__main__":
    asyncio.run(run())
