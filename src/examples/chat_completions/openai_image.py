import asyncio
import pathlib

from aiollm.contents.image_content import ImageContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.user_message import UserMessage
from aiollm.models.model import Model
from aiollm.parameters.parameters import Parameters
from aiollm.providers.openai.provider import OpenAIProvider
from aiollm.response_formats.text_response_format import TextResponseFormat
from aiollm.utils.image_url import ImageURL

BASE_PATH = pathlib.Path(__file__).parent
IMAGES_PATH = BASE_PATH / "images"


async def run():
    llm = OpenAIProvider()

    image_path = IMAGES_PATH / "tesseract_on_ocrfeeder.png"
    image = ImageURL.from_file_path(image_path)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=[
                TextContent(text="What is the content of the image?"),
                ImageContent(url=image.url),
            ]
        ),
    ]

    model = Model(id="gpt-4o-mini-2024-07-18", name="GPT 4o mini", provider="OpenAI")
    parameters = Parameters(max_tokens=256, temperature=0.2)
    response_format = TextResponseFormat()
    chat_completion = await llm.chat_completion(
        model=model,
        messages=messages,
        parameters=parameters,
        response_format=response_format,
        name="assistant",
    )
    print(chat_completion)


if __name__ == "__main__":
    asyncio.run(run())
