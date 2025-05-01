from aiollm.contents.image_content import ImageContent
from aiollm.contents.text_content import TextContent


def test_text_content_creation():
    content = TextContent(text="Hello!")
    assert content.text == "Hello!"
    assert content.type == "text"


def test_image_content_creation():
    content = ImageContent(url="http://img", detail="low")
    assert content.url == "http://img"
    assert content.detail == "low"
    assert content.type == "image"
