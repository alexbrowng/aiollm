from typing import Union

from aiollm.contents.image_content import ImageContent
from aiollm.contents.text_content import TextContent
from aiollm.messages.assistant_message import AssistantMessage
from aiollm.messages.system_message import SystemMessage
from aiollm.messages.tool_message import ToolMessage
from aiollm.messages.user_message import UserMessage
from aiollm.tools.tool_call import ToolCall


def test_system_message_creation():
    msg = SystemMessage(content="Test system", name="sys1")
    assert msg.content == "Test system"
    assert msg.name == "sys1"
    assert msg.role == "system"


def test_user_message_with_text_content():
    content: list[Union[TextContent, ImageContent]] = [TextContent(text="Hello!")]
    msg = UserMessage(content=content, name="user1")
    assert msg.content == content
    assert msg.name == "user1"
    assert msg.role == "user"


def test_user_message_with_image_content():
    img = ImageContent(url="http://img", detail="high")
    content: list[Union[TextContent, ImageContent]] = [img]
    msg = UserMessage(content=content)
    assert isinstance(msg.content[0], ImageContent)
    assert msg.content[0].url == "http://img"
    assert msg.content[0].detail == "high"
    assert msg.role == "user"


def test_assistant_message_with_tool_calls():
    content = [TextContent(text="Hi!")]
    tool_calls = [ToolCall(id="call1", name="tool", arguments={"foo": "bar"})]
    msg = AssistantMessage(content=content, tool_calls=tool_calls, name="asst1")
    assert msg.content == content
    assert msg.tool_calls == tool_calls
    assert msg.name == "asst1"
    assert msg.role == "assistant"


def test_tool_message_creation():
    msg = ToolMessage(id="call1", content="tool output")
    assert msg.content == "tool output"
    assert msg.id == "call1"
    assert msg.role == "tool"
