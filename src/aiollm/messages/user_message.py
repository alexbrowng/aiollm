import dataclasses
import typing

from aiollm.contents.audio_content import AudioContent
from aiollm.contents.document_content import DocumentContent
from aiollm.contents.image_content import ImageContent
from aiollm.contents.text_content import TextContent


@dataclasses.dataclass(frozen=True, slots=True)
class UserMessage:
    content: str | typing.Sequence[TextContent | ImageContent | DocumentContent | AudioContent]
    name: str | None = None
    role: typing.Literal["user"] = "user"

    def __str__(self) -> str:
        return f"UserMessage(content={self.content}, name={self.name})"
