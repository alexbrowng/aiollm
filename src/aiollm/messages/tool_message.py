import dataclasses
import typing

from aiollm.contents.text_content import TextContent


@dataclasses.dataclass(frozen=True, slots=True)
class ToolMessage:
    id: str
    content: str | typing.Sequence[TextContent]
    name: str | None = None
    role: typing.Literal["tool"] = "tool"

    def __str__(self) -> str:
        return f"ToolMessage(id={self.id}, content={self.content}, name={self.name})"
