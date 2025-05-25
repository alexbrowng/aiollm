import dataclasses
import typing

from aiollm.contents.text_content import TextContent


@dataclasses.dataclass(frozen=True, slots=True)
class SystemMessage:
    content: str | typing.Sequence[TextContent]
    name: str | None = None
    role: typing.Literal["system"] = "system"

    def __str__(self) -> str:
        return f"SystemMessage(content={self.content}, name={self.name})"
