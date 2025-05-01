import dataclasses
import typing


@dataclasses.dataclass(frozen=True, slots=True)
class ToolMessage:
    id: str
    content: str
    role: typing.Literal["tool"] = "tool"

    def __str__(self) -> str:
        return f"ToolMessage(id={self.id}, content={self.content})"
