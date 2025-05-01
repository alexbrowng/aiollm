import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict

    def __str__(self) -> str:
        return f"ToolCall(id={self.id}, name={self.name}, arguments={self.arguments})"
