import dataclasses

from aiollm.json_schema.object import Object


@dataclasses.dataclass(frozen=True, slots=True)
class Tool:
    name: str
    parameters: Object
    description: str | None = None
    strict: bool = True
    instructions: str | None = None

    def __str__(self) -> str:
        return f"Tool(name={self.name}, parameters={self.parameters}, description={self.description}, strict={self.strict})"
