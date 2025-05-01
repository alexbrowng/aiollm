import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict
    strict: bool = True

    def __str__(self) -> str:
        return f"Tool(name={self.name}, description={self.description}, parameters={self.parameters}, strict={self.strict})"
