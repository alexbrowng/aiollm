import dataclasses
import typing


@dataclasses.dataclass(frozen=True, slots=True)
class JsonSchema:
    name: str
    description: str
    strict: bool
    schema: dict[str, typing.Any]

    def __str__(self) -> str:
        return (
            f"JsonSchema(name={self.name}, description={self.description}, strict={self.strict}, schema={self.schema})"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class JsonSchemaResponseFormat:
    json_schema: JsonSchema
    type: typing.Literal["json_schema"] = "json_schema"

    def __str__(self) -> str:
        return f"JsonSchemaResponseFormat(json_schema={self.json_schema}, type={self.type})"
