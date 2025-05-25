import dataclasses


@dataclasses.dataclass
class ModelPrice:
    input: float | None = None
    cached_input: float | None = None
    output: float | None = None


@dataclasses.dataclass
class Model:
    id: str
    name: str
    provider: str
    price: ModelPrice | None = None
