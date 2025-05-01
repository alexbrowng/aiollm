import dataclasses
import datetime


@dataclasses.dataclass
class ModelFeatures:
    text_input: bool = True
    image_input: bool = True
    text_output: bool = True
    image_output: bool = True
    streaming: bool = True
    function_calling: bool = True
    structured_output: bool = True
    reasoning: bool = False


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
    max_context_tokens: int | None = None
    max_completion_tokens: int | None = None
    knowledge_cutoff: datetime.date | None = None
    price: ModelPrice | None = None
    features: ModelFeatures | None = None
