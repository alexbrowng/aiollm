import dataclasses
import typing

from aiollm.sources.base64_source import Base64Source


@dataclasses.dataclass(frozen=True, slots=True)
class AudioContent:
    source: Base64Source
    type: typing.Literal["audio"] = "audio"

    def __str__(self) -> str:
        return f"AudioContent(source={self.source})"
