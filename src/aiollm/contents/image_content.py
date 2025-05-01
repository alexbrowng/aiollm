import dataclasses
import typing


@dataclasses.dataclass(frozen=True, slots=True)
class ImageContent:
    url: str
    detail: typing.Literal["auto", "low", "high"] = "auto"
    type: typing.Literal["image"] = "image"

    def __str__(self) -> str:
        return f"ImageContent(url={self.url}, detail={self.detail})"
