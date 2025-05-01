import base64
import dataclasses
import mimetypes
import pathlib


@dataclasses.dataclass(frozen=True, slots=True)
class ImageURL:
    bytes: bytes
    base64: str
    mime_type: str
    url: str

    def __str__(self) -> str:
        return f"ImageURL(bytes={self.bytes}, base64={self.base64}, mime_type={self.mime_type}, url={self.url})"

    @staticmethod
    def from_file_path(file_path: pathlib.Path) -> "ImageURL":
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        file_bytes = file_path.read_bytes()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")
        file_url = f"data:{mime_type};base64,{file_base64}"

        return ImageURL(
            bytes=file_bytes,
            base64=file_base64,
            mime_type=mime_type,
            url=file_url,
        )

    @staticmethod
    def from_base64_url(base64_url: str) -> "ImageURL":
        header, base64_data = base64_url.split(";base64,")
        mime_type = header.split(":")[1]
        file_bytes = base64.b64decode(base64_data)

        return ImageURL(
            bytes=file_bytes,
            base64=base64_data,
            mime_type=mime_type,
            url=base64_url,
        )
