"""Validate inline images without fetching caller-controlled URLs."""

import base64
import binascii
from collections.abc import Iterator, Mapping
from io import BytesIO
from typing import Final

from PIL import Image
from pydantic import TypeAdapter

_OBJECT: Final = TypeAdapter(dict[str, object])
_ITEMS: Final = TypeAdapter(tuple[object, ...])
_IMAGE_TYPES: Final = frozenset(("image/png", "image/jpeg", "image/webp", "image/gif"))


def _inline_image_data(block: Mapping[str, object]) -> str | None:
    if block.get("type") == "image":
        source: Final = block.get("source")
        if isinstance(source, Mapping):
            typed_source: Final = _OBJECT.validate_python(source)
            data: Final = typed_source.get("data")
            if (
                typed_source.get("type") == "base64"
                and typed_source.get("media_type") in _IMAGE_TYPES
                and isinstance(data, str)
            ):
                return data
        return None
    if block.get("type") not in ("input_image", "image_url"):
        return None
    image_url: Final = block.get("image_url")
    url: Final = _OBJECT.validate_python(image_url).get("url") if isinstance(image_url, Mapping) else image_url
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, separator, encoded = url.partition(",")
    if separator and header[5:].split(";")[0] in _IMAGE_TYPES and header.endswith(";base64"):
        return encoded
    return None


def _image_blocks(items: object, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(items, (list, tuple)):
        for index, item in enumerate(_ITEMS.validate_python(items)):
            yield from _image_blocks(item, f"{path}[{index}]")
        return
    if not isinstance(items, Mapping):
        return
    block: Final = _OBJECT.validate_python(items)
    data: Final = _inline_image_data(block)
    if data is not None:
        yield path, data
    yield from _image_blocks(block.get("content"), f"{path}.content")
    if block.get("type") == "function_call_output":
        yield from _image_blocks(block.get("output"), f"{path}.output")


def invalid_inline_image_path(items: object, path: str) -> str | None:
    return next((image_path for image_path, encoded in _image_blocks(items, path) if not _valid_image(encoded)), None)


def _valid_image(encoded: str) -> bool:
    try:
        data: Final = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(data)) as image:
            image.verify()
        # 校验压缩像素流，避免只读图片头而漏过损坏的 PNG。
        with Image.open(BytesIO(data)) as decoded:
            decoded.load()
    except (ValueError, OSError, SyntaxError, binascii.Error, Image.DecompressionBombError):
        return False
    return True
