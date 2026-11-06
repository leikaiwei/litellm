import base64
import struct
import zlib
from io import BytesIO
from typing import Final

import pytest
from PIL import Image

from litellm.litellm_core_utils.image_validation import invalid_inline_image_path


def _image_data(format: str = "PNG") -> str:
    buffer: Final = BytesIO()
    with Image.new("RGB", (2, 2), "red") as image:
        image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def test_rejects_corrupt_deflate_with_valid_png_checksums():
    data: Final = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", b"\x78\x9c\xff\xff\xff\xff")
        + _chunk(b"IEND", b"")
    )
    with Image.open(BytesIO(data)) as header_only:
        header_only.verify()
    encoded: Final = base64.b64encode(data).decode()
    assert (
        invalid_inline_image_path(
            [{"role": "user", "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}]}],
            "input",
        )
        == "input[0].content[0]"
    )


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "GIF"])
def test_accepts_decodable_images(format: str):
    encoded: Final = _image_data(format)
    assert (
        invalid_inline_image_path(
            [{"type": "image_url", "image_url": {"url": f"data:image/{format.lower()};base64,{encoded}"}}],
            "messages",
        )
        is None
    )


@pytest.mark.parametrize("data", ["AAAA", "not base64!", ""])
def test_rejects_damaged_anthropic_tool_result_image(data: str):
    assert (
        invalid_inline_image_path(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
                            ],
                        }
                    ],
                }
            ],
            "messages",
        )
        == "messages[0].content[0].content[0]"
    )


def test_does_not_fetch_urls_or_interpret_text_and_schema_as_images():
    assert (
        invalid_inline_image_path(
            [
                {"type": "input_image", "image_url": "https://example.invalid/image.png"},
                {"type": "input_text", "text": "data:image/png;base64,AAAA"},
                {
                    "type": "function_call",
                    "arguments": '{"type":"input_image","image_url":"data:image/png;base64,AAAA"}',
                },
            ],
            "input",
        )
        is None
    )


def test_checks_responses_tool_output_image():
    assert (
        invalid_inline_image_path(
            [
                {
                    "type": "function_call_output",
                    "output": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
                }
            ],
            "input",
        )
        == "input[0].output[0]"
    )
