"""
vision_to_text guardrail 的回归测试。

跑法：.venv/bin/python -m pytest local_guardrails/ -q

测试与被测文件同目录，不放 tests/test_litellm/，避免在 upstream 跟踪的路径下留改动。
"""

import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import ModelResponse
from vision_to_text import (
    VisionToTextGuardrail,
    extract_image_references,
    image_url_for_vision_call,
    replace_images_with_text,
)

OPENAI_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
ANTHROPIC_IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
}
OTHER_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZ"}}


def _vision_response(text: Optional[str]) -> ModelResponse:
    return ModelResponse(choices=[{"index": 0, "message": {"role": "assistant", "content": text}}])


def _router(*texts: Optional[str]) -> AsyncMock:
    router = AsyncMock()
    if len(texts) == 1:
        router.acompletion.return_value = _vision_response(texts[0])
    else:
        router.acompletion.side_effect = [_vision_response(text) for text in texts]
    return router


def _make_guardrail(router: Optional[AsyncMock] = None, **overrides: Any) -> VisionToTextGuardrail:
    kwargs: Dict[str, Any] = {
        "guardrail_name": "vision-to-text",
        "event_hook": "pre_call",
        "vision_model": "vision-describer",
        "llm_router": router if router is not None else AsyncMock(),
    }
    kwargs.update(overrides)
    return VisionToTextGuardrail(**kwargs)


async def _run(
    guardrail: VisionToTextGuardrail,
    messages: List[Dict[str, Any]],
    cache: Optional[DualCache] = None,
) -> Optional[dict]:
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=cache or DualCache(),
        data={"model": "deepseek-v4-flash", "messages": messages},
        call_type="acompletion",
    )


# ---------------------------------------------------------------------------
# 形状处理：/v1/chat/completions 与 /v1/messages 的图片块形状不同
# ---------------------------------------------------------------------------


class TestImageExtraction:
    def test_openai_image_url_object(self):
        assert image_url_for_vision_call(OPENAI_IMAGE) == "data:image/png;base64,AAA"

    def test_openai_image_url_bare_string(self):
        assert image_url_for_vision_call({"type": "image_url", "image_url": "https://x/y.png"}) == "https://x/y.png"

    def test_anthropic_base64_keeps_data_url_prefix(self):
        """litellm 自带的 extract_images_from_message 会剥掉前缀，剥掉后无法回传给视觉模型"""
        assert image_url_for_vision_call(ANTHROPIC_IMAGE) == "data:image/png;base64,AAA"

    def test_anthropic_without_media_type_defaults_to_png(self):
        item = {"type": "image", "source": {"type": "base64", "data": "AAA"}}
        assert image_url_for_vision_call(item) == "data:image/png;base64,AAA"

    def test_anthropic_url_source(self):
        item = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
        assert image_url_for_vision_call(item) == "https://x/y.png"

    def test_anthropic_data_already_prefixed_is_not_double_prefixed(self):
        item = {"type": "image", "source": {"type": "base64", "data": "data:image/webp;base64,AAA"}}
        assert image_url_for_vision_call(item) == "data:image/webp;base64,AAA"

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {}},
            {"type": "image_url", "image_url": ""},
            {"type": "image", "source": {}},
            {"type": "image", "source": "not-a-dict"},
            {"type": "image"},
            {},
        ],
    )
    def test_non_images_return_none(self, item):
        assert image_url_for_vision_call(item) is None

    def test_positions_are_preserved(self):
        messages = [
            {"role": "system", "content": "hi"},
            {"role": "user", "content": [{"type": "text", "text": "q"}, OPENAI_IMAGE]},
        ]
        refs = extract_image_references(messages)
        assert len(refs) == 1
        assert (refs[0].message_index, refs[0].content_index) == (1, 1)

    @pytest.mark.parametrize(
        "content",
        ["plain string", [{"type": "text", "text": "hi"}], ["not-a-dict"], []],
    )
    def test_no_images_found(self, content):
        assert extract_image_references([{"role": "user", "content": content}]) == ()


class TestReplacement:
    def test_only_referenced_parts_are_replaced(self):
        messages = [
            {"role": "system", "content": "stay brief"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    OPENAI_IMAGE,
                    {"type": "text", "text": "after"},
                ],
            },
        ]
        out = replace_images_with_text(messages, {(1, 1): "DESC"})
        assert out[0] == {"role": "system", "content": "stay brief"}
        assert [part["text"] for part in out[1]["content"]] == ["before", "DESC", "after"]

    def test_input_is_not_mutated(self):
        messages = [{"role": "user", "content": [OPENAI_IMAGE]}]
        replace_images_with_text(messages, {(0, 0): "DESC"})
        assert messages[0]["content"][0]["type"] == "image_url"

    def test_no_replacements_returns_equivalent_list(self):
        messages = [{"role": "user", "content": "hi"}]
        assert replace_images_with_text(messages, {}) == messages

    def test_round_trip_leaves_no_images_behind(self):
        messages = [
            {
                "role": "user",
                "content": [
                    OPENAI_IMAGE,
                    {"type": "text", "text": "and"},
                    ANTHROPIC_IMAGE,
                ],
            }
        ]
        refs = extract_image_references(messages)
        out = replace_images_with_text(
            messages,
            {(r.message_index, r.content_index): f"IMG{i}" for i, r in enumerate(refs)},
        )
        assert [part["text"] for part in out[0]["content"]] == ["IMG0", "and", "IMG1"]
        assert extract_image_references(out) == ()


# ---------------------------------------------------------------------------
# hook 行为
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_part",
    [pytest.param(OPENAI_IMAGE, id="openai"), pytest.param(ANTHROPIC_IMAGE, id="anthropic")],
)
async def test_image_replaced_with_description_on_both_endpoints(image_part):
    guardrail = _make_guardrail(router=_router("digits 739"))
    messages = [{"role": "user", "content": [{"type": "text", "text": "read it"}, image_part]}]

    result = await _run(guardrail, messages)

    assert result is not None
    assert result["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "read it"},
                {"type": "text", "text": "[Image description: digits 739]"},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_image_reaches_vision_model_as_full_data_url():
    router = _router("described")
    await _run(_make_guardrail(router=router), [{"role": "user", "content": [ANTHROPIC_IMAGE]}])
    sent = router.acompletion.call_args.kwargs["messages"][0]["content"]
    assert sent[1]["image_url"]["url"] == "data:image/png;base64,AAA"


@pytest.mark.asyncio
async def test_request_without_images_is_untouched():
    router = _router("unused")
    guardrail = _make_guardrail(router=router)

    assert await _run(guardrail, [{"role": "user", "content": "plain"}]) is None
    assert await _run(guardrail, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]) is None
    router.acompletion.assert_not_called()


@pytest.mark.asyncio
async def test_other_messages_and_non_image_parts_are_preserved():
    guardrail = _make_guardrail(router=_router("described"))
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "before"}, OPENAI_IMAGE]},
    ]

    result = await _run(guardrail, messages)

    assert result is not None
    assert result["messages"][0] == {"role": "system", "content": "be brief"}
    assert result["messages"][1]["content"][0] == {"type": "text", "text": "before"}


@pytest.mark.asyncio
async def test_data_keys_other_than_messages_survive():
    guardrail = _make_guardrail(router=_router("described"))
    data = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": [OPENAI_IMAGE]}], "temperature": 0.5}

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(), data=data, call_type="acompletion"
    )

    assert result is not None
    assert result["temperature"] == 0.5
    assert result["model"] == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_vision_failure_fails_open_with_placeholder():
    """识图失败不能让请求挂掉，但模型必须知道这里原本有图"""
    router = AsyncMock()
    router.acompletion.side_effect = RuntimeError("insufficient quota")

    result = await _run(_make_guardrail(router=router), [{"role": "user", "content": [OPENAI_IMAGE]}])

    assert result is not None
    part = result["messages"][0]["content"][0]
    assert part["type"] == "text"
    assert part["text"].startswith("[Image could not be processed:")


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [None, "", "   "], ids=["none", "empty", "blank"])
async def test_empty_description_falls_back_to_placeholder(empty):
    result = await _run(_make_guardrail(router=_router(empty)), [{"role": "user", "content": [OPENAI_IMAGE]}])
    assert result is not None
    assert result["messages"][0]["content"][0]["text"].startswith("[Image could not be processed:")


@pytest.mark.asyncio
async def test_failed_description_is_not_cached_so_next_request_retries():
    router = AsyncMock()
    router.acompletion.side_effect = [RuntimeError("timeout"), _vision_response("recovered")]
    guardrail = _make_guardrail(router=router)
    cache = DualCache()
    messages = [{"role": "user", "content": [OPENAI_IMAGE]}]

    first = await _run(guardrail, messages, cache=cache)
    second = await _run(guardrail, messages, cache=cache)

    assert first is not None and first["messages"][0]["content"][0]["text"].startswith("[Image could not")
    assert second is not None
    assert second["messages"][0]["content"][0]["text"] == "[Image description: recovered]"


@pytest.mark.asyncio
async def test_duplicate_image_in_one_request_is_described_once():
    router = _router("described")
    guardrail = _make_guardrail(router=router)
    messages = [
        {"role": "user", "content": [OPENAI_IMAGE]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [OPENAI_IMAGE]},
    ]

    result = await _run(guardrail, messages)

    assert router.acompletion.call_count == 1
    assert result is not None
    assert result["messages"][0]["content"][0]["text"] == result["messages"][2]["content"][0]["text"]


@pytest.mark.asyncio
async def test_description_is_cached_across_requests_to_keep_prompt_prefix_stable():
    """改写前缀会破坏 prompt caching，除非同一张图每次产出逐字节相同的文本"""
    router = _router("described")
    guardrail = _make_guardrail(router=router)
    cache = DualCache()
    messages = [{"role": "user", "content": [OPENAI_IMAGE]}]

    first = await _run(guardrail, messages, cache=cache)
    second = await _run(guardrail, messages, cache=cache)

    assert router.acompletion.call_count == 1
    assert first is not None and second is not None
    assert first["messages"] == second["messages"]


@pytest.mark.asyncio
async def test_different_images_get_their_own_descriptions():
    router = _router("first", "second")

    result = await _run(_make_guardrail(router=router), [{"role": "user", "content": [OPENAI_IMAGE, OTHER_IMAGE]}])

    assert router.acompletion.call_count == 2
    assert result is not None
    assert [part["text"] for part in result["messages"][0]["content"]] == [
        "[Image description: first]",
        "[Image description: second]",
    ]


@pytest.mark.asyncio
async def test_max_images_leaves_excess_untouched():
    router = _router("described")

    result = await _run(
        _make_guardrail(router=router, max_images=1),
        [{"role": "user", "content": [OPENAI_IMAGE, OTHER_IMAGE]}],
    )

    assert router.acompletion.call_count == 1
    assert result is not None
    assert result["messages"][0]["content"][0]["type"] == "text"
    assert result["messages"][0]["content"][1] == OTHER_IMAGE


@pytest.mark.asyncio
async def test_vision_call_omits_guardrails_so_it_cannot_re_enter_this_hook():
    router = _router("described")

    await _run(_make_guardrail(router=router), [{"role": "user", "content": [OPENAI_IMAGE]}])

    kwargs = router.acompletion.call_args.kwargs
    assert "guardrails" not in kwargs
    assert kwargs["model"] == "vision-describer"
    assert kwargs["stream"] is False


@pytest.mark.asyncio
async def test_custom_prompt_and_template_are_used():
    router = _router("described")
    guardrail = _make_guardrail(
        router=router,
        vision_prompt="transcribe only",
        description_template="<img>{description}</img>",
    )

    result = await _run(guardrail, [{"role": "user", "content": [OPENAI_IMAGE]}])

    assert result is not None
    assert result["messages"][0]["content"][0]["text"] == "<img>described</img>"
    assert router.acompletion.call_args.kwargs["messages"][0]["content"][0]["text"] == "transcribe only"


# ---------------------------------------------------------------------------
# 配置契约
# ---------------------------------------------------------------------------


def test_only_pre_call_is_supported_because_later_hooks_cannot_change_the_request():
    assert _make_guardrail().supported_event_hooks == [GuardrailEventHooks.pre_call]


def test_vision_model_is_required():
    with pytest.raises(ValueError, match="vision_model"):
        VisionToTextGuardrail(guardrail_name="v2t", event_hook="pre_call")


def test_unknown_litellm_params_are_absorbed():
    """initialize_custom_guardrail 会把整个 litellm_params 展开传进来，多余字段不能炸"""
    guardrail = _make_guardrail(api_key="unused", api_base="unused", some_future_field=1)
    assert guardrail.vision_model == "vision-describer"


def test_default_on_defaults_to_false_so_it_never_becomes_globally_always_on():
    assert _make_guardrail().default_on is False
