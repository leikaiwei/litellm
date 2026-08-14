"""
vision_to_text guardrail 的回归测试。

跑法：.venv/bin/python -m pytest local_guardrails/ -q

测试与被测文件同目录，不放 tests/test_litellm/，避免在 upstream 跟踪的路径下留改动。
"""

import asyncio
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
# Claude Code 用 Read 读图片文件后的真实形状：图片嵌在 tool_result 自己的 content 里
ANTHROPIC_TOOL_RESULT = {"type": "tool_result", "tool_use_id": "toolu_01", "content": [ANTHROPIC_IMAGE]}


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
        assert (refs[0].message_index, refs[0].path) == (1, (1,))

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
        out = replace_images_with_text(messages, {(1, (1,)): "DESC"})
        assert out[0] == {"role": "system", "content": "stay brief"}
        assert [part["text"] for part in out[1]["content"]] == ["before", "DESC", "after"]

    def test_input_is_not_mutated(self):
        messages = [{"role": "user", "content": [OPENAI_IMAGE]}]
        replace_images_with_text(messages, {(0, (0,)): "DESC"})
        assert messages[0]["content"][0]["type"] == "image_url"

    def test_no_replacements_returns_equivalent_list(self):
        messages = [{"role": "user", "content": "hi"}]
        assert replace_images_with_text(messages, {}) == messages

    def test_tool_result_envelope_survives_replacement(self):
        """
        只换 tool_result 里面的图片，外壳必须原样保留。

        把整个 tool_result 换成 text 会让前一条 assistant 的 tool_use 失去配对，
        Anthropic 会直接拒掉整个请求。
        """
        messages = [{"role": "user", "content": [ANTHROPIC_TOOL_RESULT]}]
        out = replace_images_with_text(messages, {(0, (0, 0)): "DESC"})
        tool_result = out[0]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "toolu_01"
        assert tool_result["content"] == [{"type": "text", "text": "DESC"}]

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
            {(r.message_index, r.path): f"IMG{i}" for i, r in enumerate(refs)},
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
async def test_image_nested_in_tool_result_is_described():
    """
    Claude Code 用 Read 读图片文件时，图片嵌在 tool_result 的 content 里，不在顶层。

    只扫顶层会把这条主路径上的图片整个漏给纯文本模型（已实测：deepseek 收到原始
    base64，thinking 里在试着自己解码）。
    """
    router = _router("digits 417309")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "read screenshot.png"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {}}]},
        {"role": "user", "content": [ANTHROPIC_TOOL_RESULT, {"type": "text", "text": "what digits?"}]},
    ]

    result = await _run(_make_guardrail(router=router), messages)

    assert router.acompletion.call_count == 1
    assert result is not None
    tool_result = result["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"] == [{"type": "text", "text": "[Image description: digits 417309]"}]
    assert result["messages"][2]["content"][1] == {"type": "text", "text": "what digits?"}
    assert extract_image_references(result["messages"]) == ()


@pytest.mark.asyncio
async def test_nested_image_reaches_vision_model_as_full_data_url():
    router = _router("described")
    await _run(_make_guardrail(router=router), [{"role": "user", "content": [ANTHROPIC_TOOL_RESULT]}])
    sent = router.acompletion.call_args.kwargs["messages"][0]["content"]
    assert sent[1]["image_url"]["url"] == "data:image/png;base64,AAA"


@pytest.mark.asyncio
async def test_tool_use_and_other_messages_survive_nested_replacement():
    """tool_use 必须原样保留，否则它与 tool_result 的配对断掉，Anthropic 会拒掉整个请求"""
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"p": 1}}]},
        {"role": "user", "content": [ANTHROPIC_TOOL_RESULT]},
    ]

    result = await _run(_make_guardrail(router=_router("described")), messages)

    assert result is not None
    assert result["messages"][0] == messages[0]
    assert result["messages"][1]["content"][0]["tool_use_id"] == "toolu_01"


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
async def test_max_images_describes_the_most_recent_not_the_oldest():
    """
    用户当轮刚贴的截图排在最后。取最早 N 张会把它漏掉，等于识图对当前提问无效。
    2026-08-14 生产事故前的实现取的正是最早 N 张。
    """
    router = _router("described")

    result = await _run(
        _make_guardrail(router=router, max_images=1),
        [{"role": "user", "content": [OPENAI_IMAGE, OTHER_IMAGE]}],
    )

    assert router.acompletion.call_count == 1
    assert router.acompletion.call_args.kwargs["messages"][0]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,ZZZ"
    )
    assert result is not None
    assert result["messages"][0]["content"][1] == {"type": "text", "text": "[Image description: described]"}


@pytest.mark.asyncio
async def test_images_over_max_become_placeholders_not_raw_images():
    """
    超限的历史图片必须换成占位文本。原样留下就等于把真图透传给纯文本后端，
    那正是这个 guardrail 要消灭的 400。
    """
    result = await _run(
        _make_guardrail(router=_router("described"), max_images=1),
        [{"role": "user", "content": [OPENAI_IMAGE, OTHER_IMAGE]}],
    )

    assert result is not None
    omitted = result["messages"][0]["content"][0]
    assert omitted["type"] == "text"
    assert "Image omitted" in omitted["text"]
    assert extract_image_references(result["messages"]) == ()


@pytest.mark.asyncio
async def test_max_images_has_a_default_so_long_sessions_cannot_scale_without_bound():
    """
    生产事故的放大源：不限张数时，长会话每轮把全部历史截图重识别一遍，
    识图量随会话轮数线性涨。默认值是唯一挡住这条路的东西。
    """
    guardrail = _make_guardrail(router=_router("described"))

    assert guardrail.max_images is not None and guardrail.max_images <= 8


@pytest.mark.asyncio
async def test_max_images_none_still_describes_everything_when_explicitly_opted_in():
    router = _router(*[f"d{i}" for i in range(6)])

    result = await _run(
        _make_guardrail(router=router, max_images=None),
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"u{i}"}} for i in range(6)]}],
    )

    assert router.acompletion.call_count == 6
    assert result is not None
    assert extract_image_references(result["messages"]) == ()


@pytest.mark.asyncio
async def test_vision_call_omits_guardrails_so_it_cannot_re_enter_this_hook():
    router = _router("described")

    await _run(_make_guardrail(router=router), [{"role": "user", "content": [OPENAI_IMAGE]}])

    kwargs = router.acompletion.call_args.kwargs
    assert "guardrails" not in kwargs
    assert kwargs["model"] == "vision-describer"
    assert kwargs["stream"] is False


@pytest.mark.asyncio
async def test_vision_call_disables_fallbacks():
    """
    识图调用绝不能 fallback。视觉模型组的降级目标往往是纯文本模型，那个模型看不见图
    却会编一段"描述"，会被当成真结果写进 messages —— 比报错更糟。宁可走占位符。
    """
    router = _router("described")

    await _run(_make_guardrail(router=router), [{"role": "user", "content": [OPENAI_IMAGE]}])

    assert router.acompletion.call_args.kwargs["fallbacks"] == []


@pytest.mark.asyncio
async def test_vision_call_retries_within_the_group_by_default():
    """组内重试能换到别的成员，绕开限流或临时不可用的部署"""
    router = _router("described")

    await _run(_make_guardrail(router=router), [{"role": "user", "content": [OPENAI_IMAGE]}])

    assert router.acompletion.call_args.kwargs["num_retries"] == 2


@pytest.mark.asyncio
async def test_vision_num_retries_is_configurable():
    router = _router("described")

    await _run(
        _make_guardrail(router=router, vision_num_retries=0),
        [{"role": "user", "content": [OPENAI_IMAGE]}],
    )

    assert router.acompletion.call_args.kwargs["num_retries"] == 0


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
# 调用放大防护
#
# 2026-08-14 生产事故：视觉模型在数小时内被调用 5.5 万次，峰值 618 RPM。
# 下面每个测试对应一条独立的放大路径，都能在事故版代码上失败。
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_survives_more_images_than_the_proxy_cache_can_hold():
    """
    事故根因之一：识图缓存曾复用 proxy 传进来的 DualCache，那是 user_api_key_cache，
    内存层硬上限 200 条。累积图片数一旦超过 200，命中率不是下降而是直接归零 ——
    每张图每轮都重新识别一次。

    这里用 250 张图跑两轮：第二轮必须一次识图都不发。
    """
    count = 250
    router = AsyncMock()
    router.acompletion.side_effect = [_vision_response(f"d{i}") for i in range(count)]
    guardrail = _make_guardrail(router=router, max_images=None)
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"u{i}"}} for i in range(count)]}
    ]

    await _run(guardrail, messages, cache=DualCache())
    calls_after_first_turn = router.acompletion.call_count
    second = await _run(guardrail, messages, cache=DualCache())

    assert calls_after_first_turn == count
    assert router.acompletion.call_count == count, "第二轮应全部命中缓存，一次识图都不该发"
    assert second is not None


@pytest.mark.asyncio
async def test_vision_cache_does_not_evict_the_proxy_auth_cache():
    """
    识图键 TTL 3600 远长于认证键的 60，而 InMemoryCache 按到期时间驱逐。
    共用一个 cache 时，识图流量会把 key/team/user 认证条目挤干，
    迫使每个请求回 DB 查鉴权。识图必须完全不碰传入的 cache。
    """
    shared = DualCache()
    for i in range(50):
        await shared.async_set_cache(key=f"auth_key_{i}", value={"user": i}, ttl=60)

    router = AsyncMock()
    router.acompletion.side_effect = [_vision_response(f"d{i}") for i in range(400)]
    await _run(
        _make_guardrail(router=router, max_images=None),
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"u{i}"}} for i in range(400)]}],
        cache=shared,
    )

    survivors = [i for i in range(50) if await shared.async_get_cache(key=f"auth_key_{i}") is not None]
    assert len(survivors) == 50, f"识图流量驱逐了 {50 - len(survivors)} 个认证缓存条目"


@pytest.mark.asyncio
async def test_same_image_arriving_concurrently_is_described_once():
    """
    缓存只在第一次返回后才写入。在那之前，同一张图的 N 个并发请求全是 miss，
    于是发 N 次识图调用。长会话逐轮重发同批图片时这个倍数直接乘进放大链条。
    """
    calls = 0

    async def slow_vision(**_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return _vision_response("described")

    router = AsyncMock()
    router.acompletion = slow_vision
    guardrail = _make_guardrail(router=router)
    messages = [{"role": "user", "content": [OPENAI_IMAGE]}]

    results = await asyncio.gather(*(_run(guardrail, messages, cache=DualCache()) for _ in range(10)))

    assert calls == 1, f"同一张图的 10 个并发请求发了 {calls} 次识图调用"
    assert all(r is not None for r in results)
    assert {r["messages"][0]["content"][0]["text"] for r in results} == {"[Image description: described]"}


@pytest.mark.asyncio
async def test_client_disconnect_on_the_leading_request_does_not_fail_the_waiters():
    """
    in-flight 去重让多个请求共享一次识图调用。领头那个请求的客户端断连（CancelledError）
    绝不能顺着共享 future 传给其他请求 —— 它们各自的连接还活着，应该 fail-open。
    """

    async def slow_vision(**_kwargs):
        await asyncio.sleep(5)
        raise AssertionError("unreachable")

    router = AsyncMock()
    router.acompletion = slow_vision
    guardrail = _make_guardrail(router=router)
    messages = [{"role": "user", "content": [OPENAI_IMAGE]}]

    leading = asyncio.create_task(_run(guardrail, messages, cache=DualCache()))
    await asyncio.sleep(0.02)
    waiter = asyncio.create_task(_run(guardrail, messages, cache=DualCache()))
    await asyncio.sleep(0.02)
    leading.cancel()

    with pytest.raises(asyncio.CancelledError):
        await leading
    result = await asyncio.wait_for(waiter, timeout=2)

    assert result is not None
    assert result["messages"][0]["content"][0]["text"].startswith("[Image could not")
    assert guardrail._in_flight == {}, "in-flight 条目泄漏会让这张图永久命中一个已死的 future"


@pytest.mark.asyncio
async def test_concurrent_vision_calls_are_capped():
    """
    asyncio.gather 不限并发：一次请求 N 张图就是 N 个同时在飞的识图调用，
    会瞬间打穿视觉模型的 RPM 配额并触发限流雪崩。
    """
    in_flight = 0
    peak = 0

    async def counting_vision(**_kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return _vision_response("described")

    router = AsyncMock()
    router.acompletion = counting_vision
    images = [{"type": "image_url", "image_url": {"url": f"u{i}"}} for i in range(40)]

    await _run(
        _make_guardrail(router=router, max_images=None),
        [{"role": "user", "content": images}],
    )

    assert peak <= 8, f"峰值并发识图 {peak} 个，未受限流保护"


@pytest.mark.asyncio
async def test_a_failing_vision_model_is_not_retried_for_every_image_every_turn():
    """
    视觉模型被打限流后，失败结果不入缓存，于是每轮都对全部图片重试一遍 ——
    限流越严重、重试越多，形成正反馈。in-flight 去重至少要把同一轮内
    同一张图的重复尝试压掉。
    """
    calls = 0

    async def failing_vision(**_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        raise RuntimeError("rate limited")

    router = AsyncMock()
    router.acompletion = failing_vision
    guardrail = _make_guardrail(router=router)
    messages = [{"role": "user", "content": [OPENAI_IMAGE]}]

    results = await asyncio.gather(*(_run(guardrail, messages, cache=DualCache()) for _ in range(10)))

    assert calls == 1, f"同一张图的 10 个并发请求在失败路径上发了 {calls} 次识图调用"
    assert all(r is not None for r in results)
    assert all(r["messages"][0]["content"][0]["text"].startswith("[Image could not") for r in results)


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
