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
    TOOL_RESULT_TRAILING_TEXT,
    VisionToTextGuardrail,
    ensure_trailing_text_after_tool_result,
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
# 尾部 text 块注入：绕开 opencode 的 tool_use id 注册表校验
#
# opencode 只认自己签发过的 tool id，而 Claude Code 只回传后端给的 id，所以一次
# fallback 到别的后端就永久污染会话（单向棘轮），之后每轮必 400。自编 id 无解（存在性
# 校验），但校验是两段式的：只有请求最后一个 content 块是裸 tool_result 时才触发，
# 一触发就扫全历史。末尾追加一个非空 text 块，校验根本不启动。
# 实测判据见 .ops-runbook/findings/2026-08-01-toolid-trailing-text-绕过.md
# ---------------------------------------------------------------------------

TOOL_RESULT = {"type": "tool_result", "tool_use_id": "call_00_abc", "content": "a.txt"}
OTHER_TOOL_RESULT = {"type": "tool_result", "tool_use_id": "call_00_def", "content": "b.txt"}


def _trailing(messages: List[Dict[str, Any]]) -> Any:
    return messages[-1]["content"][-1]


class TestTrailingTextInjection:
    def test_bare_tool_result_at_end_gets_a_text_block(self):
        """裸 tool_result 收尾会触发校验 -> 400。这是生产故障的直接形状"""
        out = ensure_trailing_text_after_tool_result([{"role": "user", "content": [TOOL_RESULT]}])
        assert out[-1]["content"] == [TOOL_RESULT, {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}]

    def test_injected_text_is_not_blank_or_litellm_would_strip_it(self):
        """
        litellm 的 strip_empty_text_blocks_from_anthropic_messages 用 .strip() 判空，
        会摘掉纯空白 text 块，摘完又变回裸 tool_result 收尾。所以注入内容必须非空白。
        """
        assert TOOL_RESULT_TRAILING_TEXT.strip()

    def test_existing_text_at_end_is_left_alone(self):
        """已有非空 text 收尾时校验不触发，无需注入；重复注入会污染对话"""
        messages = [{"role": "user", "content": [TOOL_RESULT, {"type": "text", "text": "what now?"}]}]
        assert ensure_trailing_text_after_tool_result(messages) == messages

    @pytest.mark.parametrize(
        "blank_block",
        [
            pytest.param({"type": "text", "text": ""}, id="empty-string"),
            pytest.param({"type": "text", "text": "   "}, id="whitespace"),
            pytest.param({"type": "text", "text": None}, id="none"),
            pytest.param({"type": "text"}, id="missing-text-key"),
        ],
    )
    def test_blank_text_at_end_still_gets_injection(self, blank_block):
        """
        Claude Code 会回传 {"type": "text", "text": ""}；litellm 摘掉它之后就成了裸
        tool_result 收尾。所以判定必须跳过末尾的空白 text 块，与 litellm 的行为对齐。

        非 str 的 text 也要算作会被摘掉：litellm 的 _is_empty_text_block 判的是
        `not isinstance(text, str) or not text.strip()`，两边口径必须一致。
        """
        messages = [{"role": "user", "content": [TOOL_RESULT, blank_block]}]
        assert _trailing(ensure_trailing_text_after_tool_result(messages)) == {
            "type": "text",
            "text": TOOL_RESULT_TRAILING_TEXT,
        }

    def test_injection_is_idempotent(self):
        """
        注入完再跑一遍必须是 no-op。这条同时锁住注入内容不能是空白：若换成 ""，
        第二遍会认为仍是裸 tool_result 收尾而再注入一次。
        """
        once = ensure_trailing_text_after_tool_result([{"role": "user", "content": [TOOL_RESULT]}])
        assert ensure_trailing_text_after_tool_result(once) is once

    def test_message_dropped_by_stripping_does_not_hide_a_bare_tool_result(self):
        """
        litellm 摘空块后若整条 content 变空，会把**整条消息**从列表里删掉
        （common_utils.py:990-993），于是前一条又成了末条。

        只看 messages[-1] 会漏掉这个形状：判定时没有 tool_result 所以不注入，
        litellm 丢掉末条后请求变回裸 tool_result 收尾，照样 400 且不报错。
        """
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]

        out = ensure_trailing_text_after_tool_result(messages)

        assert out[0]["content"] == [TOOL_RESULT, {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}]
        assert out[1] == messages[1], "会被 litellm 丢掉的消息不该由本 guardrail 删除"

    def test_message_whose_content_is_already_empty_is_not_treated_as_dropped(self):
        """
        litellm 只丢「摘掉空块后才变空」的消息；content 本来就是 [] 的会原样保留
        （len(filtered) == len(content) 那条分支）。所以它仍是末条，不该往前找。
        """
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "assistant", "content": []},
        ]
        assert ensure_trailing_text_after_tool_result(messages) == messages

    def test_parallel_tool_results_need_only_one_text_block(self):
        """并列多个 tool_result（并行工具调用）只需在末尾补一个"""
        out = ensure_trailing_text_after_tool_result([{"role": "user", "content": [TOOL_RESULT, OTHER_TOOL_RESULT]}])
        assert out[-1]["content"] == [
            TOOL_RESULT,
            OTHER_TOOL_RESULT,
            {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT},
        ]

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([{"role": "user", "content": "plain string"}], id="string-content"),
            pytest.param([{"role": "user", "content": []}], id="empty-content"),
            pytest.param([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], id="text-only"),
            pytest.param([{"role": "user", "content": [{"type": "text", "text": ""}]}], id="all-messages-dropped"),
            pytest.param(
                [{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]}],
                id="assistant-tool-use",
            ),
            pytest.param([], id="no-messages"),
        ],
    )
    def test_no_op_when_last_message_does_not_end_with_tool_result(self, messages):
        assert ensure_trailing_text_after_tool_result(messages) == messages

    def test_only_the_last_message_is_touched(self):
        """历史里的裸 tool_result 不用管：校验只由最后一块触发"""
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [OTHER_TOOL_RESULT]},
        ]
        out = ensure_trailing_text_after_tool_result(messages)
        assert out[0] == messages[0]
        assert out[2]["content"] == [OTHER_TOOL_RESULT, {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}]

    def test_input_is_not_mutated(self):
        messages = [{"role": "user", "content": [TOOL_RESULT]}]
        ensure_trailing_text_after_tool_result(messages)
        assert messages[0]["content"] == [TOOL_RESULT]

    def test_openai_shape_is_left_alone(self):
        """
        OpenAI 形状（末条是 role:"tool"）只能另起一条 user 消息，会造成连续两条 user，
        未实测过。生产走 /v1/messages，进 hook 时是 Anthropic 形状，先留 no-op。
        """
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": "a.txt"}]
        assert ensure_trailing_text_after_tool_result(messages) == messages


@pytest.mark.asyncio
async def test_hook_injects_trailing_text_when_request_has_no_images():
    """
    最关键的回归：生产失败请求大多不带图。尾块修复若挂在图片分支下，无图时 hook
    提前 return None，整个修复就是 no-op —— 那正是这个 bug 的形状。
    """
    router = _router("unused")

    result = await _run(_make_guardrail(router=router), [{"role": "user", "content": [TOOL_RESULT]}])

    assert result is not None, "无图但需要注入尾块时不能返回 None"
    assert _trailing(result["messages"]) == {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}
    router.acompletion.assert_not_called()


@pytest.mark.asyncio
async def test_hook_applies_both_fixes_together():
    """图片替换动 tool_result 内部 content，尾块注入只在末尾追加，两者互不干扰"""
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_00_abc", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{**TOOL_RESULT, "content": [ANTHROPIC_IMAGE]}]},
    ]

    result = await _run(_make_guardrail(router=_router("digits 739")), messages)

    assert result is not None
    tool_result = result["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"] == [{"type": "text", "text": "[Image description: digits 739]"}]
    assert _trailing(result["messages"]) == {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}


@pytest.mark.asyncio
async def test_hook_still_returns_none_when_nothing_needs_changing():
    """无图又无需注入时必须返回 None，不能凭空复制一份 messages 回去"""
    router = _router("unused")
    guardrail = _make_guardrail(router=router)

    assert await _run(guardrail, [{"role": "user", "content": "plain"}]) is None
    assert await _run(guardrail, [{"role": "user", "content": [TOOL_RESULT, {"type": "text", "text": "q"}]}]) is None
    router.acompletion.assert_not_called()


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
