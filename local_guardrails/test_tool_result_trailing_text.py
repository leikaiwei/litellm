"""
tool_result_trailing_text guardrail 的回归测试。

跑法：.venv/bin/python -m pytest local_guardrails/ -q

测试与被测文件同目录，不放 tests/test_litellm/，避免在 upstream 跟踪的路径下留改动。
"""

import os
import sys
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from tool_result_trailing_text import (
    TOOL_RESULT_TRAILING_TEXT,
    ToolResultTrailingTextGuardrail,
    ensure_trailing_text_after_tool_result,
)

TOOL_RESULT = {"type": "tool_result", "tool_use_id": "call_00_abc", "content": "a.txt"}
OTHER_TOOL_RESULT = {"type": "tool_result", "tool_use_id": "call_00_def", "content": "b.txt"}
TRAILING_BLOCK = {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}


def _trailing(messages: List[Dict[str, Any]]) -> Any:
    return messages[-1]["content"][-1]


async def _run(
    guardrail: ToolResultTrailingTextGuardrail,
    messages: List[Dict[str, Any]],
) -> Optional[dict]:
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data={"model": "deepseek-v4-flash", "messages": messages},
        call_type="acompletion",
    )


def _make_guardrail(**overrides: Any) -> ToolResultTrailingTextGuardrail:
    kwargs: Dict[str, Any] = {
        "guardrail_name": "tool-result-trailing-text",
        "event_hook": "pre_call",
    }
    kwargs.update(overrides)
    return ToolResultTrailingTextGuardrail(**kwargs)


# ---------------------------------------------------------------------------
# 注入物本身的契约
#
# 注入是无条件的，所以注入物的质量就是唯一的防线：它模型看得见、用户看不见，
# 每次都会进对话。原来那个 "." 正是在这里出的事。
# ---------------------------------------------------------------------------


class TestInjectedText:
    def test_is_not_blank_or_litellm_would_strip_it(self):
        """
        litellm 的 strip_empty_text_blocks_from_anthropic_messages 用 .strip() 判空，
        会摘掉纯空白 text 块，摘完又变回裸 tool_result 收尾。所以注入内容必须非空白。
        """
        assert TOOL_RESULT_TRAILING_TEXT.strip()

    def test_is_not_a_lone_period(self):
        """
        生产故障的直接回归。注入物模型看得见、用户看不见，"." 被模型当成用户发来的谜题
        并反问（SpendLogs.reasoning_content 原话："The user sent \".\" which is just a
        period"），4.26% 请求命中、涉及 20+ 用户。
        """
        assert TOOL_RESULT_TRAILING_TEXT.strip(".") != ""

    def test_is_ascii_so_the_model_does_not_switch_reply_language(self):
        """Claude Code 系统提示是英文，注入中文有诱发模型切换回复语言的风险"""
        assert TOOL_RESULT_TRAILING_TEXT.isascii()

    def test_reads_as_a_true_statement_at_the_injection_point(self):
        """
        注入点恰是"工具刚返回、模型要决定下一步"这一刻，所以注入物必须是那个位置说得通的
        话，模型才不会去猜它是什么意思。这条锁的是"别再换回某个无语义的占位符"。
        """
        assert TOOL_RESULT_TRAILING_TEXT == "Continue."


# ---------------------------------------------------------------------------
# 尾块判定：只有裸 tool_result 收尾才触发后端那条要求
#
# 不判 tool id 来源。前缀只能证明"由 deepseek 经 opencode 签发"，证明不了"由同一个
# opencode 账号签发"（newapi 约 10 个渠道各一把 key，id 不编码账号），而漏注入的代价是
# 会话被永久钉在 fallback 上。判据见 README 的 tool_result_trailing_text 一节。
# ---------------------------------------------------------------------------


class TestTrailingTextInjection:
    def test_bare_tool_result_at_end_gets_a_text_block(self):
        """裸 tool_result 收尾会触发后端那条要求 -> 400。这是生产故障的直接形状"""
        out = ensure_trailing_text_after_tool_result([{"role": "user", "content": [TOOL_RESULT]}])
        assert out[-1]["content"] == [TOOL_RESULT, TRAILING_BLOCK]

    def test_own_prefix_id_is_injected_too(self):
        """
        全 call_ 前缀（看着"干净"）也要注入。曾按前缀做过 allowlist 想省掉这些注入，
        已否决：同前缀不代表同一个 opencode 账号，也就不代表查得到 reasoning 缓存，
        而漏注入 -> 400 -> 该会话此后每轮都打不进 deepseek，只能 fallback 到 qwen。
        """
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_00_abc", "name": "Read", "input": {}}]},
            {"role": "user", "content": [TOOL_RESULT]},
        ]
        assert _trailing(ensure_trailing_text_after_tool_result(messages)) == TRAILING_BLOCK

    @pytest.mark.parametrize(
        "foreign_id",
        [
            pytest.param("toolu_01ABC", id="toolu_-qwen"),
            pytest.param("toolu_bdrk_01ABC", id="toolu_bdrk_-kiro-claude"),
            pytest.param("tc_c96612a63b1c_0", id="tc_"),
            pytest.param("call-bc2ece10-fa3d-4e0b-9c1d", id="call-hyphen-uuid"),
            pytest.param("chatcmpl-tool-71ab", id="chatcmpl-tool-"),
            pytest.param("a100ced4-0bad-4b9d-bb8c-67aff0aea9c9", id="bare-uuid"),
        ],
    )
    def test_foreign_prefix_ids_are_injected(self, foreign_id):
        """
        生产实测到的外来前缀家族。这些是注入真正在救的场景（qwen fallback 遗留），
        与上面那条一起构成"任何来源都注入"。
        """
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": foreign_id, "name": "Read", "input": {}}]},
            {"role": "user", "content": [TOOL_RESULT]},
        ]
        assert _trailing(ensure_trailing_text_after_tool_result(messages)) == TRAILING_BLOCK

    def test_foreign_id_deep_in_history_still_triggers_injection(self):
        """
        棘轮的核心形状。Claude Code 每轮重发完整历史，qwen 签的 id 一旦进历史就永久留着
        （两天内 qwen 只签发 6 次，之后携带这些 id 的请求 3435 次）。被污染会话的末条
        tool_result 往往是 deepseek 自己签的 id，脏 id 躺在前面几十轮里，照样 400。
        """
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01ABC", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": "x"}]},
            *[{"role": "assistant", "content": [{"type": "text", "text": f"step {i}"}]} for i in range(30)],
            {"role": "user", "content": [TOOL_RESULT]},
        ]
        assert _trailing(ensure_trailing_text_after_tool_result(messages)) == TRAILING_BLOCK

    def test_existing_text_at_end_is_left_alone(self):
        """已有非空 text 收尾时后端那条要求不触发，无需注入；重复注入会污染对话"""
        messages = [{"role": "user", "content": [TOOL_RESULT, {"type": "text", "text": "what now?"}]}]
        assert ensure_trailing_text_after_tool_result(messages) is messages

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
        assert _trailing(ensure_trailing_text_after_tool_result(messages)) == TRAILING_BLOCK

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

        assert out[0]["content"] == [TOOL_RESULT, TRAILING_BLOCK]
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
        assert ensure_trailing_text_after_tool_result(messages) is messages

    def test_trailing_system_message_does_not_hide_a_bare_tool_result(self):
        """
        08-01 生产 400 的直接形状。Claude Code 会在末尾发一条独立的 role:system 提醒消息，
        而 Anthropic 的 messages 只允许 user/assistant，下游转换器把它上提到开头，
        于是后端看到的收尾又变回裸 tool_result。

        实测（Anthropic 入口）：末条 system + 前一条裸 tool_result -> 400；
        前一条补上 text 块 -> 200。判据见
        .ops-runbook/findings/2026-08-01-newapi-400-补丁未覆盖-交接.md
        """
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "system", "content": "The task tools haven't been used recently."},
        ]

        out = ensure_trailing_text_after_tool_result(messages)

        assert out[0]["content"] == [TOOL_RESULT, TRAILING_BLOCK]
        assert out[1] == messages[1], "不该改动那条 system 消息本身"

    def test_trailing_system_message_with_list_content_is_also_skipped(self):
        """
        role:system 的 content 也可能是块数组。跳过与否只看 role，不看 content 形状 ——
        08-01 之前那版正是因为拿 content 形状做判断（不是 list 就 return）才漏掉这类。
        """
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "system", "content": [{"type": "text", "text": "stay brief"}]},
        ]

        out = ensure_trailing_text_after_tool_result(messages)

        assert out[0]["content"] == [TOOL_RESULT, TRAILING_BLOCK]

    def test_consecutive_trailing_system_messages_are_all_skipped(self):
        """连发多条提醒时要一路往前找，只跳一条不够"""
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "system", "content": "The task tools haven't been used recently."},
            {"role": "system", "content": "The TodoWrite tool hasn't been used recently."},
        ]

        out = ensure_trailing_text_after_tool_result(messages)

        assert out[0]["content"] == [TOOL_RESULT, TRAILING_BLOCK]

    def test_system_message_in_the_middle_is_not_skipped(self):
        """
        只有尾部的 system 不算收尾。中间那条后面还有真正的 user 消息，
        那条才是后端看到的末条；此时无需注入。跳过规则若不限定尾部就会误注入。
        """
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": [{"type": "text", "text": "go on"}]},
        ]
        assert ensure_trailing_text_after_tool_result(messages) is messages

    def test_trailing_system_is_no_op_when_tool_result_already_has_text(self):
        """跳过尾部 system 之后，前一条已有非空 text 收尾，要求不触发，不该重复注入"""
        messages = [
            {"role": "user", "content": [TOOL_RESULT, {"type": "text", "text": "what now?"}]},
            {"role": "system", "content": "reminder"},
        ]
        assert ensure_trailing_text_after_tool_result(messages) is messages

    def test_injection_with_trailing_system_is_idempotent(self):
        """注入完再跑一遍必须 no-op，否则每轮对话都会多堆一个 text 块"""
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "system", "content": "reminder"},
        ]
        once = ensure_trailing_text_after_tool_result(messages)
        assert ensure_trailing_text_after_tool_result(once) is once

    def test_trailing_system_and_dropped_message_skip_rules_compose(self):
        """
        两类跳过规则要能叠加：末尾一条会被 litellm 摘空丢掉的消息 + 一条 system，
        真正的收尾在它们前面。任一规则单独实现都盖不住这个形状。
        """
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "system", "content": "reminder"},
        ]

        out = ensure_trailing_text_after_tool_result(messages)

        assert out[0]["content"] == [TOOL_RESULT, TRAILING_BLOCK]

    def test_only_system_messages_is_a_no_op(self):
        """全是 system 时没有可注入的目标，不能越界或凭空造消息"""
        messages = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}]
        assert ensure_trailing_text_after_tool_result(messages) is messages

    def test_parallel_tool_results_need_only_one_text_block(self):
        """并列多个 tool_result（并行工具调用）只需在末尾补一个"""
        out = ensure_trailing_text_after_tool_result([{"role": "user", "content": [TOOL_RESULT, OTHER_TOOL_RESULT]}])
        assert out[-1]["content"] == [TOOL_RESULT, OTHER_TOOL_RESULT, TRAILING_BLOCK]

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
        """后端那条要求不由裸 tool_result 触发就根本不启动，此时注入纯属噪音"""
        assert ensure_trailing_text_after_tool_result(messages) is messages

    def test_only_the_last_message_is_touched(self):
        """历史里的裸 tool_result 不用管：后端那条要求只由最后一块触发"""
        messages = [
            {"role": "user", "content": [TOOL_RESULT]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [OTHER_TOOL_RESULT]},
        ]

        out = ensure_trailing_text_after_tool_result(messages)

        assert out[0] == messages[0]
        assert out[2]["content"] == [OTHER_TOOL_RESULT, TRAILING_BLOCK]

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
        assert ensure_trailing_text_after_tool_result(messages) is messages


# ---------------------------------------------------------------------------
# hook 行为
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_injects_on_the_production_failure_shape():
    result = await _run(_make_guardrail(), [{"role": "user", "content": [TOOL_RESULT]}])

    assert result is not None, "生产失败形状必须被改写，返回 None 等于修复没生效"
    assert _trailing(result["messages"]) == TRAILING_BLOCK


@pytest.mark.asyncio
async def test_hook_injects_when_production_failure_shape_ends_with_system_message():
    """
    08-01 生产 400 的完整形状走一遍 hook：裸 tool_result + 末条独立 role:system 提醒。
    生产 11 条失败请求里注入的文本一个都没出现，因为注入打在了那条 system 上。
    这条锁住 hook 层不会因此返回 None（返回 None 就等于整个修复没生效）。
    """
    messages = [
        {"role": "user", "content": [TOOL_RESULT]},
        {"role": "system", "content": "The task tools haven't been used recently."},
    ]

    result = await _run(_make_guardrail(), messages)

    assert result is not None, "生产失败形状必须被改写，返回 None 等于修复没生效"
    assert result["messages"][0]["content"] == [TOOL_RESULT, TRAILING_BLOCK]
    assert result["messages"][1] == messages[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="no-messages"),
        pytest.param([{"role": "user", "content": "plain"}], id="no-tool-blocks"),
        pytest.param(
            [{"role": "user", "content": [TOOL_RESULT, {"type": "text", "text": "q"}]}], id="already-has-text"
        ),
        pytest.param("not a list", id="messages-not-a-list"),
    ],
)
async def test_hook_returns_none_when_nothing_needs_changing(messages):
    """无需改动时必须返回 None，不能凭空复制一份 messages 回去"""
    assert await _run(_make_guardrail(), messages) is None


# ---------------------------------------------------------------------------
# 配置契约
# ---------------------------------------------------------------------------


def test_only_pre_call_is_supported_because_later_hooks_cannot_change_the_request():
    assert _make_guardrail().supported_event_hooks == [GuardrailEventHooks.pre_call]


def test_default_on_defaults_to_false_so_it_never_becomes_globally_always_on():
    assert _make_guardrail().default_on is False


def test_unknown_litellm_params_are_absorbed():
    """config.yaml 里多写的键不该让 proxy 启动失败"""
    assert _make_guardrail(some_future_param="x") is not None
