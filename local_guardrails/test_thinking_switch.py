"""
thinking_switch guardrail 的回归测试。

跑法：.venv/bin/python -m pytest local_guardrails/ -q

测试与被测文件同目录，不放 tests/test_litellm/，避免在 upstream 跟踪的路径下留改动。

最有价值的是文件末尾那组 e2e：把 guardrail 的输出直接喂给真实的
`AnthropicMessagesConfig.transform_anthropic_messages_request`，断言"关"和"开"两态
出站可区分。它是唯一能守住本 guardrail 存在意义的测试 —— 前面那些单元测试全过、
而 e2e 挂掉，说明归一化规则本身选错了。
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm.caching.dual_cache import DualCache
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CallTypes
from thinking_switch import ThinkingSwitchGuardrail

GUARDRAIL_NAME = "thinking-switch"

# 生产实测的目标模型。它在成本表里没有 supports_adaptive_thinking，正因如此才落进
# 那个会无条件改写 thinking 的降级分支
MODEL = "deepseek-v4-flash"

# 生产 SpendLogs 里真实出现过的 output_config.format（结构化输出）。它与 effort 同处
# output_config，所以实现只能摘 effort，不能删整块
JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "schema": {"type": "object", "required": ["title"], "properties": {"title": {"type": "string"}}},
}


def _guardrail() -> ThinkingSwitchGuardrail:
    return ThinkingSwitchGuardrail(guardrail_name=GUARDRAIL_NAME, event_hook="pre_call", default_on=False)


async def _run_pre_call(data: Dict[str, Any]) -> Optional[dict]:
    return await _guardrail().async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="anthropic_messages",
    )


def _outbound_thinking(inbound: Dict[str, Any]) -> Any:
    """跑真实的 Anthropic 直通 transform，返回它最终发给下游的 thinking"""
    params = dict(inbound)
    request = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    return request.get("thinking")


# ---------------------------------------------------------------------------
# 关态：客户端没要求思考
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_thinking_becomes_disabled_and_effort_is_dropped():
    """Claude Code 关闭思考 = 不发 thinking 字段，但 effort 仍停在上次的值"""
    data: Dict[str, Any] = {"max_tokens": 32000, "output_config": {"effort": "high"}}

    await _run_pre_call(data)

    assert data["thinking"] == {"type": "disabled"}
    assert "output_config" not in data


@pytest.mark.asyncio
async def test_explicit_disabled_still_has_effort_dropped():
    """
    最容易漏的一条。客户端明确发 disabled 时，光"补 disabled"是 no-op，effort 留着就会
    让下游 transform 把 disabled 覆盖成 enabled。删掉实现里摘 effort 的分支，这条会失败
    """
    data: Dict[str, Any] = {
        "max_tokens": 32000,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "high"},
    }

    await _run_pre_call(data)

    assert data["thinking"] == {"type": "disabled"}
    assert "output_config" not in data


@pytest.mark.asyncio
async def test_unknown_thinking_type_is_treated_as_off():
    """白名单语义：未知取值按不思考处理，与下游网关规则的保守默认一致"""
    data: Dict[str, Any] = {"max_tokens": 32000, "thinking": {"type": "something-new"}}

    await _run_pre_call(data)

    assert data["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_no_output_config_at_all_still_gets_disabled():
    data: Dict[str, Any] = {"max_tokens": 32000}

    await _run_pre_call(data)

    assert data["thinking"] == {"type": "disabled"}
    assert "output_config" not in data


# ---------------------------------------------------------------------------
# output_config 的部分摘除：format 必须活下来
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_format_survives_when_effort_is_dropped():
    """
    output_config 还承载结构化输出的 json_schema（生产实测存在）。实现若图省事直接
    `data.pop("output_config")`，这条会失败，且线上表现是结构化输出静默失效
    """
    data: Dict[str, Any] = {
        "max_tokens": 32000,
        "output_config": {"effort": "high", "format": JSON_SCHEMA_FORMAT},
    }

    await _run_pre_call(data)

    assert data["output_config"] == {"format": JSON_SCHEMA_FORMAT}


@pytest.mark.asyncio
async def test_output_config_without_effort_is_left_intact():
    """没有 effort 可摘时不该把这个键搅动成别的形状"""
    data: Dict[str, Any] = {"max_tokens": 32000, "output_config": {"format": JSON_SCHEMA_FORMAT}}

    await _run_pre_call(data)

    assert data["output_config"] == {"format": JSON_SCHEMA_FORMAT}


# ---------------------------------------------------------------------------
# 开态：一个字都不许碰
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "adaptive"},
        {"type": "adaptive", "display": "summarized"},
        {"type": "enabled", "budget_tokens": 7168},
        {"type": "enabled", "budget_tokens": 31999},
    ],
    ids=["adaptive", "adaptive-summarized", "enabled-7168", "enabled-31999"],
)
async def test_thinking_on_is_never_touched(thinking):
    """
    enabled 那两个形状来自同一个模型组上的另一个客户端（不带 anthropic-beta 头、几十个
    工具），它的 thinking 本来就完好穿过 litellm，误伤它等于把好的也弄坏
    """
    original_thinking = dict(thinking)
    output_config = {"effort": "xhigh"}
    data: Dict[str, Any] = {"max_tokens": 32000, "thinking": thinking, "output_config": output_config}

    result = await _run_pre_call(data)

    assert result is None
    assert data["thinking"] == original_thinking
    assert data["output_config"] == {"effort": "xhigh"}


@pytest.mark.asyncio
async def test_messages_and_other_keys_are_untouched():
    """本 guardrail 只动 thinking / output_config.effort，不得碰 messages 等任何其他键"""
    messages = [{"role": "user", "content": "hi"}]
    data: Dict[str, Any] = {
        "max_tokens": 32000,
        "messages": messages,
        "model": MODEL,
        "tools": [{"name": "Bash"}],
        "system": "sys",
        "output_config": {"effort": "high"},
    }

    await _run_pre_call(data)

    assert data["messages"] is messages
    assert data["model"] == MODEL
    assert data["tools"] == [{"name": "Bash"}]
    assert data["system"] == "sys"


# ---------------------------------------------------------------------------
# deployment 级钩子：fallback 路径上唯一的机会
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    [CallTypes.anthropic_messages, CallTypes.acompletion],
    ids=["anthropic_messages", "acompletion"],
)
async def test_normalization_survives_deployment_level_hook(call_type):
    """
    deployment 级钩子只把返回值里的 messages 拷回 kwargs，其余键一概丢弃。所以实现必须
    原地写 data；改成 `return {**data, "thinking": ...}` 这种新建 dict 的写法这条会失败。
    模型组 fallback 只走 router 内部、不重跑 proxy 级钩子，这里是那条路上唯一的机会
    """
    kwargs: Dict[str, Any] = {
        "guardrails": [GUARDRAIL_NAME],
        "max_tokens": 32000,
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "hi"}],
    }

    result = await _guardrail().async_pre_call_deployment_hook(kwargs, call_type)

    assert result is not None
    assert result["thinking"] == {"type": "disabled"}
    assert "output_config" not in result


@pytest.mark.asyncio
async def test_deployment_hook_skips_when_guardrail_not_attached():
    """没挂在这个 deployment 上就不能改，否则等于全局常开"""
    kwargs: Dict[str, Any] = {
        "guardrails": ["some-other-guardrail"],
        "max_tokens": 32000,
        "output_config": {"effort": "high"},
    }

    result = await _guardrail().async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages)

    assert result is not None
    assert "thinking" not in result
    assert result["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# e2e：接真实 transform，断言两态出站可区分
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_off_state_reaches_downstream_as_disabled():
    """关态：归一化后过真实 transform，出站必须是 disabled 而不是被改写成 enabled"""
    data: Dict[str, Any] = {"max_tokens": 32000, "output_config": {"effort": "high"}, "temperature": 0}

    await _run_pre_call(data)

    assert _outbound_thinking(data) == {"type": "disabled"}


@pytest.mark.asyncio
async def test_e2e_on_state_reaches_downstream_as_enabled():
    """开态：出站必须是 enabled/adaptive 之一，下游据此判定要思考"""
    data: Dict[str, Any] = {
        "max_tokens": 32000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "temperature": 0,
    }

    await _run_pre_call(data)

    outbound = _outbound_thinking(data)
    assert isinstance(outbound, dict)
    assert outbound.get("type") in ("enabled", "adaptive")


@pytest.mark.asyncio
async def test_e2e_两态出站互不相同():
    """
    本 guardrail 存在的全部理由。没有它时两态出站字节级相同
    （都是 {"type":"enabled","budget_tokens":4096}），下游无从区分
    """
    off: Dict[str, Any] = {"max_tokens": 32000, "output_config": {"effort": "high"}}
    on: Dict[str, Any] = {"max_tokens": 32000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}

    await _run_pre_call(off)
    await _run_pre_call(on)

    assert _outbound_thinking(off) != _outbound_thinking(on)


def test_e2e_没有本_guardrail_时两态确实无法区分():
    """
    反向锚点：证明上面那些测试守护的是真实缺陷，不是恒真断言。这条描述的是当前 litellm
    的行为，若哪天上游修了这个改写逻辑，它会失败，提示本 guardrail 可以退役
    """
    off = {"max_tokens": 32000, "output_config": {"effort": "high"}}
    on = {"max_tokens": 32000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}

    assert _outbound_thinking(off) == _outbound_thinking(on) == {"type": "enabled", "budget_tokens": 4096}
