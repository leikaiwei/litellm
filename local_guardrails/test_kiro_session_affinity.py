"""
kiro_session_affinity guardrail 的回归测试。

跑法：.venv/bin/python -m pytest local_guardrails/ -q

测试与被测文件同目录，不放 tests/test_litellm/，避免在 upstream 跟踪的路径下留改动。
"""

import hashlib
import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypes
from kiro_session_affinity import (
    HEADER_NAME,
    KiroSessionAffinityGuardrail,
    session_fingerprint,
)

GUARDRAIL_NAME = "kiro-session-affinity"
SESSION_A = "abc12345-6789-4def-8123-456789abcdef"
SESSION_B = "99912345-6789-4def-8123-456789abcdef"

# 我们这个 fork 在 /v1/messages 上靠 data["headers"] 透传这两个头（litellm_pre_call_utils.py
# 的 add_anthropic_messages_headers_to_llm_call），且注入时机早于本 guardrail。清掉会坏
# DeepSeek Anthropic 兼容链路
PREEXISTING_HEADERS = {"anthropic-beta": "tools-2024-05-16", "anthropic-version": "2023-06-01"}


def _guardrail() -> KiroSessionAffinityGuardrail:
    return KiroSessionAffinityGuardrail(guardrail_name=GUARDRAIL_NAME, event_hook="pre_call", default_on=False)


async def _run_pre_call(data: Dict[str, Any]) -> Optional[dict]:
    return await _guardrail().async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="acompletion",
    )


@pytest.mark.asyncio
async def test_header_value_is_sha256_prefix_not_raw_session_id():
    """值必须是哈希。session_id 源自 metadata.user_id 含身份信息，而 header 会进 nginx log"""
    data: Dict[str, Any] = {"litellm_session_id": SESSION_A}

    await _run_pre_call(data)

    value = data["headers"][HEADER_NAME]
    assert value == hashlib.sha256(SESSION_A.encode()).hexdigest()[:16]
    assert SESSION_A not in value


@pytest.mark.asyncio
async def test_header_value_satisfies_gateway_contract():
    """kiro 侧契约：ASCII [A-Za-z0-9_-]，8~64 字符，非空"""
    data: Dict[str, Any] = {"litellm_session_id": SESSION_A}

    await _run_pre_call(data)

    value = data["headers"][HEADER_NAME]
    assert 8 <= len(value) <= 64
    assert value.isascii()
    assert all(char.isalnum() or char in "_-" for char in value)


def test_fingerprint_is_stable_across_turns_and_distinct_across_sessions():
    """稳定性是唯一硬要求；区分度决定负载会不会全压在一个账号上"""
    assert session_fingerprint(SESSION_A) == session_fingerprint(SESSION_A)
    assert session_fingerprint(SESSION_A) != session_fingerprint(SESSION_B)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {"litellm_session_id": SESSION_A},
        {"metadata": {"session_id": SESSION_A}},
        {"litellm_metadata": {"session_id": SESSION_A}},
    ],
    ids=["root", "metadata", "litellm_metadata"],
)
async def test_session_id_found_in_every_location_litellm_writes_it(data):
    """litellm 按 endpoint 把 session id 写到不同位置，三处都得认"""
    await _run_pre_call(data)

    assert data["headers"][HEADER_NAME] == session_fingerprint(SESSION_A)


@pytest.mark.asyncio
async def test_preexisting_headers_are_preserved():
    """本钩子跑在 add_litellm_data_to_request 之后，赋值而非合并会清掉 anthropic-* 头"""
    data: Dict[str, Any] = {"litellm_session_id": SESSION_A, "headers": dict(PREEXISTING_HEADERS)}

    await _run_pre_call(data)

    assert data["headers"][HEADER_NAME] == session_fingerprint(SESSION_A)
    for key, value in PREEXISTING_HEADERS.items():
        assert data["headers"][key] == value


@pytest.mark.asyncio
async def test_no_session_id_writes_no_header():
    """解不出会话 id 时写个随机值会让每轮都是冷 prefill；缺失才能在覆盖率里看出来"""
    data: Dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}

    result = await _run_pre_call(data)

    assert result is None
    assert "headers" not in data


@pytest.mark.asyncio
async def test_existing_kiro_header_is_not_overwritten():
    """调用方已显式指定亲和 key 时不覆盖，大小写不敏感"""
    data: Dict[str, Any] = {"litellm_session_id": SESSION_A, "headers": {"X-Kiro-Session": "caller-supplied"}}

    await _run_pre_call(data)

    assert data["headers"]["X-Kiro-Session"] == "caller-supplied"
    assert HEADER_NAME not in data["headers"]


@pytest.mark.asyncio
async def test_messages_and_other_data_keys_are_untouched():
    """本 guardrail 只写 header，不得碰 messages 或其他任何键"""
    messages = [{"role": "user", "content": "hi"}]
    data: Dict[str, Any] = {"litellm_session_id": SESSION_A, "messages": messages, "model": "claude-opus-5"}

    await _run_pre_call(data)

    assert data["messages"] is messages
    assert data["model"] == "claude-opus-5"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type",
    [CallTypes.anthropic_messages, CallTypes.acompletion],
    ids=["anthropic_messages", "acompletion"],
)
async def test_header_survives_deployment_level_hook(call_type):
    """
    最关键的一条。deployment 级钩子（fallback 路径上唯一的机会）只把返回值里的 messages
    拷回 kwargs，其余键一概丢弃。所以实现必须原地写 data["headers"]；改成
    `return {**data, "headers": ...}` 这种新建 dict 的写法，这条会失败
    """
    kwargs: Dict[str, Any] = {
        "guardrails": [GUARDRAIL_NAME],
        "litellm_session_id": SESSION_A,
        "messages": [{"role": "user", "content": "hi"}],
    }

    result = await _guardrail().async_pre_call_deployment_hook(kwargs, call_type)

    assert result is not None
    assert result["headers"][HEADER_NAME] == session_fingerprint(SESSION_A)


@pytest.mark.asyncio
async def test_deployment_hook_skips_when_guardrail_not_attached():
    """没挂在这个 deployment 上就不能写 header，否则等于全局常开"""
    kwargs: Dict[str, Any] = {"guardrails": ["some-other-guardrail"], "litellm_session_id": SESSION_A}

    result = await _guardrail().async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages)

    assert result is not None
    assert "headers" not in result
