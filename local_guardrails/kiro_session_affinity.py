"""
给发往 kiro-gateway 的请求打一个跨轮次稳定的会话指纹 header，让 kiro 侧的 nginx
`hash $http_x_kiro_session consistent` 把同一个对话线程钉在同一个上游账号上。

kiro 上游按账号隔离前缀缓存，同一会话落错账号每轮多付 3~6 秒冷 prefill。选路由 kiro
侧的 nginx 完成，litellm 这边只负责把 key 稳定地带上：本 guardrail 只写 header，不选路、
不改 messages、不阻断请求。

值 = sha256(litellm_session_id) 前 16 位 hex。不直接透出 session_id 本身：它源自
metadata.user_id，含用户身份信息，而这个 header 会进 nginx access log。

刻意只掺 session_id，不掺 system prompt。掺 system 的动机是把共享同一 session id 的并行
subagent 分到不同账号，但 Claude Code 的 system 是数组，前若干 KB 是所有会话共享的公共前缀，
按前缀截取大概率区分度为 0；而整体哈希又会被 system 里逐轮变化的内容带得每轮漂移，那比不做
更糟（每轮冷 prefill）。稳定性优先于区分度，subagent 挤在一个账号上能从 nginx 侧监控看出来
再调。

这是 litellm 的外部自定义 guardrail，不修改 litellm 源码。把本文件放在 config.yaml 同目录，
容器里按文件挂载（`./kiro_session_affinity.py:/app/kiro_session_affinity.py:ro`），然后：

    guardrails:
      - guardrail_name: "kiro-session-affinity"
        litellm_params:
          guardrail: kiro_session_affinity.KiroSessionAffinityGuardrail
          mode: "pre_call"
          default_on: false                          # 必须 false，否则变成全局常开

    model_list:
      - model_name: claude-opus-5
        litellm_params:
          model: claude-opus-5
          litellm_credential_name: kiro-gateway
          guardrails: ["kiro-session-affinity"]      # 只有这里挂了才生效
"""

import hashlib
from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail, get_session_id_from_request_data
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import CallTypesLiteral

# 小写。litellm 到上游全程按字面量传，不做规范化；HTTP header 名本身大小写不敏感
HEADER_NAME = "x-kiro-session"

# 取 16 位：满足对方契约的 8~64 字符，且 sha256 前 16 位 hex 的碰撞概率对会话量级足够
_FINGERPRINT_LEN = 16


def session_fingerprint(session_id: str) -> str:
    """会话 id 到 header 值。hexdigest 天然满足 ASCII [0-9a-f] 与长度约束"""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:_FINGERPRINT_LEN]


class KiroSessionAffinityGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        event_hook: Optional[str] = None,
        default_on: bool = False,
        **kwargs: Any,
    ) -> None:
        # 其余 litellm_params 由 **kwargs 吸收，外部 guardrail 拿不到 llm_router 也不需要
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            event_hook=event_hook,  # type: ignore[arg-type]
            default_on=default_on,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: "CallTypesLiteral",
    ) -> Optional[dict]:
        session_id = get_session_id_from_request_data(data)
        if not session_id:
            # 解不出会话 id 就不写 header。此时写一个逐次变化的值反而每轮都是冷 prefill，
            # 不如缺失 —— 缺失能在 nginx access log 的覆盖率里直接看出来
            verbose_proxy_logger.debug("kiro-session-affinity: 无 session_id，跳过")
            return None

        existing = data.get("headers")
        headers: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        if any(key.lower() == HEADER_NAME for key in headers):
            return None

        headers[HEADER_NAME] = session_fingerprint(session_id)

        # 必须原地写回 data，不能只靠返回值：deployment 级钩子
        # （custom_guardrail.py 的 async_pre_call_deployment_hook）只把返回值里的
        # messages 拷回 kwargs，其余键一概丢弃。而模型组 fallback 只走 router 内部、
        # 不重跑 proxy 级钩子，deployment 级是那条路上唯一的机会。
        # 同时合并而非覆盖 existing：本钩子跑在 add_litellm_data_to_request 之后，
        # 直接赋值会清掉它注入的 anthropic-beta / anthropic-version
        data["headers"] = headers
        return data
