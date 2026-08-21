"""
把客户端的思考开关意图归一化成下游能区分的形状，修复 Claude Code 关闭思考后仍然思考。

## 为什么需要它

`deepseek-v4-flash` 在成本表里没有 `supports_adaptive_thinking`，于是原生 Anthropic
直通路径上 `AnthropicMessagesConfig._translate_adaptive_effort_for_non_adaptive_model`
判它是"非 adaptive 模型"，走降级分支。那个分支的闸门是：

    if effort is None and not adaptive_thinking:
        return

**`output_config.effort` 非空就足以进去**，随后无条件把 `thinking` 覆盖成
`{"type": "enabled", "budget_tokens": N}`。而 Claude Code 关闭思考只是不发 `thinking`
字段，`output_config.effort` 照发（它是独立的强度档，关掉思考后仍停留在上次的值）。
于是"关"和"开"两种状态出站完全一致，下游网关无从区分。

同一个闸门还导致：客户端**明确**发 `{"type": "disabled"}`，只要 effort 还在，也会被
改写成 enabled。所以本 guardrail 必须同时摘掉 effort，只补 disabled 是不够的。

## 归一化规则

    thinking.type ∈ {enabled, adaptive}  ->  一个字都不碰
    其余情况（缺失 / disabled / 未知值） ->  thinking = {"type": "disabled"}
                                            并摘掉 output_config.effort

第一条保证不误伤另一个客户端：同一个模型组上还有一路请求发
`{"type": "enabled", "budget_tokens": 7168}`（指纹是不带 `anthropic-beta` 头、几十个
工具、max_tokens=8192），它的 thinking 本来就完好穿过 litellm，不该被动。

只摘 `effort` 而不是删掉整个 `output_config`：生产里 `output_config` 还承载
`format`（结构化输出的 json_schema），删掉会破坏它。实测保留 format 不影响闸门判定。

## 上线方式

这是 litellm 的外部自定义 guardrail，不修改 litellm 源码。把本文件放在 config.yaml
同目录，容器里按文件挂载（`./thinking_switch.py:/app/thinking_switch.py:ro`），然后：

    guardrails:
      - guardrail_name: "thinking-switch"
        litellm_params:
          guardrail: thinking_switch.ThinkingSwitchGuardrail
          mode: "pre_call"
          default_on: false                    # 必须 false，否则变成全局常开

    model_list:
      - model_name: your-model-group
        litellm_params:
          guardrails: ["vision-to-text", "thinking-switch"]   # 整列表覆盖语义，
                                                              # 漏写会把识图挤掉

下游契约：下游网关侧规则认 `enabled` / `adaptive` 为"要思考"，其余补 disabled。本
guardrail 归一化后，关态出站是 `disabled`（命中它的触发分支），开态是
`enabled`（命中跳过分支），两侧都落在正确侧。
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import CallTypesLiteral

# 视为"客户端要思考"的 type 取值。白名单而非黑名单：未知取值按不思考处理，
# 与下游网关侧规则的保守默认一致
THINKING_ON_TYPES = frozenset({"enabled", "adaptive"})

DISABLED_THINKING: Dict[str, str] = {"type": "disabled"}


def thinking_is_on(thinking: object) -> bool:
    """客户端是否明确要求思考。`{"type": "adaptive", "display": "summarized"}` 算要求"""
    return isinstance(thinking, dict) and thinking.get("type") in THINKING_ON_TYPES


def output_config_without_effort(output_config: object) -> Optional[Dict[str, Any]]:
    """摘掉 effort，保留 format 等其余键。返回 None 表示该键应整体移除"""
    if not isinstance(output_config, dict):
        return None
    residual = {key: value for key, value in output_config.items() if key != "effort"}
    return residual or None


class ThinkingSwitchGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        event_hook: Optional[str] = None,
        default_on: bool = False,
        **kwargs: Any,
    ) -> None:
        # 其余 litellm_params 由 **kwargs 吸收
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
        if thinking_is_on(data.get("thinking")):
            return None

        # 必须原地写回 data，不能只靠返回值：deployment 级钩子
        # （custom_guardrail.py 的 async_pre_call_deployment_hook）只把返回值里的
        # messages 拷回 kwargs，其余键一概丢弃。而模型组 fallback 只走 router 内部、
        # 不重跑 proxy 级钩子，deployment 级是那条路上唯一的机会。
        # 赋新对象而不是原地改嵌套 dict：上游 utils.py 只做了 kwargs 的浅拷贝，
        # 改嵌套结构会波及调用方持有的原始 dict
        data["thinking"] = dict(DISABLED_THINKING)

        if "output_config" in data:
            residual = output_config_without_effort(data["output_config"])
            if residual is None:
                data.pop("output_config", None)
            else:
                data["output_config"] = residual

        verbose_proxy_logger.debug("thinking-switch: 归一化为 disabled 并摘掉 effort")
        return data
