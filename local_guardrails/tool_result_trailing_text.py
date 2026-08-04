"""
裸 tool_result 收尾且历史带外来 tool id 时，在末尾追加一个非空 text 块，绕开 deepseek
思考模式的 reasoning_content 要求。

要解决的生产故障：hoperun 组偶发 fallback 到 qwen3.7-plus，qwen 签发的 tool id 进了会话
历史，deepseek 恢复后该会话每轮必 400。

机制（订正过一次，见下）：deepseek 思考模式要求带 tool_calls 的 assistant 消息回传
reasoning_content。opencode 用它签发的 tool id 作**缓存键**存这份 reasoning —— 自己签的
id 查得到就自动补上，外来 id 查不到就裸奔给 deepseek，被拒并报
`The reasoning_content in the thinking mode must be passed back to the API`。

所以 tool id 不是安全校验对象，只是个缓存键（实测：假 id + 手工补 reasoning_content，
哪怕只是单空格，也 200）。早前记的"opencode 对 tool id 做服务端注册表存在性校验"是错的，
按那个结论推导出的方案全部无效。

这是 litellm 的外部自定义 guardrail，不修改 litellm 源码。把本文件放在 config.yaml
同目录，然后：

    model_list:
      - model_name: deepseek-v4-flash
        litellm_params:
          model: openai/deepseek-v4-flash
          api_base: os.environ/TEXT_MODEL_API_BASE
          api_key: os.environ/TEXT_MODEL_API_KEY
          guardrails: ["tool-result-trailing-text"]   # 只有这里挂了才生效

    guardrails:
      - guardrail_name: "tool-result-trailing-text"
        litellm_params:
          guardrail: tool_result_trailing_text.ToolResultTrailingTextGuardrail
          mode: "pre_call"
          default_on: false                           # 必须 false，否则变成全局常开
"""

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, cast

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import CallTypesLiteral

# 追加在裸 tool_result 之后的文本。三条约束叠出这个取值：
#
# 1. 必须非空白：litellm 的 strip_empty_text_blocks_from_anthropic_messages 用 .strip()
#    判空并摘掉纯空白块，摘完又变回裸 tool_result 收尾，等于没修（空串与单空格均实测无效）
# 2. 必须对模型讲得通：注入的文本模型看得见、用户看不见。曾用 "." 导致模型把它当成用户
#    发来的谜题并反问（生产 reasoning_content 原话："The user sent \".\" which is just a
#    period"），命中 20+ 用户。注入点恰是"工具刚返回、模型要决定下一步"，Continue. 是真话
# 3. 必须是英文：Claude Code 系统提示是英文，注入中文有诱发模型切换回复语言的风险
TOOL_RESULT_TRAILING_TEXT = "Continue."

# 注入是无条件的（只要闸门会触发就注入），不按 tool id 来源收窄。曾试过按签发前缀做
# allowlist（只认 call_ 是"自己的"），已否决，原因是那个判断不可能做对：
#
# 前缀最多能证明"这个 id 由 deepseek 经 opencode 签发"，**证明不了"由同一个 opencode
# 账号签发"**。reasoning 缓存可能是按账号隔离的，而 newapi 有约 10 个 opencode 渠道、
# 各自一把 key，litellm 不知道这次请求被投给了哪一个 —— id 里也不编码账号。所以
# "同前缀"不等于"查得到缓存"。
#
# 两个方向的代价完全不对称：漏注入 -> 400 -> 该会话此后每轮都打不进 deepseek，只能
# fallback 到 qwen（单向棘轮，用户明确不接受）；多注入 -> 模型多看见一句 Continue.，
# 有噪音、可恢复。所以宁可多注入。
#
# 另外，那个"跨账号漏洞"从未被观测到过 —— 原来就是无条件注入，闸门没机会触发，所以
# 干净会话到底会不会 400 无数据。开启收窄等于拿会话赌一个未验证的假设。
# 要验的话见 README 的"待验"一节，验明是全局缓存之后再谈收窄。


def _survives_empty_text_stripping(block: object) -> bool:
    """
    litellm 发给后端前会摘掉纯空白 text 块（strip_empty_text_blocks_from_anthropic_messages），
    所以"是否裸 tool_result 收尾"必须按摘完之后的样子判断。否则会被 Claude Code 常带的
    {"type": "text", "text": ""} 骗过去：看着有 text 收尾，实际发出去时它已被摘掉，
    请求照样 400。
    """
    if not isinstance(block, dict) or block.get("type") != "text":
        return True
    text = block.get("text")
    return isinstance(text, str) and bool(text.strip())


def _ends_with_bare_tool_result(content: Sequence[object]) -> bool:
    last = next((block for block in reversed(content) if _survives_empty_text_stripping(block)), None)
    return isinstance(last, dict) and last.get("type") == "tool_result"


def _dropped_by_empty_text_stripping(message: object) -> bool:
    """
    litellm 摘空块后若整条 content 变空，会把**整条消息**从列表里删掉，而不是留个空数组
    （strip_empty_text_blocks_from_anthropic_messages 的 `elif filtered:` 分支）。
    这种消息不能算末条，否则它会挡住前面那条真正的裸 tool_result 收尾。

    条件必须与 litellm 逐字对齐：content 为空列表时 len 不变、走保留分支，故此处要求非空。
    """
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(not _survives_empty_text_stripping(block) for block in content)


def _counts_as_tail(message: object) -> bool:
    """
    这条消息算不算后端校验时看到的收尾。两类不算：

    1. 会被 litellm 摘空后整条删掉的（见 _dropped_by_empty_text_stripping）
    2. role: system —— Anthropic 的 messages 只允许 user / assistant，system 是顶层字段。
       但 Claude Code 确实会在末尾发独立的 role: system 提醒消息（"The task tools haven't
       been used recently…"），litellm 原样透传，下游 anthropic -> openai 转换器把它上提到
       开头，于是后端看到的收尾又变回它前面那条。

    第 2 类是 08-01 生产 400 的直接根因：注入落在一条不参与校验的消息上等于没注入
    （11 条失败请求里注入的文本一个都没出现）。判定只看 role，不看 content 形状 ——
    上一版正是因为 content 是字符串就提前 return 才漏掉它。
    """
    if isinstance(message, dict) and message.get("role") == "system":
        return False
    return not _dropped_by_empty_text_stripping(message)


def _last_effective_message_index(messages: Sequence[AllMessageValues]) -> Optional[int]:
    return next(
        (index for index in reversed(range(len(messages))) if _counts_as_tail(messages[index])),
        None,
    )


def ensure_trailing_text_after_tool_result(messages: List[AllMessageValues]) -> List[AllMessageValues]:
    """
    最后一条消息以裸 tool_result 收尾时，在其 content 末尾追加一个非空 text 块。

    改写 id 这条路走不通：问题不在 id 长什么样，而在它查不到 reasoning 缓存。但那条要求是
    两段式的 —— 只有请求最后一个 content 块是裸 tool_result 时才**触发**，一旦触发就扫全
    历史的 id。所以末尾追加一个非空 text 块，要求根本不启动，历史里有多少外来 id 都无所谓。

    无条件执行，不判 id 来源（原因见 TOOL_RESULT_TRAILING_TEXT 上方那段注释：前缀证明不了
    同账号，而漏注入的代价是会话被永久钉在 fallback 上）。注入本身的代价靠文案化解 ——
    Continue. 在"工具刚返回"这个位置是真话，模型不会像对 "." 那样反问。

    也不加"思考是否开启"这个条件：思考的最终状态由 thinking_switch 归一化后再叠加 newapi
    的规则决定，guardrail 侧看到的不是最终值，把两个 guardrail 耦起来很脆。关态多注入无害。

    OpenAI 形状（末条 role: "tool"）不处理：那种形状只能另起一条 user 消息，会造成连续两条
    user 消息，未实测过。生产走 /v1/messages，进本 hook 时是 Anthropic 形状。

    "最后一条消息"按后端实际看到的收尾算，不是 messages[-1]：会被 litellm 摘空丢掉的消息、
    以及尾部的 role: system 消息都不算（见 _counts_as_tail）。注入落在这两类上等于没注入。
    它们该由 litellm 或下游转换器自己处理，本函数不删不改。

    无需改动时返回传入的对象本身，调用方可用 `is` 判断有没有变。原 messages 不被修改。
    """
    index = _last_effective_message_index(messages)
    if index is None:
        return messages
    last_message = messages[index]
    content = last_message.get("content")
    if not isinstance(content, list) or not _ends_with_bare_tool_result(content):
        return messages
    trailing = {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}
    patched = cast(AllMessageValues, {**last_message, "content": [*content, trailing]})
    return [*messages[:index], patched, *messages[index + 1 :]]


class ToolResultTrailingTextGuardrail(CustomGuardrail):
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
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            return None
        messages = cast(List[AllMessageValues], raw_messages)

        patched = ensure_trailing_text_after_tool_result(messages)
        if patched is messages:
            return None
        return {**data, "messages": patched}
