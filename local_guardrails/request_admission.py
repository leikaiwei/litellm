import asyncio
import base64
import binascii
import copy
import os
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Final, cast
from urllib.parse import urlsplit

from openai.types.responses.response_create_params import ResponseInputParam
from opentelemetry.trace import Span
from PIL import Image
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.constants import DEFAULT_CHAT_COMPLETION_PARAM_VALUES, MAX_IMAGE_URL_DOWNLOAD_SIZE_MB
from litellm.exceptions import (
    MalformedRequestFieldError,
    RequestAdmissionContextWindowExceededError,
    RequestAdmissionError,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.litellm_core_utils.token_counter import (
    calculate_img_tokens,
    high_detail_image_token_upper_bound,
    offload_token_count,
    token_counter,  # pyright: ignore[reportUnknownVariableType]  # 上游 custom_tokenizer 包含裸 dict
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import AnthropicModelInfo, merge_anthropic_beta_headers
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import COUNT_TOKEN_OPTION_NAMES
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
    messages_carry_output_config,
)
from litellm.llms.anthropic.pass_through.messages.utils import (
    anthropic_system_to_openai_message,
    prepare_native_request_body,
)
from litellm.llms.anthropic.wif import resolve_anthropic_base
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig as ResponsesConfig,
)
from litellm.router_strategy.complexity_router.context_compaction import compaction_pending
from litellm.types.llms.anthropic import ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import CallTypes, ProviderSpecificHeader
from litellm.utils import (
    get_optional_params,  # pyright: ignore[reportUnknownVariableType]  # 上游参数转换器仍包含裸类型
)

_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_OPTIONAL_STRING: Final = TypeAdapter[str | None](str | None)
_HEADERS: Final = TypeAdapter(dict[str, str])
_PROVIDER_HEADERS: Final[TypeAdapter[ProviderSpecificHeader | tuple[ProviderSpecificHeader, ...] | None]] = TypeAdapter(
    ProviderSpecificHeader | tuple[ProviderSpecificHeader, ...] | None
)
_ITEMS: Final = TypeAdapter(tuple[object, ...])
_MESSAGES: Final = TypeAdapter(tuple[dict[str, object], ...])
_TOOL_CHOICE: Final = TypeAdapter[str | dict[str, object] | None](str | dict[str, object] | None)
_VERIFIED_BASES: Final = TypeAdapter(tuple[str, ...])
_NON_CHAT_MODES: Final = frozenset(
    (
        "embedding",
        "rerank",
        "audio_transcription",
        "audio_speech",
        "evaluation",
        "image_edit",
        "image_generation",
        "moderation",
        "realtime",
        "search",
        "vector_store",
        "video_generation",
        "extraction",
        "guardrail",
        "ocr",
    )
)
_CALLS: Final = frozenset(
    (
        CallTypes.completion,
        CallTypes.acompletion,
        CallTypes.anthropic_messages,
        CallTypes.aanthropic_messages,
        CallTypes.responses,
        CallTypes.aresponses,
    )
)
_CLAUDE_COUNT_HEADERS: Final = frozenset(
    (
        "user-agent",
        "x-app",
        "x-claude-code-session-id",
        "session-id",
        "session_id",
        "x-session-id",
        "x-session-affinity",
        "x-client-request-id",
        "anthropic-user-profile-id",
        "anthropic-workspace-id",
    )
)


@dataclass(frozen=True, slots=True)
class AdmissionProblem:
    field: str
    message: str
    malformed: bool = True


@dataclass(frozen=True, slots=True)
class Candidate:
    deployment: dict[str, object]
    model: str
    limit: int | None
    local_response: bool = False


def _mapping(value: object) -> dict[str, object]:
    return _OBJECT.validate_python(value) if isinstance(value, Mapping) else {}


def _is_non_chat(deployment: Mapping[str, object]) -> bool:
    info: Final = _mapping(deployment.get("model_info"))
    mode: Final = info.get("mode")
    if isinstance(mode, str):
        return mode in _NON_CHAT_MODES
    params: Final = _mapping(deployment.get("litellm_params"))
    model: Final = info.get("base_model") or params.get("base_model") or params.get("model")
    if not isinstance(model, str):
        return False
    try:
        return litellm.get_model_info(model=model).get("mode") in _NON_CHAT_MODES
    except Exception:  # noqa: BLE001  # 未知目录模型保持严格检查
        return False


def _candidate(deployment: dict[str, object]) -> Candidate:
    params: Final = _mapping(deployment.get("litellm_params"))
    info: Final = _mapping(deployment.get("model_info"))
    declared_model: Final = info.get("base_model") or params.get("base_model") or params.get("model")
    model: Final = declared_model if isinstance(declared_model, str) else ""
    if params.get("model") == "claude_code_review_allow/allow":
        return Candidate(deployment, model, None, True)
    explicit: Final = info.get("max_input_tokens")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return Candidate(deployment, model, explicit)
    try:
        catalog: Final = litellm.get_model_info(model=model)
    except Exception:  # noqa: BLE001  # 自定义模型可能不在目录内，只使用已知的输入上限
        return Candidate(deployment, model, None)
    maximum: Final = catalog.get("max_input_tokens")
    return Candidate(deployment, model, maximum if isinstance(maximum, int) and maximum > 0 else None)


def _image_data(block: Mapping[str, object]) -> str | AdmissionProblem | None:
    if block.get("type") == "image":
        source: Final = _mapping(block.get("source"))
        if source.get("type") != "base64":
            return None
        data: Final = source.get("data")
        return data if isinstance(data, str) else AdmissionProblem("source.data", "内嵌图片缺少 base64 数据")
    if block.get("type") not in ("input_image", "image_url"):
        return None
    image_url: Final = block.get("image_url")
    url: Final = _mapping(block.get("image_url")).get("url") if isinstance(image_url, Mapping) else image_url
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    maximum: Final = int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)
    if maximum <= 0 or len(url) > 4 * ((maximum + 2) // 3) + 128:
        return AdmissionProblem("image_url", "内嵌图片损坏或超过图片大小限制，请重新上传图片后重试")
    header, separator, encoded = url.partition(",")
    if not separator or not header.endswith(";base64"):
        return AdmissionProblem("image_url", "内嵌图片必须使用 base64 data URL")
    return encoded


def _image_blocks(items: object, path: str) -> Iterator[tuple[str, str | AdmissionProblem]]:
    if isinstance(items, (list, tuple)):
        for index, item in enumerate(_ITEMS.validate_python(items)):
            yield from _image_blocks(item, f"{path}[{index}]")
        return
    if not isinstance(items, Mapping):
        return
    block: Final = _OBJECT.validate_python(items)
    encoded: Final = _image_data(block)
    if encoded is not None:
        yield path, encoded
    yield from _image_blocks(block.get("content"), f"{path}.content")
    if block.get("type") == "function_call_output":
        yield from _image_blocks(block.get("output"), f"{path}.output")


def _valid_image(encoded: str) -> bool:
    maximum: Final = int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)
    if maximum <= 0 or len(encoded) > 4 * ((maximum + 2) // 3):
        return False
    try:
        decoded: Final = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(decoded)) as image:
            if Image.MAX_IMAGE_PIXELS is not None and image.width * image.height > Image.MAX_IMAGE_PIXELS:
                return False
            image.verify()
        with Image.open(BytesIO(decoded)) as image_pixels:
            image_pixels.load()
    except (ValueError, OSError, SyntaxError, binascii.Error, Image.DecompressionBombError):
        return False
    return True


def _image_problem(data: Mapping[str, object]) -> AdmissionProblem | None:
    for field in ("messages", "input", "system"):
        for path, encoded in _image_blocks(data.get(field), field):
            if isinstance(encoded, AdmissionProblem):
                return AdmissionProblem(f"{path}.{encoded.field}", encoded.message)
            if not _valid_image(encoded):
                return AdmissionProblem(path, "内嵌图片损坏或超过图片大小限制，请重新上传图片后重试")
    return None


def _messages(data: Mapping[str, object]) -> Sequence[object]:
    raw_messages: Final = data.get("messages")
    if raw_messages is not None:
        parsed: Final = _MESSAGES.validate_python(raw_messages)
        system: Final = anthropic_system_to_openai_message(data.get("system"))
        return (system, *parsed) if system is not None else parsed
    raw_input: Final = data.get("input")
    if not isinstance(raw_input, (str, list)):
        raise ValueError("Responses input must be a string or list")
    instructions: Final = data.get("instructions")
    return ResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # 旧裸参数
        input=cast(str | ResponseInputParam, raw_input),  # cast-ok: 共享转换器负责验证 Responses 各种输入项
        responses_api_request={"instructions": instructions} if isinstance(instructions, str) else {},
    )


def _responses_problem(data: Mapping[str, object]) -> AdmissionProblem | None:
    if "input" not in data:
        return None
    if data.get("messages") is not None:
        return AdmissionProblem("input/messages", "不能同时核验两种输入协议，请只提供当前接口的完整输入", False)
    for field in ("previous_response_id", "conversation"):
        if data.get(field) is not None:
            return AdmissionProblem(field, "无法本地核验服务端历史，请提供完整历史、使用无状态请求或开启新会话", False)
    raw_input: Final = data.get("input")
    if isinstance(raw_input, str):
        return None
    for index, item in enumerate(_MESSAGES.validate_python(raw_input)):
        if (
            item.get("type") in ("reasoning", "compaction")
            and isinstance(encrypted := item.get("encrypted_content"), str)
            and encrypted
        ):
            continue
        if item.get("type") == "item_reference" or item.get("encrypted_content") is not None:
            return AdmissionProblem(
                f"input[{index}]", "无法本地核验引用或加密历史，请提供完整可读历史或开启新会话", False
            )
        if not _messages({"input": [item]}):
            return AdmissionProblem(
                f"input[{index}]", "本地计数器无法读取此输入项，请提供完整可读历史或开启新会话", False
            )
    return None


def _remote_image(block: Mapping[str, object]) -> str | None:
    if block.get("type") == "image_url":
        value: Final = block.get("image_url")
        url: Final = _mapping(block.get("image_url")).get("url") if isinstance(value, Mapping) else value
        return url if isinstance(url, str) and url.startswith(("https://", "http://")) else None
    source: Final = _mapping(block.get("source"))
    source_url: Final = source.get("url")
    return (
        source_url
        if block.get("type") == "image" and source.get("type") == "url" and isinstance(source_url, str)
        else None
    )


def _counting_content(value: object) -> tuple[object, int]:
    if isinstance(value, (list, tuple)):
        items: Final = tuple(_counting_content(item) for item in _ITEMS.validate_python(value))
        return [item for item, _ in items], sum(extra for _, extra in items)
    if not isinstance(value, Mapping):
        return value, 0
    block: Final = _OBJECT.validate_python(value)
    remote: Final = _remote_image(block)
    if remote is not None:
        return {"type": "image_url", "image_url": {"url": remote, "detail": "auto"}}, (
            high_detail_image_token_upper_bound() - calculate_img_tokens(remote, mode="auto")
        )
    if "content" not in block:
        return block, 0
    content, extra = _counting_content(block["content"])
    return {**block, "content": content}, extra


def _count(data: Mapping[str, object], model: str) -> int | AdmissionProblem:
    if litellm.disable_token_counter:
        return AdmissionProblem("messages/input", "本地 token 计数已关闭，无法确认请求是否符合上下文上限", False)
    try:
        responses_error: Final = _responses_problem(data)
        if responses_error is not None:
            return responses_error
        messages, remote_image_extra = _counting_content(_messages(data))
        raw_tools: Final = data.get("tools")
        tools: Final = list(_MESSAGES.validate_python(raw_tools)) if raw_tools is not None else None
        return (
            token_counter(
                model=model,
                messages=cast(Sequence[AllMessageValues], messages),  # cast-ok: 计数器支持三种协议，类型声明仅列出 Chat
                tools=cast(list[ChatCompletionToolParam] | None, tools),  # cast-ok: 计数器统一格式化三种协议的工具定义
                tool_choice=_TOOL_CHOICE.validate_python(data.get("tool_choice")),
                use_default_image_token_count=False,
            )
            + remote_image_extra
        )
    except Exception as error:  # noqa: BLE001  # 计数库异常必须转成拒绝，不能继续向上游发送未知长度输入
        return AdmissionProblem(
            "messages/input/tools",
            f"无法可靠计算请求 token（{type(error).__name__}），请压缩上下文或开启新会话后重试",
            False,
        )


def _raise_invalid(problem: AdmissionProblem, model: str) -> None:
    error_type: Final = MalformedRequestFieldError if problem.malformed else litellm.BadRequestError
    raise error_type(
        message=f"{problem.field} {problem.message}",
        model=model,
        llm_provider="",
        body={"type": "invalid_request_error", "param": problem.field},
    )


def _raise_context(model: str, counts: Sequence[tuple[Candidate, int]]) -> None:
    smallest: Final = min(count for _, count in counts)
    largest: Final = max(candidate.limit or 0 for candidate, _ in counts)
    raise litellm.ContextWindowExceededError(
        message=(
            f"本地上下文预检拒绝请求，估算输入至少 {smallest} tokens，可用部署输入上限最多 {largest} tokens。"
            "请先压缩上下文（Claude Code 可用 /compact）或开启新会话"
        ),
        model=model,
        llm_provider="",
    )


def _is_counted_claude(model: str) -> bool:
    return model.removeprefix("anthropic/").startswith(("claude-opus-", "claude-sonnet-", "claude-haiku-"))


def _claude_body(data: Mapping[str, object], model: str, call_type: CallTypes) -> dict[str, JsonValue]:
    if call_type in (CallTypes.anthropic_messages, CallTypes.aanthropic_messages):
        return _JSON_OBJECT.validate_python(prepare_native_request_body(data, model))
    config: Final = AnthropicConfig()
    keys: Final = frozenset(DEFAULT_CHAT_COMPLETION_PARAM_VALUES) | frozenset(COUNT_TOKEN_OPTION_NAMES)
    request_options: Final = {
        key: copy.deepcopy(value)
        for key, value in data.items()
        if key in keys and key not in ("model", "custom_llm_provider", "messages", "metadata")
    }
    options: Final = _JSON_OBJECT.validate_python(
        get_optional_params(
            model=model,
            custom_llm_provider="anthropic",
            **request_options,  # pyright: ignore[reportArgumentType]  # 共享参数转换器负责校验输入，出口验证类型
        )
    )
    return _JSON_OBJECT.validate_python(
        config.transform_request(  # pyright: ignore[reportUnknownMemberType]  # 上游转换器仍返回裸 dict，出口验证类型
            model=model,
            messages=cast(  # cast-ok: 共享消息转换器返回 Chat 形状，供应商转换器继续校验
                list[AllMessageValues], copy.deepcopy(list(_messages(data)))
            ),
            optional_params=options,
            litellm_params={
                **data,
                **{
                    key: copy.deepcopy(data[key])
                    for key in ("metadata", "litellm_metadata", "model_info")
                    if key in data
                },
            },
            headers={},
        )
    )


def _generation_count_headers(data: Mapping[str, object]) -> Mapping[str, str]:
    scoped: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
        _PROVIDER_HEADERS.validate_python(data.get("provider_specific_header")), "anthropic"
    )
    outgoing: Final = {**_mapping(data.get("headers")), **_mapping(data.get("extra_headers")), **scoped}
    return {
        **{
            key: value
            for key, value in outgoing.items()
            if key.lower() in _CLAUDE_COUNT_HEADERS and isinstance(value, str)
        },
        "anthropic-beta": merge_anthropic_beta_headers(
            None,
            tuple(
                value for key, value in outgoing.items() if key.lower() == "anthropic-beta" and isinstance(value, str)
            ),
        ),
    }


def _reject_claude_preflight(field: str, model: str, detail: str = "无法可靠核验完整 Claude 上下文") -> None:
    raise RequestAdmissionError(
        message=f"{field} {detail}，请用纯文本摘要开启新会话；/compact 后仍有加密历史时也会拒绝。生成请求未发送",
        model=model,
        llm_provider="anthropic",
        body={"type": "invalid_request_error", "param": field, "code": "request_admission_rejected"},
    )


def _check_claude_content(content: object, path: str, model: str) -> None:
    if isinstance(content, str) or content is None:
        return
    for index, block in enumerate(_MESSAGES.validate_python(content)):
        _check_claude_block(block, f"{path}[{index}]", model)


def _check_claude_block(block: Mapping[str, object], path: str, model: str) -> None:
    kind: Final = block.get("type")
    if block.get("encrypted_content") is not None or block.get("signature") is not None:
        _reject_claude_preflight(path, model)
    if kind in ("text", "input_text", "output_text", "tool_use"):
        return
    if kind == "tool_result":
        _check_claude_content(block.get("content"), f"{path}.content", model)
        return
    if kind in ("image", "document"):
        source: Final = _mapping(block.get("source"))
        if source.get("type") == "base64" or (kind == "document" and source.get("type") == "text"):
            return
    if kind in ("image_url", "input_image"):
        image_url: Final = block.get("image_url")
        url: Final = image_url if isinstance(image_url, str) else _mapping(image_url).get("url")
        if isinstance(url, str) and url.startswith("data:image/"):
            return
    if kind == "file":
        file_data: Final = _mapping(block.get("file")).get("file_data")
        if isinstance(file_data, str) and file_data.startswith("data:application/pdf;base64,"):
            return
    _reject_claude_preflight(path, model)


def _check_claude_original_history(data: Mapping[str, object], model: str) -> None:
    # 转换器可能丢弃未知历史块，必须在转换前拒绝，避免把丢失后的低计数当成完整计数。
    _check_claude_content(data.get("system"), "system", model)
    for index, message in enumerate(_MESSAGES.validate_python(data.get("messages") or [])):
        for field in ("thinking_blocks", "reasoning_content", "reasoning", "provider_specific_fields"):
            if message.get(field):
                _reject_claude_preflight(f"messages[{index}].{field}", model)
        _check_claude_content(message.get("content"), f"messages[{index}].content", model)
    raw_input: Final = data.get("input")
    if raw_input is None or isinstance(raw_input, str):
        return
    for index, item in enumerate(_MESSAGES.validate_python(raw_input)):
        _check_claude_response_item(item, f"input[{index}]", model)


def _check_claude_response_item(item: Mapping[str, object], path: str, model: str) -> None:
    if item.get("encrypted_content") is not None or item.get("signature") is not None:
        _reject_claude_preflight(path, model)
    if item.get("type") in ("reasoning", "compaction") or item.get("provider_specific_fields"):
        _reject_claude_preflight(path, model)
    if item.get("type") not in (None, "message", "function_call", "function_call_output"):
        _reject_claude_preflight(path, model)
    if item.get("type") is None and item.get("role") not in ("user", "assistant", "system", "developer"):
        _reject_claude_preflight(path, model)
    _check_claude_content(item.get("content"), f"{path}.content", model)
    if item.get("type") == "function_call_output" and isinstance(item.get("output"), (list, tuple)):
        _check_claude_content(item["output"], f"{path}.output", model)


def _check_claude_countable_body(body: Mapping[str, object], model: str) -> None:
    # 密文不能证明计数完整；只检查协议字段，不扫描工具参数中的同名键。
    _check_claude_content(body.get("system"), "system", model)
    if body.get("mcp_servers"):
        _reject_claude_preflight("mcp_servers", model)
    tool_choice: Final = body.get("tool_choice")
    choice_type: Final = tool_choice if isinstance(tool_choice, str) else _mapping(tool_choice).get("type")
    thinking: Final = _mapping(body.get("thinking"))
    if choice_type in ("any", "tool") and (
        (bool(thinking) and thinking.get("type") != "disabled") or "effort" in _mapping(body.get("output_config"))
    ):
        _reject_claude_preflight("tool_choice", model, "强制工具调用与 thinking/effort 不能保证计数和生成一致")
    for index, tool in enumerate(_MESSAGES.validate_python(body.get("tools") or [])):
        if tool.get("type") not in (None, "custom"):
            _reject_claude_preflight(f"tools[{index}].type", model)
    for index, message in enumerate(_MESSAGES.validate_python(body.get("messages", []))):
        _check_claude_content(message.get("content"), f"messages[{index}].content", model)


def _normalized_claude_base(base: str) -> str:
    parts: Final = urlsplit(base)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.netloc.endswith(":")
        or parts.port == 0
        or any(character.isspace() for character in base)
    ):
        raise ValueError("Claude 上游地址配置无效")
    return resolve_anthropic_base(base)


def _verified_claude_base(api_base: str | None, model: str) -> str:
    # 白名单只读进程环境，不能从请求参数或 model_info 扩大信任范围。
    try:
        configured: Final = _VERIFIED_BASES.validate_json(
            os.getenv("CLAUDE_PREFLIGHT_VERIFIED_BASES", '["https://api.anthropic.com"]')
        )
        verified: Final = frozenset(_normalized_claude_base(entry) for entry in configured)
        base: Final = _normalized_claude_base(AnthropicModelInfo.get_api_base(api_base) or "https://api.anthropic.com")
        if base in verified:
            return base
    except ValueError:
        pass
    raise RequestAdmissionError(
        message="Claude 上游未登记预检验收或白名单配置无效，生成请求未发送。请管理员核验无 cloak 和请求改写后配置 CLAUDE_PREFLIGHT_VERIFIED_BASES",
        model=model,
        llm_provider="anthropic",
        body={"type": "invalid_request_error", "code": "request_admission_rejected"},
    )


async def _count_claude(data: Mapping[str, object], call_type: CallTypes) -> int:
    credential_name: Final = data.get("litellm_credential_name")
    credentials: Final = (
        CredentialAccessor.get_credential_values(credential_name) if isinstance(credential_name, str) else {}
    )
    params: Final = {**credentials, **data}
    model: Final = str(params.get("model", "")).removeprefix("anthropic/")
    provider: Final = params.get("custom_llm_provider")
    if provider not in (None, "anthropic") or not _is_counted_claude(model):
        raise ValueError("Claude 部署不是可核验的 Anthropic 上游")
    base: Final = _verified_claude_base(_OPTIONAL_STRING.validate_python(params.get("api_base")), model)
    async with asyncio.timeout(30):
        auth: Final = await AnthropicModelInfo.aget_auth_header(
            api_key=_OPTIONAL_STRING.validate_python(params.get("api_key")) or litellm.anthropic_key or litellm.api_key,
            api_base=base,
            litellm_params=params,
            allow_workload_identity=True,
        )
        if auth is None:
            raise ValueError("Claude 上游计数缺少凭据")
        body: Final = await offload_token_count(_claude_body)(params, model, call_type)
        _check_claude_countable_body(body, model)
        generation_headers: Final = _generation_count_headers(params)
        messages: Final = _JSON_MESSAGES.validate_python(body.get("messages"))
        per_turn_betas: Final = (
            (ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER,) if messages_carry_output_config(messages) else ()
        )
        config: Final = AnthropicMessagesConfig()
        count_headers: Final = _HEADERS.validate_python(
            config._finalize_messages_headers(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # 复用功能头
                headers={
                    **{key: value for key, value in generation_headers.items() if key != "anthropic-beta"},
                    **auth,
                    "anthropic-beta": merge_anthropic_beta_headers(
                        auth.get("anthropic-beta"),
                        (generation_headers.get("anthropic-beta", ""), *per_turn_betas),
                    ),
                },
                optional_params=body,
                messages=list[object](messages),
            )
        )
        response: Final = await AnthropicCountTokensHandler().handle_count_tokens_request(
            model=model,
            messages=messages,
            auth_header={key: value for key, value in count_headers.items() if key.lower() != "content-type"},
            api_base=base,
            timeout=30,
            tools=_JSON_MESSAGES.validate_python(body["tools"]) if body.get("tools") is not None else None,
            system=body.get("system"),
            optional_params={key: body[key] for key in COUNT_TOKEN_OPTION_NAMES if key in body},
        )
    count: Final = response.get("input_tokens")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Claude 上游未返回有效 input_tokens")
    return count


def _admit(
    data: Mapping[str, object], candidates: Sequence[Candidate], model: str, pending_compaction: bool = False
) -> list[dict[str, object]]:
    image_error: Final = _image_problem(data)
    if image_error is not None:
        _raise_invalid(image_error, model)
    bounded: Final = tuple(
        candidate for candidate in candidates if candidate.limit is not None or candidate.local_response
    )
    if candidates and not bounded:
        _raise_invalid(
            AdmissionProblem("model_info.max_input_tokens", "部署未配置可用输入上限，请管理员补齐后重试", False), model
        )
    if pending_compaction:
        return [candidate.deployment for candidate in bounded]
    models: Final = frozenset(
        candidate.model
        for candidate in bounded
        if candidate.limit is not None and not _is_counted_claude(candidate.model)
    )
    counts: Final = {counted_model: _count(data, counted_model) for counted_model in models}
    for count in counts.values():
        if isinstance(count, AdmissionProblem):
            _raise_invalid(count, model)
    measured: Final = tuple((candidate, counts[candidate.model]) for candidate in bounded if candidate.model in counts)
    allowed: Final = [
        candidate.deployment
        for candidate in bounded
        if candidate.local_response
        or _is_counted_claude(candidate.model)
        or (candidate.limit is not None and cast(int, counts[candidate.model]) < candidate.limit)
    ]
    if candidates and not allowed:
        _raise_context(model, cast(Sequence[tuple[Candidate, int]], measured))  # cast-ok: 上面的循环已拒绝所有计数错误
    return allowed


class RequestAdmission(CustomLogger):
    def __init__(
        self, claude_counter: Callable[[Mapping[str, object], CallTypes], Awaitable[int]] = _count_claude
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # 上游 callback 构造器的 kwargs 尚未注解
        self._claude_counter = claude_counter

    @staticmethod
    def _preflight_error(error: Exception, model: str) -> RequestAdmissionError:
        if isinstance(error, RequestAdmissionError):
            return error
        return RequestAdmissionError(
            message=f"Claude 本地预检或上游计数失败（{type(error).__name__}），生成请求未发送，请检查输入或稍后重试",
            model=model,
            llm_provider="anthropic",
            body={"type": "invalid_request_error", "code": "request_admission_rejected"},
        )

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]],
        messages: list[AllMessageValues] | None,
        request_kwargs: dict[str, object] | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict[str, object]]:
        data: Final = {**(request_kwargs or {}), **({"messages": messages} if messages is not None else {})}
        if "messages" not in data and "input" not in data:
            return healthy_deployments
        checked: Final = tuple(deployment for deployment in healthy_deployments if not _is_non_chat(deployment))
        if not checked:
            return healthy_deployments
        candidates: Final = tuple(_candidate(deployment) for deployment in checked)
        try:
            admitted: Final = await offload_token_count(_admit)(data, candidates, model, compaction_pending(data))
        except Exception as error:
            if not all(_is_counted_claude(candidate.model) for candidate in candidates):
                raise
            raise self._preflight_error(error, model) from None
        return [deployment for deployment in healthy_deployments if deployment not in checked or deployment in admitted]

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict[str, object] | None:
        if call_type is None or call_type not in _CALLS or ("messages" not in kwargs and "input" not in kwargs):
            return None
        model: Final = kwargs.get("model")
        deployment: Final = {"litellm_params": kwargs, "model_info": kwargs.get("model_info")}
        candidate: Final = _candidate(deployment)
        try:
            await offload_token_count(_admit)(kwargs, (candidate,), model if isinstance(model, str) else "")
            if not _is_counted_claude(candidate.model):
                return None
            _check_claude_original_history(kwargs, candidate.model)
            extra_body: Final = (
                _OBJECT.validate_python(kwargs["extra_body"]) if kwargs.get("extra_body") is not None else {}
            )
            if kwargs.get("container") or extra_body.get("container"):
                _reject_claude_preflight(
                    "container", candidate.model, "Claude 上游计数不支持 container，无法核验隐藏上下文"
                )
            if call_type in (CallTypes.completion, CallTypes.acompletion) and extra_body:
                _reject_claude_preflight(
                    "extra_body", candidate.model, "Claude Chat 无法核验 extra_body，请使用 Messages 或顶层参数"
                )
            problem: Final = (
                _responses_problem(kwargs) if call_type in (CallTypes.responses, CallTypes.aresponses) else None
            )
            if problem is not None:
                _reject_claude_preflight(problem.field, candidate.model, problem.message)
            if call_type in (CallTypes.responses, CallTypes.aresponses):
                endpoints: Final = _mapping(kwargs.get("model_info")).get("supported_endpoints")
                native: Final = isinstance(endpoints, (list, tuple)) and "/v1/responses" in endpoints
                if (
                    kwargs.get("custom_llm_provider") not in (None, "anthropic")
                    or not isinstance(model, str)
                    or not _is_counted_claude(model)
                    or (native and kwargs.get("use_chat_completions_api") is not True)
                ):
                    _reject_claude_preflight(
                        "input", candidate.model, "Claude 原生 Responses 无法核验，请使用 Messages 或 Chat 桥接"
                    )
                return None
            count: Final = await self._claude_counter(kwargs, call_type)
            if candidate.limit is not None and count >= candidate.limit:
                raise RequestAdmissionContextWindowExceededError(
                    message=f"Claude 上游计数 {count} tokens，达到输入上限 {candidate.limit}。请先 /compact 或开启新会话",
                    model=candidate.model,
                    llm_provider="anthropic",
                )
            return None
        except Exception as error:
            if not _is_counted_claude(candidate.model):
                raise
            raise self._preflight_error(error, candidate.model) from None


callback: Final = RequestAdmission()
