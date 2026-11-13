import asyncio
import base64
import binascii
import copy
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from itertools import accumulate
from typing import Final, Literal, cast

import httpx
from openai.types.responses.response_create_params import ResponseInputParam
from opentelemetry.trace import Span
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.anthropic_beta_gated_fields import strip_beta_gated_request_body
from litellm.anthropic_beta_headers_manager import (
    update_request_with_filtered_beta,  # pyright: ignore[reportUnknownVariableType]  # 共享过滤器仍接收裸 dict，出口已标注
)
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_CHAT_COMPLETION_PARAM_VALUES, MAX_IMAGE_URL_DOWNLOAD_SIZE_MB
from litellm.exceptions import MalformedRequestFieldError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.token_counter import (
    calculate_img_tokens,
    high_detail_image_token_upper_bound,
    offload_token_count,
    token_counter,  # pyright: ignore[reportUnknownVariableType]  # 上游 custom_tokenizer 包含裸 dict
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import COUNT_TOKEN_OPTION_NAMES
from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig
from litellm.llms.anthropic.messages.native.utils import (
    anthropic_system_to_openai_message,
    prepare_native_request_body,
)
from litellm.llms.custom_httpx.llm_http_handler import (
    _merge_messages_header_sources,  # pyright: ignore[reportPrivateUsage]  # 外部插件复用实际生成的头合并规则
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig as ResponsesConfig,
)
from litellm.router_strategy.complexity_router.context_compaction import compaction_pending
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.utils import CallTypes, CallTypesLiteral, ModelResponse, ProviderSpecificHeader
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
_REMOTE_THRESHOLD: Final = 700_000
_RECHECK_MARGIN: Final = 50_000
_MAX_LOCAL_INCREMENT: Final = 20_000
_CACHE_TTL: Final = 300
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

    @property
    def upstream_model(self) -> str:
        return str(_mapping(self.deployment.get("litellm_params")).get("model") or self.model)


@dataclass(frozen=True, slots=True)
class PreparedClaudeRequest:
    model: str
    api_base: str
    headers: dict[str, str]
    body: dict[str, JsonValue]
    cacheable: bool


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


def _is_claude(model: str) -> bool:
    return model.removeprefix("anthropic/").startswith(("claude-opus-", "claude-sonnet-", "claude-haiku-"))


def _prepare_claude_chat(
    data: Mapping[str, object],
    model: str,
    api_base: str,
    api_key: str | None,
    headers: dict[str, str],
    cacheable: bool,
) -> PreparedClaudeRequest:
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
    messages: Final = cast(  # cast-ok: 共享消息转换器返回 Chat 形状，供应商转换器继续校验
        list[AllMessageValues], copy.deepcopy(list(_messages(data)))
    )
    params: Final = {
        **data,
        **{key: copy.deepcopy(data[key]) for key in ("metadata", "litellm_metadata", "model_info") if key in data},
    }
    validated: Final = _HEADERS.validate_python(
        config.validate_environment(  # pyright: ignore[reportUnknownMemberType]  # 复用 Chat 认证优先级，出口验证类型
            api_key=api_key,
            headers=dict(headers),
            model=model,
            messages=messages,
            optional_params=options,
            litellm_params=params,
            api_base=api_base,
        )
    )
    body: Final = _JSON_OBJECT.validate_python(
        config.transform_request(  # pyright: ignore[reportUnknownMemberType]  # 上游转换器仍返回裸 dict，出口验证类型
            model=model,
            messages=messages,
            optional_params=options,
            litellm_params=params,
            headers=validated,
        )
    )
    final_headers, filtered = update_request_with_filtered_beta(validated, body, "anthropic")
    final_body: Final = _JSON_OBJECT.validate_python(
        strip_beta_gated_request_body(
            filtered, final_headers, drop_params=params.get("drop_params"), model=model, provider="anthropic"
        )
    )
    return PreparedClaudeRequest(model, api_base, final_headers, final_body, cacheable)


def _generation_count_headers(data: Mapping[str, object], call_type: CallTypes) -> dict[str, str]:
    native: Final = call_type in (CallTypes.anthropic_messages, CallTypes.aanthropic_messages)
    scoped: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
        _PROVIDER_HEADERS.validate_python(data.get("provider_specific_header")),
        "anthropic",
        native_anthropic_messages=native,
    )
    if not native:
        proxy: Final = litellm.proxy_auth.get_auth_headers() if litellm.proxy_auth is not None else {}
        return _HEADERS.validate_python(
            {**_mapping(data.get("headers")), **_mapping(data.get("extra_headers")), **proxy, **scoped}
        )
    return _HEADERS.validate_python(
        _merge_messages_header_sources(_mapping(data.get("headers")), _mapping(data.get("extra_headers")), scoped)
    )


async def _prepare_claude(data: Mapping[str, object], call_type: CallTypes) -> PreparedClaudeRequest:
    params: Final = _resolved_params(data)
    model: Final = str(params.get("model", "")).removeprefix("anthropic/")
    provider: Final = params.get("custom_llm_provider")
    if provider not in (None, "anthropic") or not _is_claude(model):
        raise ValueError("Claude 部署不是可核验的 Anthropic 上游")
    native: Final = call_type in (CallTypes.anthropic_messages, CallTypes.aanthropic_messages)
    base: Final = AnthropicModelInfo.resolve_anthropic_api_base(
        _OPTIONAL_STRING.validate_python(params.get("api_base")) or (None if native else litellm.api_base)
    )
    api_key: Final = AnthropicModelInfo.get_api_key(
        _OPTIONAL_STRING.validate_python(params.get("api_key")) or litellm.anthropic_key or litellm.api_key
    )
    async with asyncio.timeout(30):
        generation_headers: Final = await offload_token_count(_generation_count_headers)(params, call_type)
        cacheable: Final = litellm.proxy_auth is None and bool(
            api_key
            or AnthropicModelInfo.get_auth_token()
            or any(key.lower() in ("authorization", "x-api-key") and value for key, value in generation_headers.items())
        )
        if not native:
            return await offload_token_count(_prepare_claude_chat)(
                params, model, base, api_key, generation_headers, cacheable
            )
        body: Final = _JSON_OBJECT.validate_python(
            await offload_token_count(prepare_native_request_body)(params, model, headers=generation_headers)
        )
        messages: Final = _JSON_MESSAGES.validate_python(body.get("messages"))
        config: Final = AnthropicMessagesConfig()
        validated_headers, _ = await config.avalidate_anthropic_messages_environment(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # 共享认证接口的返回值在下方校验
            headers=generation_headers,
            model=model,
            optional_params=body,
            messages=list[object](messages),
            litellm_params=params,
            api_key=api_key,
            api_base=base,
        )
    return PreparedClaudeRequest(model, base, _HEADERS.validate_python(validated_headers), body, cacheable)


async def _send_count(prepared: PreparedClaudeRequest) -> int:
    body: Final = prepared.body
    async with asyncio.timeout(30):
        response: Final = await AnthropicCountTokensHandler().handle_count_tokens_request(
            model=prepared.model,
            messages=_JSON_MESSAGES.validate_python(body.get("messages")),
            auth_header={key: value for key, value in prepared.headers.items() if key.lower() != "content-type"},
            api_base=prepared.api_base,
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
    data: Mapping[str, object],
    candidates: Sequence[Candidate],
    model: str,
    pending_compaction: bool = False,
    local_counter: Callable[[Mapping[str, object], str], int | AdmissionProblem] = _count,
    known_counts: Mapping[str, int | AdmissionProblem] | None = None,
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
    models: Final = frozenset(candidate.model for candidate in bounded if candidate.limit is not None)
    counts: Final = {
        counted_model: known_counts[counted_model]
        if known_counts is not None and counted_model in known_counts
        else local_counter(data, counted_model)
        for counted_model in models
    }
    for count in counts.values():
        if isinstance(count, AdmissionProblem):
            _raise_invalid(count, model)
    measured: Final = tuple((candidate, counts[candidate.model]) for candidate in bounded if candidate.model in counts)
    allowed: Final = [
        candidate.deployment
        for candidate in bounded
        if candidate.local_response
        or (_is_claude(candidate.upstream_model) and cast(int, counts[candidate.model]) >= _REMOTE_THRESHOLD)
        or (candidate.limit is not None and cast(int, counts[candidate.model]) < candidate.limit)
    ]
    if candidates and not allowed:
        _raise_context(model, cast(Sequence[tuple[Candidate, int]], measured))  # cast-ok: 上面的循环已拒绝所有计数错误
    return allowed


class CountRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: StrictInt = Field(gt=0)
    local_tokens: StrictInt = Field(gt=0)
    messages: StrictInt = Field(ge=0)
    expires_at: float
    source: Literal["remote", "usage"]


class PendingObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str
    root: str
    prefix: str
    model: str
    local_model: str


class ObservedUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: StrictInt = Field(gt=0)
    completion_tokens: StrictInt = Field(ge=0)
    iterations: JsonValue = None
    server_tool_use: JsonValue = None


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_content(content: JsonValue) -> JsonValue:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return content
    return [_canonical_block(block) for block in content]


def _canonical_block(block: JsonValue) -> JsonValue:
    if not isinstance(block, dict):
        return block
    return {
        key: _canonical_content(value) if key == "content" and block.get("type") == "tool_result" else value
        for key, value in block.items()
        if key != "cache_control" and value is not None
    }


def _message_digest(message: Mapping[str, JsonValue]) -> str:
    return _digest(
        {
            key: _canonical_content(value) if key == "content" else value
            for key, value in message.items()
            if value is not None
        }
    )


def _extend_digest(prefix: str, message: str) -> str:
    return _digest((prefix, message))


def _fingerprints(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    root: Final = _digest(
        {
            key: _canonical_content(value) if key == "system" else value
            for key, value in body.items()
            if key not in ("messages", "stream", "cache_control", "metadata")
        }
    )
    messages: Final = _JSON_MESSAGES.validate_python(body.get("messages"))
    return tuple(accumulate((_message_digest(message) for message in messages), _extend_digest, initial=root))


def _resolved_params(data: Mapping[str, object]) -> dict[str, object]:
    name: Final = data.get("litellm_credential_name")
    credentials: Final = CredentialAccessor.get_credential_values(name) if isinstance(name, str) else {}
    return {**credentials, **data}


def _count_scope(
    data: Mapping[str, object], prepared: PreparedClaudeRequest, local_model: str, call_type: CallTypes
) -> str | None:
    metadata: Final = _mapping(data.get("litellm_metadata") or data.get("metadata"))
    caller: Final = metadata.get("user_api_key_hash")
    deployment: Final = _mapping(data.get("model_info")).get("id")
    if not prepared.cacheable or not isinstance(caller, str) or not caller or not isinstance(deployment, str):
        return None
    headers: Final = {
        key: ",".join(sorted(frozenset(beta.strip() for beta in value.split(",") if beta.strip())))
        if key.lower() == "anthropic-beta"
        else value
        for key, value in prepared.headers.items()
    }
    return _digest(
        (
            caller,
            deployment,
            call_type.value,
            prepared.model,
            local_model,
            prepared.api_base,
            headers,
            _mapping(data.get("metadata")).get("user_id"),
        )
    )


def _count_key(scope: str, fingerprint: str) -> str:
    return f"claude-admission:v1:{scope}:{fingerprint}"


def _plain_content(content: object) -> bool:
    if isinstance(content, str):
        return True
    if not isinstance(content, (list, tuple)):
        return False
    return all(_plain_block(block) for block in _ITEMS.validate_python(content))


def _plain_block(value: object) -> bool:
    block: Final = _mapping(value)
    if block.get("type") == "text":
        return isinstance(block.get("text"), str) and not frozenset(block) - {"type", "text", "cache_control"}
    if block.get("type") == "tool_result":
        return not frozenset(block) - {
            "type",
            "tool_use_id",
            "content",
            "is_error",
            "cache_control",
        } and _plain_content(block.get("content"))
    return False


def _plain_increment(messages: Sequence[Mapping[str, JsonValue]]) -> bool:
    return bool(messages) and all(
        message.get("role") == "user"
        and not frozenset(message) - {"role", "content"}
        and _plain_content(message.get("content"))
        for message in messages
    )


def _can_observe(body: Mapping[str, JsonValue]) -> bool:
    context: Final = body.get("context_management")
    return (
        context in (None, {}, {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})
        and not any(body.get(key) for key in ("container", "compaction", "mcp_servers"))
        and all(
            tool.get("type") in (None, "custom") for tool in _JSON_MESSAGES.validate_python(body.get("tools") or [])
        )
    )


class RequestAdmission(CustomLogger):
    def __init__(
        self,
        claude_counter: Callable[[Mapping[str, object], CallTypes], Awaitable[int]] | None = None,
        local_counter: Callable[[Mapping[str, object], str], int | AdmissionProblem] = _count,
        cache: DualCache | None = None,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # 上游 callback 构造器的 kwargs 尚未注解
        self._claude_counter = claude_counter
        self._local_counter = local_counter
        self._cache = cache or DualCache(
            in_memory_cache=InMemoryCache(max_size_in_memory=1000, default_ttl=_CACHE_TTL),
            default_in_memory_ttl=_CACHE_TTL,
            default_redis_ttl=_CACHE_TTL,
        )
        self._locks = tuple(asyncio.Lock() for _ in range(64))

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> None:
        from litellm.proxy.common_utils.config_sync_pubsub import coordination_redis_cache

        self._cache.attach_redis_cache(cache.redis_cache or coordination_redis_cache(), default_redis_ttl=_CACHE_TTL)

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
        admitted: Final = await offload_token_count(_admit)(
            data, candidates, model, compaction_pending(data), self._local_counter
        )
        return [deployment for deployment in healthy_deployments if deployment not in checked or deployment in admitted]

    async def _cached_count(self, scope: str, fingerprint: str) -> CountRecord | None:
        value: Final = await self._cache.async_get_cache(_count_key(scope, fingerprint))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # 缓存出口在下方用模型校验
        try:
            record: Final = (
                CountRecord.model_validate_json(value) if isinstance(value, str) else CountRecord.model_validate(value)
            )
        except ValidationError:
            return None
        return record if record.expires_at > time.time() else None

    async def _save_count(self, scope: str, fingerprint: str, record: CountRecord) -> None:
        await self._cache.async_set_cache(  # pyright: ignore[reportUnknownMemberType]  # 旧缓存接口未注解，值已序列化
            _count_key(scope, fingerprint), record.model_dump(mode="json"), ttl=_CACHE_TTL
        )

    async def _reuse_count(
        self, scope: str, fingerprints: tuple[str, ...], body: Mapping[str, JsonValue], model: str, limit: int
    ) -> int | None:
        exact: Final = await self._cached_count(scope, fingerprints[-1])
        if exact is not None and exact.source == "remote":
            return exact.tokens
        if not _can_observe(body):
            return None
        messages: Final = _JSON_MESSAGES.validate_python(body.get("messages"))
        for position in range(len(messages) - 1, max(-1, len(messages) - 9), -1):
            if (
                estimate := await self._increment_count(scope, fingerprints[position], messages, position, model, limit)
            ) is not None:
                return estimate
        return None

    async def _increment_count(
        self, scope: str, fingerprint: str, messages: list[dict[str, JsonValue]], position: int, model: str, limit: int
    ) -> int | None:
        record: Final = await self._cached_count(scope, fingerprint)
        suffix: Final = messages[position:]
        if record is None or record.messages != position or not _plain_increment(suffix):
            return None
        increment: Final = await offload_token_count(self._local_counter)({"messages": suffix}, model)
        if isinstance(increment, AdmissionProblem) or increment > _MAX_LOCAL_INCREMENT:
            return None
        estimate: Final = record.tokens + math.ceil(increment * max(1.0, record.tokens / record.local_tokens))
        return estimate if estimate < limit - _RECHECK_MARGIN else None

    async def _check_claude(
        self, data: Mapping[str, object], call_type: CallTypes, candidate: Candidate, local_tokens: int
    ) -> None:
        prepared: Final = await _prepare_claude(data, call_type)
        body: Final = prepared.body
        fingerprints: Final = await offload_token_count(_fingerprints)(body)
        scope: Final = _count_scope(data, prepared, candidate.model, call_type)
        limit: Final = candidate.limit
        if limit is None:
            return
        async with self._locks[int(fingerprints[-1][:8], 16) % len(self._locks)]:
            reused: Final = (
                await self._reuse_count(scope, fingerprints, body, candidate.model, limit)
                if scope is not None
                else None
            )
            count: Final = (
                reused
                if reused is not None
                else await _send_count(prepared)
                if self._claude_counter is None
                else await self._claude_counter(data, call_type)
            )
            if isinstance(count, bool) or count <= 0:
                raise ValueError("Claude 上游未返回有效 input_tokens")
            if reused is None and scope is not None:
                await self._save_count(
                    scope,
                    fingerprints[-1],
                    CountRecord(
                        tokens=count,
                        local_tokens=local_tokens,
                        messages=len(fingerprints) - 1,
                        expires_at=time.time() + _CACHE_TTL,
                        source="remote",
                    ),
                )
        if count >= limit:
            raise litellm.ContextWindowExceededError(
                message=f"Claude 上游计数 {count} tokens，达到输入上限 {limit}。请先 /compact 或开启新会话",
                model=candidate.model,
                llm_provider="anthropic",
            )
        logging_obj: Final = data.get("litellm_logging_obj")
        if (
            scope is not None
            and isinstance(logging_obj, Logging)
            and call_type
            in (
                CallTypes.anthropic_messages,
                CallTypes.aanthropic_messages,
            )
        ):
            seed: Final = PendingObservation(
                scope=scope,
                root=fingerprints[0],
                prefix=fingerprints[-1],
                model=prepared.model,
                local_model=candidate.model,
            )
            logging_obj.model_call_details = {**logging_obj.model_call_details, "_claude_admission_observation": seed}
        verbose_proxy_logger.debug(
            "Claude 上下文预检，本地=%s，核验=%s，复用=%s", local_tokens, count, reused is not None
        )

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: CallTypes | None
    ) -> dict[str, object] | None:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if isinstance(logging_obj, Logging):
            logging_obj.model_call_details = {
                key: value
                for key, value in _mapping(logging_obj.model_call_details).items()
                if key != "_claude_admission_observation"
            }
        if call_type is None or call_type not in _CALLS or ("messages" not in kwargs and "input" not in kwargs):
            return None
        model: Final = kwargs.get("model")
        candidate: Final = _candidate({"litellm_params": kwargs, "model_info": kwargs.get("model_info")})
        count: Final = (
            await offload_token_count(self._local_counter)(kwargs, candidate.model)
            if candidate.limit is not None
            else None
        )
        await offload_token_count(_admit)(
            kwargs,
            (candidate,),
            str(model or ""),
            False,
            self._local_counter,
            known_counts={candidate.model: count} if count is not None else {},
        )
        if not _is_claude(candidate.upstream_model):
            return None
        if kwargs.get("container") or _mapping(kwargs.get("extra_body")).get("container"):
            _raise_invalid(
                AdmissionProblem("container", "Claude container 的隐藏上下文无法本地核验", False), candidate.model
            )
        if call_type in (CallTypes.completion, CallTypes.acompletion) and kwargs.get("extra_body"):
            _raise_invalid(
                AdmissionProblem(
                    "extra_body", "Claude Chat 无法本地核验 extra_body，请使用 Messages 或顶层参数", False
                ),
                candidate.model,
            )
        if isinstance(count, AdmissionProblem):
            _raise_invalid(count, candidate.model)
            return None
        if count is None or count < _REMOTE_THRESHOLD:
            return None
        if call_type in (CallTypes.responses, CallTypes.aresponses):
            endpoints: Final = _mapping(kwargs.get("model_info")).get("supported_endpoints")
            native: Final = isinstance(endpoints, (list, tuple)) and "/v1/responses" in endpoints
            if kwargs.get("custom_llm_provider") not in (None, "anthropic") or (
                native and kwargs.get("use_chat_completions_api") is not True
            ):
                _raise_invalid(
                    AdmissionProblem(
                        "input", "Claude 原生 Responses 无法远程核验，请使用 Messages 或 Chat 桥接", False
                    ),
                    candidate.model,
                )
            return None
        try:
            await self._check_claude(kwargs, call_type, candidate, count)
        except litellm.ContextWindowExceededError:
            raise
        except Exception as error:  # noqa: BLE001  # 远程复核失败不能使用低估值继续发送
            _raise_invalid(
                AdmissionProblem(
                    "count_tokens", f"Claude 上游计数失败（{type(error).__name__}），请求未发送，请稍后重试", False
                ),
                candidate.model,
            )
        return None

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        seed: Final = kwargs.get("_claude_admission_observation")
        response: Final = kwargs.get("httpx_response")
        if (
            not isinstance(seed, PendingObservation)
            or not isinstance(response_obj, ModelResponse)
            or not isinstance(response, httpx.Response)
        ):
            return
        if (
            response.status_code != 200
            or kwargs.get("cache_hit")
            or (kwargs.get("stream") and kwargs.get("prompt_cache_response_complete") is not True)
        ):
            return
        try:
            body: Final = _JSON_OBJECT.validate_json(response.request.content)
            fingerprints: Final = await offload_token_count(_fingerprints)(body)
            usage: Final = ObservedUsage.model_validate(getattr(response_obj, "usage", None), from_attributes=True)
            if (
                seed.model != response_obj.model
                or seed.root != fingerprints[0]
                or seed.prefix != fingerprints[-1]
                or usage.iterations
                or usage.server_tool_use
                or not _can_observe(body)
            ):
                return
            result: Final = _JSON_OBJECT.validate_python(
                LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(response_obj)
                if kwargs.get("stream")
                else response.json()
            )
            content: Final = result.get("content")
            if (
                not isinstance(content, list)
                or not content
                or result.get("stop_reason") not in ("end_turn", "tool_use", "max_tokens", "stop_sequence")
                or result.get("context_management") not in (None, {}, {"applied_edits": []})
            ):
                return
            assistant: Final[dict[str, JsonValue]] = {"role": "assistant", "content": content}
            messages: Final = _JSON_MESSAGES.validate_python(body.get("messages"))
            next_body: Final = {**body, "messages": [*messages, assistant]}
            local_tokens: Final = await offload_token_count(self._local_counter)(next_body, seed.local_model)
            if isinstance(local_tokens, AdmissionProblem) or local_tokens <= 0:
                return
            await self._save_count(
                seed.scope,
                _extend_digest(seed.prefix, _message_digest(assistant)),
                CountRecord(
                    tokens=usage.prompt_tokens + usage.completion_tokens,
                    local_tokens=local_tokens,
                    messages=len(messages) + 1,
                    expires_at=time.time() + _CACHE_TTL,
                    source="usage",
                ),
            )
        except (ValidationError, ValueError, TypeError, httpx.RequestNotRead):
            return


callback: Final = RequestAdmission()
