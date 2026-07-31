"""
在请求发给后端之前改写 messages，做两件彼此独立的事：

1. 把图片交给视觉模型识别，用识别文本替换原图片块。deepseek 等模型只支持文本，但
   Claude Code 用户会贴截图；挂上本 guardrail 后带图请求不再报 400，也不会被后端
   静默丢成 [Unsupported Image]。

2. 裸 tool_result 收尾时在末尾追加一个非空 text 块，绕开 opencode 的 tool_use id
   注册表校验（见 ensure_trailing_text_after_tool_result）。

名字只覆盖第一件事。没拆成两个 guardrail 是因为挂载点完全相同，而生产上新建一个
guardrail 要改 config.yaml 再逐个 PATCH deployment；合进已挂载的文件只需替换本文件。

这是 litellm 的外部自定义 guardrail，不修改 litellm 源码。把本文件放在 config.yaml
同目录，然后：

    model_list:
      - model_name: deepseek-v4-flash
        litellm_params:
          model: openai/deepseek-v4-flash
          api_base: os.environ/TEXT_MODEL_API_BASE
          api_key: os.environ/TEXT_MODEL_API_KEY
          guardrails: ["vision-to-text"]     # 只有这里挂了才生效

      - model_name: vision-describer
        litellm_params:
          model: openai/gpt-4o
          api_key: os.environ/OPENAI_API_KEY

    guardrails:
      - guardrail_name: "vision-to-text"
        litellm_params:
          guardrail: vision_to_text.VisionToTextGuardrail
          mode: "pre_call"
          default_on: false                  # 必须 false，否则变成全局常开
          vision_model: vision-describer
"""

import asyncio
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.router import Router
    from litellm.types.utils import CallTypesLiteral

# 追加在裸 tool_result 之后的文本。必须非空白：litellm 的
# strip_empty_text_blocks_from_anthropic_messages 用 .strip() 判空并摘掉纯空白块，
# 摘完又变回裸 tool_result 收尾，等于没修。实测空串与单空格均无效
TOOL_RESULT_TRAILING_TEXT = "."

DEFAULT_VISION_PROMPT = (
    "Describe this image in detail. Transcribe any text, code, numbers or error "
    "messages exactly as they appear. Be factual and do not speculate."
)
DEFAULT_DESCRIPTION_TEMPLATE = "[Image description: {description}]"
DEFAULT_FAILURE_TEMPLATE = "[Image could not be processed: {error}]"
DEFAULT_CACHE_TTL_SECONDS = 3600
# 视觉模型常指向负载均衡组，组内成员可能限流或临时不可用。组内重试能换到别的成员，
# 所以默认给 2 次；跨组 fallback 另行禁掉（见 _call_vision_model）
DEFAULT_VISION_NUM_RETRIES = 2


@dataclass(frozen=True, slots=True)
class ImageReference:
    """messages 里一个图片 content part 的位置与可直接传给视觉模型的 URL"""

    message_index: int
    # 从 message 的 content 往下的下标路径。顶层图片是 (i,)，tool_result 里的是 (i, j)
    path: Tuple[int, ...]
    image_url: str


def image_url_for_vision_call(content_item: Mapping[str, Any]) -> Optional[str]:
    """
    从单个 content part 取出可用作 OpenAI image_url 的地址。

    同时认两种形状：/v1/chat/completions 的 {"type": "image_url", ...} 和
    /v1/messages 的 {"type": "image", "source": {...}}。非图片部分返回 None。

    不能改用 litellm 自带的 extract_images_from_message：它只认 image_url，
    而且会剥掉 data:...;base64, 前缀（那是给 Ollama 用的），剥掉后无法回传给视觉模型。
    """
    content_type = content_item.get("type")

    if content_type == "image_url":
        image_url = content_item.get("image_url")
        if isinstance(image_url, str):
            return image_url or None
        if isinstance(image_url, dict):
            url = image_url.get("url")
            return url if isinstance(url, str) and url else None
        return None

    if content_type == "image":
        source = content_item.get("source")
        if not isinstance(source, dict):
            return None
        url = source.get("url")
        if isinstance(url, str) and url:
            return url
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        if data.startswith("data:"):
            return data
        media_type = source.get("media_type")
        resolved_media_type = media_type if isinstance(media_type, str) and media_type else "image/png"
        return f"data:{resolved_media_type};base64,{data}"

    return None


def _walk_images(content: object, prefix: Tuple[int, ...]) -> Iterator[Tuple[Tuple[int, ...], str]]:
    """
    深度遍历 content，产出 (下标路径, 图片 URL)。

    必须下钻而不能只扫顶层：Anthropic 的 tool_result 把图片嵌在自己的 content 数组里，
    Claude Code 用 Read 读图片文件走的正是这个形状，只扫顶层会把它整个漏给纯文本模型。
    下标按原始列表计数，因此 content 里混有非 dict 元素时也不会错位。
    """
    if not isinstance(content, list):
        return
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        path = prefix + (index,)
        image_url = image_url_for_vision_call(part)
        if image_url is not None:
            yield path, image_url
            continue
        yield from _walk_images(part.get("content"), path)


def extract_image_references(messages: Sequence[AllMessageValues]) -> Tuple[ImageReference, ...]:
    """找出 messages 里所有图片 content part（含嵌在 tool_result 里的），保留其位置"""
    return tuple(
        ImageReference(message_index=message_index, path=path, image_url=image_url)
        for message_index, message in enumerate(messages)
        for path, image_url in _walk_images(message.get("content"), ())
    )


def _replace_part(
    part: object,
    path: Tuple[int, ...],
    replacements: Mapping[Tuple[int, ...], str],
) -> object:
    if path in replacements:
        return {"type": "text", "text": replacements[path]}
    if not isinstance(part, dict) or not isinstance(part.get("content"), list):
        return part
    return {**part, "content": _replace_in_content(part["content"], path, replacements)}


def _replace_in_content(
    content: Sequence[object],
    prefix: Tuple[int, ...],
    replacements: Mapping[Tuple[int, ...], str],
) -> List[object]:
    return [_replace_part(part, prefix + (index,), replacements) for index, part in enumerate(content)]


def _replace_message_images(
    message: AllMessageValues,
    replacements: Mapping[Tuple[int, ...], str],
) -> AllMessageValues:
    content = message.get("content")
    if not replacements or not isinstance(content, list):
        return message
    return cast(AllMessageValues, {**message, "content": _replace_in_content(content, (), replacements)})


def replace_images_with_text(
    messages: Sequence[AllMessageValues],
    replacements: Mapping[Tuple[int, Tuple[int, ...]], str],
) -> List[AllMessageValues]:
    """
    返回新的 messages，把指定位置的图片 part 换成 text part。

    {"type": "text", "text": ...} 在 OpenAI 与 Anthropic 两种形状里都合法，
    tool_result 的 content 也接受 text 块，所以这里不需要按形状分支。原 messages 不被修改。
    """
    if not replacements:
        return list(messages)
    return [
        _replace_message_images(
            message,
            {path: text for (index, path), text in replacements.items() if index == message_index},
        )
        for message_index, message in enumerate(messages)
    ]


def _survives_empty_text_stripping(block: object) -> bool:
    """
    litellm 发给后端前会摘掉纯空白 text 块（strip_empty_text_blocks_from_anthropic_messages），
    所以"是否裸 tool_result 收尾"必须按摘完之后的样子判断。否则会被 Claude Code 常带的
    {"type": "text", "text": ""} 骗过去：看着有 text 收尾就跳过注入，实际发出去时它已被摘掉，
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


def _last_surviving_message_index(messages: Sequence[AllMessageValues]) -> Optional[int]:
    return next(
        (index for index in reversed(range(len(messages))) if not _dropped_by_empty_text_stripping(messages[index])),
        None,
    )


def ensure_trailing_text_after_tool_result(messages: List[AllMessageValues]) -> List[AllMessageValues]:
    """
    最后一条消息以裸 tool_result 收尾时，在其 content 末尾追加一个非空 text 块。

    opencode 对 tool_use id 做服务端注册表校验，只认自己签发过的 id；而 Claude Code 不自己
    生成 id，只回传后端给的。于是一次 fallback 到别的后端就永久污染该会话，之后每轮必 400
    （单向棘轮）。改写 id 无解：同前缀同长度的自编 id 也被拒（真 id 只改末 4 字符即 400），
    是存在性校验而非格式校验。

    但校验是两段式的：只有请求最后一个 content 块是裸 tool_result 时才**触发**，一旦触发就
    扫全历史的 id。所以末尾追加一个非空 text 块，校验根本不启动，历史里有多少外来 id 都无所谓。
    对合法 id 注入同样无害，因此无条件执行，不必判断 id 来源。

    OpenAI 形状（末条 role: "tool"）不处理：那种形状只能另起一条 user 消息，会造成连续两条
    user 消息，未实测过。生产走 /v1/messages，进本 hook 时是 Anthropic 形状。

    "最后一条消息"按 litellm 摘空块之后的样子算：整条会被摘空的消息不算，否则它会挡住
    前面那条真正的裸 tool_result 收尾（见 _dropped_by_empty_text_stripping）。
    那些消息由 litellm 自己丢，本函数不删。

    无需改动时返回传入的对象本身，调用方可用 `is` 判断有没有变。原 messages 不被修改。
    """
    index = _last_surviving_message_index(messages)
    if index is None:
        return messages
    last_message = messages[index]
    content = last_message.get("content")
    if not isinstance(content, list) or not _ends_with_bare_tool_result(content):
        return messages
    trailing = {"type": "text", "text": TOOL_RESULT_TRAILING_TEXT}
    patched = cast(AllMessageValues, {**last_message, "content": [*content, trailing]})
    return [*messages[:index], patched, *messages[index + 1 :]]


class VisionToTextGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        event_hook: Optional[str] = None,
        default_on: bool = False,
        vision_model: Optional[str] = None,
        vision_prompt: Optional[str] = None,
        description_template: Optional[str] = None,
        failure_template: Optional[str] = None,
        max_images: Optional[int] = None,
        vision_timeout: Optional[float] = None,
        vision_num_retries: int = DEFAULT_VISION_NUM_RETRIES,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        llm_router: Optional["Router"] = None,
        **kwargs: Any,
    ) -> None:
        if not vision_model:
            raise ValueError("vision_to_text: 'vision_model' is required in litellm_params")
        self.vision_model = vision_model
        self.vision_prompt = vision_prompt or DEFAULT_VISION_PROMPT
        self.description_template = description_template or DEFAULT_DESCRIPTION_TEMPLATE
        self.failure_template = failure_template or DEFAULT_FAILURE_TEMPLATE
        self.max_images = max_images
        self.vision_timeout = vision_timeout
        # 组内重试可绕开负载均衡组里不支持图片的成员；跨组 fallback 一律禁掉（见 _call_vision_model）
        self.vision_num_retries = vision_num_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        self._llm_router = llm_router
        # 外部 guardrail 不会被注入 llm_router，其余 litellm_params 由 **kwargs 吸收
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

        # 两个修复彼此独立，各自判断：识图换掉图片块，尾块只在末尾追加，互不干扰。
        # 尾块修复不能挂在图片分支下 —— 生产上撞 tool id 校验的请求大多不带图
        patched = ensure_trailing_text_after_tool_result(await self._replace_images_in(messages, cache=cache))
        if patched is messages:
            return None
        return {**data, "messages": patched}

    async def _replace_images_in(
        self,
        messages: List[AllMessageValues],
        cache: "DualCache",
    ) -> List[AllMessageValues]:
        """把图片块换成识别文本；无图时原样返回传入对象，供调用方判断有没有变"""
        references = extract_image_references(messages)
        if not references:
            return messages

        # 超出上限的图片保持原样，避免一次请求打爆视觉模型配额
        selected = references if self.max_images is None else references[: self.max_images]
        if len(selected) < len(references):
            verbose_proxy_logger.warning(
                "vision_to_text: %d image(s) exceed max_images=%s and were left untouched",
                len(references) - len(selected),
                self.max_images,
            )

        replacements = await self._describe_images(selected, cache=cache)
        return replace_images_with_text(messages, replacements)

    async def _describe_images(
        self,
        references: Sequence[ImageReference],
        cache: "DualCache",
    ) -> Dict[Tuple[int, Tuple[int, ...]], str]:
        # 同一请求内重复出现的图片只识别一次
        unique_urls = tuple(dict.fromkeys(reference.image_url for reference in references))
        texts = await asyncio.gather(*(self._text_for_image(url, cache=cache) for url in unique_urls))
        by_url = dict(zip(unique_urls, texts))
        return {(reference.message_index, reference.path): by_url[reference.image_url] for reference in references}

    async def _text_for_image(self, image_url: str, cache: "DualCache") -> str:
        cache_key = self._cache_key(image_url)
        cached = await cache.async_get_cache(key=cache_key)
        # 命中缓存保证同图产出逐字节相同的文本，否则每轮改写都会打断 prompt caching
        if isinstance(cached, str) and cached:
            return cached

        description = await self._call_vision_model(image_url)
        if description is None:
            # 失败结果不入缓存，下次请求会重试
            return self.failure_template.format(error="vision model returned no description")

        text = self.description_template.format(description=description)
        await cache.async_set_cache(key=cache_key, value=text, ttl=self.cache_ttl_seconds)
        return text

    def _cache_key(self, image_url: str) -> str:
        fingerprint = hashlib.sha256(
            "\x00".join((self.vision_model, self.vision_prompt, self.description_template, image_url)).encode()
        ).hexdigest()
        return f"vision_to_text:{fingerprint}"

    def _router(self) -> Optional["Router"]:
        if self._llm_router is not None:
            return self._llm_router
        # 外部 guardrail 拿不到注入，延迟取 proxy 的 router 以复用 model_list 里的凭据
        try:
            from litellm.proxy.proxy_server import llm_router

            return llm_router
        except Exception:  # pragma: no cover - proxy 未启动时（如 SDK 场景）
            return None

    async def _call_vision_model(self, image_url: str) -> Optional[str]:
        messages: List[AllMessageValues] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vision_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        router = self._router()
        try:
            # 不传 guardrails，deployment 级 hook 会早退；SDK 层调用本就不过 proxy 的 pre_call_hook
            if router is not None:
                # 必须禁掉 fallback：识图失败若降级到纯文本模型，那个模型看不见图却会照样
                # 编一段"描述"，被当成真结果写进 messages。宁可 fail-open 插占位文本，
                # 也不能把幻觉描述喂给下游
                response = await router.acompletion(
                    model=self.vision_model,
                    messages=messages,
                    stream=False,
                    timeout=self.vision_timeout,
                    num_retries=self.vision_num_retries,
                    fallbacks=[],
                )
            else:
                response = await litellm.acompletion(
                    model=self.vision_model,
                    messages=messages,
                    stream=False,
                    timeout=self.vision_timeout,
                )
        except Exception as exception:
            # fail-open：识别失败也让请求继续，但要让模型知道这里原本有图
            verbose_proxy_logger.warning("vision_to_text: vision model call failed: %s", exception)
            return None
        return first_text(response)


def first_text(response: object) -> Optional[str]:
    if not isinstance(response, ModelResponse) or not response.choices:
        return None
    choice = response.choices[0]
    if not isinstance(choice, Choices):
        return None
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        return None
    return content.strip()
