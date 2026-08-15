"""
在请求发给后端之前，把图片交给视觉模型识别，用识别文本替换原图片块。

deepseek 等模型只支持文本，但 Claude Code 用户会贴截图；挂上本 guardrail 后带图请求
不再报 400，也不会被后端静默丢成 [Unsupported Image]。

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
          max_images: 4                      # 只识别最近 N 张，默认 4

调用放大防护（2026-08-14 生产事故后加固，视觉模型曾被打到 618 RPM）：只识别最近
max_images 张图、自带独立缓存、同图并发去重、并发上限。四条机制的来由见
README.md 的"调用放大防护"一节，改动前务必先读 —— 其中两条反直觉。
"""

import asyncio
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.router import Router
    from litellm.types.utils import CallTypesLiteral

DEFAULT_VISION_PROMPT = (
    "Describe this image in detail. Transcribe any text, code, numbers or error "
    "messages exactly as they appear. Be factual and do not speculate."
)
DEFAULT_DESCRIPTION_TEMPLATE = "[Image description: {description}]"
DEFAULT_FAILURE_TEMPLATE = "[Image could not be processed: {error}]"
DEFAULT_HISTORY_TEMPLATE = "[Image omitted: only the most recent {max_images} image(s) are described]"
DEFAULT_CACHE_TTL_SECONDS = 3600
# 视觉模型常指向负载均衡组，组内成员可能限流或临时不可用。组内重试能换到别的成员，
# 所以默认给 2 次；跨组 fallback 另行禁掉（见 _call_vision_model）
DEFAULT_VISION_NUM_RETRIES = 2
# 每请求最多识别几张图，超出的历史图片换成占位文本。
#
# 这只是防"单次请求塞进几百张图"的兜底闸门，不是控成本的主手段：缓存修好之后，
# 一个会话的识图次数只等于唯一图片数（与轮数无关），调这个值几乎不改变成本，
# 只决定模型能看见最近几张。所以取值要偏大 —— 正常人工会话很难贴超过 32 张不同截图，
# 对真实使用等于不限，同时对异常请求仍是硬上限。
#
# 别为了省配额把它调小：设成 4 时，每轮贴 1 张图的对话从第 5 轮起最早的图就退化成
# 占位文本，用户回头问"第一张截图里那个报错"时模型已经看不见了。事故的成因是缓存
# 命中率归零导致每张图每轮重识别一次（实测每张重复 157 次），不是张数多
DEFAULT_MAX_IMAGES = 32
# 同时在飞的识图调用上限。asyncio.gather 不限并发，一次请求 N 张图就是 N 个并发请求，
# 会瞬间打穿视觉模型的 RPM 配额并触发限流雪崩
DEFAULT_MAX_CONCURRENCY = 4
# 识图缓存独占的容量。不能共用 proxy 传进来的 DualCache：那个 cache 是 user_api_key_cache，
# 内存层硬上限 200 条且与 key/team/user 认证条目混住。识图键 TTL 3600 远长于认证键的 60，
# evict_cache 按到期时间驱逐，于是识图键既会挤掉认证条目、又会在图片数超 200 时
# 自我抖动到 0% 命中（两者均已用可执行脚本复现）
DEFAULT_CACHE_MAX_ENTRIES = 2048


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
        history_template: Optional[str] = None,
        max_images: Optional[int] = DEFAULT_MAX_IMAGES,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        vision_timeout: Optional[float] = None,
        vision_num_retries: int = DEFAULT_VISION_NUM_RETRIES,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        cache_max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
        llm_router: Optional["Router"] = None,
        **kwargs: Any,
    ) -> None:
        if not vision_model:
            raise ValueError("vision_to_text: 'vision_model' is required in litellm_params")
        self.vision_model = vision_model
        self.vision_prompt = vision_prompt or DEFAULT_VISION_PROMPT
        self.description_template = description_template or DEFAULT_DESCRIPTION_TEMPLATE
        self.failure_template = failure_template or DEFAULT_FAILURE_TEMPLATE
        self.history_template = history_template or DEFAULT_HISTORY_TEMPLATE
        self.max_images = max_images
        self.vision_timeout = vision_timeout
        # 组内重试可绕开负载均衡组里不支持图片的成员；跨组 fallback 一律禁掉（见 _call_vision_model）
        self.vision_num_retries = vision_num_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        self._llm_router = llm_router
        # 识图缓存独立于 proxy 的 user_api_key_cache，理由见 DEFAULT_CACHE_MAX_ENTRIES
        self._cache = InMemoryCache(max_size_in_memory=cache_max_entries, default_ttl=cache_ttl_seconds)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # 同一张图的并发识图只发一次请求，其余等这一次的结果（in-flight 去重）
        self._in_flight: Dict[str, "asyncio.Future[Optional[str]]"] = {}
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
        # cache 是 litellm 的钩子签名要求，但识图刻意不用它：它是 proxy 的
        # user_api_key_cache，内存层只有 200 条且与认证条目混住（见 DEFAULT_CACHE_MAX_ENTRIES）
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            return None
        messages = cast(List[AllMessageValues], raw_messages)

        patched = await self._replace_images_in(messages)
        if patched is messages:
            return None
        return {**data, "messages": patched}

    async def _replace_images_in(
        self,
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        """把图片块换成识别文本；无图时原样返回传入对象，供调用方判断有没有变"""
        references = extract_image_references(messages)
        if not references:
            return messages

        # 取最近 N 张而非最早 N 张：用户当轮刚贴的截图排在最后，正是模型要看的那些。
        # 超限的历史图片换成占位文本而不原样留下；留下等于把原图透传给纯文本后端，
        # 那正是这个 guardrail 要消灭的 400
        selected = references if self.max_images is None else references[-self.max_images :]
        skipped = references[: len(references) - len(selected)]
        if skipped:
            verbose_proxy_logger.info(
                "vision_to_text: describing the %d most recent image(s), %d older image(s) replaced with a placeholder "
                "(max_images=%s)",
                len(selected),
                len(skipped),
                self.max_images,
            )

        described = await self._describe_images(selected)
        placeholder = self.history_template.format(max_images=self.max_images)
        omitted = {(reference.message_index, reference.path): placeholder for reference in skipped}
        return replace_images_with_text(messages, {**omitted, **described})

    async def _describe_images(
        self,
        references: Sequence[ImageReference],
    ) -> Dict[Tuple[int, Tuple[int, ...]], str]:
        # 同一请求内重复出现的图片只识别一次
        unique_urls = tuple(dict.fromkeys(reference.image_url for reference in references))
        texts = await asyncio.gather(*(self._text_for_image(url) for url in unique_urls))
        by_url = dict(zip(unique_urls, texts))
        return {(reference.message_index, reference.path): by_url[reference.image_url] for reference in references}

    async def _text_for_image(self, image_url: str) -> str:
        cache_key = self._cache_key(image_url)
        cached = self._cache.get_cache(key=cache_key)
        # 命中缓存保证同图产出逐字节相同的文本，否则每轮改写都会打断 prompt caching
        if isinstance(cached, str) and cached:
            return cached

        description = await self._describe_once(cache_key, image_url)
        if description is None:
            # 失败结果不入缓存，下次请求会重试
            return self.failure_template.format(error="vision model returned no description")

        text = self.description_template.format(description=description)
        self._cache.set_cache(key=cache_key, value=text, ttl=self.cache_ttl_seconds)
        return text

    async def _describe_once(self, cache_key: str, image_url: str) -> Optional[str]:
        """
        同一张图并发到达时只真打一次视觉模型，其余请求等同一个 future。

        没有这层去重时，同一张图的 N 个并发请求就是 N 次识图调用：缓存只在第一次
        返回后才写入，在那之前所有请求都是 miss。长会话逐轮重发同批图片时，
        这个倍数直接乘在放大链条上。
        """
        in_flight = self._in_flight.get(cache_key)
        if in_flight is not None:
            # shield 让本请求被取消时不会连带取消正在跑的那次识图（别人还在等它）
            return await asyncio.shield(in_flight)

        future: "asyncio.Future[Optional[str]]" = asyncio.get_running_loop().create_future()
        self._in_flight[cache_key] = future
        try:
            async with self._semaphore:
                description = await self._call_vision_model(image_url)
        except BaseException:
            # 领头请求被取消（客户端断连）不能连累等待者：它们各自的请求还活着。
            # 给它们 None 走 fail-open 占位文本，而不是把 CancelledError 抛进去
            future.set_result(None)
            raise
        else:
            future.set_result(description)
            return description
        finally:
            self._in_flight.pop(cache_key, None)

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
