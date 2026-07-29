# 本地自定义 guardrail

放在这里的是**不修改 litellm 源码**的外部 guardrail。litellm 原生支持从 config.yaml
同目录加载自定义 guardrail（`litellm/proxy/types_utils/utils.py` 的 `get_instance_fn`），
所以升级 / 合并上游时这里零冲突。

跑测试：

```bash
.venv/bin/python -m pytest local_guardrails/ -q
```

## vision_to_text — 图片转文本

让只支持文本的模型（deepseek）能处理带截图的请求：在请求发给后端之前，把 messages 里的
图片交给视觉模型识别，用识别文本替换原图片块。

### 为什么需要

deepseek 收到图片时的两种表现（均已实测）：

| 入口 | 结果 |
|---|---|
| `/v1/chat/completions`（走 newapi） | 400 |
| `/v1/messages`（走 newapi） | 400 |
| `/v1/messages`（直连官方 `api.deepseek.com/anthropic`） | **HTTP 200 但静默降级** |

最后一种最危险：DeepSeek 自己把图换成 `[Unsupported Image]`，模型 thinking 里明说
"无法看到图片内容"，却照样作答、照样计费。用户贴了截图，得到一个看起来合理的错答案。

litellm 与其它网关都没有这个能力：`supports_vision` 全仓只是元数据，唯一按它决策的地方是
fireworks 直接抛 BadRequestError。Portkey 把媒体预处理列为使用方自己的负担，OpenRouter
只能按 modality 过滤模型列表，new-api 也没有。

### 配置

把 `vision_to_text.py` 放在 config.yaml 同目录（或用 `local_guardrails.vision_to_text.VisionToTextGuardrail`
这样的模块路径），然后：

```yaml
model_list:
  - model_name: deepseek-v4-flash
    litellm_params:
      model: openai/deepseek-v4-flash
      api_base: os.environ/TEXT_MODEL_API_BASE
      api_key: os.environ/TEXT_MODEL_API_KEY
      guardrails: ["vision-to-text"]     # 只有挂了的模型才生效

  - model_name: vision-describer            # 识图侧，换成任意支持视觉的模型
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "vision-to-text"
    litellm_params:
      guardrail: vision_to_text.VisionToTextGuardrail
      mode: "pre_call"
      default_on: false                  # 必须 false
      vision_model: vision-describer
```

可选参数：`vision_prompt`、`description_template`、`failure_template`、`max_images`、
`vision_timeout`、`vision_num_retries`（默认 2）、`cache_ttl_seconds`。

`vision_model` 可以直接指向一个负载均衡组（组里成员支持视觉即可），不必是单个部署。

### 只对指定模型生效

guardrail 写在 deployment 的 `litellm_params.guardrails` 上，由
`_check_and_merge_model_level_guardrails`（`litellm/proxy/utils.py`）合入 metadata。
所以只有挂了的模型会被改写，多模态模型完全不受干预，客户端无需改动。

`default_on` 必须保持 `false`：设成 true 会走 `should_run_guardrail` 的全局分支，
变成对所有模型常开，正好是不想要的。

### 设计要点

**挂载点**：`async_pre_call_hook` + `mode: pre_call`。只有 pre_call 的返回值会真正替换
发给后端的 data（`common_request_processing.py` 里 `self.data = await proxy_logging_obj.pre_call_hook(...)`，
随后才进 `route_request`）。`during_call` 与 LLM 调用并行、改 data 有竞态；`post_call` 太晚；
`get_chat_completion_prompt` 被 `prompt_id` 门控挡死且 `/v1/messages` 完全不走。

**形状分叉**（最容易踩）：`/v1/messages` 在 hook 处拿到的仍是 Anthropic 原生
`{"type":"image","source":{...}}`，不是 `image_url`。而 litellm 自带的
`extract_images_from_message` 只认 `image_url`，还会剥掉 `data:...;base64,` 前缀（给 Ollama 用的）。
直接复用会静默漏掉 Claude Code 的全部图片，所以本文件自带覆盖两种形状、保留完整 data URL 的提取逻辑。

**失败处理**：fail-open。识图失败或返回空时替换成 `[Image could not be processed: ...]`，
主请求继续、绝不中断，模型也知道此处原本有图。失败结果不入缓存，下次请求会重试。

**识图调用禁 fallback**（`fallbacks=[]`）：视觉模型组的降级目标往往是纯文本模型，而那个模型
看不见图却会煞有介事地编一段"描述"，会被当成真结果写进 messages。这比报错更糟——报错会走占位符，
你知道图没读到；幻觉描述则看起来完全正常。组内重试（`num_retries`，默认 2）保留，它只在同组内
换部署，能绕开限流或临时不可用的成员。

**prompt caching**：改写前缀会破坏缓存命中，故按 (视觉模型, prompt, 模板, 图片 URL) 的
sha256 缓存描述文本，保证同图产出逐字节相同。

**防重入**：视觉调用不传 `guardrails`，deployment 级 hook 会在首行早退；SDK 层调用本就不经过
`ProxyLogging.pre_call_hook`。

### 实测验证

本地 proxy + 真实 newapi 后端，同一带图请求：

- 不挂 guardrail 的对照模型：仍然 400
- 挂上后 `/v1/chat/completions`：200，DeepSeek 回答"我现在看到的是替代文字，内容是：[Image could not be processed: ...]"
- 挂上后 `/v1/messages`：guardrail 同样生效，后端答"我看到的是替代文字"

`example_config.yaml` 就是这次实测用的配置。

### 待补 / 已知问题

- 端到端识图**准确性**尚未用真实视觉模型验证。上面的实测走的是 fail-open 路径，
  证明的是改写机制通、两个端点都生效、图片没漏给后端。生产上把 `vision_model` 指向
  已有的、成员支持视觉的模型组即可，但该组只存在于生产，本地跑不到（凭据在生产 DB 里加密），
  所以这一步要在生产验证
- 视觉调用未设 `max_tokens`。若组内命中推理模型，描述那次调用可能在 reasoning 上多花 token，
  真实用量出来后可按需加上限
- 与本 guardrail 无关的既有 bug：`/v1/messages` 打 OpenAI 形状后端时 litellm 解析响应报
  `APIError: OpenAIException`（后端其实正常返回了）。纯文本、不挂 guardrail 也复现，属于独立问题
