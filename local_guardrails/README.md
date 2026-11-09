# 本地自定义 guardrail

放在这里的是**不修改 litellm 源码**的外部 guardrail。litellm 原生支持从 config.yaml
同目录加载自定义 guardrail（`litellm/proxy/types_utils/utils.py` 的 `get_instance_fn`），
所以升级 / 合并上游时这里零冲突。

跑测试：

```bash
.venv/bin/python -m pytest local_guardrails/ -q
```

## local_content_policy — 中文内容策略

这两个 guardrail 都是纯本地 `pre_call` 检测，不调用 LLM 或外部 API：

| LiteLLM 注册名 | 内部用途 | 默认状态 |
|---|---|---|
| `zh-abusive-language-filter` | 中文辱骂与人身攻击 | 开启 |
| `zh-financial-trading-filter` | 可执行金融交易决策与自动交易开发 | 开启 |

注册名直接表达管理员侧用途，便于配置、日志和运维识别；公开拦截响应仍由自定义异常统一脱敏，
不会带出注册名。实现没有继承
`ContentFilterGuardrail`：当前 LiteLLM 的原生实现会把 category、keyword、pattern、severity
等字段放进 `HTTPException.detail`，代理层又会把 dict detail 原样带进下游响应；其中文 keyword
还使用 `\b`、conditional 只按英文标点切句，无法可靠处理连续中文。独立实现只依赖稳定的
`CustomGuardrail.apply_guardrail` 接口，升级风险更低，也不需要修改 LiteLLM 核心源码。

### 判定边界

- 每次优先查看原始请求末尾消息；只有它是 `role=user` 时，才扫描其中最后一个文本块。忽略更早
  的用户历史以及 system、assistant、tool 和 tool result；请求末尾不是 user 时不回看历史。
  没有原始 messages 的入口按同样边界读取 `structured_messages`，再退回 `texts` 最后一项。
- Claude Code 在 HTTP 400 后可能把下一次输入继续追加到同一个 `user.content` 数组；因此不能把
  数组内全部文本块合并检测，否则用户随后输入“你好”也会被前一块的旧辱骂反复拦截。只取最后
  一个文本块既符合“只审核最新用户输入”，也把每次检测开销固定在本次新增文本规模。
- 文本先做 Unicode NFKC、`casefold()`，并删除所有 Unicode `Cf` 格式字符；不全局删除空格
  或标点。
- 中文词不使用 `\b`。ASCII 短码使用包含下划线的词边界，避免把 `nmsl_helper` 当成辱骂。
- 简单绕写只允许最多 3 个白名单分隔符，不使用无界 `.*` 或无界 separator。
- Precision 优先于 Recall。keyword 只对完整短句生效；辱骂 regex 主要匹配整句粗口、明确指向
  人的攻击或明确代发命令。另对不超过 2000 字的当前消息识别有限的高置信组合辱骂片段，
  并对翻译、引用、测试语料、医学和残障等明确正常语境让行。
- 金融规则只拦截完整短句的高置信执行短语、带明确请求语气的交易决策/开发 regex，以及同一条
  当前消息中同时出现强 `.mq4`、`.mq5`、`.mqh` 文件特征、明确修改指令和交易执行/业绩目标的
  三锚点请求。不会因为“金融对象 + 动作词”简单共现就拦截。
- `EA`、订单、日志、回测、突破、自动下单等歧义词不会单独触发；教育解释、财报摘要、
  新闻论文、反诈否定、接口文档、Electronic Arts、电商订单、基金会和黄金首饰等边界有
  明确回归样本。刻意拆成多轮或措辞含糊的请求可能漏过，这是为了避免影响日常交流而接受的取舍。

辱骂词候选参考了
[`houbb/sensitive-word-data`](https://github.com/houbb/sensitive-word-data) 固定提交
`fe6fc2921836217b8c90619db81b24af8b22d80f`。没有导入全量词典，也没有按上游 tags 自动抽取：
上游没有“辱骂”标签，标签不能表达本策略语义。当前完整短句词表中有 21 个上游 exact-match
候选，另有 3 个上游候选只在带方向的 anchored regex 中使用；其余缩写、变体与上下文规则为
本地人工补充。这只是经修改的词汇子集，没有复用上游 DFA，也不引用上游性能数字。第三方归属
与 Apache-2.0 许可见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

### 错误与审计日志

命中后抛出新的 HTTP 400，公开 detail 永远只有：

```text
Request rejected by content policy.
```

异常不携带 category、keyword、matched phrase、regex、severity、description、文件名、内部
rule ID 或具体检测逻辑。代理现有 OpenAI-compatible 错误封装保持不变，不伪装 HTTP 200，
也不返回 assistant 自然语言回复。

管理员日志保留单行 JSON：`guardrail_internal_type`、`rule_id`、`category`、
`matched_keyword`、`matched_pattern`、`severity`、`request_id`。regex 日志记录稳定规则名，
`matched_keyword` 留空；keyword 和 conditional 只记录规范化命中词，不复制完整 prompt 或原始
regex 源码。

### 生产配置

把三个运行文件只读挂载到容器：

```yaml
volumes:
  - ./local_content_policy.py:/app/local_content_policy.py:ro
  - ./content_policy_01.yaml:/app/content_policy_01.yaml:ro
  - ./content_policy_02.yaml:/app/content_policy_02.yaml:ro
```

然后注册两个全局 guardrail：

```yaml
guardrails:
  - guardrail_name: "zh-abusive-language-filter"
    litellm_params:
      guardrail: local_content_policy.LocalContentPolicyGuardrail
      mode: "pre_call"
      default_on: true
      policy_file: /app/content_policy_01.yaml

  - guardrail_name: "zh-financial-trading-filter"
    litellm_params:
      guardrail: local_content_policy.LocalContentPolicyGuardrail
      mode: "pre_call"
      default_on: true
      policy_file: /app/content_policy_02.yaml
```

两项互相独立。临时停用某一项时，只把对应项的 `default_on` 改为 `false` 并重建 LiteLLM
容器；恢复时改回 `true`。管理员为 key/team 配置的 `disable_global_guardrails` 或
`opted_out_global_guardrails` 仍会按 LiteLLM 原有语义生效。

### 验证与性能

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest local_guardrails/test_local_content_policy.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python local_guardrails/benchmark_local_content_policy.py
```

测试覆盖 BLOCK/ALLOW、keyword/regex/conditional/always-block、Unicode 绕写、有限 gap、当前
用户消息选择、角色过滤、公开错误脱敏、内部日志和长输入。benchmark 分别输出关闭、只开 01、
只开 02、同时开启时的短/长输入 P50/P95/P99、顺序吞吐、并发吞吐和进程 CPU。它是 matcher
微基准，不包含网络、模型推理或完整代理开销。

## vision_to_text — 图片转文本

让只支持文本的模型（deepseek）能处理带截图的请求：在请求发给后端之前，把 messages 里的
图片交给视觉模型识别，用识别文本替换原图片块。

### 为什么需要

deepseek 收到图片时的两种表现（均已实测）：

| 入口 | 结果 |
|---|---|
| `/v1/chat/completions`（经网关） | 400 |
| `/v1/messages`（经网关） | 400 |
| `/v1/messages`（直连纯文本模型的原生 anthropic 端点） | **HTTP 200 但静默降级** |

最后一种最危险：DeepSeek 自己把图换成 `[Unsupported Image]`，模型 thinking 里明说
"无法看到图片内容"，却照样作答、照样计费。用户贴了截图，得到一个看起来合理的错答案。

litellm 与其它网关都没有这个能力：`supports_vision` 全仓只是元数据，唯一按它决策的地方是
fireworks 直接抛 BadRequestError。Portkey 把媒体预处理列为使用方自己的负担，OpenRouter
只能按 modality 过滤模型列表，其它自建网关也没有。

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

可选参数：`vision_prompt`、`description_template`、`failure_template`、`history_template`、
`max_images`（默认 32）、`max_concurrency`（默认 4）、`vision_timeout`、
`vision_num_retries`（默认 2）、`cache_ttl_seconds`（默认 3600）、
`cache_max_entries`（默认 2048）。

`vision_model` 可以直接指向一个负载均衡组（组里成员支持视觉即可），不必是单个部署。

### 调用放大防护（2026-08-14 生产事故后加固）

视觉模型曾在数小时内被调用约 5.5 万次、峰值 618 RPM，起因是一个持续累积的超长会话。
四条独立的放大路径，每条都有对应回归测试：

| 机制 | 事故时行为 | 现在 |
|---|---|---|
| 历史图片重复识别 | `max_images` 默认 `None`，单次请求无上界 | 默认识别**最近 32 张**，更早的换成占位文本 |
| 缓存容量 | 复用 proxy 传入的 `DualCache`（即 `user_api_key_cache`），内存层硬上限 200 条 | 自带 `InMemoryCache`，默认 2048 条，与 proxy 缓存完全隔离 |
| 并发去重 | 同一张图的 N 个并发请求 = N 次识图调用（缓存要等第一次返回才写入） | in-flight 去重，同图并发只真打一次 |
| 并发上限 | `asyncio.gather` 不限并发，一次请求 N 张图就是 N 个同时在飞的调用 | semaphore 限 4 |

两个反直觉的点，改前请先读：

- **缓存不能共用传入的 `cache` 参数**。它是 `user_api_key_cache`，内存层 200 条上限，
  且与 key / team / user 认证条目混住。识图键 TTL 3600 远长于认证键的 60，而
  `InMemoryCache.evict_cache` 按到期时间驱逐 —— 于是识图流量会把认证条目挤干（实测
  50 个认证键在 300 个识图键写入后**全部**被驱逐，迫使每请求回 DB 鉴权），
  同时图片数一旦超 200，识图缓存自身命中率**不是下降而是归零**（250 张图 x 40 轮 =
  10400 次调用，命中率 0.0%）。这两件事都能用可执行脚本复现。
- **`max_images` 不是控成本的手段，别为省配额调小**。缓存修好之后，一个会话的识图次数
  只等于**唯一图片数**（与轮数、与这个值都无关），调它几乎不改变成本，只决定模型能看见
  最近几张。曾经设成 4，结果每轮贴 1 张图的对话从第 5 轮起最早的图就退化成占位文本，
  用户回头问"第一张截图里那个报错"时模型已经看不见了。它的定位是防"单次请求塞进几百张图"
  的兜底闸门，取值要偏大（现 32，正常人工会话贴不到这个数，等于不限）。
- **`max_images` 必须取最近 N 张，不是最早 N 张**。用户当轮刚贴的截图排在 messages 最后；
  取最早 N 张会把它漏掉，识图对当前提问完全无效。事故版实现取的正是最早 N 张。
  另外超限的图片必须换成占位文本而不是原样留下 —— 留下就等于把真图透传给纯文本后端，
  那正是这个 guardrail 要消灭的 400。

还有一条**未修的正反馈**：视觉模型被打限流后失败结果不入缓存（这是有意的，为了下次能重试），
于是每轮都对全部图片重试一遍，限流越严重重试越多。真正压住它规模的是 `max_concurrency`
（同时在飞的调用限 4）和 in-flight 去重，不是 `max_images` —— 后者已放宽到 32。
根治需要熔断器，当前判断是不值得为此引入状态机。

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

**嵌套下钻**（第二个坑，已实测踩到）：Claude Code 用 Read 工具读图片文件时，图片不在顶层
content 里，而嵌在 `tool_result` 自己的 `content` 数组内。只扫顶层会把这条主路径整个漏掉：
实测 deepseek 收到了原始 base64，thinking 里在试着自己解码。所以提取与替换都递归下钻，
且只替换里层的图片块、保留 `tool_result` 外壳 —— 换掉外壳会让前一条 assistant 的 `tool_use`
失去配对，Anthropic 直接拒掉整个请求。位置用下标路径（如 `(2, (0, 1))`）表示，多图混合时不会串位。

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

本地 proxy + 真实纯文本后端（两个纯文本模型）+ 真实视觉模型组，
识图**准确性**已端到端验证，不再只是机制验证。

识图正确性，问只有真看到图才能答对的细节：

| 图片 | 场景 | 结果 |
|---|---|---|
| 生成的数字图 | 2 模型 x 2 端点 x 流式/非流式，共 14 组合 | 14/14 答出正确数字 |
| 真实 UI 截图（126KB） | 顶层图与 `tool_result` 嵌套图，两端点，流式/非流式 | 4/4 场景，每次 3/3 细节全中 |

真实截图那组问的是错误弹窗里的错误码、团队名、下拉框取值三项，全部答对，
说明走的是真识图而不是猜。

安全性与不退化：

- 对照组（同后端不挂 guardrail）带图仍然 400，证明差异确实来自 guardrail
- 图片泄漏用假后端做确定性取证（抓 litellm 实际发出的字节，不靠读日志）：
  不挂 guardrail 时 base64 原样到达，挂上后 content 变成两个 text 块、**零 base64**
- 多图混合（顶层 1 张 + `tool_result` 内 2 张）：三段描述各自贴在正确位置，无串位
- fail-open：识图组整体不可达时 HTTP 200 主请求不中断，模型 reasoning 里明确说
  "image could not be processed, we have no information"，**没有编造**
- `fallbacks=[]` 有效：故意给识图组配上纯文本兜底目标，全程该目标调用计数一次未增长
- 回归：纯文本单轮 / content 块数组 / 带 system / 多轮 / 流式 / 工具调用 / 工具调用流式，
  两个端点各 7 项，挂与不挂 guardrail 表现一致，21/21 全通
- 全程 proxy 日志里的真实异常全部归因到故意制造的两个失败源（对照组带图 400、
  坏识图组连接拒绝），无法解释的错误为零

### 待补 / 已知问题

- 视觉调用未设 `max_tokens`。若组内命中推理模型，描述那次调用可能在 reasoning 上多花 token，
  真实用量出来后可按需加上限
- 超出 `max_images` 的图片保持原样，会原封不动送给纯文本后端并触发 400。
  默认不限张数，配了才有这个风险
- 纯文本后端配 `openai/` 前缀时，`/v1/messages` 会被路由到 Responses API，
  litellm 解析响应报 `APIError`（后端其实返回了正确内容）。与本 guardrail 无关，
  配成 `custom_openai` 或 `anthropic` 即正常，两者都已实测通过

- **`vision_model` 必须指向一个「每个成员都确认能识图」的组**（已踩，2026-07-30）。
  fail-open 只在调用**抛异常**时触发；若视觉模型返回 HTTP 200 加一句客气的拒绝
  （实测有后端回 "I cannot describe the image..."），那句拒绝会被当成合法描述
  套进 `description_template` 发给下游，**并按 `cache_ttl_seconds` 缓存**，
  整个 TTL 内不再重试，且不打任何 warning。下游模型于是如实回答"看不到图片"。
  `num_retries` 救不了 —— 重试只在异常时触发，200 不算失败。

  没有在代码里加"识别拒绝式回复"的判断：靠文本判断"这是描述还是客套"本质是猜，两个方向都会错
  （真实截图里完全可能出现 `cannot process image` 字样，本来就在转录报错弹窗；而拒绝的表达
  无穷多、还跨语言）。真正的约束在配置层 —— 别把图片交给成员能力不受控的负载均衡组。
  若确实需要代码层兜底，唯一不算猜的方向是要求模型返回
  `{"can_see_image": bool, "description": str}`，把判断从"猜语气"变成"它自己声明"。

- 判定某个模型能否识图，**不要用手搓的点阵数字图**：字形太糊会让能识图的模型读错数字，
  从而被误判成不支持视觉（生产选型时据此错杀了 qwen3.7-plus）。用真实截图，
  问只有真看到图才知道的细节。同理，`supports_vision` 为 `None` 只表示
  `model_prices_and_context_window.json` 里查无此条目，不代表不支持

## kiro_session_affinity — 会话亲和 header

给发往 kiro-gateway 的请求打一个跨轮次稳定的 `x-kiro-session`，让 kiro 侧 nginx 的
`hash $http_x_kiro_session consistent` 把同一个对话线程钉在同一个上游账号上。

### 为什么需要

kiro 上游按账号隔离前缀缓存，同一会话落到不同账号要重新 prefill，每轮多付 3~6 秒。
选路由 kiro 侧 nginx 完成，litellm 这边只负责把 key 稳定带上：本 guardrail 只写 header，
不选路、不改 messages、不阻断请求。

kiro 那个 model group 必须继续只有一个 deployment 指向 nginx，注册成多个会让两层路由打架。

### 配置

按文件挂载（不是挂目录），所以新增文件要改 compose：

```yaml
volumes:
  - ./kiro_session_affinity.py:/app/kiro_session_affinity.py:ro
```

```yaml
guardrails:
  - guardrail_name: "kiro-session-affinity"
    litellm_params:
      guardrail: kiro_session_affinity.KiroSessionAffinityGuardrail
      mode: "pre_call"
      default_on: false          # 必须 false，否则变成全局常开
```

再给每个 kiro deployment 的 `litellm_params.guardrails` 加上 `"kiro-session-affinity"`。

### 设计要点

**值 = `sha256(litellm_session_id)` 前 16 位 hex。** 不透出 session_id 原值：它源自
`metadata.user_id`，含用户身份信息，而这个 header 会进 nginx access log。

**刻意只掺 session_id，不掺 system prompt。** 掺 system 的动机是把共享同一 session id 的
并行 subagent 分到不同账号，但 Claude Code 的 system 是数组，前若干 KB 是所有会话共享的公共
前缀，按前缀截取大概率区分度为 0；整体哈希又会被 system 里逐轮变化的内容带得每轮漂移，
那比不做更糟（每轮冷 prefill）。稳定性优先于区分度。

**解不出会话 id 时不写 header**，而不是退化成随机值 —— 随机值每轮都是冷 prefill，
且缺失能在 nginx 侧的覆盖率里直接看出来。

**必须原地写 `data["headers"]`，不能只靠返回值。** deployment 级钩子
（`custom_guardrail.py` 的 `async_pre_call_deployment_hook`）只把返回值里的 `messages`
拷回 kwargs，其余键一概丢弃；而模型组 fallback 只走 router 内部、不重跑 proxy 级钩子，
deployment 级是 fallback 路径上唯一的机会。改成 `return {**data, ...}` 会让 header
在 fallback 上静默丢失。

**合并而非覆盖已有 headers。** 本钩子跑在 `add_litellm_data_to_request` 之后，直接赋值会
清掉它注入的 `anthropic-beta` / `anthropic-version`（见上文 DeepSeek Anthropic 兼容链路）。

**没走 `forward_client_headers_to_llm_api` 纯配置方案。** 那条会转发所有 `x-` 开头的头
（`litellm_pre_call_utils.py` 的 `_get_forwardable_headers` 只排除 `x-stainless`），
把调用方的 `x-api-key` 一起送上去；而 messages 路径的认证注入是
`if "x-api-key" not in headers`（`messages/transformation.py:244`），于是真 key 不注入，
上游拿到一个不认识的 key。

### 实测验证

本地 proxy 加假后端抓上游实收字节（`litellm_params.guardrails` 挂载 vs 未挂载对照）：

| 用例 | 上游实收 `x-kiro-session` |
|---|---|
| 挂载，session A 第 1 轮 | `609a4b6453120e27` |
| 挂载，session A 第 2 轮 | `609a4b6453120e27`（跨轮次一致） |
| 挂载，session B | `2af9371cca8b4329`（不同会话不同值） |
| 挂载，session id 走 header 来源 | 与 body 来源同值 |
| **未挂载，session A** | **缺失**（只对挂了的 deployment 生效） |
| 挂载，无 session id | 缺失（不退化成随机值） |
| **fallback 落到挂载组** | `609a4b6453120e27`（deployment 级钩子补上） |

最后一行是关键：fallback 路径上 proxy 级钩子不重跑，header 由 deployment 级钩子写入。

`anthropic-version` 与 `x-kiro-session` 在上游同时收到，即证明是合并不是覆盖。
用一个编造的 `anthropic-beta` 值测不出这条 —— 它会被 provider config 的 beta 过滤
（`should_filter_anthropic_beta_headers` 默认 True）当成不支持的 beta 丢掉，
未挂载的对照组同样缺失，与本 guardrail 无关。

### 覆盖面：只有能解出会话 id 的客户端才拿到 header

生产实测约 **81%**（2026-08-02，498 个请求）。分界完全按客户端格式走：

| 入口 | call_type | header |
|---|---|---|
| `/v1/messages` | `anthropic_messages` | 有 |
| `/chat/completions` | `acompletion` | **无** |

litellm 认会话 id 只有两条路（`proxy/litellm_pre_call_utils.py`）：header 路径
`get_chain_id_from_headers()`（:430），或 body 路径从 `metadata.user_id` 里的
`..._session_<uuid>` 抠（:456）。Claude Code 靠后者；OpenAI 格式的请求两条都不命中，
于是本 guardrail 静默跳过。

给这类客户端补上不用改代码，让它每轮发一个稳定的头即可，litellm 会自动认出来：

```
头名  匹配 ^x-.+-session-id$      （大小写不敏感，:47）
值    匹配 ^[a-zA-Z0-9_\-]{8,}$   （:51）
```

即 `x-openclaw-session-id: <该对话固定的 id>`，或显式的 `x-litellm-session-id`（优先级更高）。
硬要求仍是同一对话所有轮次完全一致，别掺时间戳或消息条数。

**验证覆盖率别看 `applied_guardrails`**，那个字段是无条件记录的（`proxy/utils.py:1226`），
静默 no-op 时同样会出现本 guardrail 的名字；`SpendLogs.session_id` 也会 fallback 到自动生成的
trace id（`litellm_logging.py:5110`），永不为空。真判据是同一 `session_id` 有没有跨轮复用。

### 挂给其他模型

只写一个自定义 header，不改 body。已确认 `headers` 不参与缓存 key
（`ModelParamHelper._get_all_llm_api_params()` 里没有 `headers` / `extra_headers`），
所以不会打散缓存。对不认识这个 header 的上游是多一个被忽略的头。

未实测：bedrock / vertex 这类要对请求头做 SigV4 之类签名的 provider，多一个头是否影响签名。
当前 6 个 kiro deployment 全是 anthropic provider，不涉及；将来挂到 bedrock 前先验这一条。

## thinking_switch — 思考开关归一化

把客户端的思考开关意图归一化成下游能区分的形状。修复 Claude Code 里关掉思考、上游
deepseek 仍然思考。

### 为什么需要

`deepseek-v4-flash` 在成本表里没有 `supports_adaptive_thinking`，于是原生 Anthropic 直通
路径上 `AnthropicMessagesConfig._translate_adaptive_effort_for_non_adaptive_model` 判它是
非 adaptive 模型，走降级分支。那个分支的闸门是：

```python
if effort is None and not adaptive_thinking:
    return
```

**`output_config.effort` 非空就足以进去**，随后无条件把 `thinking` 覆盖成
`{"type": "enabled", "budget_tokens": N}`（effort -> budget：low=1024 / medium=2048 /
high=4096 / xhigh=8192 / max=16384，再按 `max_tokens` 截断）。

而 Claude Code 关闭思考只是**不发 `thinking` 字段**，`output_config.effort` 照发 —— effort
是独立的强度档，关掉思考后仍停留在上次的值。于是两态出站字节级相同，下游无从区分。生产
实测（08-05 会话 `d7ad84b9`，两态 effort 都是 high，故出站都是 `enabled/4096`）：

| 客户端操作 | 实际发出 | litellm 出站 |
|---|---|---|
| 关闭思考 | 无 `thinking`，`effort: high` | `enabled/4096` |
| 开启思考 | `{"type":"adaptive"}`，`effort: high` | `enabled/4096` |

同一个闸门还导致：客户端**明确**发 `{"type":"disabled"}`，只要 effort 还在，也会被改写成
enabled。所以只补 disabled 是不够的，必须同时摘掉 effort。

### 配置

按文件挂载（不是挂目录），所以新增文件要改 compose：

```yaml
volumes:
  - ./thinking_switch.py:/app/thinking_switch.py:ro
```

```yaml
guardrails:
  - guardrail_name: "thinking-switch"
    litellm_params:
      guardrail: thinking_switch.ThinkingSwitchGuardrail
      mode: "pre_call"
      default_on: false          # 必须 false，否则变成全局常开
```

再给目标模型组每个 deployment 的 `litellm_params.guardrails` 加上 `"thinking-switch"`。
**`guardrails` 是整列表覆盖语义**，写成 `["thinking-switch"]` 会把 `vision-to-text`
挤掉，必须写全：`["vision-to-text", "thinking-switch"]`。

### 归一化规则

```
thinking.type ∈ {enabled, adaptive}  ->  一个字都不碰
其余（缺失 / disabled / 未知值）      ->  thinking = {"type": "disabled"}
                                         并摘掉 output_config.effort
```

### 设计要点

**白名单而非黑名单。** 未知取值按不思考处理，与下游网关侧规则的保守默认一致。

**开态一个字都不碰，是为了不误伤另一个客户端。** 同一个模型组上还有一路请求发
`{"type":"enabled","budget_tokens":7168}`（指纹：不带 `anthropic-beta` 头、几十个工具、
`max_tokens=8192`、单轮），它的 thinking 本来就完好穿过 litellm —— 闸门对它早退，因为它
既不带 `output_config` 也不是 adaptive。误伤它等于把好的也弄坏。

**只摘 `effort`，不删整个 `output_config`。** 生产里 `output_config` 还承载 `format`
（结构化输出的 json_schema，24 小时内 97 条），删掉会静默破坏结构化输出。实测保留 format
不影响闸门判定。

**必须原地写 `data`，不能只靠返回值。** deployment 级钩子只把返回值里的 `messages` 拷回
kwargs，其余键一概丢弃；而模型组 fallback 只走 router 内部、不重跑 proxy 级钩子，
deployment 级是那条路上唯一的机会。与 `kiro_session_affinity` 同一个坑。

### 与识图 guardrail 的关系

两者挂在同一个模型组上，互不干扰：识图动 `messages` 里的图片块，本 guardrail 只动
`thinking` / `output_config.effort`。

### 实测验证

单元 + e2e 共 18 个测试（`test_thinking_switch.py`）。突变检查：不摘 effort / 整块删
output_config / 返回新 dict 而非原地改 / 白名单漏掉 enabled，四个突变各杀掉至少 2 个测试。

端到端起假后端抓出站字节（`.ops-runbook/scripts/thinking_switch_e2e.py`，拓扑复刻生产：
provider=anthropic + drop_params=True + deployment 挂 guardrails）：

| 入站 | 对照组（不挂） | 实验组（挂） |
|---|---|---|
| 关：无 thinking + effort=high | `enabled/4096` | **`disabled`** |
| 开：adaptive + effort=high | `enabled/4096` | `enabled/4096` |
| 显式 disabled + effort=high | `enabled/4096` | **`disabled`** |
| 另一客户端 `enabled/7168` | 原样 | 原样 |

对照组两态相同（复现缺陷），实验组可区分。

判据坑：关态与开态**必须用同一个 effort**。第一版脚手架给关态 high、开态 xhigh，对照组
出站 4096 vs 8192 看起来"可区分"，误判成缺陷不存在 —— 那是不同 effort 导致的，不是开关。

### 下游契约

下游网关侧规则认 `enabled` / `adaptive` 为要思考，其余补 `{"type":"disabled"}`。归一化后
关态出站 `disabled`（命中它的触发分支）、开态 `enabled`（命中跳过分支），两侧都落在正确侧，
网关那边不用改。

前提是下游网关**转换时别再丢 `thinking`** —— 若它在 anthropic -> openai 转换时把该字段
整个丢掉，litellm 这侧改了也传不过去。

## error_sanitizer — 错误脱敏

把返回给下游的错误体收敛成"只给状态码和一句固定文案"，详细内容只留在内部日志。

### 为什么需要

litellm 默认把内部细节原样透给客户端。限速是最典型的一例，下游看到的是：

```
Error Code: 429
Message: litellm.RateLimitError: Rate limit exceeded for model_per_key:
  deadbeef0123456789abcdef0123456789abcdef0123456789abcdef0badc0de:ratelimit-test.
  Limit type: requests. Current limit: 3, Remaining: 0.
  Limit resets at: 2026-08-28 09:57:17 UTC
```

一行里泄漏了限速维度、key 的哈希、当前额度、剩余量和重置时刻，还带 `litellm.` 前缀暴露
了网关实现。客户端要做的只是退避重试，这些都不必知道。改写后：

```json
{"error": {"message": "Rate limit exceeded, please retry later", "type": "rate_limit_error", "code": "429"}}
```

### 改写规则

| 状态码 | 处理 |
|---|---|
| 429 / 500 / 502 / 503 / 504 | 换成固定文案 |
| 其余（含 400 / 401 / 403 / 404） | 一个字都不碰 |

400 保留原文是刻意的：客户端得知道自己请求哪里写错了，换成一句 "Bad request" 会让人无从
下手。401 / 403 / 404 同理，本身不含内部细节，且下游要靠原文区分是 key 无效还是模型不存在。

改写只影响 HTTP 响应体。原始异常照旧由 litellm 自己的 failure logging 落进
`LiteLLM_ErrorLogs` 与容器日志，取证能力不受影响。

`retry-after` 头必须透传，客户端退避只能靠它。同一异常上的 `reset_at`（绝对时刻，暴露
限速窗口边界）和 `rate_limit_type`（暴露限速维度）都丢掉。这里有个真踩过的坑：
`_handle_llm_api_exception` 取的是**被顶替后**那个异常的 `headers`，原异常的头不会自动
继承，第一版漏了它，下游只能靠猜退避间隔。

### 上线方式

```yaml
guardrails:
  - guardrail_name: "error-sanitizer"
    litellm_params:
      guardrail: error_sanitizer.ErrorSanitizerGuardrail
      mode: "post_call"
      default_on: true
```

`default_on: true` 与其他 guardrail 相反，但**它其实不起作用**：
`async_post_call_failure_hook` 由 `litellm.callbacks` 无条件遍历
（`litellm/proxy/utils.py:2158`），不看 `default_on` 也不看模型级 `guardrails` 列表。
写 `true` 只是让配置读起来不误导人，`mode` 取哪个值同样不影响，`post_call` 是最贴近语义
的那个。因为绕过模型级开关，模型级 `guardrails: [...]` 列表里**不需要**写它，漏写也不会
把它关掉。

要临时关掉只能从 config 里摘掉这条再重启。

### 两个不生效的边界

都是上游结构决定的，不是本文件的缺陷。

**认证阶段的错误改不了。** `user_api_key_auth` 里抛的异常（key 无效、模型不在白名单、
预算超限）走 `ProxyLogging._handle_logging_proxy_only_error`，那条路调
`post_call_failure_hook` 只为记日志，返回值不参与响应构造。好在这类错误本身文案就干净，
泄漏面小。

**流式首字节之后的错误不改。** `async_data_generator` 拿到 `HTTPException` 会直接 raise
而不是 yield SSE error 帧（`common_request_processing.py:3037`），连接被截断，客户端看到
的是传输层异常而非 JSON 错误体。原路径至少会给一个 error 帧，所以中途改写会更糟。判据
是 `request_data` 里有没有 `combined_usage_object`：`post_call_failure_hook` 会把已交付
chunk 的用量提上来（`utils.py:2143`），有它就说明字节已经出门了。

### 测试

26 个用例，变异测试 9/9 全杀：不脱敏、流已交付也脱敏、400 也脱敏、丢 retry-after、透传
全部 header、日志不记原文、detail 改回 dict、错误 type 抹掉、状态码写死。

最有价值的是末尾那组 e2e：把 guardrail 挂进 `litellm.callbacks`，让真实的
`ProxyRateLimitError`（限速器实际抛的那个类，不是自造的 `HTTPException`）走完
`_handle_llm_api_exception`，断言下游最终拿到的 `ProxyException` 里一个内部细节都不剩。
前面的单元测试全过而它挂掉，说明改写虽然发生了却没落到响应体上，那正是这个补丁的意义。

写测试时踩到的：第一版 `bool` 状态码防护是死代码（`True in {429: ...}` 本来就是 False），
变异测试里它存活，说明没有测试能杀掉它，删掉了。


## request_admission 请求上下文准入

`request_admission.py` 使用现有 `CustomLogger` 扩展点，不修改 Router 源码。它在异步候选部署
筛选时检查输入长度，并在选定部署、向上游发送之前再次检查，覆盖直接指定 deployment ID 和
fallback。注册后对 Chat Completions、Anthropic Messages 和 Responses 生效，不依赖客户端
请求的模型名称。同步 SDK、批处理上传、Embedding 和文件上传不属于这条检测路径

候选筛选根据部署自身 `model_info.mode` 或模型目录排除明确的非 Chat 类型，包括 Embedding、
Rerank、音频转写和 evaluation，避免 Router 内部占位消息误触发检查。自定义非 Chat 部署应
明确配置 mode；未知 mode 仍严格检查。客户端请求中的 mode 不作为豁免依据，最终部署检查
继续按实际调用类型执行

输入上限优先使用部署已有 `model_info.max_input_tokens`，未声明时查现有模型目录。没有可解析
正数上限的自定义模型被排除，所有候选都缺上限时返回配置错误，首次启用前应补齐该字段。
唯一例外是内部固定响应 provider `claude_code_review_allow/allow`，不按价格为零豁免普通模型。
等于或超过上限的
候选部署被排除；全部候选超限时返回 `ContextWindowExceededError`，保留已有
`context_window_fallbacks`。例如希望在 100 万处拒绝，可把相应部署的
`model_info.max_input_tokens` 设为 `1000000`，不需要改变供应商模型目录

2026-11-07 已在 patch.4 镜像配套 `0b852acb03f` 外部插件上启用。百万级输入上限封顶
`1000000`，当前 DeepSeek V4.1 Flash 同源部署为 `970000`，保留其他已有较小上限。
两个暂不使用的本地 LM Studio 模型保持原配置，未按未经核验的记忆值补写容量

未启用下述 Claude 上游计数的部署复用 LiteLLM 的 tokenizer、Anthropic system 提取和 Responses 输入转换，包含 system、
instructions、工具定义和工具历史，不重命名工具，不修改用户消息。计数失败或全局计数关闭时，
返回明确的本地无法核验提示，建议 `/compact` 或开启新会话，不再把未知长度的请求继续发给
上游。Responses 的正常加密推理和加密压缩历史除外，这类内容按下述规则保留并交由上游校验

内嵌图片在受限后台线程里复用 Pillow 校验结构和压缩像素。解码前按现有
`MAX_IMAGE_URL_DOWNLOAD_SIZE_MB` 限制分配，解码后遵守 Pillow 像素上限。返回错误时包含
字段路径，不包含图片或请求正文。需要 `proxy-runtime` extra 中的 Pillow，发布镜像安装此 extra 并在构建时验证图片解码。
不向 SDK 核心依赖新增包，也不在请求时下载远端图片

图片大小和像素限制按单张应用，不是整条请求的累计图片预算。大量图片仍会增加解码 CPU 成本，
候选筛选和最终部署校验都会执行；线程池只限制并发，不消除总处理时间

计数副本中的远端图片使用已有 high-detail tile 上界估算，实际请求和 detail 参数保持原样；
本地图片继续使用已有尺寸计数。这仍是估算，不是所有供应商的精确视觉 token 公式。
Responses 的 `reasoning`、`compaction` 输入项携带非空字符串 `encrypted_content` 时，不再因
密文不可本地读取而拒绝。仍通过共享转换器检查可读正文、推理摘要、系统指令、工具定义和工具
历史，原始密文与请求保持不变。此时本地计数只覆盖可读部分，不代表完整上下文长度，密文的
有效性和实际上下文上限由上游最终校验，不能用密文字符数代替 token 数。Messages 中的
`redacted_thinking` 转入 Responses 后同样适用。协议依据见
[OpenAI 推理历史](https://developers.openai.com/api/docs/guides/reasoning)和
[加密压缩历史](https://developers.openai.com/api/docs/guides/compaction)，核对日期 2026-11-09

Responses 的 `previous_response_id`、`conversation`、`item_reference`，以及
共享转换器无法读取的未知输入项仍返回无法核验提示，要求完整可读历史或无状态新会话。启用前必须
确认客户端能接受这项有状态 Responses 限制；不会把这些有效协议项称为非法格式。工具参数中
同名普通 JSON 字段不受此限制。远端文件等供应商持有的内容仍无法仅靠当前请求完整还原。
模型 tokenizer 不一致、后端模板及供应商额外注入也可能造成计数误差，
保守输入上限只能留出余量，不能证明上游绝不会报超限

2026-11-09 加密历史兼容修复仅在本地验证，等待维护窗口与 Opus 计数修复一同部署。
该修复只更新外部准入插件，不额外修改 LiteLLM 核心或 CPA。Opus 仍使用上游计数，计数失败
仍拒绝请求，不因本项兼容修复退回本地估算

本机隔离代理重放同一份失败请求，普通和流式调用真实上游均返回 HTTP 200，实际模型均为
`gpt-6.1-terra`。普通响应输入 26065 tokens，输出限制 32 tokens；流式收到 `message_stop`
且没有错误事件。169 项相关测试通过，覆盖加密历史转发、可读内容超限和 Opus 计数失败拒绝。
部署时需使用本次提交的外部插件，不能继续使用此前已暂存的旧候选插件

已有复杂路由自动压缩处于 pending 时，候选筛选只校验图片，把长度复核留给压缩完成后的
deployment hook。该路径有真实 Router 离线回归，不会新增自动压缩调用。Messages 的客户端
`context_management` 等其他压缩机制可能发生在更晚的 provider 转换阶段，这类请求仍按当前
完整输入严格检查，必要时先由客户端压缩。单字段超大文本是否抽样沿用
`TOKEN_COUNTER_MAX_EXACT_CHARS`，没有添加固定倍率或独立 tokenizer

tokenizer 首次加载遵循 LiteLLM 既有配置。若部署要求禁止 Hugging Face 下载，可使用
`litellm_settings.disable_hf_tokenizer_download: true`，但这会让相关模型回落到通用 tokenizer，
需要接受估算精度变化。本插件不临时改动进程级 tokenizer 开关，也不发起模型生成请求

### Opus、Sonnet、Haiku 上游计数

2026-11-09 晚间 review 补充两项修复，Chat 计数改用生成路径的参数转换，遵守
`additional_drop_params`，并保留输出预算参与 thinking 转换。按上游已合并的
[#47363](https://github.com/BerriAI/litellm/pull/47363)，旧版 `output_format` 仅在计数时
归一为 `output_config.format`，保留 effort 等其他配置。前者避免把生成已删除的字段送去
计数，后者避免计数接口拒绝合法结构化输出。需重新构建核心镜像并配套最终插件，下面的
旧候选镜像和历史测试记录不代表这次修复已发布，服务器旧暂存插件和 manifest 不可直接用于上线

晚间修复后，全部外部插件及相关核心回归合计 786 项通过，新增用例已验证旧实现失败、新实现通过。
以今天改动前的 `cd630e62698` 为基线运行完整 `scripts/pre_commit_lint.sh` 通过，五项 lint gate
均未超出基线，插件单独 basedpyright 为零错误。当前插件 SHA256 为
`99b7ba1cd607f1387ee60128d46d5868e57f4cca55c0875e5dddd63ad6a0ab82`。
这些是本地源码验收结果。随后已构建并预拉取 `v1.108.0-fork.patch.5`，源码 `3b0a3d4a6e9`，
镜像 digest 为 `sha256:30a88b8e23a89cce022ed042c54259b25ab99376cf910237fea58b6c9a17a4ee`。
正式镜像与上述插件通过断网校验，覆盖源码哈希、Pillow、Chat 参数删除、结构化输出、加密历史
和三系列准入。配套文件已独立暂存，尚未执行新镜像真实上游验收或生产切换

三个 Claude 系列候选不再以本地 tokenizer 的估算值判断是否可发送。候选阶段保留图片校验和输入上限检查，
选定部署后调用该部署上游的 `/v1/messages/count_tokens`，等于或超过输入上限即拒绝。
识别依据是部署的模型或 base_model 的 `claude-opus-`、`claude-sonnet-`、`claude-haiku-` 前缀，
不是客户端组名。同组其他后端仍沿用原有检查，不因组名包含 Claude 而调用 Anthropic 计数。
覆盖这三个系列的各个部署渠道，使用选中部署自身的凭据、api_base 和输入上限，不只绑定某个
CPA 地址。实际出站模型仍须是这三个系列且 provider 为 Anthropic，其他协议拒绝发送，避免误用计数端点

计数直接复用现有 `AnthropicCountTokensHandler`，不经 Router 重选部署，也不调用会回退本地的
`acount_tokens()`。命名凭据在副本中按 LiteLLM 原有优先级合并，实际请求不变。Messages 保留
system、tools 和官方计数接口支持的输入参数。Messages 与 baseline 预测共用无预测预算限制的
请求规范化函数，包含原生 thinking 转换、drop_params 和上下文管理；Chat 复用现有 Anthropic
请求转换器。计数投影包含 output_format、context_management、mcp_servers、cache_control、
speed 和 compaction，不静默漏掉这些字段。官方计数接口接受但忽略 compaction，因此结果
仍针对当前消息，不能当作压缩后上下文的预计长度
计数请求保留客户端和鉴权 beta，并复用 Messages 的功能头推导。消息携带 `output_config`
时还加入现有 `mid-conversation-output-config` beta，避免合法的逐轮配置被计数端点拒绝。
Responses 在外层检查图片和隐藏历史，再由现有 Chat 桥接后的 deployment hook 计数，避免
对转换前后请求重复调用。声明原生 `/v1/responses` 透传的上述 Claude 部署会被拒绝，除非显式启用
`use_chat_completions_api`，确保必经 Chat 检查。服务端上下文编辑仍按编辑前的完整输入检查

非空 container 会被明确拒绝，避免漏计隐藏上下文。上述 Claude Chat 非空 extra_body 同样拒绝，
因为当前生成路径将其嵌套发送，无法与计数拍平语义保持一致；应改用原生 Messages 或顶层
受支持参数。失败均不回退本地估算

计数最多等待 30 秒，鉴权失败、超时、上游错误或缺少有效的正整数 `input_tokens` 都拒绝当前
Claude 请求，不回退本地估算。错误只显示异常类型，不回显上游正文或凭据。供应商计数也有延迟、
限流和可用性成本；中转是否真正调用官方计数取决于供应商实现。原有 Router fallback 策略不变

计数请求由计数处理器统一添加单个 `Content-Type: application/json`，移除功能头准备阶段
产生的同名大小写变体，避免重复类型头让严格的 JSON 网关返回 422。User-Agent、X-App
和功能 beta 透传同样覆盖这三个系列的计数请求，正式生成请求头仍沿用原有行为

2026-11-09 本地扩展验证覆盖五个启用部署，计数接口均成功。主上游的三个系列普通生成均
返回 HTTP 200，Sonnet、Haiku 的流式响应也正常结束。两个备用部署生成时报模型或订阅权限
错误，关闭准入插件同样复现，因此不能宣称所有渠道的生成服务正常；本轮未改变生产路由。
223 项相关测试、ruff 和 basedpyright 通过，覆盖三系列的客户端头、计数与生成内容一致、
计数失败拒绝、失败请求隔离和 Responses 桥接。这项扩展只修改外部插件，不增加核心镜像改动，
等待用户安排维护窗口部署，需使用本次最新插件

当前生产已撤回上游计数插件，恢复旧版准入检查。下述两次上线记录均不是当前生产状态。
后续修复须同时验证 LiteLLM 的输入转换与 CPA 的计数兼容性，不能仅凭短文本或合成请求
成功判断可上线。部署时需备份旧插件、核验新文件哈希并重启主代理加载；回滚恢复旧插件再重启。
2026-11-09 首次部署提交 `2b6fab9af53` 后，带逐轮 `output_config` 的计数请求被上游拒绝，
导致正常 Opus 请求进入 fallback。已于 UTC 04:07 回滚旧插件，UTC 04:08 观察到真实 Opus
请求成功。首次部署的健康检查与短文本计数不足以覆盖 Claude Code 实际请求，镜像仍为 patch.4

本地验证通过全部 569 项外部插件回归，其中准入测试 60 项，ruff 和 basedpyright 通过。
新增测试使用真实 Router 和模拟 HTTP 传输，覆盖三种接口的拒绝路径、Chat/Responses 计数与
推理内容一致、命名凭据、无效计数和原生 Responses 透传防漏检，没有发送真实推理请求

部署验收在正式镜像断网容器验证了放行、超限、503 和缺失计数，并在生产容器的独立检查进程中
调用真实上游计数，验证低于上限放行、达到上限拒绝。该检查未调用推理端点，也未修改生产模型
限额；不能视为真实客户端生成请求的端到端验收

此次修复验证通过全部 574 项插件回归，其中准入测试 65 项，ruff 和 basedpyright 通过。
新增回归覆盖原生 Messages 的逐轮 `output_config`、功能请求头，以及同一个 Router 中超限
或计数失败后下一条正常请求仍能使用 Opus。正式镜像独立进程调用真实上游，验证计数
`1200376` 的请求在 100 万上限处拒绝，Messages、Chat 和 Responses 的真实生成均返回 Opus

修复提交 `54f206439cb` 已于 2026-11-09 UTC 04:16 部署，仅替换插件并重启主代理，约
48 秒不可用。正式镜像临时 HTTP 代理依次验证正常请求 200、超限请求 400、后续正常请求
200。生产 `/v1/messages` 的逐轮配置、工具和 thinking 请求返回 Opus，重试与 fallback 均为 0。
但实际用户的 `thinking.display=updates` 随后被 CPA 计数接口拒绝。已于 UTC 04:22:45 再次
回滚旧插件，UTC 04:24 的受影响用户真实请求恢复 Opus，fallback 为 0。生产尚未启用可靠的
上游计数准入，不能把合成请求验收结果当成真实业务已恢复的证据

后续本地候选修复已修改 LiteLLM 核心，需新构建，不能只替换插件。相关回归 778 项通过，
随后补充原生 extra_body 和原始客户端头回归，准入测试总计 101 项通过。原生 Messages 生成路径忽略
extra_body，计数保持相同语义，不将它拍平覆盖顶层参数。两份实际失败请求在本地真实 Router
和受控 HTTP 传输下，计数与生成
的全部计数输入字段一致。该结果尚不等于真实 CPA 验收通过

CPA v8.0.61 对未识别为原生客户端的计数请求重建基础 beta，导致 display updates 等字段的
功能头丢失；它已有原生 Claude Code 透传机制。保留真实客户端的 User-Agent、X-App 和
原始 beta 后，两份实际失败请求在未修改的 CPA 上均返回 200，计数为 52176 和 171941。
因此当前 Claude Code 场景无需修改 CPA 源码或全局配置，也不需要新增 CPA 插件。不得伪造
缺失的客户端身份，也不得将客户端 Authorization、x-api-key 或 Cookie 转发到上游

候选计数插件仅从原始 proxy_server_request.headers、litellm_metadata.headers 或
metadata.headers 读取上述三个白名单头，生成请求头保持不变。缺失原始身份时不合成 CLI
标识，计数失败仍拒绝发送。未识别为原生 Claude Code 的客户端仍受 CPA 现有兼容范围限制

候选核心镜像与新插件已在隔离 HTTP 代理连接未修改的生产 CPA 验收。两份实际请求数据及
原始客户端头回放分别完成普通生成和流式生成，均返回 Opus；中间插入的请求上游计数
1217342，返回超限拒绝，随后正常请求继续成功。这里只限制了验证请求的输出长度，没有
执行返回的客户端工具调用。生产尚未重新上线，CPA 自定义构建与源码补丁路线已停止

CPA 计数仍会移除 context_management 等字段，计数与生成的系统提示注入也有差异。上游
count_tokens 本身不是最终 usage 的严格等值保证，不能仅以返回 200 声称精确准入。
[官方计数说明](https://platform.claude.com/docs/en/build-with-claude/token-counting)

### 注册

将文件放到配置同目录，或只读挂载到 `/app/request_admission.py`，合并进已有 callbacks

```yaml
litellm_settings:
  callbacks:
    - request_admission.callback
```

首次注册和移除需要重新加载服务配置。模型输入限额仍走已有模型管理配置，不需要改插件。
该插件不会随镜像构建自动启用。多模型候选只对不同 tokenizer 模型分别计数，最终发出前还会
复核实际部署，因此对大请求有额外 CPU 开销，使用 LiteLLM 既有计数线程并发限制控制负载

### 验证

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest local_guardrails/test_request_admission.py -q
```

测试禁止 TCP，覆盖三协议图片、上下文边界、system/tools 计数、合法服务端工具历史、计数失败、
模型候选过滤、直接 deployment ID、上下文 fallback，以及已有自动压缩后发送的真实 Router
流程。模拟 provider 验证被拒请求没有到达后端，不代表真实供应商精确计数已经通过验收

## claude_code_review: Claude Code 两种审核模式

`claude_code_review.py` 提供全局 `pre_call` callback 和本地固定响应 provider。
独立审核通过 callback 分流，原生 `safeguards` 复用 LiteLLM 的审核、鉴权和 SSE 实现。
核心补丁只补配置字段、放行分支和原生路由接管，不另写流式协议解析器

模式和作用范围都属于 `general_settings`，对所有模型组的 Messages 请求统一生效，不按主模型名称区分。
`claude_code_review_model_scope` 只决定指定模型模式的作用范围，不是第三种审核模式

| 本地放行 | 审核模型 | 模型作用范围 | 独立客户端审核 | 服务端 `safeguards` |
|---|---|---|---|---|
| `false` | `hoperun` | `client_only` | 转给 `hoperun` | 原样交给上游，保留上游结果 |
| `false` | `hoperun` | `all` | 转给 `hoperun` | LiteLLM 调用 `hoperun` 审核 |
| `true` | 任意值或未配置 | 任意值 | 本地放行 | 本地放行 |

作用范围默认 `all`，保留已有 `safeguards_classifier_model` 的启用语义。
希望优先保留上游原生审核、只把客户端 fallback 转给指定模型时，显式设置 `client_only`。
上游不支持审核时，LiteLLM 在 `client_only` 下不会额外发起服务端分类调用，由 Claude Code 自己回退到独立审核

非原生 Anthropic 后端走 Chat Completions 或 Responses 适配器时，`client_only` 不会把原主模型再调用一次当分类器。
这些接口不能表示 `safeguards` 时，保留原有适配行为，生成响应没有 `safeguard_results`，由 Claude Code 回退到独立审核。
它不伪造允许结果，也不返回表示单个动作审查失败的 `unavailable` 来代替客户端回退

默认未配置时保留原路由。模型模式没有另一个 true/false 开关，模型名为空或 `null` 时不接管。
放行设为 `false` 后会恢复指定模型模式，保留模型名和作用范围即可。放行优先，不受模型作用范围限制

独立审核仍要求系统提示以 `You are a security monitor for autonomous AI coding agents` 开头，
包含完整的 `<block>yes/no</block>` 约定，没有 `<severity>`、工具定义或非用户文本消息。
停止符可缺省或为 `</block>`，不依赖 `max_tokens`。普通对话中引用审核提示不会命中。
未知独立审核模板和 severity 协议保留原路由，不能承诺所有未来 Claude Code 模板零费用

独立审核的模型模式保留原审核提示，改走配置的模型组，先检查调用者对审核模型的权限、预算和限流。
检查失败直接返回 403 或 429，不退回原模型。请求级路由覆盖、mock 和组级 fallback 被关闭，避免偏离指定策略。
放行模式走隐藏内部组，固定返回 `<block>no</block>`，审核输入输出 token 和费用均为零。
客户端不能直访内部组。审计 metadata 的 `claude_code_review` 为 `classifier_model` 或 `direct_allow`，
`claude_code_review_requested_model` 保留原请求模型

服务端模式只接管 `safeguards` 中的 `dangerous_tool_use`。本地放行或模型作用范围为 `all` 时包括原生 Claude 路由，
这是本 fork 的全局策略，区别于上游默认交给 Claude 原生供应商审核的行为。
服务端分类调用同样禁用跨组 fallback，指定审核模型失败时返回 unavailable，不转到其他收费模型。
发往供应商时移除这项审核请求，其他 safeguard 请求保留。返回时替换该项结果，其他结果保留。
工具调用不完整或 JSON 损坏时仍返回 unavailable，固定放行不会把不能执行的工具调用变成有效调用

本地放行只取消审核模型调用，正常生成请求仍按原规则计费。
带 `dangerous_tool_use` 的请求关闭响应缓存读写，即使当前两个字段都关闭，也不复用旧模式的审核结果。
普通请求的缓存不受影响。每次审核在开始时读取配置快照，已开始生成的请求不在中途更换审核模式

### 注册与部署

将文件放到配置文件同目录并只读挂载，保留已有模型和 callbacks

```yaml
volumes:
  - ./claude_code_review.py:/app/claude_code_review.py:ro
```

```yaml
model_list:
  - model_name: __claude_code_review_allow
    litellm_params:
      model: claude_code_review_allow/allow
    model_info:
      input_cost_per_token: 0
      output_cost_per_token: 0

litellm_settings:
  callbacks:
    - claude_code_review.callback
  custom_provider_map:
    - provider: claude_code_review_allow
      custom_handler: claude_code_review.provider
```

callback、本地 provider 和内部组必须一起注册。内部组不配 API key、API base 或外部 fallback。
首次安装需要部署包含核心补丁的镜像并加载插件，之后通过已有管理 API 热切换。
要使用热切换，不要在 YAML 声明这三个配置字段。LiteLLM 的 YAML 所有权规则会拒绝管理 API 修改同名字段，
连 YAML 中写 `false` 或 `null` 也会锁定它们。原来已在 YAML 声明的字段需先移除并重载配置

### 热切换

用代理管理员凭据调用现有 `/config/field/update`，`LITELLM_BASE_URL` 指向自己的代理。
以下命令只更新配置，不调用模型。首次配置建议先选择仅客户端独立审核，保留上游原生审核

```bash
curl -sS "$LITELLM_BASE_URL/config/field/update" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"config_type":"general_settings","field_name":"claude_code_review_model_scope","field_value":"client_only"}'
```

再配置指定审核模型，要求模型组已存在且调用者有权使用

```bash
curl -sS "$LITELLM_BASE_URL/config/field/update" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"config_type":"general_settings","field_name":"safeguards_classifier_model","field_value":"hoperun"}'
```

要统一服务端和客户端审核，都转给指定模型，将 `claude_code_review_model_scope` 通过相同接口改成 `all`。
要恢复上游原生审核，将它改回 `client_only`，不需要重启

切到本地放行，保留审核模型名和作用范围，方便切回

```bash
curl -sS "$LITELLM_BASE_URL/config/field/update" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"config_type":"general_settings","field_name":"claude_code_review_always_allow","field_value":true}'
```

切回指定模型，把上一条 `field_value` 改成 `false`。要停止接管，放行和模型字段分别设为 `false` 和 `null`，作用范围可以保留。
停止接管不代表客户端不再审核，Claude Code 可能恢复单独的收费审核请求

配置沿用现有数据库持久化、缓存失效和 worker 同步。接收更新的 worker 立即应用，其他 worker 通过配置变更通知
和周期重载同步，不是整个集群同一时刻原子切换。周期由 `proxy_config_reload_interval_seconds` 控制。
本地测试覆盖管理 HTTP 请求、数据库替身、SettingsStore、下一次审核的完整链路，未做真实多 worker 集群验收

### 验证

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest local_guardrails/test_claude_code_review.py tests/unit/llms/anthropic/pass_through/test_safeguards.py -q
```

回归覆盖已知独立审核模板、普通请求和未知模板、内部组隔离、权限预算拒绝、请求路由覆盖、
模式优先级及快照、原生 Claude 与适配器的非流式和 SSE，以及开启响应缓存后的上游透传、指定模型、本地放行和恢复透传热切换。
管理端点回归在 `tests/unit/proxy/test_proxy_server_endpoints_and_startup.py`。
全部使用本地替身或禁网测试，未调用付费模型，也未验证真实 Claude Code 客户端与生产多 worker 切换

生产部署记录，2026-11-07

审核配置为 `client_only`、`hoperun`、固定通过 `false`，三个字段由数据库持有且可热更新。
正式镜像在断网容器中通过审核分流、ACL、970000/1000000 两道准入检查，以及真实本地
HTTP 超长输入和损坏图片拒绝测试。生产已读回配置、插件注册和健康状态，未发起付费模型
请求，也未将配置验收视作真实 Claude Code 或多 worker 端到端验证
