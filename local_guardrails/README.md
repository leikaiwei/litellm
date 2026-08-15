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

本地 proxy + 真实纯文本后端（newapi 上的 deepseek 两个模型）+ 真实视觉模型组，
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

- **`vision_model` 必须指向一个「每个成员都确认能识图」的组**（生产已踩，2026-07-30）。
  fail-open 只在调用**抛异常**时触发；若视觉模型返回 HTTP 200 加一句客气的拒绝
  （实测：讯飞 `xopglm52` 回 "I cannot describe the image..."），那句拒绝会被当成合法描述
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

再给每个 hoperun deployment 的 `litellm_params.guardrails` 加上 `"thinking-switch"`。
**`guardrails` 是整列表覆盖语义**，写成 `["thinking-switch"]` 会把 `vision-to-text`
挤掉，必须写全：`["vision-to-text", "thinking-switch"]`。

### 归一化规则

```
thinking.type ∈ {enabled, adaptive}  ->  一个字都不碰
其余（缺失 / disabled / 未知值）      ->  thinking = {"type": "disabled"}
                                         并摘掉 output_config.effort
```

### 设计要点

**白名单而非黑名单。** 未知取值按不思考处理，与下游 newapi 侧规则的保守默认一致。

**开态一个字都不碰，是为了不误伤另一个客户端。** hoperun 上还有一路请求发
`{"type":"enabled","budget_tokens":7168}`（指纹：不带 `anthropic-beta` 头、68/71 个工具、
`max_tokens=8192`、单轮），它的 thinking 本来就完好穿过 litellm —— 闸门对它早退，因为它
既不带 `output_config` 也不是 adaptive。误伤它等于把好的也弄坏。

**只摘 `effort`，不删整个 `output_config`。** 生产里 `output_config` 还承载 `format`
（结构化输出的 json_schema，24 小时内 97 条），删掉会静默破坏结构化输出。实测保留 format
不影响闸门判定。

**必须原地写 `data`，不能只靠返回值。** deployment 级钩子只把返回值里的 `messages` 拷回
kwargs，其余键一概丢弃；而模型组 fallback 只走 router 内部、不重跑 proxy 级钩子，
deployment 级是那条路上唯一的机会。与 `kiro_session_affinity` 同一个坑。

### 与识图 guardrail 的关系

两者都挂在 hoperun 上，互不干扰：识图动 `messages` 里的图片块，本 guardrail 只动
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

newapi 侧规则认 `enabled` / `adaptive` 为要思考，其余补 `{"type":"disabled"}`。归一化后
关态出站 `disabled`（命中它的触发分支）、开态 `enabled`（命中跳过分支），两侧都落在正确侧，
newapi 那边不用改。

前提是 newapi **转换时别再丢 `thinking`** —— 它当前在 anthropic -> openai 转换时把该字段
整个丢掉，不修的话 litellm 这侧改了也传不过去。
