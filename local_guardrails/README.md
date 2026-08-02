# 本地自定义 guardrail

放在这里的是**不修改 litellm 源码**的外部 guardrail。litellm 原生支持从 config.yaml
同目录加载自定义 guardrail（`litellm/proxy/types_utils/utils.py` 的 `get_instance_fn`），
所以升级 / 合并上游时这里零冲突。

跑测试：

```bash
.venv/bin/python -m pytest local_guardrails/ -q
```

## vision_to_text — 请求改写

在请求发给后端之前改写 messages，做两件彼此独立的事：

1. **图片转文本**：把图片交给视觉模型识别，用识别文本替换原图片块，让纯文本模型能处理带截图的请求
2. **尾部 text 块注入**：裸 `tool_result` 收尾时补一个非空 text 块，绕开 opencode 的 tool id 校验

名字只覆盖第一件事。没拆成两个 guardrail 是因为挂载点完全相同（都是 opencode 这条路上的
`pre_call`），而生产上新建一个 guardrail 要改 config.yaml 的 `guardrails:` 段、再逐个 PATCH
deployment 的 `guardrails` 数组；合进已挂载的文件只需替换一个 py 文件加重启。

### 图片转文本

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

### 尾部 text 块注入 — 绕开 opencode 的 tool id 校验

opencode（`opencode.ai/zen/go`）对 `tool_use.id` 做**服务端注册表校验**，只接受它自己签发过的 id。
而 Claude Code 不自己生成 tool id，只回传后端给的，于是形成**单向棘轮**：

```
opencode 服务这轮 -> id 由 opencode 签发 -> 下轮 opencode 认
其它后端服务这轮  -> id 由它签发        -> 下轮 opencode 必拒 -> 又落该后端 -> 再加一个外来 id
```

**一次 fallback 就永久污染该会话**，之后每轮加深，表现为「上下文进了别的模型就再也回不到 opencode」。
生产 2026-08-01 凌晨命中：`hoperun` 组唯一活跃成员经 newapi 打 opencode，对被污染会话 100% 返回
400 `Error from provider (Console Go): Upstream request failed`，全部降级到 qwen。

改写 id 这条路走不通：同前缀同长度的自编 id 也被拒（真 id 只改末 4 字符即 400），
是存在性校验而非格式校验。

**但校验是两段式的**：只有请求最后一个 content 块是裸 `tool_result` 时才**触发**，
一旦触发就扫全历史的 id。所以末尾追加一个非空 text 块，校验根本不启动，历史里有多少外来 id
都无所谓。对合法 id 注入同样无害，因此无条件执行，不必判断 id 来源。

直连 newapi 实测（`.ops-runbook/scripts/toolid_verify.py`，每例重复 3 次，全部一致）：

| 用例 | 结果 |
|---|---|
| 真 id + 裸 `tool_result` 收尾（正对照） | 200 |
| 假 id + 裸 `tool_result` 收尾（故障复现） | **400** |
| 假 id + 尾部 text `"."`（本修复） | **200** |
| 真 id + 尾部 text `"."`（注入无害） | 200 |
| 假 id + 尾部 text `""`（空串） | **400** |
| 假 id + 另起一条 user 消息收尾 | 200 |

实现要点：

**注入内容必须非空白**。litellm 发给后端前会跑
`strip_empty_text_blocks_from_anthropic_messages`（`llms/anthropic/common_utils.py`），
用 `.strip()` 判空并摘掉纯空白 text 块 —— 摘完又变回裸 `tool_result` 收尾，等于没修。
同理，判断「是否裸 tool_result 收尾」时也要**跳过末尾的空白 text 块**：Claude Code 常回传
`{"type": "text", "text": ""}`，看着有 text 收尾，实际发出去时已被摘掉。
`text` 非 str（`None` 或缺键）也算会被摘掉，与 litellm 的 `_is_empty_text_block` 逐条对齐。

**「最后一条消息」不是 `messages[-1]`**。有两类消息不参与后端校验，注入落在它们身上等于没注入。
判定实现为 `_counts_as_tail` + `_last_effective_message_index`，注入打在真正会被校验的那条上；
这两类消息由 litellm 或下游转换器自己处理，本 guardrail 不删不改。

第一类：**会被摘空后整条删掉的消息**。这是空白 text 块那个坑的消息级版本，容易只做一半 ——
清理函数在 content 摘空后把**整条消息**从列表里删掉，不是留个空数组。所以末条消息整条只含
空白块时，前一条裸 `tool_result` 会重新成为末条，只看 `messages[-1]` 会认为无需注入，
litellm 丢掉末条后请求变回裸收尾，照样 400 且不报错。边界要与 litellm 逐字对齐：
content **本来就是** `[]` 的消息不会被丢（走 `len` 相等那条分支），仍算末条。

第二类：**尾部的 `role: system` 消息**。Anthropic 的 `messages` 只允许 `user` / `assistant`，
`system` 是顶层字段；但 Claude Code 确实会在末尾发独立的 `role: system` 提醒消息
（`The task tools haven't been used recently…` / `The TodoWrite tool hasn't been used recently…`），
litellm 原样透传，下游 anthropic -> openai 转换器把它上提到开头，于是后端看到的收尾又变回
它前面那条。**这是 08-01 生产 400 的直接根因**：生产 11 条失败请求里注入的 `"."` 一个都没出现，
因为注入全打在了这条不参与校验的 system 消息上（它 content 是字符串，旧版判定直接提前 return）。
判定只看 `role`，不看 content 形状 —— 拿 content 形状做判断正是漏掉它的原因。
只跳过**尾部**的：中间那条 system 后面还有真正的 user 收尾，此时无需注入。

Anthropic 入口实测（生产真实入口）：

| 用例 | 结果 |
|---|---|
| 裸 `tool_result` + 末条 `role:system` | **400** ← 生产失败形状 |
| `tool_result` + text 收尾 + 末条 `role:system` | **200** ← 本修复 |

**不能挂在图片分支下**。生产上撞这个校验的请求大多不带图，而识图那条路无图时提前返回。
两个修复各自独立判断，都不需要改写时才返回 `None`。

**只处理 Anthropic 形状**。末条是 `role: "tool"` 的 OpenAI 形状留 no-op：那种形状只能另起一条
user 消息，会造成连续两条 user，未实测过。生产走 `/v1/messages`，进 hook 时是 Anthropic 原生形状
（转换发生在 hook 之后，见 `llms/anthropic/experimental_pass_through/adapters/`）。

副作用：注入的 `"."` 会进入对话被模型看到。实测模型正常作答，但确实是注入内容。

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
