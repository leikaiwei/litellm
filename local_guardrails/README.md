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

> 2026-08-05 前本文件还兼管"裸 `tool_result` 收尾时补一个 text 块"，那件事与识图完全独立，
> 已拆到 `tool_result_trailing_text.py`。当初合在一起只是因为生产按文件挂载、新增一个 .py
> 要改 compose 重建容器，不是设计判断。

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

## tool_result_trailing_text — 尾部 text 块注入

会话被 fallback 污染过、且请求以裸 `tool_result` 收尾时，在末尾追加一个非空 text 块。

### 为什么需要

deepseek 思考模式要求带 `tool_calls` 的 assistant 消息回传 `reasoning_content`。opencode
用它签发的 tool id 作**缓存键**存这份 reasoning：自己签的 id 查得到就自动补上，外来 id
查不到就裸奔给 deepseek，被拒并报

```
[invalid_request_error] The `reasoning_content` in the thinking mode must be passed back to the API
```

这句与 `litellm/llms/deepseek/chat/transformation.py:87` 的 docstring 逐字一致。

> **订正**：此前本文件与两份 findings 记的是"opencode 对 `tool_use.id` 做服务端注册表存在性
> 校验"，**那个结论是错的**，据它推导出的方案（改写 id、伪造同形 id）全部无效。
> tool id 不是安全校验对象，只是个缓存键 —— 决定性证据是**假 id + 手工补
> `assistant.reasoning_content`（哪怕只是单空格）就 200**。
> 另一处订正：早前记"末块必须是非空 text"不准确，text 块放在 `tool_result` 前面
> （`tool_result` 仍是最后一块）同样 200，所以闸门并非严格看"最后一个块"。

于是形成**单向棘轮**：Claude Code 每轮重发完整历史，qwen 签的 id 一旦进历史就永久留着。
生产两天实测 qwen 只签发 6 次，而后续携带这些 id 的请求有 3435 次，比例 1:572。按会话看更直白，
三个会话在首次 fallback 之后 100% 的请求都带着外来 id。**fallback 罕见，但影响永久且单向。**

### 注入条件是两个的合取

```
最后一条消息以裸 tool_result 收尾
  AND
整个历史里存在外来 tool id
```

早前版本只判第一个条件、无条件注入，理由写的是"对合法 id 注入同样无害"。这个理由不成立：

注入物**模型看得见、用户看不见**（注入在 litellm 侧、客户端发出之后）。原来注的是一个 `.`，
模型于是把它当成用户发来的谜题并反问"你输入一个点是想表达什么"。铁证在 `SpendLogs.response`
的 `reasoning_content` 里，是模型自己的原话：

```
"The user sent \".\" which is just a period. This likely means they're still watching"
"The user has sent just a period. This is likely a nudge to continue."
```

命中率按请求算：`hoperun` 4.26%（12125 中 517）、`deepseek-v4-flash` 4.43%、
`deepseek-v4-pro` 1.28%；挂 `kiro-session-affinity` 的几个组是 0–0.05%。分界干净落在
"有没有挂这个注入"上。涉及 20+ 用户，一个会话几十上百请求，所以体感是"频繁"。

而同期实测 **72.9% 的注入是白打**（`messages[-1].content[-1].type == 'tool_result'` 为分母，
即注入的真实触发条件）：

| model_group | 总请求 | 注入实际触发 | 其中含外来 id | 白注入 |
|---|---|---|---|---|
| hoperun | 15041 | 8732 | 2510 | 71.3% |
| deepseek-v4-pro | 1216 | 313 | 32 | 89.8% |
| deepseek-v4-flash | 731 | 335 | 0 | **100%** |
| 合计 | 16988 | 9380 | 2542 | **72.9%** |

加上第二个条件后，日常 deepseek 多轮（全 `call_` 前缀）一次都不注入。

**不要因此整个删掉注入**：thinking 开关已修好并上生产（`thinking_switch.py`，commit
`b1d45ff442`），思考现在能真开，也就是 400 能真回来。条件注入在两种状态下都安全：
关态几乎不触发，开态自动接管，不用再改代码。

**也不要把"思考是否开启"加进条件**。看着能更省，但思考的最终状态由 `thinking_switch`
归一化后再叠加 newapi 那条规则决定，guardrail 侧看到的不是最终值，把两个 guardrail 的逻辑
耦起来很脆。按外来 id 判就够，关态多注入几次无害。

### 怎么判"外来"

**allowlist 而非 blocklist**：只认 `call_` 前缀是自己的，其余一律视为外来。这样失败方向偏向
多注入（无害）而不是漏注入（400）。同理，工具块的 id 缺失或不是字符串时也算外来。

生产两天在请求侧实测到的 id 家族：

| 家族 | 出现次数 | 不同 id 数 | 判定 |
|---|---|---|---|
| `call_`（deepseek 经 opencode 签发） | 2005338 | 18457 | 自己人 |
| `toolu_`（qwen3.7-plus） | 481110 | 3359 | 外来 |
| `tc_` | 20298 | 373 | 外来 |
| `call-<uuid>`（**连字符**） | 10788 | 54 | 外来 |
| `toolu_bdrk_`（kiro 的 claude） | 5322 | 511 | 外来 |
| `chatcmpl-tool-` | 2482 | 15 | 外来 |
| `<工具名>_xxx`（`Read_g33kfzc4vw7`） | 1532 | 8 | 外来 |

`call-<uuid>` 是个陷阱：它与自己人只差一个字符，前缀判断写成"`call` 开头"就会把这一万次
误判成自己人而漏注入。所以 `OWN_TOOL_ID_PREFIX` 必须精确匹配到下划线。

**已接受的漏网**：`call_` 前缀的 id 若来自**另一个 opencode 账号**，同样不在该账号的
reasoning 缓存里，但按前缀会被判成"自己的"而不注入，于是 400。会话亲和能压低概率，消不掉。
漏网后果是回到修复前的 400，不产生新问题。

### 必须扫全历史

后端那条要求是两段式的：**触发**看最后一块是不是裸 `tool_result`（不是就根本不校验，历史
多脏都无所谓）；**一旦触发就扫全历史**，任何一个外来 id 都 400。

所以第二个条件不能只看末条。被污染会话里最后那个 `tool_result` 往往是 deepseek 自己签的
干净 id，脏 id 躺在前面几十轮里，照样 400。`tool_use`（assistant 侧）与 `tool_result`
（user 侧）两侧的 id 都要扫 —— 后端查缓存用的正是 assistant 侧的 id（实测 assistant 假 /
tool_result 真 -> 400，反过来报的是另一个错）。

代码里先判尾块再扫历史：尾块判定只看一条消息，历史扫描是全量，而生产上约半数请求不以裸
`tool_result` 收尾，这些直接短路掉。

### 注入内容

`Continue.`，三条约束叠出来的：

1. **非空白**。litellm 发给后端前会跑 `strip_empty_text_blocks_from_anthropic_messages`
   （`llms/anthropic/common_utils.py`），用 `.strip()` 判空并摘掉纯空白 text 块 —— 摘完又变回
   裸 `tool_result` 收尾，等于没修。空串、单空格、NBSP 实测均无效
2. **对模型讲得通**。注入点恰是"工具刚返回、模型要决定下一步"这一刻，所以 `Continue.` 是真话；
   而 `.` 是个谜题，模型只能猜，于是有了生产上那些反问
3. **英文**。Claude Code 系统提示是英文，注入中文有诱发模型切换回复语言的风险

**要记的代价**：如果那一刻正确行为是**停下**（任务已完成），`Continue.` 可能推着模型多做
一步。`.` 没这个风险但有困惑风险。条件注入把频率降下来后这个代价可接受，但要观察。

已排除的候选：不可见字符（ZWSP、WORD JOINER、BOM、HANGUL FILLER、SOFT HYPHEN）在 HTTP 层
可行（活过两道关，假 id 400 -> 200），但**"模型看不见它"是未验证假设** —— 模型看 token 不看
像素，很可能说"用户发了个空消息"。

### 判定"最后一条消息"不是 `messages[-1]`

有两类消息不参与后端校验，注入落在它们身上等于没注入。判定实现为 `_counts_as_tail` +
`_last_effective_message_index`；这两类消息由 litellm 或下游转换器自己处理，本 guardrail
不删不改。

第一类：**会被摘空后整条删掉的消息**。这是空白 text 块那个坑的消息级版本，易只做一半 ——
清理函数在 content 摘空后把**整条消息**从列表里删掉，不是留个空数组。所以末条消息整条只含
空白块时，前一条裸 `tool_result` 会重新成为末条，只看 `messages[-1]` 会认为无需注入，
litellm 丢掉末条后请求变回裸收尾，照样 400 且不报错。边界要与 litellm 逐字对齐：
content **本来就是** `[]` 的消息不会被丢（走 `len` 相等那条分支），仍算末条。

第二类：**尾部的 `role: system` 消息**。Anthropic 的 `messages` 只允许 `user` / `assistant`，
`system` 是顶层字段；但 Claude Code 确实会在末尾发独立的 `role: system` 提醒消息
（`The task tools haven't been used recently…`），litellm 原样透传，下游 anthropic -> openai
转换器把它上提到开头，于是后端看到的收尾又变回它前面那条。**这是 08-01 生产 400 的直接根因**：
生产 11 条失败请求里注入的文本一个都没出现，因为注入全打在了这条不参与校验的 system 消息上
（它 content 是字符串，旧版判定直接提前 return）。判定只看 `role`，不看 content 形状。
只跳过**尾部**的：中间那条 system 后面还有真正的 user 收尾，此时无需注入。

**只处理 Anthropic 形状**。末条是 `role: "tool"` 的 OpenAI 形状留 no-op：那种形状只能另起一条
user 消息，会造成连续两条 user，未实测过。生产走 `/v1/messages`，进 hook 时是 Anthropic 原生形状
（转换发生在 hook 之后，见 `llms/anthropic/experimental_pass_through/adapters/`）。

### 配置

按文件挂载（不是挂目录），所以新增文件要改 compose 再 `docker compose up -d litellm` 重建容器：

```yaml
volumes:
  - ./tool_result_trailing_text.py:/app/tool_result_trailing_text.py:ro
```

```yaml
guardrails:
  - guardrail_name: "tool-result-trailing-text"
    litellm_params:
      guardrail: tool_result_trailing_text.ToolResultTrailingTextGuardrail
      mode: "pre_call"
      default_on: false        # 必须 false，否则变成全局常开
```

然后给需要的 deployment 挂上（与 `vision-to-text` 并列，两者互不干扰）：

```yaml
      guardrails: ["vision-to-text", "tool-result-trailing-text"]
```

多个 guardrail 在同一请求里是**串行**执行的（`proxy/utils.py:1418` 那个循环，
`data = result` 逐个接力），所以两个都改 `messages` 也不会互相覆盖。
`asyncio.gather` 那条并发路径只对显式 `run_in_parallel: true` 的 guardrail 生效，
而且并发分支会**丢弃**它们返回的改写（设计上只用于阻断）。

### 实测验证

直连 newapi（`.ops-runbook/scripts/toolid_verify.py`，每例重复 3 次，全部一致）：

| 用例 | 结果 |
|---|---|
| 真 id + 裸 `tool_result` 收尾（正对照） | 200 |
| 假 id + 裸 `tool_result` 收尾（故障复现） | **400** |
| 假 id + 尾部 text 块（本修复） | **200** |
| 真 id + 尾部 text 块（注入无害） | 200 |
| 假 id + 尾部 text `""`（空串） | **400** |
| 假 id + 补 `assistant.reasoning_content`（含单空格） | **200** ← 坐实"只是缓存键" |
| 假 id + 另起一条 user 消息收尾 | 200 |

Anthropic 入口（生产真实入口）：

| 用例 | 结果 |
|---|---|
| 裸 `tool_result` + 末条 `role:system` | **400** ← 生产失败形状 |
| `tool_result` + text 收尾 + 末条 `role:system` | **200** ← 本修复 |

单元测试 62 项（`test_tool_result_trailing_text.py`）。变异测试 8/8 全杀：

| 变异 | 被杀测试数 |
|---|---|
| 前缀去掉下划线（`call_` -> `call`） | 2 |
| 删掉外来 id 条件（退回无条件注入） | 6 |
| 注入内容退回 `.` | 1 |
| 历史扫描只看末条 | 30 |
| id 缺失算自己人 | 4 |
| 只认 `tool_result` 侧的 id | 28 |
| 尾部 system 不跳过 | 5 |
| 不跳过空白 text 块 | 6 |

测试里每个断言"不注入"的用例都显式让**另一个**条件成立（`_polluted()` 辅助函数），
否则干净会话里无论尾块什么形状都不注入，那种用例删掉尾块判断也照样通过 —— 等于空断言。

### 待验

- **`Continue.` 的行为影响未实测**，特别是"任务已完成时被推着多做一步"这个代价
- 不可见字符"模型看不见"这条无法用小样本证明。两版行为测试的 `dot` 对照组都是 0/5 提及，
  与生产铁证矛盾，自检判定脚手架无效 —— 生产是 76k–500k token 长会话加 Claude Code 系统提示，
  本地两百 token 复现不出。要验只能拿生产真实长会话 payload 回放

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

### 与另外两个 guardrail 的关系

三者都挂在 hoperun 上，互不干扰：识图动 `messages` 里的图片块，`tool_result_trailing_text`
只在 `messages` 末尾追加，本 guardrail 只动 `thinking` / `output_config.effort`。

但与 `tool_result_trailing_text` 有个连带效应：那个注入是为了绕开外来 tool id 被拒，而
那个 400 的真因是**思考模式要求回传 `reasoning_content`**。所以思考关着时注入是白打、
开着时必需。**不要因此删掉注入**，也**不要把思考状态加进注入条件** —— 最终状态由本
guardrail 归一化后再叠加 newapi 那条规则决定，那边看到的不是最终值，耦起来很脆。
它按外来 id 判就够了。

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
