# LiteLLM (Fork)

Fork 自 [BerriAI/litellm](https://github.com/BerriAI/litellm)，在上游基础上打了以下补丁：

**PostgreSQL 空字节修复** — `proxy/utils.py`
- 清洗 spend logs 中的 `\x00` 空字节，避免 PostgreSQL jsonb 写入失败（22P05）

**DeepSeek V4 支持及兼容性修复** — `llms/deepseek/chat/transformation.py`、`llms/deepseek/messages/transformation.py`
- 注册 deepseek-v4-flash / deepseek-v4-pro 模型（1M input, 384K output），支持裸名路由
- 修复 thinking mode 多轮对话中 reasoning_content 缺失导致 API 400 的问题
- 修复 tool schema 中 Anthropic `type:"custom"` 未转换为标准 `"object"` 的问题
- 修复 Anthropic thinking_blocks 到 DeepSeek reasoning_content 的转换
- 修复 Anthropic 兼容端点 fallback（如 qwen/claude 回退到 deepseek）时，历史里外来 thinking 块导致 DeepSeek 400 的问题；命中该 400 时自动修复 assistant 历史（redacted_thinking 转占位 thinking 块、给含 tool_use 却无 thinking 块的消息注入占位块）并重试，直连场景不受影响

**Anthropic 流式适配器空 choices 防护** — `llms/anthropic/experimental_pass_through/adapters/transformation.py`、`responses/litellm_completion_transformation/streaming_iterator.py`
- 症状：Claude Code 等 `/v1/messages` 客户端流式调用 OpenAI 形状后端（如 `custom_openai`）时，内容能完整收到，但请求被记为错误且 tokens=0，日志里是 `IndexError: list index out of range`
- 根因：OpenAI 兼容后端会在流末尾发出 `choices: []` 的空帧，而适配器多处裸取 `choices[0]`。空帧有两种来源：OpenAI 规范里 `include_usage` 的 usage-only 尾帧（适配器对流式请求强制开启该选项），以及网关自定义的非标准帧（实测有后端在流末尾追加一个私有类型的计费帧，`usage` 为 null，真实 usage 挂在前一个 `finish_reason` 帧上）。崩溃发生在响应头已发出之后，所以状态码仍是 200，只是缺了 `message_delta` 和 `message_stop`
- 修复：空 choices 的 chunk 不丢弃，加判空守卫后继续走已有的 usage 合并路径，usage 仍能进 `message_delta`。同类问题在 Responses API 桥接层一并修掉（Azure 前导 `prompt_filter_results` 空帧也走这条路）
- 上游进展（1.99.1 同步时核对）：`adapters/streaming_iterator.py` 上游已自行修复（PR [#35314](https://github.com/BerriAI/litellm/pull/35314)，走 `_handle_choiceless_chunk` 在两条流式循环开头统一 `continue` 掉空帧，比逐点判空更彻底），我们在该文件里的三处判空已成死代码，整个文件改回上游实现。我们最初跟踪的 PR [#34455](https://github.com/BerriAI/litellm/pull/34455) 未被合并。`transformation.py` 的 `finish_reason` 裸取与 Responses 桥接层的三处裸取（`_is_reasoning_end`、`_ensure_output_item_for_chunk`、`_get_delta_string_from_streaming_choices`）上游仍未防护，且调用点也没有前置守卫，故这两个文件的补丁继续保留
- 1.100.1 同步时复核：`transformation.py` 的 `finish_reason` 裸取与 Responses 桥接层那三处仍未防护，调用点也仍无前置守卫，补丁原样保留
- 1.101.0 同步时复核：结论同上，上游这一版没有动这两处，补丁原样保留

**OpenRouter OpenAI 系列模型兼容性修复** — `llms/openrouter/chat/transformation.py`
- 修复 Claude Code `Agent` tool schema 中 Anthropic `type:"custom"` 透传导致 OpenRouter 下游 OpenAI/Azure 模型 API 400 的问题
- 支持 Claude Code -> LiteLLM -> OpenRouter(OpenAI 系列模型) 链路正常调用

**Docker 自动发布** — `docker_release_auto.yml`
- tag/release 时自动构建多架构镜像推送 DockerHub 和 GHCR

**流式 tokens 统计修复** — `litellm_core_utils/streaming_chunk_builder_utils.py`、`litellm_core_utils/streaming_handler.py`
- 症状：OpenAI 兼容后端流式调用时 tokens 严重偏低。实测同一 prompt 真实 87/18，客户端与日志只记 15/2；不带 `include_usage` 的那条路径更是直接记 0
- 根因：统计代码用 `"prompt_tokens" in usage` 判断字段存在，而只有 litellm 自己的 `Usage` 定义了 `__contains__`；OpenAI 兼容后端解析出的是 SDK 原生 `CompletionUsage`，`in` 退化为遍历 `(key, value)` 元组，恒为 False，于是真实 token 数被丢弃、回落到本地 `token_counter` 估算
- 影响面：所有走流式的客户端，不限于 Claude Code。tokens 是限流、配额与用量分析的依据，偏低会让这些全部失真
- 修复：改用 shape-aware 读取（dict 用 `.get()`，pydantic 模型用 `getattr`），三种 usage 形状均正确；后端确实未返回 usage 时仍保留估算兜底
- 上游同样未修，且无对应 issue / PR

**Anthropic 适配层参数泄漏与 thinking-only 消息修复** — `llms/anthropic/experimental_pass_through/adapters/transformation.py`、`llms/anthropic/experimental_pass_through/adapters/handler.py`
- 前提：`/v1/messages` 打 OpenAI 形状后端（`custom_openai`）时要经 anthropic↔openai 双向适配层，原生 anthropic 端点那条路径上的既有补丁全部不参与。以下两个缺陷都用同一请求打原生 anthropic 端点做对照确认：原生 200、经适配层失败
- 症状一：请求带 `stop_sequences` / `mcp_servers` / `speed` / `cache_control` / `inference_geo` 任一参数时 500，报 `AsyncCompletions.create() got an unexpected keyword argument`
- 根因一：适配层 `translatable_anthropic_params()` 是反向白名单，名单外的 Anthropic 顶层参数被 `_copy_untranslated_anthropic_params` 原样拷进 `litellm.acompletion(**kwargs)`，一路漏到 OpenAI SDK。`drop_params: true` 拦不住——它只比对 `supported_openai_params`，这些名字压根不在表里。参数有两个注入点：命名参数经 `request_data` 进适配层，其余经 handler 的 `extra_kwargs` 回注，两处都要堵
- 修复一：无 OpenAI 对应的四个参数收进 `ANTHROPIC_ONLY_PARAMS_WITHOUT_OPENAI_EQUIVALENT`，并复用 handler 既有的 `ANTHROPIC_ONLY_REQUEST_KEYS` 机制堵住第二个注入点，清单保持单一来源。`stop_sequences` 原本由我们映射成 OpenAI 标准的 `stop`，上游 1.99.1 起自带了同语义的 `_translate_stop_sequences_to_openai`，1.99.1 那次合并把两份都留下了，于是方法定义、白名单项、调用点各重复一次（后定义胜出，语义相同故无实际故障）。1.100.1 同步时删掉我们那份，只留上游实现
- 症状二：assistant 历史消息只含 thinking 块时上游 400，且上游只回一句通用兜底错误串，不指出违规字段
- 根因二：`strip_empty_text_blocks_from_anthropic_messages` 先摘掉伴随的空 text 块（那是为原生 anthropic 路径加的，见上游 #22930），适配层随后落到 `assistant_content = assistant_message_str` 拿到 `None`，再被 `utils.py` 的 `cleanup_none_field_in_message` 连键一起删掉，产出既无 `content` 又无 `tool_calls` 的非法 OpenAI 消息。唯一变量法钉死了上游规则：只看 `content` 键存不存在，与其值无关、与 `tools` 无关
- 修复二：无 `content` 且无 `tool_calls` 的 assistant 消息补 `content: ""`。OpenAI 规范只允许带 `tool_calls` 时省略 `content`，故不影响工具调用那条路
- 四个无 OpenAI 对应的参数上游至今未处理，无对应 issue / PR；症状二上游同样未修
- 已知遗留（本轮未修）：`top_k` 同样会漏成 500，但 vertex_ai 等 provider 真支持它，而适配层此处拿不到解析后的 provider（`model` 是 model_group 名），无条件丢弃会让 gemini 等后端静默失去该参数，需按 provider 能力决定去留；`stop_reason` 缺 `stop_sequence` 与 `content_filter` 映射；`count_tokens` 漏算 system 与 tools（两条路径都中）；历史轮 thinking 以非标准 `thinking_blocks` 字段发给 `custom_openai`，未转成 `reasoning_content`（fork 里那个转换挂在 `DeepSeekChatConfig` 上，provider 为 `custom_openai` 时拿到的是 `OpenAILikeChatConfig`，补丁不参与），多轮会丢失上一轮推理链

**deployment 级 guardrail 在 `/v1/messages` 上不执行** — `integrations/custom_guardrail.py`
- 症状：挂在 deployment `litellm_params.guardrails` 上的 guardrail，在模型组 fallback 落到该 deployment 时完全不执行。表现为带图请求打主模型组、退到挂了识图 guardrail 的纯文本 deployment 时仍然报错，而直呼同一个 deployment 一切正常
- 根因：`async_pre_call_deployment_hook` 的 call_type 门只认 `completion` / `acompletion`，而 `/v1/messages` 原生直通的 call_type 是 `anthropic_messages`（`utils.py` 取被装饰函数名，再由 `CallTypes()` 转成枚举）。proxy 级 `pre_call_hook` 又只按**请求里的组名**取 guardrail 并集，且 fallback 只在 router 内部重试、不重跑 proxy 级钩子。两者叠加，这个 deployment 钩子是 fallback 路径上唯一的机会，却被门挡住
- 修复：门改为查 `_DEPLOYMENT_PRE_CALL_TYPES` 映射表，加入 `anthropic_messages`，同时把 call_type 到下游字面量的转换收敛到该表（原先是内联三元表达式）。`CallTypesLiteral` 本就含这个值，未放宽到 embedding / responses 等无 messages 可改写的 call_type
- 实测：本地 proxy 构造真实组级 fallback（A 组指向不可达地址必失败 -> B 组为真实纯文本模型 + 识图 guardrail），补丁前后同一请求对照。带图 fallback 补丁前 500（surface 出 A 组的连接错误，即 `Error doing the fallback` 形态），补丁后 200 且 `input_tokens` 与直呼对照逐一致（125），纯文本模型答出图片真实颜色即证明描述已注入；嵌在 `tool_result` 里的图同样生效（178 vs 补丁前原样透传的 237）
- 上游未修，无对应 issue / PR
- 注意：`vision_model` 指向的组不能挂这个 guardrail。识图那次 `router.acompletion` 的 call_type 是 `acompletion`，天然穿过门，一旦该组自身也挂上就会无限递归，guardrail 里没有递归保护

**空回复不写响应缓存** — `caching/caching_handler.py`、`llms/anthropic/experimental_pass_through/messages/response_cache.py`
- 症状：客户端调用后长时间无输出，之后自动重试全部瞬时返回同样的空回复，会话彻底卡死，只能手动发一句"继续"才恢复
- 根因：上游偶发静默拒答，回 HTTP 200 加空 `content` 加 `finish_reason: stop`，litellm 判为 success（`attempted_retries: 0`、`error_information: null`，故 `num_retries` 与模型组 fallback 全部绕过）。这个空回复随后被写进响应缓存，默认 TTL 60 秒，于是窗口内每次重试都命中缓存秒回同一个空回复。生产实测同一 request_id 三条记录：首条真实请求耗时 181 秒，随后两条 `cache_hit=True` 各 0.01 秒返回，重试彻底失去意义
- 修复：`_should_store_result_in_cache` 加一条判空，所有 choices 都没有实质内容时不写缓存。判空覆盖 `content` 之外的 `tool_calls` / `function_call` / `reasoning_content` / `thinking_blocks` / `audio` / `images`，故纯工具调用与纯推理回复不受影响；空格等仍算内容，不猜测语义。该函数是 sync 与 async、流式与非流式四条路径的唯一决策点（流式经 `_add_streaming_response_to_cache` 汇入），改一处全覆盖
- 注意：这只让重试重新有意义，治不了空回复本身，那个根因在上游
- 1.99.1 同步时扩面：上游 1.99 新增了原生 `/v1/messages` 响应缓存，`anthropic_messages` / `aanthropic_messages` 直接进了 `DEFAULT_CACHING_SUPPORTED_CALL_TYPES`，等于默认打开。它自带的两道跳过判定只看「有没有 `message_stop`」和「是不是 error 帧」，而静默拒答回的正是一条格式完好、带 `message_stop`、也不是 error 的空流，两道都放行。原补丁挂在 `_should_store_result_in_cache` 上，够不着这条新路径，升级即等于把这个已修的生产 bug 重新打开。故补两处：流式在 `response_cache.py` 的 `_persist` 里按 SSE 形状判空（非 text 的内容块一律算内容，text 块要求真有 delta 文本），非流式把 `_is_contentless_result` 从只认 `ModelResponse` 扩到也认 Anthropic Messages 的 content 块数组，认不出的形状一律放行
- 1.100.1 同步时复核：上游 `_persist` 仍只有那两道判定，`_should_store_result_in_cache` 也没加判空，两处补丁原样保留。上游 1.100 新增的 `test_async_cache_write_completes_when_asyncio_run_closes_the_loop` 拿空 `ModelResponse()` 当样本，正好被本补丁判成空回复而不写缓存，测试跟着失败；它要测的是「写操作能挺过 event loop 关闭」，与内容无关，故把样本换成带 content 的 response，断言不变
- 1.101.0 同步时复核：上游 `_persist` 与 `_should_store_result_in_cache` 均无变化，两处补丁原样保留
- 部署状态：生产曾用派生镜像 `v1.98.0-fork.patch.1-cachefix`（在 patch.1 上叠一层 COPY 替换该文件），未发 release、未打 tag。1.99.1 起随正式 release 构建，该临时镜像可弃用
- 上游未修，无对应 issue / PR

**只按本地单价计费** — `__init__.py`、`cost_calculator.py`、`litellm_core_utils/litellm_logging.py`
- 场景：上游是自建网关时，它会回传 `x-litellm-response-cost`，litellm 优先采信这个值，于是本地 `model_info` 里配的单价被覆盖。同一个模型流式走上游价、非流式走本地价，两套口径对不上，用量报表没法看
- 修复：加 `litellm.always_use_local_pricing` 开关（环境变量 `LITELLM_ALWAYS_USE_LOCAL_PRICING`，默认关）。打开后 `get_response_cost_from_hidden_params` 直接返回 None，`litellm_logging` 里两处采信 `hidden_params["response_cost"]` 的分支一并跳过，计费全部落回本地单价
- 默认关，不打开时行为与上游完全一致

**预算窗口 reset_at 按 UTC 比较** — `proxy/common_utils/reset_budget_job.py`
- 症状：key / team 的 budget 窗口在非 UTC 时区的机器上不按时重置
- 根因：`reset_at` 用 `.replace(tzinfo=None)` 把带偏移的时间戳直接砍成裸时间（拿到的是当地墙钟），却拿去和裸 UTC `datetime.utcnow()` 比，UTC+8 下整整差 8 小时
- 修复：`reset_at` 统一 `astimezone(timezone.utc)`，`now` 改用 `datetime.now(timezone.utc)`，两边都是 aware UTC
- 上游未修：1.101.0 里 `_reset_expired_window` 仍是 `.replace(tzinfo=None)`，`reset_budget_windows` 仍是 `datetime.utcnow()`
- 补丁边界（1.100.1 同步时核对过，结论是不用扩大）：同文件里 `_reset_budget_for_litellm_keys_chunk` / `users_chunk` / `teams_chunk` 三处的 `datetime.utcnow()` 不是同一类错位。它们只有两个去向，一是 Prisma 的 where 过滤（`expires` / `budget_reset_at`），prisma-client-py 0.11.0 的 `builder.serialize_datetime` 明确把 naive 当 UTC 再打上 tzinfo，`utcnow()` 与 `now(timezone.utc)` 序列化出的查询字节完全一致；二是 `current_time` 传进 `_reset_budget_common`，而那个函数根本没读这个参数，新的 `budget_reset_at` 来自 `compute_budget_reset_at`，后者内部自己取 aware UTC。窗口那条路之所以真的错，是因为 `reset_at` 存在 JSON 里是**字符串**（带 `+08:00` 偏移），偏移一丢就变成当地墙钟；列字段那条路上 tzinfo 两头都由 Prisma 归一，从没被丢过
- 遗留隐患（未改）：上面那三处的正确性依赖「naive 即 UTC」这个 Prisma 契约，`datetime.utcnow()` 恰好满足。Python 3.12 起该方法被弃用，若有人顺手改成 `datetime.now()`，在 UTC+8 上会静默变成 8 小时偏差且无测试拦截；安全的现代化写法是 `datetime.now(timezone.utc)`

**Anthropic 能力头定向透传** — `proxy/litellm_pre_call_utils.py`
- 打 `/v1/messages` 时把客户端的 `anthropic-beta` / `anthropic-version` 透传给下游。Claude Code 的工具 / Agent 能力由 beta 头声明，不透传下游就当这些能力不存在。只放这两个非敏感能力头，不做通用头转发

**本地价格表补缺远端** — `litellm_core_utils/get_model_cost_map.py`
- 远端价格表拉下来后，把本地 `model_prices_and_context_window.json` 里远端还没有的条目补进去。远端优先，本地只补缺，故上游发布某个模型后自动以上游为准
- 需要它是因为 fork 里加了上游尚未收录的模型（如 `anthropic/deepseek-v4-pro`），不补的话联网拉表会把这些条目整个丢掉

1.101.0 同步核对结论：上游这一版没有吸收上面任何一个补丁，逐文件核对后全部保留。`llms/deepseek/` 与 `llms/openrouter/chat/transformation.py` 上游在 1.100.1 到 1.101.0 之间没有改动；价格表与上游只差 `anthropic/deepseek-v4-pro` 这一条镜像

已移除的补丁（保留记录，便于回溯）：

- ~~**Anthropic passthrough 非标准 SSE 帧健壮性**~~ — 我们提交的 PR [#26000](https://github.com/BerriAI/litellm/pull/26000) 已并入上游，本地补丁移除
- ~~**UI 会话 team sentinel 被当成已删除团队**~~ — 上游 1.99 已在 `_token_can_vouch_for_team` 里加了同语义的 `UI_TEAM_ID` 豁免（`UI_TEAM_ID` 与我们用的 `UI_SESSION_TOKEN_TEAM_ID` 同为 `litellm-dashboard`），判据与我们那版一致，本地补丁移除，该文件已与上游完全一致
- ~~**代管的上游 otel cache token 属性**~~ — 这本来就不是我们的补丁，是上游 PR [#38716](https://github.com/BerriAI/litellm/pull/38716)。它当初只进了 1.99.x 这条 stable 线，1.100.x 里没有，所以 1.100.1 同步时由我们保住。1.101.0 已原生自带且实现更完整（多了 `prompt_tokens_details` 兜底取值），`integrations/otel/` 整目录改回上游，与上游差异归零

---

Use it as a **Python SDK** for direct library integration, or deploy the **AI Gateway (Proxy Server)** as a centralized service for your team or organization.

[**Jump to LiteLLM Proxy (LLM Gateway) Docs**](https://docs.litellm.ai/docs/simple_proxy) <br>
[**Jump to Supported LLM Providers**](https://docs.litellm.ai/docs/providers)

---

## Why LiteLLM

Managing LLM calls across providers gets complicated fast — different SDKs, auth patterns, request formats, and error types for every model. LiteLLM removes that friction:

- **Unified API** — one interface for 100+ LLMs, no provider-specific SDK juggling
- **Drop-in OpenAI compatibility** — swap providers without rewriting your code
- **Production-ready gateway** — virtual keys, spend tracking, guardrails, load balancing, and an admin dashboard out of the box
- **8ms P95 latency** at 1k RPS ([benchmarks](https://docs.litellm.ai/docs/benchmarks))

### OSS Adopters

<table>
  <tr>
    <td><img height="60" alt="Stripe" src="https://github.com/user-attachments/assets/f7296d4f-9fbd-460d-9d05-e4df31697c4b" /></td>
    <td><img height="60" alt="image" src="https://github.com/user-attachments/assets/436fca71-988b-40bb-b5fe-8450c80fdbd0" /></td>
    <td><img height="60" alt="Google ADK" src="https://github.com/user-attachments/assets/caf270a2-5aee-45c4-8222-41a2070c4f19" /></td>
    <td><img height="60" alt="Greptile" src="https://github.com/user-attachments/assets/3db0ae72-0843-4005-a56d-bba1dde2193d" /></td>
    <td><img height="60" alt="OpenHands" src="https://github.com/user-attachments/assets/a6150c4c-149e-4cae-888b-8b92be6e003f" /></td>
    <td><h2>Netflix</h2></td>
    <td><img height="60" alt="OpenAI Agents SDK" src="https://github.com/user-attachments/assets/c02f7be0-8c2e-4d27-aea7-7c024bfaebc0" /></td>
  </tr>
</table>

---

## Features

<details open>
<summary><b>LLMs</b> - Call 100+ LLMs (Python SDK + AI Gateway)</summary>

[**All Supported Endpoints**](https://docs.litellm.ai/docs/supported_endpoints) - `/chat/completions`, `/responses`, `/embeddings`, `/images`, `/audio`, `/batches`, `/rerank`, `/a2a`, `/messages` and more.

### Python SDK

```shell
uv add litellm
```

```python
from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# OpenAI
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])

# Anthropic  
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hello!"}])
```

### AI Gateway (Proxy Server)

[**Getting Started - E2E Tutorial**](https://docs.litellm.ai/docs/proxy/docker_quick_start) - Setup virtual keys, make your first request

```shell
uv tool install 'litellm[proxy]'
litellm --model gpt-4o
```

```python
import openai

client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

[**Docs: LLM Providers**](https://docs.litellm.ai/docs/providers)

</details>

<details>
<summary><b>Agents</b> - Invoke A2A Agents (Python SDK + AI Gateway)</summary>

[**Supported Providers**](https://docs.litellm.ai/docs/a2a#add-a2a-agents) - LangGraph, Vertex AI Agent Engine, Azure AI Foundry, Bedrock AgentCore, Pydantic AI

### Python SDK - A2A Protocol

```python
from litellm.a2a_protocol import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams
from uuid import uuid4

client = A2AClient(base_url="http://localhost:10001")

request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello!"}],
            "messageId": uuid4().hex,
        }
    )
)
response = await client.send_message(request)
```

### AI Gateway (Proxy Server)

**Step 1.** [Add your Agent to the AI Gateway](https://docs.litellm.ai/docs/a2a#adding-your-agent) — set `protocolVersion` to `1.0` or `0.3` per agent

**Step 2.** Call Agent via A2A SDK (requires `a2a-sdk>=1.1.0`)

```python
import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, SendMessageRequest
from a2a.utils.constants import TransportProtocol
from uuid import uuid4

base_url = "http://localhost:4000/a2a/my-agent"  # LiteLLM proxy + agent name
headers = {"Authorization": "Bearer sk-1234"}    # LiteLLM Virtual Key

async with httpx.AsyncClient(headers=headers, timeout=60.0) as http_client:
    resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
    agent_card = await resolver.get_agent_card()
    config = ClientConfig(
        httpx_client=http_client,
        streaming=False,
        supported_protocol_bindings=[TransportProtocol.JSONRPC, TransportProtocol.HTTP_JSON],
    )
    client = ClientFactory(config).create(agent_card)

    request = SendMessageRequest(
        message=Message(
            message_id=uuid4().hex,
            role=Role.ROLE_USER,
            parts=[Part(text="Hello!")],
        )
    )
    async for event in client.send_message(request):
        populated = event.ListFields()
        if populated and populated[0][0].name in ("message", "msg"):
            print("".join(getattr(p, "text", "") or "" for p in populated[0][1].parts))
```

[**Docs: A2A Agent Gateway**](https://docs.litellm.ai/docs/a2a)

</details>

<details>
<summary><b>MCP Tools</b> - Connect MCP servers to any LLM (Python SDK + AI Gateway)</summary>

### Python SDK - MCP Bridge

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from litellm import experimental_mcp_client
import litellm

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # Load MCP tools in OpenAI format
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")

        # Use with any LiteLLM model
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's 3 + 5?"}],
            tools=tools
        )
```

### AI Gateway - MCP Gateway

**Step 1.** [Add your MCP Server to the AI Gateway](https://docs.litellm.ai/docs/mcp#adding-your-mcp)

**Step 2.** Call MCP tools via `/chat/completions`

```bash
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Summarize the latest open PR"}],
    "tools": [{
      "type": "mcp",
      "server_url": "litellm_proxy/mcp/github",
      "server_label": "github_mcp",
      "require_approval": "never"
    }]
  }'
```

### Use with Cursor IDE

```json
{
  "mcpServers": {
    "LiteLLM": {
      "url": "http://localhost:4000/mcp/",
      "headers": {
        "x-litellm-api-key": "Bearer sk-1234"
      }
    }
  }
}
```

[**Docs: MCP Gateway**](https://docs.litellm.ai/docs/mcp)

</details>

### Supported Providers ([Website Supported Models](https://models.litellm.ai/) | [Docs](https://docs.litellm.ai/docs/providers))

| Provider                                                                            | `/chat/completions` | `/messages` | `/responses` | `/embeddings` | `/image/generations` | `/audio/transcriptions` | `/audio/speech` | `/moderations` | `/batches` | `/rerank` |
|-------------------------------------------------------------------------------------|---------------------|-------------|--------------|---------------|----------------------|-------------------------|-----------------|----------------|-----------|-----------|
| [Abliteration (`abliteration`)](https://docs.litellm.ai/docs/providers/abliteration) | ✅ |  |  |  |  |  |  |  |  |  |
| [AI/ML API (`aiml`)](https://docs.litellm.ai/docs/providers/aiml) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |
| [AI21 (`ai21`)](https://docs.litellm.ai/docs/providers/ai21) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [AI21 Chat (`ai21_chat`)](https://docs.litellm.ai/docs/providers/ai21) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Aleph Alpha](https://docs.litellm.ai/docs/providers/aleph_alpha) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Amazon Nova](https://docs.litellm.ai/docs/providers/amazon_nova) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Anthropic (`anthropic`)](https://docs.litellm.ai/docs/providers/anthropic) | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |  |
| [Anthropic Text (`anthropic_text`)](https://docs.litellm.ai/docs/providers/anthropic) | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |  |
| [Anyscale](https://docs.litellm.ai/docs/providers/anyscale) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [AssemblyAI (`assemblyai`)](https://docs.litellm.ai/docs/pass_through/assembly_ai) | ✅ | ✅ | ✅ |  |  | ✅ |  |  |  |  |
| [Auto Router (`auto_router`)](https://docs.litellm.ai/docs/proxy/auto_routing) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [AWS - Bedrock (`bedrock`)](https://docs.litellm.ai/docs/providers/bedrock) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |
| [AWS - Sagemaker (`sagemaker`)](https://docs.litellm.ai/docs/providers/aws_sagemaker) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [Azure (`azure`)](https://docs.litellm.ai/docs/providers/azure) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| [Azure AI (`azure_ai`)](https://docs.litellm.ai/docs/providers/azure_ai) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| [Azure Text (`azure_text`)](https://docs.litellm.ai/docs/providers/azure) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ | ✅ | ✅ |  |
| [Baseten (`baseten`)](https://docs.litellm.ai/docs/providers/baseten) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Bytez (`bytez`)](https://docs.litellm.ai/docs/providers/bytez) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Cerebras (`cerebras`)](https://docs.litellm.ai/docs/providers/cerebras) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Clarifai (`clarifai`)](https://docs.litellm.ai/docs/providers/clarifai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Cloudflare AI Workers (`cloudflare`)](https://docs.litellm.ai/docs/providers/cloudflare_workers) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Codestral (`codestral`)](https://docs.litellm.ai/docs/providers/codestral) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Cognition (`cognition`)](https://docs.litellm.ai/docs/providers/cognition) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Cohere (`cohere`)](https://docs.litellm.ai/docs/providers/cohere) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |
| [Cohere Chat (`cohere_chat`)](https://docs.litellm.ai/docs/providers/cohere) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [CometAPI (`cometapi`)](https://docs.litellm.ai/docs/providers/cometapi) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [CompactifAI (`compactifai`)](https://docs.litellm.ai/docs/providers/compactifai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Custom (`custom`)](https://docs.litellm.ai/docs/providers/custom_llm_server) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Custom OpenAI (`custom_openai`)](https://docs.litellm.ai/docs/providers/openai_compatible) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ | ✅ | ✅ |  |
| [Dashscope (`dashscope`)](https://docs.litellm.ai/docs/providers/dashscope) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |
| [Databricks (`databricks`)](https://docs.litellm.ai/docs/providers/databricks) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [DataRobot (`datarobot`)](https://docs.litellm.ai/docs/providers/datarobot) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Deepgram (`deepgram`)](https://docs.litellm.ai/docs/providers/deepgram) | ✅ | ✅ | ✅ |  |  | ✅ |  |  |  |  |
| [DeepInfra (`deepinfra`)](https://docs.litellm.ai/docs/providers/deepinfra) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Deepseek (`deepseek`)](https://docs.litellm.ai/docs/providers/deepseek) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [ElevenLabs (`elevenlabs`)](https://docs.litellm.ai/docs/providers/elevenlabs) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ |  |  |  |
| [Empower (`empower`)](https://docs.litellm.ai/docs/providers/empower) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Fal AI (`fal_ai`)](https://docs.litellm.ai/docs/providers/fal_ai) | ✅ | ✅ | ✅ |  | ✅ |  |  |  |  |  |
| [Featherless AI (`featherless_ai`)](https://docs.litellm.ai/docs/providers/featherless_ai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Fireworks AI (`fireworks_ai`)](https://docs.litellm.ai/docs/providers/fireworks_ai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [FriendliAI (`friendliai`)](https://docs.litellm.ai/docs/providers/friendliai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Galadriel (`galadriel`)](https://docs.litellm.ai/docs/providers/galadriel) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [GitHub Copilot (`github_copilot`)](https://docs.litellm.ai/docs/providers/github_copilot) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [GitHub Models (`github`)](https://docs.litellm.ai/docs/providers/github) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Google - PaLM](https://docs.litellm.ai/docs/providers/palm) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Google - Vertex AI (`vertex_ai`)](https://docs.litellm.ai/docs/providers/vertex) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |
| [Google AI Studio - Gemini (`gemini`)](https://docs.litellm.ai/docs/providers/gemini) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [GradientAI (`gradient_ai`)](https://docs.litellm.ai/docs/providers/gradient_ai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Groq AI (`groq`)](https://docs.litellm.ai/docs/providers/groq) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Heroku (`heroku`)](https://docs.litellm.ai/docs/providers/heroku) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Hosted VLLM (`hosted_vllm`)](https://docs.litellm.ai/docs/providers/vllm) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Huggingface (`huggingface`)](https://docs.litellm.ai/docs/providers/huggingface) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |
| [Hyperbolic (`hyperbolic`)](https://docs.litellm.ai/docs/providers/hyperbolic) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [IBM - Watsonx.ai (`watsonx`)](https://docs.litellm.ai/docs/providers/watsonx) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [Infinity (`infinity`)](https://docs.litellm.ai/docs/providers/infinity) |  |  |  | ✅ |  |  |  |  |  |  |
| [Jina AI (`jina_ai`)](https://docs.litellm.ai/docs/providers/jina_ai) |  |  |  | ✅ |  |  |  |  |  |  |
| [Lambda AI (`lambda_ai`)](https://docs.litellm.ai/docs/providers/lambda_ai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Lemonade (`lemonade`)](https://docs.litellm.ai/docs/providers/lemonade) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [LiteLLM Proxy (`litellm_proxy`)](https://docs.litellm.ai/docs/providers/litellm_proxy) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |
| [Llamafile (`llamafile`)](https://docs.litellm.ai/docs/providers/llamafile) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [LM Studio (`lm_studio`)](https://docs.litellm.ai/docs/providers/lm_studio) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Maritalk (`maritalk`)](https://docs.litellm.ai/docs/providers/maritalk) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Meta - Llama API (`meta_llama`)](https://docs.litellm.ai/docs/providers/meta_llama) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Mistral AI API (`mistral`)](https://docs.litellm.ai/docs/providers/mistral) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [ModelScope (`modelscope`)](https://docs.litellm.ai/docs/providers/modelscope) | ✅ | ✅ | ✅ |  | ✅ |  |  |  |  |  |
| [Moonshot (`moonshot`)](https://docs.litellm.ai/docs/providers/moonshot) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Morph (`morph`)](https://docs.litellm.ai/docs/providers/morph) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Nebius AI Studio (`nebius`)](https://docs.litellm.ai/docs/providers/nebius) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [NLP Cloud (`nlp_cloud`)](https://docs.litellm.ai/docs/providers/nlp_cloud) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Novita AI (`novita`)](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Nscale (`nscale`)](https://docs.litellm.ai/docs/providers/nscale) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Nvidia NIM (`nvidia_nim`)](https://docs.litellm.ai/docs/providers/nvidia_nim) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [OCI (`oci`)](https://docs.litellm.ai/docs/providers/oci) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Ollama (`ollama`)](https://docs.litellm.ai/docs/providers/ollama) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [Ollama Chat (`ollama_chat`)](https://docs.litellm.ai/docs/providers/ollama) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Oobabooga (`oobabooga`)](https://docs.litellm.ai/docs/providers/openai_compatible) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ | ✅ | ✅ |  |
| [OpenAI (`openai`)](https://docs.litellm.ai/docs/providers/openai) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| [OpenAI-like (`openai_like`)](https://docs.litellm.ai/docs/providers/openai_compatible) |  |  |  | ✅ |  |  |  |  |  |  |
| [OpenRouter (`openrouter`)](https://docs.litellm.ai/docs/providers/openrouter) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [OVHCloud AI Endpoints (`ovhcloud`)](https://docs.litellm.ai/docs/providers/ovhcloud) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Perplexity AI (`perplexity`)](https://docs.litellm.ai/docs/providers/perplexity) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Petals (`petals`)](https://docs.litellm.ai/docs/providers/petals) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Pinstripes (`pinstripes`)](https://docs.litellm.ai/docs/providers/pinstripes) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Predibase (`predibase`)](https://docs.litellm.ai/docs/providers/predibase) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Qwen AI Platform (`qwen_ai_platform`)](https://docs.litellm.ai/docs/providers/qwencloud) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  | ✅ |
| [QwenCloud (`qwencloud`)](https://docs.litellm.ai/docs/providers/qwencloud) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  | ✅ |
| [Recraft (`recraft`)](https://docs.litellm.ai/docs/providers/recraft) |  |  |  |  | ✅ |  |  |  |  |  |
| [Replicate (`replicate`)](https://docs.litellm.ai/docs/providers/replicate) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Sagemaker Chat (`sagemaker_chat`)](https://docs.litellm.ai/docs/providers/aws_sagemaker) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Sambanova (`sambanova`)](https://docs.litellm.ai/docs/providers/sambanova) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Snowflake (`snowflake`)](https://docs.litellm.ai/docs/providers/snowflake) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Text Completion Codestral (`text-completion-codestral`)](https://docs.litellm.ai/docs/providers/codestral) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Text Completion OpenAI (`text-completion-openai`)](https://docs.litellm.ai/docs/providers/text_completion_openai) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ | ✅ | ✅ |  |
| [Together AI (`together_ai`)](https://docs.litellm.ai/docs/providers/togetherai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Topaz (`topaz`)](https://docs.litellm.ai/docs/providers/topaz) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Triton (`triton`)](https://docs.litellm.ai/docs/providers/triton-inference-server) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [V0 (`v0`)](https://docs.litellm.ai/docs/providers/v0) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Vercel AI Gateway (`vercel_ai_gateway`)](https://docs.litellm.ai/docs/providers/vercel_ai_gateway) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [VLLM (`vllm`)](https://docs.litellm.ai/docs/providers/vllm) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Volcengine (`volcengine`)](https://docs.litellm.ai/docs/providers/volcano) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Voyage AI (`voyage`)](https://docs.litellm.ai/docs/providers/voyage) |  |  |  | ✅ |  |  |  |  |  |  |
| [WandB Inference (`wandb`)](https://docs.litellm.ai/docs/providers/wandb_inference) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Watsonx Text (`watsonx_text`)](https://docs.litellm.ai/docs/providers/watsonx) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [xAI (`xai`)](https://docs.litellm.ai/docs/providers/xai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Xinference (`xinference`)](https://docs.litellm.ai/docs/providers/xinference) |  |  |  | ✅ |  |  |  |  |  |  |

[**Read the Docs**](https://docs.litellm.ai/docs/)

---

## Get Started

You can use LiteLLM through either the Proxy Server or Python SDK. Both give you a unified interface to access multiple LLMs (100+ LLMs). Choose the option that best fits your needs:

<table style={{width: '100%', tableLayout: 'fixed'}}>
<thead>
<tr>
<th style={{width: '14%'}}></th>
<th style={{width: '43%'}}><strong><a href="https://docs.litellm.ai/docs/simple_proxy">LiteLLM AI Gateway</a></strong></th>
<th style={{width: '43%'}}><strong><a href="https://docs.litellm.ai/docs/">LiteLLM Python SDK</a></strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style={{width: '14%'}}><strong>Use Case</strong></td>
<td style={{width: '43%'}}>Central service (LLM Gateway) to access multiple LLMs</td>
<td style={{width: '43%'}}>Use LiteLLM directly in your Python code</td>
</tr>
<tr>
<td style={{width: '14%'}}><strong>Who Uses It?</strong></td>
<td style={{width: '43%'}}>Gen AI Enablement / ML Platform Teams</td>
<td style={{width: '43%'}}>Developers building LLM projects</td>
</tr>
<tr>
<td style={{width: '14%'}}><strong>Key Features</strong></td>
<td style={{width: '43%'}}>Centralized API gateway with authentication and authorization, multi-tenant cost tracking and spend management per project/user, per-project customization (logging, guardrails, caching), virtual keys for secure access control, admin dashboard UI for monitoring and management</td>
<td style={{width: '43%'}}>Direct Python library integration in your codebase, Router with retry/fallback logic across multiple deployments (e.g. Azure/OpenAI) - <a href="https://docs.litellm.ai/docs/routing">Router</a>, application-level load balancing and cost tracking, exception handling with OpenAI-compatible errors, observability callbacks (Lunary, MLflow, Langfuse, etc.)</td>
</tr>
</tbody>
</table>

**Stable Release:** Use docker images with the `-stable` tag. These have undergone 12 hour load tests, before being published. [More information about the release cycle here](https://docs.litellm.ai/docs/proxy/release_cycle)

Support for more providers. Missing a provider or LLM Platform, raise a [feature request](https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+).

### Deploy on AWS or GCP with Terraform

Run the LiteLLM proxy as a production-ready componentized stack (gateway, backend, UI on separate services; managed Postgres + Redis + object store) using the published Terraform modules. Both modules are on the [public Terraform Registry](https://registry.terraform.io/namespaces/BerriAI) — no auth needed.

#### AWS — ECS Fargate + Aurora + ElastiCache + ALB

[![Launch in AWS CloudShell](https://img.shields.io/badge/Launch-AWS_CloudShell-FF9900?logo=amazon-aws&logoColor=white)](https://console.aws.amazon.com/cloudshell/home) — opens an in-browser shell, already authenticated to your AWS account. Once inside, run:

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm/terraform/litellm/aws/examples/default
cp terraform.tfvars.example terraform.tfvars   # edit region/tenant/env
terraform init && terraform apply
```

[Module page →](https://registry.terraform.io/modules/BerriAI/litellm/aws/latest)

Or call the module from your own root config:

```hcl
# main.tf
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.60" }
  }
}

provider "aws" {
  region = "us-west-2"
}

module "litellm" {
  source  = "BerriAI/litellm/aws"
  version = "~> 1.89"

  region = "us-west-2"
  azs    = ["us-west-2a", "us-west-2b"]
  tenant = "acme"
  env    = "prod"

  # Production: provide an ACM cert. Without one, set allow_plaintext_alb = true
  # (dev/trial only).
  # acm_certificate_arn = "arn:aws:acm:us-west-2:111122223333:certificate/..."
  allow_plaintext_alb = true
}

output "litellm_url" {
  value = module.litellm.alb_dns_name
}
```

```bash
terraform init
terraform apply
```

Provider API keys live in AWS Secrets Manager; reference ARNs via `gateway_extra_secrets`. Full input list and architecture diagram on the [registry page](https://registry.terraform.io/modules/BerriAI/litellm/aws/latest?tab=inputs).

#### GCP — Cloud Run + Cloud SQL + Memorystore + HTTPS LB

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.png)](https://ssh.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FBerriAI%2Flitellm&cloudshell_workspace=terraform%2Flitellm%2Fgcp%2Fexamples%2Fdefault&cloudshell_tutorial=TUTORIAL.md&cloudshell_image=gcr.io/ds-artifacts-cloudshell/deploystack_custom_image&shellonly=true)

Real 1-click. Opens Cloud Shell, clones this repo, and walks you through `terraform apply` via a built-in [DeployStack tutorial](./terraform/litellm/gcp/examples/default/TUTORIAL.md) — pick the project, the tutorial sets up the Artifact Registry remote repo, writes `terraform.tfvars` from your answers, and runs apply.

[Module page →](https://registry.terraform.io/modules/BerriAI/litellm/google/latest)

To call the module from your own config instead, Cloud Run can't pull from `ghcr.io` directly, so first set up a one-time Artifact Registry remote repo backed by GHCR:

```bash
gcloud artifacts repositories create litellm \
  --location=us-central1 \
  --repository-format=docker \
  --mode=remote-repository \
  --remote-docker-repo=https://ghcr.io \
  --project=my-gcp-project
```

Then:

```hcl
# main.tf
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google      = { source = "hashicorp/google",      version = "~> 6.10" }
    google-beta = { source = "hashicorp/google-beta", version = "~> 6.10" }
  }
}

provider "google"      { project = "my-gcp-project"; region = "us-central1" }
provider "google-beta" { project = "my-gcp-project"; region = "us-central1" }

module "litellm" {
  source  = "BerriAI/litellm/google"
  version = "~> 1.89"

  project_id = "my-gcp-project"
  region     = "us-central1"
  tenant     = "acme"
  env        = "prod"

  # Replace my-gcp-project with your GCP project ID (same value as project_id above).
  image_registry = "us-central1-docker.pkg.dev/my-gcp-project/litellm/berriai"

  # Production: provide DNS already pointing at the LB IP for Google-managed certs.
  # Without one, set allow_plaintext_lb = true (dev/trial only).
  # lb_domains         = ["proxy.example.com"]
  allow_plaintext_lb = true
}

output "litellm_url" {
  value = module.litellm.load_balancer_url
}
```

```bash
terraform init
terraform apply
```

Provider API keys live in Secret Manager; reference resource IDs (e.g. `projects/my-gcp-project/secrets/openai-api-key`) via `gateway_extra_secrets`. Full input list and architecture diagram on the [registry page](https://registry.terraform.io/modules/BerriAI/litellm/google/latest?tab=inputs).

#### Both stacks include

- The full componentized split (gateway / backend / UI as independent services)
- Managed Postgres (writer + reader) and Redis
- Versioned object store for proxy state + file uploads
- An auto-generated `LITELLM_MASTER_KEY` in your cloud's secret manager
- A one-off migration job that runs `prisma migrate deploy` before the proxy starts
- The same `proxy_config` surface as the [Helm chart](./helm/litellm/) — pass YAML as a typed map

The Terraform modules live at [`terraform/litellm/aws/`](./terraform/litellm/aws/) and [`terraform/litellm/gcp/`](./terraform/litellm/gcp/) in this repo; the registry entries are read-only mirrors updated on each release.

### Run in Developer Mode
#### Services
1. Setup .env file in root
2. Run dependent services `docker-compose up db prometheus`

#### Backend
1. Run `make bootstrap`
2. Start proxy backend: `uv run python litellm/proxy/proxy_cli.py`

#### Frontend
1. Navigate to `ui/litellm-dashboard` (dependencies were already installed w/ `make bootstrap`)
2. Start dashboard: `npm run dev`

### Verify Docker Image Signatures

All LiteLLM Docker images published to GHCR are signed with [cosign](https://docs.sigstore.dev/cosign/overview/). Every release is signed with the same key introduced in [commit `0112e53`](https://github.com/BerriAI/litellm/commit/0112e53046018d726492c814b3644b7d376029d0).

**Verify using the pinned commit hash (recommended):**

A commit hash is cryptographically immutable, so this is the strongest way to ensure you are using the original signing key:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/0112e53046018d726492c814b3644b7d376029d0/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

**Verify using a release tag (convenience):**

Tags are protected in this repository and resolve to the same key. This option is easier to read but relies on tag protection rules:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/<release-tag>/cosign.pub \
  ghcr.io/berriai/litellm:<release-tag>
```

Replace `<release-tag>` with the version you are deploying (e.g. `v1.83.0-stable`).

---

# Enterprise
For companies that need better security, user management and professional support

[Get an Enterprise License](https://litellm.ai/enterprise)
[Talk to founders](https://enterprise.litellm.ai/demo)

This covers:
- ✅ **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise):**
- ✅ **Feature Prioritization**
- ✅ **Custom Integrations**
- ✅ **Professional Support - Dedicated discord + slack**
- ✅ **Custom SLAs**
- ✅ **Secure access with Single Sign-On**

# Contributing

We welcome contributions to LiteLLM! Whether you're fixing bugs, adding features, or improving documentation, we appreciate your help.

## Quick Start for Contributors

This requires uv to be installed.

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-dev    # Install development dependencies
make format         # Format your code
make lint           # Run all linting checks
make test-unit      # Run unit tests
make format-check   # Check formatting only
```

For detailed contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

> **📖 Contributing to documentation?** The LiteLLM docs have moved to a separate repository: [BerriAI/litellm-docs](https://github.com/BerriAI/litellm-docs). Please open doc PRs there. Docs are served at [docs.litellm.ai](https://docs.litellm.ai).

## Code Quality / Linting

LiteLLM follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Our automated checks include:
- **Black** for code formatting
- **Ruff** for linting and code quality
- **MyPy** for type checking
- **Circular import detection**
- **Import safety checks**


All these checks must pass before your PR can be merged.


# Support / talk with founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- [Community Slack 💭](https://www.litellm.ai/support)
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/BerriAI/litellm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BerriAI/litellm" />
</a>
