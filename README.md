# LiteLLM (Fork)

Fork 自 [BerriAI/litellm](https://github.com/BerriAI/litellm)，在上游基础上打了以下补丁：

2026-11-13 已将上游稳定版 [v1.109.0](https://github.com/BerriAI/litellm/releases/tag/v1.109.0) 的 fork 部署到生产，发布源码为 `4972faa706f`，[v1.109.0-fork 镜像构建](https://github.com/leikaiwei/litellm/actions/runs/43062709338)成功。主代理和 usage-api 使用同一固定镜像，外部准入及审核插件已配套更新

Claude 预检查命中输入上限时抛出 `ContextWindowExceededError`，不发送该部署的生成请求，保留既有 fallback。当前生产 `num_retries=0`，全局及模型组重试策略均未设置，因此不会在该模型上重复尝试。若以后显式设置 `BadRequestErrorRetries` 等策略，上游允许它覆盖默认的不重试行为，不能把本插件描述成无条件禁止重试

2026-11-13 取消外部准入插件在每次 Claude 生成前追加的远程 `count_tokens`，Opus、Sonnet、Haiku 统一复用现有本地 tokenizer 检查输入长度。客户端主动调用的计数端点和 `thinking.display=updates` 参数限制保留，已有部署输入上限及路由不变。该改动只更新外部插件，不需要重建镜像

生产插件已替换并重启验证。隔离代理的真实普通、流式请求均确认自动计数为零，计数端点模拟 503 时生成仍成功。上线后普通和流式请求均返回 200，`thinking.display=updates` 仍返回 400

本地预检查仍可能低估 CPA 改写后的完整输入。同日凌晨上线时已将六个 Opus、Sonnet、Haiku 部署的输入上限设为 `950000`，CPA 保持现有改写。Haiku 5.5 的[官方窗口为 100 万](https://platform.claude.com/docs/en/models/haiku-5-5/whats-new-haiku-5-5)，当时同时替换两个部署遗留的 20 万输入上限。95 万只提供余量，不保证所有百万 token 超限都在生成前被识别

2026-11-06 发布范围调整，撤回 `27e276c5112` 和 `0d923ebc042` 的请求准入及错误路径补丁，保留 1.108.0 升级和此前旧补丁。`v1.108.0-fork.patch.1` 已含准入修改，`v1.108.0-fork.patch.2` 进一步包含错误路径修改，两者均不作为本次上线候选，也不移动既有标签

不含这批修改的已发布镜像是 `leikaiwei/litellm:v1.108.0-fork`，发布 digest 为 `sha256:a677652db2903c8b637121e87b04a01721e236df9969e033865aba858bc0490a`。其核心运行逻辑与撤回后的版本一致，源码仅保留 Responses schema 参数的类型声明，并说明重试处理多种供应商异常所需的 BLE001 豁免，均不改变运行行为。外部 Claude Code 审核改动另存于 `f72874dd9cd`，不随该镜像启用。此段记录当时的源码回退与候选镜像，当前部署见下文

2026-11-07 已部署 `v1.108.0-fork.patch.4`，镜像固定为 `leikaiwei/litellm@sha256:2338a4f06e9a5e1a92e4934bf965ea4630ab7aa07ade5e2935a62850328421c0`。镜像源码为 `aad820d4d01`，外部只读挂载插件配套 `0b852acb03f`，后者补齐非聊天请求边界。生产启用 `client_only`，独立审核交给 `hoperun`，固定通过关闭。严格准入已启用，百万级输入上限封顶 `1000000`，当前 DeepSeek V4.1 Flash 同源部署为 `970000`，已有较小限额保持不变。审核三字段由数据库管理，可通过管理 API 热更新。备份和回滚入口已保留，只有主代理容器重启

宿主机日志维护：[基础日志永久保存、正文保留90天](scripts/spend_log_body_retention.md)，独立于代理镜像运行，避免官方整行保留策略删除费用和token明细

**Claude Code 两种审核模式**：外部 [claude_code_review callback](local_guardrails/README.md#claude_code_review-claude-code-两种审核模式) 加原生 safeguards 小补丁
- `general_settings.safeguards_classifier_model` 指定审核模型组，没有额外的模型启用开关，清空模型名即停止接管。`general_settings.claude_code_review_model_scope=client_only` 只接管已识别的客户端独立审核，保留上游服务端审核；设为 `all` 则同时接管 Messages `dangerous_tool_use`。范围默认 `all`，保留已有配置的启用语义，按需分流时应显式选择 `client_only`
- `general_settings.claude_code_review_always_allow=true` 优先本地放行，两种协议均生效，不受范围或主模型名称限制。模型名与范围可保留，关闭放行后恢复指定模型策略。固定模式只免除审核调用费用，正常生成仍收费
- 独立审核由 callback 分流，服务端审核复用上游 evaluator 与 SSE。接管服务端审核时包括原生 Claude 路由，移除送往供应商的该项审核请求，并替换返回的同类结果，其他 safeguards 保留
- 补齐配置 schema 后，复用管理 API、SettingsStore 和数据库配置同步热切换，不另建管理端点或轮询服务。热配置字段不能同时由 YAML 锁定，首次安装插件仍需部署。跨 worker 为最终一致，已开始的审核保持配置快照
- 带审核标记的请求禁用响应缓存读写，避免模式切换后复用旧结论。未知独立模板不接管。callback、本地 provider 和隐藏内部模型组一起注册，镜像升级不会自动启用

两种形态按协议区分，不按客户端版本推断。[Claude Code 官方说明](https://code.claude.com/docs/en/permission-modes#server-side-classifier-review) 明确新版也可能在服务端不支持审核时回退为独立请求。`client_only` 不额外发起服务端分类调用，实际是否回退取决于客户端及上游协议支持

此补丁保留上游审核协议实现，升级时重点核对 `safeguards.py`、两个 Messages handler 和配置字段。相关回归覆盖管理 HTTP 热更新、权限预算、双协议、原生与桥接路由、SSE、缓存和优先级。已随 patch.4 部署，完成断网测试及生产配置读回，未执行付费模型或真实 Claude Code 端到端验收

**工具数组 token 计数兼容**：`litellm_core_utils/token_counter.py`
- 回移上游 [PR #46180](https://github.com/BerriAI/litellm/pull/46180) 的最小修复，兼容工具数组未声明 `items`、布尔及旧版元组 schema，避免计数抛出 `KeyError` 或 `AttributeError`
- 仅修复计数异常，不改变上下文阈值、计数失败策略或路由规则，本地 token 计数仍是估算

2026-11-07 补充合法历史计数兼容，局部回移已合并的 [#44784](https://github.com/BerriAI/litellm/pull/44784) 压缩摘要、[#48699](https://github.com/BerriAI/litellm/pull/48699) 搜索结果、[#48624](https://github.com/BerriAI/litellm/pull/48624) Anthropic 工具选择和 [#48526](https://github.com/BerriAI/litellm/pull/48526) 内嵌 auto 图片尺寸计数。只取 tokenizer 及必要类型，不回移这些 PR 的其他端点或路由变更。另按 Anthropic SDK 的格式核对后，复用已有 tool use/result 字段计数 `server_tool_use` 及空或错误的 `web_search_tool_result`，不改变原始请求，也不为基础安装新增 Anthropic SDK 依赖

加密搜索结果不能仅按标题和 URL 当作完整正文，仍报告本地无法计数。未知块也不按零 token 放行。[#43371](https://github.com/BerriAI/litellm/pull/43371) 已关闭但未合并，其未知块转 JSON 方案未采用。[#47335](https://github.com/BerriAI/litellm/pull/47335) 的 `original` 图片算法仍在讨论，本次不把它当作精确计数回移。图片计数修复不替代图片有效性检查，模型 tokenizer 差异也仍存在

本次在已有 `test_token_counter.py` 中验证摘要与签名分离、搜索结果嵌套计数、服务端工具的空和错误结果、加密内容不静默漏计、两协议 tool choice 等价及内嵌 auto 图片的无网络计数。升级上游时按这四个 PR 逐项核对，已有同义实现即移除回移部分。这些计数修复已随 patch.4 部署

2026-11-07 发布验收发现 `patch.3` 镜像缺少 Pillow，因此未部署该镜像。该依赖此前仅属于 CI 组，现补入 `proxy-runtime` 并增加镜像内图片解码检查，避免开发环境依赖掩盖运行时缺失。`patch.3` 不作为本次上线候选，标签保持不变

**请求上下文严格准入**：[request_admission callback](local_guardrails/README.md#request_admission-请求上下文准入)
- 复用候选部署过滤与发送前 hook，不再修改 Router 核心。支持异步 Chat Completions、Messages、Responses，包含直接 deployment ID、fallback 与已有路由自动压缩后的复核。按部署或目录中的 mode 排除明确非 Chat 模型，避免 Router 为转写、重排生成的占位参数误触发准入，未知类型仍严格检查
- 使用部署已有 `model_info.max_input_tokens`，例如设为 `1000000`，在估算值达到上限时拒绝。计数失败、计数关闭或部署无可用上限均不继续发送，损坏内嵌图片在本地返回字段路径，使用已有 tokenizer 和受限计数线程
- 严格模式会拒绝无法本地核验的有效历史，包括服务端加密搜索结果和有状态 Responses 引用。它是可选运营策略，不改变上游默认兼容性。计数仍有模型与图片公式差异，阈值余量不能保证供应商绝不报超限。批量大图仍有 CPU 开销

1.108 已含 [#44544](https://github.com/BerriAI/litellm/pull/44544)，上游有意在计数失败时继续其他筛选，并不保证阻止超长输入，因此把严格策略放到外部插件。system/tools 和 Responses 基础计数已有 [#39663](https://github.com/BerriAI/litellm/pull/39663)、[#33706](https://github.com/BerriAI/litellm/pull/33706)，本次复用现有转换。原始超长请求的离线复放可在本地拒绝，没有发出付费调用；生产已按上述限额启用。正式镜像的断网 HTTP 验收确认超长输入与损坏图片均在本地返回 400，生产只读取健康和配置，未发起付费测试

历史工具名跨供应商不兼容暂不做全局重命名。[#47434](https://github.com/BerriAI/litellm/pull/47434) 和 [#42963](https://github.com/BerriAI/litellm/pull/42963) 分别涉及 guardrail 恢复和 Bedrock 长度，未确认覆盖此次同一问题。修改名称还需保证声明、调用和结果关联，以及不同目的供应商往返一致，不能用某一供应商的修复宣称通用兼容

**2026-11-09 已部署补丁，Claude 上游计数与加密历史兼容**

以下远程计数行为为历史记录，2026-11-13 已取消生成前自动计数，当前行为见本文开头和[插件说明](local_guardrails/README.md#request_admission-请求上下文准入)

2026-11-10 外部插件候选修复让准入计数使用实际生成的出站身份与功能头，避免计数独自透传原始 Claude Code 身份而生成仍使用 LiteLLM 身份。默认 User-Agent 继续由共享 HTTP 客户端提供，显式 headers、extra_headers 与供应商专属身份头保持一致。该修改已于 2026-11-13 随 1.109 升级启用。若要求消除 CPA 输入转换差异，需要配套修复并完成真实验收。2026-11-11 后续决定接受部分漏拦，沿用本文开头说明的边界。下面的客户端身份透传行为记录 patch.5 发布时状态

Opus、Sonnet、Haiku 按实际部署调用现有 Anthropic `count_tokens` 处理器，使用该部署的凭据和输入上限，计数失败不退回本地估算。原生 Messages 与缓存预测共用请求转换，Chat 复用生成路径的参数转换，包含 `additional_drop_params` 和输出预算对 thinking 的影响。保留真实客户端 User-Agent、X-App 和功能 beta，只发送一个 Content-Type，不转发客户端鉴权头，也不伪造客户端身份。无需修改 CPA，现有路由与 fallback 保持原配置

计数参数包含上下文管理、工具、结构化输出等输入。参考已合并的 [#47363](https://github.com/BerriAI/litellm/pull/47363)，计数时将旧版 `output_format` 归一到 `output_config.format`，保留其他输出配置，避免计数接口拒绝生成接口接受的格式。该 PR 的 Responses 转换复用、[#44784](https://github.com/BerriAI/litellm/pull/44784) 的压缩历史计数以及 [#46815](https://github.com/BerriAI/litellm/pull/46815) 的计数错误传播均可参考，但不能直接替代发送前准入。2026-11-09 核对上游 main，尚无完整同义实现，升级时需逐项核对再删除本地差异。仍未合并的 [#49764](https://github.com/BerriAI/litellm/pull/49764) 放宽非 OpenAI 估算阈值，与本插件的严格限额策略不同，不回移

Responses 正常的加密 reasoning、compaction 历史改为保留并交给上游校验，本地继续检查可读正文、摘要、指令与工具。密文不按字符数估算，也不宣称已完整计数。`previous_response_id`、`conversation`、`item_reference` 和未知输入项仍受原有无法核验策略限制。这是上述严格准入的明确例外，详细行为与验证记录见 [插件说明](local_guardrails/README.md#request_admission-请求上下文准入)

本批上线前，生产保持 patch.4 和回滚后的旧准入插件。曾两次仅更新插件后因实际客户端字段兼容问题回滚，不能把当时短文本验收当作发布依据。晚间 review 又修复了 Chat 计数参数漂移和旧版结构化输出计数问题，已从最终提交重新构建核心镜像并配套最新外置插件，禁止直接复用早先的候选镜像、暂存插件或 manifest。已有真实回放覆盖主渠道三系列生成及超限拒绝，两个备用渠道的生成权限故障在关闭插件时也存在，不能宣称所有渠道均正常

2026-11-09 已完成 patch.5 构建，源码固定为 `3b0a3d4a6e9`，镜像为 `leikaiwei/litellm:v1.108.0-fork.patch.5`，digest 为 `sha256:30a88b8e23a89cce022ed042c54259b25ab99376cf910237fea58b6c9a17a4ee`。[发布构建](https://github.com/leikaiwei/litellm/actions/runs/42623399610)成功，正式镜像与新插件的断网校验通过，包含源码哈希、图片依赖、参数删除、结构化输出、加密历史和三系列准入

同日 UTC+8 23:05 完成生产切换，镜像与配套插件一起更新，约 44 秒不可用。主代理健康、数据库连接、回调与源码哈希核对通过，其他五个容器、路由、限额和无关挂载保持不变。Opus、Sonnet、Haiku 及加密历史的八个普通和流式真实验收请求均返回目标模型，重试与降级均为 0，流式正常结束并返回用量。超限请求只在关闭该请求 fallback 时返回 400，保留原路由时会按既有策略降级，不能把严格准入描述成禁止全部跨模型回退。观察窗口另有 129 条普通用户请求成功，尚无普通用户 Claude 完成请求，不能将管理员回放算作自然流量。镜像和插件的成对回滚入口已保留

**PostgreSQL 空字节修复** — `proxy/utils.py`
- 清洗 spend logs 中的 `\x00` 空字节，避免 PostgreSQL jsonb 写入失败（22P05）

**DeepSeek V4 支持及兼容性修复** — `llms/deepseek/chat/transformation.py`、`llms/deepseek/messages/transformation.py`
- 注册 deepseek-v4-flash / deepseek-v4-pro 模型（1M input, 384K output），支持裸名路由
- 修复 thinking mode 多轮对话中 reasoning_content 缺失导致 API 400 的问题
- 修复 tool schema 中 Anthropic `type:"custom"` 未转换为标准 `"object"` 的问题
- 修复 Anthropic thinking_blocks 到 DeepSeek reasoning_content 的转换
- 修复 Anthropic 兼容端点 fallback（如 qwen/claude 回退到 deepseek）时，历史里外来 thinking 块导致 DeepSeek 400 的问题；命中该 400 时自动修复 assistant 历史（redacted_thinking 转占位 thinking 块、给含 tool_use 却无 thinking 块的消息注入占位块）并重试，直连场景不受影响

**Anthropic 流式适配器空 choices 防护** — `llms/anthropic/pass_through/adapters/transformation.py`
- 症状：Claude Code 等 `/v1/messages` 客户端流式调用 OpenAI 形状后端（如 `custom_openai`）时，内容能完整收到，但请求被记为错误且 tokens=0，日志里是 `IndexError: list index out of range`
- 根因：OpenAI 兼容后端会在流末尾发出 `choices: []` 的空帧，而适配器多处裸取 `choices[0]`。空帧有两种来源：OpenAI 规范里 `include_usage` 的 usage-only 尾帧（适配器对流式请求强制开启该选项），以及网关自定义的非标准帧（实测有后端在流末尾追加一个私有类型的计费帧，`usage` 为 null，真实 usage 挂在前一个 `finish_reason` 帧上）。崩溃发生在响应头已发出之后，所以状态码仍是 200，只是缺了 `message_delta` 和 `message_stop`
- 修复：空 choices 的 chunk 不丢弃，加判空守卫后继续走已有的 usage 合并路径，usage 仍能进 `message_delta`。同类问题在 Responses API 桥接层一并修掉（Azure 前导 `prompt_filter_results` 空帧也走这条路）
- 上游进展（1.99.1 同步时核对）：`adapters/streaming_iterator.py` 上游已自行修复（PR [#35314](https://github.com/BerriAI/litellm/pull/35314)，走 `_handle_choiceless_chunk` 在两条流式循环开头统一 `continue` 掉空帧，比逐点判空更彻底），我们在该文件里的三处判空已成死代码，整个文件改回上游实现。我们最初跟踪的 PR [#34455](https://github.com/BerriAI/litellm/pull/34455) 未被合并。`transformation.py` 的 `finish_reason` 裸取与 Responses 桥接层的三处裸取（`_is_reasoning_end`、`_ensure_output_item_for_chunk`、`_get_delta_string_from_streaming_choices`）上游仍未防护，且调用点也没有前置守卫，故这两个文件的补丁继续保留
- 1.100.1 同步时复核：`transformation.py` 的 `finish_reason` 裸取与 Responses 桥接层那三处仍未防护，调用点也仍无前置守卫，补丁原样保留
- 1.101.0 同步时复核：结论同上，上游这一版没有动这两处，补丁原样保留
- 1.103.1 同步时复核：我们提交的 PR [#34455](https://github.com/BerriAI/litellm/pull/34455) 已于 09-17 并入上游，覆盖的正是 Responses 桥接层那三处裸取，`responses/litellm_completion_transformation/streaming_iterator.py` 合并后与上游完全一致，该文件不再算补丁（上面「#34455 未被合并」是当时的状态）。`transformation.py` 的 `finish_reason` 判空上游仍没有，保留；但它的两个调用点都在 `_handle_choiceless_chunk` 之后，空帧到不了这里，现在只是纵深防护
- 1.104.0 同步时复核：结论同上。上游这一版在该文件里只加了 compaction 块的回译与 usage 组装的重写，`finish_reason` 判空仍没有，补丁原样保留

**OpenRouter OpenAI 系列模型兼容性修复** — `llms/openrouter/chat/transformation.py`
- 修复 Claude Code `Agent` tool schema 中 Anthropic `type:"custom"` 透传导致 OpenRouter 下游 OpenAI/Azure 模型 API 400 的问题
- 支持 Claude Code -> LiteLLM -> OpenRouter(OpenAI 系列模型) 链路正常调用

**Docker 自动发布** — `docker_release_auto.yml`
- release 或手动发布时，只构建生产服务器使用的 `linux/amd64` 镜像，推送 DockerHub 和 GHCR
- 保留一次构建，不恢复 ARM64、重复扫描构建或未使用的发布链路

**流式 tokens 统计修复** — `litellm_core_utils/streaming_handler.py`
- 症状：OpenAI 兼容后端流式调用时 tokens 严重偏低。实测同一 prompt 真实 87/18，客户端与日志只记 15/2；不带 `include_usage` 的那条路径更是直接记 0
- 根因：统计代码用 `"prompt_tokens" in usage` 判断字段存在，而只有 litellm 自己的 `Usage` 定义了 `__contains__`；OpenAI 兼容后端解析出的是 SDK 原生 `CompletionUsage`，`in` 退化为遍历 `(key, value)` 元组，恒为 False，于是真实 token 数被丢弃、回落到本地 `token_counter` 估算
- 影响面：所有走流式的客户端，不限于 Claude Code。tokens 是限流、配额与用量分析的依据，偏低会让这些全部失真
- 修复：改用 shape-aware 读取（dict 用 `.get()`，pydantic 模型用 `getattr`），三种 usage 形状均正确；后端确实未返回 usage 时仍保留估算兜底
- 上游同样未修，且无对应 issue / PR
- 1.104.0 同步时复核，按文件拆出两个结论。`streaming_chunk_builder_utils.py` 交还上游：上游 08-31 起（1.103.1 里就有，前两轮同步漏看了）在 `_usage_chunk_calculation_helper` 的唯一调用点前加了 `_extract_usage_chunk`，把原生 `CompletionUsage` 先归一成 litellm `Usage`，那个 `in` 判断从此成立，我们改成 `getattr` 的那段已是冗余。判据是把 fork 的回归测试原样放到纯上游 v1.104.0 上跑，`test_provider_native_completion_usage_is_not_replaced_by_estimate` 通过。`streaming_handler.py` 的 `calculate_total_usage` 上游仍是裸的 `"prompt_tokens" in usage`，同一轮对照里 `test_calculate_total_usage_with_provider_native_completion_usage` 在纯上游下失败（0 ≠ 87），补丁保留。两条回归测试都留着
- 上游 1.104 同时改了缺省语义：usage 帧明确报 0 就记 0，不再回落到本地估算，只有完全没有 usage 帧才估算

**Anthropic 适配层参数泄漏与 thinking-only 消息修复** — `llms/anthropic/pass_through/adapters/transformation.py`、`llms/anthropic/pass_through/adapters/handler.py`
- 前提：`/v1/messages` 打 OpenAI 形状后端（`custom_openai`）时要经 anthropic↔openai 双向适配层，原生 anthropic 端点那条路径上的既有补丁全部不参与。以下两个缺陷都用同一请求打原生 anthropic 端点做对照确认：原生 200、经适配层失败
- 症状一：请求带 `stop_sequences` / `mcp_servers` / `speed` / `cache_control` / `inference_geo` 任一参数时 500，报 `AsyncCompletions.create() got an unexpected keyword argument`
- 根因一：适配层 `translatable_anthropic_params()` 是反向白名单，名单外的 Anthropic 顶层参数被 `_copy_untranslated_anthropic_params` 原样拷进 `litellm.acompletion(**kwargs)`，一路漏到 OpenAI SDK。`drop_params: true` 拦不住——它只比对 `supported_openai_params`，这些名字压根不在表里。参数有两个注入点：命名参数经 `request_data` 进适配层，其余经 handler 的 `extra_kwargs` 回注，两处都要堵
- 修复一：无 OpenAI 对应的四个参数收进 `ANTHROPIC_ONLY_PARAMS_WITHOUT_OPENAI_EQUIVALENT`，并复用 handler 既有的 `ANTHROPIC_ONLY_REQUEST_KEYS` 机制堵住第二个注入点，清单保持单一来源。`stop_sequences` 原本由我们映射成 OpenAI 标准的 `stop`，上游 1.99.1 起自带了同语义的 `_translate_stop_sequences_to_openai`，1.99.1 那次合并把两份都留下了，于是方法定义、白名单项、调用点各重复一次（后定义胜出，语义相同故无实际故障）。1.100.1 同步时删掉我们那份，只留上游实现
- 症状二：assistant 历史消息只含 thinking 块时上游 400，且上游只回一句通用兜底错误串，不指出违规字段
- 根因二：`strip_empty_text_blocks_from_anthropic_messages` 先摘掉伴随的空 text 块（那是为原生 anthropic 路径加的，见上游 #22930），适配层随后落到 `assistant_content = assistant_message_str` 拿到 `None`，再被 `utils.py` 的 `cleanup_none_field_in_message` 连键一起删掉，产出既无 `content` 又无 `tool_calls` 的非法 OpenAI 消息。唯一变量法钉死了上游规则：只看 `content` 键存不存在，与其值无关、与 `tools` 无关
- 修复二：无 `content` 且无 `tool_calls` 的 assistant 消息补 `content: ""`。OpenAI 规范只允许带 `tool_calls` 时省略 `content`，故不影响工具调用那条路
- 四个无 OpenAI 对应的参数上游至今未处理，无对应 issue / PR；症状二上游同样未修
- 1.104.0 同步时复核：上游往 handler 的 `ANTHROPIC_ONLY_REQUEST_KEYS` 里加了 `safeguards`（它自己已在翻译层映射过的键），与我们那行冲突，解法是取并集 `{"output_config", "safeguards"} | ANTHROPIC_ONLY_PARAMS_WITHOUT_OPENAI_EQUIVALENT`。四个参数与症状二上游仍未处理，补丁保留
- 已知遗留（本轮未修）：`top_k` 同样会漏成 500，但 vertex_ai 等 provider 真支持它，而适配层此处拿不到解析后的 provider（`model` 是 model_group 名），无条件丢弃会让 gemini 等后端静默失去该参数，需按 provider 能力决定去留；`stop_reason` 缺 `stop_sequence` 与 `content_filter` 映射；`count_tokens` 漏算 system 与 tools（两条路径都中）；历史轮 thinking 以非标准 `thinking_blocks` 字段发给 `custom_openai`，未转成 `reasoning_content`（fork 里那个转换挂在 `DeepSeekChatConfig` 上，provider 为 `custom_openai` 时拿到的是 `OpenAILikeChatConfig`，补丁不参与），多轮会丢失上一轮推理链

**deployment 级 guardrail 在 `/v1/messages` 上不执行** — `integrations/custom_guardrail.py`
- 症状：挂在 deployment `litellm_params.guardrails` 上的 guardrail，在模型组 fallback 落到该 deployment 时完全不执行。表现为带图请求打主模型组、退到挂了识图 guardrail 的纯文本 deployment 时仍然报错，而直呼同一个 deployment 一切正常
- 根因：`async_pre_call_deployment_hook` 的 call_type 门只认 `completion` / `acompletion`，而 `/v1/messages` 原生直通的 call_type 是 `anthropic_messages`（`utils.py` 取被装饰函数名，再由 `CallTypes()` 转成枚举）。proxy 级 `pre_call_hook` 又只按**请求里的组名**取 guardrail 并集，且 fallback 只在 router 内部重试、不重跑 proxy 级钩子。两者叠加，这个 deployment 钩子是 fallback 路径上唯一的机会，却被门挡住
- 修复：门改为查 `_DEPLOYMENT_PRE_CALL_TYPES` 映射表，加入 `anthropic_messages`，同时把 call_type 到下游字面量的转换收敛到该表（原先是内联三元表达式）。`CallTypesLiteral` 本就含这个值，未放宽到 embedding / responses 等无 messages 可改写的 call_type
- 实测：本地 proxy 构造真实组级 fallback（A 组指向不可达地址必失败 -> B 组为真实纯文本模型 + 识图 guardrail），补丁前后同一请求对照。带图 fallback 补丁前 500（surface 出 A 组的连接错误，即 `Error doing the fallback` 形态），补丁后 200 且 `input_tokens` 与直呼对照逐一致（125），纯文本模型答出图片真实颜色即证明描述已注入；嵌在 `tool_result` 里的图同样生效（178 vs 补丁前原样透传的 237）
- 上游未修，无对应 issue / PR
- 注意：`vision_model` 指向的组不能挂这个 guardrail。识图那次 `router.acompletion` 的 call_type 是 `acompletion`，天然穿过门，一旦该组自身也挂上就会无限递归，guardrail 里没有递归保护

**空回复不写响应缓存** — `caching/caching_handler.py`、`llms/anthropic/pass_through/messages/response_cache.py`
- 症状：客户端调用后长时间无输出，之后自动重试全部瞬时返回同样的空回复，会话彻底卡死，只能手动发一句"继续"才恢复
- 根因：上游偶发静默拒答，回 HTTP 200 加空 `content` 加 `finish_reason: stop`，litellm 判为 success（`attempted_retries: 0`、`error_information: null`，故 `num_retries` 与模型组 fallback 全部绕过）。这个空回复随后被写进响应缓存，默认 TTL 60 秒，于是窗口内每次重试都命中缓存秒回同一个空回复。生产实测同一 request_id 三条记录：首条真实请求耗时 181 秒，随后两条 `cache_hit=True` 各 0.01 秒返回，重试彻底失去意义
- 修复：`_should_store_result_in_cache` 加一条判空，所有 choices 都没有实质内容时不写缓存。判空覆盖 `content` 之外的 `tool_calls` / `function_call` / `reasoning_content` / `thinking_blocks` / `audio` / `images`，故纯工具调用与纯推理回复不受影响；空格等仍算内容，不猜测语义。该函数是 sync 与 async、流式与非流式四条路径的唯一决策点（流式经 `_add_streaming_response_to_cache` 汇入），改一处全覆盖
- 注意：这只让重试重新有意义，治不了空回复本身，那个根因在上游
- 1.99.1 同步时扩面：上游 1.99 新增了原生 `/v1/messages` 响应缓存，`anthropic_messages` / `aanthropic_messages` 直接进了 `DEFAULT_CACHING_SUPPORTED_CALL_TYPES`，等于默认打开。它自带的两道跳过判定只看「有没有 `message_stop`」和「是不是 error 帧」，而静默拒答回的正是一条格式完好、带 `message_stop`、也不是 error 的空流，两道都放行。原补丁挂在 `_should_store_result_in_cache` 上，够不着这条新路径，升级即等于把这个已修的生产 bug 重新打开。故补两处：流式在 `response_cache.py` 的 `_persist` 里按 SSE 形状判空（非 text 的内容块一律算内容，text 块要求真有 delta 文本），非流式把 `_is_contentless_result` 从只认 `ModelResponse` 扩到也认 Anthropic Messages 的 content 块数组，认不出的形状一律放行
- 1.100.1 同步时复核：上游 `_persist` 仍只有那两道判定，`_should_store_result_in_cache` 也没加判空，两处补丁原样保留。上游 1.100 新增的 `test_async_cache_write_completes_when_asyncio_run_closes_the_loop` 拿空 `ModelResponse()` 当样本，正好被本补丁判成空回复而不写缓存，测试跟着失败；它要测的是「写操作能挺过 event loop 关闭」，与内容无关，故把样本换成带 content 的 response，断言不变
- 1.101.0 同步时复核：上游 `_persist` 与 `_should_store_result_in_cache` 均无变化，两处补丁原样保留
- 1.103.1 同步时复核：结论同上，两处补丁原样保留
- 1.104.0 同步时复核：上游对 `caching_handler.py` 的改动全在 embedding 缓存（格式版本、部分命中合并），没有新增绕过 `_should_store_result_in_cache` 的写入路径；`_persist` 只是把 payload 收成一行。两处补丁原样保留。上游把测试从 `tests/test_litellm/caching/` 搬到 `tests/unit/caching/` 并大幅扩写，git 认不出是改名，我们的判空用例与那条换样本的修改已手工移植到新位置
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
- 上游未修：1.104.0 里 `_reset_expired_window` 仍是 `.replace(tzinfo=None)`，`reset_budget_windows` 仍是 `datetime.utcnow()`
- 补丁边界（1.100.1 同步时核对过，结论是不用扩大）：同文件里 `_reset_budget_for_litellm_keys_chunk` / `users_chunk` / `teams_chunk` 三处的 `datetime.utcnow()` 不是同一类错位。它们只有两个去向，一是 Prisma 的 where 过滤（`expires` / `budget_reset_at`），prisma-client-py 0.11.0 的 `builder.serialize_datetime` 明确把 naive 当 UTC 再打上 tzinfo，`utcnow()` 与 `now(timezone.utc)` 序列化出的查询字节完全一致；二是 `current_time` 传进 `_reset_budget_common`，而那个函数根本没读这个参数，新的 `budget_reset_at` 来自 `compute_budget_reset_at`，后者内部自己取 aware UTC。窗口那条路之所以真的错，是因为 `reset_at` 存在 JSON 里是**字符串**（带 `+08:00` 偏移），偏移一丢就变成当地墙钟；列字段那条路上 tzinfo 两头都由 Prisma 归一，从没被丢过
- 遗留隐患（未改）：上面那三处的正确性依赖「naive 即 UTC」这个 Prisma 契约，`datetime.utcnow()` 恰好满足。Python 3.12 起该方法被弃用，若有人顺手改成 `datetime.now()`，在 UTC+8 上会静默变成 8 小时偏差且无测试拦截；安全的现代化写法是 `datetime.now(timezone.utc)`

**Anthropic 能力头定向透传** — `proxy/litellm_pre_call_utils.py`
- 打 `/v1/messages` 时把客户端的 `anthropic-beta` / `anthropic-version` 透传给下游。Claude Code 的工具 / Agent 能力由 beta 头声明，不透传下游就当这些能力不存在。只放这两个非敏感能力头，不做通用头转发

**本地价格表补缺远端** — `litellm_core_utils/get_model_cost_map.py`
- 远端价格表拉下来后，把本地 `model_prices_and_context_window.json` 里远端还没有的条目补进去。远端优先，本地只补缺，故上游发布某个模型后自动以上游为准
- 需要它是因为 fork 里加了上游尚未收录的模型（如 `anthropic/deepseek-v4-pro`），不补的话联网拉表会把这些条目整个丢掉
- 1.103.1 同步时扩面：上游把远端加载重构成三条路径，启动时同步首拉（`get_model_cost_map`）、首拉遇瞬时错误后的后台重试（`_retry_remote_fetch_in_background`）、proxy 运行期重载（`refetch_model_cost_map`）。原补丁只挂在同步首拉上，另外两条一旦生效就会用纯远端表整体替换，把本地条目丢掉。现在三处统一走 `_finalize_remote_model_cost_map`，测试覆盖首拉与运行期重载两条入口
- 1.104.0 同步时复核：上游在 `_finalize_loaded_model_cost_map` 里加了一份「最近一次加载的目录」快照（`loaded_model_cost_map`，供模型管理接口读）。我们的 `_finalize_remote_model_cost_map` 是先补缺再调它，所以快照里同样带着本地条目，不用改

1.108.0 同步核对结论（2026-11-05）：以正式发布标签为基线，保留现有 fork 补丁。主线已合并的工具数组计数修复仍未进入该标签，继续保留回移。空回复缓存判定保留新版缓存目标追踪和 provider 排除规则；deployment guardrail 保留新版执行标记的 `finally`，继续覆盖 Messages。新增的 `_provider_reported_cost` 分支同样遵守 `always_use_local_pricing`，避免绕过本地定价。DeepSeek HTTP 错误重试 override 同步接收新版 `request_body` 参数。旧补丁按新版类型和不可变规则收敛，不调整 lint 或类型预算

上游新增 `general_settings.safeguards_classifier_model`，可调用指定模型完成服务端工具审核，已不只是透传；未配置该选项时不会启用这条模型审核路径

105.1 到 108.0 新增 25 份数据库迁移，没有修改旧迁移，没有 SpendLogs DDL 或迁移时全表数据重写。包含 ManagedFileTable、Lens 等表的索引和新表，晚间切换仍需预留迁移时间。发布源码和镜像不等于生产部署，自定义 hook 需要另行注册和挂载

1.105.1 同步核对结论：同步 105 系列最新稳定补丁版，包含 105.0 的模型用量聚合锁争用与 Responses 桥接重复日志修复。现有生产补丁上游仍未吸收，全部保留。Anthropic 目录从 `experimental_pass_through` 改为 `pass_through`，生产补丁、自有测试与 thinking-switch 测试随之迁移。空回复缓存判定保留上游新的 call-type 规则与后台写入机制，回归测试等待后台写入后再检查缓存；两条上游缓存生命周期测试改用非空样本，仍验证原来的写入时序。deployment guardrail 同时保留我们的 `anthropic_messages` 支持和上游从 metadata 提取鉴权信息的新逻辑。旧 UI sentinel 回归测试已由上游拆分后的 `test_user_api_key_auth_request_flow.py` 中三条同类用例覆盖，不再保留重复测试。DeepSeek Anthropic 镜像条目继续与本地表一起补缺远端价格表

105 新增六份数据库迁移，没有业务数据重写；SpendLogs 仅新增可空 `billing_agent_id`，仍需在部署窗口获取表锁。数据库初始化失败现在始终退出，`ENFORCE_PRISMA_MIGRATION_CHECK=false` 不能再绕过，`--no-enforce_prisma_migration_check` 已移除。SpendLogs 手工索引改为启动后后台并发补建。构建和推送镜像不代表生产部署，生产切换须另行执行

1.105.1 验证记录（2026-10-15）：补丁所在 18 个测试文件与两个 `local_` 目录共 1,703 项通过，扩大模块回归 9,384 passed、1 skipped。扩大回归用 UTC 运行，排除 macOS/Python MIME 映射缺少 `text/markdown` 的一条上游用例，以及上游 Realtime 测试文件：后者的旧 mock 未适配新的异步日志 dispatcher，原样运行有 24 项在日志 flush 超时，仅在测试进程内补齐 mock 后整文件 122 项通过，源码与测试文件均未改动。`make lint` 全部通过，未修改预算文件。`make check` 的 API-types 同步仍有上游基线漂移，纯 v1.105.1 在同一环境也生成完全相同的 1,097 行新增内容，保留上游生成文件。临时本地 proxy 调真实 DeepSeek V4 的七条 curl 请求全部 200，覆盖 chat/messages、原生/适配层、流式/非流式及 custom tool schema；流式包含结束事件与 token 用量。wheel 构建通过并核对包内补丁源码及 Anthropic 新路径，生产部署另行执行

1.104.0 同步核对结论：本轮没有补丁被上游新吸收，但查出一处早已冗余的（流式 tokens 统计在 `streaming_chunk_builder_utils.py` 里的那一半，见上），该文件交还上游，其余逐文件核对后全部保留。与上游有差异的文件 42 -> 41（不含 `local_` 目录）。上游这一版把 `tests/test_litellm` 的大部分测试搬到了 `tests/unit`，补丁测试跟着搬：多数由 git 按改名自动带过去，`test_caching_handler.py` 因上游同时大幅扩写而认不出改名，手工移植；fork 独有的两份根目录测试（`test_anthropic_messages_header_forwarding.py`、`test_model_cost_map_local_merge.py`）一并挪到 `tests/unit/`。`llms/deepseek/` 下两个文件上游本轮没有改动；`llms/openrouter/chat/transformation.py` 上游只换了 tokenizer 的类型标注。价格表与上游只差 `anthropic/deepseek-v4-pro` 这一条镜像，native 的 `deepseek/deepseek-v4-pro` 本轮没有调价，镜像仍未跟 `off_peak_pricing`（那是 DeepSeek 官方直连的计费规则）

1.109.0 同步核对结论（2026-11-12），按稳定发布标签合并。工具数组缺少 `items`、压缩摘要、搜索结果、Anthropic tool choice 和内嵌 auto 图片计数已由上游覆盖，删除对应回移实现，只保留上游尚未支持的服务端工具与搜索错误结果兼容，`token_counter.py` 相对上游缩减为 6 行差异。其他仍未被上游修复的缓存、定价、流式用量、DeepSeek、guardrail 和预算时区补丁继续保留

Anthropic 模块按上游移动到 `messages/native` 和 `messages/adapters`，准入复用新的公开功能头接口，在请求转换前推导功能头，保留真实出站身份与计数正文的一致性。共享计数参数继续覆盖 native 结构化输出和上下文管理，不重复实现上游已有的 Chat 计数转换。新版供应商 `usage.cost` 快捷计费也遵守 `always_use_local_pricing`，避免绕过本地单价

同步时复现了上游适配器对深层工具 schema 的输入污染，JSON 深拷贝失败后，附加工具参数会写回调用方的原 schema。仅将原地更新改为新字典合并，上游原有行为测试由失败变为通过，不调整测试断言或递归限制

108.0 到 109.0 新增 5 份数据库迁移，旧迁移未改，没有 SpendLogs DDL 或数据重写。两个并发索引涉及的既有表在上线前只读核验中均无数据，另有配置 revision trigger、账本和订阅账户新表。仍由启动时正常迁移，不绕过迁移检查

发布镜像固定为 `leikaiwei/litellm@sha256:d533be118fdf405b127ced8c87e3bbe76e57f9c37849a3588a94a90dc63f2ef2`，最终源码的 `make check BASE_REF=v1.109.0` 及相关回归通过。正式镜像的 25 个补丁模块和准入插件哈希与发布源码一致，Pillow、参数删除、结构化输出和加密历史检查通过。禁网容器内使用模拟上游，经实际 HTTP 入口验证三系列正常请求、超限拒绝、计数失败拒绝及跨模型 fallback。默认重试次数设为 3 时，两种拒绝均只计数一次，没有 Claude 生成请求

2026-11-13 UTC+8 01:47 完成生产切换，探测到约 138 秒不可用。5 项迁移全部成功，readiness healthy、数据库 connected，回调和 guardrail 注册齐全，usage-api 查询正常。六个 Claude 部署在数据库和运行 API 中均为 950,000，运行容器内注入计数的边界检查确认 949,999 放行、950,000 拒绝。模型其他字段、全局配置和无关挂载未变，其余四个基础容器未重启。管理员页面及静态资源通过直连检查，公网健康入口返回 200。旧镜像、插件、限额快照和成对回滚入口已保留，本次未发起付费模型或长历史真实供应商验收

已移除的补丁（保留记录，便于回溯）：

- ~~**Anthropic passthrough 非标准 SSE 帧健壮性**~~ — 我们提交的 PR [#26000](https://github.com/BerriAI/litellm/pull/26000) 已并入上游，本地补丁移除
- ~~**UI 会话 team sentinel 被当成已删除团队**~~ — 上游 1.99 已在 `_token_can_vouch_for_team` 里加了同语义的 `UI_TEAM_ID` 豁免（`UI_TEAM_ID` 与我们用的 `UI_SESSION_TOKEN_TEAM_ID` 同为 `litellm-dashboard`），判据与我们那版一致，本地补丁移除，该文件已与上游完全一致
- ~~**Responses 桥接层空 choices 防护**~~ — `responses/litellm_completion_transformation/streaming_iterator.py` 里 `_is_reasoning_end`、`_ensure_output_item_for_chunk`、`_get_delta_string_from_streaming_choices` 三处判空。我们提交的 PR [#34455](https://github.com/BerriAI/litellm/pull/34455) 已并入上游，1.103.1 起该文件与上游一致。回归测试 `test_empty_choices_streaming_iterator.py` 保留
- ~~**流式 tokens 统计在 chunk builder 里的那一半**~~ — `litellm_core_utils/streaming_chunk_builder_utils.py` 的 `_usage_chunk_calculation_helper` 改 `getattr` 读取。上游自 08-31 起在调用前统一归一成 litellm `Usage`，补丁冗余，1.104.0 起该文件与上游一致。回归测试保留
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

An independent `litellm-core` distribution provides the Python SDK with the same
`import litellm` API and runtime dependencies. It has no optional extras, CLI entry
points, or bundled dashboard. Install one SDK distribution per environment because
`litellm` and `litellm-core` own overlapping Python files. Use `litellm` for the
proxy, CLI, and optional extras

Build and install core from this checkout while its release integration is pending:

```shell
python scripts/build_core_distribution.py --out-dir dist/core
python -m pip install dist/core/litellm_core-*.whl
```

The builder requires Git, `uv` and the Rust build toolchain. It reads the release version
from the root `pyproject.toml`, leaves source files unchanged, and produces a wheel
and a self-contained sdist in `dist/core`. Run installation in a fresh environment
without `litellm`

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

</details>

<details>
<summary><b>Auto Routing</b> - Choose a model for each prompt (Python SDK + AI Gateway)</summary>

### Python SDK

[**Docs: Autorouter SDK**](https://docs.litellm.ai/docs/autorouter_sdk)

Install with `uv add litellm` and set `OPENAI_API_KEY` for the models below.

```python
import asyncio
from litellm import Router

router = Router(model_list=[
    {"model_name": "fast", "litellm_params": {"model": "openai/gpt-6-luna"}},
    {"model_name": "capable", "litellm_params": {"model": "openai/gpt-6.1-sol"}},
    {"model_name": "auto", "litellm_params": {
        "model": "auto_router/complexity_router",
        "complexity_router_config": {
            "classifier_type": "heuristic",
            "tiers": {"SIMPLE": "fast", "MEDIUM": "fast",
                      "COMPLEX": "capable", "REASONING": "capable"},
            "return_raw_model_name": True,
        },
        "complexity_router_default_model": "capable",
    }},
])
response = asyncio.run(router.acompletion(
    model="auto", messages=[{"role": "user", "content": "What is 2+2?"}],
))
print(response.model, response.choices[0].message.content)
```

### AI Gateway (Proxy Server)

[**Create an Auto Router**](https://docs.litellm.ai/docs/auto_router/setup), then call its model name. Replace `auto` and the URL below with yours, and set `LITELLM_API_KEY` to your virtual key.

```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

</details>

<details>
<summary><b>A2A</b> - Invoke A2A Agents (Python SDK + AI Gateway)</summary>

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
headers = {"Authorization": "Bearer <your-master-key>"}    # LiteLLM master key or a virtual key

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
from litellm import mcp_client
import litellm

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # Load MCP tools in OpenAI format
        tools = await mcp_client.load_mcp_tools(session=session, format="openai")

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
  -H 'Authorization: Bearer <your-master-key>' \
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
        "x-litellm-api-key": "Bearer <your-master-key>"
      }
    }
  }
}
```

For MCP OAuth, an upstream may advertise dynamic client registration but refuse requests with HTTP 401 or 403. If the provider requires a pre-registered OAuth app, configure its `credentials.client_id` and, when required, `credentials.client_secret` on the MCP server. This skips dynamic registration in the gateway sign-in flow. The provider must approve the app for MCP access; reaching its authorization page does not establish that login or tool calls will succeed

[**Docs: MCP Gateway**](https://docs.litellm.ai/docs/mcp)

</details>

<details>
<summary><b>Agents</b> - Run Claude Code, Codex, OpenCode, pi or Deep Agents on any model (Python SDK)</summary>

### Python SDK - Agents

```python
import litellm
from litellm import Harness, sandbox

result = litellm.agent(
    Harness.CLAUDE_CODE,  # or Harness.CODEX, Harness.OPENCODE, Harness.PI, Harness.DEEPAGENTS
    "Find why tests/test_router.py is flaky and fix it.",
    sandbox=sandbox.local("./repo"),
    model="litellm_proxy/claude-sonnet-4-5",  # a model group on your AI Gateway
)

print(result.text, result.cost, [f.path for f in result.files])
```

Set `LITELLM_PROXY_API_BASE` and `LITELLM_PROXY_API_KEY` and every model call the agent makes goes through your AI Gateway, tagged `harness,claude_code`. Drop the `litellm_proxy/` prefix to call a provider directly. Install `starlette uvicorn` plus the agent's CLI (`claude`, `codex`, `opencode` or `pi`), or `deepagents langchain-litellm` for Deep Agents.

[**Docs: Agent Harnesses**](https://docs.litellm.ai/docs/harness)

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
| [CLF AI Gateway (`clf_ai_gateway`)](https://docs.litellm.ai/docs/providers/clf_ai_gateway) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Cloudflare AI Workers (`cloudflare`)](https://docs.litellm.ai/docs/providers/cloudflare_workers) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Codestral (`codestral`)](https://docs.litellm.ai/docs/providers/codestral) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Cognition (`cognition`)](https://docs.litellm.ai/docs/providers/cognition) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Cohere (`cohere`)](https://docs.litellm.ai/docs/providers/cohere) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |
| [Cohere Chat (`cohere_chat`)](https://docs.litellm.ai/docs/providers/cohere) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [CometAPI (`cometapi`)](https://docs.litellm.ai/docs/providers/cometapi) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |
| [CompactifAI (`compactifai`)](https://docs.litellm.ai/docs/providers/compactifai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [CoralBricks (`coralbricks`)](https://docs.litellm.ai/docs/providers/coralbricks) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Custom (`custom`)](https://docs.litellm.ai/docs/providers/custom_llm_server) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Custom OpenAI (`custom_openai`)](https://docs.litellm.ai/docs/providers/openai_compatible) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ | ✅ | ✅ |  |
| [Dashscope (`dashscope`)](https://docs.litellm.ai/docs/providers/dashscope) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  | ✅ |
| [Databricks (`databricks`)](https://docs.litellm.ai/docs/providers/databricks) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [DataRobot (`datarobot`)](https://docs.litellm.ai/docs/providers/datarobot) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Deepgram (`deepgram`)](https://docs.litellm.ai/docs/providers/deepgram) | ✅ | ✅ | ✅ |  |  | ✅ |  |  |  |  |
| [DeepInfra (`deepinfra`)](https://docs.litellm.ai/docs/providers/deepinfra) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Deepseek (`deepseek`)](https://docs.litellm.ai/docs/providers/deepseek) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Eden AI (`edenai`)](https://docs.litellm.ai/docs/providers/edenai) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |
| [ElevenLabs (`elevenlabs`)](https://docs.litellm.ai/docs/providers/elevenlabs) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ |  |  |  |
| [Empower (`empower`)](https://docs.litellm.ai/docs/providers/empower) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [eu/jev Decisions API (`eu_jev`)](https://docs.litellm.ai/docs/pass_through/eu_jev) |  |  |  |  |  |  |  |  |  |  |
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
| [Nace (`nace`)](https://console.nace.ai/docs/drex) |  |  |  |  |  |  |  |  |  |  |
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
| [Qianwen AI Platform (`qwen_ai_platform`)](https://docs.litellm.ai/docs/providers/qwencloud) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  | ✅ |
| [QwenCloud (`qwencloud`)](https://docs.litellm.ai/docs/providers/qwencloud) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  | ✅ |
| [Recraft (`recraft`)](https://docs.litellm.ai/docs/providers/recraft) |  |  |  |  | ✅ |  |  |  |  |  |
| [Replicate (`replicate`)](https://docs.litellm.ai/docs/providers/replicate) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Sagemaker Chat (`sagemaker_chat`)](https://docs.litellm.ai/docs/providers/aws_sagemaker) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Sail (`sail`)](https://docs.litellm.ai/docs/providers/sail) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Sambanova (`sambanova`)](https://docs.litellm.ai/docs/providers/sambanova) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [ScaleDown (`scaledown`)](https://docs.litellm.ai/docs/providers/scaledown) | ✅ |  |  |  |  |  |  |  |  |  |
| [Snowflake (`snowflake`)](https://docs.litellm.ai/docs/providers/snowflake) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Strands Decider (`strands_decider`)](https://docs.litellm.ai/docs/providers) |  |  |  |  |  |  |  |  |  |  |
| [Text Completion Codestral (`text-completion-codestral`)](https://docs.litellm.ai/docs/providers/codestral) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Text Completion OpenAI (`text-completion-openai`)](https://docs.litellm.ai/docs/providers/text_completion_openai) | ✅ | ✅ | ✅ |  |  | ✅ | ✅ | ✅ | ✅ |  |
| [Together AI (`together_ai`)](https://docs.litellm.ai/docs/providers/togetherai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Topaz (`topaz`)](https://docs.litellm.ai/docs/providers/topaz) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Triton (`triton`)](https://docs.litellm.ai/docs/providers/triton-inference-server) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Typesafe Decisions API (`typesafe`)](https://docs.litellm.ai/docs/providers) |  |  |  |  |  |  |  |  |  |  |
| [V0 (`v0`)](https://docs.litellm.ai/docs/providers/v0) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Vercel AI Gateway (`vercel_ai_gateway`)](https://docs.litellm.ai/docs/providers/vercel_ai_gateway) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [VLLM (`vllm`)](https://docs.litellm.ai/docs/providers/vllm) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Volcengine (`volcengine`)](https://docs.litellm.ai/docs/providers/volcano) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [VoyageAI by MongoDB (`voyage`)](https://docs.litellm.ai/docs/providers/voyage) |  |  |  | ✅ |  |  |  |  |  |  |
| [WandB Inference (`wandb`)](https://docs.litellm.ai/docs/providers/wandb_inference) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Watsonx Text (`watsonx_text`)](https://docs.litellm.ai/docs/providers/watsonx) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [xAI (`xai`)](https://docs.litellm.ai/docs/providers/xai) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| [Xinference (`xinference`)](https://docs.litellm.ai/docs/providers/xinference) |  |  |  | ✅ |  |  |  |  |  |  |
| [ZeroGPU (`zerogpu`)](https://docs.litellm.ai/docs/providers/zerogpu) | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |

[**Read the Docs**](https://docs.litellm.ai/docs/)

---

## Get Started

You can use LiteLLM through either the Proxy Server or Python SDK. Both give you a unified interface to access multiple LLMs (100+ LLMs). Choose the option that best fits your needs:

<table style={{width: '100%', tableLayout: 'fixed'}}>
<thead>
<tr>
<th style={{width: '14%'}}></th>
<th style={{width: '43%'}}><strong><a href="https://docs.litellm.ai/docs/simple_proxy">LiteLLM AI Gateway</a></strong></th>
<th style={{width: '43%'}}><strong><a href="https://docs.litellm.ai/docs/python_sdk">LiteLLM Python SDK</a></strong></th>
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

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.png)](https://ssh.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FBerriAI%2Fterraform-google-litellm&cloudshell_git_branch=v1.105.0&cloudshell_workspace=examples%2Fdefault&cloudshell_tutorial=TUTORIAL.md&cloudshell_image=gcr.io/ds-artifacts-cloudshell/deploystack_custom_image&shellonly=true)

Real 1-click. Opens Cloud Shell, clones the [terraform-google-litellm](https://github.com/BerriAI/terraform-google-litellm) module mirror at the latest stable release (a small download instead of the full litellm repo), and walks you through `terraform apply` via a built-in [DeployStack tutorial](./terraform/litellm/gcp/examples/default/TUTORIAL.md). Pick the project, the tutorial sets up the Artifact Registry remote repo, writes `terraform.tfvars` from your answers, and runs apply

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

All LiteLLM Docker images published to GHCR are signed with [cosign](https://docs.sigstore.dev/cosign/overview/). Releases are signed with the key introduced in [commit `e82a3c6`](https://github.com/BerriAI/litellm/commit/e82a3c646908ba0a6ee0beddc4310d7a6cac4065).

**Verify using the pinned commit hash (recommended):**

A commit hash is cryptographically immutable, so this is the strongest way to ensure you are using the published signing key:

```bash
cosign verify \
  --key https://raw.githubusercontent.com/BerriAI/litellm/e82a3c646908ba0a6ee0beddc4310d7a6cac4065/cosign.pub \
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
- ✅ **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/enterprise):**
- ✅ **Feature Prioritization**
- ✅ **Custom Integrations**
- ✅ **Professional Support - Dedicated discord + slack**
- ✅ **Custom SLAs**
- ✅ **Secure access with Single Sign-On**

# Contributing

We welcome contributions to LiteLLM! Whether you're fixing bugs, adding features, or improving documentation, we appreciate your help.

If you want to work closely with our team, check out [LiteLLM OSS Champions](https://www.litellm.ai/champions), our program for trusted OSS contributors with their own Slack channel and a voice in the roadmap

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
- **Ruff** for formatting, linting, and code quality
- **basedpyright** for type checking
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
