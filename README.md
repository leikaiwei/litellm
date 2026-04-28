# LiteLLM (Fork)

Fork 自 [BerriAI/litellm](https://github.com/BerriAI/litellm)，在上游基础上打了以下补丁：

**日志健壮性修复** — `anthropic_passthrough_logging_handler.py`
- 跳过上游厂商混入的 OpenAI 风格 `[DONE]` 控制帧
- 捕获非 JSON SSE 行的解析异常
- 确保任何上游非标准响应都不会中断日志写入

**PostgreSQL 空字节修复** — `proxy/utils.py`
- 清洗 spend logs 中的 `\x00` 空字节，避免 PostgreSQL jsonb 写入失败（22P05）

**Anthropic tool type 字段泄漏修复** — `anthropic/.../adapters/transformation.py`
- 过滤 Anthropic tool 顶层 `type` 字段（如 `"custom"`），防止泄漏到 OpenAI function parameters
- 使用 `deepcopy` 隔离 `input_schema`，避免原始数据被污染
- 修复 Claude Code → LiteLLM → DeepSeek（OpenAI 兼容接口）链路的请求被拒问题
- 注意：仍需在 LiteLLM Params 中配置 `"drop_params": true`

**Docker 自动发布** — `docker_release_auto.yml`
- tag/release 时自动构建多架构镜像推送 DockerHub 和 GHCR
