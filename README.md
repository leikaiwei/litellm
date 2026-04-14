# LiteLLM (Fork)

Fork 自 [BerriAI/litellm](https://github.com/BerriAI/litellm)，在上游基础上打了以下补丁：

**日志健壮性修复** — `anthropic_passthrough_logging_handler.py`
- 跳过上游厂商混入的 OpenAI 风格 `[DONE]` 控制帧
- 捕获非 JSON SSE 行的解析异常
- 确保任何上游非标准响应都不会中断日志写入

**Docker 自动发布** — `docker_release_auto.yml`
- tag/release 时自动构建多架构镜像推送 DockerHub 和 GHCR
