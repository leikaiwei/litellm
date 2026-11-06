# Docker Development Guide

This guide provides instructions for building and running the LiteLLM application using Docker and Docker Compose.

> **Just want to run LiteLLM?** This guide builds from source. To run the published
> image instead, use `docker-compose.quickstart.yml` in this directory — the
> two-service stack (gateway + Postgres) that the
> [Docker quickstart](https://docs.litellm.ai/docs/proxy/docker_quick_start) documents:
>
> ```bash
> curl -sSLO https://github.com/BerriAI/litellm/raw/main/docker/docker-compose.quickstart.yml
> printf 'LITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\n' "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > .env
> docker compose -f docker-compose.quickstart.yml up -d
> ```

## Fork 发布流程

本 fork 的部署服务器使用 x86_64，`.github/workflows/docker_release_auto.yml` 只构建 `linux/amd64`，不构建 ARM64，也不启用 QEMU

发布 Release 或手动运行该工作流时，使用指定标签的源码，只构建一次并推送 GHCR。配置 Docker Hub 凭据时，同一次构建也推送 Docker Hub。两个仓库均写入版本标签和 `latest`，镜像通过 `LITELLM_RELEASE_TAG` 记录发布版本

补发已有版本时，在 Actions 中选择主分支上的工作流并填写原始发布标签，避免使用旧标签内的工作流。镜像发布成功后再执行服务器升级，工作流不会自动部署生产

### 请求准入补丁 v1.108.0-fork.patch.1

2026-11-06 的补丁保留客户端协议与模型路由。Claude Code 可以继续使用 `/v1/messages` 调用配置为 OpenAI provider 的模型，LiteLLM 负责协议转换

启用 `enable_pre_call_checks` 且部署有输入上限时，计数异常会明确返回本地 400，不再放行。内嵌 PNG/JPEG/WebP/GIF 会校验文件结构和像素解码，损坏图片返回具体字段路径，不下载远端图片 URL。上述校验在计数工作线程执行。本地确定性校验失败不重试，也不通过 fallback 绕过

工具 JSON Schema 的整数约束（例如 `minItems: 1.0`）规范成整数 `1`，只遍历 schema 节点，不修改 `default`、`examples`、`const` 或真实小数。GPT Chat/Responses 的非法历史工具名称在发往上游前返回 400，保留错误字段路径，不擅自改名而破坏调用与结果的对应关系

部署设置需要单独保留。Gemini 各部署及完整可达 fallback 链使用不超过 `model_info.max_input_tokens: 1000000` 的限制，其他模型保留各自更小的输入限制。动态路由无法解析实际窗口时，100 万只作为网关绝对上限，仍可能先触及供应商更小的窗口。仅对不支持 Responses `metadata` 的兼容上游设置 `additional_drop_params: [metadata]`，不应全局删除 Anthropic `metadata.user_id` 或本地日志 metadata

本地 token 数仍是估算，100 万是保守准入阈值，不保证与供应商计数完全一致。补丁不会增加供应商容量或账号配额，也不能修复远端 URL 指向的图片。服务端真正的上下文压缩仍先压缩、再验证

发布前回归覆盖三入口（Chat Completions、Anthropic Messages、Responses）的超限与坏图，以及 Claude Code 到 GPT 的格式转换。通过本地受控上游验证拒绝请求的上游调用次数为零，再发布新标签。生产部署使用新标签或其 digest，保留旧镜像与数据库配置备份用于回滚

## Prerequisites

- Docker
- Docker Compose

## Building and Running the Application

To build and run the application, you will use the `docker-compose.yml` file located in the root of the project. This file is configured to use the `Dockerfile.non_root` for a secure, non-root container environment.

### 1. Set the Master Key

The application requires a `LITELLM_MASTER_KEY` for signing and validating tokens. You must set this key as an environment variable before running the application.

Create a `.env` file in the root of the project and add the following line:

```
LITELLM_MASTER_KEY=your-secret-key
```

Replace `your-secret-key` with a strong, randomly generated secret.

### 2. Build and Run the Containers

Once you have set the `LITELLM_MASTER_KEY`, you can build and run the containers using the following command:

```bash
docker compose up -d --build
```

This command will:

-   Build the Docker image using `Dockerfile.non_root`.
-   Start the `litellm`, `litellm_db`, and `prometheus` services in detached mode (`-d`).
-   The `--build` flag ensures that the image is rebuilt if there are any changes to the Dockerfile or the application code.

### 3. Verifying the Application is Running

You can check the status of the running containers with the following command:

```bash
docker compose ps
```

To view the logs of the `litellm` container, run:

```bash
docker compose logs -f litellm
```

### 4. Stopping the Application

To stop the running containers, use the following command:

```bash
docker compose down
```

## Embedded LiteAdmin MCP

Source builds containing embedded LiteAdmin MCP can serve it at `/admin/mcp` on the existing LiteLLM port. This capability is unreleased. Keep your existing database, master key, and proxy configuration, then add these settings to the serving container's environment:

```bash
LITELLM_ENABLE_ADMIN_MCP=true
LITELLM_LICENSE="your-enterprise-license"
PROXY_BASE_URL=https://gateway.example.com
```

For the unified source deployment described above, put them in its `.env` file and rebuild:

```bash
docker compose up -d --build
```

In componentized deployments, set the flag and license on the backend container and route `/admin/mcp` to the backend service. The gateway component excludes this endpoint. The unified, database, non-root, and backend image builds bundle the connector

Hosting is disabled by default. Opting in requires a valid base Enterprise license; an unlicensed opt-in or invalid flag value prevents startup. Enabling it reserves `/admin`, so rename any MCP server alias called `admin` first

With native key authentication, connect with a personal proxy-admin bearer key. When `enable_oauth2_proxy_auth` is enabled, the existing trusted-proxy identity headers select the user instead; the MCP bearer is required by the connector but does not select the native user. The resolved user must have the stored `proxy_admin` role, and `trusted_proxy_ranges` applies to the original caller's direct peer

Embedded responses default to `full`; selecting `LITELLM_ADMIN_RESPONSE_VIEW=compact` requires subsequent saved-result reads to reach the same worker process, including within a multi-worker pod

See the [LiteAdmin MCP guide](https://docs.litellm.ai/docs/proxy/liteadmin_mcp#run-liteadmin-mcp-inside-litellm) for client configuration, tool restrictions, and verification

## Hardened / Offline Testing

To ensure changes are safe for non-root, read-only root filesystems and restricted egress, always validate with the hardened compose file:

```bash
docker compose -f docker-compose.yml -f docker-compose.hardened.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.hardened.yml up -d
```

This setup:
- Builds from `docker/Dockerfile.non_root` with Prisma engines and Node toolchain baked into the image.
- Runs the proxy as a non-root user with a read-only rootfs and only writable tmpfs mounts:
  - `/app/cache` (Prisma/NPM cache; backing `PRISMA_BINARY_CACHE_DIR`, `NPM_CONFIG_CACHE`, `XDG_CACHE_HOME`)
  - `/app/migrations` (Prisma migration workspace; backing `LITELLM_MIGRATION_DIR`)
- Pre-builds and serves the admin UI from read-only paths:
  - `/var/lib/litellm/ui` (pre-restructured Next.js UI with `.litellm_ui_ready` marker)
  - `/var/lib/litellm/assets` (UI logos and assets)
- Routes all outbound traffic through a local Squid proxy that denies egress, so Prisma migrations must use the cached CLI and engines.

You should also verify offline Prisma behaviour with:

```bash
docker run --rm --network none --entrypoint prisma ghcr.io/berriai/litellm:main-stable --version
```

This command should succeed (showing engine versions) even with `--network none`, confirming that Prisma binaries are available without network access.

## Troubleshooting

-   **`build_admin_ui.sh: not found`**: This error can occur if the Docker build context is not set correctly. Ensure that you are running the `docker-compose` command from the root of the project.
-   **`Master key is not initialized`**: This error means the `LITELLM_MASTER_KEY` environment variable is not set. Make sure you have created a `.env` file in the project root with the `LITELLM_MASTER_KEY` defined.
