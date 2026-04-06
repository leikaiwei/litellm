# 当前仓库的最小发布流程

本仓库 CI/CD 调整为“最小可用发布链路”：

1. 手动创建 GitHub Release（会自动打 tag）
2. Release 发布后自动构建并推送 Docker 镜像（Docker Hub + GHCR）

不再保留 PyPI 发布工作流。

---

## 1) 创建 Release

工作流：`Create Release`  
触发方式：`workflow_dispatch`

输入参数：
- `tag`：发布标签（例如 `v1.0.1`）
- `target`：目标分支或提交（默认 `main`）

说明：
- 自动生成 Release Notes
- `tag` 需要符合 `vX.Y.Z`（可带后缀如 `-rc.1`）

## 2) 自动发布 Docker 镜像

工作流：`Auto Publish Docker & GHCR`  
触发方式：Release `published`

镜像推送目标：
- GHCR：`ghcr.io/${{ github.repository }}`
- Docker Hub：`${{ vars.DOCKERHUB_REPOSITORY || github.repository }}`
  - 仅当配置了 `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN` 时启用
  - 未配置时自动跳过（不会导致工作流失败）

镜像标签：
- `${release_tag}`
- `latest`

---

## 需要提前配置的 Secrets / Variables

### Secrets
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### Variables（可选）
- `DOCKERHUB_REPOSITORY`（不配置则默认使用 `owner/repo`）
