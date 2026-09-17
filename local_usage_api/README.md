# 虚拟 key 自查用量

个人 key 统一成 `llm_api_routes` 后调不了 litellm 的用量接口，而 `/user/daily/activity` 按 user_id 隔离、放开就全员互见。
这个独立小服务让调用方用自己的 key 查自己的今天与近 7 天用量，不改 litellm 源码，也不碰 DB

跑测试：

```bash
.venv/bin/python -m pytest local_usage_api/ -q
```

## 用法

```bash
curl -s -H "Authorization: Bearer sk-xxxxx" http://<host>:4100/usage
```

```json
{
  "as_of": "2026-09-17T13:45:00+08:00",
  "today": {"since": "2026-09-17T00:00:00+08:00", "requests": 1200, "tokens": 98000000, "spend": 12.3456},
  "last_7_days": {"since": "2026-09-11T00:00:00+08:00", "requests": 6800, "tokens": 520000000, "spend": 70.5}
}
```

- `today` 从中国时间当天 00:00 算起，`last_7_days` 是含今天在内的 7 个中国自然日
- 没带 `Authorization: Bearer` 直接拒绝，不会查 Prometheus；key 写错或从没用过，三项都是 0
- Prometheus 不可用返回 502

## 原理

litellm 上报的 Prometheus 指标带 `hashed_api_key` 标签，值就是 `sha256(明文 key)`，与 litellm 的 `hash_token`
以及 DB 里 `LiteLLM_VerificationToken.token` 相同（生产实测对上过）。服务收到明文 key 后本地算 hash，对下面三个指标各查一次
`sum(increase(<metric>{hashed_api_key="<hash>"}[<窗口>s]))`，两个窗口共 6 条并发发出

| 字段 | 指标 |
|---|---|
| `requests` | `litellm_proxy_total_requests_metric_total`（含失败请求） |
| `tokens` | `litellm_total_tokens_metric_total` |
| `spend` | `litellm_spend_metric_total` |

只接受明文 key，不接受直接传 hash：hash 在 Prometheus 里谁都看得到，收 hash 就等于谁都能查别人。
明文 key 只在本服务内存里算一次 hash，不会发给 Prometheus，也不进访问日志（放在请求头里而不是 URL）

## 口径

- 与 DB 日表 `LiteLLM_DailyUserSpend` 按整天对账，请求数、token、费用都差在 1% 以内（2026-09-15 全部 52 把 key）。
  单个 key 可能差很多：DB 会给失败请求记估算的输入 token，Prometheus 不记，所以这里的 token 更接近实际消耗
- token 包含缓存读取，与 litellm UI 同口径。上游网关如果把已含缓存的 `prompt_tokens` 直接当成 Anthropic 的 `input_tokens` 返回，这部分会多算大约一倍
- 费用是 litellm 的账面值，部分模型有已知漏算（0 费用配置被 provider 分发绕过、流式不计费），只能当参考
- `increase()` 看不到新序列的第一个样本，某个 key 第一次用某个模型的那次请求可能漏算
- Prometheus 保留 15 天，7 天窗口够用

## 性能

2026-09-17 在生产实测，单个 key 查 7 天窗口、3 个指标并发：普通 key 约 40ms；请求量最大的几把 key 冷查询 0.25 到 1.3 秒，
热查询 40 到 120ms。今天窗口都在几毫秒内

## 生产部署

直接复用 litellm 镜像：它自带 Python 3.13、fastapi、httpx、uvicorn，不用另外打镜像，也不用去拉新镜像。
把 `usage_api.py` 放到 `docker-compose.yml` 同目录，再加一个服务：

```yaml
  usage-api:
    image: leikaiwei/litellm:v1.99.1-fork
    restart: unless-stopped
    entrypoint: ["/app/.venv/bin/uvicorn", "--factory", "usage_api:build_app", "--app-dir", "/opt/usage_api", "--host", "0.0.0.0", "--port", "4100"]
    environment:
      PROMETHEUS_URL: http://prometheus:9090
    volumes:
    - ./usage_api.py:/opt/usage_api/usage_api.py:ro
    ports:
    - 4100:4100
```

然后 `docker compose up -d usage-api`，只会新建这一个容器，litellm 本身不受影响。
镜像标签写死，以后 litellm 升级不会连带改动这个服务

Prometheus 的 `9090:9090` 宿主机端口建议一起换掉。Grafana 走 docker 内网 `http://prometheus:9090` 访问 Prometheus，本服务也是，
都不依赖宿主机端口。换端口要重建 prometheus 容器，数据在命名卷 `litellm_prometheus_data` 里不会丢。
compose 里的 `image: prom/prometheus` 没写版本，重建时别先 `pull`，否则会顺带升级 Prometheus
