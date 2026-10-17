# 基础日志永久保留，正文保留90天

`spend_log_body_retention.py` 是宿主机维护任务，不改变代理写入路径，也不需要重建镜像或重启代理

官方 `maximum_spend_logs_retention_period` 会删除整条 SpendLogs。使用本任务前先将该项设为 `null`，保持 `store_prompts_in_spend_logs=true`。不要同时启用官方整行保留策略，否则基础费用和token明细仍可能被删除

## 清理范围

任务只更新早于滚动90天的 `messages`、`response`、`proxy_server_request` 和明确含正文的metadata路径，不删除任何请求行。请求时间、费用、所有基础token、模型、身份、状态和其他非目标列保持不变，包括 `updated_at`

历史响应中的 `usage` 和 `message.usage` 保留。metadata中的usage、费用拆分、缓存token、压缩节省统计和未知键保留，只剥离请求副本、guardrail正文、错误回显、工具参数与结果、检索文本和评估理由。SQL NULL与JSON null保持原值，合法历史JSON文本按原类型保存；无法解析的metadata保留并报告，不把它当空对象

每批事务锁定快照，更新后逐行比较所有基础字段，任一变化即回滚。主键与基础投影哈希在更新前写审计并fsync，成功后追加提交记录。审计、游标和可选首批备份只允许root读取，不打印正文或凭据

## 执行与恢复

依赖宿主机Python标准库及Docker CLI，通过现有PostgreSQL容器中的psql连接，不安装系统依赖。先预览一批，确认数据库和保留期后显式使用 `--apply`

```bash
python3 scripts/spend_log_body_retention.py \
  --state-file /root/litellm/maintenance/body-retention/state.json

python3 scripts/spend_log_body_retention.py --apply \
  --state-file /root/litellm/maintenance/body-retention/state.json \
  --batch-size 50 --run-budget-seconds 300 --sleep-seconds 1 \
  --min-free-gib 64
```

默认连接 `litellm_db` 容器的 `litellm` 库，用户为 `llmproxy`，可用 `--database`、`--container`、`--db-user`指定隔离测试环境。单批statement_timeout为15秒，lock_timeout为1秒，最多1000行一批。运行时间、批次和磁盘空间均有边界，达到边界保留进度，失败批次不推进游标

手动运行与定时任务必须复用同一state文件，文件锁会阻止重入。不要在任务运行中替换脚本或删除状态文件

未完成一轮时沿用原截止时间与游标。完成后下一轮只扫描上次截止时间前一天的重叠范围和新过期记录，避免每次重扫全部永久基础记录。超过一天的迟到记录或历史回灌，需要暂停常态任务后用独立新state完整补扫，再恢复常态任务

首批验证可增加 `--backup-dir /root/litellm/maintenance/body-retention-canary`。该目录使用0700、文件0600，压缩原始正文在更新前落盘；启用备份时清理超过7天的同名前缀备份，并限制总量10GiB。常态清理不重复备份已过期正文，避免把正文换一个目录永久保存。若首批后不再传该参数，部署者须另设7天到期清理

## 定时运行

使用独立systemd oneshot service及timer，显式指定Asia/Shanghai低峰时段。业务容器不重启。不要设置 `RemainAfterExit=yes`，以免后续timer不能再次执行

timer设置 `Persistent=false`，错过低峰时不在白天补跑，下次低峰继续。服务的运行预算应大于每日新增过期数据与一天重叠范围所需时间，历史积压则在监控下单独分批回填，不能把完成部署等同于历史已清完

随附service与timer模板使用北京时间每日04:00、100条一批、每批间隔1秒、每次最多30分钟，以及64GiB磁盘保护水位。先确认目标路径与策略，获准后在宿主机安装

```bash
install -d -m 700 /root/litellm/maintenance/body-retention /root/litellm/maintenance/body-retention-canary
install -m 700 scripts/spend_log_body_retention.py /root/litellm/maintenance/spend_log_body_retention.py
install -m 644 scripts/litellm-spend-log-body-retention.service /etc/systemd/system/
install -m 644 scripts/litellm-spend-log-body-retention.timer /etc/systemd/system/
systemd-analyze verify /etc/systemd/system/litellm-spend-log-body-retention.service /etc/systemd/system/litellm-spend-log-body-retention.timer
systemctl daemon-reload
systemctl enable --now litellm-spend-log-body-retention.timer
```

停止任务时只停用这个timer和service，保留整行保留期为null。代理升级时备份并保留维护脚本、两个unit、游标与审计文件，不恢复会删除基础明细的旧90d设置

只清正文会产生UPDATE、WAL和死元组，普通vacuum主要让空间供数据库内部复用，不保证操作系统文件立即缩小。不要顺手执行VACUUM FULL、全表UPDATE或在启动migration中回填数据

## 测试

使用独立且名称以 `test_body_retention_` 开头的PostgreSQL测试库，所有数据为合成fixture。测试逐行比较基础字段与计费metadata，覆盖边界、历史JSON文本、首批备份、事务回滚、幂等、续跑和锁

```bash
RETENTION_TEST_DATABASE=test_body_retention_example \
RETENTION_TEST_SSH_HOST=proxy-host \
  .venv/bin/python tests/integration/spend/test_spend_log_body_retention.py
```

[官方整行清理](https://docs.litellm.ai/docs/proxy/spend_logs_deletion)、[正文写入设置](https://docs.litellm.ai/docs/proxy/ui_spend_log_settings)、[PostgreSQL JSON操作](https://www.postgresql.org/docs/16/functions-json.html)
