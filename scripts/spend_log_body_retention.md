# 永久保留基础账单，按期清理正文

`spend_log_body_retention.py` 是宿主机维护任务，不改变代理写入路径，也不需要重建镜像或重启代理

官方 `maximum_spend_logs_retention_period` 会删除整条 SpendLogs。使用本任务前先将该项设为 `null`，保持 `store_prompts_in_spend_logs=true`。不要同时启用官方整行保留策略，否则基础费用和token明细仍可能被删除

## 清理范围

随附服务按请求的 `startTime` 保留 30 天，只更新已到期的 `messages`、`response`、`proxy_server_request` 和明确含正文的metadata路径，不删除任何请求行。请求时间、费用、所有基础token、模型、身份、状态和其他非目标列保持不变，包括 `updated_at`

历史响应中的 `usage`、`message.usage` 和 metadata 中的 `usage_object`、`additional_usage_values` 仅移除 `attribution`。metadata中的usage、费用拆分、缓存token、压缩节省统计和未知键保留，只剥离请求副本、guardrail正文、错误回显、工具参数与结果、检索文本和评估理由。SQL NULL与JSON null保持原值，合法历史JSON文本按原类型保存；无法解析的metadata保留并报告，不把它当空对象

候选批次读取行版本，大正文字段通过 `pg_column_size` 判断是否需要清空。每批在更新前锁行并复核行版本，更新后逐行比较全部基础字段，任一变化即回滚。主键与基础投影哈希在更新前写审计并fsync，成功后追加提交记录。审计、游标和可选首批备份只允许root读取，不打印正文或凭据

## 执行与恢复

宿主机只需要 Python 标准库和 Docker CLI。换表时先排空写入，记录新表开始接收请求的实际时刻，保留时区。初始化新目录后，用相同的参数预览一批。下例中 `CUTOVER_AT` 由操作人员设置

```bash
python3 scripts/initialize_retention.py --cutover "${CUTOVER_AT:?请先设置实际切表时刻}"
. /root/litellm/maintenance/body-retention-30d/window.env
python3 scripts/spend_log_body_retention.py \
  --state-file /root/litellm/maintenance/body-retention-30d/state.json \
  --not-before "$SPENDLOGS_CUTOVER_UTC" --retention-days 30 --min-free-gib 10
```

初始化可重复执行。同一时刻即使换了时区，已有游标也会保留。若目录已记录其他切换时刻，程序退出。原保留期的状态、审计和备份另存。回滚前先停止新 timer 和 service，再恢复原脚本、unit 及原定时状态

带 `--not-before` 时，清理区间为 `切换时刻 <= startTime < 当前时刻减去保留天数`。首次到期前直接退出，无需访问数据库。预览、只读重跑、缺失游标或为迟到记录向前重看一天，都受同一下界约束。若应用允许在切换后补写 `startTime` 早于切换时刻的记录，须按实际回填范围单独处理

脚本独立运行时默认保留 90 天，支持原有游标。默认连接 `litellm_db` 容器的 `litellm` 库，用户为 `llmproxy`，可用 `--database`、`--container`、`--db-user` 指定隔离测试库。单批 statement_timeout 为 15 秒，lock_timeout 为 1 秒。预览完成后加 `--apply` 才会提交变更。失败批次不推进游标

手动运行与定时任务必须复用同一state文件，文件锁会阻止重入。不要在任务运行中替换脚本或删除状态文件

未完成一轮时沿用原截止时间与游标。完成后下一轮只扫描上次截止时间前一天的重叠范围和新过期记录，避免每次重扫全部永久基础记录。超过一天的迟到记录或历史回灌，需要暂停常态任务后用独立游标和明确的历史区间补扫，再恢复常态任务

首批验证可增加 `--backup-dir /root/litellm/maintenance/body-retention-canary`。该目录使用0700、文件0600，压缩原始正文在更新前落盘；启用备份时清理超过7天的同名前缀备份，并限制总量10GiB。常态清理不重复备份已过期正文，避免把正文换一个目录永久保存。若首批后不再传该参数，部署者须另设7天到期清理

## 定时运行

使用独立systemd oneshot service及timer，显式指定Asia/Shanghai低峰时段。业务容器不重启。不要设置 `RemainAfterExit=yes`，以免后续timer不能再次执行

timer设置 `Persistent=false`，错过低峰时不在白天补跑，下次低峰继续。服务的运行预算应大于每日新增过期数据与一天重叠范围所需时间，历史积压则在监控下单独分批回填，不能把完成部署等同于历史已清完

随附 service 在北京时间每日 23 点启动，每批 50 条，间隔 1 秒。每次最多运行 7 小时、最多 6000 批，磁盘可用空间低于 10 GiB 即停止并记录水位。批量应结合写入延迟和夜间积压调整。更新 unit 前，先停旧 timer 和正在运行的 service，并备份旧文件

```bash
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

使用独立且名称以 `test_body_retention_` 开头的PostgreSQL测试库，所有数据为合成fixture。测试主机上的两个维护脚本应放在同一目录。测试逐行比较基础字段与计费 metadata，覆盖切表下界、保留期上界、初始化冲突、历史 JSON 文本归因、首批备份、事务回滚、幂等、续跑和锁

```bash
RETENTION_TEST_DATABASE=test_body_retention_example \
RETENTION_TEST_SSH_HOST=proxy-host \
RETENTION_TEST_SCRIPT=/path/to/scripts/spend_log_body_retention.py \
  .venv/bin/python tests/integration/spend/test_spend_log_body_retention.py
```

[官方整行清理](https://docs.litellm.ai/docs/proxy/spend_logs_deletion)、[正文写入设置](https://docs.litellm.ai/docs/proxy/ui_spend_log_settings)、[PostgreSQL JSON操作](https://www.postgresql.org/docs/16/functions-json.html)
