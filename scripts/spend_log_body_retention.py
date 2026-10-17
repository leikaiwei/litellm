#!/usr/bin/env python3
"""保留账单明细，分批清理超过保留期的请求正文"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, cast

Json = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
LOGGER: Final = logging.getLogger(__name__)

SQL_SETUP: Final = r"""
CREATE OR REPLACE FUNCTION pg_temp.retention_parse(v jsonb) RETURNS jsonb
LANGUAGE plpgsql IMMUTABLE AS $fn$
BEGIN
  IF jsonb_typeof(v) <> 'string' THEN RETURN v; END IF;
  RETURN (v #>> '{}')::jsonb;
EXCEPTION WHEN invalid_text_representation THEN RETURN NULL;
END $fn$;

CREATE OR REPLACE FUNCTION pg_temp.retention_clean(v jsonb, context text) RETURNS jsonb
LANGUAGE plpgsql IMMUTABLE AS $fn$
DECLARE parsed jsonb; cleaned jsonb;
BEGIN
  IF v IS NULL THEN RETURN NULL; END IF;
  IF context <> 'body' AND jsonb_typeof(v) = 'string' THEN
    parsed := pg_temp.retention_parse(v);
    IF parsed IS NULL THEN
      IF context = 'response' THEN RETURN '{}'::jsonb; END IF;
      IF context = 'guardrail_response' THEN RETURN 'null'::jsonb; END IF;
      RETURN v;
    END IF;
    cleaned := pg_temp.retention_clean(parsed, context);
    IF cleaned IS NOT DISTINCT FROM parsed THEN RETURN v; END IF;
    RETURN to_jsonb(cleaned::text);
  END IF;
  IF context = 'body' THEN
    IF v IN ('null'::jsonb, '{}'::jsonb, '[]'::jsonb) THEN RETURN v; END IF;
    RETURN '{}'::jsonb;
  END IF;
  IF context = 'response' THEN
    IF v IN ('null'::jsonb, '{}'::jsonb, '[]'::jsonb) THEN RETURN v; END IF;
    IF jsonb_typeof(v) <> 'object' THEN RETURN '{}'::jsonb; END IF;
    RETURN (CASE WHEN v ? 'usage' THEN jsonb_build_object('usage', v->'usage') ELSE '{}'::jsonb END)
      || (CASE WHEN jsonb_typeof(v->'message') = 'object' AND (v->'message') ? 'usage'
          THEN jsonb_build_object('message', jsonb_build_object('usage', v#>'{message,usage}'))
          ELSE '{}'::jsonb END);
  END IF;
  IF context = 'guardrail_response' THEN
    IF jsonb_typeof(v) <> 'object' THEN RETURN 'null'::jsonb; END IF;
    RETURN COALESCE((SELECT jsonb_object_agg(k, val) FROM jsonb_each(v) AS stats(k, val)
      WHERE k IN ('tokens_before', 'tokens_after', 'tokens_saved', 'compression_ratio')
        AND jsonb_typeof(val) = 'number'), 'null'::jsonb);
  END IF;
  IF jsonb_typeof(v) = 'array' THEN
    RETURN (SELECT COALESCE(jsonb_agg(pg_temp.retention_clean(item, context) ORDER BY ord), '[]'::jsonb)
      FROM jsonb_array_elements(v) WITH ORDINALITY AS items(item, ord));
  END IF;
  IF jsonb_typeof(v) <> 'object' THEN RETURN v; END IF;
  RETURN (SELECT COALESCE(jsonb_object_agg(key,
    CASE
      WHEN context = 'metadata' AND key = 'proxy_server_request' THEN 'null'::jsonb
      WHEN context = 'metadata' AND key IN ('guardrail_information', 'error_information',
        'mcp_tool_call_metadata', 'vector_store_request_metadata', 'eval_information')
        THEN pg_temp.retention_clean(value, key)
      WHEN context = 'guardrail_information' AND key IN ('guardrail_request', 'match_details', 'classification')
        THEN 'null'::jsonb
      WHEN context = 'guardrail_information' AND key = 'guardrail_response' THEN
        pg_temp.retention_clean(value, 'guardrail_response')
      WHEN context = 'error_information' AND key IN ('error_message', 'traceback') THEN 'null'::jsonb
      WHEN context = 'mcp_tool_call_metadata' AND key IN ('arguments', 'result') THEN 'null'::jsonb
      WHEN context = 'vector_store_request_metadata' AND key = 'query' THEN 'null'::jsonb
      WHEN context = 'vector_store_request_metadata' AND key = 'vector_store_search_response'
        THEN pg_temp.retention_clean(value, 'vector_response')
      WHEN context = 'vector_response' AND key = 'data' THEN pg_temp.retention_clean(value, 'vector_data')
      WHEN context = 'vector_data' AND key = 'content' THEN pg_temp.retention_clean(value, 'vector_content')
      WHEN context = 'vector_content' AND key = 'text' THEN 'null'::jsonb
      WHEN context = 'eval_information' AND key = 'eval_error' THEN 'null'::jsonb
      WHEN context = 'eval_information' AND key = 'verdicts' THEN pg_temp.retention_clean(value, 'eval_verdict')
      WHEN context = 'eval_verdict' AND key = 'reasoning' THEN 'null'::jsonb
      ELSE value
    END), '{}'::jsonb) FROM jsonb_each(v));
END $fn$;
"""


@dataclass(frozen=True, slots=True)
class Cursor:
    start_time: str
    request_id: str


@dataclass(frozen=True, slots=True)
class Options:
    container: str
    database: str
    db_user: str
    state_file: Path
    backup_dir: Path | None
    batch_size: int
    run_budget_seconds: float
    sleep_seconds: float
    max_batches: int
    retention_days: int
    cutoff: datetime | None
    apply: bool
    min_free_gib: float


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def object_value(value: Json) -> dict[str, Json]:
    if not isinstance(value, dict):
        raise ValueError("数据库返回格式不正确")
    return value


def run_sql(options: Options, sql: str) -> dict[str, Json]:
    result: Final = subprocess.run(
        (
            "docker",
            "exec",
            "-i",
            options.container,
            "psql",
            "-X",
            "-qAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            options.db_user,
            "-d",
            options.database,
        ),
        input="BEGIN; SET LOCAL standard_conforming_strings=on; SET LOCAL timezone='UTC'; "
        "SET LOCAL statement_timeout='15s'; SET LOCAL lock_timeout='1s';\n" + SQL_SETUP + sql + "\nCOMMIT;\n",
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if result.returncode:
        # PostgreSQL 错误上下文可能包含正文，只记录退出码
        raise RuntimeError(f"数据库操作失败，退出码 {result.returncode}，本批未确认成功")
    return object_value(cast(Json, json.loads(result.stdout)))


def select_batch_sql(
    cutoff: str, cursor: Cursor | None, batch_size: int, include_body: bool = False, lower_bound: str | None = None
) -> str:
    after: Final = (
        f'AND ("startTime", request_id) > ({literal(cursor.start_time)}::timestamp, {literal(cursor.request_id)})'
        if cursor
        else ""
    )
    body_fields: Final = (
        " || jsonb_build_object('messages', s.messages, 'response', s.response, "
        "'proxy_server_request', s.proxy_server_request, 'metadata', s.metadata)"
        if include_body
        else ""
    )
    minimum: Final = f'AND "startTime" >= {literal(lower_bound)}::timestamp' if lower_bound else ""
    return f"""
WITH scanned AS MATERIALIZED (
  SELECT * FROM "LiteLLM_SpendLogs"
  WHERE "startTime" < {literal(cutoff)}::timestamp {minimum} {after}
  ORDER BY "startTime", request_id LIMIT {batch_size}
), marked AS (
  SELECT s.*, pg_temp.retention_clean(messages::jsonb, 'body') AS clean_messages,
    pg_temp.retention_clean(response::jsonb, 'response') AS clean_response,
    pg_temp.retention_clean(proxy_server_request::jsonb, 'body') AS clean_request,
    pg_temp.retention_clean(metadata::jsonb, 'metadata') AS clean_metadata
  FROM scanned s
), changed AS (
  SELECT * FROM marked WHERE messages::jsonb IS DISTINCT FROM clean_messages
    OR response::jsonb IS DISTINCT FROM clean_response
    OR proxy_server_request::jsonb IS DISTINCT FROM clean_request
    OR metadata::jsonb IS DISTINCT FROM clean_metadata
)
SELECT jsonb_build_object(
 'scanned', (SELECT count(*) FROM scanned),
 'metadata_unparsed', (SELECT count(*) FROM scanned WHERE jsonb_typeof(metadata::jsonb) = 'string'
   AND pg_temp.retention_parse(metadata::jsonb) IS NULL),
 'cursor', (SELECT jsonb_build_object('start_time', "startTime", 'request_id', request_id)
   FROM scanned ORDER BY "startTime" DESC, request_id DESC LIMIT 1),
 'rows', COALESCE((SELECT jsonb_agg(jsonb_build_object('request_id', s.request_id, 'start_time', s."startTime",
   'fingerprint', md5(to_jsonb(s)::text),
   'base_fingerprint', md5((to_jsonb(s) - ARRAY['messages','response','proxy_server_request','metadata'])::text))
     {body_fields})
   FROM scanned s JOIN changed c USING (request_id, "startTime")), '[]'::jsonb));
"""


def apply_batch_sql(rows: list[Json], cutoff: str) -> str:
    snapshots: Final = tuple(
        {key: object_value(row)[key] for key in ("request_id", "start_time", "fingerprint")} for row in rows
    )
    payload: Final = literal(json.dumps(snapshots, ensure_ascii=True, separators=(",", ":")))
    return f"""
CREATE TEMP TABLE retention_snapshot ON COMMIT DROP AS
 SELECT * FROM jsonb_to_recordset({payload}::jsonb)
 AS x(request_id text, start_time timestamp, fingerprint text);
CREATE TEMP TABLE retention_before ON COMMIT DROP AS
 SELECT s.request_id, s."startTime", to_jsonb(s) - ARRAY['messages','response','proxy_server_request','metadata']
   AS base_projection, pg_temp.retention_clean(s.metadata::jsonb, 'metadata') AS expected_metadata,
   pg_temp.retention_clean(s.response::jsonb, 'response') AS expected_response
 FROM "LiteLLM_SpendLogs" s JOIN retention_snapshot b
   ON s.request_id = b.request_id AND s."startTime" = b.start_time
 WHERE md5(to_jsonb(s)::text) = b.fingerprint AND s."startTime" < {literal(cutoff)}::timestamp
 FOR UPDATE OF s;
DO $assert$ BEGIN
 IF (SELECT count(*) FROM retention_before) <> (SELECT count(*) FROM retention_snapshot) THEN
   RAISE EXCEPTION '正文快照已变化，本批回滚';
 END IF;
END $assert$;
UPDATE "LiteLLM_SpendLogs" s SET
 messages = pg_temp.retention_clean(s.messages::jsonb, 'body'),
 response = pg_temp.retention_clean(s.response::jsonb, 'response'),
 proxy_server_request = pg_temp.retention_clean(s.proxy_server_request::jsonb, 'body'),
 metadata = pg_temp.retention_clean(s.metadata::jsonb, 'metadata')
 FROM retention_before b WHERE s.request_id = b.request_id AND s."startTime" = b."startTime";
DO $assert$ BEGIN
 IF EXISTS (SELECT 1 FROM retention_before b LEFT JOIN "LiteLLM_SpendLogs" s
     ON s.request_id = b.request_id AND s."startTime" = b."startTime"
   WHERE s.request_id IS NULL
     OR to_jsonb(s) - ARRAY['messages','response','proxy_server_request','metadata']
       IS DISTINCT FROM b.base_projection
     OR s.metadata::jsonb IS DISTINCT FROM b.expected_metadata
     OR s.response::jsonb IS DISTINCT FROM b.expected_response
     OR s.messages::jsonb IS DISTINCT FROM pg_temp.retention_clean(s.messages::jsonb, 'body')
     OR s.proxy_server_request::jsonb IS DISTINCT FROM pg_temp.retention_clean(s.proxy_server_request::jsonb, 'body')) THEN
   RAISE EXCEPTION '账单明细校验失败，本批回滚';
 END IF;
END $assert$;
SELECT jsonb_build_object('updated', (SELECT count(*) FROM retention_before));
"""


def private_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)


def atomic_json(path: Path, value: object) -> None:
    temporary: Final = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", opener=lambda name, flags: os.open(name, flags, 0o600)) as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    descriptor: Final = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def backup_rows(directory: Path, rows: list[Json]) -> None:
    private_directory(directory)
    now: Final = time.time()
    for path in directory.glob("body-*.json.gz"):
        if now - path.stat().st_mtime > 7 * 86400:
            path.unlink()
    if sum(path.stat().st_size for path in directory.glob("body-*.json.gz")) >= 10 * 1024**3:
        raise RuntimeError("正文备份达到 10 GiB 上限，停止本次清理")
    destination: Final = directory / f"body-{time.time_ns()}.json.gz"
    with open(destination, "xb", opener=lambda name, flags: os.open(name, flags, 0o600)) as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
            compressed.write(json.dumps(rows, ensure_ascii=True).encode())
        raw.flush()
        os.fsync(raw.fileno())
    sync_directory(directory)
    if sum(path.stat().st_size for path in directory.glob("body-*.json.gz")) > 10 * 1024**3:
        destination.unlink()
        raise RuntimeError("正文备份将超过 10 GiB 上限，本批未更新")


def read_cursor(value: Json) -> Cursor | None:
    if value is None:
        return None
    fields: Final = object_value(value)
    if not isinstance(fields.get("start_time"), str) or not isinstance(fields.get("request_id"), str):
        raise ValueError("游标格式不正确")
    return Cursor(cast(str, fields["start_time"]), cast(str, fields["request_id"]))


def append_audit(state_file: Path, value: object) -> None:
    audit: Final = state_file.with_suffix(state_file.suffix + ".audit.jsonl")
    with open(audit, "a", opener=lambda name, flags: os.open(name, flags, 0o600)) as handle:
        handle.write(json.dumps(value, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(audit.parent)


def run(options: Options) -> int:
    private_directory(options.state_file.parent)
    with options.state_file.with_suffix(options.state_file.suffix + ".lock").open("a") as lock:
        lock_path: Final = Path(lock.name)
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOGGER.info("已有正文清理任务运行，本次退出")
            return 0
        return run_locked(options)


def run_locked(options: Options) -> int:
    state: Final = (
        object_value(cast(Json, json.loads(options.state_file.read_text())))
        if options.apply and options.state_file.exists()
        else {}
    )
    identity: Final = f"{options.container}/{options.database}/{options.retention_days}"
    if state and state.get("identity") != identity:
        raise ValueError("状态文件与当前数据库或保留期不匹配")
    cutoff_value: Final = state.get("cutoff") if state.get("cursor") else None
    cutoff: Final = (
        str(cutoff_value)
        if cutoff_value
        else (options.cutoff or datetime.now(timezone.utc) - timedelta(days=options.retention_days))
        .astimezone(timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
    )
    completed_cutoff: Final = state.get("completed_cutoff")
    lower_bound: Final = (
        cast(str | None, state.get("lower_bound"))
        if state.get("cursor")
        else (datetime.fromisoformat(str(completed_cutoff)) - timedelta(days=1)).isoformat()
        if completed_cutoff
        else None
    )
    cursor = read_cursor(state.get("cursor"))
    deadline: Final = time.monotonic() + options.run_budget_seconds
    count = 0
    updated = 0
    while time.monotonic() < deadline and count < options.max_batches:
        if shutil.disk_usage(options.state_file.parent).free < options.min_free_gib * 1024**3:
            raise RuntimeError("磁盘可用空间低于保护水位，停止本次清理")
        result: Final = run_sql(
            options, select_batch_sql(cutoff, cursor, options.batch_size, bool(options.backup_dir), lower_bound)
        )
        raw_rows: Final = result.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError("数据库未返回正文快照")
        rows: Final = raw_rows
        next_cursor: Final = read_cursor(result.get("cursor"))
        if options.apply and rows:
            if options.backup_dir:
                backup_rows(options.backup_dir, rows)
            batch_id: Final = str(time.time_ns())
            append_audit(
                options.state_file,
                {
                    "phase": "planned",
                    "batch_id": batch_id,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "cutoff": cutoff,
                    "rows": [
                        {
                            "request_id": object_value(row)["request_id"],
                            "start_time": object_value(row)["start_time"],
                            "base_fingerprint": object_value(row)["base_fingerprint"],
                        }
                        for row in rows
                    ],
                },
            )
            applied: Final = run_sql(options, apply_batch_sql(rows, cutoff))
            if applied.get("updated") != len(rows):
                raise RuntimeError("本批更新数量未通过校验，停止推进游标")
            append_audit(
                options.state_file,
                {
                    "phase": "committed",
                    "batch_id": batch_id,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "updated": len(rows),
                },
            )
        count += 1
        updated += len(rows)
        if options.apply:
            atomic_json(
                options.state_file,
                {
                    "identity": identity,
                    "cutoff": cutoff,
                    "cursor": {"start_time": next_cursor.start_time, "request_id": next_cursor.request_id}
                    if next_cursor
                    else None,
                    "lower_bound": lower_bound if next_cursor else None,
                    "completed_cutoff": completed_cutoff if next_cursor else cutoff,
                },
            )
        LOGGER.info(
            "%s：扫描 %s 条，正文 %s 条，未解析元数据 %s 条",
            "已清理" if options.apply else "只读预览",
            result.get("scanned"),
            len(rows),
            result.get("metadata_unparsed"),
        )
        if next_cursor is None or not options.apply:
            break
        cursor = next_cursor
        time.sleep(options.sleep_seconds)
    LOGGER.info("本次完成 %s 批，%s正文 %s 条", count, "清理" if options.apply else "预计清理", updated)
    return 0


def parse_options(argv: list[str] | None = None) -> Options:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="litellm_db")
    parser.add_argument("--database", default="litellm")
    parser.add_argument("--db-user", default="llmproxy")
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--run-budget-seconds", type=float, default=300)
    parser.add_argument("--sleep-seconds", type=float, default=1)
    parser.add_argument("--max-batches", type=int, default=1000)
    parser.add_argument("--retention-days", type=int, default=90)
    parser.add_argument("--cutoff", type=datetime.fromisoformat)
    parser.add_argument("--min-free-gib", type=float, default=20)
    parser.add_argument("--apply", action="store_true")
    args: Final = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 1000 or args.run_budget_seconds <= 0 or args.sleep_seconds < 0:
        parser.error("批量必须为 1 至 1000，预算必须大于 0，间隔不可为负数")
    if args.retention_days < 1 or args.max_batches < 1 or args.min_free_gib < 0:
        parser.error("保留天数和批次必须大于 0，磁盘水位不可为负数")
    if args.cutoff is not None and args.cutoff.tzinfo is None:
        parser.error("截止时间必须包含时区")
    return Options(**vars(args))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        return run(parse_options(argv))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        LOGGER.error("正文清理未完成，已停止且未推进失败批次游标，请核查维护记录")
        return 1


if __name__ == "__main__":
    sys.exit(main())
