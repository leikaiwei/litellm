"""在独立测试库执行，禁止指向生产库

RETENTION_TEST_DATABASE=test_body_retention_20261018 RETENTION_TEST_SSH_HOST=litellm \
  .venv/bin/python tests/integration/spend/test_spend_log_body_retention.py
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import unittest
import uuid
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

Json: TypeAlias = "None | bool | int | float | str | list[Json] | dict[str, Json]"
Row: TypeAlias = dict[str, Json]
CUTOFF: Final = "2026-07-20T00:00:00+00:00"
BODY: Final = "synthetic-body-marker:" + "上下文" * 2000
USAGE: Final[Row] = {
    "input_tokens": 101,
    "output_tokens": 27,
    "cache_creation_input_tokens": 19,
    "cache_read_input_tokens": 41,
    "output_tokens_details": {"reasoning_tokens": 11},
}
METADATA: Final[Row] = {
    "user_api_key_alias": "fixture-key",
    "user_api_key_spend": 1.25,
    "team_id": "fixture-team",
    "cache_hit": True,
    "cache_creation_input_tokens": 19,
    "cache_read_input_tokens": 41,
    "usage": USAGE,
    "cost_breakdown": {"input_cost": 0.75, "output_cost": 0.5},
    "unknown_future_accounting": {"amount": 7, "description": "preserve unknown keys"},
    "proxy_server_request": {"body": {"messages": BODY}, "headers": {"x-fixture": BODY}},
    "guardrail_information": [
        {
            "guardrail_name": "fixture-compression",
            "guardrail_status": "success",
            "guardrail_request": BODY,
            "guardrail_response": {
                "tokens_before": 10,
                "tokens_after": 6,
                "tokens_saved": 4,
                "compression_ratio": 0.6,
                "messages": BODY,
            },
            "match_details": [{"text": BODY}],
            "classification": {"excerpt": BODY},
            "guardrail_usage": {"units": 4},
            "guardrail_cost": 0.125,
            "guardrail_cost_by_unit": {"units": 0.125},
            "guardrail_cost_in_spend": True,
        },
        {
            "guardrail_name": "fixture-echo",
            "guardrail_response": BODY,
            "guardrail_cost": 0.25,
        },
    ],
    "error_information": {"error_code": "429", "error_type": "rate_limit", "error_message": BODY, "traceback": BODY},
    "mcp_tool_call_metadata": {
        "name": "fixture-tool",
        "arguments": {"text": BODY},
        "result": {"content": BODY},
        "mcp_server_cost_info": {"cost_per_query": 0.25},
    },
    "vector_store_request_metadata": [
        {
            "vector_store_id": "fixture-vector",
            "query": BODY,
            "vector_store_search_response": {
                "data": [{"file_id": "fixture-file", "score": 0.9, "content": [{"type": "text", "text": BODY}]}],
                "usage": {"tokens": 9},
            },
        }
    ],
    "eval_information": [
        {
            "eval_name": "fixture-eval",
            "overall_score": 82,
            "eval_error": BODY,
            "verdicts": [{"criterion_name": "fixture", "score": 82, "reasoning": BODY, "weight": 100}],
        }
    ],
}
CLEAN_METADATA: Final[Row] = {
    **METADATA,
    "proxy_server_request": None,
    "guardrail_information": [
        {
            "guardrail_name": "fixture-compression",
            "guardrail_status": "success",
            "guardrail_request": None,
            "guardrail_response": {"tokens_before": 10, "tokens_after": 6, "tokens_saved": 4, "compression_ratio": 0.6},
            "match_details": None,
            "classification": None,
            "guardrail_usage": {"units": 4},
            "guardrail_cost": 0.125,
            "guardrail_cost_by_unit": {"units": 0.125},
            "guardrail_cost_in_spend": True,
        },
        {"guardrail_name": "fixture-echo", "guardrail_response": None, "guardrail_cost": 0.25},
    ],
    "error_information": {"error_code": "429", "error_type": "rate_limit", "error_message": None, "traceback": None},
    "mcp_tool_call_metadata": {
        "name": "fixture-tool",
        "arguments": None,
        "result": None,
        "mcp_server_cost_info": {"cost_per_query": 0.25},
    },
    "vector_store_request_metadata": [
        {
            "vector_store_id": "fixture-vector",
            "query": None,
            "vector_store_search_response": {
                "data": [{"file_id": "fixture-file", "score": 0.9, "content": [{"type": "text", "text": None}]}],
                "usage": {"tokens": 9},
            },
        }
    ],
    "eval_information": [
        {
            "eval_name": "fixture-eval",
            "overall_score": 82,
            "eval_error": None,
            "verdicts": [{"criterion_name": "fixture", "score": 82, "reasoning": None, "weight": 100}],
        }
    ],
}
ENCODED_CONTEXTS: Final = frozenset(
    (
        "guardrail_information",
        "error_information",
        "mcp_tool_call_metadata",
        "vector_store_request_metadata",
        "eval_information",
    )
)
ENCODED_METADATA: Final[Row] = {
    key: json.dumps(value) if key in ENCODED_CONTEXTS else value for key, value in METADATA.items()
}
CLEAN_ENCODED_METADATA: Final[Row] = {
    key: json.dumps(value) if key in ENCODED_CONTEXTS else value for key, value in CLEAN_METADATA.items()
}
GUARDRAIL_METADATA: Final[Row] = {
    "spend": 1.25,
    "guardrail_information": {
        "guardrail_name": "fixture-object-shape",
        "guardrail_response": {"tokens_before": 10, "tokens_after": 7, "tokens_saved": True, "compression_ratio": BODY},
        "guardrail_request": BODY,
        "guardrail_cost": 0.125,
    },
}
CLEAN_GUARDRAIL_METADATA: Final[Row] = {
    "spend": 1.25,
    "guardrail_information": {
        "guardrail_name": "fixture-object-shape",
        "guardrail_response": {"tokens_before": 10, "tokens_after": 7},
        "guardrail_request": None,
        "guardrail_cost": 0.125,
    },
}
DDL: Final = """
CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogs" (
  request_id text PRIMARY KEY, call_type text NOT NULL, api_key text NOT NULL,
  spend double precision NOT NULL, total_tokens integer NOT NULL,
  prompt_tokens integer NOT NULL, completion_tokens integer NOT NULL,
  "startTime" timestamp NOT NULL, "endTime" timestamp NOT NULL,
  request_duration_ms integer, "completionStartTime" timestamp,
  model text NOT NULL, model_id text, model_group text, custom_llm_provider text,
  api_base text, "user" text, metadata jsonb, cache_hit text, cache_key text,
  request_tags jsonb, team_id text, organization_id text, end_user text,
  requester_ip_address text, messages jsonb, response jsonb, session_id text,
  status text, mcp_namespaced_tool_name text, billing_agent_id text, agent_id text,
  proxy_server_request jsonb, litellm_call_id text,
  created_at timestamp NOT NULL, updated_at timestamp NOT NULL
);
CREATE INDEX IF NOT EXISTS spend_logs_retention_fixture_cursor ON "LiteLLM_SpendLogs" ("startTime",request_id);
CREATE TABLE IF NOT EXISTS "LiteLLM_SpendLogToolIndex" (request_id text PRIMARY KEY, token_total integer);
"""


def _sql_literal(value: Json) -> str:
    return "'" + json.dumps(value, ensure_ascii=False).replace("'", "''") + "'"


def _row(request_id: str, start_time: str, **changes: Json) -> Row:
    return {
        "request_id": request_id,
        "call_type": "completion",
        "api_key": "fixture-hash",
        "spend": 1.25,
        "total_tokens": 128,
        "prompt_tokens": 101,
        "completion_tokens": 27,
        "startTime": start_time,
        "endTime": start_time,
        "request_duration_ms": 456,
        "completionStartTime": start_time,
        "model": "fixture-model",
        "model_id": "fixture-model-id",
        "model_group": "fixture-model-group",
        "custom_llm_provider": "fixture-provider",
        "api_base": "https://fixture.invalid/v1",
        "user": "fixture-user",
        "metadata": METADATA,
        "cache_hit": "True",
        "cache_key": "fixture-cache-key",
        "request_tags": ["fixture-tag"],
        "team_id": "fixture-team",
        "organization_id": "fixture-organization",
        "end_user": "fixture-end-user",
        "requester_ip_address": "192.0.2.1",
        "messages": [{"role": "user", "content": BODY}],
        "response": {"usage": USAGE, "message": {"usage": USAGE, "content": BODY}, "choices": [{"text": BODY}]},
        "session_id": "fixture-session",
        "status": "success",
        "mcp_namespaced_tool_name": "fixture/tool",
        "billing_agent_id": "fixture-billing-agent",
        "agent_id": "fixture-agent",
        "proxy_server_request": {"body": {"messages": BODY}, "headers": {"x-fixture": BODY}},
        "litellm_call_id": "fixture-call",
        "created_at": start_time,
        "updated_at": start_time,
        **changes,
    }


@dataclass(frozen=True, slots=True)
class FixtureCase:
    row: Row
    changes: Row


CASES: Final = (
    FixtureCase(
        _row("a-old-object", "2026-06-01T00:00:00"),
        {
            "messages": {},
            "proxy_server_request": {},
            "response": {"usage": USAGE, "message": {"usage": USAGE}},
            "metadata": CLEAN_METADATA,
        },
    ),
    FixtureCase(
        _row(
            "b-old-encoded",
            "2026-06-01T00:00:00",
            metadata=json.dumps(METADATA),
            response=json.dumps({"usage": USAGE, "text": BODY}),
        ),
        {
            "messages": {},
            "proxy_server_request": {},
            "response": json.dumps({"usage": USAGE}),
            "metadata": json.dumps(CLEAN_METADATA),
        },
    ),
    FixtureCase(
        _row(
            "c-old-legacy",
            "2026-06-02T00:00:00",
            metadata="historical non-JSON metadata",
            response="historical response " + BODY,
        ),
        {"messages": {}, "proxy_server_request": {}, "response": {}},
    ),
    FixtureCase(
        _row(
            "d-old-null", "2026-06-03T00:00:00", metadata=None, messages=None, response=None, proxy_server_request=None
        ),
        {},
    ),
    FixtureCase(
        _row("e-old-empty", "2026-06-04T00:00:00", metadata={}, messages=[], response={}, proxy_server_request={}),
        {},
    ),
    FixtureCase(
        _row("h-old-nested-encoded", "2026-06-05T00:00:00", metadata=ENCODED_METADATA),
        {
            "messages": {},
            "proxy_server_request": {},
            "response": {"usage": USAGE, "message": {"usage": USAGE}},
            "metadata": CLEAN_ENCODED_METADATA,
        },
    ),
    FixtureCase(
        _row("i-old-guardrail-object", "2026-06-06T00:00:00", metadata=GUARDRAIL_METADATA),
        {
            "messages": {},
            "proxy_server_request": {},
            "response": {"usage": USAGE, "message": {"usage": USAGE}},
            "metadata": CLEAN_GUARDRAIL_METADATA,
        },
    ),
    FixtureCase(
        _row(
            "j-old-json-null",
            "2026-06-07T00:00:00",
            metadata=None,
            messages=None,
            response=None,
            proxy_server_request=None,
        ),
        {},
    ),
    FixtureCase(
        _row(
            "k-old-encoded-guardrail-response",
            "2026-06-08T00:00:00",
            metadata={
                "guardrail_information": {
                    "guardrail_name": "fixture-encoded-response",
                    "guardrail_response": json.dumps(
                        {"tokens_before": 10, "tokens_after": 6, "tokens_saved": 4, "text": BODY}
                    ),
                }
            },
        ),
        {
            "messages": {},
            "proxy_server_request": {},
            "response": {"usage": USAGE, "message": {"usage": USAGE}},
            "metadata": {
                "guardrail_information": {
                    "guardrail_name": "fixture-encoded-response",
                    "guardrail_response": json.dumps({"tokens_before": 10, "tokens_after": 6, "tokens_saved": 4}),
                }
            },
        },
    ),
    FixtureCase(_row("f-exact-boundary", "2026-07-20T00:00:00"), {}),
    FixtureCase(_row("g-recent", "2026-07-21T00:00:00"), {}),
)


def _normalized(row: Row) -> Row:
    def decode(value: Json) -> Json:
        if not isinstance(value, str):
            return value
        try:
            return cast(Json, json.loads(value))
        except json.JSONDecodeError:
            return value

    def metadata(value: Json) -> Json:
        decoded: Final = decode(value)
        if not isinstance(decoded, dict):
            return decoded
        return {
            key: guardrail(item)
            if key == "guardrail_information"
            else decode(item)
            if key in ENCODED_CONTEXTS
            else item
            for key, item in decoded.items()
        }

    def guardrail(value: Json) -> Json:
        decoded: Final = decode(value)
        if isinstance(decoded, list):
            return [guardrail(item) for item in decoded]
        if isinstance(decoded, dict):
            return {key: decode(item) if key == "guardrail_response" else item for key, item in decoded.items()}
        return decoded

    return {
        key: metadata(value) if key == "metadata" else decode(value) if key == "response" else value
        for key, value in row.items()
    }


class SpendLogBodyRetentionPostgresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database = os.environ.get("RETENTION_TEST_DATABASE", "")
        if not cls.database:
            raise unittest.SkipTest("显式设置 RETENTION_TEST_DATABASE 后才运行独立数据库测试")
        if not cls.database.startswith("test_body_retention_"):
            raise RuntimeError("测试库名称必须以 test_body_retention_ 开头，禁止操作生产库")
        cls.host = os.environ.get("RETENTION_TEST_SSH_HOST", "")
        cls.container = os.environ.get("RETENTION_TEST_CONTAINER", "litellm_db")
        cls.user = os.environ.get("RETENTION_TEST_DB_USER", "llmproxy")
        cls.script = os.environ.get("RETENTION_TEST_SCRIPT", "/root/litellm/maintenance/spend_log_body_retention.py")
        cls.directory = "/tmp/litellm-body-retention-test-" + uuid.uuid4().hex
        cls.command(("mkdir", "-m", "700", cls.directory))
        cls.sql(DDL)

    @classmethod
    def command(cls, args: tuple[str, ...], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        command: Final = ("ssh", "-o", "BatchMode=yes", cls.host, shlex.join(args)) if cls.host else args
        return subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=600, check=False)

    @classmethod
    def sql(cls, statement: str) -> str:
        started: Final = time.monotonic()
        result: Final = cls.command(
            (
                "docker",
                "exec",
                "-i",
                cls.container,
                "psql",
                "-X",
                "-qAt",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                cls.user,
                "-d",
                cls.database,
            ),
            stdin=statement,
        )
        elapsed: Final = time.monotonic() - started
        if elapsed > 2:
            sys.stderr.write(f"测试 SQL {statement.lstrip().split()[0]} 耗时 {elapsed:.2f} 秒\n")
        if result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout.strip()

    def setUp(self) -> None:
        self.started = time.monotonic()
        self.state = self.directory + "/" + self.id().rsplit(".", 1)[-1] + ".json"
        inserts: Final = tuple(
            'INSERT INTO "LiteLLM_SpendLogs" SELECT * FROM jsonb_populate_record(NULL::"LiteLLM_SpendLogs", '
            + _sql_literal(case.row)
            + "::jsonb);"
            for case in CASES
        )
        self.sql(
            'BEGIN; DROP TRIGGER IF EXISTS corrupt_accounting ON "LiteLLM_SpendLogs"; '
            'DELETE FROM "LiteLLM_SpendLogs"; DELETE FROM "LiteLLM_SpendLogToolIndex";\n'
            + "\n".join(inserts)
            + "\nINSERT INTO \"LiteLLM_SpendLogToolIndex\" VALUES ('fixture-tool-row',128);\n"
            + """UPDATE "LiteLLM_SpendLogs" SET metadata='null'::jsonb,messages='null'::jsonb,
response='null'::jsonb,proxy_server_request='null'::jsonb WHERE request_id='j-old-json-null'; COMMIT;"""
        )

    def tearDown(self) -> None:
        sys.stderr.write(f"用例耗时 {time.monotonic() - self.started:.2f} 秒\n")

    def snapshot(self) -> dict[str, Row]:
        return cast(
            dict[str, Row],
            json.loads(self.sql('SELECT jsonb_object_agg(request_id,to_jsonb(s)) FROM "LiteLLM_SpendLogs" s;')),
        )

    def audit(self) -> tuple[Row, ...]:
        result: Final = self.command(
            (
                "python3",
                "-c",
                "import json,sys; print(json.dumps([json.loads(line) for line in open(sys.argv[1])]))",
                self.state + ".audit.jsonl",
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("synthetic-body-marker:", result.stdout)
        self.assertNotIn("fixture-hash", result.stdout)
        return tuple(cast(list[Row], json.loads(result.stdout)))

    def cleanup(self, *extra: str, success: bool = True) -> subprocess.CompletedProcess[str]:
        started: Final = time.monotonic()
        result: Final = self.command(
            (
                "python3",
                self.script,
                "--container",
                self.container,
                "--database",
                self.database,
                "--db-user",
                self.user,
                "--state-file",
                self.state,
                "--cutoff",
                CUTOFF,
                "--batch-size",
                "2",
                "--run-budget-seconds",
                "300",
                "--sleep-seconds",
                "0",
                *extra,
            )
        )
        sys.stderr.write(f"清理 CLI 耗时 {time.monotonic() - started:.2f} 秒\n")
        if success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def assert_cleaned(self, before: dict[str, Row]) -> None:
        after: Final = self.snapshot()
        self.assertEqual(set(before), set(after), "请求记录数量必须完全保留")
        for case in CASES:
            request_id: Final = cast(str, case.row["request_id"])
            expected: Final = {**before[request_id], **case.changes}
            self.assertEqual(_normalized(after[request_id]), _normalized(expected), request_id)
        self.assertIsInstance(after["b-old-encoded"]["metadata"], str)
        self.assertIsInstance(after["b-old-encoded"]["response"], str)
        nested: Final = cast(Row, after["h-old-nested-encoded"]["metadata"])
        for key in ENCODED_CONTEXTS:
            self.assertIsInstance(nested[key], str, key)
        self.assertEqual(
            self.sql("""SELECT request_id,metadata IS NULL,messages IS NULL,response IS NULL,proxy_server_request IS NULL
FROM "LiteLLM_SpendLogs" WHERE request_id IN ('d-old-null','j-old-json-null') ORDER BY request_id;"""),
            "d-old-null|t|t|t|t\nj-old-json-null|f|f|f|f",
        )
        self.assertEqual(
            self.sql('SELECT request_id,token_total FROM "LiteLLM_SpendLogToolIndex";'), "fixture-tool-row|128"
        )

    def test_dry_run_preserves_every_value(self) -> None:
        before: Final = self.snapshot()
        self.cleanup()
        self.assertEqual(self.snapshot(), before)

    def test_only_expired_bodies_are_cleared_and_all_accounting_survives(self) -> None:
        before: Final = self.snapshot()
        self.cleanup("--apply")
        self.assert_cleaned(before)

    def test_successful_batches_resume_and_second_run_is_idempotent(self) -> None:
        before: Final = self.snapshot()
        self.cleanup("--apply", "--max-batches", "1")
        partial: Final = self.snapshot()
        self.assertNotEqual(partial["a-old-object"], before["a-old-object"])
        self.assertEqual(partial["c-old-legacy"], before["c-old-legacy"], "首批不能越过未处理记录")
        self.cleanup("--apply")
        self.assert_cleaned(before)
        cleaned: Final = self.snapshot()
        self.cleanup("--apply")
        self.assertEqual(self.snapshot(), cleaned)

    def test_accounting_trigger_change_rolls_back_batch_and_does_not_skip_rows(self) -> None:
        before: Final = self.snapshot()
        self.sql("""
CREATE OR REPLACE FUNCTION fixture_corrupt_accounting() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN NEW.spend := OLD.spend + 999; RETURN NEW; END $$;
CREATE TRIGGER corrupt_accounting BEFORE UPDATE ON "LiteLLM_SpendLogs"
FOR EACH ROW EXECUTE FUNCTION fixture_corrupt_accounting();
""")
        self.cleanup("--apply", success=False)
        self.assertEqual(self.snapshot(), before, "校验失败必须回滚同批全部更新")
        self.assertEqual(tuple(entry["phase"] for entry in self.audit()), ("planned",))
        self.sql('DROP TRIGGER corrupt_accounting ON "LiteLLM_SpendLogs";')
        self.cleanup("--apply")
        self.assert_cleaned(before)

    def test_failed_second_batch_resumes_from_last_successful_batch(self) -> None:
        before: Final = self.snapshot()
        self.sql("""
CREATE OR REPLACE FUNCTION fixture_corrupt_accounting() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN NEW.spend := OLD.spend + 999; RETURN NEW; END $$;
CREATE TRIGGER corrupt_accounting BEFORE UPDATE ON "LiteLLM_SpendLogs"
FOR EACH ROW WHEN (OLD.request_id='c-old-legacy') EXECUTE FUNCTION fixture_corrupt_accounting();
""")
        self.cleanup("--apply", success=False)
        partial: Final = self.snapshot()
        self.assertNotEqual(partial["a-old-object"], before["a-old-object"])
        self.assertEqual(partial["c-old-legacy"], before["c-old-legacy"])
        self.assertEqual(partial["h-old-nested-encoded"], before["h-old-nested-encoded"])
        self.sql('DROP TRIGGER corrupt_accounting ON "LiteLLM_SpendLogs";')
        self.cleanup("--apply")
        self.assert_cleaned(before)

    def test_canary_backup_contains_exact_original_bodies_with_private_permissions(self) -> None:
        before: Final = self.snapshot()
        backup: Final = self.directory + "/backup-" + uuid.uuid4().hex
        self.cleanup("--apply", "--max-batches", "1", "--backup-dir", backup)
        result: Final = self.command(
            (
                "python3",
                "-c",
                "import glob,gzip,json,os,sys; paths=glob.glob(sys.argv[1]+'/*.json.gz'); "
                "print(json.dumps({'rows':[row for path in paths for row in json.load(gzip.open(path))], "
                "'modes':[os.stat(path).st_mode&0o777 for path in paths], 'directory_mode':os.stat(sys.argv[1]).st_mode&0o777}))",
                backup,
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload: Final = cast(Row, json.loads(result.stdout))
        self.assertEqual(payload["modes"], [0o600])
        self.assertEqual(payload["directory_mode"], 0o700)
        rows: Final = cast(list[Row], payload["rows"])
        self.assertEqual({row["request_id"] for row in rows}, {"a-old-object", "b-old-encoded"})
        for row in rows:
            for column in ("messages", "response", "proxy_server_request", "metadata"):
                self.assertEqual(row[column], before[cast(str, row["request_id"])][column], column)

    def test_completed_cycle_advances_cutoff_and_covers_recent_backfill_without_rescanning_ancient_rows(self) -> None:
        self.cleanup("--apply")
        late: Final = _row("overlap-backfill", "2026-07-19T00:00:00")
        ancient: Final = _row("ancient-backfill-needs-full-rescan", "2026-06-01T00:00:00")
        self.sql(
            "\n".join(
                'INSERT INTO "LiteLLM_SpendLogs" SELECT * FROM jsonb_populate_record(NULL::"LiteLLM_SpendLogs", '
                + _sql_literal(row)
                + "::jsonb);"
                for row in (late, ancient)
            )
        )
        before: Final = self.snapshot()
        result: Final = self.cleanup("--apply", "--cutoff", "2026-07-22T00:00:00+00:00")
        after: Final = self.snapshot()
        self.assertEqual(
            sum(int(count) for count in re.findall(r"扫描 (\d+) 条", result.stdout + result.stderr)),
            3,
            result.stdout + result.stderr,
        )
        self.assertEqual(set(after), set(before))
        for request_id, row in before.items():
            changes: Final[Row] = (
                {
                    "messages": {},
                    "proxy_server_request": {},
                    "response": {"usage": USAGE, "message": {"usage": USAGE}},
                    "metadata": CLEAN_METADATA,
                }
                if request_id in ("overlap-backfill", "f-exact-boundary", "g-recent")
                else {}
            )
            self.assertEqual(_normalized(after[request_id]), _normalized({**row, **changes}), request_id)

    def test_existing_file_lock_prevents_concurrent_run(self) -> None:
        before: Final = self.snapshot()
        code: Final = (
            "import fcntl,subprocess,sys; "
            "handle=open(sys.argv[1]+'.lock','w'); fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB); "
            "result=subprocess.run(sys.argv[2:],capture_output=True,text=True); "
            "print(result.stdout); print(result.stderr,file=sys.stderr); sys.exit(result.returncode)"
        )
        result: Final = self.command(
            (
                "python3",
                "-c",
                code,
                self.state,
                "python3",
                self.script,
                "--container",
                self.container,
                "--database",
                self.database,
                "--db-user",
                self.user,
                "--state-file",
                self.state,
                "--cutoff",
                CUTOFF,
                "--apply",
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("已有正文清理任务运行", result.stdout + result.stderr)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
