#!/usr/bin/env python3
"""为新表的首个请求时间建立 30 天保留下界，重复执行不重置游标。"""

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

from spend_log_body_retention import private_directory, read_state, sync_directory


def create_if_absent(path: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent) as handle:
        _ = handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        try:
            os.link(handle.name, path)
        except FileExistsError:
            return
    sync_directory(path.parent)


class Arguments(argparse.Namespace):
    cutover: datetime
    directory: Path


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--cutover", required=True, type=datetime.fromisoformat)
    _ = parser.add_argument("--directory", type=Path, default=Path("/root/litellm/maintenance/body-retention-30d"))
    args: Final = parser.parse_args(namespace=Arguments())
    requested: Final = args.cutover
    directory: Final = args.directory
    if requested.tzinfo is None:
        parser.error("切换时间必须带时区")
    cutover: Final = requested.astimezone(timezone.utc)
    if cutover > datetime.now(timezone.utc):
        parser.error("切换时间不能晚于现在")
    private_directory(directory)
    window: Final = directory / "window.env"
    state: Final = directory / "state.json"
    line: Final = f"SPENDLOGS_CUTOVER_UTC={cutover.isoformat()}\n"
    if state.exists() and not window.exists():
        parser.error("目录已有游标但缺少切换时间，请核对原窗口记录")
    create_if_absent(window, line)
    if window.read_text() != line:
        parser.error("已记录另一个切换时间，请核对原窗口记录")
    identity: Final = "litellm_db/litellm/30"
    create_if_absent(
        state, json.dumps({"identity": identity, "completed_cutoff": cutover.replace(tzinfo=None).isoformat()}) + "\n"
    )
    if read_state(state).get("identity") != identity:
        parser.error("目标目录已有其他数据库或保留期的游标")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("30 天状态已核对，北京时间 %s", cutover.astimezone(timezone(timedelta(hours=8))).isoformat())


if __name__ == "__main__":
    main()
