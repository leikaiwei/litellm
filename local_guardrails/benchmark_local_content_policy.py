from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import Callable, Sequence

from local_guardrails.local_content_policy import LocalPolicyMatcher


POLICY_DIR = Path(__file__).parent
SHORT_TEXT = "请解释二分查找的时间复杂度。"
LONG_TEXT = "这是正常的软件工程性能分析和缓存设计讨论。" * 1_024


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    input_size: str
    characters: int
    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    sequential_rps: float
    sequential_cpu_ms_per_request: float
    sequential_cpu_percent: float
    concurrent_workers: int
    concurrent_rps: float
    concurrent_cpu_ms_per_request: float
    concurrent_cpu_percent: float


def _percentile(samples: Sequence[int], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index] / 1_000_000


def _benchmark(
    mode: str,
    input_size: str,
    text: str,
    iterations: int,
    workers: int,
    check: Callable[[str], None],
) -> BenchmarkResult:
    for _ in range(100):
        check(text)

    samples: list[int] = []
    cpu_start = process_time_ns()
    wall_start = perf_counter_ns()
    for _ in range(iterations):
        started = perf_counter_ns()
        check(text)
        samples.append(perf_counter_ns() - started)
    wall_ns = perf_counter_ns() - wall_start
    cpu_ns = process_time_ns() - cpu_start

    cpu_start = process_time_ns()
    wall_start = perf_counter_ns()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tuple(executor.map(check, (text for _ in range(iterations))))
    concurrent_wall_ns = perf_counter_ns() - wall_start
    concurrent_cpu_ns = process_time_ns() - cpu_start

    return BenchmarkResult(
        mode=mode,
        input_size=input_size,
        characters=len(text),
        iterations=iterations,
        p50_ms=round(_percentile(samples, 0.50), 4),
        p95_ms=round(_percentile(samples, 0.95), 4),
        p99_ms=round(_percentile(samples, 0.99), 4),
        sequential_rps=round(iterations * 1_000_000_000 / wall_ns, 1),
        sequential_cpu_ms_per_request=round(cpu_ns / iterations / 1_000_000, 4),
        sequential_cpu_percent=round(cpu_ns * 100 / wall_ns, 1),
        concurrent_workers=workers,
        concurrent_rps=round(iterations * 1_000_000_000 / concurrent_wall_ns, 1),
        concurrent_cpu_ms_per_request=round(concurrent_cpu_ns / iterations / 1_000_000, 4),
        concurrent_cpu_percent=round(concurrent_cpu_ns * 100 / concurrent_wall_ns, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 Content Policy matcher 微基准")
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--long-iterations", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 1) * 2))
    args = parser.parse_args()

    policy_01 = LocalPolicyMatcher.from_file(str(POLICY_DIR / "content_policy_01.yaml"))
    policy_02 = LocalPolicyMatcher.from_file(str(POLICY_DIR / "content_policy_02.yaml"))

    def checker(matchers: tuple[LocalPolicyMatcher, ...]) -> Callable[[str], None]:
        def check(text: str) -> None:
            for matcher in matchers:
                if matcher.detect(text) is not None:
                    raise AssertionError("benchmark input must remain allowed")

        return check

    modes = (
        ("off", ()),
        ("zh-abusive-language-filter", (policy_01,)),
        ("zh-financial-trading-filter", (policy_02,)),
        ("both", (policy_01, policy_02)),
    )
    results = [
        _benchmark(mode, size, text, iterations, args.workers, checker(matchers))
        for mode, matchers in modes
        for size, text, iterations in (
            ("short", SHORT_TEXT, args.iterations),
            ("long", LONG_TEXT, args.long_iterations),
        )
    ]
    sys.stdout.write(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
