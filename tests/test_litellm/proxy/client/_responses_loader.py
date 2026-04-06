"""Load the third-party `responses` package even when local namespace packages shadow it."""

import importlib
import site
import sys
import sysconfig
from types import ModuleType


def _iter_site_packages() -> list[str]:
    candidates: list[str] = []
    candidates.extend(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(user_site)
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.append(purelib)

    # 去重并保持顺序
    seen: set[str] = set()
    ordered: list[str] = []
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _load_responses() -> ModuleType:
    original_sys_path = list(sys.path)
    try:
        # 让 site-packages 优先，避免命中仓库内的同名 namespace 包
        candidates = _iter_site_packages()
        sys.path = candidates + [p for p in sys.path if p not in candidates]

        sys.modules.pop("responses", None)
        module = importlib.import_module("responses")
        if hasattr(module, "activate"):
            return module
    finally:
        sys.path = original_sys_path

    # 兜底：按当前环境正常导入
    import responses as fallback_responses

    return fallback_responses


responses = _load_responses()
