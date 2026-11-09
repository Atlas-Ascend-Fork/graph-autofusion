"""Small, source-independent values shared by report orchestration."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def issue(code: str, message: str, *, severity: str = "warning", **extra: Any) -> dict:
    payload = {
        "domain": "parsing",
        "severity": severity,
        "code": code,
        "message": message,
        **extra,
    }
    payload["id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    return payload


def unavailable(source: str = "") -> dict:
    return {"status": "unavailable", "nodes": [], "edges": [], "source": source}


def empty_result(parser: str, diagnostics: list | None = None) -> dict:
    return {
        "parser": parser,
        "instance_id": "mi01",
        "before": unavailable(),
        "after": unavailable(),
        "scopes": [],
        "user_scopes": [],
        "fusions": [],
        "queues": [],
        "runtime": {
            "status": "insufficient",
            "conclusion": "缺少运行诊断证据",
            "issues": [],
            "evidence": {},
        },
        "diagnostics": diagnostics or [],
        "capabilities": dict.fromkeys(
            ("scopes", "fusions", "queues", "runtime"), "unavailable"
        ),
        "source_files": [],
    }
