"""Discover inputs, execute independent parsers, and build isolated model reports."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from .common import empty_result, issue
from .compare import resolve_model
from .exploration import enrich_model

JSON_NAMES = ("sk_graph_origin.json", "sk_graph_updated.json", "sk_task_queue.json")
LOG_NAMES = (
    "super_kernel.log",
    "sk_scope_split.log",
    "sk_node_detail.log",
    "sk_fused_nodes.log",
    "sk_device_args.log",
    "sk_fusion_fail_reasons.log",
)


def discover_models(input_path: Path, mode: str) -> list[Path]:
    names = (
        JSON_NAMES
        if mode == "json"
        else LOG_NAMES
        if mode == "log"
        else JSON_NAMES + LOG_NAMES
    )
    found = []
    for root, dirs, files in os.walk(input_path):
        dirs[:] = sorted(
            d
            for d in dirs
            if d
            not in {
                "reports",
                "bin_files",
                "kernel_meta",
                "__pycache__",
                ".cache",
                "__MACOSX",
            }
            and not d.startswith(".")
        )
        if any(name in files for name in names):
            found.append(Path(root))
    return sorted(found)


def _parse_log(path: Path) -> list[dict]:
    from .log_parser import parse_models

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return parse_models(path)


def _parse_json(path: Path) -> dict:
    from .json_parser import parse_model

    return parse_model(path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _fingerprint(path: Path, mode: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.absolute()).encode())
    digest.update(mode.encode())
    # Code content invalidates cache even when a developer forgets a manual version bump.
    scripts = Path(__file__).resolve().parent
    for code in sorted(scripts.glob("*.py")) + [
        scripts.parent / "sk_library_extractor.py",
        scripts.parent / "diagnose_run.py",
    ]:
        digest.update(code.read_bytes())
    names = (
        JSON_NAMES
        if mode == "json"
        else LOG_NAMES
        if mode == "log"
        else JSON_NAMES + LOG_NAMES
    )
    for name in names:
        digest.update(name.encode())
        file = path / name
        if not file.is_file():
            digest.update(b"missing")
            continue
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _collect(payload: tuple) -> tuple[list[dict], dict, bool]:
    path_text, mode, cache_text, use_cache = payload
    path = Path(path_text)
    fingerprint = _fingerprint(path, mode) if cache_text and use_cache else None
    cache = Path(cache_text) / f"{fingerprint}.json" if fingerprint else None
    if cache:
        try:
            cached = json.loads(cache.read_text())
            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
                and isinstance(cached.get("models"), list)
            ):
                return cached["models"], cached["parses"], True
        except (OSError, ValueError, KeyError, TypeError):
            pass
    j = None
    logs = []
    if mode in {"json", "both"}:
        try:
            j = _parse_json(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            j = empty_result(
                "json",
                [
                    issue(
                        "json_parser_failed",
                        str(exc),
                        severity="error",
                        source={"file": str(path)},
                    )
                ],
            )
    if mode in {"log", "both"}:
        try:
            logs = _parse_log(path)
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            logs = [
                empty_result(
                    "log",
                    [
                        issue(
                            "log_parser_failed",
                            str(exc),
                            severity="error",
                            source={"file": str(path)},
                        )
                    ],
                )
            ]
    models = []
    base = str(path.absolute())
    if len(logs) > 1:
        json_usable = j is not None and (
            any(j[stage]["status"] != "unavailable" for stage in ("before", "after"))
            or j["capabilities"].get("queues") != "unavailable"
        )
        for log in logs:
            # Unavailable JSON carries only diagnostics and can safely accompany each
            # instance. A usable unassigned JSON cannot validate any log instance.
            model = resolve_model(
                f"{base}/{log['instance_id']}",
                f"{path.parent.name}/{path.name}/{log['instance_id']}",
                j if mode == "both" and not json_usable else None,
                log,
                "log" if json_usable else mode,
            )
            if json_usable:
                model["diagnostics"].append(
                    issue(
                        "ambiguous_json_instance",
                        "JSON 未标明 optimize 实例，未与此 Log 实例合并",
                        impact="本实例使用独立 Log 结果，不能完成双侧校验",
                    )
                )
            models.append(model)
        if json_usable:
            j["instance_id"] = "json-unassigned"
            model = resolve_model(
                f"{base}/json-unassigned",
                f"{path.parent.name}/{path.name}/JSON（实例未对齐）",
                j,
                None,
                "json",
            )
            model["diagnostics"].append(
                issue(
                    "ambiguous_json_instance",
                    "JSON 保留为独立快照，无法唯一归属 Log 实例",
                    impact="没有对跨实例同名节点进行比较或融合",
                    next_step="补充 JSON 实例标识或提供每实例独立目录",
                )
            )
            models.append(model)
    else:
        log = logs[0] if logs else None
        instance = (log or j or {}).get("instance_id", "mi01")
        models.append(
            resolve_model(
                f"{base}/{instance}",
                f"{path.parent.name}/{path.name}/{instance}",
                j,
                log,
                mode,
            )
        )
    parses = {"json": j, "log": logs}
    if cache:
        atomic_json(
            cache, {"fingerprint": fingerprint, "models": models, "parses": parses}
        )
    return models, parses, False


def attach_hierarchy(model: dict, input_path: Path) -> None:
    """Use complete directory identity; process labels describe paths, never guessed PIDs."""
    directory = Path(model["key"]).parent
    process = directory.parent
    while process.name == "sk_meta":
        process = process.parent
    try:
        relative = process.relative_to(input_path)
        label = relative.as_posix() if relative.parts else "当前进程目录"
    except ValueError:
        label = process.name + "（上级目录）"
    model.update(
        process_key=str(process),
        process_label=label,
        model_key=str(directory),
        model_label=directory.name,
        asset_path=str(directory),
    )
    model["label"] = f"{label} / {directory.name} / {model['instance_id']}"


def analyze(
    input_path: Path,
    *,
    parser_mode: str = "both",
    jobs: int = 1,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    progress=None,
) -> dict:
    if parser_mode not in {"both", "json", "log"}:
        raise ValueError("Unknown parser mode")
    input_path = Path(input_path).resolve()
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")
    start = time.perf_counter()
    paths = discover_models(input_path, parser_mode)
    models, parses, diagnostics = [], {}, []
    hits = 0
    workers = min(max(1, jobs), len(paths)) if paths else 1
    payloads = [
        (str(p), parser_mode, str(cache_dir) if cache_dir else "", use_cache)
        for p in paths
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(_collect, p): p[0] for p in payloads}
            for index, future in enumerate(as_completed(pending), 1):
                items, raw, hit = future.result()
                models.extend(items)
                parses[pending[future]] = raw
                hits += int(hit)
                if progress:
                    progress(index, len(paths))
    else:
        for index, payload in enumerate(payloads, 1):
            items, raw, hit = _collect(payload)
            models.extend(items)
            parses[payload[0]] = raw
            hits += int(hit)
            if progress:
                progress(index, len(paths))
    models.sort(key=lambda m: m["key"])
    # Namespaced IDs make links unambiguous across model instances.
    for model in models:
        attach_hierarchy(model, input_path)
        enrich_model(model)
        prefix = hashlib.sha256(model["key"].encode()).hexdigest()[:10]
        for group in (model["diagnostics"], model.get("runtime", {}).get("issues", [])):
            for index, item in enumerate(group):
                item["id"] = f"{prefix}-{item.get('id', index)}"
                item["model_key"] = model["key"]
        diagnostics.extend(model["diagnostics"])
    if not paths:
        diagnostics.append(
            issue(
                "no_model_assets",
                "未发现所选解析模式支持的模型资产",
                severity="error",
                impact="无法还原执行图",
                next_step="提供包含 graph JSON 或 SK 日志的目录",
                source={"file": str(input_path)},
            )
        )
    return {
        "schema_version": "1.0",
        "parser_mode": parser_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "models": models,
        "diagnostics": diagnostics,
        "parses": parses,
        "profile": {
            "total_seconds": round(time.perf_counter() - start, 3),
            "cache_hits": hits,
            "model_directories": len(paths),
            "workers": workers,
        },
    }
