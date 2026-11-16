"""Generate portable reports and structured diagnostics from the public CLI."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from .archive import archive_report
from .common import issue
from .dfx_projection import project_report
from .pipeline import analyze, atomic_json
from .render import render_report


def nonnegative(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a nonnegative number") from exc
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("Expected a finite nonnegative number")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected at least one worker")
    return number


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input", help="Result root, sk_meta directory, or model directory"
    )
    parser.add_argument(
        "--parser",
        dest="parser_mode",
        choices=("both", "json", "log"),
        default="both",
        help="Independent parser mode (default: both; compare and prefer JSON)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory; source assets are read only",
    )
    parser.add_argument(
        "--jobs",
        type=positive_int,
        help="Model worker count (automatic default capped at 4)",
    )
    parser.add_argument("--no-parallel", action="store_true")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--cache",
        dest="use_cache",
        action="store_true",
        help="Read and write parser cache (disabled by default)",
    )
    cache.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable both parser cache reads and writes (default)",
    )
    parser.set_defaults(use_cache=False)
    parser.add_argument(
        "--export-summary",
        action="store_true",
        help="Write data/run-summary.json for structured consumers",
    )
    parser.add_argument(
        "--export-parsers",
        action="store_true",
        help="Write independent parser snapshots to data/parser-results.json",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Write generation stage timing JSON"
    )
    parser.add_argument(
        "--dfx-debug",
        action="store_true",
        help="Generate a separate DFX debug report with raw parser evidence",
    )
    parser.add_argument(
        "--archive", choices=("auto", "none", "zip", "tar.gz"), default="auto"
    )
    parser.add_argument(
        "--archive-threshold-mb",
        type=nonnegative,
        default=50,
        help="Auto ZIP threshold in MiB (default: 50)",
    )
    parser.add_argument(
        "--with-ai",
        action="store_true",
        help="Compatibility flag; no AI backend is configured",
    )


def _default_output(input_path: Path) -> Path:
    # Keep the established result-root convention, without using it for parsing.
    from sk_library_extractor import infer_model_asset_root, infer_result_root

    from .pipeline import discover_models

    models = discover_models(input_path, "both")
    root = infer_result_root(
        input_path,
        infer_model_asset_root(input_path, models) if models else None,
        models,
    )
    return root / "reports"


def _atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def generate(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    input_path = Path(args.input).resolve()
    if not input_path.is_dir():
        print(f"Input directory does not exist: {input_path}", file=sys.stderr)
        return 1
    output = (args.output or _default_output(input_path)).resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
        print("Stage 1/3: 独立提取、校验与图建模", flush=True)
        report = analyze(
            input_path,
            parser_mode=args.parser_mode,
            jobs=1 if args.no_parallel else args.jobs or min(4, os.cpu_count() or 1),
            use_cache=args.use_cache,
            cache_dir=output / ".cache" if args.use_cache else None,
            progress=lambda i, n: print(f"  Models {i}/{n}", flush=True),
        )
        if args.with_ai:
            report["diagnostics"].append(
                issue(
                    "ai_not_configured",
                    "未接入 AI 后端，基础报告不受影响",
                    severity="info",
                )
            )
        raw = report.pop("parses")
        if args.export_parsers:
            atomic_json(output / "data/parser-results.json", raw)
        del raw
        usable = any(
            m[stage]["status"] != "unavailable"
            for m in report["models"]
            for stage in ("before", "after")
        ) or any(
            m.get("runtime", {}).get("status") == "abnormal" for m in report["models"]
        )
        html_name = "run-portal-debug.html" if args.dfx_debug else "run-portal.html"
        report["generation"] = {
            "status": "ok" if usable else "insufficient",
            "html": html_name,
            "archive": None,
            "dfx_profile": "debug" if args.dfx_debug else "user",
        }
        print("Stage 2/3: 生成 Graph / DFX 单 HTML", flush=True)
        render_start = time.perf_counter()
        html = output / html_name
        content = render_report(report, include_debug=args.dfx_debug)
        archive_mode = args.archive
        if archive_mode == "auto":
            archive_mode = (
                "zip"
                if len(content.encode("utf-8"))
                > args.archive_threshold_mb * 1024 * 1024
                else "none"
            )
        if archive_mode != "none":
            report["generation"]["archive"] = html.with_suffix("." + archive_mode).name
            content = render_report(report, include_debug=args.dfx_debug)
        _atomic_text(html, content)
        report["profile"]["render_seconds"] = round(
            time.perf_counter() - render_start, 3
        )
        print("Stage 3/3: 离线交付与可选数据导出", flush=True)
        try:
            archive_report(html, archive_mode, args.archive_threshold_mb)
        except (OSError, ValueError) as exc:
            report["generation"].update(status="delivery_failed", archive=None)
            report["diagnostics"].append(
                issue(
                    "archive_failed",
                    str(exc),
                    severity="error",
                    impact="HTML 已保留，但压缩交付失败",
                    next_step="检查磁盘空间和输出权限后重新执行",
                )
            )
            _atomic_text(html, render_report(report, include_debug=args.dfx_debug))
        report["profile"]["generation_seconds"] = round(time.perf_counter() - start, 3)
        report["profile"]["html_bytes"] = html.stat().st_size
        if args.profile:
            atomic_json(output / "data/diagnose-profile.json", report["profile"])
        if args.export_summary:
            atomic_json(
                output / "data/run-summary.json",
                project_report(
                    report,
                    include_debug=args.dfx_debug,
                    preserve_raw=not args.dfx_debug,
                ),
            )
        print(f"Report: {html}", flush=True)
        if report["generation"]["archive"]:
            print(f"Archive: {output / report['generation']['archive']}", flush=True)
        print(
            f"Models: {len(report['models'])} | Parser: {args.parser_mode} | Status: {report['generation']['status']}",
            flush=True,
        )
        return 0 if usable and report["generation"]["status"] == "ok" else 1
    except (OSError, ValueError) as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        return 1
