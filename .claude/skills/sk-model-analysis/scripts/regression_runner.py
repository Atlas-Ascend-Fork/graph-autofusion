#!/usr/bin/env python3
"""Read-only sample regression; all reports and caches go to temporary output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from graph_report.pipeline import discover_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_root", type=Path)
    parser.add_argument("--parser", choices=("both", "json", "log"), default="both")
    parser.add_argument(
        "--case",
        action="append",
        choices=("analyze_result_root", "analyze_model_dir", "hang_result_root"),
        default=[],
    )
    args = parser.parse_args()
    root = args.sample_root.resolve()
    if not root.is_dir():
        parser.error("Sample directory does not exist")
    models = discover_models(root, args.parser)
    if not models:
        parser.error("No model assets for selected parser")
    cases = [
        ("analyze_result_root", "analyze", root),
        ("analyze_model_dir", "analyze", models[0]),
        ("hang_result_root", "diagnose-hang-crash", root),
    ]
    for name, command, path in cases:
        if args.case and name not in args.case:
            continue
        with tempfile.TemporaryDirectory(
            prefix="sk-model-analysis-regression-"
        ) as temporary:
            output = Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("model_analysis.py")),
                    command,
                    str(path),
                    "--output",
                    str(output),
                    "--parser",
                    args.parser,
                    "--archive",
                    "none",
                    "--export-summary",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                raise SystemExit(
                    f"{name} failed:\n{completed.stdout}\n{completed.stderr}"
                )
            report = json.loads((output / "data/run-summary.json").read_text())
            assert report["generation"]["status"] == "ok"
            assert report["models"] and (output / "run-portal.html").is_file()
            assert not list(output.rglob("performance-report.html"))
            print(f"[ok] {name}: {len(report['models'])} models")


if __name__ == "__main__":
    main()
