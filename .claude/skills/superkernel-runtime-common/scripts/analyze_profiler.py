#!/usr/bin/env python3
"""Compatibility launcher for the sibling fusion-performance skill."""

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[2]
    / "superkernel-fusion-performance-analysis"
    / "scripts"
    / "analyze_fusion_performance.py"
)

if not TARGET.is_file():
    raise SystemExit(
        "missing sibling skill superkernel-fusion-performance-analysis; "
        "install it before running profiling analysis"
    )

runpy.run_path(str(TARGET), run_name="__main__")
