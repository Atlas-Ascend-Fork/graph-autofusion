#!/usr/bin/env python3
"""Forward the standalone source-range entry to the shared session controller."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv=None):
    controller = (
        Path(__file__).resolve().parents[2]
        / "superkernel-auto-tune"
        / "scripts"
        / "auto_tune_session.py"
    )
    if not controller.is_file():
        raise SystemExit(
            "missing sibling skill superkernel-auto-tune; install the source-range "
            "entrypoint with its dependencies"
        )
    arguments = sys.argv[1:] if argv is None else argv
    os.execv(
        sys.executable,
        [sys.executable, str(controller), "source-range-from-smap", *arguments],
    )


if __name__ == "__main__":
    main()
