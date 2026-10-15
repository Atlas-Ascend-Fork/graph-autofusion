#!/usr/bin/env python3
"""Validate that Stage-A SuperKernel configs explicitly disable all option maps."""

import argparse
import json
from pathlib import Path

from analyze_performance import validate_stage_a_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    try:
        configs = [validate_stage_a_config(path) for path in args.config]
    except ValueError as error:
        parser.error(str(error))

    report = {
        "schema_version": "superkernel-stage-a-option-validation-v1",
        "policy": "explicit_empty_option_maps",
        "valid": True,
        "configs": configs,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
