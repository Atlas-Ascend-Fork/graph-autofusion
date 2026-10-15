#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Wrap one verified scope action as a critical-path scope derivative manifest."""

import argparse
import json
from pathlib import Path

import multistream_source_transform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-action", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent-action", type=Path, required=True)
    parser.add_argument("--single-stream-projection-before", required=True)
    parser.add_argument("--single-stream-projection-after", required=True)
    parser.add_argument("--dependency-before", required=True)
    parser.add_argument("--dependency-after", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError(f"output already exists: {args.out}")
        result = multistream_source_transform.wrap_scope_derivative(
            json.loads(args.scope_action.read_text()),
            json.loads(args.candidate.read_text()),
            json.loads(args.parent_action.read_text()),
            single_stream_projection_fingerprint_before=args.single_stream_projection_before,
            single_stream_projection_fingerprint_after=args.single_stream_projection_after,
            dependency_evidence_fingerprint_before=args.dependency_before,
            dependency_evidence_fingerprint_after=args.dependency_after,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
