#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Collapse per-core sk_prof events into SK E2E, Cube, and Vector lanes."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CORE_PREFIX = re.compile(r"^\[\d+/\d+\]\s*")
STATIC_KERNEL = re.compile(r"static_kernel_(.+?)_[0-9a-f]{32,}_[0-9]+_d[0-9]+")


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Incrementally decode a top-level JSON array without third-party packages."""
    decoder, buffer, offset, opened, eof = json.JSONDecoder(), "", 0, False, False
    with path.open("r", encoding="utf-8") as source:
        while not eof or offset < len(buffer):
            if not eof:
                block = source.read(1 << 20)
                if block:
                    buffer += block
                else:
                    eof = True
            while True:
                while offset < len(buffer) and (
                    buffer[offset].isspace() or buffer[offset] == ","
                ):
                    offset += 1
                if offset >= len(buffer):
                    break
                if not opened:
                    if buffer[offset] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    opened, offset = True, offset + 1
                    continue
                if buffer[offset] == "]":
                    return
                try:
                    event, end = decoder.raw_decode(buffer, offset)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(
                            f"Malformed JSON near byte buffer offset {offset}"
                        ) from None
                    break
                offset = end
                if isinstance(event, dict):
                    yield event
            if offset:
                buffer, offset = buffer[offset:], 0
            elif eof:
                break
    if not opened:
        raise ValueError(f"{path} is empty")


def canonical_name(value: Any) -> str:
    return CORE_PREFIX.sub("", str(value or "unnamed_static_kernel"))


def operator_name(raw_name: str) -> str:
    match = STATIC_KERNEL.search(raw_name)
    return match.group(1) if match else raw_name[:180]


def create_database(
    temp_dir: Path | None, keep: bool
) -> tuple[sqlite3.Connection, Path]:
    if keep:
        directory = temp_dir or Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        db_path = directory / "sk_prof_compact.sqlite"
        if db_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite temporary database: {db_path}"
            )
    else:
        handle = tempfile.NamedTemporaryFile(
            prefix="sk_prof_compact_", suffix=".sqlite", dir=temp_dir, delete=False
        )
        handle.close()
        db_path = Path(handle.name)
    return sqlite3.connect(db_path), db_path


def add_events(
    connection: sqlite3.Connection,
    input_path: Path,
    cube_pid: str,
    vector_pid: str,
    include_parents: bool,
) -> dict[str, int]:
    connection.execute(
        """CREATE TABLE events (component TEXT NOT NULL, model_id TEXT NOT NULL, sk_id TEXT NOT NULL, node_id TEXT NOT NULL, raw_name TEXT NOT NULL, operator TEXT NOT NULL, start REAL NOT NULL, finish REAL NOT NULL)"""
    )
    accepted = ignored = excluded_parent_spans = 0
    batch: list[tuple[str, str, str, str, str, str, float, float]] = []
    components = {cube_pid: "Cube", vector_pid: "Vector"}
    for event in iter_json_array(input_path):
        if event.get("ph") != "X" or str(event.get("pid")) not in components:
            ignored += 1
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        if "skId" not in args:
            ignored += 1
            continue
        is_parent = "nodeId" not in args
        if is_parent and not include_parents:
            excluded_parent_spans += 1
            continue
        try:
            start, duration = float(event["ts"]), float(event["dur"])
        except (KeyError, TypeError, ValueError):
            ignored += 1
            continue
        if duration < 0 or not math.isfinite(start) or not math.isfinite(duration):
            ignored += 1
            continue
        raw_name = canonical_name(event.get("name"))
        component = "SK E2E" if is_parent else components[str(event.get("pid"))]
        batch.append(
            (
                component,
                str(args.get("modelId", "")),
                str(args["skId"]),
                str(args.get("nodeId", "__sk_span__")),
                raw_name,
                operator_name(raw_name),
                start,
                start + duration,
            )
        )
        accepted += 1
        if len(batch) == 10_000:
            connection.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
            batch.clear()
    if batch:
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch
        )
    connection.commit()
    connection.execute(
        "CREATE INDEX events_by_identity_time ON events (component, model_id, sk_id, node_id, raw_name, start)"
    )
    connection.execute(
        """CREATE TABLE aggregate (component TEXT NOT NULL, model_id TEXT NOT NULL, sk_id TEXT NOT NULL, node_id TEXT NOT NULL, raw_name TEXT NOT NULL, operator TEXT NOT NULL, occurrence INTEGER NOT NULL, start REAL NOT NULL, finish REAL NOT NULL, core_event_count INTEGER NOT NULL)"""
    )
    return {
        "accepted_raw_events": accepted,
        "ignored_events": ignored,
        "excluded_parent_spans": excluded_parent_spans,
    }


def aggregate_events(connection: sqlite3.Connection, gap_us: float) -> int:
    query = "SELECT component, model_id, sk_id, node_id, raw_name, operator, start, finish FROM events ORDER BY component, model_id, sk_id, node_id, raw_name, start"
    current_key: tuple[str, str, str, str, str, str] | None = None
    occurrence = aggregates = count = 0
    previous_start = start = finish = 0.0
    pending: list[tuple[Any, ...]] = []

    def flush() -> None:
        nonlocal aggregates
        if current_key is not None:
            pending.append((*current_key, occurrence, start, finish, count))
            aggregates += 1
            if len(pending) == 10_000:
                connection.executemany(
                    "INSERT INTO aggregate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    pending,
                )
                pending.clear()

    for row in connection.execute(query):
        key, item_start, item_finish = tuple(row[:6]), float(row[6]), float(row[7])
        if current_key != key:
            flush()
            current_key, occurrence, start, finish, count = (
                key,
                1,
                item_start,
                item_finish,
                1,
            )
        elif item_start - previous_start > gap_us:
            flush()
            occurrence, start, finish, count = (
                occurrence + 1,
                item_start,
                item_finish,
                1,
            )
        else:
            finish, count = max(finish, item_finish), count + 1
        previous_start = item_start
    flush()
    if pending:
        connection.executemany(
            "INSERT INTO aggregate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pending
        )
    connection.commit()
    return aggregates


def write_trace(connection: sqlite3.Connection, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": 0,
            "args": {"name": "SuperKernel components"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 0,
            "tid": 0,
            "args": {"name": "SK E2E"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 0,
            "tid": 1,
            "args": {"name": "Cube"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": 0,
            "tid": 2,
            "args": {"name": "Vector"},
        },
        {
            "ph": "M",
            "name": "thread_sort_index",
            "pid": 0,
            "tid": 0,
            "args": {"sort_index": 0},
        },
        {
            "ph": "M",
            "name": "thread_sort_index",
            "pid": 0,
            "tid": 1,
            "args": {"sort_index": 1},
        },
        {
            "ph": "M",
            "name": "thread_sort_index",
            "pid": 0,
            "tid": 2,
            "args": {"sort_index": 2},
        },
    ]
    total = 0
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    try:
        destination = temporary
        # Chrome Trace timestamps are microseconds by convention.  Its legacy viewer
        # reads the raw sk_prof top-level array more reliably than an object wrapper.
        destination.write("[")
        first = True
        for event in metadata:
            if not first:
                destination.write(",")
            json.dump(
                event,
                destination,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            first = False
        query = "SELECT component, model_id, sk_id, node_id, raw_name, operator, occurrence, start, finish, core_event_count FROM aggregate ORDER BY start, component, sk_id, node_id, occurrence"
        for (
            component,
            model_id,
            sk_id,
            node_id,
            raw_name,
            operator,
            occurrence,
            start,
            finish,
            count,
        ) in connection.execute(query):
            if not first:
                destination.write(",")
            first = False
            total += 1
            tid = {"SK E2E": 0, "Cube": 1, "Vector": 2}[component]
            label = "SuperKernel E2E" if component == "SK E2E" else operator
            event = {
                "ph": "X",
                "pid": 0,
                "tid": tid,
                "name": f"{label} | sk={sk_id} node={node_id} occurrence={occurrence}",
                "ts": start,
                "dur": finish - start,
                "args": {
                    "component": component,
                    "modelId": model_id,
                    "skId": sk_id,
                    "nodeId": node_id,
                    "occurrence": occurrence,
                    "merged_core_event_count": count,
                    "raw_kernel_name": raw_name,
                },
            }
            json.dump(
                event,
                destination,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        destination.write("]")
        destination.close()
        os.replace(temporary.name, output)
    except BaseException:
        temporary.close()
        Path(temporary.name).unlink(missing_ok=True)
        raise
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Raw sk_prof_device_*.json trace"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Compact Chrome Trace JSON"
    )
    parser.add_argument(
        "--cube-pid", default="AIC", help="Raw pid label for Cube events (default: AIC)"
    )
    parser.add_argument(
        "--vector-pid",
        default="AIV",
        help="Raw pid label for Vector events (default: AIV)",
    )
    parser.add_argument(
        "--occurrence-gap-us",
        type=float,
        default=100.0,
        help="Start-time gap that starts a new occurrence",
    )
    parser.add_argument(
        "--exclude-superkernel-spans",
        dest="include_superkernel_spans",
        action="store_false",
        default=True,
        help="Omit original SK end-to-end parent spans",
    )
    parser.add_argument(
        "--include-superkernel-spans",
        dest="include_superkernel_spans",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Directory for the temporary SQLite aggregation database",
    )
    parser.add_argument(
        "--keep-temp-db",
        action="store_true",
        help="Keep the temporary SQLite database for inspection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.occurrence_gap_us <= 0:
        raise ValueError("--occurrence-gap-us must be positive")
    connection, database = create_database(args.temp_dir, args.keep_temp_db)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        summary = add_events(
            connection,
            args.input,
            args.cube_pid,
            args.vector_pid,
            args.include_superkernel_spans,
        )
        summary["operator_component_occurrences"] = aggregate_events(
            connection, args.occurrence_gap_us
        )
        summary["compact_trace_events"] = write_trace(connection, args.output)
        summary.update(
            {
                "input": str(args.input),
                "output": str(args.output),
                "cube_pid": args.cube_pid,
                "vector_pid": args.vector_pid,
                "occurrence_gap_us": args.occurrence_gap_us,
                "include_superkernel_spans": args.include_superkernel_spans,
            }
        )
        args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        connection.close()
        if not args.keep_temp_db:
            database.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
