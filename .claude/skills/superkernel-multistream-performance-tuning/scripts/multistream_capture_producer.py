#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run a fingerprinted model plugin and seal a conforming multistream capture."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath

import multistream_operator_order
import multistream_trace_analysis
import multistream_logical_graph
import multistream_critical_path


PLUGIN_API_VERSION = "superkernel-multistream-capture-plugin-v1"
JOB_SCHEMA = "superkernel-multistream-capture-job-v1"
RECEIPT_SCHEMA = "superkernel-multistream-capture-production-receipt-v1"
CAPTURE_SCHEMAS = {
    "logical_graph": multistream_logical_graph.SCHEMA,
    "critical_path": multistream_critical_path.CAPTURE_SCHEMA,
    "short_trace": multistream_trace_analysis.CAPTURE_SCHEMA,
    "operator_order": multistream_operator_order.CAPTURE_SCHEMA,
    "post_dispatch": multistream_operator_order.DISPATCH_CAPTURE_SCHEMA,
}


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value):
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path):
    """Load strict JSON for use by capture plugins."""
    try:
        return json.loads(
            Path(path).read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot load JSON {path}: {error}") from error


def read_csv_rows(path, required_columns=()):
    """Load a CSV with a unique header and require the requested raw columns."""
    try:
        with Path(path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if (
                not fields
                or any(not item for item in fields)
                or len(fields) != len(set(fields))
            ):
                raise ValueError("CSV header must contain unique non-empty columns")
            missing = sorted(set(required_columns) - set(fields))
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {', '.join(missing)}"
                )
            return [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise ValueError(f"cannot load CSV {path}: {error}") from error


def parse_integer(value, label, *, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer >= {minimum}") from error
    if str(value).strip() not in {str(parsed), f"+{parsed}"} or parsed < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return parsed


def parse_number(value, label, *, minimum=0.0, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed) or parsed < minimum or (positive and parsed <= 0):
        raise ValueError(
            f"{label} must be a finite {'positive' if positive else 'non-negative'} number"
        )
    return parsed


def source_file_record(path, artifact_root):
    root = Path(artifact_root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("source file escapes artifact root") from error
    if not path.is_file():
        raise ValueError(f"source file does not exist: {relative}")
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "file_fingerprint": file_fingerprint(path),
    }


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical non-empty string")
    return value


def _relative(value, label):
    value = _text(value, label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"{label} must be a safe relative path")
    return value


def _rooted(root, value, label, *, must_exist=True):
    root = Path(root).resolve()
    relative = _relative(value, label)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error
    if must_exist and not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    return path


def _atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ValueError(f"output already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_plugin(plugin_root, binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "file_fingerprint"}:
        raise ValueError("plugin must contain exactly path and file_fingerprint")
    path = _rooted(plugin_root, binding["path"], "plugin.path")
    actual = file_fingerprint(path)
    if binding["file_fingerprint"] != actual:
        raise ValueError("capture plugin fingerprint mismatch")
    module_name = "_superkernel_capture_plugin_" + actual.split(":", 1)[1]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("capture plugin cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "PLUGIN_API_VERSION", None) != PLUGIN_API_VERSION:
        raise ValueError(f"capture plugin must use {PLUGIN_API_VERSION}")
    plugin_id = _text(getattr(module, "PLUGIN_ID", None), "plugin.PLUGIN_ID")
    capabilities = getattr(module, "CAPABILITIES", None)
    if not isinstance(capabilities, (list, tuple)) or not capabilities:
        raise ValueError("plugin.CAPABILITIES must be a non-empty list")
    capabilities = tuple(_text(item, "plugin.CAPABILITIES[]") for item in capabilities)
    if len(capabilities) != len(set(capabilities)) or set(capabilities) - set(
        CAPTURE_SCHEMAS
    ):
        raise ValueError("plugin.CAPABILITIES contains duplicates or unsupported kinds")
    producer = getattr(module, "produce_capture", None)
    if not callable(producer):
        raise ValueError("capture plugin must define produce_capture(kind, context)")
    return module, plugin_id, capabilities, producer, path, actual


def inspect_plugin(plugin_root, binding):
    _, plugin_id, capabilities, _, path, actual = _load_plugin(plugin_root, binding)
    return {
        "api_version": PLUGIN_API_VERSION,
        "plugin_id": plugin_id,
        "capabilities": list(capabilities),
        "path": path.relative_to(Path(plugin_root).resolve()).as_posix(),
        "file_fingerprint": actual,
    }


def _validate_inputs(root, value):
    if not isinstance(value, dict) or not value:
        raise ValueError("inputs must be a non-empty object")
    result = {}
    for name in sorted(value):
        _text(name, "inputs key")
        binding = value[name]
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "file_fingerprint",
        }:
            raise ValueError(
                f"inputs.{name} must contain exactly path and file_fingerprint"
            )
        path = _rooted(root, binding["path"], f"inputs.{name}.path")
        actual = file_fingerprint(path)
        if binding["file_fingerprint"] != actual:
            raise ValueError(f"input file changed: {binding['path']}")
        result[name] = {
            "path": path,
            "relative_path": path.relative_to(Path(root).resolve()).as_posix(),
            "file_fingerprint": actual,
        }
    return result


def _seal_capture(kind, value):
    if not isinstance(value, dict):
        raise ValueError("capture plugin must return an object")
    if "capture_fingerprint" in value:
        raise ValueError("capture plugin must not provide capture_fingerprint")
    expected_schema = CAPTURE_SCHEMAS[kind]
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"{kind} capture must use {expected_schema}")
    if kind == "logical_graph":
        return multistream_logical_graph.validate(value)
    sealed = json.loads(_canonical(value))
    sealed["capture_fingerprint"] = fingerprint(sealed)
    return sealed


def _capture_fingerprint(kind, capture):
    if kind == "logical_graph":
        return capture["graph_fingerprint"]
    return capture["capture_fingerprint"]


def _conform(kind, capture_path, root, validation):
    if not isinstance(validation, dict):
        raise ValueError("validation must be an object")
    if kind == "short_trace":
        if set(validation) != {"request"}:
            raise ValueError("short_trace validation must contain exactly request")
        request = _rooted(root, validation["request"], "validation.request")
        result = multistream_trace_analysis.analyze(request, capture_path, root)
        return {
            "analyzer": multistream_trace_analysis.ANALYSIS_SCHEMA,
            "target_count": len(result["targets"]),
            "blockers": result["blockers"],
        }
    if kind == "logical_graph":
        if set(validation) != {"request_fingerprint"}:
            raise ValueError(
                "logical_graph validation must contain exactly request_fingerprint"
            )
        graph = multistream_logical_graph.validate(read_json(capture_path))
        expected = _text(
            validation["request_fingerprint"], "validation.request_fingerprint"
        )
        if graph["request_fingerprint"] != expected:
            raise ValueError(
                "logical graph request_fingerprint differs from validation"
            )
        return {
            "analyzer": multistream_logical_graph.SCHEMA,
            "graph_id": graph["graph_id"],
            "join_count": len(graph["joins"]),
        }
    if kind == "critical_path":
        if set(validation) != {"request_fingerprint"}:
            raise ValueError(
                "critical_path validation must contain exactly request_fingerprint"
            )
        result = multistream_critical_path.analyze(read_json(capture_path))
        expected = _text(
            validation["request_fingerprint"], "validation.request_fingerprint"
        )
        if result["request_fingerprint"] != expected:
            raise ValueError(
                "critical path request_fingerprint differs from validation"
            )
        return {
            "analyzer": multistream_critical_path.ANALYSIS_SCHEMA,
            "decision": result["decision"],
            "target_count": len(result["targets"]),
        }
    if kind == "operator_order":
        if set(validation) != {"request_fingerprint"}:
            raise ValueError(
                "operator_order validation must contain exactly request_fingerprint"
            )
        result = multistream_operator_order.analyze(
            capture_path,
            root,
            expected_request_fingerprint=_text(
                validation["request_fingerprint"], "validation.request_fingerprint"
            ),
        )
        return {
            "analyzer": multistream_operator_order.ANALYSIS_SCHEMA,
            "target_count": len(result["targets"]),
            "authorized_target_count": sum(
                item["multistream_reorder_authorized"] for item in result["targets"]
            ),
        }
    if set(validation) != {"action_manifest"}:
        raise ValueError(
            "post_dispatch validation must contain exactly action_manifest"
        )
    action = _rooted(root, validation["action_manifest"], "validation.action_manifest")
    result = multistream_operator_order.build_dispatch_evidence(
        action, capture_path, root
    )
    return {
        "analyzer": multistream_operator_order.DISPATCH_EVIDENCE_SCHEMA,
        "decision": result["decision"],
        "aligned_occurrence_count": result["aligned_occurrence_count"],
    }


def produce(job_path, plugin_root):
    job = read_json(job_path)
    required = {
        "schema_version",
        "job_id",
        "capture_kind",
        "artifact_root",
        "plugin",
        "inputs",
        "parameters",
        "validation",
        "capture_path",
        "receipt_path",
    }
    if (
        not isinstance(job, dict)
        or set(job) != required
        or job.get("schema_version") != JOB_SCHEMA
    ):
        raise ValueError(
            f"capture job must use {JOB_SCHEMA} with exactly {sorted(required)}"
        )
    job_id = _text(job["job_id"], "job_id")
    kind = _text(job["capture_kind"], "capture_kind")
    if kind not in CAPTURE_SCHEMAS:
        raise ValueError(
            f"capture_kind must be one of: {', '.join(sorted(CAPTURE_SCHEMAS))}"
        )
    root = Path(_text(job["artifact_root"], "artifact_root")).resolve()
    if not root.is_dir() or not root.is_absolute():
        raise ValueError("artifact_root must be an existing absolute directory")
    if not isinstance(job["parameters"], dict):
        raise ValueError("parameters must be an object")
    inputs = _validate_inputs(root, job["inputs"])
    _, plugin_id, capabilities, producer, plugin_path, plugin_fp = _load_plugin(
        plugin_root, job["plugin"]
    )
    if kind not in capabilities:
        raise ValueError(f"capture plugin {plugin_id} does not support {kind}")
    capture_path = _rooted(root, job["capture_path"], "capture_path", must_exist=False)
    receipt_path = _rooted(root, job["receipt_path"], "receipt_path", must_exist=False)
    if capture_path == receipt_path or capture_path.exists() or receipt_path.exists():
        raise ValueError("capture and receipt outputs must be distinct and absent")
    context = {
        "job_id": job_id,
        "artifact_root": str(root),
        "inputs": {name: str(binding["path"]) for name, binding in inputs.items()},
        "parameters": json.loads(_canonical(job["parameters"])),
    }
    try:
        produced = producer(kind, context)
    except Exception as error:
        raise ValueError(
            f"capture plugin {plugin_id} failed: {type(error).__name__}: {error}"
        ) from error
    capture = _seal_capture(kind, produced)
    _atomic_json(capture_path, capture)
    try:
        conformance = _conform(kind, capture_path, root, job["validation"])
    except Exception:
        capture_path.unlink(missing_ok=True)
        raise
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "job_id": job_id,
        "job_fingerprint": fingerprint(job),
        "capture_kind": kind,
        "plugin": {
            "plugin_id": plugin_id,
            "api_version": PLUGIN_API_VERSION,
            "path": plugin_path.relative_to(Path(plugin_root).resolve()).as_posix(),
            "file_fingerprint": plugin_fp,
        },
        "inputs": {
            name: {
                "path": binding["relative_path"],
                "file_fingerprint": binding["file_fingerprint"],
            }
            for name, binding in inputs.items()
        },
        "capture": {
            "path": capture_path.relative_to(root).as_posix(),
            "schema_version": capture["schema_version"],
            "capture_fingerprint": _capture_fingerprint(kind, capture),
            "file_fingerprint": file_fingerprint(capture_path),
        },
        "conformance": conformance,
    }
    receipt["receipt_fingerprint"] = fingerprint(receipt)
    try:
        _atomic_json(receipt_path, receipt)
    except Exception:
        capture_path.unlink(missing_ok=True)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-plugin")
    inspect.add_argument("--plugin-root", type=Path, required=True)
    inspect.add_argument("--plugin", type=Path, required=True)
    inspect.add_argument("--plugin-fingerprint", required=True)
    run = commands.add_parser("produce")
    run.add_argument("--job", type=Path, required=True)
    run.add_argument("--plugin-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-plugin":
            result = inspect_plugin(
                args.plugin_root,
                {
                    "path": args.plugin.as_posix(),
                    "file_fingerprint": args.plugin_fingerprint,
                },
            )
        else:
            result = produce(args.job, args.plugin_root)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
