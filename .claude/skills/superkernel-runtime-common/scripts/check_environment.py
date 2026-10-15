#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check whether the active torch_npu runtime can run SuperKernel experiments."""

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path


OPTIMIZE_OPTION_SAMPLES = {
    "auto_op_parallel": [0, 1],
    "early_start": [0, 1],
    "dcci_disable_on_kernel": [[".*"]],
    "dcci_before_kernel_start": [[".*"]],
    "dcci_after_kernel_end": [[".*"]],
    "aggressive_opt_strategies": [
        {"task_breaker_bypass": 0, "value_breaker_bypass": 0},
        {"task_breaker_bypass": 1, "value_breaker_bypass": 0},
        {"task_breaker_bypass": 0, "value_breaker_bypass": 1},
        {"task_breaker_bypass": 0, "value_breaker_bypass": 2},
        {"task_breaker_bypass": 0, "value_breaker_bypass": 3},
        {
            "event_breaker_bypass": 1,
            "task_breaker_bypass": 0,
            "value_breaker_bypass": 0,
        },
    ],
}
DEBUG_OPTION_SAMPLES = {
    "debug_sync_all": [0, 1],
    "debug_dcci_disable_on_kernel": [[".*"]],
    "dcci_disable_on_kernel": [[".*"]],
    "debug_op_exec_trace": [0, 1],
    "debug_cross_core_sync_check": [0, 1],
    "debug_per_op_max_core_num": [1],
}
CANN_HOME_ENV_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_AICPU_PATH",
)
VISIBLE_DEVICE_ENV_VARS = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
)
CANN_LOADER_ENV_VARS = (
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
)
CANN_VERSION_PATTERN = re.compile(
    r"^(?:Version|version|version_name|CANN_VERSION)\s*=\s*[\"']?([^\"'\s]+)"
)


class RuntimeEnvironmentError(RuntimeError):
    """Raised when the probe is not running in the inference runtime environment."""


def _cann_loader_environment(environ):
    homes = _candidate_cann_homes(environ)
    loader_paths = {
        key: [item for item in environ.get(key, "").split(os.pathsep) if item]
        for key in CANN_LOADER_ENV_VARS
    }
    cann_library_paths = [
        item
        for item in loader_paths["LD_LIBRARY_PATH"]
        if "cann" in item.lower() or "/ascend/" in item.lower()
    ]
    return {
        "cann_home_candidates": homes,
        "cann_library_paths": cann_library_paths,
        "loaded": bool(homes and cann_library_paths),
    }


def _require_cann_loader_environment(environ):
    evidence = _cann_loader_environment(environ)
    if not evidence["loaded"]:
        raise RuntimeEnvironmentError(
            "CANN runtime environment is not loaded. Run this probe after the same "
            "approved CANN setup used by inference (typically source the active "
            "CANN bin/setenv.bash in the probe subprocess). CANN/HCCL paths must be "
            "present in LD_LIBRARY_PATH before importing torch_npu. Option probing "
            "was not run; do not classify any option value as rejected."
        )
    return evidence


def _load_torch():
    import torch
    import torch_npu

    return torch, getattr(torch_npu, "__version__", "unknown")


def _classify_device(device_name):
    if "910_93" in device_name or "A3" in device_name:
        return "A3"
    if "910B" in device_name or "A2" in device_name:
        return "A2"
    return "unsupported"


def _current_python_info():
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable": sys.executable,
    }


def _candidate_cann_homes(environ):
    homes = []
    for key in CANN_HOME_ENV_VARS:
        value = environ.get(key)
        if value and value not in homes:
            homes.append(value)
    opp_path = environ.get("ASCEND_OPP_PATH")
    if opp_path:
        parent = str(Path(opp_path).parent)
        if parent not in homes:
            homes.append(parent)
    return homes


def _read_cann_version(home):
    for relative in ("version.info", "version.cfg", "opp/version.info"):
        version_file = Path(home) / relative
        try:
            lines = version_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        for line in lines:
            match = CANN_VERSION_PATTERN.match(line.strip())
            if match:
                return match.group(1), str(version_file)
    return "unknown", None


def _collect_cann(environ):
    homes = _candidate_cann_homes(environ)
    for home in homes:
        version, version_file = _read_cann_version(home)
        if version != "unknown":
            return {"home": home, "version": version, "version_file": version_file}
    return {
        "home": homes[0] if homes else None,
        "version": "unknown",
        "version_file": None,
    }


def _collect_devices(npu):
    get_name = getattr(npu, "get_device_name", None)
    count_fn = getattr(npu, "device_count", None)
    try:
        if count_fn:
            count = count_fn()
            names = [get_name(index) for index in range(count)] if get_name else []
        else:
            names = [get_name()] if get_name else []
    except Exception as exc:
        return [], f"device topology could not be detected: {exc}"
    return names, None


def _probe_explicit_none_exclusion(npu):
    begin = getattr(npu, "super_kernel_scope_begin", None)
    end = getattr(npu, "super_kernel_scope_end", None)
    if not callable(begin) or not callable(end):
        return {
            "accepted": False,
            "status": "not_run",
            "error": "balanced SuperKernel scope APIs are unavailable",
            "semantic_validation": "not_run",
        }
    try:
        begin(None)
        end(None)
    except Exception as exc:  # The active wrapper controls exception types.
        return {
            "accepted": False,
            "status": "rejected",
            "error": str(exc),
            "semantic_validation": "not_proven_by_api_probe",
        }
    return {
        "accepted": True,
        "status": "accepted",
        "error": None,
        "semantic_validation": "not_proven_by_api_probe",
    }


def _probe_options(graph_type, category, samples):
    validator = getattr(graph_type, "_validate_options", None)
    results = {}
    for name, candidate_values in samples.items():
        if validator is None:
            results[name] = {
                "accepted": False,
                "accepted_values": [],
                "rejected_values": [],
                "error": "_validate_options is unavailable",
            }
            continue
        accepted_values = []
        rejected_values = []
        for value in candidate_values:
            try:
                validator(None, category, {name: value})
            except Exception as exc:  # The active wrapper controls exception types.
                rejected_values.append({"value": value, "error": str(exc)})
            else:
                accepted_values.append(value)
        results[name] = {
            "accepted": bool(accepted_values),
            "accepted_values": accepted_values,
            "rejected_values": rejected_values,
            "error": None
            if accepted_values
            else (
                rejected_values[0]["error"] if rejected_values else "no values tested"
            ),
        }
    return results


def _merge_probe_samples(defaults, custom):
    merged = {name: list(values) for name, values in defaults.items()}
    for name, values in (custom or {}).items():
        if not isinstance(values, list):
            raise ValueError(f"custom probe values for {name} must be a list")
        target = merged.setdefault(name, [])
        for value in values:
            if value not in target:
                target.append(value)
    return merged


def _load_probe_file(path):
    if not path:
        return {}, {}
    data = json.loads(Path(path).read_text(errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("option probe file must contain a JSON object")
    optimize = data.get("optimize_options", {})
    debug = data.get("debug_options", {})
    if not isinstance(optimize, dict) or not isinstance(debug, dict):
        raise ValueError("optimize_options and debug_options must be JSON objects")
    return optimize, debug


def collect_environment(
    torch_module=None,
    check_device=True,
    environ=None,
    python_info=None,
    optimize_option_samples=None,
    debug_option_samples=None,
):
    """Return runtime capabilities without modifying torch or torch_npu."""
    environ = os.environ if environ is None else environ
    python_info = _current_python_info() if python_info is None else python_info
    runtime_environment = _cann_loader_environment(environ)
    if torch_module is None:
        runtime_environment = _require_cann_loader_environment(environ)
        torch_module, torch_npu_version = _load_torch()
    else:
        torch_npu_version = getattr(torch_module, "_torch_npu_version", "unknown")

    npu = getattr(torch_module, "npu", None)
    graph_type = getattr(npu, "NPUGraph", None)
    names, device_limitation = _collect_devices(npu)
    device_name = names[0] if names else "unavailable"
    cann = _collect_cann(environ)
    visible_devices = {
        key: environ[key] for key in VISIBLE_DEVICE_ENV_VARS if environ.get(key)
    }

    apis = {
        "npugraph_ex": hasattr(npu, "npugraph_ex"),
        "static_kernel": graph_type is not None,
        "scope_begin": hasattr(npu, "super_kernel_scope_begin"),
        "scope_end": hasattr(npu, "super_kernel_scope_end"),
    }
    scope_capabilities = {
        "explicit_none_exclusion": _probe_explicit_none_exclusion(npu),
    }
    platform_name = _classify_device(device_name)
    required_apis_present = all(apis.values())
    supported_device = bool(names) and all(
        _classify_device(name) in {"A2", "A3"} for name in names
    )
    ready = required_apis_present and (not check_device or supported_device)
    limitations = []
    if cann["version"] == "unknown":
        limitations.append("CANN version could not be detected")
    if not visible_devices:
        limitations.append("visible-device selection is not set")
    if device_limitation:
        limitations.append(device_limitation)

    return {
        "python": python_info,
        "runtime_environment": runtime_environment,
        "cann": cann,
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "torch_npu_version": torch_npu_version,
        "visible_devices": visible_devices,
        "device": {
            "name": device_name,
            "names": names,
            "count": len(names),
            "platform": platform_name,
        },
        "apis": apis,
        "scope_capabilities": scope_capabilities,
        "options": {
            "optimize_options": _probe_options(
                graph_type,
                "optimize_options",
                _merge_probe_samples(OPTIMIZE_OPTION_SAMPLES, optimize_option_samples),
            ),
            "debug_options": _probe_options(
                graph_type,
                "debug_options",
                _merge_probe_samples(DEBUG_OPTION_SAMPLES, debug_option_samples),
            ),
        },
        "device_check_skipped": not check_device,
        "ready": ready,
        "limitations": limitations,
    }


def _print_human(report):
    print(f"ready: {report['ready']}")
    if "error" in report:
        if report.get("error_code"):
            print(f"error code: {report['error_code']}")
        print(f"error: {report['error']}")
        if report.get("guidance"):
            print(f"guidance: {report['guidance']}")
        return
    print(
        f"python: {report['python']['implementation']} "
        f"{report['python']['version']} ({report['python']['executable']})"
    )
    print(
        f"CANN: {report['cann']['version']} ({report['cann']['home'] or 'unknown home'})"
    )
    print(f"PyTorch: {report['torch_version']}")
    print(f"torch_npu: {report['torch_npu_version']}")
    print(
        f"devices: {report['device']['count']} "
        f"({', '.join(report['device']['names']) or 'unavailable'})"
    )
    print(
        "visible devices: "
        + (
            ", ".join(
                f"{key}={value}" for key, value in report["visible_devices"].items()
            )
            or "not set"
        )
    )
    print(
        "apis: " + ", ".join(f"{key}={value}" for key, value in report["apis"].items())
    )
    none_scope = report["scope_capabilities"]["explicit_none_exclusion"]
    print(
        "explicit None exclusion: "
        + ("accepted" if none_scope["accepted"] else none_scope["status"])
    )
    for category, options in report["options"].items():
        accepted = [
            f"{name}={result['accepted_values']}"
            for name, result in options.items()
            if result["accepted"]
        ]
        print(f"{category}: {', '.join(accepted) if accepted else 'none accepted'}")
    for limitation in report["limitations"]:
        print(f"limitation: {limitation}")


def main(argv=None, torch_module=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="print the full report as JSON"
    )
    parser.add_argument(
        "--no-device-check",
        action="store_true",
        help="do not require an A2 or A3 device for a successful exit",
    )
    parser.add_argument(
        "--option-probes",
        type=Path,
        help=(
            "JSON with optimize_options/debug_options candidate value lists; use this "
            "to validate exact targeted regex or version-specific values"
        ),
    )
    args = parser.parse_args(argv)

    try:
        optimize_samples, debug_samples = _load_probe_file(args.option_probes)
        report = collect_environment(
            torch_module=torch_module,
            check_device=not args.no_device_check,
            optimize_option_samples=optimize_samples,
            debug_option_samples=debug_samples,
        )
        report["custom_option_probe_file"] = (
            str(args.option_probes) if args.option_probes else None
        )
    except RuntimeEnvironmentError as exc:
        report = {
            "ready": False,
            "error_code": "cann_environment_not_loaded",
            "option_probe_status": "not_run",
            "error": str(exc),
            "guidance": (
                "Load the exact approved inference CANN environment in this "
                "subprocess, then rerun the probe."
            ),
        }
    except Exception as exc:
        message = str(exc)
        runtime_load_failure = (
            "cannot open shared object file" in message
            or "Failed to load the backend extension" in message
        )
        report = {
            "ready": False,
            "error_code": (
                "torch_npu_runtime_load_failed"
                if runtime_load_failure
                else "environment_probe_failed"
            ),
            "option_probe_status": "not_run",
            "error": message,
        }
        if runtime_load_failure:
            report["guidance"] = (
                "Verify that the probe inherited the same CANN/HCCL library paths "
                "as inference; this failure is not evidence that an option value "
                "was rejected."
            )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
