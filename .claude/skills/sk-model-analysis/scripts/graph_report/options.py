"""Source-bound graph configuration; no inferred runtime defaults."""

import copy
import json
import re
from .common import issue


def read_log_options(path, span=None):
    result = {}
    if not path.is_file():
        return result
    with path.open(errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if span and not span["begin_line"] <= line_no <= span["end_line"]:
                continue
            m = re.search(
                r"OptionName:\s*([\w.]+),\s*(?:set\s+)?value(?:\[(\d+)\])?\s*:\s*(.*)",
                line,
            )
            if not m:
                continue
            name, index, raw = m.groups()
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                value = raw.strip()
            row = {"value": value, "source": {"file": str(path), "line": line_no}}
            if index is not None:
                old = result.get(name, {}).get("value", [])
                values = list(old) if isinstance(old, list) else []
                while len(values) <= int(index):
                    values.append(None)
                values[int(index)] = value
                row["value"] = values
            result[name] = row
    return result


def resolve_options(j, log_data, mode, diagnostics):
    out = {}
    for stage in ("before", "after"):
        js = j.get(stage, {}).get("options", {}) if mode != "log" else {}
        ls = log_data.get("options", {}) if mode != "json" else {}
        out[stage] = {k: {**copy.deepcopy(v), "parser": "log"} for k, v in ls.items()}
        for key, value in js.items():
            if key in ls and value.get("value") != ls[key].get("value"):
                diagnostics.append(
                    issue(
                        "graph_option_mismatch",
                        f"{key} 的 JSON 与 Log 记录不一致，采用 JSON。",
                        source=value.get("source", {}),
                        object={"stage": stage},
                        details={"option": key, "json": value, "log": ls[key]},
                    )
                )
            out[stage][key] = {**copy.deepcopy(value), "parser": "json"}
        if mode == "both" and not js and ls:
            diagnostics.append(
                issue(
                    "graph_options_fallback",
                    "该阶段未采集 JSON 图配置，展示 Log 中显式记录的配置。",
                    object={"stage": stage},
                )
            )
    return out
