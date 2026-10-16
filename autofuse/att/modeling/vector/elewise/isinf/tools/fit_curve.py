#!/usr/bin/env python3
"""
IsInf perf 拟合工具：读取 samples.csv，按 combo（源码分支）分组拟合 y = a * repeat + b。

样本输入为多组 (loop_count, repeat, measured_cycles)：
    y = measured_cycles / loop_count   （计时循环内单次 API 调用的周期）
    y = a * repeat + b                 （最小二乘，求 a、b 最优值）

每种分支一组独立拟合结果；report.md 总是生成并保留（数据不足的分支在报告中标注原因）。

用法：
    python3 fit_curve.py                       # 读 tools/samples.csv，写 tools/report.md
    python3 fit_curve.py --result r.csv --report report.md
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_result(path):
    """读取 samples.csv，返回 {combo: {"valid": [(loop_count, repeat, measured)...], "invalid": n}}。

    measured_cycles 为空或非正的行计入 invalid（0 值占位数据不参与拟合）。
    """
    groups = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for field in ("combo", "repeat", "loop_count", "measured_cycles"):
            if field not in fields:
                raise SystemExit(
                    "samples file {} missing column: {}".format(path, field)
                )
        for line_no, row in enumerate(reader, start=2):
            combo = (row.get("combo") or "").strip()
            repeat_text = (row.get("repeat") or "").strip()
            loop_text = (row.get("loop_count") or "").strip()
            measured_text = (row.get("measured_cycles") or "").strip()
            if not combo and not repeat_text:
                continue
            entry = groups.setdefault(combo, {"valid": [], "invalid": 0})
            if not measured_text:
                entry["invalid"] += 1
                continue
            loop_count = int(loop_text)
            if loop_count <= 0:
                raise SystemExit(
                    "samples file {} line {}: invalid loop_count: {}".format(
                        path, line_no, loop_text
                    )
                )
            measured = float(measured_text)
            if measured <= 0:
                entry["invalid"] += 1
                continue
            entry["valid"].append((loop_count, int(repeat_text), measured))
    return groups


def fit_linear(samples):
    """最小二乘拟合 y = a * repeat + b，samples 为多组 (loop_count, repeat, measured_cycles)。

    返回 (a, b, r2, rows)；rows 为 [(repeat, loop_count, measured, y, y_pred, resid), ...]。
    repeat 全相同时抛 ValueError。
    """
    points = []
    for loop_count, repeat, measured in samples:
        y = measured / loop_count
        points.append((repeat, loop_count, measured, y))
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[3] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)
    sum_xy = sum(p[0] * p[3] for p in points)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        raise ValueError("all repeats are identical")
    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n

    rows = []
    ss_res = 0.0
    ss_tot = 0.0
    y_mean = sum_y / n
    for repeat, loop_count, measured, y in points:
        y_pred = a * repeat + b
        resid = y - y_pred
        ss_res += resid * resid
        ss_tot += (y - y_mean) * (y - y_mean)
        rows.append((repeat, loop_count, measured, y, y_pred, resid))
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return a, b, r2, rows


def build_report(groups, fits, result_path):
    shown_path = result_path
    if os.path.commonpath([os.path.abspath(result_path), SCRIPT_DIR]) == SCRIPT_DIR:
        shown_path = os.path.relpath(result_path, SCRIPT_DIR)
    lines = ["# IsInf perf 拟合报告", ""]
    lines.append("数据来源：`{}`".format(shown_path))
    lines.append(
        "拟合公式：`y = a * repeat + b`，其中 `y = measured_cycles / loop_count`（单次调用周期）"
    )
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| combo | 有效样本 | 无效样本 | a | b | 公式 | R2 | 状态 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for combo in sorted(groups):
        entry = groups[combo]
        fit = fits.get(combo)
        if fit is None:
            lines.append(
                "| {} | {} | {} | NA | NA | NA | NA | 未拟合 |".format(
                    combo, len(entry["valid"]), entry["invalid"]
                )
            )
        else:
            a, b, r2, rows = fit
            formula = "cycles = {:.6g} * repeat + {:.6g}".format(a, b)
            lines.append(
                "| {} | {} | {} | {:.6g} | {:.6g} | {} | {:.6f} | 已拟合 |".format(
                    combo, len(rows), entry["invalid"], a, b, formula, r2
                )
            )
    lines.append("")

    for combo in sorted(groups):
        entry = groups[combo]
        fit = fits.get(combo)
        lines.append("## {}".format(combo))
        lines.append("")
        lines.append(
            "- 有效样本：{}，无效样本（measured_cycles 为空或非正）：{}".format(
                len(entry["valid"]), entry["invalid"]
            )
        )
        if fit is None:
            reason = (
                "有效样本不足 2 个"
                if len(entry["valid"]) < 2
                else "repeat 全部相同，无法拟合"
            )
            if len(entry["valid"]) == 0 and entry["invalid"] > 0:
                reason += "（measured_cycles 均为空或非正，如为 CPU 仿真占位数据需目标设备实测后再拟合）"
            lines.append("- **未拟合：{}**".format(reason))
            lines.append("")
            continue
        a, b, r2, rows = fit
        lines.append("- **a = {:.6g}，b = {:.6g}**".format(a, b))
        lines.append("- 公式：`cycles = {:.6g} * repeat + {:.6g}`".format(a, b))
        lines.append("- R2 = {:.6f}".format(r2))
        resid_abs = [abs(row[5]) for row in rows]
        resid_rel = [abs(row[5]) / row[3] * 100.0 if row[3] else 0.0 for row in rows]
        lines.append(
            "- 残差：max {:.3f}，mean {:.3f}；相对误差 mean {:.2f}%".format(
                max(resid_abs),
                sum(resid_abs) / len(resid_abs),
                sum(resid_rel) / len(resid_rel),
            )
        )
        lines.append("")
        lines.append(
            "| repeat | loop_count | measured_cycles | y（单次周期） | 预测 y | 残差 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for repeat, loop_count, measured, y, y_pred, resid in rows:
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                    repeat, loop_count, measured, y, y_pred, resid
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="IsInf perf curve fitting")
    parser.add_argument("--result", default=os.path.join(SCRIPT_DIR, "samples.csv"))
    parser.add_argument("--report", default=os.path.join(SCRIPT_DIR, "report.md"))
    args = parser.parse_args()

    if not os.path.isfile(args.result):
        raise SystemExit("samples file not found: {}".format(args.result))
    groups = load_result(args.result)
    if not groups:
        raise SystemExit("no combo found in {}".format(args.result))

    fits = {}
    for combo in sorted(groups):
        entry = groups[combo]
        samples = entry["valid"]
        if len(samples) < 2:
            print("skip {}: only {} valid sample(s)".format(combo, len(samples)))
            continue
        try:
            a, b, r2, rows = fit_linear(samples)
        except ValueError as exc:
            print("skip {}: {}".format(combo, exc))
            continue
        fits[combo] = (a, b, r2, rows)
        print(
            "{}: n={}, a={:.6g}, b={:.6g}, R2={:.6f}".format(combo, len(rows), a, b, r2)
        )

    report = build_report(groups, fits, args.result)
    with open(args.report, "w") as handle:
        handle.write(report)
    print("report: {}".format(args.report))
    if not fits:
        print("no combo fitted, see report for reasons")
        return 1
    if len(fits) < len(groups):
        print(
            "warning: {} of {} combos not fitted".format(
                len(groups) - len(fits), len(groups)
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
