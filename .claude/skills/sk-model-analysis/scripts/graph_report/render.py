"""Render a self-contained offline graph report; capture data is always inert."""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

from sk_visualizer_shared import (
    COMMON_GRAPH_FRAME_CSS,
    COMMON_GRAPH_TOOLBAR_CSS,
    COMMON_KEYBOARD_NAV_JS,
    COMMON_TOOLBAR_CSS,
    build_visualizer_styles,
    render_graph_toolbar,
    render_page_header,
)

from .dfx_projection import project_report

COMPRESS_THRESHOLD_BYTES = 8 * 1024 * 1024
_MODEL_SUMMARY_FIELDS = (
    "key",
    "label",
    "process_key",
    "process_label",
    "model_key",
    "model_label",
    "instance_id",
    "asset_path",
    "statistics",
    "generation",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _inert_json(data: str) -> str:
    for char, escaped in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        data = data.replace(char, escaped)
    return data


def _payload(report: dict) -> tuple[str, str]:
    data = _json(report)
    if len(data.encode("utf-8")) <= COMPRESS_THRESHOLD_BYTES or not report.get(
        "models"
    ):
        return _inert_json(data), ""
    del data
    # Keep the hierarchy cheap to parse. Every full model, including raw evidence,
    # remains in its own inert gzip block and is decoded only when selected.
    models = report["models"]
    header = {key: value for key, value in report.items() if key != "models"}
    header["payload_encoding"] = "gzip-base64-models-v1"
    header["models"] = []
    global_diagnostics = []
    diagnostics_by_model: dict[int, list] = {}
    model_keys = {
        str(m.get("key")): i for i, m in enumerate(models) if m.get("key") is not None
    }
    for diagnostic in report.get("diagnostics", []):
        index = (
            model_keys.get(str(diagnostic.get("model_key")))
            if diagnostic.get("model_key") is not None
            else None
        )
        if index is None:
            global_diagnostics.append(diagnostic)
        else:
            diagnostics_by_model.setdefault(index, []).append(diagnostic)
    if "diagnostics" in header:
        header["diagnostics"] = global_diagnostics
    blocks = []
    for index, original in enumerate(models):
        summary = {
            key: original[key] for key in _MODEL_SUMMARY_FIELDS if key in original
        }
        summary["runtime"] = {
            "status": original.get("runtime", {}).get("status", "insufficient")
        }
        header["models"].append(summary)
        model = original
        if index in diagnostics_by_model:
            items = list(original.get("diagnostics", []))
            existing = {_json(item) for item in items}
            for item in diagnostics_by_model[index]:
                fingerprint = _json(item)
                if fingerprint not in existing:
                    items.append(item)
                    existing.add(fingerprint)
            model = {**original, "diagnostics": items}
        packed = base64.b64encode(
            gzip.compress(_json(model).encode("utf-8"), compresslevel=6, mtime=0)
        ).decode("ascii")
        blocks.append(
            f'<script type="application/octet-stream" id="report-model-{index}">{packed}</script>'
        )
    return _inert_json(_json(header)), "".join(blocks)


def render_report(report: dict, *, include_debug: bool = False) -> str:
    assets = Path(__file__).with_name("assets")
    report = project_report(report, include_debug=include_debug)
    if not include_debug:
        # User reports carry per-model conclusions only; global raw diagnostics
        # remain available in the explicit debug profile and structured outputs.
        report["diagnostics"] = []
    data, model_blocks = _payload(report)
    css = build_visualizer_styles(
        COMMON_TOOLBAR_CSS
        + COMMON_GRAPH_TOOLBAR_CSS
        + COMMON_GRAPH_FRAME_CSS
        + (assets / "style.css").read_text(encoding="utf-8")
        + (assets / "dependencies.css").read_text(encoding="utf-8")
        + (assets / "cosmic_blueprint.css").read_text(encoding="utf-8")
    )
    js = (
        (assets / "app.js")
        .read_text(encoding="utf-8")
        .replace("/* SHARED_KEYBOARD_NAV */", COMMON_KEYBOARD_NAV_JS)
        .replace(
            "/* CONTEXT_CONTROLS */",
            (assets / "controls.js").read_text(encoding="utf-8"),
        )
    )
    legacy_start = "/* LEGACY_DFX_START */"
    legacy_end = "/* LEGACY_DFX_END */"
    begin = js.find(legacy_start)
    end = js.find(legacy_end, begin + len(legacy_start)) if begin >= 0 else -1
    if begin >= 0 and end >= 0:
        js = js[:begin] + js[end + len(legacy_end) :]
    js = (assets / "payload.js").read_text(encoding="utf-8") + "\n" + js
    js = (assets / "relations.js").read_text(encoding="utf-8") + "\n" + js
    js = js.replace(
        "/* OBJECT_ACTIONS */",
        (assets / "context_actions.js").read_text(encoding="utf-8"),
    )
    js = js.replace(
        "/* DFX_VIEW */",
        (assets / "dfx_view.js").read_text(encoding="utf-8"),
    )
    js = js.replace(
        "/* SCOPE_CHILD_LAYOUT */",
        (assets / "scope_children.js").read_text(encoding="utf-8"),
    )
    js = js.replace(
        "/* USER_SCOPE_OVERLAY */",
        (assets / "user_scopes.js").read_text(encoding="utf-8"),
    )
    js = js.replace(
        "/* MINIMAP */", (assets / "minimap.js").read_text(encoding="utf-8")
    )
    js = (assets / "taskgraph.js").read_text(encoding="utf-8") + "\n" + js
    js = (assets / "network.js").read_text(encoding="utf-8") + "\n" + js
    js = (assets / "dependencies.js").read_text(encoding="utf-8") + "\n" + js
    header = render_page_header(
        icon="SK",
        kicker="SuperKernel",
        title="Graph 视图",
        note_html="Stream 泳道 · SK 融合与内部任务流",
        stat_badges_html="",
    )
    toolbar = render_graph_toolbar(
        label="当前选择",
        nav_html="",
        select_html="""<label>进程 <select id="process" class="toolbar-input" aria-label="进程"></select></label><span class="context-arrow">›</span><label>Model <select id="model" class="toolbar-input" aria-label="Model"></select></label><span id="model-count" class="context-count"></span>""",
        trailing_html='<span id="parser" class="toolbar-chip"></span><label class="switch graph-only"><input type="checkbox" id="update"><span></span>整图 Update</label><span class="graph-only" id="stage-label">更新前</span>',
        controls_html="""<button id="graph-back" class="toolbar-btn" disabled>← 返回</button><form id="search-form"><input id="search" class="toolbar-input" aria-label="搜索节点或 SK" placeholder="查找 SK / 节点 / sk_…"><button type="submit" class="toolbar-btn">查找</button></form><button id="search-prev" class="toolbar-btn" title="上一个搜索结果">上一项</button><button id="search-next" class="toolbar-btn" title="下一个搜索结果">下一项</button><span id="search-status" role="status"></span>
<button id="zoom-in" class="toolbar-btn" aria-label="放大">放大</button><button id="zoom-out" class="toolbar-btn" aria-label="缩小">缩小</button><button id="reset" class="toolbar-btn">重置视野</button><button id="fit" class="toolbar-btn">适配视图</button><button id="focus" class="toolbar-btn">定位选中</button>
<button id="kbdToggleBtn" class="toolbar-btn" aria-pressed="true">键盘导航：开</button><select id="kbdModeSelect" class="toolbar-input" aria-label="键盘导航模式"><option value="wasd_zoom">WASD：左右平移 + 缩放</option><option value="wasd_pan">WASD：四向平移</option><option value="arrows_zoom">方向键：左右平移 + 缩放</option><option value="vim_zoom">H/L 平移 + K/J 缩放</option></select>

<button id="collapse-scopes" class="toolbar-btn">折叠全部 SK 区域</button><button id="expand-scopes" class="toolbar-btn">展开全部 SK 区域</button><button id="immersive" class="toolbar-btn" aria-pressed="false">专注看图</button>""",
    )
    return (
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SK · Graph & DFX</title><style>"""
        + css
        + """</style></head><body>
<div class="page-shell">
"""
        + header
        + """
<nav class="view-nav" aria-label="报告视图"><div class="view-nav-kicker">视图</div><button class="view-nav-link is-active" id="graph-tab" aria-selected="true">Graph 视图</button><button class="view-nav-link" id="dfx-tab" aria-selected="false">DFX 视图 <span id="issue-count"></span></button></nav>
<main>
"""
        + toolbar
        + """
<section id="model-context"><div id="current-context" class="toolbar-chip" role="status"></div><details id="model-overview" class="surface-card"><summary id="overview-summary">进程 / Model 总览</summary><div id="overview-body" class="table-wrap"></div></details></section>
<section id="graph-view"><div class="summary"><div id="stats" class="stats"></div></div>
<div class="graph-frame"><div class="graph-heading"><h2 id="model-title">完整执行图</h2><details id="graph-options"><summary id="graph-options-summary">图配置</summary><div id="graph-options-body"></div></details><details id="view-settings"><summary>视图设置 / 快捷键</summary><div id="view-settings-body"></div></details></div>
<div id="canvas-toolbar" role="toolbar" aria-label="整图操作"><div class="canvas-toolbar-main"><div id="canvas-search"></div><div id="canvas-navigation"></div></div><div id="canvas-options"><details id="graph-actions"><summary>整图操作</summary><div id="graph-actions-body"><div id="graph-display-options"><label><input id="show-selected-waiting" type="checkbox" checked><span>选中时显示等待连线</span></label><label><input id="show-all-waiting" type="checkbox"><span>显示整图等待连线</span></label><label><input id="show-user-scopes" type="checkbox" checked aria-describedby="user-scope-status"><span>显示用户 Scope 边框<small id="user-scope-status" role="status"></small></span></label></div><div id="graph-batch-actions" aria-label="批量展开与收起"><button id="expand-all-internal" class="toolbar-btn">展开全部内部任务流</button><button id="collapse-all-internal" class="toolbar-btn">收起全部内部任务流</button></div></div></details></div></div>
<div id="node-filters" class="node-filters" aria-label="节点显示过滤"></div><details id="graph-filters" class="graph-filters"><summary>过滤 <span id="filter-status"></span></summary><div id="filter-controls"></div></details>
<div id="selection-bar"><span id="selection-label">单击高亮关联 · 双击查看详情</span><button id="show-details" class="toolbar-btn" disabled>查看详情</button><button id="clear-selection" class="toolbar-btn" disabled>取消选择</button><div id="selection-controls"></div><div id="scope-actions"></div></div>


<div id="canvas-shell" class="svg-container"><canvas id="graph" tabindex="0" aria-label="执行图。拖动平移、滚轮缩放，支持 WASD 键盘导航和 Task 搜索。"></canvas><div id="scope-headers"></div><div id="sk-subgraphs"></div><div id="sk-tools-layer"></div><div id="empty" hidden></div><div class="canvas-note">单击高亮 · 双击详情 · 拖动平移 · WASD 导航</div><div id="position" class="position"></div><details id="minimap-panel" open><summary>缩略图</summary><canvas id="minimap" width="240" height="140" tabindex="0" aria-label="缩略图，点击或拖动定位；方向键移动视野"></canvas><span>点击定位 · 拖动视野框</span></details>
<aside id="inspector" hidden aria-label="Task 详情"><div class="panel-toolbar"><button id="back">← 返回</button><span class="grow"></span><button id="pin" aria-pressed="false">固定</button><button id="expand" aria-expanded="false">展开</button><button id="close" aria-label="关闭详情">×</button></div><div id="panel-body"></div></aside></div>
<footer><div class="graph-legend"><div class="legend"><span><i class="kernel"></i>普通 Kernel</span><span><i class="sync"></i>同步节点</span><span><i class="sk"></i>已融合 / SK</span><span><i class="break-key"></i>导致断开</span><span><i class="failed-key"></i>整体融合失败</span><span><i class="placeholder"></i>占位 / INVALID</span><span><i class="active-key"></i>激活</span><span><i class="related-key"></i>关联</span><span><i class="waiting-key"></i>等待依赖</span><span title="相同 Event 的信号关联，箭头表示通知到等待的方向，不代表已确认唯一消费轮次"><i class="event-association-key"></i>同 Event 关联</span><span><i class="fusion-scale"></i>融合成员：少 → 多</span><span><i class="user-scope-key"></i>用户 Scope 边框</span></div></div><span id="source"></span></footer></div></section>
<section id="dfx-view" hidden><div class="summary"><div><span class="eyebrow">DFX</span><h1>结果状态</h1></div></div><div id="dfx-content"></div></section></main></div>
<script type="application/json" id="report-data">"""
        + data
        + """</script>"""
        + model_blocks
        + """<script>"""
        + js
        + """</script></body></html>"""
    )
