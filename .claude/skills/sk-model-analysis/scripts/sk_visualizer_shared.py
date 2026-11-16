#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared visual building blocks for SK visualizer HTML pages."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as html_escape
from typing import Iterable, Sequence

VISUAL_FONT_UI_STACK = "PingFang SC, Hiragino Sans GB, Noto Sans SC, sans-serif"
VISUAL_FONT_DISPLAY_STACK = "Songti SC, Noto Serif SC, Georgia, serif"
VISUAL_FONT_MONO_STACK = "JetBrains Mono, SFMono-Regular, Consolas, monospace"


@dataclass(frozen=True)
class VisualStatBadge:
    caption: str
    value: str
    foot: str
    tone: str
    dom_id: str = ""
    value_dom_id: str = ""


@dataclass(frozen=True)
class ScopeExplainerEntry:
    kind: str
    label: str
    color: str = ""
    extra_class: str = ""
    hidden: bool = False
    dash_style: str = "solid"


SVG_NODE_TOKENS = {
    "radius_sm": 6,
    "radius_md": 9,
    "radius_lg": 12,
    "radius_pill": 9,
    "border_width": 1.5,
    "border_width_emphasis": 2.0,
    "border_width_active": 3.2,
    "title_font_size": 11,
    "subtitle_font_size": 9,
    "badge_font_size": 10,
    "badge_mono_font_size": 8,
    "queue_label_font_size": 15,
    "lane_title_font_size": 12,
}

SVG_EDGE_TOKENS = {
    "width_stream": 1.5,
    "width_queue": 2.0,
    "width_match": 2.0,
    "width_sync": 2.4,
    "width_emphasis": 2.5,
    "width_all_sync": 3.0,
    "dash_memory": "3,3",
    "dash_event": "6,3",
    "dash_sync": "8,4",
    "dash_wait": "5,3",
    "dash_set": "1,0",
    "arrow_size": 6,
    "label_radius_sm": 7,
    "label_radius_md": 9,
    "label_font_size": 9,
    "label_font_size_sm": 8,
}

SVG_LANE_TOKENS = {
    "queue_bg_radius": 14,
    "queue_label_radius": 12,
    "stripe_radius": 4,
    "line_width": 1.5,
}


COMMON_THEME_CSS = (
    r"""
:root {
    --sk-font-ui: """
    + VISUAL_FONT_UI_STACK
    + r""";
    --sk-font-display: """
    + VISUAL_FONT_DISPLAY_STACK
    + r""";
    --sk-font-mono: """
    + VISUAL_FONT_MONO_STACK
    + r""";
    --sk-bg-0: #edf2f7;
    --sk-bg-1: #e6edf5;
    --sk-bg-2: #f4f7fb;
    --sk-surface: rgba(255,255,255,0.94);
    --sk-surface-strong: rgba(255,255,255,0.98);
    --sk-surface-soft: rgba(248,251,255,0.96);
    --sk-ink: #172033;
    --sk-ink-soft: #243045;
    --sk-muted: #5b667a;
    --sk-line: #d8e1ea;
    --sk-line-strong: rgba(148,163,184,0.22);
    --sk-accent: #1d4ed8;
    --sk-accent-2: #4f46e5;
    --sk-accent-soft: #dbe7ff;
    --sk-accent-wash: linear-gradient(135deg,#eff4ff 0%,#d9e7ff 100%);
    --sk-radius-xl: 24px;
    --sk-radius-lg: 18px;
    --sk-radius-md: 14px;
    --sk-text-xs: 11px;
    --sk-text-sm: 12px;
    --sk-text-md: 13px;
    --sk-text-lg: 15px;
    --sk-text-xl: 18px;
    --sk-text-hero: 24px;
    --sk-text-stat: 28px;
    --sk-weight-medium: 600;
    --sk-weight-bold: 700;
    --sk-weight-heavy: 800;
    --sk-shadow-xl: 0 20px 48px rgba(15,23,42,0.08);
    --sk-shadow-lg: 0 14px 32px rgba(20,33,61,.08);
    --sk-shadow-md: 0 10px 28px rgba(20,33,61,.05);
    --sk-shadow-sm: 0 6px 16px rgba(15,23,42,.04);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: var(--sk-font-ui);
    background:
        radial-gradient(circle at top left, rgba(37,99,235,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(124,58,237,0.08), transparent 24%),
        linear-gradient(180deg, var(--sk-bg-0) 0%, var(--sk-bg-1) 52%, var(--sk-bg-2) 100%);
    color: var(--sk-ink-soft);
    overflow-x: hidden;
    overflow-y: auto;
    min-height: 100vh;
}
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(255,255,255,0.18) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,0.18) 1px, transparent 1px);
    background-size: 24px 24px;
    opacity: 0.35;
    pointer-events: none;
}
h1 {
    color: var(--sk-ink);
    font-family: var(--sk-font-display);
    letter-spacing: -0.02em;
}
h2, h3, h4 {
    color: var(--sk-ink);
    font-family: var(--sk-font-ui);
    font-weight: var(--sk-weight-heavy);
    letter-spacing: -0.01em;
}
p { line-height: 1.6; }
a { color: var(--sk-accent); }
.hint { color: var(--sk-muted); font-size: var(--sk-text-md); }
.mono {
    font-family: var(--sk-font-mono);
    font-size: var(--sk-text-xs);
    word-break: break-all;
}
.eyebrow {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--sk-accent-soft);
    color: var(--sk-accent);
    font-size: var(--sk-text-xs);
    font-weight: var(--sk-weight-heavy);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.surface-card {
    background: var(--sk-surface);
    border: 1px solid var(--sk-line);
    border-radius: var(--sk-radius-lg);
    box-shadow: var(--sk-shadow-md);
}
.surface-card.soft { background: var(--sk-surface-soft); }
.table-wrap {
    margin-top: 12px;
    overflow: auto;
    max-width: 100%;
    border: 1px solid var(--sk-line);
    border-radius: var(--sk-radius-md);
    background: #fff;
}
.table-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--sk-text-md);
}
.table-wrap th,
.table-wrap td {
    border: 1px solid var(--sk-line);
    padding: 7px 9px;
    vertical-align: top;
    overflow-wrap: anywhere;
    word-break: break-word;
}
.table-wrap th {
    background: linear-gradient(180deg, #edf4ff 0%, #e7eef9 100%);
    color: #13233d;
    text-align: left;
    font-size: var(--sk-text-sm);
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}
.summary-metric-card {
    background: var(--sk-surface);
    border-radius: var(--sk-radius-md);
    padding: 14px 16px;
    border: 1px solid var(--sk-line);
    box-shadow: var(--sk-shadow-sm);
    border-left: 4px solid var(--sk-accent);
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-height: 92px;
}
.summary-metric-label {
    font-size: var(--sk-text-xs);
    color: var(--sk-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: var(--sk-weight-heavy);
}
.summary-metric-value {
    font-size: 24px;
    font-weight: var(--sk-weight-heavy);
    color: var(--sk-ink);
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.summary-metric-note {
    font-size: var(--sk-text-sm);
    color: var(--sk-muted);
    line-height: 1.45;
}
"""
)


COMMON_VISUALIZER_CSS = (
    COMMON_THEME_CSS
    + r"""
.page-shell {
    width: min(1760px, calc(100vw - 32px));
    margin: 18px auto 28px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    position: relative;
    z-index: 1;
}
.header {
    background: linear-gradient(145deg, rgba(255,255,255,0.94) 0%, rgba(245,248,252,0.92) 100%);
    border: 1px solid var(--sk-line-strong);
    border-radius: 22px;
    padding: 18px 22px;
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
    gap: 18px;
    box-shadow: var(--sk-shadow-xl);
    position: relative;
    overflow: hidden;
}
.header::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        linear-gradient(135deg, rgba(37,99,235,0.12), transparent 36%),
        linear-gradient(315deg, rgba(124,58,237,0.10), transparent 34%);
    pointer-events: none;
}
.header-left {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    position: relative;
    z-index: 1;
}
.header-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: calc(var(--sk-text-lg) + 2px);
    font-weight: var(--sk-weight-bold);
    color: #fff;
    box-shadow: 0 12px 24px rgba(37,99,235,0.22);
    flex-shrink: 0;
}
.header-copy {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.header h1 {
    font-size: var(--sk-text-hero);
    color: var(--sk-ink);
    font-weight: var(--sk-weight-heavy);
    letter-spacing: 0.01em;
    line-height: 1.1;
}
.header-kicker {
    font-size: var(--sk-text-xs);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sk-muted);
    font-weight: var(--sk-weight-heavy);
}
.header-note {
    color: var(--sk-muted);
    font-size: var(--sk-text-md);
}
.hero-stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    position: relative;
    z-index: 1;
    align-self: stretch;
}
.stat-badge {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 92px;
    padding: 14px 16px;
    border-radius: 18px;
    font-weight: var(--sk-weight-bold);
    letter-spacing: 0.02em;
    border: 1px solid rgba(148,163,184,0.18);
    backdrop-filter: blur(8px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.42);
}
.stat-badge .stat-caption {
    font-size: var(--sk-text-xs);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    opacity: 0.86;
}
.stat-badge .stat-value {
    font-size: var(--sk-text-stat);
    line-height: 1;
    margin-top: 8px;
}
.stat-badge .stat-foot {
    font-size: var(--sk-text-sm);
    margin-top: 8px;
    opacity: 0.78;
}
.stat-aic {
    background: linear-gradient(180deg, rgba(111,151,177,0.12), rgba(111,151,177,0.05));
    color: #35566c;
    border-color: rgba(111,151,177,0.18);
}
.stat-aiv {
    background: linear-gradient(180deg, rgba(154,132,175,0.12), rgba(154,132,175,0.05));
    color: #654f78;
    border-color: rgba(154,132,175,0.18);
}
.stat-total {
    background: linear-gradient(180deg, rgba(120,150,125,0.12), rgba(120,150,125,0.05));
    color: #45604a;
    border-color: rgba(120,150,125,0.18);
}
.view-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 18px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 14px 28px rgba(15,23,42,0.05);
}
.view-nav-link {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.22);
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    color: #334155;
    text-decoration: none;
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-bold);
    transition: all 0.16s ease;
}
.view-nav-link:hover {
    color: #1d4ed8;
    border-color: rgba(59,130,246,0.28);
    background: #eef4ff;
}
.view-nav-link.is-active {
    color: #ffffff;
    background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%);
    border-color: rgba(79,70,229,0.48);
    box-shadow: 0 10px 22px rgba(37,99,235,0.22);
}
.view-nav-kicker {
    font-size: var(--sk-text-xs);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sk-muted);
    font-weight: var(--sk-weight-heavy);
    margin-right: 6px;
    align-self: center;
}
@media (max-width: 1100px) {
    .page-shell { width: min(100vw - 18px, 100%); margin: 10px auto 20px; }
    .header { grid-template-columns: 1fr; }
    .hero-stats { grid-template-columns: 1fr; }
}
"""
)


COMMON_TOOLBAR_CSS = r"""
.toolbar {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 20px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    box-shadow: 0 14px 28px rgba(15,23,42,0.05);
}
.toolbar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.toolbar-label {
    color: #334155;
    font-size: var(--sk-text-xs);
    font-weight: var(--sk-weight-heavy);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.toolbar-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 12px;
    border-radius: 999px;
    background: #e8eff8;
    color: #23405c;
    font-size: var(--sk-text-xs);
    font-weight: var(--sk-weight-heavy);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.toolbar-input {
    min-width: 220px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.24);
    background: #ffffff;
    color: #334155;
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-medium);
}
.toolbar-input:focus {
    outline: none;
    border-color: rgba(59,130,246,0.34);
    box-shadow: 0 0 0 3px rgba(59,130,246,0.10);
}
.toolbar-btn {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid rgba(148,163,184,0.24);
    color: #334155;
    padding: 7px 14px;
    border-radius: 999px;
    cursor: pointer;
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-bold);
    transition: all 0.16s ease;
}
.toolbar-btn:hover {
    background: #eef4ff;
    color: #1d4ed8;
    border-color: rgba(59,130,246,0.28);
}
.toolbar-btn:active { transform: scale(0.97); }
"""


COMMON_GRAPH_TOOLBAR_CSS = r"""
.graph-toolbar-primary {
    justify-content: space-between;
    gap: 16px;
}
.graph-toolbar-left,
.graph-toolbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    min-width: 0;
}
.graph-toolbar-right {
    justify-content: flex-end;
}
.graph-nav {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px;
    border-radius: 999px;
    background: #f8fafc;
    border: 1px solid rgba(148,163,184,0.20);
}
.graph-nav-btn {
    width: 34px;
    height: 34px;
    border: none;
    border-radius: 999px;
    background: #ffffff;
    color: #334155;
    font-size: var(--sk-text-xl);
    font-weight: var(--sk-weight-bold);
    cursor: pointer;
    box-shadow: 0 1px 2px rgba(15,23,42,0.08);
}
.graph-nav-btn:hover {
    background: #eef4ff;
    color: #1d4ed8;
}
.graph-nav-btn:disabled {
    opacity: 0.38;
    cursor: not-allowed;
}
.graph-index-chip {
    white-space: nowrap;
}
.graph-select {
    min-width: 320px;
}
.graph-meta-note {
    color: #6b7280;
    font-size: var(--sk-text-sm);
}
"""


COMMON_TABLE_CONTROLS_CSS = r"""
.table-search, .table-filter, .detail-search, .detail-filter {
    border: 1px solid rgba(148,163,184,0.28);
    background: rgba(255,255,255,0.92);
    color: #334155;
    border-radius: 12px;
    padding: 7px 11px;
    font-size: var(--sk-text-sm);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
}
.table-search, .detail-search { min-width: 220px; }
.table-pager, .detail-pager {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    padding-top:10px;
    flex-wrap:wrap;
}
.table-pager-main, .table-pager-side, .detail-pager-main, .detail-pager-side {
    display:flex;
    align-items:center;
    gap:8px;
    flex-wrap:wrap;
}
.table-pager-btn, .table-page-size, .detail-pager-btn, .detail-page-size {
    border: 1px solid rgba(148,163,184,0.28);
    background: rgba(255,255,255,0.92);
    color: #334155;
    border-radius: 12px;
    padding: 7px 11px;
    font-size: var(--sk-text-sm);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
}
.table-pager-btn, .detail-pager-btn { cursor:pointer; font-weight:var(--sk-weight-medium); }
.table-pager-btn:disabled, .detail-pager-btn:disabled { opacity:.45; cursor:not-allowed; }
.table-page-status, .detail-page-status { color: var(--sk-muted); font-size: var(--sk-text-sm); }
.table-page-jump, .detail-page-jump {
    display:flex;
    align-items:center;
    gap:6px;
    color: var(--sk-muted);
    font-size: var(--sk-text-sm);
}
.table-jump-input, .detail-jump-input {
    width: 72px;
    border: 1px solid rgba(148,163,184,0.28);
    background: rgba(255,255,255,0.92);
    color: #334155;
    border-radius: 12px;
    padding: 7px 11px;
    font-size: var(--sk-text-sm);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
}
.table-jump-btn, .detail-jump-btn {
    border: 1px solid rgba(148,163,184,0.28);
    background: rgba(255,255,255,0.92);
    color: #334155;
    border-radius: 12px;
    padding: 7px 11px;
    font-size: var(--sk-text-sm);
    cursor: pointer;
    font-weight: var(--sk-weight-medium);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
}
.table-shell-tools {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    flex-wrap:wrap;
    margin-bottom:14px;
}
.table-shell-tools-main, .table-shell-tools-side {
    display:flex;
    align-items:center;
    gap:8px;
    flex-wrap:wrap;
}
"""


COMMON_DETAIL_TABLE_CSS = r"""
.detail-panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,251,255,0.94) 100%);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 14px 16px 16px;
    box-shadow: 0 18px 34px rgba(15,23,42,0.05);
    overflow: hidden;
}
.detail-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin: -2px -4px 10px;
    padding: 10px 12px;
    border-radius: 16px;
    cursor: pointer;
    transition: background .16s ease, box-shadow .16s ease;
    user-select: none;
}
.detail-head:hover {
    background: rgba(239,244,255,0.82);
    box-shadow: inset 0 0 0 1px rgba(191,219,254,0.55);
}
.detail-head.is-compact {
    padding: 8px 10px;
    justify-content: flex-end;
    background: rgba(248,250,252,0.58);
}
.detail-head-main {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    min-width: 0;
    flex: 1 1 auto;
}
.detail-head.is-compact .detail-head-side {
    width: 100%;
    justify-content: flex-end;
}
.detail-title {
    font-size: var(--sk-text-lg);
    font-weight: var(--sk-weight-heavy);
    color: #1f2a40;
    margin: 0;
    letter-spacing: -0.01em;
}
.detail-subtitle {
    color: #64748b;
    font-size: var(--sk-text-sm);
    margin: 0 0 12px;
    min-width: 0;
    overflow-wrap: anywhere;
    word-break: break-word;
    padding: 0 12px;
}
.detail-head-main .detail-subtitle {
    padding: 0;
    margin: 0;
}
.detail-head-side {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
    color: #64748b;
    flex-shrink: 0;
}
.detail-head-state {
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    font-weight: var(--sk-weight-heavy);
}
.detail-head-arrow {
    width: 22px;
    height: 22px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(148,163,184,0.24);
    color: #475569;
    font-size: 11px;
    line-height: 1;
    transition: transform .16s ease, background .16s ease, border-color .16s ease;
}
.detail-panel.is-open .detail-head-arrow {
    transform: rotate(180deg);
}
.detail-head:hover .detail-head-arrow {
    background: rgba(239,244,255,0.96);
    border-color: rgba(147,197,253,0.65);
}
.detail-tools {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}
.detail-content.is-collapsed {
    display: none;
}
.detail-table-wrap {
    overflow: auto;
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 16px;
    background: #fff;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
}
.detail-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--sk-text-sm);
}
.detail-table th {
    text-align: left;
    padding: 11px 13px;
    background: linear-gradient(180deg, #eef4ff 0%, #e7eef9 100%);
    color: #334155;
    border-bottom: 1px solid rgba(148,163,184,0.16);
    position: sticky;
    top: 0;
    z-index: 1;
    font-weight: var(--sk-weight-bold);
}
.detail-table td {
    padding: 10px 13px;
    border-bottom: 1px solid rgba(226,232,240,0.8);
    vertical-align: top;
    color: #334155;
}
.detail-table tr:hover { background: #f8fbff; }
.detail-snippet {
    max-width: 360px;
    white-space: normal;
}
"""


COMMON_TOOLTIP_CSS = r"""
.tooltip, .tip {
    position: fixed;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: var(--sk-text-sm);
    line-height: 1.7;
    max-width: 480px;
    pointer-events: none;
    z-index: 1000;
    display: none;
    box-shadow: 0 8px 32px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.02);
}
.tooltip .tt-header {
    padding-bottom: 8px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(0,0,0,0.06);
}
.tooltip .tt-title {
    font-weight: var(--sk-weight-bold);
    font-size: var(--sk-text-lg);
    margin-bottom: 2px;
    color: #2c3140;
}
.tooltip .tt-queue {
    font-size: var(--sk-text-xs);
    font-weight: var(--sk-weight-medium);
    padding: 2px 8px;
    border-radius: 10px;
    display: inline-block;
}
.tooltip .tt-row {
    display: flex;
    justify-content: space-between;
    gap: 20px;
}
.tooltip .tt-label { color: #8892a4; font-size: var(--sk-text-xs); }
.tooltip .tt-value {
    color: #3a3f4b;
    font-family: var(--sk-font-mono);
    font-size: var(--sk-text-xs);
}
.tip b { color:#81d4fa; }
"""


COMMON_META_STRIP_CSS = r"""
.meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}
.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(148,163,184,0.18);
    color: #334155;
    font-size: var(--sk-text-sm);
    box-shadow: var(--sk-shadow-sm);
}
.meta-chip-label {
    color: var(--sk-muted);
    font-weight: var(--sk-weight-bold);
    letter-spacing: .04em;
}
.meta-chip-value {
    color: var(--sk-ink);
    font-weight: var(--sk-weight-bold);
}
"""


COMMON_EMPTY_STATE_CSS = r"""
.empty-note {
    margin-top: 12px;
    padding: 14px 16px;
    border-radius: 14px;
    border: 1px dashed rgba(148,163,184,0.32);
    background: linear-gradient(180deg, rgba(248,251,255,.96) 0%, rgba(243,248,255,.92) 100%);
    color: var(--sk-muted);
    font-size: var(--sk-text-md);
}
"""


COMMON_INLINE_LEGEND_CSS = r"""
.inline-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 14px 20px;
    background: var(--sk-surface);
    border-radius: var(--sk-radius-md);
    margin-bottom: 16px;
    box-shadow: var(--sk-shadow-sm);
    font-size: var(--sk-text-md);
    align-items: center;
    border: 1px solid var(--sk-line);
}
.inline-legend-title {
    font-weight: var(--sk-weight-bold);
    color: var(--sk-ink-soft);
}
.inline-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #334155;
}
.inline-legend-box {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    flex-shrink: 0;
}
"""


COMMON_GRAPH_FRAME_CSS = r"""
.graph-frame {
    background: var(--sk-surface);
    border-radius: var(--sk-radius-md);
    padding: 20px;
    box-shadow: var(--sk-shadow-sm);
    margin-bottom: 20px;
    border: 1px solid var(--sk-line);
}
.graph-frame > h2 {
    font-size: var(--sk-text-lg);
    color: var(--sk-ink);
    margin-bottom: 10px;
}
.svg-container {
    overflow: hidden;
    position: relative;
    cursor: grab;
    background:
        linear-gradient(180deg, rgba(248,250,252,0.96), rgba(240,245,250,0.94)),
        #f2f5f8;
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 24px;
    box-shadow: 0 20px 44px rgba(15,23,42,0.07);
    min-height: clamp(700px, 76vh, 1080px);
}
.svg-container:active { cursor: grabbing; }
.svg-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(circle at top left, rgba(255,255,255,0.48), transparent 24%),
        linear-gradient(90deg, rgba(148,163,184,0.07) 1px, transparent 1px),
        linear-gradient(rgba(148,163,184,0.07) 1px, transparent 1px);
    background-size: auto, 24px 24px, 24px 24px;
    pointer-events: none;
    z-index: 0;
}
.svg-container svg {
    position: absolute;
    top: 0;
    left: 0;
    transform-origin: 0 0;
    z-index: 1;
    shape-rendering: geometricPrecision;
    text-rendering: geometricPrecision;
    display: block;
}
.svg-container text { user-select: none; }
.zoom-toolbar {
    position: absolute;
    top: 8px; right: 8px; z-index: 10;
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: rgba(255,255,255,.92);
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,.15);
    padding: 4px;
}
.zoom-toolbar button {
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 6px;
    background: none;
    cursor: pointer;
    font-size: var(--sk-text-xl);
    color: #555;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background .15s;
}
.zoom-toolbar button:hover { background: #e8e8e8; }
.zoom-toolbar button:active { background: #d0d0d0; }
.zoom-toolbar .zoom-lvl {
    text-align: center;
    font-size: 10px;
    color: #999;
    font-family: var(--sk-font-mono);
    padding: 2px 0;
    user-select: none;
}
"""


COMMON_LEGEND_CSS = r"""
.legend {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 20px;
    padding: 16px 18px;
    font-size: var(--sk-text-xs);
    box-shadow: 0 14px 28px rgba(15,23,42,0.05);
}
.legend-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}
.legend-title {
    font-size: var(--sk-text-xs);
    margin-bottom: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: var(--sk-weight-heavy);
}
.legend-section {
    margin-bottom: 10px;
    padding: 12px 12px 10px;
    border-radius: 14px;
    background: #f8fbff;
    border: 1px solid rgba(148,163,184,0.14);
}
.legend-section:last-child { margin-bottom: 0; }
.legend-item {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
    color: #334155;
}
.legend-color {
    width: 16px;
    height: 12px;
    border-radius: 3px;
    flex-shrink: 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.legend-line {
    width: 28px;
    height: 0;
    border-top: 2px dashed #8f6fd1;
    flex-shrink: 0;
    margin-top: 1px;
}
"""

# Canonical explainer pattern sourced from the original Scope View.
COMMON_SCOPE_SECTION_CSS = r"""
.scope-explainer {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 14px 20px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 20px;
    box-shadow: 0 14px 28px rgba(15,23,42,0.05);
    font-size: var(--sk-text-sm);
    align-items: center;
}
.scope-explainer-group {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.scope-explainer-title {
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-bold);
    color: #64748b;
    white-space: nowrap;
}
.scope-explainer-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #334155;
}
.scope-explainer-box {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    flex-shrink: 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.scope-explainer-line {
    width: 30px;
    height: 0;
    border-top: 2px dashed #8f6fd1;
    flex-shrink: 0;
}
"""


COMMON_REPORT_CSS = (
    COMMON_THEME_CSS
    + COMMON_META_STRIP_CSS
    + COMMON_TABLE_CONTROLS_CSS
    + COMMON_DETAIL_TABLE_CSS
    + r"""
.page-shell {
    width: min(1760px, calc(100vw - 32px));
    margin: 18px auto 28px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    position: relative;
    z-index: 1;
}
.header {
    background: linear-gradient(145deg, rgba(255,255,255,0.94) 0%, rgba(245,248,252,0.92) 100%);
    border: 1px solid var(--sk-line-strong);
    border-radius: 22px;
    padding: 18px 22px;
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
    gap: 18px;
    box-shadow: var(--sk-shadow-xl);
    position: relative;
    overflow: hidden;
}
.header::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        linear-gradient(135deg, rgba(37,99,235,0.12), transparent 36%),
        linear-gradient(315deg, rgba(124,58,237,0.10), transparent 34%);
    pointer-events: none;
}
.header-left {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    position: relative;
    z-index: 1;
}
.header-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: calc(var(--sk-text-lg) + 2px);
    font-weight: var(--sk-weight-bold);
    color: #fff;
    box-shadow: 0 12px 24px rgba(37,99,235,0.22);
    flex-shrink: 0;
}
.header-copy {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.header h1 {
    font-size: var(--sk-text-hero);
    color: var(--sk-ink);
    font-weight: var(--sk-weight-heavy);
    letter-spacing: 0.01em;
    line-height: 1.1;
}
.header-kicker {
    font-size: var(--sk-text-xs);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sk-muted);
    font-weight: var(--sk-weight-heavy);
}
.header-note {
    color: var(--sk-muted);
    font-size: var(--sk-text-md);
    max-width: 920px;
    line-height: 1.55;
}
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; margin-top: 18px; }
.grid-tight { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 18px; }
.card {
    background: var(--sk-surface);
    border: 1px solid var(--sk-line);
    border-radius: var(--sk-radius-lg);
    padding: 18px;
    box-shadow: var(--sk-shadow-md);
}
.card.primary { border-top: 5px solid var(--sk-accent); }
.card.secondary { border-top: 5px solid #cbd5e1; }
.card.tertiary { background: var(--sk-surface-soft); }
.subblock { margin-top: 14px; }
.report-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
    margin-top: 14px;
}
.report-summary.is-compact {
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 8px;
    margin-top: 12px;
}
.report-summary-item {
    background: linear-gradient(180deg, rgba(248,251,255,.98) 0%, rgba(241,246,252,.96) 100%);
    border: 1px solid rgba(191,204,217,.42);
    border-radius: 14px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.report-summary-label {
    font-size: var(--sk-text-xs);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    font-weight: var(--sk-weight-heavy);
}
.report-summary-value {
    color: var(--sk-ink);
    font-size: var(--sk-text-md);
    line-height: 1.45;
    font-weight: var(--sk-weight-bold);
}
.report-summary.is-compact .report-summary-item {
    border-radius: 12px;
    padding: 10px 12px;
    gap: 4px;
}
.report-summary.is-compact .report-summary-value {
    font-size: var(--sk-text-sm);
    line-height: 1.4;
}
.report-intro {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.info-panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,251,255,0.94) 100%);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 14px 16px 16px;
    box-shadow: 0 18px 34px rgba(15,23,42,0.05);
    display: flex;
    flex-direction: column;
    gap: 12px;
    position: relative;
    z-index: 1;
}
.info-panel.is-compact {
    gap: 10px;
}
.info-panel-head {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 2px 2px 0;
}
.info-panel-title {
    margin: 0;
    font-size: var(--sk-text-lg);
    font-weight: var(--sk-weight-heavy);
    color: #1f2a40;
    letter-spacing: -0.01em;
}
.info-panel-note {
    margin: 0;
    color: var(--sk-muted);
    font-size: var(--sk-text-sm);
    line-height: 1.6;
    max-width: 980px;
}
.info-panel-body {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.info-panel.is-compact .info-panel-body {
    gap: 10px;
}
.info-panel .meta-strip {
    margin-top: 0;
}
.report-section {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.report-section-head {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.report-section-title {
    margin: 0;
}
.report-section-note {
    color: var(--sk-muted);
    font-size: var(--sk-text-sm);
    line-height: 1.6;
    margin: 0;
}
.report-section-body {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.view-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 18px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 14px 28px rgba(15,23,42,0.05);
    position: relative;
    z-index: 1;
}
.view-nav-link {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.22);
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    color: #334155;
    text-decoration: none;
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-bold);
    transition: all 0.16s ease;
}
.view-nav-link:hover {
    color: #1d4ed8;
    border-color: rgba(59,130,246,0.28);
    background: #eef4ff;
}
.view-nav-link.is-active {
    color: #ffffff;
    background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%);
    border-color: rgba(79,70,229,0.48);
    box-shadow: 0 10px 22px rgba(37,99,235,0.22);
}
.view-nav-kicker {
    font-size: var(--sk-text-xs);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sk-muted);
    font-weight: var(--sk-weight-heavy);
    margin-right: 6px;
    align-self: center;
}
code { background: #eef4fb; padding: 2px 5px; border-radius: 6px; }
.decision-list, .compact-list { margin: 0; padding-left: 18px; }
.decision-list li { margin: 10px 0; }
.compact-list li { margin: 8px 0; }
details.fold {
    margin-top: 16px;
    border: 1px solid var(--sk-line);
    border-radius: var(--sk-radius-md);
    background: #fbfdff;
}
details.fold > summary {
    cursor: pointer;
    list-style: none;
    font-weight: var(--sk-weight-bold);
    padding: 14px 16px;
}
details.fold > summary::-webkit-details-marker { display: none; }
details.fold > div { padding: 0 16px 16px; }
.capability-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    background: #eef4fb;
    color: #1f3a5b;
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-bold);
    margin: 6px 8px 0 0;
    border: 1px solid rgba(31,79,209,.08);
}
.priority-box {
    background: var(--sk-accent-wash);
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 16px;
}
.priority-box strong { color: var(--sk-accent); }
.muted-card {
    background: var(--sk-surface-soft);
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 14px;
}
.eyebrow-muted {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    background: #e2e8f0;
    color: #475569;
    font-weight: var(--sk-weight-bold);
    font-size: var(--sk-text-xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.entry-list { list-style: none; padding: 0; margin: 0; }
.entry-list li {
    margin: 12px 0 0;
    padding-top: 12px;
    border-top: 1px solid rgba(191,204,217,.45);
}
.entry-list li:first-child { margin-top: 0; padding-top: 0; border-top: 0; }
.entry-reason { margin-top: 6px; color: var(--sk-muted); font-size: var(--sk-text-md); }
.report-card h3 { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: var(--sk-text-xs);
    font-weight: var(--sk-weight-heavy);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.status-pill.ok { background: #dbeafe; color: #1e40af; }
.status-pill.failed { background: #fee2e2; color: #b91c1c; }
.entry-grid { display: grid; gap: 18px; margin-top: 18px; }
.entry-grid.primary { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.entry-grid.secondary { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.graph-panel { overflow: hidden; }
.graph-frame-wrap {
    margin-top: 14px;
    border: 1px solid var(--sk-line);
    border-radius: 16px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.4);
}
.graph-frame-wrap iframe {
    display: block;
    width: 100%;
    min-height: 720px;
    border: 0;
    background: #fff;
}
.paged-table-wrap {
    overflow: hidden;
    width: 100%;
    max-width: 100%;
    margin: 12px 0 0;
}
.paged-table-wrap.prominent { border-color: #bfd3eb; box-shadow: var(--sk-shadow-lg); }
.paged-table { width: 100%; border-collapse: collapse; }
.paged-table thead th {
    background: linear-gradient(180deg, #edf4ff 0%, #e2ecfb 100%);
    color: #13233d;
    font-size: var(--sk-text-sm);
    letter-spacing: .02em;
    text-transform: uppercase;
    text-align: center;
}
.paged-table.prominent thead th {
    background: linear-gradient(180deg, #dcecff 0%, #cfe2ff 100%);
    color: #16325c;
}
.paged-table tbody td { text-align: center; vertical-align: middle; }
.paged-table tbody tr:nth-child(even) td { background: rgba(247,250,253,.92); }
.paged-table tbody tr:hover td { background: rgba(231,240,255,.78); }
.table-pager {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--sk-line);
    background: linear-gradient(180deg, rgba(249,251,255,.98) 0%, rgba(242,247,253,.96) 100%);
    flex-wrap: wrap;
}
.table-pager-meta { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.table-pager-title { font-size: var(--sk-text-sm); text-transform: uppercase; letter-spacing: .08em; color: #355070; }
.table-pager-meta span { color: var(--sk-muted); font-size: var(--sk-text-md); }
.table-pager-actions { display: flex; align-items: center; gap: 8px; }
.table-pager-btn {
    appearance: none;
    border: 1px solid #c8d5e6;
    background: #fff;
    color: var(--sk-accent);
    border-radius: 999px;
    padding: 6px 12px;
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-bold);
    cursor: pointer;
}
.table-pager-btn[disabled] {
    color: #94a3b8;
    border-color: #e2e8f0;
    cursor: not-allowed;
    background: #f8fafc;
}
.dfx-stack, .raw-evidence-stack {
    display: grid;
    gap: 18px;
}
.dfx-stack {
    margin-top: 18px;
}
.dfx-stack > *,
.raw-evidence-stack > * {
    width: 100%;
    max-width: 100%;
    margin-left: 0;
    margin-right: 0;
    text-align: left;
}
.focus-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit,minmax(220px,1fr));
    gap: 14px;
    margin-top: 18px;
}
.focus-card {
    background: linear-gradient(180deg, rgba(255,255,255,.98) 0%, rgba(243,248,255,.96) 100%);
    border: 1px solid rgba(191,204,217,.62);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 10px 24px rgba(20,33,61,.04);
}
.focus-card strong { display: block; font-size: var(--sk-text-lg); color: var(--sk-ink); margin-bottom: 8px; }
.focus-card p { margin: 4px 0 0; color: var(--sk-muted); font-size: var(--sk-text-md); }
.focus-card .focus-kv { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.focus-card .focus-kv span {
    display: inline-flex;
    padding: 5px 9px;
    border-radius: 999px;
    background: #eef4fb;
    color: #334155;
    font-size: var(--sk-text-sm);
    font-weight: var(--sk-weight-bold);
}
.focus-card.primary { border-color: #bfdbfe; box-shadow: 0 14px 30px rgba(29,78,216,.08); }
.focus-card.muted { background: linear-gradient(180deg, rgba(251,253,255,.98) 0%, rgba(247,250,253,.96) 100%); }
.stage-grid {
    display: grid;
    grid-template-columns: minmax(0,1.6fr) minmax(280px,.9fr);
    gap: 18px;
    align-items: start;
    margin-top: 18px;
}
.stage-side { display: grid; gap: 14px; }
@media (max-width: 1100px) {
    .page-shell { width: min(100vw - 18px, 100%); margin: 10px auto 20px; }
    .header { grid-template-columns: 1fr; }
    .stage-grid { grid-template-columns: 1fr; }
}
"""
)


def build_visualizer_styles(extra_css: str = "") -> str:
    return COMMON_VISUALIZER_CSS + ("\n" + extra_css if extra_css else "")


def build_report_styles(extra_css: str = "") -> str:
    return COMMON_REPORT_CSS + ("\n" + extra_css if extra_css else "")


def render_meta_strip(items: Sequence[tuple[str, str]]) -> str:
    chips: list[str] = []
    for label, value in items:
        if not str(value).strip():
            continue
        chips.append(
            "<div class='meta-chip'>"
            f"<span class='meta-chip-label'>{html_escape(label)}</span>"
            f"<span class='meta-chip-value'>{html_escape(value)}</span>"
            "</div>"
        )
    if not chips:
        return ""
    return f"<div class='meta-strip'>{''.join(chips)}</div>"


def render_empty_note(message: str, dom_id: str = "", hidden: bool = False) -> str:
    dom_attr = f' id="{html_escape(dom_id)}"' if dom_id else ""
    style_attr = ' style="display:none;"' if hidden else ""
    return f"<div class='empty-note'{dom_attr}{style_attr}>{html_escape(message)}</div>"


def render_inline_legend(items: Sequence[str], title: str = "图例:") -> str:
    rows = "".join(f"<div class='inline-legend-item'>{item}</div>" for item in items)
    return f"<div class='inline-legend'><span class='inline-legend-title'>{html_escape(title)}</span>{rows}</div>"


def render_legend_block(sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    blocks: list[str] = []
    for title, items in sections:
        rows = "".join(f"<div class='legend-item'>{item}</div>" for item in items)
        blocks.append(
            f"<div class='legend-section'><h3 class='legend-title'>{html_escape(title)}</h3>{rows}</div>"
        )
    return "<div class='legend'><div class='legend-grid'>{}</div></div>".format(
        "".join(blocks)
    )


def make_scope_explainer_box(
    label: str,
    *,
    color: str,
    extra_class: str = "",
    hidden: bool = False,
) -> ScopeExplainerEntry:
    return ScopeExplainerEntry(
        kind="box",
        label=label,
        color=color,
        extra_class=extra_class,
        hidden=hidden,
    )


def make_scope_explainer_line(
    label: str,
    *,
    color: str,
    dash_style: str = "solid",
    extra_class: str = "",
    hidden: bool = False,
) -> ScopeExplainerEntry:
    return ScopeExplainerEntry(
        kind="line",
        label=label,
        color=color,
        dash_style=dash_style,
        extra_class=extra_class,
        hidden=hidden,
    )


def make_scope_explainer_text(
    label: str,
    *,
    extra_class: str = "",
    hidden: bool = False,
) -> ScopeExplainerEntry:
    return ScopeExplainerEntry(
        kind="text",
        label=label,
        extra_class=extra_class,
        hidden=hidden,
    )


def _render_scope_explainer_item(item: ScopeExplainerEntry | str) -> str:
    if isinstance(item, str):
        return f"<div class='scope-explainer-item'>{item}</div>"
    class_attr = f" class='{html_escape(item.extra_class)}'" if item.extra_class else ""
    visibility_style = ";display:none" if item.hidden else ""
    hidden_attr = ' style="display:none"' if item.hidden else ""
    item_class_suffix = f" {item.extra_class}" if item.extra_class else ""
    if item.kind == "box":
        style = f"background:{html_escape(item.color)}{visibility_style}"
        return (
            "<div class='scope-explainer-item'>"
            f"<div class='scope-explainer-box{item_class_suffix}' "
            f"style='{style}'></div>"
            f"<span{class_attr}{hidden_attr}>{html_escape(item.label)}</span>"
            "</div>"
        )
    if item.kind == "line":
        dash_style = html_escape(item.dash_style or "solid")
        style = f"border-top-style:{dash_style};border-top-color:{html_escape(item.color)}{visibility_style}"
        return (
            "<div class='scope-explainer-item'>"
            f"<div class='scope-explainer-line{item_class_suffix}' "
            f"style='{style}'></div>"
            f"<span{class_attr}{hidden_attr}>{html_escape(item.label)}</span>"
            "</div>"
        )
    return f"<div class='scope-explainer-item'><span{class_attr}{hidden_attr}>{html_escape(item.label)}</span></div>"


def render_scope_section_block(
    sections: Sequence[tuple[str, Sequence[ScopeExplainerEntry | str]]],
) -> str:
    groups: list[str] = []
    for title, items in sections:
        rows = "".join(_render_scope_explainer_item(item) for item in items)
        groups.append(
            "<div class='scope-explainer-group'>"
            f"<span class='scope-explainer-title'>{html_escape(title)}</span>"
            f"{rows}"
            "</div>"
        )
    return "<div class='scope-explainer'>{}</div>".format("".join(groups))


def render_graph_nav(
    prev_id: str, next_id: str, prev_title: str, next_title: str
) -> str:
    prev_button = (
        f"<button type='button' class='graph-nav-btn' id='{html_escape(prev_id)}' "
        f"title='{html_escape(prev_title)}'>‹</button>"
    )
    next_button = (
        f"<button type='button' class='graph-nav-btn' id='{html_escape(next_id)}' "
        f"title='{html_escape(next_title)}'>›</button>"
    )
    return f"<div class='graph-nav'>{prev_button}{next_button}</div>"


def render_graph_index_chip(chip_id: str, text: str, extra_class: str = "") -> str:
    extra = f" {extra_class}" if extra_class else ""
    return f"<span class='toolbar-chip graph-index-chip{extra}' id='{html_escape(chip_id)}'>{html_escape(text)}</span>"


def render_graph_toolbar(
    *,
    label: str,
    nav_html: str,
    search_html: str = "",
    select_html: str = "",
    index_chip_html: str = "",
    trailing_html: str = "",
    controls_html: str = "",
) -> str:
    left_parts = [
        f"<span class='toolbar-label'>{html_escape(label)}</span>" if label else "",
        nav_html,
        search_html,
        select_html,
    ]
    left_html_parts = []
    for part in left_parts:
        if part:
            left_html_parts.append(part)
    left_html = "".join(left_html_parts)
    right_html_parts = []
    for part in (index_chip_html, trailing_html):
        if part:
            right_html_parts.append(part)
    right_html = "".join(right_html_parts)
    controls_row = (
        f"<div class='toolbar-row controls'>{controls_html}</div>"
        if controls_html
        else ""
    )
    return (
        "<div class='toolbar'>"
        "<div class='toolbar-row graph-toolbar-primary'>"
        f"<div class='graph-toolbar-left'>{left_html}</div>"
        f"<div class='graph-toolbar-right'>{right_html}</div>"
        "</div>"
        f"{controls_row}"
        "</div>"
    )


def render_detail_table_panel(
    *,
    title: str,
    table_html: str,
    pager_html: str,
    tools_html: str = "",
    subtitle: str = "",
    subtitle_html: str = "",
    title_id: str = "",
    subtitle_id: str = "",
    toggle_id: str = "",
    content_id: str = "",
    toggle_label: str = "展开",
    initially_collapsed: bool = True,
    empty_html: str = "",
) -> str:
    title_markup = ""
    if title:
        title_markup = (
            f"<div class='detail-title' id='{html_escape(title_id)}'>{html_escape(title)}</div>"
            if title_id
            else f"<div class='detail-title'>{html_escape(title)}</div>"
        )
    subtitle_markup = ""
    if subtitle_html:
        subtitle_markup = (
            f"<div class='detail-subtitle' id='{html_escape(subtitle_id)}'>{subtitle_html}</div>"
            if subtitle_id
            else f"<div class='detail-subtitle'>{subtitle_html}</div>"
        )
    elif subtitle:
        subtitle_markup = (
            f"<div class='detail-subtitle' id='{html_escape(subtitle_id)}'>{html_escape(subtitle)}</div>"
            if subtitle_id
            else f"<div class='detail-subtitle'>{html_escape(subtitle)}</div>"
        )
    expanded = not initially_collapsed
    head_main_markup = f"{title_markup}{subtitle_markup}"
    if not head_main_markup:
        head_main_markup = "<div class='detail-title'>详情</div>"
    state_markup = (
        f"<span class='detail-head-state' id='{html_escape(toggle_id)}'>"
        f"{html_escape('收起' if expanded else toggle_label)}</span>"
        if toggle_id
        else (
            f"<span class='detail-head-state'>{html_escape('收起' if expanded else toggle_label)}</span>"
        )
    )
    panel_class = "detail-panel is-open" if expanded else "detail-panel"
    content_class = (
        "detail-content is-collapsed" if initially_collapsed else "detail-content"
    )
    content_attr = f" id='{html_escape(content_id)}'" if content_id else ""
    tools_markup = f"<div class='detail-tools'>{tools_html}</div>" if tools_html else ""
    head_markup = (
        "<div class='detail-head' data-detail-toggle role='button' tabindex='0'"
        f" aria-expanded='{'true' if expanded else 'false'}'"
        + (f" aria-controls='{html_escape(content_id)}'" if content_id else "")
        + ">"
        + f"<div class='detail-head-main'>{head_main_markup}</div>"
        + "<div class='detail-head-side'>"
        + state_markup
        + "<span class='detail-head-arrow' aria-hidden='true'>⌄</span>"
        + "</div>"
        + "</div>"
    )
    return (
        f"<div class='{panel_class}'>"
        f"{head_markup}"
        f"<div class='{content_class}'{content_attr}>"
        f"{tools_markup}"
        f"{table_html}"
        f"{pager_html}"
        f"{empty_html}"
        "</div>"
        "</div>"
    )


def _normalize_standard_page_size(page_size: int) -> int:
    normalized = int(page_size)
    if normalized in (10, 20, 50, 100):
        return normalized
    return 20


def _render_standard_page_size_options(selected: int) -> str:
    normalized = _normalize_standard_page_size(selected)
    return "".join(
        "<option value='{value}'{selected}>每页 {value} 条</option>".format(
            value=value,
            selected=" selected" if value == normalized else "",
        )
        for value in (10, 20, 50, 100)
    )


def render_standard_table_tools_html(
    *,
    main_html: str,
    jump_input_id: str = "",
    jump_btn_id: str = "",
    row_count_id: str = "",
    page_size_id: str = "",
    page_size: int = 20,
) -> str:
    jump_input_id_attr = f" id='{html_escape(jump_input_id)}'" if jump_input_id else ""
    jump_btn_id_attr = f" id='{html_escape(jump_btn_id)}'" if jump_btn_id else ""
    row_count_id_attr = f" id='{html_escape(row_count_id)}'" if row_count_id else ""
    page_size_id_attr = f" id='{html_escape(page_size_id)}'" if page_size_id else ""
    return (
        "<div class='table-shell-tools'>"
        "<div class='table-shell-tools-main'>"
        f"{main_html}"
        "</div>"
        "<div class='table-shell-tools-side'>"
        "<span class='table-page-jump'>"
        "<span>跳转到</span>"
        f"<input type='number' min='1' step='1' class='table-jump-input'{jump_input_id_attr} placeholder='输入页码' />"
        f"<button type='button' class='table-jump-btn'{jump_btn_id_attr}>跳转</button>"
        "</span>"
        f"<span class='table-page-status'{row_count_id_attr}>共 0 条</span>"
        f"<select class='table-page-size'{page_size_id_attr}>"
        f"{_render_standard_page_size_options(page_size)}"
        "</select>"
        "</div>"
        "</div>"
    )


def render_standard_table_pager_html(
    *,
    prev_btn_id: str = "",
    next_btn_id: str = "",
    status_id: str = "",
) -> str:
    prev_id_attr = f" id='{html_escape(prev_btn_id)}'" if prev_btn_id else ""
    next_id_attr = f" id='{html_escape(next_btn_id)}'" if next_btn_id else ""
    status_id_attr = f" id='{html_escape(status_id)}'" if status_id else ""
    return (
        "<div class='table-pager'>"
        "<div class='table-pager-main'>"
        f"<button type='button' class='table-pager-btn'{prev_id_attr}>上一页</button>"
        f"<button type='button' class='table-pager-btn'{next_id_attr}>下一页</button>"
        f"<span class='table-page-status'{status_id_attr}>第 1 / 1 页</span>"
        "</div>"
        "</div>"
    )


def render_paginated_table_shell(
    *,
    table_id: str,
    headers: Sequence[str],
    rows_html: str,
    title: str,
    min_width: int = 760,
    page_size: int = 20,
    prominent: bool = False,
) -> str:
    table_class = "paged-table prominent" if prominent else "paged-table"
    header_html = "".join(f"<th>{html_escape(label)}</th>" for label in headers)
    return (
        "<table id='{table_id}' class='{table_class}'>"
        "<thead><tr>{header_html}</tr></thead>"
        "<tbody>{rows_html}</tbody>"
        "</table>"
    ).format(
        table_id=html_escape(table_id),
        table_class=table_class,
        header_html=header_html,
        rows_html=rows_html,
    )


def render_standard_table_panel(
    *,
    panel_id: str,
    title: str = "",
    table_html: str,
    subtitle: str = "",
    subtitle_html: str = "",
    search_placeholder: str = "搜索当前表格",
    min_width: int = 760,
    page_size: int = 20,
    prominent: bool = False,
    initially_collapsed: bool = True,
    extra_tools_html: str = "",
) -> str:
    normalized_page_size = _normalize_standard_page_size(page_size)
    wrapped_table_html = (
        f"<div class='table-wrap detail-table-wrap{' prominent' if prominent else ''}' "
        f"style='--table-min-width:{int(min_width)}px' data-standard-table data-page-size='{normalized_page_size}'>"
        f"{table_html}"
        "</div>"
    )
    tools_html = (
        "<div class='table-shell-tools'>"
        "<div class='table-shell-tools-main'>"
        f"{extra_tools_html}"
        f"<input type='search' class='table-search' data-table-search "
        f"placeholder='{html_escape(search_placeholder)}' />"
        "</div>"
        "<div class='table-shell-tools-side'>"
        "<span class='table-page-jump'>"
        "<span>跳转到</span>"
        "<input type='number' min='1' step='1' class='table-jump-input' data-table-jump-input placeholder='输入页码' />"
        "<button type='button' class='table-jump-btn' data-table-jump-btn>跳转</button>"
        "</span>"
        "<span class='table-page-status' data-table-row-count>共 0 条</span>"
        "<select class='table-page-size' data-table-page-size>"
        f"{_render_standard_page_size_options(normalized_page_size)}"
        "</select>"
        "</div>"
        "</div>"
    )
    pager_html = render_standard_table_pager_html()
    return render_detail_table_panel(
        title=title,
        subtitle=subtitle,
        subtitle_html=subtitle_html,
        toggle_id=f"{panel_id}ToggleBtn",
        content_id=f"{panel_id}Content",
        tools_html=tools_html,
        table_html=wrapped_table_html,
        pager_html=pager_html,
        initially_collapsed=initially_collapsed,
    )


COMMON_STANDARD_TABLE_JS = r"""
function initStandardTablePanel(root) {
    const wrap = root.matches('[data-standard-table]') ? root : root.querySelector('[data-standard-table]');
    if (!wrap) return;
    const panel = wrap.closest('.detail-panel') || wrap.parentElement || wrap;
    const detailHead = panel.querySelector('[data-detail-toggle]');
    const detailContent = panel.querySelector('.detail-content');
    const detailState = panel.querySelector('.detail-head-state');
    const table = wrap.querySelector('table');
    const tbody = table && table.querySelector('tbody');
    if (!table || !tbody) return;
    const allRows = Array.from(tbody.querySelectorAll('tr'));
    const emptyRows = allRows.filter((row) => {
        if (row.hasAttribute('data-table-empty-row')) return true;
        const cells = row.querySelectorAll('td');
        return cells.length === 1 && cells[0].hasAttribute('colspan') && !row.hasAttribute('data-search-text');
    });
    const rows = allRows.filter((row) => !emptyRows.includes(row));
    const searchInput = panel.querySelector('[data-table-search]');
    const status = panel.querySelector('[data-table-status]');
    const rowCount = panel.querySelector('[data-table-row-count]');
    const prevBtn = panel.querySelector('[data-table-prev]');
    const nextBtn = panel.querySelector('[data-table-next]');
    const pageSizeSelect = panel.querySelector('[data-table-page-size]');
    const jumpInput = panel.querySelector('[data-table-jump-input]');
    const jumpBtn = panel.querySelector('[data-table-jump-btn]');
    let page = 1;
    let pageSize = Math.max(parseInt(wrap.getAttribute('data-page-size') || '20', 10), 1);
    if (pageSizeSelect) pageSizeSelect.value = String(pageSize);

    function syncCollapsedState() {
        if (!detailHead || !detailContent) return;
        const expanded = !detailContent.classList.contains('is-collapsed');
        panel.classList.toggle('is-open', expanded);
        detailHead.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        if (detailState) detailState.textContent = expanded ? '收起' : '展开';
    }

    function toggleCollapsed() {
        if (!detailContent) return;
        detailContent.classList.toggle('is-collapsed');
        syncCollapsedState();
    }

    function getFilteredRows() {
        const q = ((searchInput && searchInput.value) || '').trim().toLowerCase();
        if (!q) return rows;
        return rows.filter((row) => {
            const text = (row.getAttribute('data-search-text') || row.textContent || '').toLowerCase();
            return text.includes(q);
        });
    }

    function render() {
        const filtered = getFilteredRows();
        const totalRows = filtered.length;
        const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));
        if (page > totalPages) page = totalPages;
        if (page < 1) page = 1;
        const start = (page - 1) * pageSize;
        const end = start + pageSize;
        allRows.forEach((row) => { row.hidden = true; });
        if (totalRows > 0) {
            filtered.slice(start, end).forEach((row) => { row.hidden = false; });
        } else {
            emptyRows.forEach((row) => { row.hidden = false; });
        }
        if (status) status.textContent = '第 ' + page + ' / ' + totalPages + ' 页';
        if (rowCount) rowCount.textContent = '共 ' + totalRows + ' 条';
        if (prevBtn) prevBtn.disabled = page <= 1;
        if (nextBtn) nextBtn.disabled = page >= totalPages;
    }

    const jump = () => {
        if (!jumpInput) return;
        const target = parseInt(jumpInput.value || '', 10);
        if (!Number.isFinite(target)) return;
        page = target;
        render();
    };
    if (searchInput) searchInput.addEventListener('input', () => { page = 1; render(); });
    if (pageSizeSelect) pageSizeSelect.addEventListener('change', (ev) => {
        pageSize = Math.max(parseInt(ev.target.value || '20', 10), 1);
        page = 1;
        render();
    });
    if (prevBtn) prevBtn.addEventListener('click', () => { if (page > 1) { page -= 1; render(); } });
    if (nextBtn) nextBtn.addEventListener('click', () => { page += 1; render(); });
    if (jumpBtn) jumpBtn.addEventListener('click', jump);
    if (jumpInput) jumpInput.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') jump(); });
    if (detailHead && detailContent) {
        detailHead.addEventListener('click', (ev) => {
            const target = ev.target;
            if (target && target.closest('input, select, button, a, textarea, summary')) return;
            toggleCollapsed();
        });
        detailHead.addEventListener('keydown', (ev) => {
            if (ev.key !== 'Enter' && ev.key !== ' ') return;
            ev.preventDefault();
            toggleCollapsed();
        });
        syncCollapsedState();
    }
    render();
}
"""


COMMON_SVG_VIEWPORT_JS = r"""
function getViewportConfig() {
    const cfg = window.__SK_VIEWPORT_CONFIG || {};
    return {
        containerId: cfg.containerId || 'svgContainer',
        svgId: cfg.svgId || 'mainSvg',
        minScale: cfg.minScale != null ? cfg.minScale : 0.34,
        maxScale: cfg.maxScale != null ? cfg.maxScale : 2.2,
        topPad: cfg.topPad != null ? cfg.topPad : 24,
        sidePad: cfg.sidePad != null ? cfg.sidePad : 28,
        safeX: cfg.safeX != null ? cfg.safeX : null,
        safeY: cfg.safeY != null ? cfg.safeY : null,
        fitMode: cfg.fitMode || 'max',
    };
}

function ensureViewportPrefs() {
    if (!window.__SK_VIEWPORT_PREFS) {
        window.__SK_VIEWPORT_PREFS = { smallX: 'left', smallY: 'center' };
    }
    return window.__SK_VIEWPORT_PREFS;
}

function getViewportMetrics() {
    const cfg = getViewportConfig();
    const container = document.getElementById(cfg.containerId);
    const svg = document.getElementById(cfg.svgId);
    return {
        cfg,
        container,
        svg,
        cw: container.clientWidth,
        ch: container.clientHeight,
        sw: parseFloat(svg.getAttribute('width')) || 0,
        sh: parseFloat(svg.getAttribute('height')) || 0,
    };
}

function clampScaleValue(value) {
    const cfg = getViewportConfig();
    return Math.max(Math.min(value, cfg.maxScale), cfg.minScale);
}

function clampViewportTransform() {
    const { cfg, cw, ch, sw, sh } = getViewportMetrics();
    const prefs = ensureViewportPrefs();
    scale = clampScaleValue(scale);
    const scaledW = sw * scale;
    const scaledH = sh * scale;
    const safeX = cfg.safeX != null ? cfg.safeX : Math.min(72, cw * 0.08);
    const safeY = cfg.safeY != null ? cfg.safeY : Math.min(56, ch * 0.08);

    let minX;
    let maxX;
    if (scaledW <= cw - safeX * 2) {
        if (prefs.smallX === 'center') {
            const centeredX = (cw - scaledW) / 2;
            minX = centeredX - 12;
            maxX = centeredX + 12;
        } else {
            minX = safeX - 12;
            maxX = safeX + 12;
        }
    } else {
        minX = cw - scaledW - safeX;
        maxX = safeX;
    }

    let minY;
    let maxY;
    if (scaledH <= ch - safeY * 2) {
        if (prefs.smallY === 'top') {
            minY = safeY - 12;
            maxY = safeY + 12;
        } else {
            const centeredY = (ch - scaledH) / 2;
            minY = centeredY - 12;
            maxY = centeredY + 12;
        }
    } else {
        minY = ch - scaledH - safeY;
        maxY = safeY;
    }

    translateX = Math.max(minX, Math.min(maxX, translateX));
    translateY = Math.max(minY, Math.min(maxY, translateY));
}

function updateTransform() {
    const { svg } = getViewportMetrics();
    clampViewportTransform();
    svg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    const zoomEl = document.getElementById('zoomLevel') || document.getElementById('zoomLvl');
    if (zoomEl) zoomEl.textContent = Math.round(scale * 100) + '%';
}

function zoomAroundPoint(nextScale, anchorX, anchorY) {
    const oldScale = scale;
    const clampedNext = clampScaleValue(nextScale);
    if (Math.abs(clampedNext - oldScale) < 1e-6) {
        updateTransform();
        return;
    }
    scale = clampedNext;
    translateX = anchorX - (anchorX - translateX) * (scale / oldScale);
    translateY = anchorY - (anchorY - translateY) * (scale / oldScale);
    updateTransform();
}

function zoomAroundViewportCenter(nextScale) {
    const { cw, ch } = getViewportMetrics();
    zoomAroundPoint(nextScale, cw / 2, ch / 2);
}

function zoomIn() {
    zoomAroundViewportCenter(scale * 1.2);
}

function zoomOut() {
    zoomAroundViewportCenter(scale / 1.2);
}

function fitView() {
    const { cfg, cw, ch, sw, sh } = getViewportMetrics();
    const prefs = ensureViewportPrefs();
    const fitHeightScale = Math.max((ch - cfg.topPad * 2) / sh, 0.1);
    const fitWidthScale = Math.max((cw - cfg.sidePad * 2) / sw, 0.1);
    scale = clampScaleValue(Math.min(fitHeightScale, fitWidthScale));
    prefs.smallX = 'center';
    prefs.smallY = 'center';
    translateX = (cw - sw * scale) / 2;
    translateY = (ch - sh * scale) / 2;
    updateTransform();
}

function resetView() {
    const { cfg, cw, ch, sw, sh } = getViewportMetrics();
    const prefs = ensureViewportPrefs();
    const fitHeightScale = Math.max((ch - cfg.topPad * 2) / sh, 0.1);
    const fitWidthScale = Math.max((cw - cfg.sidePad * 2) / sw, 0.1);
    scale = clampScaleValue(Math.max(fitHeightScale, fitWidthScale));
    prefs.smallX = 'left';
    prefs.smallY = 'center';
    translateX = cfg.sidePad;
    translateY = (ch - sh * scale) / 2;
    updateTransform();
}

function attachViewportInteractions() {
    const { container } = getViewportMetrics();
    if (!container || container.dataset.viewportBound === '1') return;
    container.dataset.viewportBound = '1';

    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const oldScale = scale;
        scale = e.deltaY < 0
            ? clampScaleValue(scale * 1.12)
            : clampScaleValue(scale / 1.12);
        translateX = mx - (mx - translateX) * (scale / oldScale);
        translateY = my - (my - translateY) * (scale / oldScale);
        updateTransform();
    }, { passive: false });

    container.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        isPanning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        panStartTransX = translateX;
        panStartTransY = translateY;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        translateX = panStartTransX + (e.clientX - panStartX);
        translateY = panStartTransY + (e.clientY - panStartY);
        updateTransform();
    });

    window.addEventListener('mouseup', () => { isPanning = false; });
}
"""

COMMON_KEYBOARD_NAV_JS = r"""
function ensureKeyboardNavDefaults() {
    if (typeof keyboardNavEnabled === 'undefined') window.keyboardNavEnabled = true;
    if (typeof keyboardNavMode === 'undefined') window.keyboardNavMode = 'wasd_zoom';
}

function toggleKeyboardNav() {
    ensureKeyboardNavDefaults();
    keyboardNavEnabled = !keyboardNavEnabled;
    const btn = document.getElementById('kbdToggleBtn');
    if (btn) btn.textContent = keyboardNavEnabled ? '键盘导航：开' : '键盘导航：关';
}

function syncKeyboardModeControl() {
    ensureKeyboardNavDefaults();
    const select = document.getElementById('kbdModeSelect');
    if (select) select.value = keyboardNavMode;
}

function applyKeyboardAction(action, panStep) {
    if (!action) return false;
    if (action === 'zoom_in') {
        zoomIn();
        return true;
    }
    if (action === 'zoom_out') {
        zoomOut();
        return true;
    }
    if (action === 'pan_left') {
        translateX += panStep;
        updateTransform();
        return true;
    }
    if (action === 'pan_right') {
        translateX -= panStep;
        updateTransform();
        return true;
    }
    if (action === 'pan_up') {
        translateY += panStep;
        updateTransform();
        return true;
    }
    if (action === 'pan_down') {
        translateY -= panStep;
        updateTransform();
        return true;
    }
    return false;
}

function getKeyboardAction(key) {
    ensureKeyboardNavDefaults();
    const normalized = String(key || '').toLowerCase();
    const modeMap = {
        wasd_zoom: {
            w: 'zoom_in',
            s: 'zoom_out',
            a: 'pan_left',
            d: 'pan_right',
        },
        wasd_pan: {
            w: 'pan_up',
            s: 'pan_down',
            a: 'pan_left',
            d: 'pan_right',
        },
        arrows_zoom: {
            arrowup: 'zoom_in',
            arrowdown: 'zoom_out',
            arrowleft: 'pan_left',
            arrowright: 'pan_right',
        },
        vim_zoom: {
            k: 'zoom_in',
            j: 'zoom_out',
            h: 'pan_left',
            l: 'pan_right',
        },
    };
    return (modeMap[keyboardNavMode] || modeMap.wasd_zoom)[normalized] || '';
}

function attachKeyboardNav(options = {}) {
    if (window.__SK_KEYBOARD_NAV_BOUND === true) return;
    window.__SK_KEYBOARD_NAV_BOUND = true;
    ensureKeyboardNavDefaults();
    const panStep = options.panStep || 80;
    const previous = typeof options.previous === 'function' ? options.previous : null;
    const next = typeof options.next === 'function' ? options.next : null;

    window.addEventListener('keydown', (e) => {
        if (typeof options.isActive === 'function' && !options.isActive()) return;
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
        if (e.altKey || e.ctrlKey || e.metaKey) return;
        if (typeof options.beforeAction === 'function') options.beforeAction();

        if (e.key === '[' && previous) {
            e.preventDefault();
            previous();
            return;
        }
        if (e.key === ']' && next) {
            e.preventDefault();
            next();
            return;
        }
        if (e.key === '0' && typeof resetView === 'function') {
            e.preventDefault();
            resetView();
            return;
        }
        if ((e.key === 'f' || e.key === 'F') && typeof fitView === 'function') {
            e.preventDefault();
            fitView();
            return;
        }
        if (!keyboardNavEnabled) return;

        const action = getKeyboardAction(e.key);
        if (action) {
            e.preventDefault();
            applyKeyboardAction(action, panStep);
        }
    });
}
"""


def render_stat_badges(badges: Iterable[VisualStatBadge]) -> str:
    parts: list[str] = []
    for badge in badges:
        dom_id = f' id="{html_escape(badge.dom_id)}"' if badge.dom_id else ""
        value_dom_id = (
            f' id="{html_escape(badge.value_dom_id)}"' if badge.value_dom_id else ""
        )
        parts.append(
            f'<div class="stat-badge stat-{html_escape(badge.tone)}"{dom_id}>'
            f'<span class="stat-caption">{html_escape(badge.caption)}</span>'
            f'<span class="stat-value"{value_dom_id}>{html_escape(str(badge.value))}</span>'
            f'<span class="stat-foot">{html_escape(badge.foot)}</span>'
            f"</div>"
        )
    return "".join(parts)


def render_page_header(
    *,
    icon: str,
    kicker: str,
    title: str,
    note_html: str,
    stat_badges_html: str,
) -> str:
    kicker_block = (
        f'<div class="header-kicker">{html_escape(kicker)}</div>' if kicker else ""
    )
    note_block = f'<div class="header-note">{note_html}</div>' if note_html else ""
    stats_block = (
        f'<div class="hero-stats">{stat_badges_html}</div>' if stat_badges_html else ""
    )
    return (
        '<div class="header">'
        '  <div class="header-left">'
        f'    <div class="header-icon">{html_escape(icon)}</div>'
        '    <div class="header-copy">'
        f"      {kicker_block}"
        f"      <h1>{html_escape(title)}</h1>"
        f"      {note_block}"
        "    </div>"
        "  </div>"
        f"  {stats_block}"
        "</div>"
    )


def render_report_header(
    *,
    icon: str,
    kicker: str,
    title: str,
    note_html: str,
    nav_html: str = "",
) -> str:
    nav_block = nav_html if nav_html else ""
    page_header = render_page_header(
        icon=icon,
        kicker=kicker,
        title=title,
        note_html=note_html,
        stat_badges_html="",
    )
    return f"{page_header}{nav_block}"


def render_report_summary(
    items: Sequence[tuple[str, str]], *, compact: bool = False
) -> str:
    if not items:
        return ""
    blocks = []
    for label, value in items:
        blocks.append(
            "<div class='report-summary-item'>"
            f"<span class='report-summary-label'>{html_escape(label)}</span>"
            f"<div class='report-summary-value'>{html_escape(value)}</div>"
            "</div>"
        )
    css_class = "report-summary is-compact" if compact else "report-summary"
    return f"<div class='{css_class}'>{''.join(blocks)}</div>"


def render_report_intro_section(
    *,
    title: str,
    note_html: str,
    summary_html: str,
    section_id: str = "",
) -> str:
    id_attr = f" id='{html_escape(section_id)}'" if section_id else ""
    note_block = f"<p class='info-panel-note'>{note_html}</p>" if note_html else ""
    return (
        f"<section class='info-panel'{id_attr}>"
        f"<div class='info-panel-head'><h2 class='info-panel-title'>{html_escape(title)}</h2></div>"
        f"<div class='info-panel-body'><div class='report-intro'>{note_block}{summary_html}</div></div>"
        "</section>"
    )


def render_report_top_strip(
    *,
    items: Sequence[tuple[str, str]],
    note_html: str = "",
) -> str:
    chips_html = render_meta_strip(items)
    note_block = f"<p class='info-panel-note'>{note_html}</p>" if note_html else ""
    if not chips_html and not note_block:
        return ""
    return (
        "<section class='info-panel is-compact'>"
        f"<div class='info-panel-body'>{chips_html}{note_block}</div></section>"
    )


def render_report_section(
    *,
    title: str,
    note_html: str,
    body_html: str,
    tone: str = "",
    section_id: str = "",
    extra_class: str = "",
) -> str:
    if not title and not note_html:
        return body_html
    classes = ["card", "report-section"]
    if tone:
        classes.append(tone)
    if extra_class:
        classes.append(extra_class)
    id_attr = f" id='{html_escape(section_id)}'" if section_id else ""
    note_block = f"<p class='report-section-note'>{note_html}</p>" if note_html else ""
    head_markup = ""
    if title or note_block:
        title_block = (
            f"<h2 class='report-section-title'>{html_escape(title)}</h2>"
            if title
            else ""
        )
        head_markup = (
            f"<div class='report-section-head'>{title_block}{note_block}</div>"
        )
    return (
        f"<section class='{' '.join(classes)}'{id_attr}>"
        f"{head_markup}"
        f"<div class='report-section-body'>{body_html}</div>"
        "</section>"
    )


def render_view_nav(
    items: Iterable[tuple[str, str, bool]], kicker: str = "Views"
) -> str:
    links: list[str] = (
        [f'<div class="view-nav-kicker">{html_escape(kicker)}</div>'] if kicker else []
    )
    for label, href, active in items:
        cls = "view-nav-link is-active" if active else "view-nav-link"
        links.append(
            f'<a class="{cls}" href="{html_escape(href)}">{html_escape(label)}</a>'
        )
    return f'<nav class="view-nav">{"".join(links)}</nav>'
