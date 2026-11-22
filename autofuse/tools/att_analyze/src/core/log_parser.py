#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .evidence_schema import FinalTilingRecord, FinalTilingSummaryRecord


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class OperatorSummary:
    operator_name: str
    graph: int = 0
    result: int = 0
    group: int = -1
    case: int = 0
    aiv_mte2: Optional[float] = None
    aiv_mte3: Optional[float] = None
    objective_value: Optional[float] = None
    result_performance: Optional[float] = None
    tiling_values: Dict[str, float] = field(default_factory=dict)
    # 状态用于区分缺失证据与真实的 0 值；不改变既有 CSV 字段。
    parse_status: str = "ok"


class LogParser:
    def __init__(self):
        self.patterns = {
            "operator_name": re.compile(r"\[([^\]]+)\]\s*\[PROF\]"),
            "graph_result": re.compile(
                r"\[PROF\]Among all schedule results,\s*graph(\d+)_result(\d+)\s+is the best choice"
            ),
            "group_case": re.compile(
                r"\[PROF\]Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph(\d+)_result(\d+)_g(\d+)\s+is the best choice"
            ),
            "tiling_value": re.compile(
                rf"\[PROF\]The value of\s+(\w+)\s+is\s+({_NUMBER})\s+in\s+graph(\d+)_result(\d+)_g(\d+)_(\d+)"
            ),
            "objective_value": re.compile(
                rf"\[PROF\]The objective value of the tiling data is\s+({_NUMBER})\s+in\s+graph(\d+)_result(\d+)_g(\d+)_(\d+)"
            ),
            "result_performance": re.compile(
                rf"\[([^\]]+)\]\s+The value of\s+graph(\d+)_result(\d+)\s+is\s+({_NUMBER})"
            ),
        }

    @staticmethod
    def _envelope(line: str) -> Dict[str, str]:
        """Decode the space separated, shell-escaped ATT envelope."""
        values: Dict[str, str] = {}
        index = 0
        length = len(line)
        while index < length:
            while index < length and line[index].isspace():
                index += 1
            key_start = index
            while index < length and line[index] not in "= \t\r\n":
                index += 1
            if index >= length or line[index] != "=":
                while index < length and not line[index].isspace():
                    index += 1
                continue
            key = line[key_start:index]
            index += 1
            if index < length and line[index] == '"':
                index += 1
                chars = []
                closed = False
                while index < length:
                    char = line[index]
                    index += 1
                    if char == '"':
                        closed = True
                        break
                    if char == "\\" and index < length:
                        escaped = line[index]
                        index += 1
                        if escaped == "n":
                            chars.append("\n")
                        elif escaped == "r":
                            chars.append("\r")
                        elif escaped == "t":
                            chars.append("\t")
                        elif escaped in ("\\", '"'):
                            chars.append(escaped)
                        else:
                            chars.extend(("\\", escaped))
                    else:
                        chars.append(char)
                if not closed:
                    return {}
                values[key] = "".join(chars)
            else:
                value_start = index
                while index < length and not line[index].isspace():
                    index += 1
                values[key] = line[value_start:index]
        return values

    @staticmethod
    def _json_field(value: Optional[str]):
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _integer(value: Optional[str]) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _payload_hash(payload: str, algorithm: str) -> Optional[str]:
        if algorithm == "sha256":
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if algorithm != "att_mix64_v1":
            return None
        value = 0x6D2B79F5
        for byte in payload.encode("utf-8"):
            value ^= (
                byte + 0x9E3779B9 + ((value << 6) & 0xFFFFFFFFFFFFFFFF) + (value >> 2)
            )
            value &= 0xFFFFFFFFFFFFFFFF
            value = ((value << 13) | (value >> 51)) & 0xFFFFFFFFFFFFFFFF
        return f"{value:016x}"

    def _final_record(
        self,
        fields: Dict[str, str],
        source_path: str,
        source_line: int,
        tiling_repr=None,
        repr_hash: Optional[str] = None,
        status: str = "ok",
    ) -> FinalTilingRecord:
        pipe_value = fields.get("pipe_estimates")
        pipe_est = self._json_field(pipe_value)
        if pipe_est is None and pipe_value not in (None, "null"):
            status = "invalid_final_tiling"
        if isinstance(pipe_est, dict):
            for value in pipe_est.values():
                if value is not None and (
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    status = "invalid_final_tiling"
        repr_value = fields.get("tiling_repr")
        repr_kind = fields.get("repr_kind")
        if tiling_repr is None and repr_value is not None:
            tiling_repr = self._json_field(repr_value)
            if tiling_repr is None and repr_kind == "full_json" and repr_value:
                # Inductor's GetTilingDataRepr is a C++ designated-initializer
                # string rather than JSON. Preserve it verbatim for hashing
                # and downstream inspection.
                tiling_repr = repr_value
            elif tiling_repr is None and not (
                repr_kind == "unavailable" and repr_value == ""
            ):
                status = "invalid_final_tiling"
        graph = self._integer(fields.get("graph"))
        result = self._integer(fields.get("result"))
        group = self._integer(fields.get("group"))
        key = self._integer(fields.get("tiling_key"))
        score = self._integer(fields.get("score"))
        if fields.get("score") is not None and score is None:
            status = "invalid_final_tiling"
        if (
            not fields.get("operator")
            or graph is None
            or result is None
            or group is None
            or self._integer(fields.get("case_id")) is None
            or key is None
            or key < 0
        ):
            status = "invalid_final_tiling"
        schema = fields.get("schema")
        if schema not in ("1", 1):
            status = "invalid_final_tiling"
        if tiling_repr is not None and repr_hash is None:
            encoded = (
                json.dumps(tiling_repr, ensure_ascii=False, separators=(",", ":"))
                if not isinstance(tiling_repr, str)
                else tiling_repr
            )
            repr_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        output_index = self._integer(fields.get("output_index"))
        if fields.get("source") == "topn" and output_index is None:
            status = "invalid_final_tiling"
        return FinalTilingRecord(
            output_index=output_index,
            schema=fields.get("schema"),
            source=fields.get("source"),
            selection_mode=fields.get("selection_mode"),
            op=fields.get("operator"),
            graph=graph,
            result=result,
            group=group,
            case=self._integer(fields.get("case_id")),
            tiling_key=key,
            score=score,
            sub_case_tag=fields.get("sub_case_tag"),
            template_name=fields.get("template"),
            pipe_est=pipe_est,
            tiling_repr=tiling_repr,
            repr_kind=repr_kind,
            repr_hash=repr_hash,
            parse_status=status,
            source_path=source_path,
            source_line=source_line,
        )

    def _parse_final_tiling_end(
        self, fields: Dict[str, str], pending: Dict[str, Dict], source_path: str
    ):
        item = pending.pop(fields.get("id"), None)
        if item is None:
            return None
        begin = item["fields"]
        begin_chunks = self._integer(begin.get("chunks"))
        end_chunks = self._integer(fields.get("chunks"))
        chunks = item["chunks"]
        ordered = end_chunks is not None and list(sorted(chunks)) == list(
            range(end_chunks)
        )
        payload = "".join(chunks[i] for i in range(end_chunks)) if ordered else None
        begin_len = self._integer(begin.get("len"))
        end_len = self._integer(fields.get("len"))
        algorithm = begin.get("hash_alg", "sha256")
        valid = (
            ordered
            and payload is not None
            and not item["bad"]
            and begin_chunks is not None
            and end_chunks == begin_chunks
            and begin_len is not None
            and end_len == begin_len
            and bool(begin.get("hash"))
            and fields.get("hash") == begin.get("hash")
            and fields.get("hash_alg", algorithm) == algorithm
            and self._payload_hash(payload, algorithm) is not None
        )
        if valid and end_len is not None and len(payload.encode("utf-8")) != end_len:
            valid = False
        actual_hash = (
            self._payload_hash(payload, algorithm) if payload is not None else None
        )
        if valid and actual_hash != fields.get("hash"):
            valid = False
        framed_repr = self._json_field(payload) if valid else None
        if (
            valid
            and framed_repr is None
            and begin.get("repr_kind") == "full_json"
            and payload
        ):
            framed_repr = payload
        if valid and framed_repr is None:
            valid = False
        return self._final_record(
            begin,
            source_path,
            item["line"],
            tiling_repr=framed_repr,
            repr_hash=actual_hash if valid else fields.get("hash"),
            status="ok" if valid else "incomplete_final_tiling",
        )

    def _append_legacy_final_records(
        self, records: List[FinalTilingRecord], log_content: str, source_path: str
    ):
        if records:
            return
        for summary in self._parse_log_content(log_content):
            records.append(
                FinalTilingRecord(
                    schema=None,
                    source="legacy",
                    selection_mode=None,
                    op=summary.operator_name,
                    graph=summary.graph,
                    result=summary.result,
                    group=summary.group,
                    case=summary.case,
                    tiling_key=None,
                    score=None,
                    sub_case_tag=None,
                    template_name=None,
                    pipe_est=None,
                    tiling_repr=None,
                    repr_kind=None,
                    repr_hash=None,
                    parse_status="inferred_legacy",
                    source_path=source_path,
                    source_line=None,
                )
            )

    @staticmethod
    def _mark_duplicate_final_records(records: List[FinalTilingRecord]):
        identities = {}
        for record in records:
            identity = (
                record.source,
                record.op,
                record.graph,
                record.result,
                record.group,
                record.case,
                record.tiling_key,
            )
            identities.setdefault(identity, []).append(record)
        for group in identities.values():
            for record in group[1:]:
                if record.parse_status == "ok":
                    record.parse_status = "duplicate_final_tiling"

    def _tiling_envelope(self, payload: str, kind: str) -> Dict[str, str]:
        fields = self._envelope(payload)
        if kind == "TOPN_TILING":
            match = re.match(r"\s*stage=topn_output output\[(\d+)\]\s", payload)
            if match is not None:
                fields["output_index"] = match.group(1)
        return fields

    def _extract_tiling_records(
        self, log_content: str, source_path: str, kind: str
    ) -> List[FinalTilingRecord]:
        records: List[FinalTilingRecord] = []
        pending: Dict[str, Dict] = {}
        begin, chunk, end, single = (
            f"[ATT][{kind}{suffix}]" for suffix in ("_BEGIN", "_CHUNK", "_END", "")
        )
        for line_no, line in enumerate(log_content.splitlines(), 1):
            if begin in line:
                fields = self._tiling_envelope(line.split(begin, 1)[1], kind)
                if fields.get("id"):
                    pending[fields["id"]] = {
                        "fields": fields,
                        "line": line_no,
                        "chunks": {},
                        "bad": False,
                    }
            elif chunk in line:
                fields = self._envelope(line.split(chunk, 1)[1])
                item = pending.get(fields.get("id"))
                seq = self._integer(fields.get("seq"))
                if (
                    item is None
                    or seq is None
                    or seq in item["chunks"]
                    or seq != len(item["chunks"])
                ):
                    if item is not None:
                        item["bad"] = True
                else:
                    item["chunks"][seq] = fields.get("data", "")
            elif end in line:
                record = self._parse_final_tiling_end(
                    self._envelope(line.split(end, 1)[1]), pending, source_path
                )
                if record is not None:
                    records.append(record)
            elif single in line:
                fields = self._tiling_envelope(line.split(single, 1)[1], kind)
                records.append(self._final_record(fields, source_path, line_no))
        for item in pending.values():
            records.append(
                self._final_record(
                    item["fields"],
                    source_path,
                    item["line"],
                    status="incomplete_final_tiling",
                )
            )
        return records

    def extract_topn_tiling_records(
        self, log_content: str, source_path: str = ""
    ) -> List[FinalTilingRecord]:
        """Parse exported candidates separately from runtime final tiling."""
        records = self._extract_tiling_records(log_content, source_path, "TOPN_TILING")
        return sorted(records, key=lambda record: record.source_line or 0)

    def extract_final_tiling_records(
        self, log_content: str, source_path: str = ""
    ) -> List[FinalTilingRecord]:
        """Parse FINAL_TILING records, including BEGIN/CHUNK/END framing."""
        records = self._extract_tiling_records(log_content, source_path, "FINAL_TILING")
        self._append_legacy_final_records(records, log_content, source_path)
        self._mark_duplicate_final_records(records)
        return sorted(records, key=lambda record: record.source_line or 0)

    # Explicit alias retained for callers that use the schema terminology.
    parse_final_tiling_records = extract_final_tiling_records

    def extract_final_tiling_summaries(
        self, log_content: str, source_path: str = ""
    ) -> List[FinalTilingSummaryRecord]:
        """Parse result-level group mappings without changing group records."""
        summaries: List[FinalTilingSummaryRecord] = []
        pending: Dict[str, Dict] = {}
        for line_no, line in enumerate(log_content.splitlines(), 1):
            if "[ATT][FINAL_TILING_SUMMARY_BEGIN]" in line:
                fields = self._envelope(
                    line.split("[ATT][FINAL_TILING_SUMMARY_BEGIN]", 1)[1]
                )
                if fields.get("id"):
                    pending[fields["id"]] = {
                        "fields": fields,
                        "line": line_no,
                        "chunks": {},
                        "bad": False,
                    }
                continue
            if "[ATT][FINAL_TILING_SUMMARY_CHUNK]" in line:
                fields = self._envelope(
                    line.split("[ATT][FINAL_TILING_SUMMARY_CHUNK]", 1)[1]
                )
                item = pending.get(fields.get("id"))
                seq = self._integer(fields.get("seq"))
                if (
                    item is None
                    or seq is None
                    or seq in item["chunks"]
                    or seq != len(item["chunks"])
                ):
                    if item is not None:
                        item["bad"] = True
                else:
                    item["chunks"][seq] = fields.get("data", "")
                continue
            if "[ATT][FINAL_TILING_SUMMARY_END]" in line:
                fields = self._envelope(
                    line.split("[ATT][FINAL_TILING_SUMMARY_END]", 1)[1]
                )
                item = pending.pop(fields.get("id"), None)
                if item is None:
                    continue
                begin = item["fields"]
                begin_count = self._integer(begin.get("chunks"))
                count = self._integer(fields.get("chunks"))
                payload = (
                    "".join(item["chunks"].get(index, "") for index in range(count))
                    if count is not None
                    and list(sorted(item["chunks"])) == list(range(count))
                    else None
                )
                expected_len = self._integer(fields.get("len"))
                begin_len = self._integer(begin.get("len"))
                algorithm = begin.get("hash_alg", "sha256")
                valid = (
                    payload is not None
                    and not item["bad"]
                    and begin_count is not None
                    and begin_count == count
                    and begin_len is not None
                    and begin_len == expected_len
                    and expected_len == len(payload.encode("utf-8"))
                    and fields.get("hash") == begin.get("hash")
                    and fields.get("hash") == self._payload_hash(payload, algorithm)
                    and fields.get("hash_alg", algorithm) == algorithm
                )
                merged = dict(begin)
                merged["groups"] = payload if valid else None
                summaries.append(
                    self._make_final_tiling_summary(
                        merged, source_path, item["line"], valid
                    )
                )
                continue
            marker = "[ATT][FINAL_TILING_SUMMARY]"
            if marker not in line:
                continue
            fields = self._envelope(line.split(marker, 1)[1])
            summaries.append(
                self._make_final_tiling_summary(fields, source_path, line_no)
            )
        for item in pending.values():
            summaries.append(
                self._make_final_tiling_summary(
                    item["fields"], source_path, item["line"], False
                )
            )
        return summaries

    def _make_final_tiling_summary(
        self,
        fields: Dict[str, str],
        source_path: str,
        source_line: int,
        framed_valid: Optional[bool] = None,
    ) -> FinalTilingSummaryRecord:
        groups = self._json_field(fields.get("groups"))
        status = "ok"
        graph = self._integer(fields.get("graph"))
        result = self._integer(fields.get("result"))
        if fields.get("schema") not in ("1", 1):
            status = "invalid_final_tiling_summary"
        if framed_valid is False:
            status = "incomplete_final_tiling_summary"
        if framed_valid is not False and (
            not fields.get("operator")
            or graph is None
            or result is None
            or not isinstance(groups, dict)
        ):
            status = "invalid_final_tiling_summary"
        return FinalTilingSummaryRecord(
            schema=fields.get("schema"),
            source=fields.get("source"),
            selection_mode=fields.get("selection_mode"),
            op=fields.get("operator"),
            graph=graph,
            result=result,
            groups=groups if isinstance(groups, dict) else None,
            parse_status=status,
            source_path=source_path,
            source_line=source_line,
        )

    def extract_operator_names(self, log_content: str) -> List[str]:
        operators = []
        seen = set()
        for match in self.patterns["operator_name"].finditer(log_content):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                operators.append(name)
        return operators

    def _extract_graph_result_status(self, log_content: str, operator_name: str):
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among all schedule results,\s*graph(\d+)_result(\d+)\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(1)), int(match.group(2)), "ok"
        # 模板行也带有 graph/result，可用于兼容旧日志，但要明确标记推断来源。
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+\d+\s+of\s+graph(\d+)_result(\d+)_g\d+\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(1)), int(match.group(2)), "inferred_graph_result"
        return None, None, "missing_graph_result"

    def extract_graph_result(self, log_content: str, operator_name: str) -> tuple:
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among all schedule results,\s*graph(\d+)_result(\d+)\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(1)), int(match.group(2))
        graph, result, _ = self._extract_graph_result_status(log_content, operator_name)
        # 保留历史 API 的 0 默认值；parse_log_file 使用带状态接口，避免丢失缺失信息。
        return (graph if graph is not None else 0), (
            result if result is not None else 0
        )

    def extract_group_case(
        self, log_content: str, operator_name: str, graph_id: int, result_id: int
    ) -> tuple:
        """单 group 接口（兼容旧调用），返回第一个匹配的 (group_id, case_id)"""
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph{graph_id}_result{result_id}_g(\d+)\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(2)), int(match.group(1))
        return -1, 0

    def extract_all_group_cases(
        self, log_content: str, operator_name: str, graph_id: int, result_id: int
    ) -> Dict[int, int]:
        """多 group 接口，返回 {group_id: case_id}，同一 group 重复出现时取最后一次"""
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph{graph_id}_result{result_id}_g(\d+)\s+is the best choice"
        )
        result: Dict[int, int] = {}
        for match in pattern.finditer(log_content):
            case_id = int(match.group(1))
            group_id = int(match.group(2))
            result[group_id] = case_id  # 后出现的覆盖前面的
        return result

    def extract_all_graph_results(
        self, log_content: str, operator_name: str
    ) -> List[Tuple[int, int]]:
        results = set()

        result_perf_pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s+The value of\s+graph(\d+)_result(\d+)\s+is\b"
        )
        for match in result_perf_pattern.finditer(log_content):
            results.add((int(match.group(1)), int(match.group(2))))

        group_case_pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+\d+\s+of\s+graph(\d+)_result(\d+)_g\d+\s+is the best choice"
        )
        for match in group_case_pattern.finditer(log_content):
            results.add((int(match.group(1)), int(match.group(2))))

        if not results:
            results.add(self.extract_graph_result(log_content, operator_name))
        return sorted(results)

    def extract_tiling_values(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        group_id: int,
        case_id: int,
    ) -> Dict[str, float]:
        tiling_values = {}
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]The value of\s+(\w+)\s+is\s+({_NUMBER})\s+in\s+graph{graph_id}_result{result_id}_g{group_id}_{case_id}"
        )
        for match in pattern.finditer(log_content):
            tiling_values[match.group(1)] = float(match.group(2))
        return tiling_values

    def extract_performance_metrics(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        group_id: int,
        case_id: int,
    ) -> Dict[str, float]:
        metrics = {}
        objective_pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]The objective value of the tiling data is\s+({_NUMBER})\s+in\s+graph{graph_id}_result{result_id}_g{group_id}_{case_id}"
        )
        match = objective_pattern.search(log_content)
        if match:
            metrics["objective_value"] = float(match.group(1))
        tiling_values = self.extract_tiling_values(
            log_content, operator_name, graph_id, result_id, group_id, case_id
        )
        if "AIV_MTE2" in tiling_values:
            metrics["aiv_mte2"] = tiling_values["AIV_MTE2"]
        if "AIV_MTE3" in tiling_values:
            metrics["aiv_mte3"] = tiling_values["AIV_MTE3"]
        return metrics

    def extract_result_performance(
        self, log_content: str, operator_name: str, graph_id: int, result_id: int
    ) -> Optional[float]:
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s+The value of\s+graph{graph_id}_result{result_id}\s+is\s+({_NUMBER})"
        )
        match = pattern.search(log_content)
        if match:
            return float(match.group(1).rstrip("."))
        return None

    def _build_summary(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        group_id: int,
        case_id: int,
        parse_status: str = "ok",
    ) -> OperatorSummary:
        metrics = self.extract_performance_metrics(
            log_content, operator_name, graph_id, result_id, group_id, case_id
        )
        tiling_values = self.extract_tiling_values(
            log_content, operator_name, graph_id, result_id, group_id, case_id
        )
        result_performance = self.extract_result_performance(
            log_content, operator_name, graph_id, result_id
        )
        result_perf_missing = result_performance is None
        if result_perf_missing and parse_status == "ok":
            parse_status = "missing_result_performance"

        return OperatorSummary(
            operator_name=operator_name,
            graph=graph_id,
            result=result_id,
            group=group_id,
            case=case_id,
            aiv_mte2=metrics.get("aiv_mte2"),
            aiv_mte3=metrics.get("aiv_mte3"),
            objective_value=metrics.get("objective_value"),
            result_performance=result_performance,
            tiling_values=tiling_values,
            parse_status=parse_status,
        )

    def _collect_result_summaries(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        parse_status: str = "ok",
    ) -> List[OperatorSummary]:
        summaries: List[OperatorSummary] = []
        group_cases = self.extract_all_group_cases(
            log_content, operator_name, graph_id, result_id
        )
        if group_cases:
            for group_id, case_id in sorted(group_cases.items()):
                status = parse_status
                if status == "ok":
                    status = "ok"
                summaries.append(
                    self._build_summary(
                        log_content,
                        operator_name,
                        graph_id,
                        result_id,
                        group_id,
                        case_id,
                        status,
                    )
                )
            return summaries

        group_id, case_id = self.extract_group_case(
            log_content, operator_name, graph_id, result_id
        )
        status = parse_status if parse_status != "ok" else "missing_group_case"
        summaries.append(
            self._build_summary(
                log_content,
                operator_name,
                graph_id,
                result_id,
                group_id,
                case_id,
                status,
            )
        )
        return summaries

    def parse_log_file(
        self, file_path: str, summary_mode: str = "best_result_all_groups"
    ) -> List[OperatorSummary]:
        with open(file_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        return self._parse_log_content(log_content, summary_mode)

    def _parse_log_content(
        self, log_content: str, summary_mode: str = "best_result_all_groups"
    ) -> List[OperatorSummary]:
        """Parse summaries from content already loaded by a caller."""

        operator_names = self.extract_operator_names(log_content)
        summaries = []

        for operator_name in operator_names:
            graph_status = "ok"
            if summary_mode == "all_results_all_groups":
                graph_results = self.extract_all_graph_results(
                    log_content, operator_name
                )
            else:
                graph, result, graph_status = self._extract_graph_result_status(
                    log_content, operator_name
                )
                graph_results = [] if graph is None else [(graph, result)]

            if not graph_results:
                continue

            for graph_id, result_id in graph_results:
                summaries.extend(
                    self._collect_result_summaries(
                        log_content, operator_name, graph_id, result_id, graph_status
                    )
                )

        return summaries
