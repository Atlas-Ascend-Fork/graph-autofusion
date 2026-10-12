# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Logical-stream merger translated from ``dag_stream_merger.cc``.

The implementation intentionally keeps the same level/unit model and
tie-break order as GE.  It operates on node indices and never mutates the
input graph; the caller applies the returned logical-to-physical mapping.
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..common.log import logger


K_MERGE_IMPROVE_EPS = 1e-9


def _is_load_balance_strategy(strategy: object) -> bool:
    """Return whether *strategy* selects the load-balance solver.

    ``_stream_allocator`` passes the normalized ``"load_balance"`` spelling;
    a ``MergeStrategy`` member (whose value is ``"LoadBalance"``) is accepted
    as well so the two spellings cannot diverge into a silent MainStream
    fallback.  C++ compares an enum here.
    """

    if getattr(strategy, "name", "") == "LOAD_BALANCE":
        return True
    value = getattr(strategy, "value", strategy)
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_") in (
            "loadbalance",
            "load_balance",
        )
    return False


def _compare_double(lhs: float, rhs: float) -> int:
    diff = lhs - rhs
    if abs(diff) <= K_MERGE_IMPROVE_EPS:
        return 0
    return -1 if diff < 0.0 else 1


class StreamMergeOptions:
    __slots__ = (
        "strategy",
        "physical_stream_limit",
        "window_width",
        "candidate_limit",
        "low_conflict_limit",
        "light_stream_limit",
        "repair_moves",
        "event_local_weight",
        "window_balance_weight",
        "total_load_weight",
        "parallel_conflict_weight",
        "new_flow_penalty_weight",
        "main_stream_bonus_weight",
        "main_stream",
    )

    def __init__(
        self,
        strategy: object = "main_stream",
        physical_stream_limit: int = 8,
        window_width: int = 6,
        candidate_limit: int = 6,
        low_conflict_limit: int = 3,
        light_stream_limit: int = 3,
        repair_moves: int = 0,
        event_local_weight: float = 3.0,
        window_balance_weight: float = 4.0,
        total_load_weight: float = 1.0,
        parallel_conflict_weight: float = 4.0,
        new_flow_penalty_weight: float = 2.0,
        main_stream_bonus_weight: float = 4.0,
        main_stream: int = 0,
    ) -> None:
        self.strategy = strategy
        self.physical_stream_limit = physical_stream_limit
        self.window_width = window_width
        self.candidate_limit = candidate_limit
        self.low_conflict_limit = low_conflict_limit
        self.light_stream_limit = light_stream_limit
        self.repair_moves = repair_moves
        self.event_local_weight = event_local_weight
        self.window_balance_weight = window_balance_weight
        self.total_load_weight = total_load_weight
        self.parallel_conflict_weight = parallel_conflict_weight
        self.new_flow_penalty_weight = new_flow_penalty_weight
        self.main_stream_bonus_weight = main_stream_bonus_weight
        self.main_stream = main_stream


class _UnitProfile:
    __slots__ = (
        "unit_id",
        "node_indices",
        "size",
        "earliest_level",
        "latest_level",
        "peak_level_load",
        "level_hist",
        "interaction",
    )

    def __init__(
        self,
        unit_id: int,
        node_indices: List[int],
        size: int,
        earliest_level: int,
        latest_level: int,
        peak_level_load: int,
        level_hist: List[int],
        interaction: int = 0,
    ) -> None:
        self.unit_id = unit_id
        self.node_indices = node_indices
        self.size = size
        self.earliest_level = earliest_level
        self.latest_level = latest_level
        self.peak_level_load = peak_level_load
        self.level_hist = level_hist
        self.interaction = interaction


class _ConcurrentScore:
    __slots__ = (
        "total",
        "event_local",
        "parallel_conflict",
        "new_flow_penalty",
        "main_stream_bonus",
    )

    def __init__(
        self,
        total: float = 0.0,
        event_local: int = 0,
        parallel_conflict: int = 0,
        new_flow_penalty: int = 0,
        main_stream_bonus: int = 0,
    ) -> None:
        self.total = total
        self.event_local = event_local
        self.parallel_conflict = parallel_conflict
        self.new_flow_penalty = new_flow_penalty
        self.main_stream_bonus = main_stream_bonus


class _LiteScore:
    __slots__ = (
        "total",
        "event_local",
        "window_balance",
        "total_load",
        "candidate_stream",
    )

    def __init__(
        self,
        total: float = 0.0,
        event_local: int = 0,
        window_balance: float = 0.0,
        total_load: int = 0,
        candidate_stream: int = 0,
    ) -> None:
        self.total = total
        self.event_local = event_local
        self.window_balance = window_balance
        self.total_load = total_load
        self.candidate_stream = candidate_stream


def _concurrent_less(lhs: _ConcurrentScore, rhs: _ConcurrentScore) -> bool:
    cmp = _compare_double(lhs.total, rhs.total)
    if cmp:
        return cmp < 0
    if lhs.event_local != rhs.event_local:
        return lhs.event_local < rhs.event_local
    if lhs.parallel_conflict != rhs.parallel_conflict:
        return lhs.parallel_conflict < rhs.parallel_conflict
    if lhs.new_flow_penalty != rhs.new_flow_penalty:
        return lhs.new_flow_penalty < rhs.new_flow_penalty
    return lhs.main_stream_bonus < rhs.main_stream_bonus


def _lite_less(lhs: _LiteScore, rhs: _LiteScore) -> bool:
    cmp = _compare_double(lhs.total, rhs.total)
    if cmp:
        return cmp < 0
    if lhs.event_local != rhs.event_local:
        return lhs.event_local < rhs.event_local
    cmp = _compare_double(lhs.window_balance, rhs.window_balance)
    if cmp:
        return cmp < 0
    if lhs.total_load != rhs.total_load:
        return lhs.total_load < rhs.total_load
    return lhs.candidate_stream < rhs.candidate_stream


class _StreamMergeSolver:
    __slots__ = (
        "dag",
        "logical_stream_routes",
        "options",
        "topo_nodes",
        "node_name_to_index",
        "preds",
        "succs",
        "topo_order",
        "topo_position",
        "levels",
        "max_level",
        "edge_pairs",
        "unit_profiles",
        "node_to_unit",
        "unit_count",
        "global_level_loads",
        "unit_solo_mass",
        "unit_pred_edges",
        "unit_succ_edges",
        "units_by_earliest_level",
    )

    def __init__(
        self,
        dag,
        logical_stream_routes: Sequence[Sequence[int]],
        options: StreamMergeOptions,
    ):
        self.dag = dag
        self.logical_stream_routes = logical_stream_routes
        self.options = options
        self.topo_nodes: List[object] = []
        self.node_name_to_index: Dict[str, int] = {}
        self.preds: List[List[int]] = []
        self.succs: List[List[int]] = []
        self.topo_order: List[int] = []
        self.topo_position: List[int] = []
        self.levels: List[int] = []
        self.max_level = 0
        self.edge_pairs: List[Tuple[int, int]] = []
        self.unit_profiles: List[_UnitProfile] = []
        self.node_to_unit: List[int] = []
        self.unit_count = 0
        self.global_level_loads: List[int] = []
        self.unit_solo_mass: List[int] = []
        self.unit_pred_edges: List[Dict[int, int]] = []
        self.unit_succ_edges: List[Dict[int, int]] = []
        self.units_by_earliest_level: Dict[int, List[int]] = {}

    def solve(self) -> List[int]:
        self._build_node_order()
        self._build_edges()
        self._build_levels()
        self._build_units()
        if self.unit_count == 0:
            return []
        self._build_unit_dependency_graph()
        self._build_units_by_earliest_level()
        strategy = self.options.strategy
        if _is_load_balance_strategy(strategy):
            logger.info("Use kLoadBalance strategy.")
            return self._solve_load_balance()
        self._build_global_level_loads()
        self._build_unit_solo_mass()
        logger.info("Use kMainStream strategy.")
        return self._solve_main_stream()

    def _build_node_order(self) -> None:
        nodes = self.dag.get_nodes()
        for idx, node in enumerate(nodes):
            if node is None:
                logger.error("Node at index %d is nullptr.", idx)
                raise RuntimeError("merge graph contains a null node")
        self.topo_nodes = sorted(nodes, key=lambda node: node._index)
        self.node_name_to_index = {
            node._name: idx for idx, node in enumerate(self.topo_nodes)
        }

    def _build_edges(self) -> None:
        n = len(self.topo_nodes)
        self.preds = [[] for _ in range(n)]
        self.succs = [[] for _ in range(n)]
        self.edge_pairs = []
        for node_index, node in enumerate(self.topo_nodes):
            pred_indices = set()
            for pred in node._get_input_nodes():
                pred_index = self.node_name_to_index.get(pred._name)
                if pred_index is not None:
                    pred_indices.add(pred_index)
            self.preds[node_index] = sorted(pred_indices)
            for pred_index in self.preds[node_index]:
                self.succs[pred_index].append(node_index)
                self.edge_pairs.append((pred_index, node_index))

    def _build_levels(self) -> None:
        indegree = [len(preds) for preds in self.preds]
        ready = [idx for idx, degree in enumerate(indegree) if degree == 0]
        heapq.heapify(ready)
        self.topo_order = []
        while ready:
            node_index = heapq.heappop(ready)
            self.topo_order.append(node_index)
            for succ_index in self.succs[node_index]:
                indegree[succ_index] -= 1
                if indegree[succ_index] == 0:
                    # Keeping this sorted emulates priority_queue<greater<>>.
                    heapq.heappush(ready, succ_index)
        if len(self.topo_order) != len(self.topo_nodes):
            logger.error("Merge only supports DAG graphs.")
            raise RuntimeError("graph contains a cycle; merge only supports DAG graphs")
        self.topo_position = [-1] * len(self.topo_nodes)
        self.levels = [0] * len(self.topo_nodes)
        self.max_level = 0
        for position, node_index in enumerate(self.topo_order):
            self.topo_position[node_index] = position
            level = 0
            for pred_index in self.preds[node_index]:
                level = max(level, self.levels[pred_index] + 1)
            self.levels[node_index] = level
            self.max_level = max(self.max_level, level)

    def _build_units(self) -> None:
        self.unit_count = len(self.logical_stream_routes)
        self.unit_profiles = []
        self.node_to_unit = [-1] * len(self.topo_nodes)
        for route_id, route in enumerate(self.logical_stream_routes):
            indices: List[int] = []
            for node_index in route:
                if (
                    not isinstance(node_index, int)
                    or isinstance(node_index, bool)
                    or not (0 <= node_index < len(self.topo_nodes))
                ):
                    logger.error(
                        "Node index %s is out of range [0, %d).",
                        node_index,
                        len(self.topo_nodes),
                    )
                    raise ValueError(f"node index {node_index!r} is out of range")
                if self.node_to_unit[node_index] != -1:
                    logger.error(
                        "Node %s appears in more than one logical stream.",
                        self.topo_nodes[node_index]._name,
                    )
                    raise ValueError(
                        f"node {node_index} appears in more than one logical stream"
                    )
                self.node_to_unit[node_index] = route_id
                indices.append(node_index)
            if not indices:
                logger.error("Logical stream %d should not be empty.", route_id)
                raise ValueError(f"logical stream {route_id} should not be empty")
            indices.sort(key=lambda idx: self.topo_position[idx])
            level_hist = [0] * (self.max_level + 1)
            earliest = self.max_level
            latest = 0
            for idx in indices:
                level = self.levels[idx]
                level_hist[level] += 1
                earliest = min(earliest, level)
                latest = max(latest, level)
            peak = max(level_hist) if level_hist else 0
            self.unit_profiles.append(
                _UnitProfile(
                    route_id, indices, len(indices), earliest, latest, peak, level_hist
                )
            )
        for idx, unit in enumerate(self.node_to_unit):
            if unit < 0:
                logger.error(
                    "Node %s is not assigned to any logical stream.",
                    self.topo_nodes[idx]._name,
                )
                raise ValueError(f"node {idx} is not assigned to any logical stream")

    def _build_global_level_loads(self) -> None:
        self.global_level_loads = [0] * (self.max_level + 1)
        for level in self.levels:
            self.global_level_loads[level] += 1

    def _build_unit_solo_mass(self) -> None:
        self.unit_solo_mass = [0] * self.unit_count
        for unit in self.unit_profiles:
            self.unit_solo_mass[unit.unit_id] = sum(
                count
                for level, count in enumerate(unit.level_hist)
                if self.global_level_loads[level] == 1
            )

    def _build_unit_dependency_graph(self) -> None:
        self.unit_pred_edges = [dict() for _ in range(self.unit_count)]
        self.unit_succ_edges = [dict() for _ in range(self.unit_count)]
        interactions = [0] * self.unit_count
        for src, dst in self.edge_pairs:
            src_unit = self.node_to_unit[src]
            dst_unit = self.node_to_unit[dst]
            if src_unit == dst_unit:
                continue
            self.unit_succ_edges[src_unit][dst_unit] = (
                self.unit_succ_edges[src_unit].get(dst_unit, 0) + 1
            )
            self.unit_pred_edges[dst_unit][src_unit] = (
                self.unit_pred_edges[dst_unit].get(src_unit, 0) + 1
            )
            interactions[src_unit] += 1
            interactions[dst_unit] += 1
        for unit in self.unit_profiles:
            unit.interaction = interactions[unit.unit_id]

    def _build_units_by_earliest_level(self) -> None:
        self.units_by_earliest_level = {}
        for unit in self.unit_profiles:
            self.units_by_earliest_level.setdefault(unit.earliest_level, []).append(
                unit.unit_id
            )
        for ids in self.units_by_earliest_level.values():
            ids.sort(
                key=lambda uid: (
                    -self.unit_profiles[uid].size,
                    -self.unit_profiles[uid].interaction,
                    -self.unit_profiles[uid].peak_level_load,
                    self.unit_profiles[uid].earliest_level,
                    uid,
                )
            )

    def _window_end(self, start_level: int) -> int:
        return min(self.max_level, start_level + self.options.window_width - 1)

    def _window_mass(self, hist: Sequence[int], start_level: int) -> int:
        return sum(hist[start_level : self._window_end(start_level) + 1])

    def _stream_window_load(
        self, loads: Sequence[Sequence[int]], stream: int, level: int
    ) -> int:
        return self._window_mass(loads[stream], level)

    def _must_assign_main(self, unit_id: int, level: int) -> bool:
        unit = self.unit_profiles[unit_id]
        return (
            self.global_level_loads[level] == 1
            or self.unit_solo_mass[unit_id] == unit.size
        )

    def _adjacent_assigned_flows(
        self, unit_id: int, assignment: Sequence[int]
    ) -> Dict[int, int]:
        weights: Dict[int, int] = {}
        for neighbor, count in self.unit_pred_edges[unit_id].items():
            flow = assignment[neighbor]
            if flow >= 0:
                weights[flow] = weights.get(flow, 0) + count
        for neighbor, count in self.unit_succ_edges[unit_id].items():
            flow = assignment[neighbor]
            if flow >= 0:
                weights[flow] = weights.get(flow, 0) + count
        return weights

    def _parallel_conflict(
        self, unit_hist: Sequence[int], stream_hist: Sequence[int], level: int
    ) -> int:
        return sum(
            unit_hist[lvl] * stream_hist[lvl]
            for lvl in range(level, self._window_end(level) + 1)
        )

    def _dedup_candidates(
        self, candidates: Iterable[int], active_count: int, include_new: bool
    ) -> List[int]:
        values = list(candidates)
        if include_new and active_count < self.options.physical_stream_limit:
            values.append(active_count)
        unique: List[int] = []
        seen = set()
        for candidate in values:
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
            if len(unique) >= self.options.candidate_limit:
                break
        if unique:
            return unique
        if include_new and active_count < self.options.physical_stream_limit:
            return [active_count]
        return list(range(min(active_count, self.options.candidate_limit)))

    def _add_adjacent_with_size(
        self, out: List[int], uid: int, assignment: Sequence[int], sizes: Sequence[int]
    ) -> None:
        pairs = list(self._adjacent_assigned_flows(uid, assignment).items())
        pairs.sort(key=lambda item: (-item[1], sizes[item[0]], item[0]))
        out.extend(stream for stream, _ in pairs)

    def _add_light(
        self,
        out: List[int],
        level: int,
        sizes: Sequence[int],
        loads: Sequence[Sequence[int]],
        active: int,
    ) -> None:
        values = [
            (self._stream_window_load(loads, stream, level), sizes[stream], stream)
            for stream in range(active)
        ]
        values.sort()
        out.extend(stream for _, _, stream in values[: self.options.light_stream_limit])

    def _load_candidates(
        self,
        uid: int,
        level: int,
        assignment: Sequence[int],
        sizes: Sequence[int],
        loads: Sequence[Sequence[int]],
        active: int,
        include_new: bool,
    ) -> List[int]:
        candidates: List[int] = []
        self._add_adjacent_with_size(candidates, uid, assignment, sizes)
        self._add_light(candidates, level, sizes, loads, active)
        return self._dedup_candidates(candidates, active, include_new)

    def _calc_event_local(
        self, uid: int, candidate: int, assignment: Sequence[int]
    ) -> int:
        event = 0
        for neighbor, count in self.unit_pred_edges[uid].items():
            stream = assignment[neighbor]
            if stream >= 0 and stream != candidate:
                event += count
        for neighbor, count in self.unit_succ_edges[uid].items():
            stream = assignment[neighbor]
            if stream >= 0 and stream != candidate:
                event += count
        return event

    def _add_main_candidate(
        self, out: List[int], active: int, include_new: bool
    ) -> None:
        if active > self.options.main_stream:
            out.append(self.options.main_stream)
        elif include_new and active == 0:
            out.append(self.options.main_stream)

    def _add_adjacent(
        self, out: List[int], uid: int, assignment: Sequence[int]
    ) -> None:
        pairs = list(self._adjacent_assigned_flows(uid, assignment).items())
        pairs.sort(key=lambda item: (-item[1], item[0]))
        out.extend(stream for stream, _ in pairs)

    def _add_low_conflict(
        self,
        out: List[int],
        uid: int,
        level: int,
        loads: Sequence[Sequence[int]],
        active: int,
    ) -> None:
        unit_hist = self.unit_profiles[uid].level_hist
        values = [
            (self._parallel_conflict(unit_hist, loads[stream], level), stream)
            for stream in range(active)
        ]
        values.sort()
        out.extend(stream for _, stream in values[: self.options.low_conflict_limit])

    def _main_candidates(
        self,
        uid: int,
        level: int,
        assignment: Sequence[int],
        loads: Sequence[Sequence[int]],
        active: int,
        include_new: bool,
    ) -> List[int]:
        if self._must_assign_main(uid, level):
            return [self.options.main_stream]
        candidates: List[int] = []
        self._add_main_candidate(candidates, active, include_new)
        self._add_adjacent(candidates, uid, assignment)
        self._add_low_conflict(candidates, uid, level, loads, active)
        return self._dedup_candidates(candidates, active, include_new)

    def _evaluate_lite(
        self,
        uid: int,
        candidate: int,
        level: int,
        assignment: Sequence[int],
        sizes: Sequence[int],
        loads: Sequence[Sequence[int]],
        active: int,
    ) -> _LiteScore:
        unit = self.unit_profiles[uid]
        opens = candidate == active
        projected = active + (1 if opens else 0)
        score = _LiteScore()
        score.event_local = self._calc_event_local(uid, candidate, assignment)
        window_loads = [
            self._stream_window_load(loads, stream, level) for stream in range(active)
        ]
        if opens:
            window_loads.append(0)
        window_loads[candidate] += self._window_mass(unit.level_hist, level)
        mean = sum(window_loads) / float(projected)
        peak = max(window_loads) if window_loads else 0
        variance = sum((float(load) - mean) ** 2 for load in window_loads) / float(
            projected
        )
        score.window_balance = float(peak) + variance
        score.total_load = (0 if opens else sizes[candidate]) + unit.size
        score.candidate_stream = candidate
        score.total = (
            self.options.event_local_weight * score.event_local
            + self.options.window_balance_weight * score.window_balance
            + self.options.total_load_weight * score.total_load
            + self.options.new_flow_penalty_weight * (1 if opens else 0)
        )
        return score

    def _evaluate_concurrent(
        self,
        uid: int,
        candidate: int,
        level: int,
        assignment: Sequence[int],
        loads: Sequence[Sequence[int]],
        active: int,
    ) -> _ConcurrentScore:
        score = _ConcurrentScore()
        opens = candidate == active
        score.event_local = self._calc_event_local(uid, candidate, assignment)
        if not opens:
            score.parallel_conflict = self._parallel_conflict(
                self.unit_profiles[uid].level_hist, loads[candidate], level
            )
        score.new_flow_penalty = 1 if opens else 0
        score.main_stream_bonus = (
            self.unit_solo_mass[uid] if candidate == self.options.main_stream else 0
        )
        score.total = (
            self.options.event_local_weight * score.event_local
            + self.options.parallel_conflict_weight * score.parallel_conflict
            + self.options.new_flow_penalty_weight * score.new_flow_penalty
            - self.options.main_stream_bonus_weight * score.main_stream_bonus
        )
        return score

    def _apply_load(
        self,
        uid: int,
        candidate: int,
        assignment: List[int],
        sizes: List[int],
        loads: List[List[int]],
        active: int,
    ) -> int:
        if candidate == active:
            sizes.append(0)
            loads.append([0] * (self.max_level + 1))
            active += 1
        if candidate < 0 or candidate >= active:
            raise RuntimeError("candidate physical stream is out of range")
        assignment[uid] = candidate
        unit = self.unit_profiles[uid]
        sizes[candidate] += unit.size
        for level, count in enumerate(unit.level_hist):
            loads[candidate][level] += count
        return active

    def _apply_main(
        self,
        uid: int,
        candidate: int,
        assignment: List[int],
        loads: List[List[int]],
        active: int,
    ) -> int:
        if candidate == active:
            loads.append([0] * (self.max_level + 1))
            active += 1
        if candidate < 0 or candidate >= active:
            raise RuntimeError("candidate physical stream is out of range")
        assignment[uid] = candidate
        for level, count in enumerate(self.unit_profiles[uid].level_hist):
            loads[candidate][level] += count
        return active

    def _remove_load(
        self, uid: int, assignment: List[int], sizes: List[int], loads: List[List[int]]
    ) -> int:
        stream = assignment[uid]
        if stream < 0:
            logger.error("Try to remove unassigned unit %d.", uid)
            raise RuntimeError("try to remove an unassigned unit")
        assignment[uid] = -1
        unit = self.unit_profiles[uid]
        sizes[stream] -= unit.size
        for level, count in enumerate(unit.level_hist):
            loads[stream][level] -= count
        return stream

    def _remove_main(
        self, uid: int, assignment: List[int], loads: List[List[int]]
    ) -> int:
        stream = assignment[uid]
        if stream < 0:
            logger.error("Try to remove unassigned unit %d.", uid)
            raise RuntimeError("try to remove an unassigned unit")
        assignment[uid] = -1
        for level, count in enumerate(self.unit_profiles[uid].level_hist):
            loads[stream][level] -= count
        return stream

    def _solve_load_balance(self) -> List[int]:
        assignment = [-1] * self.unit_count
        sizes: List[int] = []
        loads: List[List[int]] = []
        active = 0
        for level in range(self.max_level + 1):
            recent_units: List[int] = []
            for uid in self.units_by_earliest_level.get(level, []):
                candidates = self._load_candidates(
                    uid, level, assignment, sizes, loads, active, True
                )
                if not candidates:
                    logger.error("No candidate physical stream for unit %d.", uid)
                    raise RuntimeError("no candidate physical stream")
                best = candidates[0]
                best_score = self._evaluate_lite(
                    uid, best, level, assignment, sizes, loads, active
                )
                for candidate in candidates[1:]:
                    score = self._evaluate_lite(
                        uid, candidate, level, assignment, sizes, loads, active
                    )
                    if _lite_less(score, best_score):
                        best, best_score = candidate, score
                active = self._apply_load(uid, best, assignment, sizes, loads, active)
                recent_units.append(uid)
            if self.options.repair_moves > 0 and recent_units:
                self._repair_load_balance_recent_units(
                    level, recent_units, assignment, sizes, loads, active
                )
        return self._compact(assignment)

    def _solve_main_stream(self) -> List[int]:
        assignment = [-1] * self.unit_count
        loads: List[List[int]] = []
        active = 0
        for level in range(self.max_level + 1):
            recent_units: List[int] = []
            for uid in self.units_by_earliest_level.get(level, []):
                candidates = self._main_candidates(
                    uid, level, assignment, loads, active, True
                )
                if not candidates:
                    logger.error("No candidate physical stream for unit %d.", uid)
                    raise RuntimeError("no candidate physical stream")
                best = candidates[0]
                best_score = self._evaluate_concurrent(
                    uid, best, level, assignment, loads, active
                )
                for candidate in candidates[1:]:
                    score = self._evaluate_concurrent(
                        uid, candidate, level, assignment, loads, active
                    )
                    if _concurrent_less(score, best_score):
                        best, best_score = candidate, score
                active = self._apply_main(uid, best, assignment, loads, active)
                recent_units.append(uid)
            if self.options.repair_moves > 0 and recent_units:
                self._repair_main_stream_recent_units(
                    level, recent_units, assignment, loads, active
                )
        return self._compact(assignment)

    def _repair_load_balance_recent_units(
        self,
        level: int,
        recent_units: Sequence[int],
        assignment: List[int],
        sizes: List[int],
        loads: List[List[int]],
        active: int,
    ) -> None:
        """Apply the bounded local repair pass from ``dag_stream_merger.cc``.

        ``repair_moves`` defaults to zero in GE, so this pass normally does
        nothing; it is translated because the option exists in the baseline.
        Every trial is rolled back before the next candidate is evaluated; the
        final move is committed only after the complete level has been scored.
        """

        move_count = 0
        while move_count < self.options.repair_moves:
            best_unit = -1
            best_candidate = -1
            best_improvement = 0.0
            for uid in recent_units:
                current_stream = self._remove_load(uid, assignment, sizes, loads)
                current_score = self._evaluate_lite(
                    uid, current_stream, level, assignment, sizes, loads, active
                )
                candidates = self._load_candidates(
                    uid, level, assignment, sizes, loads, active, False
                )
                if current_stream not in candidates:
                    candidates.append(current_stream)

                best_for_unit = current_stream
                best_score = self._evaluate_lite(
                    uid, current_stream, level, assignment, sizes, loads, active
                )
                for candidate in candidates:
                    score = self._evaluate_lite(
                        uid, candidate, level, assignment, sizes, loads, active
                    )
                    if _lite_less(score, best_score):
                        best_for_unit, best_score = candidate, score

                # Restore the state before considering another unit.  Passing
                # the original active count is intentional: removing a unit
                # never retires a physical stream in the C++ implementation.
                self._apply_load(uid, current_stream, assignment, sizes, loads, active)
                improvement = current_score.total - best_score.total
                if (
                    best_for_unit != current_stream
                    and improvement > K_MERGE_IMPROVE_EPS
                    and improvement > best_improvement
                ):
                    best_unit = uid
                    best_candidate = best_for_unit
                    best_improvement = improvement

            if best_unit < 0:
                return
            self._remove_load(best_unit, assignment, sizes, loads)
            self._apply_load(
                best_unit, best_candidate, assignment, sizes, loads, active
            )
            move_count += 1

    def _repair_main_stream_recent_units(
        self,
        level: int,
        recent_units: Sequence[int],
        assignment: List[int],
        loads: List[List[int]],
        active: int,
    ) -> None:
        """Apply the bounded local repair pass for MainStream strategy."""

        move_count = 0
        while move_count < self.options.repair_moves:
            best_unit = -1
            best_candidate = -1
            best_improvement = 0.0
            for uid in recent_units:
                if self._must_assign_main(uid, level):
                    continue
                current_stream = self._remove_main(uid, assignment, loads)
                current_score = self._evaluate_concurrent(
                    uid, current_stream, level, assignment, loads, active
                )
                candidates = self._main_candidates(
                    uid, level, assignment, loads, active, False
                )
                if current_stream not in candidates:
                    candidates.append(current_stream)

                best_for_unit = current_stream
                best_score = self._evaluate_concurrent(
                    uid, current_stream, level, assignment, loads, active
                )
                for candidate in candidates:
                    score = self._evaluate_concurrent(
                        uid, candidate, level, assignment, loads, active
                    )
                    if _concurrent_less(score, best_score):
                        best_for_unit, best_score = candidate, score

                self._apply_main(uid, current_stream, assignment, loads, active)
                improvement = current_score.total - best_score.total
                if (
                    best_for_unit != current_stream
                    and improvement > K_MERGE_IMPROVE_EPS
                    and improvement > best_improvement
                ):
                    best_unit = uid
                    best_candidate = best_for_unit
                    best_improvement = improvement

            if best_unit < 0:
                return
            self._remove_main(best_unit, assignment, loads)
            self._apply_main(best_unit, best_candidate, assignment, loads, active)
            move_count += 1

    @staticmethod
    def _compact(assignment: Sequence[int]) -> List[int]:
        # An empty route set is a valid no-op merge.  The C++ helper returns an
        # empty mapping for this case; the path-cover boundary is responsible
        # for rejecting an unexpectedly empty mapping when routes were given.
        if not assignment:
            return []
        if any(stream < 0 for stream in assignment):
            logger.error(
                "There is still an unassigned unit when compacting assignment."
            )
            raise RuntimeError(
                "multistream internal error: unassigned unit when compacting assignment"
            )
        used = sorted(set(assignment))
        if not used:
            logger.error("Merge should produce at least one physical stream.")
            raise RuntimeError(
                "multistream internal error: merge produced no physical stream"
            )
        remap = {stream: idx for idx, stream in enumerate(used)}
        return [remap[stream] for stream in assignment]


class _Merger:
    """Entry point used by ``_stream_allocator``; mirrors ``StreamMerger``."""

    __slots__ = ("options",)

    def __init__(self, options: Optional[StreamMergeOptions] = None):
        self.options = options if options is not None else StreamMergeOptions()

    def merge(self, graph, routes: Sequence[Sequence[int]]) -> List[int]:
        options = self.options
        # Check one option at a time, in StreamMerger::Merge order, so the log
        # names the invalid option.
        if options.physical_stream_limit <= 0:
            logger.error("Merge physical stream limit must be greater than 0.")
            raise ValueError("invalid stream merge options")
        if options.window_width <= 0:
            logger.error("Merge window width must be greater than 0.")
            raise ValueError("invalid stream merge options")
        if options.candidate_limit <= 0:
            logger.error("Merge candidate limit must be greater than 0.")
            raise ValueError("invalid stream merge options")
        if options.repair_moves < 0:
            logger.error("Merge repair moves must be greater than or equal to 0.")
            raise ValueError("invalid stream merge options")
        if _is_load_balance_strategy(options.strategy):
            if options.light_stream_limit <= 0:
                logger.error("LoadBalance light stream limit must be greater than 0.")
                raise ValueError("invalid light stream limit")
        else:
            if options.low_conflict_limit <= 0:
                logger.error("MainStream low conflict limit must be greater than 0.")
                raise ValueError("invalid main stream options")
            if options.main_stream < 0:
                logger.error(
                    "MainStream main stream must be greater than or equal to 0."
                )
                raise ValueError("invalid main stream options")
        if not routes:
            return []
        return _StreamMergeSolver(graph, routes, options).solve()


def merge(
    graph, routes: Sequence[Sequence[int]], options: Optional[StreamMergeOptions] = None
) -> List[int]:
    """Convenience entry point mirroring the C++ ``StreamMerger::Merge``.

    The implementation uses :class:`_Merger` internally. It is intentionally
    not part of ``__all__``; the package allocator uses
    the ``_Merger`` entry point directly, while this helper remains available
    for explicit callers.
    """

    return _Merger(options).merge(graph, routes)


__all__ = ["StreamMergeOptions", "_Merger"]
