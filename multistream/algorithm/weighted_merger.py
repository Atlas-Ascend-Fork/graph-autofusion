# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Duration/resource-aware logical stream merger.

This module is a direct, standard-library-only translation of
``dag_weighted_stream_merger.cc``.  The public package only uses the small
``_WeightedMerger`` entry point; the solver below deliberately keeps the same
staging (resource profiles, candidate scoring, schedule re-simulation and
bounded repair) as the GE implementation so that a profiling-enabled graph
does not silently take a different scheduling path.
"""

from __future__ import annotations

from collections import deque
from functools import cmp_to_key
import heapq
from typing import Dict, List, Optional, Sequence, Tuple

from ..common.log import logger
from ..minidag.types import NodeCost, _cost_of


K_IMPROVE_EPS = 1e-9


def _compare_double(lhs: float, rhs: float) -> int:
    diff = lhs - rhs
    if abs(diff) <= K_IMPROVE_EPS:
        return 0
    return -1 if diff < 0.0 else 1


class WeightedStreamMergeOptions:
    """Options corresponding to ``WeightedStreamMergeOptions`` in GE."""

    __slots__ = (
        "physical_stream_limit",
        "window_width",
        "candidate_limit",
        "light_stream_limit",
        "repair_moves",
        "resim_candidate_limit",
        "ressimspan_weight",
        "stage_var_weight",
        "event_weight",
        "time_imbalance_weight",
        "stream_count_weight",
        "event_local_weight",
        "time_conflict_weight",
        "time_load_weight",
        "new_flow_penalty_weight",
    )

    def __init__(
        self,
        physical_stream_limit: int = 8,
        window_width: int = 6,
        candidate_limit: int = 8,
        light_stream_limit: int = 3,
        repair_moves: int = 0,
        resim_candidate_limit: int = 3,
        ressimspan_weight: float = 8.0,
        stage_var_weight: float = 3.0,
        event_weight: float = 3.0,
        time_imbalance_weight: float = 2.0,
        stream_count_weight: float = 2.0,
        event_local_weight: float = 3.0,
        time_conflict_weight: float = 4.0,
        time_load_weight: float = 2.0,
        new_flow_penalty_weight: float = 2.0,
    ) -> None:
        self.physical_stream_limit = physical_stream_limit
        self.window_width = window_width
        self.candidate_limit = candidate_limit
        self.light_stream_limit = light_stream_limit
        self.repair_moves = repair_moves
        self.resim_candidate_limit = resim_candidate_limit
        self.ressimspan_weight = ressimspan_weight
        self.stage_var_weight = stage_var_weight
        self.event_weight = event_weight
        self.time_imbalance_weight = time_imbalance_weight
        self.stream_count_weight = stream_count_weight
        self.event_local_weight = event_local_weight
        self.time_conflict_weight = time_conflict_weight
        self.time_load_weight = time_load_weight
        self.new_flow_penalty_weight = new_flow_penalty_weight


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


class _LiteScore:
    __slots__ = (
        "total",
        "time_conflict",
        "event_local",
        "time_load",
        "candidate_stream",
    )

    def __init__(
        self,
        total: float = 0.0,
        time_conflict: float = 0.0,
        event_local: int = 0,
        time_load: float = 0.0,
        candidate_stream: int = 0,
    ) -> None:
        self.total = total
        self.time_conflict = time_conflict
        self.event_local = event_local
        self.time_load = time_load
        self.candidate_stream = candidate_stream


class _ResourceScheduleResult:
    __slots__ = ("feasible", "makespan")

    def __init__(self, feasible: bool = False, makespan: float = 0.0) -> None:
        self.feasible = feasible
        self.makespan = makespan


class _RunningNode:
    __slots__ = ("finish_time", "topo_pos", "node_index")

    def __init__(self, finish_time: float, topo_pos: int, node_index: int) -> None:
        self.finish_time = finish_time
        self.topo_pos = topo_pos
        self.node_index = node_index


class _SimulationContext:
    __slots__ = (
        "node_streams",
        "stream_to_queue",
        "filtered_preds",
        "finished",
        "running_in_stream",
        "running_heap",
        "current_time",
        "executed_count",
    )

    def __init__(self) -> None:
        self.node_streams: List[int] = []
        self.stream_to_queue: Dict[int, deque] = {}
        self.filtered_preds: List[List[int]] = []
        self.finished: List[bool] = []
        self.running_in_stream: List[int] = []
        # A heap of ``_RunningNode`` objects is kept as a list.  The helper
        # methods below use the GE epsilon-aware comparator when selecting and
        # removing entries; native tuple ordering would distinguish values
        # that C++ intentionally treats as equal.
        self.running_heap: List[_RunningNode] = []
        self.current_time = 0.0
        self.executed_count = 0


class _StreamingState:
    __slots__ = (
        "assignment",
        "stream_total_durations",
        "stream_duration_hists",
        "origin_stream_to_flow_counts",
        "active_stream_count",
    )

    def __init__(self) -> None:
        self.assignment: List[int] = []
        self.stream_total_durations: List[float] = []
        self.stream_duration_hists: List[List[float]] = []
        self.origin_stream_to_flow_counts: Dict[int, Dict[int, int]] = {}
        self.active_stream_count = 0


class _WeightedSolver:
    __slots__ = (
        "graph",
        "routes",
        "options",
        "costs",
        "nodes",
        "n",
        "name_to_idx",
        "preds",
        "succs",
        "topo_order",
        "position",
        "levels",
        "max_level",
        "edge_pairs",
        "unit_profiles",
        "owner",
        "unit_count",
        "pred_edges",
        "succ_edges",
        "node_durations",
        "node_origin_stream_hints",
        "node_aiv_cores",
        "node_aic_cores",
        "node_bottom_ranks",
        "unit_total_duration",
        "unit_total_aiv_time",
        "unit_total_aic_time",
        "unit_duration_hist",
        "unit_peak_aiv_hist",
        "unit_peak_aic_hist",
        "unit_peak_aiv",
        "unit_peak_aic",
        "unit_rank",
        "unit_origin_stream_hint",
        "unit_active_levels",
        "units_by_earliest_level",
    )

    def __init__(
        self,
        graph,
        routes: Sequence[Sequence[int]],
        options: WeightedStreamMergeOptions,
        costs: Optional[Sequence[NodeCost]] = None,
    ) -> None:
        self.graph = graph
        self.routes = routes
        self.options = options
        self.nodes = sorted(graph.get_nodes(), key=lambda node: node._index)
        self.n = len(self.nodes)
        self.name_to_idx = {node._name: idx for idx, node in enumerate(self.nodes)}
        self.preds = [[] for _ in range(self.n)]
        self.succs = [[] for _ in range(self.n)]
        self.topo_order: List[int] = []
        self.position: List[int] = []
        self.levels: List[int] = []
        self.max_level = 0
        self.edge_pairs: List[Tuple[int, int]] = []
        self.unit_profiles: List[_UnitProfile] = []
        self.owner: List[int] = []
        self.unit_count = 0
        self.pred_edges: List[Dict[int, int]] = []
        self.succ_edges: List[Dict[int, int]] = []
        self.node_durations: List[float] = []
        self.node_origin_stream_hints: List[int] = []
        self.node_aiv_cores: List[int] = []
        self.node_aic_cores: List[int] = []
        self.node_bottom_ranks: List[float] = []
        self.unit_total_duration: List[float] = []
        self.unit_total_aiv_time: List[float] = []
        self.unit_total_aic_time: List[float] = []
        self.unit_duration_hist: List[List[float]] = []
        self.unit_peak_aiv_hist: List[List[int]] = []
        self.unit_peak_aic_hist: List[List[int]] = []
        self.unit_peak_aiv: List[int] = []
        self.unit_peak_aic: List[int] = []
        self.unit_rank: List[float] = []
        self.unit_origin_stream_hint: List[int] = []
        self.unit_active_levels: List[List[int]] = []
        self.units_by_earliest_level: Dict[int, List[int]] = {}

        self._build_edges()
        self._build_topology()
        self._build_units()
        self.costs = costs
        self._load_node_cost_profiles()

    def _build_edges(self) -> None:
        # Keep the staging helpers idempotent, as their C++ counterparts are:
        # reusing the old adjacency arrays would append duplicate edges and
        # alter every subsequent level/route decision.
        self.preds = [[] for _ in range(self.n)]
        self.succs = [[] for _ in range(self.n)]
        self.edge_pairs = []
        for idx, node in enumerate(self.nodes):
            pred_indices = set()
            for pred in node._get_input_nodes():
                pred_index = self.name_to_idx.get(pred._name)
                if pred_index is not None:
                    pred_indices.add(pred_index)
            self.preds[idx] = sorted(pred_indices)
            for pred_index in self.preds[idx]:
                self.succs[pred_index].append(idx)
                self.edge_pairs.append((pred_index, idx))

    def _build_topology(self) -> None:
        self.topo_order = []
        self.max_level = 0
        indegree = [len(preds) for preds in self.preds]
        ready = list(i for i, degree in enumerate(indegree) if degree == 0)
        heapq.heapify(ready)
        while ready:
            idx = heapq.heappop(ready)
            self.topo_order.append(idx)
            for succ in self.succs[idx]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    heapq.heappush(ready, succ)
        if len(self.topo_order) != self.n:
            logger.error("Weighted merge only supports DAG graphs.")
            raise RuntimeError(
                "graph contains a cycle; weighted merge only supports DAG graphs"
            )
        self.position = [-1] * self.n
        self.levels = [0] * self.n
        for pos, idx in enumerate(self.topo_order):
            self.position[idx] = pos
            self.levels[idx] = max(
                (self.levels[p] + 1 for p in self.preds[idx]), default=0
            )
            self.max_level = max(self.max_level, self.levels[idx])

    def _build_units(self) -> None:
        self.unit_count = len(self.routes)
        self.owner = [-1] * self.n
        self.unit_profiles = []
        for uid, route in enumerate(self.routes):
            indices: List[int] = []
            for node_index in route:
                if (
                    isinstance(node_index, bool)
                    or not isinstance(node_index, int)
                    or node_index < 0
                    or node_index >= self.n
                ):
                    logger.error(
                        "Node index %s is out of range [0, %d).", node_index, self.n
                    )
                    raise ValueError(f"node index {node_index!r} is out of range")
                if self.owner[node_index] != -1:
                    logger.error(
                        "Node %s appears in more than one logical stream.",
                        self.nodes[node_index]._name,
                    )
                    raise ValueError(
                        f"node {node_index} appears in more than one logical stream"
                    )
                self.owner[node_index] = uid
                indices.append(node_index)
            if not indices:
                logger.error("Weighted logical stream %d should not be empty.", uid)
                raise ValueError(f"logical stream {uid} should not be empty")
            indices.sort(key=lambda idx: self.position[idx])
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
                    uid, indices, len(indices), earliest, latest, peak, level_hist
                )
            )
        for node_index, value in enumerate(self.owner):
            if value < 0:
                logger.error(
                    "Node %s is not assigned to any logical stream.",
                    self.nodes[node_index]._name,
                )
                raise ValueError("not all graph nodes are assigned to logical streams")

    @staticmethod
    def _size_to_int32(value: int) -> int:
        return min(int(value), 2**31 - 1)

    def _load_node_cost_profiles(self) -> None:
        self.node_durations = [0.0] * self.n
        # The C++ implementation uses the logical stream id as the origin
        # hint.  Keep this value even though most route sets have one unique
        # hint per unit; it is consumed by the candidate/repair path.
        self.node_origin_stream_hints = list(self.owner)
        self.node_aiv_cores = [0] * self.n
        self.node_aic_cores = [0] * self.n
        self.node_bottom_ranks = [0.0] * self.n
        valid_cost_count = 0
        missing_cost_count = 0
        for idx, node in enumerate(self.nodes):
            cost = (
                _cost_of(node, self.costs)
                if self.costs is not None
                else node.get_node_cost()
            )
            self.node_durations[idx] = (
                float(cost.execution_time) if cost.execution_time >= 0.0 else 0.0
            )
            self.node_aiv_cores[idx] = self._size_to_int32(cost.vec_block_num)
            self.node_aic_cores[idx] = self._size_to_int32(cost.cube_block_num)
            if cost.execution_time >= 0.0:
                valid_cost_count += 1
            else:
                missing_cost_count += 1
        if missing_cost_count > 0:
            logger.warning(
                "Weighted stream NodeCost summary: valid=%d, missing=%d.",
                valid_cost_count,
                missing_cost_count,
            )
        else:
            logger.info(
                "Weighted stream NodeCost summary: valid=%d, missing=%d.",
                valid_cost_count,
                missing_cost_count,
            )
        self._compute_bottom_ranks()
        self._build_resource_unit_profiles()

    def _compute_bottom_ranks(self) -> None:
        for idx in reversed(self.topo_order):
            succ_rank = 0.0
            for succ in self.succs[idx]:
                succ_rank = max(succ_rank, self.node_bottom_ranks[succ])
            self.node_bottom_ranks[idx] = self.node_durations[idx] + succ_rank

    def _init_resource_unit_profiles(self) -> None:
        level_count = self.max_level + 1
        self.unit_total_duration = [0.0] * self.unit_count
        self.unit_total_aiv_time = [0.0] * self.unit_count
        self.unit_total_aic_time = [0.0] * self.unit_count
        self.unit_duration_hist = [[0.0] * level_count for _ in range(self.unit_count)]
        self.unit_peak_aiv_hist = [[0] * level_count for _ in range(self.unit_count)]
        self.unit_peak_aic_hist = [[0] * level_count for _ in range(self.unit_count)]
        self.unit_peak_aiv = [0] * self.unit_count
        self.unit_peak_aic = [0] * self.unit_count
        self.unit_rank = [0.0] * self.unit_count
        self.unit_origin_stream_hint = [-1] * self.unit_count
        self.unit_active_levels = [[] for _ in range(self.unit_count)]

    def _build_resource_unit_profiles(self) -> None:
        self._init_resource_unit_profiles()
        for unit in self.unit_profiles:
            uid = unit.unit_id
            origin_counter: Dict[int, int] = {}
            for node_index in unit.node_indices:
                level = self.levels[node_index]
                duration = self.node_durations[node_index]
                aiv = self.node_aiv_cores[node_index]
                aic = self.node_aic_cores[node_index]
                origin = self.node_origin_stream_hints[node_index]
                origin_counter[origin] = origin_counter.get(origin, 0) + 1
                self.unit_total_duration[uid] += duration
                self.unit_total_aiv_time[uid] += duration * float(aiv)
                self.unit_total_aic_time[uid] += duration * float(aic)
                self.unit_duration_hist[uid][level] += duration
                self.unit_peak_aiv_hist[uid][level] = max(
                    self.unit_peak_aiv_hist[uid][level], aiv
                )
                self.unit_peak_aic_hist[uid][level] = max(
                    self.unit_peak_aic_hist[uid][level], aic
                )
                self.unit_rank[uid] = max(
                    self.unit_rank[uid], self.node_bottom_ranks[node_index]
                )
            self.unit_peak_aiv[uid] = max(self.unit_peak_aiv_hist[uid], default=0)
            self.unit_peak_aic[uid] = max(self.unit_peak_aic_hist[uid], default=0)
            best_count = -1
            best_stream = -1
            for origin, count in sorted(origin_counter.items()):
                if count > best_count or (count == best_count and origin < best_stream):
                    best_count, best_stream = count, origin
            self.unit_origin_stream_hint[uid] = best_stream
            for level in range(self.max_level + 1):
                if (
                    self.unit_duration_hist[uid][level] > K_IMPROVE_EPS
                    or self.unit_peak_aiv_hist[uid][level] > 0
                    or self.unit_peak_aic_hist[uid][level] > 0
                ):
                    self.unit_active_levels[uid].append(level)

    def _build_unit_dependency_graph(self) -> None:
        self.pred_edges = [{} for _ in range(self.unit_count)]
        self.succ_edges = [{} for _ in range(self.unit_count)]
        interactions = [0] * self.unit_count
        for src, dst in self.edge_pairs:
            src_unit, dst_unit = self.owner[src], self.owner[dst]
            if src_unit == dst_unit:
                continue
            self.succ_edges[src_unit][dst_unit] = (
                self.succ_edges[src_unit].get(dst_unit, 0) + 1
            )
            self.pred_edges[dst_unit][src_unit] = (
                self.pred_edges[dst_unit].get(src_unit, 0) + 1
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

        def compare(lhs: int, rhs: int) -> int:
            cmp = _compare_double(
                self.unit_total_duration[rhs], self.unit_total_duration[lhs]
            )
            if cmp:
                return cmp
            lhs_resource = self.unit_total_aiv_time[lhs] + self.unit_total_aic_time[lhs]
            rhs_resource = self.unit_total_aiv_time[rhs] + self.unit_total_aic_time[rhs]
            cmp = _compare_double(rhs_resource, lhs_resource)
            if cmp:
                return cmp
            lhs_peak = max(self.unit_peak_aiv[lhs], self.unit_peak_aic[lhs])
            rhs_peak = max(self.unit_peak_aiv[rhs], self.unit_peak_aic[rhs])
            if lhs_peak != rhs_peak:
                return -1 if lhs_peak > rhs_peak else 1
            lhs_interaction = self.unit_profiles[lhs].interaction
            rhs_interaction = self.unit_profiles[rhs].interaction
            if lhs_interaction != rhs_interaction:
                return -1 if lhs_interaction > rhs_interaction else 1
            lhs_level = self.unit_profiles[lhs].earliest_level
            rhs_level = self.unit_profiles[rhs].earliest_level
            if lhs_level != rhs_level:
                return -1 if lhs_level < rhs_level else 1
            return -1 if lhs < rhs else (1 if lhs > rhs else 0)

        for ids in self.units_by_earliest_level.values():
            ids.sort(key=cmp_to_key(compare))

    def _new_streaming_state(self) -> _StreamingState:
        state = _StreamingState()
        state.assignment = [-1] * self.unit_count
        state.stream_total_durations = [0.0] * self.options.physical_stream_limit
        state.stream_duration_hists = [
            [0.0] * (self.max_level + 1)
            for _ in range(self.options.physical_stream_limit)
        ]
        return state

    def _window_end(self, start_level: int) -> int:
        return min(self.max_level, start_level + self.options.window_width - 1)

    def _window_mass(self, hist: Sequence[float], start_level: int) -> float:
        return sum(hist[start_level : self._window_end(start_level) + 1])

    def _stream_window_duration_load(
        self, state: _StreamingState, stream: int, level: int
    ) -> float:
        return self._window_mass(state.stream_duration_hists[stream], level)

    def _window_active_levels(self, uid: int, start_level: int) -> List[int]:
        end_level = self._window_end(start_level)
        levels = [
            level
            for level in self.unit_active_levels[uid]
            if start_level <= level <= end_level
        ]
        return levels if levels else list(self.unit_active_levels[uid])

    def _adjacent_assigned_flows(
        self, uid: int, assignment: Sequence[int]
    ) -> Dict[int, int]:
        flows: Dict[int, int] = {}
        for neighbor, count in self.pred_edges[uid].items():
            stream = assignment[neighbor]
            if stream >= 0:
                flows[stream] = flows.get(stream, 0) + count
        for neighbor, count in self.succ_edges[uid].items():
            stream = assignment[neighbor]
            if stream >= 0:
                flows[stream] = flows.get(stream, 0) + count
        return flows

    def _origin_hint_flows(self, uid: int, state: _StreamingState) -> List[int]:
        hint = self.unit_origin_stream_hint[uid]
        if hint < 0 or hint not in state.origin_stream_to_flow_counts:
            return []
        counts = state.origin_stream_to_flow_counts[hint]

        def compare(lhs: int, rhs: int) -> int:
            if counts[lhs] != counts[rhs]:
                return -1 if counts[lhs] > counts[rhs] else 1
            cmp = _compare_double(
                state.stream_total_durations[lhs], state.stream_total_durations[rhs]
            )
            if cmp:
                return cmp
            return -1 if lhs < rhs else (1 if lhs > rhs else 0)

        return sorted(counts, key=cmp_to_key(compare))

    def _adjacent_candidate_flows(self, uid: int, state: _StreamingState) -> List[int]:
        weights = self._adjacent_assigned_flows(uid, state.assignment)

        def compare(lhs: int, rhs: int) -> int:
            if weights[lhs] != weights[rhs]:
                return -1 if weights[lhs] > weights[rhs] else 1
            cmp = _compare_double(
                state.stream_total_durations[lhs], state.stream_total_durations[rhs]
            )
            if cmp:
                return cmp
            return -1 if lhs < rhs else (1 if lhs > rhs else 0)

        return sorted(weights, key=cmp_to_key(compare))

    def _light_candidate_flows(self, level: int, state: _StreamingState) -> List[int]:
        streams = list(range(state.active_stream_count))

        def compare(lhs: int, rhs: int) -> int:
            cmp = _compare_double(
                self._stream_window_duration_load(state, lhs, level),
                self._stream_window_duration_load(state, rhs, level),
            )
            if cmp:
                return cmp
            cmp = _compare_double(
                state.stream_total_durations[lhs], state.stream_total_durations[rhs]
            )
            if cmp:
                return cmp
            return -1 if lhs < rhs else (1 if lhs > rhs else 0)

        streams.sort(key=cmp_to_key(compare))
        return streams[: self.options.light_stream_limit]

    def _unique_limited_candidates(self, candidates: Sequence[int]) -> List[int]:
        result: List[int] = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
            if len(result) >= self.options.candidate_limit:
                break
        return result

    def _fallback_candidate_flows(
        self, state: _StreamingState, include_new: bool
    ) -> List[int]:
        if (
            include_new
            and state.active_stream_count < self.options.physical_stream_limit
        ):
            return [state.active_stream_count]
        return list(range(min(state.active_stream_count, self.options.candidate_limit)))

    def _candidate_flows(
        self, uid: int, level: int, state: _StreamingState, include_new: bool
    ) -> List[int]:
        candidates = self._adjacent_candidate_flows(uid, state)
        candidates.extend(self._origin_hint_flows(uid, state))
        candidates.extend(self._light_candidate_flows(level, state))
        if (
            include_new
            and state.active_stream_count < self.options.physical_stream_limit
        ):
            candidates.append(state.active_stream_count)
        unique = self._unique_limited_candidates(candidates)
        return unique if unique else self._fallback_candidate_flows(state, include_new)

    def _calc_event_local(
        self, uid: int, candidate: int, assignment: Sequence[int]
    ) -> int:
        event = 0
        for neighbor, count in self.pred_edges[uid].items():
            stream = assignment[neighbor]
            if stream >= 0 and stream != candidate:
                event += count
        for neighbor, count in self.succ_edges[uid].items():
            stream = assignment[neighbor]
            if stream >= 0 and stream != candidate:
                event += count
        return event

    def _evaluate_lite_score(
        self, uid: int, candidate: int, level: int, state: _StreamingState
    ) -> _LiteScore:
        score = _LiteScore()
        opens = candidate == state.active_stream_count
        score.event_local = self._calc_event_local(uid, candidate, state.assignment)
        for current_level in self._window_active_levels(uid, level):
            score.time_conflict += (
                self.unit_duration_hist[uid][current_level]
                * state.stream_duration_hists[candidate][current_level]
            )
        score.time_load = (
            state.stream_total_durations[candidate] + self.unit_total_duration[uid]
        )
        score.candidate_stream = candidate
        score.total = (
            self.options.event_local_weight * float(score.event_local)
            + self.options.time_conflict_weight * score.time_conflict
            + self.options.time_load_weight * score.time_load
            + self.options.new_flow_penalty_weight * float(opens)
        )
        return score

    @staticmethod
    def _lite_score_less(lhs: _LiteScore, rhs: _LiteScore) -> bool:
        cmp = _compare_double(lhs.total, rhs.total)
        if cmp:
            return cmp < 0
        cmp = _compare_double(lhs.time_conflict, rhs.time_conflict)
        if cmp:
            return cmp < 0
        if lhs.event_local != rhs.event_local:
            return lhs.event_local < rhs.event_local
        cmp = _compare_double(lhs.time_load, rhs.time_load)
        if cmp:
            return cmp < 0
        return lhs.candidate_stream < rhs.candidate_stream

    def _apply_unit_to_stream(
        self, uid: int, candidate: int, state: _StreamingState
    ) -> bool:
        opens = candidate == state.active_stream_count
        if candidate < 0 or candidate >= self.options.physical_stream_limit:
            raise RuntimeError("candidate physical stream is out of range")
        if opens:
            state.active_stream_count += 1
        if candidate >= state.active_stream_count:
            raise RuntimeError("candidate physical stream is out of range")
        state.assignment[uid] = candidate
        state.stream_total_durations[candidate] += self.unit_total_duration[uid]
        for level in self.unit_active_levels[uid]:
            state.stream_duration_hists[candidate][level] += self.unit_duration_hist[
                uid
            ][level]
        hint = self.unit_origin_stream_hint[uid]
        if hint >= 0:
            counter = state.origin_stream_to_flow_counts.setdefault(hint, {})
            counter[candidate] = counter.get(candidate, 0) + 1
        return opens

    def _remove_unit_from_stream(self, uid: int, state: _StreamingState) -> int:
        stream = state.assignment[uid]
        if stream < 0:
            logger.error("Try to remove unassigned weighted unit %d.", uid)
            return -1
        state.assignment[uid] = -1
        state.stream_total_durations[stream] -= self.unit_total_duration[uid]
        for level in self.unit_active_levels[uid]:
            state.stream_duration_hists[stream][level] -= self.unit_duration_hist[uid][
                level
            ]
        hint = self.unit_origin_stream_hint[uid]
        if hint >= 0 and hint in state.origin_stream_to_flow_counts:
            counter = state.origin_stream_to_flow_counts[hint]
            count = counter.get(stream, 0) - 1
            if count <= 0:
                counter.pop(stream, None)
            else:
                counter[stream] = count
            if not counter:
                state.origin_stream_to_flow_counts.pop(hint, None)
        return stream

    def _build_simulation_node_streams(
        self,
        assignment: Sequence[int],
        include_unassigned: bool,
        node_streams: List[int],
        active_mask: List[int],
    ) -> int:
        active_nodes = 0
        for node_index in range(self.n):
            unit_id = self.owner[node_index]
            if unit_id < 0 or unit_id >= len(assignment):
                return -1
            stream = assignment[unit_id]
            if stream < 0:
                if include_unassigned:
                    logger.error(
                        "There is still an unassigned weighted unit in final simulation."
                    )
                    return -1
                continue
            node_streams[node_index] = stream
            active_mask[node_index] = 1
            active_nodes += 1
        return active_nodes

    def _build_simulation_queues(
        self, node_streams: Sequence[int], active_mask: Sequence[int]
    ) -> Dict[int, deque]:
        queues: Dict[int, deque] = {}
        for node_index in self.topo_order:
            if active_mask[node_index]:
                queues.setdefault(node_streams[node_index], deque()).append(node_index)
        return queues

    def _build_simulation_filtered_preds(
        self, active_mask: Sequence[int]
    ) -> List[List[int]]:
        filtered = [[] for _ in range(self.n)]
        for node_index in range(self.n):
            if active_mask[node_index]:
                filtered[node_index] = [
                    p for p in self.preds[node_index] if active_mask[p]
                ]
        return filtered

    def _new_simulation_context(
        self, node_streams: List[int], active_mask: List[int]
    ) -> _SimulationContext:
        context = _SimulationContext()
        context.node_streams = node_streams
        context.stream_to_queue = self._build_simulation_queues(
            node_streams, active_mask
        )
        context.filtered_preds = self._build_simulation_filtered_preds(active_mask)
        context.finished = [False] * self.n
        context.running_in_stream = [0] * self.options.physical_stream_limit
        return context

    @staticmethod
    def _are_simulation_preds_finished(
        node_index: int,
        filtered_preds: Sequence[Sequence[int]],
        finished: Sequence[bool],
    ) -> bool:
        return all(finished[pred] for pred in filtered_preds[node_index])

    def _sort_simulation_ready_heads(self, heads: List[int]) -> None:
        heads.sort(key=cmp_to_key(self._simulation_node_compare))

    def _simulation_node_compare(self, lhs: int, rhs: int) -> int:
        cmp = _compare_double(self.node_bottom_ranks[rhs], self.node_bottom_ranks[lhs])
        if cmp:
            return cmp
        cmp = _compare_double(self.node_durations[rhs], self.node_durations[lhs])
        if cmp:
            return cmp
        if self.position[lhs] != self.position[rhs]:
            return -1 if self.position[lhs] < self.position[rhs] else 1
        return -1 if lhs < rhs else (1 if lhs > rhs else 0)

    def _simulation_ready_heads(self, context: _SimulationContext) -> List[int]:
        """Return ready queue heads in deterministic priority order."""

        stream_to_queue = context.stream_to_queue
        filtered_preds = context.filtered_preds
        finished = context.finished
        running_in_stream = context.running_in_stream
        heads: List[int] = []
        for stream in sorted(stream_to_queue):
            queue = stream_to_queue[stream]
            if not queue or running_in_stream[stream]:
                continue
            if self._are_simulation_preds_finished(queue[0], filtered_preds, finished):
                heads.append(queue[0])
        self._sort_simulation_ready_heads(heads)
        return heads

    @staticmethod
    def _running_less(lhs: _RunningNode, rhs: _RunningNode) -> bool:
        cmp = _compare_double(lhs.finish_time, rhs.finish_time)
        if cmp:
            return cmp < 0
        if lhs.topo_pos != rhs.topo_pos:
            return lhs.topo_pos < rhs.topo_pos
        return lhs.node_index < rhs.node_index

    @classmethod
    def _heap_push_running(cls, heap: List[_RunningNode], value: _RunningNode) -> None:
        heap.append(value)
        idx = len(heap) - 1
        while idx:
            parent = (idx - 1) // 2
            if not cls._running_less(heap[idx], heap[parent]):
                break
            heap[idx], heap[parent] = heap[parent], heap[idx]
            idx = parent

    @classmethod
    def _heap_pop_running(cls, heap: List[_RunningNode]) -> _RunningNode:
        if not heap:
            raise RuntimeError("weighted simulation heap is empty")
        result = heap[0]
        tail = heap.pop()
        if heap:
            heap[0] = tail
            idx = 0
            while True:
                left = idx * 2 + 1
                if left >= len(heap):
                    break
                right = left + 1
                child = left
                if right < len(heap) and cls._running_less(heap[right], heap[left]):
                    child = right
                if not cls._running_less(heap[child], heap[idx]):
                    break
                heap[idx], heap[child] = heap[child], heap[idx]
                idx = child
        return result

    @classmethod
    def _heap_peek_min(cls, heap: Sequence[_RunningNode]) -> _RunningNode:
        if not heap:
            raise RuntimeError("weighted simulation heap is empty")
        # ``running_heap`` models the C++ ``std::priority_queue`` used by
        # MiniDAG's simulator.  Its ``top()`` operation returns the heap root,
        # rather than searching the backing array for a globally minimal
        # element.  The epsilon-aware comparator is intentionally kept in the
        # same form as the reference implementation (and is not a strict
        # total order for values within the tolerance), so scanning here can
        # select an element that the heap would not expose through ``top()``.
        # Returning the root preserves the reference event ordering exactly.
        return heap[0]

    def _start_simulation_ready_head(
        self, heads: Sequence[int], context: _SimulationContext
    ) -> bool:
        if not heads:
            return True
        node_index = heads[0]
        stream = context.node_streams[node_index]
        queue = context.stream_to_queue.get(stream)
        if queue is None or not queue or queue[0] != node_index:
            logger.error("Weighted stream simulation queue state is inconsistent.")
            return False
        queue.popleft()
        context.running_in_stream[stream] = 1
        self._heap_push_running(
            context.running_heap,
            _RunningNode(
                context.current_time + self.node_durations[node_index],
                self.position[node_index],
                node_index,
            ),
        )
        return True

    def _finish_next_simulation_nodes(self, context: _SimulationContext) -> None:
        next_finish = self._heap_peek_min(context.running_heap).finish_time
        context.current_time = next_finish
        while context.running_heap:
            candidate = self._heap_peek_min(context.running_heap)
            if _compare_double(candidate.finish_time, next_finish) != 0:
                break
            finished = self._heap_pop_running(context.running_heap)
            context.finished[finished.node_index] = True
            context.running_in_stream[context.node_streams[finished.node_index]] = 0
            context.executed_count += 1

    def _run_simulation_step(self, context: _SimulationContext) -> bool:
        # Keep the same two-stage operation as the C++ reference: enumerate
        # and sort every ready stream head, then start only the first one.
        # The epsilon-aware comparator is deliberately not a mathematical
        # total order for values inside its tolerance; replacing this sort
        # with a linear ``min`` changes tie behaviour on such inputs.
        heads = self._simulation_ready_heads(context)
        if not self._start_simulation_ready_head(heads, context):
            return False
        if not context.running_heap:
            return False
        self._finish_next_simulation_nodes(context)
        return True

    def _run_simulation(
        self, active_nodes: int, context: _SimulationContext
    ) -> _ResourceScheduleResult:
        while context.executed_count < active_nodes:
            if not self._run_simulation_step(context):
                return _ResourceScheduleResult(False, context.current_time)
        return _ResourceScheduleResult(True, context.current_time)

    def _build_and_run_simulation(
        self, assignment: Sequence[int], include_unassigned: bool
    ) -> _ResourceScheduleResult:
        node_streams = [-1] * self.n
        active_mask = [0] * self.n
        active_nodes = self._build_simulation_node_streams(
            assignment, include_unassigned, node_streams, active_mask
        )
        if active_nodes < 0:
            return _ResourceScheduleResult(False, 0.0)
        if active_nodes == 0:
            return _ResourceScheduleResult(True, 0.0)
        return self._run_simulation(
            active_nodes, self._new_simulation_context(node_streams, active_mask)
        )

    def _select_best_stream(
        self, uid: int, level: int, candidates: Sequence[int], state: _StreamingState
    ) -> int:
        scored = [
            self._evaluate_lite_score(uid, candidate, level, state)
            for candidate in candidates
        ]
        scored.sort(
            key=cmp_to_key(
                lambda lhs, rhs: -1
                if self._lite_score_less(lhs, rhs)
                else (1 if self._lite_score_less(rhs, lhs) else 0)
            )
        )
        if self.options.resim_candidate_limit <= 0:
            return scored[0].candidate_stream
        best_resim: Optional[_LiteScore] = None
        best_span = 0.0
        limit = min(len(scored), self.options.resim_candidate_limit)
        for candidate in scored[:limit]:
            previous_active = state.active_stream_count
            self._apply_unit_to_stream(uid, candidate.candidate_stream, state)
            schedule = self._build_and_run_simulation(state.assignment, False)
            self._remove_unit_from_stream(uid, state)
            state.active_stream_count = previous_active
            if not schedule.feasible:
                continue
            if (
                best_resim is None
                or _compare_double(schedule.makespan, best_span) < 0
                or (
                    _compare_double(schedule.makespan, best_span) == 0
                    and self._lite_score_less(candidate, best_resim)
                )
            ):
                best_resim = candidate
                best_span = schedule.makespan
        return (
            best_resim.candidate_stream
            if best_resim is not None
            else scored[0].candidate_stream
        )

    def _repair_recent_units(
        self, level: int, recent_units: Sequence[int], state: _StreamingState
    ) -> None:
        move_count = 0
        while move_count < self.options.repair_moves:
            best_unit = -1
            best_candidate = -1
            best_improvement = 0.0
            for uid in recent_units:
                previous_active = state.active_stream_count
                current_stream = self._remove_unit_from_stream(uid, state)
                if current_stream < 0:
                    return
                current_score = self._evaluate_lite_score(
                    uid, current_stream, level, state
                )
                candidates = self._candidate_flows(uid, level, state, False)
                if current_stream not in candidates:
                    candidates = list(candidates) + [current_stream]
                selected = self._select_best_stream(uid, level, candidates, state)
                best_score = self._evaluate_lite_score(uid, selected, level, state)
                self._apply_unit_to_stream(uid, current_stream, state)
                state.active_stream_count = previous_active
                improvement = current_score.total - best_score.total
                if (
                    selected != current_stream
                    and improvement > K_IMPROVE_EPS
                    and improvement > best_improvement
                ):
                    best_unit, best_candidate, best_improvement = (
                        uid,
                        selected,
                        improvement,
                    )
            if best_unit < 0:
                return
            self._remove_unit_from_stream(best_unit, state)
            self._apply_unit_to_stream(best_unit, best_candidate, state)
            move_count += 1

    def _compact_assignment(self, assignment: Sequence[int]) -> List[int]:
        # Empty logical routes are a valid no-op merge.  Keep the helper's
        # return shape aligned with the C++ implementation; callers that
        # supplied non-empty routes validate the mapping length afterwards.
        if not assignment:
            return []
        if any(stream < 0 for stream in assignment):
            logger.error(
                "There is still an unassigned weighted unit when compacting assignment."
            )
            raise RuntimeError(
                "multistream internal error: unassigned weighted unit when compacting assignment"
            )
        used = sorted(set(assignment))
        if not used:
            logger.error("Weighted merge should produce at least one physical stream.")
            raise RuntimeError(
                "multistream internal error: weighted merge produced no physical stream"
            )
        remap = {stream: idx for idx, stream in enumerate(used)}
        return [remap[stream] for stream in assignment]

    def solve(self) -> List[int]:
        if self.unit_count == 0:
            return []
        self._build_unit_dependency_graph()
        self._build_units_by_earliest_level()
        state = self._new_streaming_state()
        for level in range(self.max_level + 1):
            unit_ids = self.units_by_earliest_level.get(level)
            if not unit_ids:
                continue
            recent_units: List[int] = []
            for uid in unit_ids:
                candidates = self._candidate_flows(uid, level, state, True)
                if not candidates:
                    logger.error(
                        "No candidate physical stream for weighted unit %d.", uid
                    )
                    raise RuntimeError("no candidate physical stream for weighted unit")
                best = self._select_best_stream(uid, level, candidates, state)
                self._apply_unit_to_stream(uid, best, state)
                recent_units.append(uid)
            if self.options.repair_moves > 0 and recent_units:
                self._repair_recent_units(level, recent_units, state)
        return self._compact_assignment(state.assignment)


class _WeightedMerger:
    """Entry point used by the path-cover allocator."""

    __slots__ = ("options",)

    def __init__(self, options: Optional[WeightedStreamMergeOptions] = None):
        self.options = options if options is not None else WeightedStreamMergeOptions()

    def merge(
        self,
        graph,
        routes: Sequence[Sequence[int]],
        costs: Optional[Sequence[NodeCost]] = None,
    ) -> List[int]:
        options = self.options
        _validate_options(options)
        if not routes:
            return []
        return _WeightedSolver(graph, routes, options, costs).solve()


def merge(
    graph,
    routes: Sequence[Sequence[int]],
    options: Optional[WeightedStreamMergeOptions] = None,
    costs: Optional[Sequence[NodeCost]] = None,
) -> List[int]:
    """Convenience wrapper for the weighted logical-stream merger."""

    return _WeightedMerger(options).merge(graph, routes, costs)


def _validate_options(options: WeightedStreamMergeOptions) -> None:
    integer_options = (
        "physical_stream_limit",
        "window_width",
        "candidate_limit",
        "light_stream_limit",
        "repair_moves",
        "resim_candidate_limit",
    )
    for name in integer_options:
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if options.physical_stream_limit <= 0:
        logger.error(
            "Weighted stream merge physical stream limit must be greater than 0."
        )
        raise ValueError("physical stream limit must be greater than 0")
    if options.window_width <= 0:
        logger.error("Weighted stream merge window width must be greater than 0.")
        raise ValueError("window width must be greater than 0")
    if options.candidate_limit <= 0:
        logger.error("Weighted stream merge candidate limit must be greater than 0.")
        raise ValueError("candidate limit must be greater than 0")
    if options.light_stream_limit <= 0:
        logger.error("Weighted stream merge light_stream_limit must be greater than 0.")
        raise ValueError("light stream limit must be greater than 0")
    if options.repair_moves < 0:
        logger.error(
            "Weighted stream merge repair_moves must be greater than or equal to 0."
        )
        raise ValueError("repair_moves must be non-negative")
    if options.resim_candidate_limit < 0:
        logger.error(
            "Weighted stream merge resim_candidate_limit must be greater than or equal to 0."
        )
        raise ValueError("resim_candidate_limit must be non-negative")


__all__ = ["WeightedStreamMergeOptions", "_WeightedMerger"]
