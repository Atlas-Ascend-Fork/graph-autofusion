/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_gen_utils.h"
#include <iostream>
#include <regex>
#include "common/ge_common/error_codes_define_af.h"
#include "gtest/gtest.h"
#include "gen_model_info.h"
#include "graph/types_af.h"
#include "ascir_ops.h"
#include "test_fa_ascir_graph.h"
#include "base/att_const_values.h"
#include "util/thread_local_context.h"
#include "gen_tiling_impl.h"
#include "api_tiling_gen/gen_api_tiling.h"
#include "tiling_code_generator.h"
#include "transpose_base_type.h"
#include "graph_construct_utils.h"
#include "common/platform_context.h"

using namespace att;
using att::test::CombineTilings;
using namespace af::ascir_op;
namespace {
af::Status CreateFaAscGraphWithAllModelInfos(const af::char_t *tiling_data_name, const af::char_t *algorithm_type,
                                             af::AscGraph &graph, FusedParsedScheduleResult &all_model_infos,
                                             std::map<std::string, std::string> &options) {
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  if (strlen(tiling_data_name) != 0) {
    options.emplace(kTilingDataTypeName, tiling_data_name);
  }
  options.emplace(kOutputFilePath, kDefaultFilePath);
  options.emplace(kGenConfigType, "HighPerf");

  std::vector<ascir::ScheduledResult> schedule_results1;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.emplace_back(graph);
  schedule_result.schedule_groups.emplace_back(schedule_group);
  schedule_results1.emplace_back(schedule_result);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(schedule_results1);
  return GetModelInfoMap(fused_schedule_result, options, all_model_infos);
}

ascir::FusedScheduledResult CreateScheduleResultWithSingleAscGraph(
    const std::vector<size_t> &schedule_results_group_num, const af::AscGraph &graph,
    const size_t sub_impl_graph_in_group = 1UL) {
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results;
  for (size_t i = 0UL; i < schedule_results_group_num.size(); i++) {
    ascir::ScheduledResult schedule_result;
    for (size_t j = 0UL; j < schedule_results_group_num[i]; j++) {
      ascir::ScheduleGroup schedule_group;
      for (size_t k = 0UL; k < sub_impl_graph_in_group; k++) {
        schedule_group.impl_graphs.emplace_back(graph);
      }
      schedule_result.schedule_groups.emplace_back(schedule_group);
    }
    schedule_results.emplace_back(schedule_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  return fused_schedule_result;
}

bool HasVariableName(const ModelInfo &model_info, const std::string &name) {
  for (const auto &item : model_info.variable_name_map) {
    if (item.second == name) {
      return true;
    }
  }
  return false;
}
}  // namespace

namespace af {
namespace ascir {
namespace cg {

// 辅助函数：验证排列有效性
bool IsValidPermutation(const std::vector<int64_t> &perm) {
  std::set<int64_t> unique(perm.begin(), perm.end());
  if (unique.size() != perm.size()) return false;
  for (auto idx : perm) {
    if (idx < 0 || idx >= static_cast<int64_t>(perm.size())) return false;
  }
  return true;
}

void SetMemReuseIds(const std::vector<std::pair<std::string, int>> &id_map, af::AscGraph &graph) {
  for (const auto &[name, id] : id_map) {
    graph.FindNode(name.c_str())->outputs[0].attr.mem.reuse_id = id;
  }
}

std::vector<af::Expression> CalcTransposedStrides(const std::vector<af::Expression> &dims,
                                                  const std::vector<int64_t> &perm) {
  std::vector<af::Expression> strides(perm.size());
  af::Expression stride = af::Symbol(1);
  for (int i = static_cast<int>(perm.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride = stride * dims[perm[i]];
  }
  return strides;
}

static af::Expression One = af::Symbol(1);
Status Build2DTransposeAscendGraph(af::AscGraph &graph, const std::vector<int64_t> &perm = {1, 0}) {
  if (perm.size() != 2 || !IsValidPermutation(perm)) {
    return af::FAILED;
  }
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  std::vector<af::Axis> axes = {z0, z1};
  std::vector<af::Expression> dims = {s0, s1};

  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z1}, 0, FORMAT_ND);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0, z1}, 1, FORMAT_ND);
  *data1.repeats = {s0, s1};
  *data2.repeats = {s0, s1};
  *data1.strides = {s1, One};
  *data2.strides = {s1, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto add = Add("add", load1, load2).TBuf(Position::kPositionVecCalc);
      *(add.vectorized_axis) = {z0.id, z1.id};
      *(add.axis) = {z0.id, z1.id};
      *(add.repeats) = {s0, s1};
      *(add.strides) = {s1, One};

      auto transpose = Transpose("transpose", add).TQue(Position::kPositionVecOut, 1, 2);
      *(transpose.vectorized_axis) = {axes[perm[0]].id, axes[perm[1]].id};
      *(transpose.axis) = {axes[perm[0]].id, axes[perm[1]].id};
      *(transpose.repeats) = {dims[perm[0]], dims[perm[1]]};
      *(transpose.strides) = CalcTransposedStrides(dims, perm);
      auto store = Store("store", transpose);
      auto output = Output("output", store);
      SetMemReuseIds({{"load1", 0}, {"load2", 1}, {"transpose", 2}, {"add", 3}}, graph);
    }
  }

  auto transpose_node = graph.FindNode("transpose");
  GE_ASSERT_NOTNULL(transpose_node);
  transpose_node->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

Status Build2DPadAscendGraph(af::AscGraph &graph, const std::vector<int64_t> &perm = {1, 0}) {
  if (perm.size() != 2 || !IsValidPermutation(perm)) {
    return af::FAILED;
  }
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  std::vector<af::Axis> axes = {z0, z1};
  std::vector<af::Expression> dims = {s0, s1};

  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z1}, 0, FORMAT_ND);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0, z1}, 1, FORMAT_ND);
  *data1.repeats = {s0, s1};
  *data2.repeats = {s0, s1};
  *data1.strides = {s1, One};
  *data2.strides = {s1, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto add = Add("add", load1, load2).TBuf(Position::kPositionVecCalc);
      *(add.vectorized_axis) = {z0.id, z1.id};
      *(add.axis) = {z0.id, z1.id};
      *(add.repeats) = {s0, s1};
      *(add.strides) = {s1, One};

      auto pad = Pad("pad", add).TQue(Position::kPositionVecOut, 1, 2);
      *(pad.vectorized_axis) = {axes[perm[0]].id, axes[perm[1]].id};
      *(pad.axis) = {axes[perm[0]].id, axes[perm[1]].id};
      *(pad.repeats) = {dims[perm[0]], dims[perm[1]]};
      *(pad.strides) = CalcTransposedStrides(dims, perm);
      auto store = Store("store", pad);
      auto output = Output("output", store);
      SetMemReuseIds({{"load1", 0}, {"load2", 1}, {"pad", 2}, {"add", 3}}, graph);
    }
  }

  auto pad_node = graph.FindNode("pad");
  GE_ASSERT_NOTNULL(pad_node);
  pad_node->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

Status BuildTransposeSplitAscendGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(128);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z1, z2}, 0, FORMAT_ND);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0, z1, z2}, 1, FORMAT_ND);
  *data1.repeats = {s0, s1, s2};
  *data2.repeats = {s0, s1, s2};
  *data1.strides = {s1 * s2, s2, One};
  *data2.strides = {s1 * s2, s2, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto add = Add("add", load1, load2).TBuf(Position::kPositionVecCalc);
      *(add.axis) = {z0T->id, z0t->id, z1.id, z2.id};
      *(add.repeats) = {s0 / z0t->size, z0t->size, s1, s2};
      *(add.strides) = {z0t->size * s1 * s2, s1 * s2, s2, One};
      *(add.vectorized_axis) = {z0t->id, z1.id, z2.id};

      auto transpose = Transpose("transpose", add).TQue(Position::kPositionVecOut, 1, 2);
      *(transpose.axis) = {z0T->id, z0t->id, z2.id, z1.id};
      *(transpose.repeats) = {s0 / z0t->size, z0t->size, s2, s1};
      *(transpose.strides) = {z0t->size * s1 * s2, s1 * s2, s1, One};
      *(transpose.vectorized_axis) = {z0t->id, z2.id, z1.id};
      auto store = Store("store", transpose);
      auto output = Output("output", store);
      SetMemReuseIds({{"load1", 0}, {"load2", 1}, {"transpose", 2}, {"add", 3}}, graph);
    }
  }

  auto transpose_node = graph.FindNode("transpose");
  GE_ASSERT_NOTNULL(transpose_node);
  transpose_node->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

Status BuildTransposeAscendGraph(af::AscGraph &graph, const std::vector<int64_t> &perm = {0, 2, 1}) {
  if (perm.size() != 3 || !IsValidPermutation(perm)) {
    return af::FAILED;
  }
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  std::vector<af::Axis> axes = {z0, z1, z2};
  std::vector<af::Expression> dims = {s0, s1, s2};

  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z1, z2}, 0, FORMAT_ND);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0, z1, z2}, 1, FORMAT_ND);
  *data1.repeats = {s0, s1, s2};
  *data2.repeats = {s0, s1, s2};
  *data1.strides = {s1 * s2, s2, One};
  *data2.strides = {s1 * s2, s2, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto add = Add("add", load1, load2).TBuf(Position::kPositionVecCalc);
      *(add.vectorized_axis) = {z0.id, z1.id, z2.id};
      *(add.axis) = {z0.id, z1.id, z2.id};
      *(add.repeats) = {s0, s1, s2};
      *(add.strides) = {s1 * s2, s2, One};

      auto transpose = Transpose("transpose", add).TQue(Position::kPositionVecOut, 1, 2);
      *(transpose.vectorized_axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id};
      *(transpose.axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id};
      *(transpose.repeats) = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
      *(transpose.strides) = CalcTransposedStrides(dims, perm);
      auto store = Store("store", transpose);
      auto output = Output("output", store);
      SetMemReuseIds({{"load1", 0}, {"load2", 1}, {"transpose", 2}, {"add", 3}}, graph);
    }
  }

  auto transpose_node = graph.FindNode("transpose");
  GE_ASSERT_NOTNULL(transpose_node);
  transpose_node->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

Status Build4DTransposeAscendGraph(af::AscGraph &graph, const std::vector<int64_t> &perm = {0, 1, 2, 3}) {
  if (perm.size() != 4 || !IsValidPermutation(perm)) {
    return af::FAILED;
  }
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  std::vector<af::Axis> axes = {z0, z1, z2, z3};
  std::vector<af::Expression> dims = {s0, s1, s2, s3};

  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data = graph.CreateContiguousData("input", DT_FLOAT, {z0, z1, z2, z3}, 0, FORMAT_ND);
  *data.repeats = {s0, s1, s2, s3};
  *data.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load = Load("load", data).TQue(Position::kPositionVecIn, 1, 2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, z0, z1, z2, z3}, {load}, 1));
      auto transpose = Transpose("transpose", load).TQue(Position::kPositionVecOut, 1, 2);
      *(transpose.vectorized_axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id, axes[perm[3]].id};
      *(transpose.axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id, axes[perm[3]].id};
      *(transpose.repeats) = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
      *(transpose.strides) = CalcTransposedStrides(dims, perm);
      auto store = Store("store", transpose);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, *z0t, z1, z2, z3}, {store}, 1));
      auto output = Output("output", store);
      SetMemReuseIds({{"load", 0}, {"transpose", 1}}, graph);
    }
  }

  auto transpose_node = graph.FindNode("transpose");
  GE_ASSERT_NOTNULL(transpose_node);
  transpose_node->attr.api.unit = ComputeUnit::kUnitVector;
  GE_ASSERT_TRUE(!transpose_node->outputs[0].attr.vectorized_axis.empty());
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  GELOGD("Got input vectorized_axis=[%s], output vectorized_axis=[%s]",
         DebugString(transpose_node->inputs[0].attr.vectorized_axis).c_str(),
         DebugString(transpose_node->outputs[0].attr.vectorized_axis).c_str());
  return af::SUCCESS;
}

Status BuildFlashSoftmaxAscendGraph(af::AscGraph &graph) {
  auto ND = af::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
      auto [softmax_out1, softmax_out2, softmax_out3] = FlashSoftmax("softmax", broadcast, load2, load2);
      auto store1 = Store("store1", softmax_out1);
      auto store2 = Store("store2", softmax_out2);
      auto store3 = Store("store3", softmax_out3);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndbt},
          {load1, load2, broadcast, softmax_out1, softmax_out2, softmax_out3, store1, store2, store3}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
    }
  }
  auto softmax = graph.FindNode("softmax");
  GE_ASSERT_NOTNULL(softmax);
  softmax->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

Status BuildWorkSpaceAscendGraph(af::AscGraph &graph) {
  auto ND = af::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Workspace("workspace1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Workspace("workspace2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
      auto [softmax_out1, softmax_out2, softmax_out3] = FlashSoftmax("softmax", broadcast, load2, load2);
      auto store1 = Workspace("workspace3", softmax_out1);
      auto store2 = Workspace("workspace4", softmax_out2);
      auto store3 = Workspace("workspace5", softmax_out3);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndbt},
          {load1, load2, broadcast, softmax_out1, softmax_out2, softmax_out3, store1, store2, store3}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
    }
  }
  auto softmax = graph.FindNode("softmax");
  GE_ASSERT_NOTNULL(softmax);
  softmax->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}

Status BuildTilingBroadcastAscendGraph(af::AscGraph &graph) {
  auto R = af::Symbol("R");
  auto A = af::Symbol("A");
  auto r = graph.CreateAxis("r", R);
  auto a = graph.CreateAxis("a", A);
  auto [rT, rt] = graph.TileSplit(r.id);
  auto [rTB, rTb] = graph.BlockSplit(rT->id);
  auto [aT, at] = graph.TileSplit(a.id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {r, a}, 0);
  LOOP(*rT) {
    LOOP(*rTB) {
      LOOP(*rTb) {
        LOOP(*aT) {
          auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
          auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
          auto store1 = Store("store1", broadcast);
          GE_ASSERT_SUCCESS(
              GraphConstructUtils::UpdateOutputTensorAxes({*rTB, *rTb, *aT, *rt, *at}, {load1, broadcast, store1}, 2));
          auto output1 = Output("output1", store1);
          (*broadcast.repeats)[2] = af::Symbol(100);  // rTb
          (*broadcast.strides)[2] = af::Symbol(100);
          (*broadcast.repeats)[3] = af::Symbol(1);  // rt
          (*broadcast.strides)[3] = af::Symbol(0);
          (*broadcast.repeats)[4] = af::Symbol(200);  // at
          (*broadcast.strides)[4] = af::Symbol(200);
        }
      }
    }
  }
  auto broadcast1 = graph.FindNode("broadcast");
  GE_ASSERT_NOTNULL(broadcast1);
  broadcast1->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

Status BuildHeavyOpTilingAscendGraph(af::AscGraph &graph) {
  auto R = af::Symbol("R");
  auto A = af::Symbol("A");
  auto r = graph.CreateAxis("r", R);
  auto a = graph.CreateAxis("a", A);
  auto [rT, rt] = graph.TileSplit(r.id);
  auto [rTB, rTb] = graph.BlockSplit(rT->id);
  auto [aT, at] = graph.TileSplit(a.id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {r, a}, 0);
  LOOP(*rT) {
    LOOP(*rTB) {
      LOOP(*rTb) {
        LOOP(*aT) {
          auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
          auto pow1 = Pow("pow", load1, load1).TBuf(Position::kPositionVecOut);
          auto store1 = Store("store1", pow1);
          GE_ASSERT_SUCCESS(
              GraphConstructUtils::UpdateOutputTensorAxes({*rTB, *rTb, *aT, *rt, *at}, {load1, pow1, store1}, 2));
          auto output1 = Output("output1", store1);
          (*pow1.repeats)[2] = af::Symbol(100);  // rTb
          (*pow1.strides)[2] = af::Symbol(100);
          (*pow1.repeats)[3] = af::Symbol(1);  // rt
          (*pow1.strides)[3] = af::Symbol(0);
          (*pow1.repeats)[4] = af::Symbol(200);  // at
          (*pow1.strides)[4] = af::Symbol(200);
        }
      }
    }
  }
  auto pow1 = graph.FindNode("pow");
  GE_ASSERT_NOTNULL(pow1);
  pow1->attr.api.unit = ComputeUnit::kUnitVector;
  pow1->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}

Status BuildMatMulDemoAscendGraph(af::AscGraph &graph) {
  auto ND = af::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
      auto mat_mul_out = Add("mat_mul", broadcast, load2);
      auto store1 = Store("store1", mat_mul_out);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt},
                                                                    {load1, load2, broadcast, mat_mul_out, store1}, 1));
      auto output1 = Output("output1", store1);
    }
  }
  auto mat_mul = graph.FindNode("mat_mul");
  GE_ASSERT_NOTNULL(mat_mul);
  mat_mul->attr.api.unit = ComputeUnit::kUnitVector;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  return af::SUCCESS;
}
}  // namespace cg
}  // namespace ascir
}  // namespace af
class TestGenModelInfo : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    unsetenv("AUTOFUSE_FLAGS");
    unsetenv("AUTOFUSE_DFX_FLAGS");
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    att::AutoFuseConfig::MutablePgoStrategyConfig() = att::PgoStrategyConfig();
  }
  void TearDown() override {}
};

TEST_F(TestGenModelInfo, GetModelMap) {
  af::AscGraph graph("graph");
  FusedParsedScheduleResult all_model_infos;
  std::map<std::string, std::string> options;
  ASSERT_EQ(CreateFaAscGraphWithAllModelInfos("NpuKernel0TilingData", "HighPerf", graph, all_model_infos, options),
            af::SUCCESS);
  EXPECT_EQ(all_model_infos.size(), 1);
  EXPECT_EQ(all_model_infos[0][0].groups_tiling_model_info.size(), 1);
  auto schedule_results = CreateScheduleResultWithSingleAscGraph({2, 1}, graph);
  ASSERT_EQ(schedule_results.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(schedule_results.node_idx_to_scheduled_results[0].size(), 2UL);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_TRUE(GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true));
}

TEST_F(TestGenModelInfo, GetModelMap_set_env_debug) {
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
  af::AscGraph graph("graph");
  FusedParsedScheduleResult all_model_infos;
  std::map<std::string, std::string> options;
  ASSERT_EQ(CreateFaAscGraphWithAllModelInfos("NpuKernel0TilingData", "HighPerf", graph, all_model_infos, options),
            af::SUCCESS);
  EXPECT_EQ(all_model_infos.size(), 1);
  EXPECT_EQ(all_model_infos[0][0].groups_tiling_model_info.size(), 1);
  auto schedule_results = CreateScheduleResultWithSingleAscGraph({2, 1}, graph);
  ASSERT_EQ(schedule_results.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(schedule_results.node_idx_to_scheduled_results[0].size(), 2UL);
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_TRUE(GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true));
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find(R"( GELOGD("[%s]" fmt, name, ##__VA_ARGS__))"), std::string::npos);
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
}

TEST_F(TestGenModelInfo, GenTilingImplAutoFuseV3EmptyTilingDataName) {
  std::string graph_name("graph");
  af::AscGraph graph(graph_name.c_str());
  FusedParsedScheduleResult all_model_infos;
  std::map<std::string, std::string> options;
  options[kTilingDataTypeName] = graph_name + "TilingData";
  ASSERT_EQ(CreateFaAscGraphWithAllModelInfos("", "HighPerf", graph, all_model_infos, options), af::SUCCESS);
  EXPECT_EQ(all_model_infos.size(), 1);
  EXPECT_EQ(all_model_infos[0][0].groups_tiling_model_info.size(), 1);
  auto schedule_results = CreateScheduleResultWithSingleAscGraph({2, 1}, graph);
  ASSERT_EQ(schedule_results.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(schedule_results.node_idx_to_scheduled_results[0].size(), 2UL);
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_TRUE(GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true));
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find(R"(graphTilingData)"), std::string::npos);
}

TEST_F(TestGenModelInfo, GenTilingImplAutoFuseV3WithScoreFunc) {
  std::string graph_name("graph");
  af::AscGraph graph(graph_name.c_str());
  FusedParsedScheduleResult all_model_infos;
  std::map<std::string, std::string> options;
  options[kTilingDataTypeName] = graph_name + "TilingData";
  ASSERT_EQ(CreateFaAscGraphWithAllModelInfos("", "HighPerf", graph, all_model_infos, options), af::SUCCESS);
  EXPECT_EQ(all_model_infos.size(), 1);
  EXPECT_EQ(all_model_infos[0][0].groups_tiling_model_info.size(), 1);
  auto schedule_results = CreateScheduleResultWithSingleAscGraph({2, 1}, graph);
  ASSERT_EQ(schedule_results.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(schedule_results.node_idx_to_scheduled_results[0].size(), 2UL);
  schedule_results.node_idx_to_scheduled_results[0][0].score_func =
      "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_TRUE(GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true));
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find(R"(graphTilingData)"), std::string::npos);
}

TEST_F(TestGenModelInfo, GenTilingImplAutoFuseV3WithScoreFuncByEnv) {
  setenv("AUTOFUSE_DFX_FLAGS",
         "--autofuse_att_algorithm=HighPerf;--att_accuracy_level=0;--att_ub_threshold=20;--force_template_op_name="
         "Concat;--force_tiling_case=0;--force_schedule_result=0",
         1);
  af::AscGraph graph("graph");
  FusedParsedScheduleResult all_model_infos;
  std::map<std::string, std::string> options;
  options[kTilingDataTypeName] = "graphTilingData";
  ASSERT_EQ(CreateFaAscGraphWithAllModelInfos("", "HighPerf", graph, all_model_infos, options), af::SUCCESS);
  EXPECT_EQ(all_model_infos.size(), 1);
  EXPECT_EQ(all_model_infos[0][0].groups_tiling_model_info.size(), 1);
  auto schedule_results = CreateScheduleResultWithSingleAscGraph({2, 1}, graph);
  ASSERT_EQ(schedule_results.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(schedule_results.node_idx_to_scheduled_results[0].size(), 2UL);
  schedule_results.node_idx_to_scheduled_results[0][0].score_func =
      "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_TRUE(GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true));
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find(R"(graphTilingData)"), std::string::npos);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenModelInfo, GenTilingImplAutoFuseV3WithPGOByEnv) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  setenv("AUTOFUSE_DFX_FLAGS",
         "--autofuse_att_algorithm=HighPerf;--att_accuracy_level=0;--att_ub_threshold=20;--force_template_op_name="
         "Concat;--force_tiling_case=0;--force_schedule_result=0;--autofuse_pgo_algo=algo_invalid;--autofuse_pgo_step_"
         "max=32",
         1);
  af::AscGraph graph("graph");
  FusedParsedScheduleResult all_model_infos;
  std::map<std::string, std::string> options;
  options[kTilingDataTypeName] = "graphTilingData";
  ASSERT_EQ(CreateFaAscGraphWithAllModelInfos("", "HighPerf", graph, all_model_infos, options), af::SUCCESS);
  EXPECT_EQ(all_model_infos.size(), 1);
  EXPECT_EQ(all_model_infos[0][0].groups_tiling_model_info.size(), 1);
  auto schedule_results = CreateScheduleResultWithSingleAscGraph({2, 1}, graph);
  ASSERT_EQ(schedule_results.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(schedule_results.node_idx_to_scheduled_results[0].size(), 2UL);
  schedule_results.node_idx_to_scheduled_results[0][0].score_func =
      "int32_t CalcScore(graph_normalTilingData &tiling_data) { return 1;}";
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_TRUE(GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true));
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo, "true");
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().autofuse_pgo_algo_select, "core_select");
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().autofuse_pgo_algo_step_max, 32);
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find(R"(graphTilingData)"), std::string::npos);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenModelInfo, ModelInfoParser) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  att::FaBeforeAutoFuse(graph);
  att::FaAfterScheduler(graph);
  att::FaAfterQueBufAlloc(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  graphs.emplace_back(graph);
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list), af::SUCCESS);
  EXPECT_EQ(MakeJson(model_info_list, json_info), af::SUCCESS);
}

TEST_F(TestGenModelInfo, RefreshCommonUbExprKeepsNonContainerVariables) {
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::Build2DTransposeAscendGraph(graph, {1, 0}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  ASSERT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);

  ASSERT_EQ(model_info_list.size(), 1U);
  EXPECT_TRUE(HasVariableName(model_info_list[0], "transpose_output_0"));
  EXPECT_TRUE(HasVariableName(model_info_list[0], "add_output_0"));
  EXPECT_NE(model_info_list[0].container_exprs.find("q0_size"), model_info_list[0].container_exprs.end());
}

TEST_F(TestGenModelInfo, ModelInfoParserForTranspose10ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::Build2DTransposeAscendGraph(graph, {1, 0}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserForV2TransposeWithoutApiTilingTypeName) {
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::BuildTransposeAscendGraph(graph, {1, 0, 2}), af::SUCCESS);
  graphs.emplace_back(graph);
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(model_info_list.size(), 1);
  EXPECT_TRUE(model_info_list[0].node_name_to_api_code.empty());
}

TEST_F(TestGenModelInfo, Pad) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::Build2DPadAscendGraph(graph, {1, 0}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["pad"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(got_api_code.function_impl.empty());  // GetPadTilingDefine为空
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserForTranspose102ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::BuildTransposeAscendGraph(graph, {1, 0, 2}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserForTranspose021ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::BuildTransposeAscendGraph(graph, {0, 2, 1}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserForTranspose021SplitApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::BuildTransposeSplitAscendGraph(graph), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserForTranspose210ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::BuildTransposeAscendGraph(graph, {2, 1, 0}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserForTranspose201ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::BuildTransposeAscendGraph(graph, {2, 0, 1}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserFor4DTranspose0213ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::Build4DTransposeAscendGraph(graph, {0, 2, 1, 3}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserFor4DTranspose2103ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::Build4DTransposeAscendGraph(graph, {2, 1, 0, 3}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserFor4DTranspose0321ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::Build4DTransposeAscendGraph(graph, {0, 3, 2, 1}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::SUCCESS);
  ASSERT_EQ(model_info_list.size(), 1);
  ASSERT_EQ(model_info_list[0].node_name_to_api_code.size(), 1);
  auto &got_api_code = model_info_list[0].node_name_to_api_code["transpose"];
  EXPECT_TRUE(!got_api_code.function_invoke.empty());
  EXPECT_TRUE(!got_api_code.function_impl.empty());
  EXPECT_TRUE(!got_api_code.head_files.empty());
}

TEST_F(TestGenModelInfo, ModelInfoParserFor4DTranspose0123ApiTiling) {
  std::string json_info;
  std::vector<af::AscGraph> graphs;
  TilingModelInfo model_info_list;
  af::AscGraph graph("graph");
  ASSERT_EQ(af::ascir::cg::Build4DTransposeAscendGraph(graph, {0, 1, 2, 3}), af::SUCCESS);
  graphs.emplace_back(graph);
  const auto &tiling_data_name = graph.GetName() + "TilingData";
  EXPECT_EQ(GenerateModelInfo(graphs, model_info_list, {{kTilingDataTypeName, tiling_data_name}}), af::PARAM_INVALID);
}

TEST_F(TestGenModelInfo, GetModelMap_set_env_small_shape) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=1;--att_enable_small_shape_strategy=true", 1);
  af::AscGraph graph("graph");
  FusedParsedScheduleResult all_model_infos;
  std::map<std::string, std::string> options;
  ASSERT_EQ(CreateFaAscGraphWithAllModelInfos("NpuKernel0TilingData", "HighPerf", graph, all_model_infos, options),
            af::SUCCESS);
  EXPECT_EQ(all_model_infos.size(), 1);
  EXPECT_EQ(all_model_infos[0][0].groups_tiling_model_info.size(), 1);
  auto schedule_results = CreateScheduleResultWithSingleAscGraph({2, 1}, graph);
  ASSERT_EQ(schedule_results.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(schedule_results.node_idx_to_scheduled_results[0].size(), 2UL);
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_TRUE(GenTilingImplAutoFuseV3("AddLayerNorm", schedule_results, options, tiling_funcs, true));
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("TrySmallShapeTiling"), std::string::npos);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenModelInfo, gen_softmax_api_tiling_success) {
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  setenv("AUTOFUSE_DFX_FLAGS", "--att_enable_small_shape_strategy=true", 1);
  std::vector<af::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);

    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("flash_softmax_tiling_func.cpp", std::ios::out);
  oss << "#include \"FlashSoftmax_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_NE(tiling_func.find("TrySmallShapeTiling"), std::string::npos);
  unsetenv("AUTOFUSE_DFX_FLAGS");
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
}

TEST_F(TestGenModelInfo, gen_schedule_group_cache_success) {
  // 设置环境变量启用 OperatorCache
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
  att::AutoFuseConfig::MutablePgoStrategyConfig() = att::PgoStrategyConfig();

  std::vector<af::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);
  // TTODO 当前仅检查是否有生成使能cache后的字符串，后续需要增加端到端验证用例
  // 更新：新API使用SaveOperatorCache替代SaveCache
  EXPECT_NE(tiling_func.find("SaveOperatorCache"), std::string::npos);
  std::ofstream oss;
  oss.open("cache_tiling_func.cpp", std::ios::out);
  oss << "#include \"FlashSoftmax_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();

  // 清理环境变量
  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
}

TEST_F(TestGenModelInfo, gen_workspace_with_tensor_id) {
  std::vector<af::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildWorkSpaceAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  // 选中新候选时先清空本子图全部 workspace 占用，再直接写入当前候选的 size（无复用场景不做 max）
  EXPECT_NE(tiling_func.find("tiling_data.set_workspace0(0U);"), std::string::npos);
  EXPECT_NE(tiling_func.find("tiling_data.set_workspace0(it0->second);"), std::string::npos);
  EXPECT_EQ(tiling_func.find("std::max(tiling_data.get_workspace0()"), std::string::npos);
}

TEST_F(TestGenModelInfo, gen_schedule_group_reduce_tile_r) {
  std::vector<af::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildTilingBroadcastAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::string tiling_func;
  std::ofstream oss;
  oss.open("flash_softmax_tiling_func.cpp", std::ios::out);
  oss << "#include \"FlashSoftmax_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_schedule_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode("FlashSoftmax", all_model_infos, generator_config, tiling_res), af::SUCCESS);
  oss.open("FlashSoftmax_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("tilingCaseImplPtr = &caseR1101"), std::string::npos);
  EXPECT_TRUE(tiling_func.find("solver.Run(runtime_enable_multicore_ub_tradeoff, ") != std::string::npos);
}

TEST_F(TestGenModelInfo, gen_tiling_with_heavy_op) {
  std::vector<af::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildHeavyOpTilingAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::string tiling_func;
  std::ofstream oss;
  oss.open("flash_softmax_tiling_func.cpp", std::ios::out);
  oss << "#include \"FlashSoftmax_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_schedule_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode("FlashSoftmax", all_model_infos, generator_config, tiling_res), af::SUCCESS);
  oss.open("FlashSoftmax_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_TRUE(tiling_func.find("solver.Run(runtime_enable_multicore_ub_tradeoff, ") != std::string::npos);
}

TEST_F(TestGenModelInfo, gen_schedule_group_reduce_tile_r_force_tiling_r) {
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_att_algorithm=AxesReorder;--force_tiling_case=0_R;--force_schedule_result=0",
         1);
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildTilingBroadcastAscendGraph(graph_normal), af::SUCCESS);
  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::string tiling_func;
  std::ofstream oss;
  oss.open("flash_softmax_tiling_func.cpp", std::ios::out);
  oss << "#include \"FlashSoftmax_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_schedule_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(att::AutoFuseConfig::MutableAttStrategyConfig().Init(), af::SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::GetAttStrategyConfig().set_force_tiling_case, true);
  EXPECT_EQ(ge::AttStrategyConfigUtils::ParseForceTilingCase(
                att::AutoFuseConfig::GetAttStrategyConfig().force_tiling_case, generator_config.force_tiling_case),
            af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode("FlashSoftmax", all_model_infos, generator_config, tiling_res), af::SUCCESS);
  oss.open("FlashSoftmax_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("tilingCaseImplPtr = &caseR1101"), std::string::npos);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenModelInfo, gen_schedule_group_with_var_relation) {
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  setenv("experimental_autofusion_att_enable_small_shape_strategy", "1", 1);
  std::vector<af::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    Expr src_var = af::Symbol("ND");
    scheduled_result.var_relations = {{1, {{0, {{"ND", src_var}}}}}};
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(
      tiling_func.find(
          "graph0_result0_g1_ND_var_relation_value = static_cast<double>(graph0_result0_g0_tiling_data.get_ND())"),
      std::string::npos);
  EXPECT_NE(tiling_func.find("graph0_result0_g1_tiling_data.set_ND(static_cast<uint32_t>("
                             "graph0_result0_g1_ND_var_relation_value))"),
            std::string::npos);
}

// ============================================================================
// ATT Tiling缓存功能单元测试
// ============================================================================

/**
 * @brief 测试用例1：算子级缓存基础功能测试
 *
 * 测试项：ATT算子级缓存代码生成
 * 重要级别：高 (P0)
 *
 * 预置条件：
 * - 编译环境正常
 * - FlashSoftmax图构建正常
 * - AxesReorder求解器可用
 *
 * 操作步骤：
 * 1. 创建FlashSoftmax图结构（2个schedule_group）
 * 2. 调用GenTilingImplAutoFuseV3生成Tiling代码
 * 3. 合并Tiling函数
 * 4. 验证生成代码中的缓存相关内容
 *
 * 输入：
 * - TilingKey: 1101u
 * - 图名: FlashSoftmax
 * - 求解器类型: AxesReorder
 * - 2个schedule_group
 *
 * 预期结果：
 * 生成的代码包含：
 * - `using OperatorLevelCache`
 * - `FixedSizeHashMap<kInputShapeSize, kOperatorCacheCapacity`
 * - `bool FindOperatorCache`
 * - `OperatorCacheSaveResult SaveOperatorCache`
 *
 * 备注：验证缓存类型定义和函数声明正确生成
 */
TEST_F(TestGenModelInfo, op_level_cache_basic) {
  // 设置环境变量启用 OperatorCache
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();

  std::vector<af::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);

  // 验证缓存类型定义
  EXPECT_NE(tiling_func.find("using OperatorLevelCache"), std::string::npos);
  EXPECT_NE(tiling_func.find("FixedSizeHashMap<kInputShapeSize, kOperatorCacheCapacity"), std::string::npos);

  // 验证缓存函数定义（在TilingCacheContext类中）
  // 使用通用的类型匹配，不依赖具体的TilingData类型名称
  EXPECT_NE(tiling_func.find("* FindOperatorCache(const OperatorCacheKey&"), std::string::npos);
  EXPECT_NE(tiling_func.find("OperatorCacheSaveResult SaveOperatorCache(const OperatorCacheKey&"), std::string::npos);

  // 注意：缓存查询代码(input_shapes数组构建)只在有缓存复用信息时生成
  // 这是当前设计的限制，算子级缓存类型和函数已正确生成

  // 清理环境变量
  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
}

/**
 * @brief 两级缓存同时生成测试
 *
 * 验证生成的代码同时包含：
 * - OperatorLevelCache (第一级，thread_local)
 * - GroupLevelCache (第二级，栈上)
 *
 * 备注：验证两级缓存类型和函数正确生成
 */
TEST_F(TestGenModelInfo, two_level_cache_generation) {
  // 设置环境变量启用 OperatorCache
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();

  std::vector<af::AscGraph> graphs;
  af::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(af::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);

  std::string tiling_func;
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  CombineTilings(tiling_funcs, tiling_func);

  // 验证两级缓存类型定义
  EXPECT_NE(tiling_func.find("using OperatorLevelCache"), std::string::npos);
  EXPECT_NE(tiling_func.find("using GroupLevelCache"), std::string::npos);

  // 验证两级缓存函数（在TilingCacheContext类中）
  // 使用通用的类型匹配，不依赖具体的TilingData类型名称
  EXPECT_NE(tiling_func.find("* FindOperatorCache(const OperatorCacheKey&"), std::string::npos);
  EXPECT_NE(tiling_func.find("OperatorCacheSaveResult SaveOperatorCache(const OperatorCacheKey&"), std::string::npos);

  // 验证TilingCacheContext类生成（使用unique_ptr避免栈溢出）
  EXPECT_NE(tiling_func.find("class TilingCacheContext"), std::string::npos);
  EXPECT_NE(tiling_func.find("thread_local std::unique_ptr<OperatorLevelCache<TilingData>> operator_cache_"),
            std::string::npos);

  // 清理环境变量
  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
}
