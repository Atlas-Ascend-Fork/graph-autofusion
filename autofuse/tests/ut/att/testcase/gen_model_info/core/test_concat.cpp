/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "gen_tiling_impl.h"
#include "base/att_const_values.h"
#include "graph_construct_utils.h"
#include "test_common_utils.h"
#include "common_gen_utils.h"

using namespace af::ascir_op;
namespace {
void InitDataNode(Data &node, const std::vector<int64_t> &axis, af::DataType dtype,
                  const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void InitLoadNode(Load &node, const std::vector<int64_t> &axis, af::DataType dtype,
                  const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void InitStoreNode(Store &node, const std::vector<int64_t> &axis, af::DataType dtype,
                   const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void InitOutputNode(Output &node, const std::vector<int64_t> &axis, af::DataType dtype,
                    const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}
}  // namespace
namespace ascir {
constexpr int64_t ID_NONE = -1;  // 取多少？
using namespace af;
using HintGraph = AscGraph;
}  // namespace ascir
using namespace att;
using att::test::CombineTilings;

class TestGenConcat : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
  }
};
void InitConcatAxes(Concat &node, const std::vector<int64_t> &axis, af::DataType dtype,
                    const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.attr.sched.axis = axis;
  node.y.dtype = dtype;
  *node.y.axis = axis;
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void Concat_Normal_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = af::sym::kSymbolOne, ZERO = af::sym::kSymbolZero;
  auto A = af::Symbol("A"), R = af::Symbol("R"), BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A), r = graph.CreateAxis("R", R), bl = graph.CreateAxis("BL", BL);
  auto axis = std::vector<int64_t>{a.id, r.id, bl.id};
  auto rep_ar = std::vector<af::Expression>{A, R, ONE};
  auto str_ar = std::vector<af::Expression>{R, ONE, ZERO};
  auto rep_a = std::vector<af::Expression>{A, ONE, ONE};
  auto str_a = std::vector<af::Expression>{ONE, ZERO, ZERO};

  Data x1("x1", graph);
  InitDataNode(x1, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Load x1L("x1Local");
  x1L.x = x1.y;
  InitLoadNode(x1L, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Data x2("x2", graph);
  InitDataNode(x2, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Load x2L("x2Local");
  x2L.x = x2.y;
  InitLoadNode(x2L, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Data bias("bias", graph);
  InitDataNode(bias, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Load biasL("biasLocal");
  biasL.x = bias.y;
  InitLoadNode(biasL, axis, af::DT_FLOAT16, rep_ar, str_ar);

  Concat mean("mean");
  mean.x = {x1L.y, x2L.y, biasL.y};
  InitConcatAxes(mean, axis, af::DT_FLOAT, rep_ar, str_ar);
  Store x_out("x_out");
  x_out.x = mean.y;
  InitStoreNode(x_out, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Store mean_out("mean_out");
  mean_out.x = mean.y;
  InitStoreNode(mean_out, axis, af::DT_FLOAT, rep_a, str_a);

  Data one("one", graph);
  InitDataNode(one, axis, af::DT_FLOAT, {ONE, ONE, BL}, {ZERO, ZERO, ONE});
  Concat rstd("rstd");
  rstd.x = {mean.y, mean.y, one.y};
  InitConcatAxes(rstd, axis, af::DT_FLOAT, rep_ar, str_ar);
  Store rstd_out("rstd_out");
  rstd_out.x = rstd.y;
  InitStoreNode(rstd_out, axis, af::DT_FLOAT, rep_a, str_a);

  auto rep_r = std::vector<af::Expression>{ONE, R, ONE};
  auto str_r = std::vector<af::Expression>{ZERO, ONE, ZERO};
  Data beta("beta", graph);
  InitDataNode(beta, axis, af::DT_FLOAT16, rep_r, str_r);
  Load betaL("betaLocal");
  betaL.x = beta.y;
  InitLoadNode(betaL, axis, af::DT_FLOAT16, rep_r, str_r);
  Data gamma("gamma", graph);
  InitDataNode(gamma, axis, af::DT_FLOAT16, rep_r, str_r);
  Load gammaL("gammaLocal");
  gammaL.x = gamma.y;
  InitLoadNode(gammaL, axis, af::DT_FLOAT16, rep_r, str_r);

  Concat y("y");
  y.attr.api.unit = af::ComputeUnit::kUnitVector;
  y.attr.sched.axis = axis;
  y.x = {rstd.y, betaL.y, gammaL.y, rstd.y};
  y.y.dtype = af::DT_FLOAT16;
  *y.y.axis = axis, *y.y.repeats = rep_ar, *y.y.strides = str_ar;

  Concat concat("concat");
  concat.x = {x1L.y, x2L.y};
  concat.attr.sched.axis = axis;
  *concat.y.axis = axis, *concat.y.repeats = rep_ar, *concat.y.strides = str_ar;
  Store cat_out("cat_out");
  cat_out.x = y.y;
  InitStoreNode(cat_out, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Store y_out("y_out");
  y_out.x = y.y;
  InitStoreNode(y_out, axis, af::DT_FLOAT16, rep_ar, str_ar);

  Output buf1("buf1");
  buf1.x = x_out.y;
  InitOutputNode(buf1, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Output buf2("buf2");
  buf2.x = mean_out.y;
  InitOutputNode(buf2, axis, af::DT_FLOAT, rep_a, str_a);
  Output buf3("buf3");
  buf3.x = rstd_out.y;
  InitOutputNode(buf3, axis, af::DT_FLOAT, rep_a, str_a);
  Output buf("buf");
  buf.x = y_out.y;
  InitOutputNode(buf, axis, af::DT_FLOAT16, rep_ar, str_ar);
  Output buf4("buf4");
  buf4.x = cat_out.y;
  InitOutputNode(buf4, axis, af::DT_FLOAT16, rep_ar, str_ar);
}

/*
for aBO
  for aBIO
    for aBII
      for r
        load x1
        load x2
        load bias
        CalcMean
        CalcRstd
        Store X
        Store mean
        Load beta
        Load gamma
        CalcRstd
        Store rstd
        CalcY
        Store y
*/

void ApplySplitToNode(ascir::HintGraph &graph, const char *name, int64_t aBO, int64_t aBI, int64_t aBIO, int64_t aBII,
                      std::vector<int64_t> vec_axis) {
  auto node = graph.FindNode(name);
  graph.ApplySplit(node, aBO, aBI);
  graph.ApplySplit(node, aBIO, aBII);
  node->attr.sched.loop_axis = aBIO;
  node->outputs[0].attr.vectorized_axis = vec_axis;
}

void Concat_Normal_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;
  auto bl = graph.FindAxis(2)->id;
  auto [aBO, aBI] = graph.BlockSplit(a, "nbi", "nbo");
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");
  auto vec_ar = std::vector<int64_t>{aBII->id, r};
  auto vec_r = std::vector<int64_t>{r};

  const char *ar_nodes[] = {"x1",    "x2",       "bias", "x1Local",  "x2Local", "biasLocal", "mean",
                            "x_out", "mean_out", "rstd", "rstd_out", "y",       "cat_out"};
  for (auto name : ar_nodes) {
    ApplySplitToNode(graph, name, aBO->id, aBI->id, aBIO->id, aBII->id, vec_ar);
  }
  graph.FindNode("mean")->attr.api.unit = af::ComputeUnit::kUnitVector;
  graph.FindNode("rstd")->attr.api.unit = af::ComputeUnit::kUnitVector;
  graph.FindNode("rstd_out")->attr.api.unit = af::ComputeUnit::kUnitVector;

  const char *r_nodes[] = {"betaLocal", "gammaLocal"};
  for (auto name : r_nodes) {
    ApplySplitToNode(graph, name, aBO->id, aBI->id, aBIO->id, aBII->id, vec_r);
  }

  ApplySplitToNode(graph, "concat", aBO->id, aBI->id, aBIO->id, aBII->id, vec_ar);
  graph.FindNode("concat")->attr.api.unit = af::ComputeUnit::kUnitVector;
}

void SetGmOutput(ascir::HintGraph &graph, const char *name) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  node->outputs[0].attr.mem.position = af::Position::kPositionGM;
}

void SetQueueOutput(ascir::HintGraph &graph, const char *name, int &tensor_id, int que_id, af::Position position) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.tensor_id = tensor_id++;
  node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  node->outputs[0].attr.mem.position = position;
  node->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  node->outputs[0].attr.buf.id = ascir::ID_NONE;
  node->outputs[0].attr.que.id = que_id;
  node->outputs[0].attr.que.depth = 1;
  node->outputs[0].attr.que.buf_num = 1;
  node->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;
}

void Concat_Normal_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensor_id = 0;
  int que_id = 0;
  int buf_id = 0;
  int x1Que = que_id++;
  int x2Que = que_id++;
  int biasQue = que_id++;
  int gammaQue = que_id++;
  int betaQue = que_id++;
  int meanQue = que_id++;
  int rstdQue = que_id++;
  int yQue = que_id++;
  int xQue = que_id++;
  int x32Queue = que_id++;
  int oneTBuf = buf_id++;

  SetGmOutput(graph, "x1");
  SetGmOutput(graph, "x2");
  SetGmOutput(graph, "bias");

  SetQueueOutput(graph, "x1Local", tensor_id, x1Que, af::Position::kPositionVecIn);
  SetQueueOutput(graph, "x2Local", tensor_id, x2Que, af::Position::kPositionVecIn);
  SetQueueOutput(graph, "biasLocal", tensor_id, biasQue, af::Position::kPositionVecIn);
  SetQueueOutput(graph, "mean", tensor_id, meanQue, af::Position::kPositionVecOut);

  SetGmOutput(graph, "x_out");
  SetGmOutput(graph, "mean_out");

  auto one = graph.FindNode("one");
  one->outputs[0].attr.mem.tensor_id = tensor_id++;
  one->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  one->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  one->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  one->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  one->outputs[0].attr.buf.id = oneTBuf;
  one->outputs[0].attr.que.id = ascir::ID_NONE;
  one->outputs[0].attr.que.depth = ascir::ID_NONE;
  one->outputs[0].attr.que.buf_num = ascir::ID_NONE;
  one->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;

  SetQueueOutput(graph, "rstd", tensor_id, yQue, af::Position::kPositionVecOut);

  SetGmOutput(graph, "rstd_out");
  SetGmOutput(graph, "beta");

  SetQueueOutput(graph, "betaLocal", tensor_id, betaQue, af::Position::kPositionVecIn);

  SetGmOutput(graph, "gamma");

  SetQueueOutput(graph, "gammaLocal", tensor_id, gammaQue, af::Position::kPositionVecIn);
  SetQueueOutput(graph, "y", tensor_id, yQue, af::Position::kPositionVecOut);

  SetGmOutput(graph, "y_out");
  SetQueueOutput(graph, "concat", tensor_id, yQue, af::Position::kPositionVecOut);

  SetGmOutput(graph, "cat_out");
}

TEST_F(TestGenConcat, case_axes_reorder) {
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["dump_debug_info"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
}

TEST_F(TestGenConcat, case_axes_reorder_by_env) {
  setenv("AUTOFUSE_DFX_FLAGS",
         "--autofuse_att_algorithm=AxesReorder;--att_enable_small_shape_strategy=true;--att_enable_multicore_ub_"
         "tradeoff=true;--att_corenum_threshold=70;--att_ub_threshold=30",
         1);
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["dump_debug_info"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestGenConcat, case_axes_reorder_replace) {
  std::vector<ascir::AscGraph> graphs;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;

  // dtype 固定fp16, 1000
  // 固定需要输出x-out  100
  // 分三个模板 normal(0)  slice(1) welford(5)
  // 固定带bias，并且不需要broadcast 1

  // 1101
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  Concat_Normal_BeforeAutofuse(graph_normal);
  Concat_Normal_AfterScheduler(graph_normal);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph_normal);
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);
  graphs.emplace_back(graph_normal);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["dump_debug_info"] = "./";
  options["gen_extra_info"] = "1";
  options["duration_level"] = "1";
  options["do_variable_replace"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
}
namespace af {
namespace ascir {
namespace cg {
Status BuildTqueTbufAscendGraph_single_case(af::AscGraph &graph, bool reuse_temp_buffer) {
  auto A = af::Symbol(10, "A");
  auto R = af::Symbol(20, "R");
  auto BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = af::Symbol(10, "ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TBuf(Position::kPositionVecOut);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt}, {load1, load2, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  auto data = graph.FindNode("load1");
  data->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, reuse_temp_buffer ? 1 : 0});
  return af::SUCCESS;
}

Status BuildTqueTbufAscendGraph_multi_case_g0(af::AscGraph &graph) {
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto ND = af::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd}, 2);
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {nd}, 3);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load2", data2).TBuf(Position::kPositionVecIn);
      auto load_tbuf1 = Load("load3", data3).TBuf(Position::kPositionVecIn);
      auto load_tbuf2 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tbuf0);
      auto store3 = Store("store2", load_tbuf1);
      auto store4 = Store("store2", load_tbuf2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndb, *ndbt}, {load_tque0, load_tbuf0, load_tbuf1, load_tbuf2, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  auto data1_node = graph.FindNode("load2");
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{af::Symbol(1024), 0}, {}, 0});
  data1_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{af::Symbol(2 * 1024), 0}, {}, 0});
  return af::SUCCESS;
}

Status BuildTqueTbufAscendGraph_multi_case_g1(af::AscGraph &graph) {
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = af::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0B, z0b] = graph.BlockSplit(z0.id);
  auto [z0bT, z0bt] = graph.TileSplit(z0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0}, 1);
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {z0}, 2);
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {z0}, 3);
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque1 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque2 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tque1);
      auto store3 = Store("store2", load_tque2);
      auto store4 = Store("store2", load_tbuf0);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*z0B, *z0bT, *z0b, *z0bt}, {load_tque0, load_tque1, load_tque2, load_tbuf0, store1, store2, store3, store4},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
      auto output4 = Output("output2", store4);
    }
  }
  auto data_node = graph.FindNode("load1");
  data_node->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{af::Symbol(16 * 1024), 0}, {}, 0});
  return af::SUCCESS;
}
}  // namespace cg
}  // namespace ascir
}  // namespace af
bool IsFileContainsString(const std::string &filename, const std::string &searchString) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find(searchString) != std::string::npos) {
      file.close();
      return true;
    }
  }
  file.close();
  return false;
}

std::string GenerateConcatTilingData(const ascir::FusedScheduledResult &fused_result,
                                     const std::map<std::string, std::string> &options, const std::string &graph_name) {
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options.at(af::sym::kTilingDataTypeName);
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = false;
  EXPECT_EQ(generator.GenTilingCode("Concat", all_model_infos, generator_config, tiling_res), af::SUCCESS);
  return tiling_res.at(graph_name + "TilingData");
}

void TestGenConcat_tque_tbuf_test(bool reuse_temp_buffer) {
  ascir::ScheduleGroup schedule_group1;
  std::vector<ascir::ScheduledResult> schedule_results;

  const std::string kFirstGraphName = "tque_tbuf_case0";
  ascir::AscGraph tque_tbuf_case0(kFirstGraphName.c_str());
  ASSERT_EQ(af::ascir::cg::BuildTqueTbufAscendGraph_single_case(tque_tbuf_case0, reuse_temp_buffer), af::SUCCESS);
  tque_tbuf_case0.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(tque_tbuf_case0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  std::map<std::string, std::string> tiling_funcs;
  auto res = GenTilingImplAutoFuseV3("Concat", fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  ASSERT_TRUE(res);
  EXPECT_FALSE(tiling_func.empty());
  const auto tiling_data = GenerateConcatTilingData(fused_scheduled_result, options, kFirstGraphName);
  EXPECT_NE(tiling_data.find("q0_size"), std::string::npos);
  EXPECT_NE(tiling_data.find("b1_size"), std::string::npos);
  if (!reuse_temp_buffer) {
    EXPECT_NE(tiling_data.find("b0_size"), std::string::npos);
  }
}

TEST_F(TestGenConcat, tque_tbuf_case0) {
  TestGenConcat_tque_tbuf_test(false);
}

TEST_F(TestGenConcat, tque_tbuf_reuse_temp_buffer) {
  TestGenConcat_tque_tbuf_test(true);
}

TEST_F(TestGenConcat, tque_tbuf_case1) {
  const std::string kFirstGraphName = "case0";
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1("case1");
  ASSERT_EQ(af::ascir::cg::BuildTqueTbufAscendGraph_multi_case_g0(graph_0), af::SUCCESS);
  graph_0.SetTilingKey(0U);
  ASSERT_EQ(af::ascir::cg::BuildTqueTbufAscendGraph_multi_case_g1(graph_1), af::SUCCESS);
  graph_1.SetTilingKey(1U);
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  std::map<std::string, std::string> tiling_funcs;
  auto res = GenTilingImplAutoFuseV3("Concat", fused_scheduled_result, options, tiling_funcs, true);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  ASSERT_TRUE(res);
  EXPECT_NE(tiling_func.find("AscGraph0ScheduleResult0G0"), std::string::npos);
  EXPECT_NE(tiling_func.find("AscGraph0ScheduleResult1G0"), std::string::npos);
  const auto tiling_data = GenerateConcatTilingData(fused_scheduled_result, options, kFirstGraphName);
  EXPECT_NE(tiling_data.find("q0_size"), std::string::npos);
  EXPECT_NE(tiling_data.find("q1_size"), std::string::npos);
  EXPECT_NE(tiling_data.find("q2_size"), std::string::npos);
}
