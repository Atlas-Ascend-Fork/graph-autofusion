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
#include "base/base_types.h"
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "tiling_code_generator.h"
#include "graph_construct_utils.h"
#include "result_checker_utils.h"
#include "gen_tiling_impl.h"
#include "graph/utils/graph_utils.h"
#include "autofuse_config/auto_fuse_config.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"
using namespace af::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1;
using namespace ge;
using HintGraph = AscGraph;
}  // namespace ascir
// 公共函数声明：生成tiling代码并写入文件
std::map<std::string, std::string> GenerateTilingCodeCommon(const std::string &op_name,
                                                            const ascir::FusedScheduledResult &fused_scheduled_result,
                                                            const std::map<std::string, std::string> &options,
                                                            const std::string &tiling_func,
                                                            const std::string &header_file, bool gen_extra_infos);

namespace {

std::string kRunTilingFuncMain = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";

std::string kConcatTilingTestHead = R"(
#include "Concat_tiling_api.h"
)";

std::string kRunTilingFuncMainSameND = R"(
#include <iostream>
#include <unordered_map>
#include "Concat_tiling_api.h"
#include "Concat_tiling_test.h"

using namespace optiling;

int main() {
  // ========== Group 级别缓存测试 ==========
  std::cout << "\n========== Group-level cache test ==========" << std::endl;
  AscGraph0ScheduleResult0G0::GroupLevelCache group_cache;
  // Test 1: ND = 1024，首次执行（不命中）
  std::cout << "\n--- Test 1: ND = 1024 (first execution) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData1;
  tilingData1.set_ND(1024);
  tilingData1.set_block_dim(64);
  tilingData1.set_ub_size(245760);
  std::unordered_map<int64_t, uint64_t> workspace_map1;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData1, workspace_map1, -1, &group_cache)) {
    std::cout << "Test 1 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData1.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData1.get_ND() << std::endl;
  // Test 2: ND = 1024，再次执行（应命中 Group 缓存）
  std::cout << "\n--- Test 2: ND = 1024 (should hit Group cache) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData2;
  tilingData2.set_ND(1024);
  tilingData2.set_block_dim(64);
  tilingData2.set_ub_size(245760);

  std::unordered_map<int64_t, uint64_t> workspace_map2;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData2, workspace_map2, -1, &group_cache)) {
    std::cout << "Test 2 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData2.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData2.get_ND() << std::endl;
  // Test 3: ND = 2048，首次执行（不命中）
  std::cout << "\n--- Test 3: ND = 2048 (first execution) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData3;
  tilingData3.set_ND(2048);
  tilingData3.set_block_dim(64);
  tilingData3.set_ub_size(245760);

  std::unordered_map<int64_t, uint64_t> workspace_map3;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData3, workspace_map3, -1, &group_cache)) {
    std::cout << "Test 3 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData3.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData3.get_ND() << std::endl;

  // Test 4: ND = 2048，再次执行（应命中 Group 缓存）
  std::cout << "\n--- Test 4: ND = 2048 (should hit Group cache) ---" << std::endl;
  AscGraph0ScheduleResult0G0TilingData tilingData4;
  tilingData4.set_ND(2048);
  tilingData4.set_block_dim(64);
  tilingData4.set_ub_size(245760);

  std::unordered_map<int64_t, uint64_t> workspace_map4;
  if (!AscGraph0ScheduleResult0G0::GetTiling(
        tilingData4, workspace_map4, -1, &group_cache)) {
    std::cout << "Test 4 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData4.get_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData4.get_ND() << std::endl;

  // ========== Operator 级别缓存测试 ==========
  std::cout << "\n========== Operator-level cache test ==========" << std::endl;

  // Test 5: ND = 4096，首次执行（不命中）
  std::cout << "\n--- Test 5: ND = 4096 (first execution) ---" << std::endl;
  graph_ndTilingData tilingData5;
  tilingData5.set_block_dim(64);
  tilingData5.set_ub_size(245760);
  tilingData5.graph0_result0_g0_tiling_data.set_ND(4096);

  if (!GetTiling(tilingData5, -1, nullptr)) {
    std::cout << "Test 5 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData5.get_graph0_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData5.graph0_result0_g0_tiling_data.get_ND() << std::endl;

  // Test 6: ND = 4096，再次执行（应命中 Operator 缓存）
  std::cout << "\n--- Test 6: ND = 4096 (should hit Operator cache) ---" << std::endl;
  graph_ndTilingData tilingData6;
  tilingData6.set_block_dim(64);
  tilingData6.set_ub_size(245760);
  tilingData6.graph0_result0_g0_tiling_data.set_ND(4096);

  if (!GetTiling(tilingData6, -1, nullptr)) {
    std::cout << "Test 6 failed" << std::endl;
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData6.get_graph0_tiling_key() << std::endl;
  std::cout << "ND = " << tilingData6.graph0_result0_g0_tiling_data.get_ND() << std::endl;
  std::cout << "\n========== All tests passed ==========" << std::endl;
  return 0;
}
)";
}  // namespace
using namespace att;
using namespace att::test;

namespace {
void SaveConcatTilingApi(const std::map<std::string, std::string> &tiling_funcs) {
  const auto header = tiling_funcs.find("TilingHead");
  ASSERT_NE(header, tiling_funcs.end());
  std::ofstream output("Concat_tiling_api.h", std::ios::out);
  ASSERT_TRUE(output.is_open());
  output << "#pragma once\n#include \"Concat_tiling_data.h\"\n";
  output << att::test::RemoveAutoFuseTilingHeadGuards(header->second);
  output.close();
  ASSERT_TRUE(output.good());
}
}  // namespace

namespace {
// 编译命令常量
const std::string kCompileCmd =
    "g++ tiling_func_main_concat.cpp Concat_tiling_func.cpp "
    "-o tiling_func_main_concat -I ./ -DSTUB_LOG";
const std::string kCompileDebugCmd =
    "g++ -ggdb3 -O0 tiling_func_main_concat.cpp Concat_tiling_func.cpp "
    "-o tiling_func_main_concat -I ./ -DSTUB_LOG";
const std::string kCompileAllCmd =
    "g++ tiling_func_main_concat.cpp Concat_*_tiling_func.cpp "
    "-o tiling_func_main_concat -I ./ -DSTUB_LOG";
const std::string kCompileEmptyCmd =
    "g++ tiling_func_main_concat_empty.cpp Concat_tiling_func.cpp "
    "-o tiling_func_main_concat_empty -I ./ -DSTUB_LOG";
}  // namespace

class STestGenConcat : public ::testing::Test {
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
    setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
    AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
    AutoFuseConfig::MutableAttStrategyConfig().force_template_op_name = "";
    AutoFuseConfig::MutableAttStrategyConfig().force_tiling_case = "";
    AutoFuseConfig::MutableAttStrategyConfig().force_schedule_result = -1L;
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    unsetenv("AUTOFUSE_DFX_FLAGS");
  }

 protected:
  // 辅助函数：验证Group缓存行为
  void VerifyGroupCacheBehavior();
  // 辅助函数：验证静态shape缓存行为
  void VerifyStaticShapeCacheBehavior();

  // 辅助函数：构建单个图到schedule_group
  void BuildSingleGraphToScheduleGroup(ascir::AscGraph &graph, ascir::ScheduleGroup &schedule_group,
                                       uint32_t tiling_key);

  // 辅助函数：生成tiling函数并写入文件
  void GenerateTilingFunctionAndWriteToFile(const std::string &op_name,
                                            const ascir::FusedScheduledResult &fused_scheduled_result,
                                            std::map<std::string, std::string> &options);

  // 辅助函数：生成tiling数据和头文件
  void GenerateTilingDataAndHeader(const std::string &op_name, const std::string &graph_name,
                                   const ascir::FusedScheduledResult &fused_scheduled_result,
                                   std::map<std::string, std::string> &options);

  // 辅助函数：准备测试环境文件
  void PrepareTestEnvironmentFiles(const std::string &test_header_content = "");

  // 辅助函数：编译生成的tiling代码
  void CompileGeneratedTilingCode();

  // 辅助函数：编译并运行tiling测试，验证输出
  void CompileAndRunTilingTest(const std::string &output_file = "./info.log");

  // 辅助函数：构造两个图+两个schedule_result的fused_scheduled_result
  void ConstructTwoGraphTwoResult(const std::string &graph_name, ascir::AscGraph &graph1, ascir::AscGraph &graph2,
                                  ascir::FusedScheduledResult &fused_scheduled_result);

  // 静态成员：静态shape缓存测试主函数代码
  static const std::string kStaticShapeCacheTestMain;
};

// Concat Normal 辅助结构：轴信息
struct ConcatNormalAxisInfo {
  af::Symbol A, R, BL, ONE, ZERO;
  int64_t a_id, r_id, bl_id;
};

// 创建Data节点（输入数据）并设置属性
void CreateDataNode(Data &node, ascir::HintGraph &graph, const char *name, const ConcatNormalAxisInfo &ax,
                    ge::DataType dtype, const std::vector<af::Expression> &repeats,
                    const std::vector<af::Expression> &strides) {
  node.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  node.y.dtype = dtype;
  *node.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

// 创建Load节点并设置属性
void CreateLoadNode(Load &node, const Data &src, const ConcatNormalAxisInfo &ax, ge::DataType dtype,
                    const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.x = src.y;
  node.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  node.y.dtype = dtype;
  *node.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

// 创建Store节点并设置属性
void CreateStoreNode(Store &node, const ConcatNormalAxisInfo &ax, ge::DataType dtype,
                     const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  node.y.dtype = dtype;
  *node.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

// 创建Output节点并设置属性
void CreateOutputNode(Output &node, const ConcatNormalAxisInfo &ax, ge::DataType dtype,
                      const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  node.y.dtype = dtype;
  *node.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *node.y.repeats = repeats;
  *node.y.strides = strides;
}

void BuildMeanConcatNode(Concat &mean, const ConcatNormalAxisInfo &ax, const std::vector<af::AscOpOutput> &inputs) {
  auto aoo = std::vector<af::Expression>{ax.A, ax.ONE, ax.ONE};
  auto oss_v = std::vector<af::Expression>{ax.ONE, ax.ZERO, ax.ZERO};
  mean.attr.api.unit = af::ComputeUnit::kUnitVector;
  mean.x = inputs;
  mean.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  mean.y.dtype = ge::DT_FLOAT;
  *mean.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *mean.y.repeats = aoo;
  *mean.y.strides = oss_v;
}

void BuildConcatOutputNodes(const ConcatNormalAxisInfo &ax, const Store &x_out,
                            const std::vector<Store> &output_stores) {
  auto arb = std::vector<af::Expression>{ax.A, ax.R, ax.ONE};
  auto rs = std::vector<af::Expression>{ax.R, ax.ONE, ax.ZERO};
  auto aoo = std::vector<af::Expression>{ax.A, ax.ONE, ax.ONE};
  auto oss_v = std::vector<af::Expression>{ax.ONE, ax.ZERO, ax.ZERO};
  Output buf1("buf1");
  buf1.x = x_out.y;
  buf1.y.dtype = ge::DT_FLOAT16;
  *buf1.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *buf1.y.repeats = arb;
  *buf1.y.strides = rs;

  Output buf2("buf2");
  CreateOutputNode(buf2, ax, ge::DT_FLOAT, aoo, oss_v);
  buf2.x = output_stores[0].y;
  Output buf3("buf3");
  CreateOutputNode(buf3, ax, ge::DT_FLOAT, aoo, oss_v);
  buf3.x = output_stores[1].y;
  Output buf("buf");
  CreateOutputNode(buf, ax, ge::DT_FLOAT16, arb, rs);
  buf.x = output_stores[2].y;
  Output buf4("buf4");
  CreateOutputNode(buf4, ax, ge::DT_FLOAT16, arb, rs);
  buf4.x = output_stores[3].y;
}

struct ConcatVecExprs {
  std::vector<af::Expression> arb;
  std::vector<af::Expression> rs;
  std::vector<af::Expression> aoo;
  std::vector<af::Expression> oss_v;
};

void CreateStoreFp16Node(Store &store, const ConcatNormalAxisInfo &ax, const af::AscOpOutput &input,
                         const ConcatVecExprs &vecs) {
  store.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  store.x = input;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *store.y.repeats = vecs.arb;
  *store.y.strides = vecs.rs;
}

void BuildConcatRstdYAndOutputs(const ConcatNormalAxisInfo &ax, const ConcatVecExprs &vecs,
                                const af::AscOpOutput &mean_y, const Store &x_out, const Store &mean_out,
                                ascir::HintGraph &graph, const af::AscOpOutput &x1_out, const af::AscOpOutput &x2_out) {
  auto oob = std::vector<af::Expression>{ax.ONE, ax.ONE, ax.BL};
  auto oso = std::vector<af::Expression>{ax.ZERO, ax.ZERO, ax.ONE};
  Data one("one", graph);
  CreateDataNode(one, graph, "one", ax, ge::DT_FLOAT, oob, oso);
  Concat rstd("rstd");
  rstd.attr.api.unit = af::ComputeUnit::kUnitVector;
  rstd.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  rstd.x = {mean_y, mean_y, one.y};
  rstd.y.dtype = ge::DT_FLOAT;
  *rstd.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *rstd.y.repeats = vecs.arb;
  *rstd.y.strides = vecs.rs;

  Store rstd_out("rstd_out");
  CreateStoreNode(rstd_out, ax, ge::DT_FLOAT, vecs.aoo, vecs.oss_v);
  rstd_out.x = rstd.y;

  auto orb = std::vector<af::Expression>{ax.ONE, ax.R, ax.ONE};
  auto bg_strides = std::vector<af::Expression>{ax.ZERO, ax.ONE, ax.ZERO};
  Data beta("beta", graph);
  CreateDataNode(beta, graph, "beta", ax, ge::DT_FLOAT16, orb, bg_strides);
  Load betaLocal("betaLocal");
  CreateLoadNode(betaLocal, beta, ax, ge::DT_FLOAT16, orb, bg_strides);
  Data gamma("gamma", graph);
  CreateDataNode(gamma, graph, "gamma", ax, ge::DT_FLOAT16, orb, bg_strides);
  Load gammaLocal("gammaLocal");
  CreateLoadNode(gammaLocal, gamma, ax, ge::DT_FLOAT16, orb, bg_strides);

  Concat y("y");
  y.attr.api.unit = af::ComputeUnit::kUnitVector;
  y.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  y.x = {rstd.y, betaLocal.y, gammaLocal.y, rstd.y};
  y.y.dtype = ge::DT_FLOAT16;
  *y.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *y.y.repeats = vecs.arb;
  *y.y.strides = vecs.rs;

  Concat concat("concat");
  concat.attr.api.unit = af::ComputeUnit::kUnitVector;
  concat.x = {x1_out, x2_out};
  concat.attr.sched.axis = {ax.a_id, ax.r_id, ax.bl_id};
  concat.y.dtype = ge::DT_FLOAT16;
  *concat.y.axis = {ax.a_id, ax.r_id, ax.bl_id};
  *concat.y.repeats = vecs.arb;
  *concat.y.strides = vecs.rs;

  Store y_out("y_out");
  CreateStoreFp16Node(y_out, ax, y.y, vecs);
  Store cat_out("cat_out");
  CreateStoreFp16Node(cat_out, ax, y.y, vecs);
  (void)y_out;
  (void)cat_out;

  BuildConcatOutputNodes(ax, x_out, {mean_out, rstd_out, y_out, cat_out});
}

void Concat_Normal_BeforeAutofuse(ascir::HintGraph &graph) {
  auto ONE = af::sym::kSymbolOne;
  auto ZERO = af::sym::kSymbolZero;
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);
  ConcatNormalAxisInfo ax{A, R, BL, ONE, ZERO, a.id, r.id, bl.id};
  ConcatVecExprs vecs{{A, R, ONE}, {R, ONE, ZERO}, {A, ONE, ONE}, {ONE, ZERO, ZERO}};

  auto arb = std::vector<af::Expression>{ax.A, ax.R, ax.ONE};
  auto rs = std::vector<af::Expression>{ax.R, ax.ONE, ax.ZERO};
  Data x1("x1", graph);
  CreateDataNode(x1, graph, "x1", ax, ge::DT_FLOAT16, arb, rs);
  Load x1Local("x1Local");
  CreateLoadNode(x1Local, x1, ax, ge::DT_FLOAT16, arb, rs);
  Data x2("x2", graph);
  CreateDataNode(x2, graph, "x2", ax, ge::DT_FLOAT16, arb, rs);
  Load x2Local("x2Local");
  CreateLoadNode(x2Local, x2, ax, ge::DT_FLOAT16, arb, rs);
  Data bias("bias", graph);
  CreateDataNode(bias, graph, "bias", ax, ge::DT_FLOAT16, arb, rs);
  Load biasLocal("biasLocal");
  CreateLoadNode(biasLocal, bias, ax, ge::DT_FLOAT16, arb, rs);
  Concat mean("mean");
  BuildMeanConcatNode(mean, ax, {x1Local.y, x2Local.y, biasLocal.y});

  Store x_out("x_out");
  CreateStoreFp16Node(x_out, ax, mean.y, vecs);
  Store mean_out("mean_out");
  CreateStoreNode(mean_out, ax, ge::DT_FLOAT, vecs.aoo, vecs.oss_v);
  mean_out.x = mean.y;

  BuildConcatRstdYAndOutputs(ax, vecs, mean.y, x_out, mean_out, graph, x1Local.y, x2Local.y);
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
                      std::initializer_list<int64_t> vec_axis, bool set_unit_vector = false) {
  auto node = graph.FindNode(name);
  graph.ApplySplit(node, aBO, aBI);
  graph.ApplySplit(node, aBIO, aBII);
  if (set_unit_vector) {
    node->attr.api.unit = af::ComputeUnit::kUnitVector;
  }
  node->attr.sched.loop_axis = aBIO;
  node->outputs[0].attr.vectorized_axis = vec_axis;
}

void SetGmNode(ascir::HintGraph &graph, const char *name) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  node->outputs[0].attr.mem.position = af::Position::kPositionGM;
}

void SetQueueNode(ascir::HintGraph &graph, const char *name, int &tensor_id, int que_id, af::Position position) {
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

void SetBufferNode(ascir::HintGraph &graph, const char *name, int &tensor_id, int buf_id) {
  auto node = graph.FindNode(name);
  node->outputs[0].attr.mem.tensor_id = tensor_id++;
  node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  node->outputs[0].attr.mem.reuse_id = ascir::ID_NONE;
  node->outputs[0].attr.buf.id = buf_id;
  node->outputs[0].attr.que.id = ascir::ID_NONE;
  node->outputs[0].attr.que.depth = ascir::ID_NONE;
  node->outputs[0].attr.que.buf_num = ascir::ID_NONE;
  node->outputs[0].attr.opt.ref_tensor = ascir::ID_NONE;
}

void Concat_Normal_AfterScheduler(ascir::HintGraph &graph) {
  auto a = graph.FindAxis(0)->id;
  auto r = graph.FindAxis(1)->id;
  auto [aBO, aBI] = graph.BlockSplit(a, "nbi", "nbo");
  auto [aBIO, aBII] = graph.TileSplit(aBI->id, "nii", "nio");
  ApplySplitToNode(graph, "x1", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "x2", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "bias", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "x1Local", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "x2Local", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "biasLocal", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "mean", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "x_out", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "mean_out", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "rstd", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "rstd_out", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "betaLocal", aBO->id, aBI->id, aBIO->id, aBII->id, {r});
  ApplySplitToNode(graph, "gammaLocal", aBO->id, aBI->id, aBIO->id, aBII->id, {r});
  ApplySplitToNode(graph, "y", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "concat", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r}, true);
  ApplySplitToNode(graph, "y_out", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
  ApplySplitToNode(graph, "cat_out", aBO->id, aBI->id, aBIO->id, aBII->id, {aBII->id, r});
}

void Concat_Normal_AfterQueBufAlloc(ascir::HintGraph &graph) {
  int tensor_id = 0;
  int que_id = 0;
  int buf_id = 0;
  int x1_que = que_id++;
  int x2_que = que_id++;
  int bias_que = que_id++;
  int gamma_que = que_id++;
  int beta_que = que_id++;
  int mean_que = que_id++;
  int rstd_que = que_id++;
  int y_que = que_id++;
  int x_que = que_id++;
  int x32_queue = que_id++;
  int one_t_buf = buf_id++;

  SetGmNode(graph, "x1");
  SetGmNode(graph, "x2");
  SetGmNode(graph, "bias");
  SetQueueNode(graph, "x1Local", tensor_id, x1_que, af::Position::kPositionVecIn);
  SetQueueNode(graph, "x2Local", tensor_id, x2_que, af::Position::kPositionVecIn);
  SetQueueNode(graph, "biasLocal", tensor_id, bias_que, af::Position::kPositionVecIn);
  SetQueueNode(graph, "mean", tensor_id, mean_que, af::Position::kPositionVecOut);
  SetGmNode(graph, "x_out");
  SetGmNode(graph, "mean_out");
  SetBufferNode(graph, "one", tensor_id, one_t_buf);
  SetQueueNode(graph, "rstd", tensor_id, y_que, af::Position::kPositionVecOut);
  SetGmNode(graph, "rstd_out");
  SetGmNode(graph, "beta");
  SetQueueNode(graph, "betaLocal", tensor_id, beta_que, af::Position::kPositionVecIn);
  SetGmNode(graph, "gamma");
  SetQueueNode(graph, "gammaLocal", tensor_id, gamma_que, af::Position::kPositionVecIn);
  SetQueueNode(graph, "y", tensor_id, y_que, af::Position::kPositionVecOut);
  SetGmNode(graph, "y_out");
  SetQueueNode(graph, "concat", tensor_id, y_que, af::Position::kPositionVecOut);
  SetGmNode(graph, "cat_out");
}

namespace af {
namespace ascir {
namespace cg {
Status BuildConcatGroupAscendGraphND(af::AscGraph &graph) {
  // create default axis
  auto ND = af::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0S1MultiTiling(af::AscGraph &graph) {
  auto S0 = af::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto S1 = af::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto [s2T, s2t] = graph.TileSplit(s0.id);
  auto [s1T, s1t] = graph.TileSplit(s1.id);
  auto s1Ts2T = *graph.MergeAxis({s1T->id, s2T->id});
  auto [s1Ts2TB, s1Ts2Tb] = graph.BlockSplit(s1Ts2T.id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s0, s1}, 0);
  LOOP(*s1Ts2TB) {
    LOOP(*s1Ts2Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto abs = Abs("abs1", load1).TQue(Position::kPositionVecOut, 1, 1);
      auto store1 = Store("store1", abs);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*s1Ts2TB, *s1Ts2Tb, *s1t, *s2t}, {load1, store1}, 1));
      *load1.axis = {s1Ts2Tb->id, s2t->id, s1t->id};
      *load1.repeats = {s1Ts2Tb->size, s2t->size, s1t->size};
      *load1.strides = {s2t->size * s1t->size, s1t->size, CreateExpr(1)};
      *load1.vectorized_axis = {s1t->id, s2t->id};

      *abs.axis = {s1Ts2Tb->id, s1t->id, s2t->id};
      *abs.repeats = {s1Ts2Tb->size, s1t->size, s2t->size};
      *abs.strides = {s2t->size * s1t->size, s2t->size, CreateExpr(1)};
      *abs.vectorized_axis = {s1t->id, s2t->id};

      *store1.axis = {s1Ts2Tb->id, s1t->id, s2t->id};
      *store1.repeats = {s1Ts2Tb->size, s1t->size, s2t->size};
      *store1.strides = {s2t->size * s1t->size, s2t->size, CreateExpr(1)};
      *store1.vectorized_axis = {s1t->id, s2t->id};
      auto output1 = Output("output1", store1);
    }
  }
  for (auto node : graph.GetAllNodes()) {
    if (node->outputs().empty()) {
      continue;
    }
    auto last_dim_name = GetVecString(node->outputs()[0]->attr.repeats);
    GELOGD("Found Tile split axis %s in load/store node", last_dim_name.c_str());
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0S1_Reorder(af::AscGraph &graph) {
  // create default axis
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S0 = af::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto S1 = af::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto [ndB, ndb] = graph.BlockSplit(s0.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s0}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {s1}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphS1S0_Reorder(af::AscGraph &graph) {
  // create default axis
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);

  auto S1 = af::Symbol("S1");
  auto s1 = graph.CreateAxis("s1", S1);
  auto S0 = af::Symbol("S0");
  auto s0 = graph.CreateAxis("s0", S0);
  auto [ndB, ndb] = graph.BlockSplit(s1.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s1}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {s0}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphS0(af::AscGraph &graph) {
  // create default axis
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
  LOOP(*z0B) {
    LOOP(*z0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0B, *z0bT, *z0b, *z0bt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphND2(af::AscGraph &graph) {
  // create default axis
  auto A = af::Symbol("A");
  auto R = af::Symbol("R");
  auto BL = af::Symbol(8, "BL");
  auto a = graph.CreateAxis("A", A);
  auto r = graph.CreateAxis("R", R);
  auto bl = graph.CreateAxis("BL", BL);
  auto ND = af::Symbol("ND2");
  auto nd = graph.CreateAxis("nd2", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphND2WithAbs(af::AscGraph &graph) {
  // create default axis
  auto ND = af::Symbol("ND2");
  auto nd = graph.CreateAxis("nd2", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto abs = Abs("load1", load1).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", abs);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, abs, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphND2TB(af::AscGraph &graph) {
  // create default axis
  auto ND = af::Symbol("ND2");
  auto nd = graph.CreateAxis("nd2", ND);
  auto [ndT, ndt] = graph.TileSplit(nd.id);
  auto [ndTB, ndTb] = graph.BlockSplit(ndT->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndTB) {
    LOOP(*ndTb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(
          GraphConstructUtils::UpdateOutputTensorAxes({*ndTB, *ndTb, *ndt}, {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildConcatGroupAscendGraphStatic(af::AscGraph &graph) {
  // create default axis
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
      auto load2_perm = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2_perm);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndb, *ndbt},
                                                                    {load1, load2_perm, store1, store2}, 2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
    }
  }
  return af::SUCCESS;
}

Status BuildTqueTbufAscendGraph_single_case(af::AscGraph &graph) {
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
  data->attr.tmp_buffers.emplace_back(TmpBuffer{TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  return af::SUCCESS;
}

Status BuildTqueTbufAscendGraphMultiCaseG0(af::AscGraph &graph) {
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

Status BuildTqueTbufAscendGraphMultiCaseG1(af::AscGraph &graph) {
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

Status BuildMultiCaseG0(af::AscGraph &graph) {
  auto ND = af::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndT, ndt] = graph.TileSplit(nd.id);
  auto [ndTB, ndTb] = graph.BlockSplit(ndT->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {nd}, 2);
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {nd}, 3);
  LOOP(*ndTB) {
    LOOP(*ndTb) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load2", data2).TBuf(Position::kPositionVecIn);
      auto load_tbuf1 = Load("load3", data3).TBuf(Position::kPositionVecIn);
      auto load_tbuf2 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tbuf0);
      auto store3 = Store("store2", load_tbuf1);
      auto store4 = Store("store2", load_tbuf2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndTB, *ndTb, *ndt}, {load_tque0, load_tbuf0, load_tbuf1, load_tbuf2, store1, store2, store3, store4}, 1));
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

Status BuildMultiCaseG1(af::AscGraph &graph) {
  auto S0 = af::Symbol("S0");
  auto z0 = graph.CreateAxis("z0", S0);
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0}, 1);
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {z0}, 2);
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {z0}, 3);
  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load_tque0 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque1 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tque2 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 1);
      auto load_tbuf0 = Load("load4", data4).TBuf(Position::kPositionVecIn);
      auto store1 = Store("store1", load_tque0);
      auto store2 = Store("store2", load_tque1);
      auto store3 = Store("store2", load_tque2);
      auto store4 = Store("store2", load_tbuf0);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*z0TB, *z0Tb, *z0t}, {load_tque0, load_tque1, load_tque2, load_tbuf0, store1, store2, store3, store4}, 1));
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

Status BuildSevenInputsMiddleAxisCacheLineConflict(ge::AscGraph &graph) {
  auto S0 = af::Symbol("S0");
  auto S1 = af::Symbol("S1");
  auto s0 = graph.CreateAxis("s0", S0);
  auto s1 = graph.CreateAxis("s1", S1);
  auto [s0B, s0b] = graph.BlockSplit(s0.id);
  auto [s0bT, s0bt] = graph.TileSplit(s0b->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {s0, s1}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {s0, s1}, 1);
  auto data3 = graph.CreateContiguousData("input3", DT_FLOAT, {s0, s1}, 2);
  auto data4 = graph.CreateContiguousData("input4", DT_FLOAT, {s0, s1}, 3);
  auto data5 = graph.CreateContiguousData("input5", DT_FLOAT, {s0, s1}, 4);
  auto data6 = graph.CreateContiguousData("input6", DT_FLOAT, {s0, s1}, 5);
  auto data7 = graph.CreateContiguousData("input7", DT_FLOAT, {s0, s1}, 6);
  LOOP(*s0B) {
    LOOP(*s0bT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto load3 = Load("load3", data3).TQue(Position::kPositionVecIn, 1, 3);
      auto load4 = Load("load4", data4).TQue(Position::kPositionVecIn, 1, 4);
      auto load5 = Load("load5", data5).TQue(Position::kPositionVecIn, 1, 5);
      auto load6 = Load("load6", data6).TQue(Position::kPositionVecIn, 1, 6);
      auto load7 = Load("load7", data7).TQue(Position::kPositionVecIn, 1, 7);
      auto store1 = Store("store1", load1);
      auto store2 = Store("store2", load2);
      auto store3 = Store("store3", load3);
      auto store4 = Store("store4", load4);
      auto store5 = Store("store5", load5);
      auto store6 = Store("store6", load6);
      auto store7 = Store("store7", load7);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*s0B, *s0bT, *s0b, *s0bt},
          {load1, load2, load3, load4, load5, load6, load7, store1, store2, store3, store4, store5, store6, store7},
          2));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output3", store3);
      auto output4 = Output("output4", store4);
      auto output5 = Output("output5", store5);
      auto output6 = Output("output6", store6);
      auto output7 = Output("output7", store7);
    }
  }
  return af::SUCCESS;
}
}  // namespace cg
}  // namespace ascir
}  // namespace af
const std::string kTqueTbufCase0Main = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(tque_tbuf_case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "b0_size"<< " = " << tilingData.get_b0_size() << std::endl;
  std::cout << "q0_size"<< " = " << tilingData.get_q0_size() << std::endl;
  std::cout << "b1_size"<< " = " << tilingData.get_b1_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  tque_tbuf_case0TilingData tilingData;
  tilingData.set_b0_size(64);
  tilingData.set_q0_size(128);
  tilingData.set_b1_size(512);
  PrintResult(tilingData);
  return 0;
}
)";

const std::string kFirstGraphName = "case0";
const std::string kSecondGraphName = "case1";

TEST_F(STestGenConcat, tque_tbuf_case0) {
  const std::string kCaseName = "tque_tbuf_case0";
  ascir::AscGraph graph(kCaseName.c_str());
  ASSERT_EQ(af::ascir::cg::BuildTqueTbufAscendGraph_single_case(graph), af::SUCCESS);
  graph.SetTilingKey(0U);

  ascir::ScheduleGroup schedule_group;
  ascir::ScheduledResult schedule_result;
  schedule_group.impl_graphs.emplace_back(graph);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group.impl_graphs);
  schedule_result.schedule_groups.emplace_back(schedule_group);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result});
  auto res = GenTilingImplAutoFuseV3("Concat", fused_scheduled_result, options, tiling_funcs, true);
  SaveConcatTilingApi(tiling_funcs);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_EQ(res, true);

  auto tiling_res =
      GenerateTilingCodeCommon("Concat", fused_scheduled_result, options, tiling_func, "Concat_tiling_data.h", false);
  std::ofstream oss;
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[kCaseName + "TilingData"];
  oss.close();
  auto ret = autofuse::test::CopyOpLog(TOP_DIR);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kTqueTbufCase0Main;
  oss.close();
  ret = std::system(kCompileDebugCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 公共函数定义：生成tiling代码并写入文件
std::map<std::string, std::string> GenerateTilingCodeCommon(const std::string &op_name,
                                                            const ascir::FusedScheduledResult &fused_scheduled_result,
                                                            const std::map<std::string, std::string> &options,
                                                            const std::string &tiling_func,
                                                            const std::string &header_file, bool gen_extra_infos) {
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"" << header_file << "\"\n";
  oss << tiling_func;
  oss.close();

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options.at(af::sym::kTilingDataTypeName);
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = gen_extra_infos;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), af::SUCCESS);

  return tiling_res;
}

// 完整的Concat tiling生成流水线：GenTiling → 写文件 → GenTilingCode → 写data → CopyStub
void GenConcatTilingFullPipeline(const std::string &op_name, const ascir::FusedScheduledResult &fused_scheduled_result,
                                 std::map<std::string, std::string> &options, const std::string &graph_name) {
  std::map<std::string, std::string> tiling_funcs;
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  SaveConcatTilingApi(tiling_funcs);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
  EXPECT_EQ(res, true);

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), af::SUCCESS);
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
  auto ret = autofuse::test::CopyOpLog(TOP_DIR);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
}

// 写main文件、编译、运行并检查输出
void WriteMainCompileRunAndCheck(const std::string &main_content, const std::string &expected_output) {
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << main_content;
  oss.close();
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  if (!expected_output.empty()) {
    EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", expected_output), true);
  }
}

// IsStaticShape测试的raw string常量
const std::string kIsStaticShapeTestMain = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;
extern "C" bool IsStaticShape();
int main() {
  if (IsStaticShape()) {
    std::cout << "Got static graph" << std::endl;
  } else {
    std::cout << "Got dynamic graph" << std::endl;
  }
  return 0;
}
)";

// equal_priority_tiling测试的raw string常量
const std::string kEqualPriorityTradeoffMain = R"(
#include <sstream>
#include <iostream>
#include <stdio.h>
double g_input_ub_threshold = 0.1;
double g_input_corenum_threshold = 1.0;

#define MY_ASSERT_EQ(x, y)                                                                                    \
 do {                                                                                                        \
   const auto &xv = (x);                                                                                     \
   const auto &yv = (y);                                                                                     \
   if (xv != yv) {                                                                                           \
     std::stringstream ss;                                                                                   \
     ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv;                \
     printf("%s\n", ss.str().c_str());                                                             \
     std::exit(1);                                                                                           \
   }                                                                                                         \
 } while (false)
#include <iostream>
#include <vector>
#include "Concat_tiling_api.h"
using namespace optiling;

#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Rational(a, b) ((double)(a) / (double)(b))
inline bool ReferenceIsEqual(double a, double b)
{
  const double epsilon = 1e-8;
  double abs = (a > b) ? (a - b) : (b - a);
  return abs < epsilon;
}
template<typename T>
inline T ReferenceCeiling(T a)
{
  T value = static_cast<T>(static_cast<int64_t>(a));
  return (ReferenceIsEqual(value, a)) ? value : (value + 1);
}

struct TestCase {
  const char* name;
  uint32_t S0;
  uint32_t S1;
  uint32_t ub_size;
  bool need_square{true};
  uint32_t expect_s0t_size{0U};
  uint32_t expect_s1t_size{0U};
  uint32_t expect_ub_size{0U};
  uint32_t expect_block_dim{0U};
};

int main() {
  std::vector<TestCase> test_cases = {
      {"Case1_0: Equal priority", 20, 20, 40000},
      {"Case1_1: Equal priority", 30, 30, 40000},
      {"Case1_2: Equal priority", 40, 40, 40000},
      {"Case1_3: Equal priority", 40, 200, 40000, false, 32},
      {"Case1_4: Equal priority", 40, 2000, 40000, false, 40},
      {"Case1_5: Equal priority", 40, 20000, 40000, false, 40},
      {"Case1: Equal priority", 50, 50, 40000},
      {"Case2: Equal priority", 100, 100, 40000},
      {"Case3: Equal priority", 150, 150, 40000},
      {"Case4: Equal priority", 200, 200, 40000},
      {"Case5: Equal priority", 250, 250, 40000},
      {"Case6: Equal priority", 300, 300, 40000},
      {"Case7: Equal priority", 350, 350, 40000},
      {"Case8: Equal priority", 400, 400, 40000},
      {"Case9: Equal priority", 500, 500, 40000},
      {"Case10: Equal priority", 550, 550, 40000},
      {"Case11: Equal priority", 600, 600, 40000},
      {"Case12: Equal priority", 750, 750, 40000},
      {"Case13: Equal priority", 800, 800, 40000},
      {"Case14: Equal priority", 850, 850, 40000},
      {"Case15: Equal priority", 900, 900, 40000},
      {"Case16: Equal priority", 950, 950, 40000},
      {"Case17: Equal priority", 1000, 1000, 40000},
  };

  int64_t passed = 0;
  int64_t failed = 0;
  constexpr uint32_t kHardwareCoreNum = 72U;
  constexpr uint32_t kCoreNumThreshold = kHardwareCoreNum * 0.9;
  constexpr uint32_t kSquareTileThreshold = 5U;
  for (const auto& tc : test_cases) {
    case0TilingData tilingData;
    tilingData.set_ub_size(tc.ub_size);
    tilingData.set_S1(tc.S1);
    tilingData.set_S0(tc.S0);
    tilingData.set_block_dim(kHardwareCoreNum);

    std::cout << "\n========== Testing: " << tc.name << " ==========" << std::endl;
    std::cout << "Input: S0=" << tc.S0 << ", S1=" << tc.S1 << ", ub_size=" << tc.ub_size << std::endl;

    if (GetTiling(tilingData, -1, nullptr)) {
      std::cout << "tc.name s0t: " << tc.name << " " << tilingData.get_s0t_size() << std::endl;
      std::cout << "tc.name s1t: " << tc.name << " " << tilingData.get_s1t_size() << std::endl;
      std::cout << "tc.name q0_size: " << tc.name << " " << tilingData.get_q0_size() << std::endl;

      double tensor_0 = (4 * tilingData.s1t_size_);
      double tensor_1 = (4 * tilingData.s0t_size_ * tilingData.s1t_size_);
      int64_t ub_size = ((32 * ReferenceCeiling((Rational(1, 32) * tensor_0))) + (32 * ReferenceCeiling((Rational(1, 32) * tensor_1))));
      int64_t all_ub_size = ub_size * tilingData.s1Ts0Tb_size_ * tilingData.block_dim_;
      std::cout << "tc.name ub_size: " << tc.name << " " << ub_size << std::endl;
      std::cout << "tc.name block_dim: " << tc.name << " " << tilingData.block_dim_ << std::endl;
      std::cout << "tc.name all_ub_size: " << tc.name << " " << all_ub_size << std::endl;
      bool is_pass = true;
      int64_t ub_ratio_10 = int(tc.ub_size * g_input_ub_threshold);
      std::cout << "tc.name ub_ratio_10: " << tc.name << " " << ub_ratio_10 << std::endl;
      auto block_dim = Max(0, ReferenceCeiling((ReferenceCeiling((static_cast<double>(tc.S0) /
                         static_cast<double>(tilingData.s0t_size_))) * ReferenceCeiling((static_cast<double>(tc.S1) /
                         static_cast<double>(tilingData.s1t_size_))) / (tilingData.s1Ts0Tb_size_))));
      std::cout << "tc.name block_dim: " << tc.name << " " << block_dim << std::endl;
      auto raw_block_dim = ReferenceCeiling((ReferenceCeiling((static_cast<double>(tc.S0) / static_cast<double>(tilingData.s0t_size_)))
                             * ReferenceCeiling((static_cast<double>(tc.S1) / static_cast<double>(tilingData.s1t_size_)))));
      std::cout << "tc.name raw_block_dim: " << tc.name << " " << raw_block_dim << std::endl;

      if (tilingData.block_dim_ == kHardwareCoreNum && ub_size < ub_ratio_10) {
        is_pass = false;
      }
      if (tilingData.block_dim_ < kCoreNumThreshold && raw_block_dim >= kCoreNumThreshold) {
        is_pass = true;
      }
      if ((tc.expect_s0t_size > 0) && (tilingData.s0t_size_ != tc.expect_s0t_size)) {
        std::cout << "expect_s0t_size=" << tc.expect_s0t_size << ", s0t=" << tilingData.s0t_size_ << std::endl;
        is_pass = false;
      }
      if ((tc.expect_s1t_size > 0) && (tilingData.s1t_size_ != tc.expect_s1t_size)) {
        std::cout << "expect_s1t_size=" << tc.expect_s1t_size << ", s1t=" << tilingData.s1t_size_ << std::endl;
        is_pass = false;
      }
      if ((tc.expect_ub_size > 0) && (ub_size != tc.expect_ub_size)) {
        std::cout << "expect_ub_size=" << tc.expect_s0t_size << ", ub_size=" << ub_size << std::endl;
        is_pass = false;
      }
      if ((tc.expect_block_dim > 0) && (tilingData.block_dim_ != tc.expect_block_dim)) {
        std::cout << "expect_block_dim=" << tc.expect_block_dim << ", s0t=" << tilingData.block_dim_ << std::endl;
        is_pass = false;
      }
      bool is_square_tile = std::abs((static_cast<int64_t>(tilingData.get_s0t_size()) -
                                      static_cast<int64_t>(tilingData.get_s1t_size()))) < kSquareTileThreshold;
      if (!tc.need_square) {
        is_square_tile = true;
        if (tilingData.s1Ts0Tb_size_ > 2) {
          is_pass = false;
        }
      }
      if (is_pass && tilingData.get_s0t_size() > 0 && tilingData.get_s1t_size() > 0 && (ub_size <= tc.ub_size) &&
          (all_ub_size >= (tc.S0 * tc.S1 * 4)) && (tilingData.block_dim_ <= kHardwareCoreNum) &&
          (is_square_tile)) {
        std::cout << "PASSED - Both axes are tiled" << std::endl;
        passed++;
      } else {
        std::cout << "FAILED - One or both axes not tiled, is_pass=" << is_pass << ", s0t=" << tilingData.get_s0t_size()
                  << ", s1t=" << tilingData.get_s1t_size() << ", ub_size=" << ub_size << ", all_ub_size=" << all_ub_size
                  << ", block_dim=" << tilingData.block_dim_ << std::endl;
        failed++;
      }
    } else {
      std::cout << "FAILED - GetTiling returned false" << std::endl;
      failed++;
    }
  }

  std::cout << "\n==================== Test Summary ====================" << std::endl;
  std::cout << "Total: " << (passed + failed) << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << "====================================================" << std::endl;

  return (failed == 0) ? 0 : -1;
}
)";

const std::string kEqualPriorityTilingMain = R"(
#include <iostream>
#include <vector>
#include "Concat_tiling_api.h"
using namespace optiling;

struct TestCase {
  const char *name;
  uint32_t S0;
  uint32_t S1;
  uint32_t ub_size;
};

int main() {
  std::vector<TestCase> test_cases = {
    {"Case1: Equal priority 16x16", 16, 16, 10240},
    {"Case2: Equal priority 32x16", 32, 16, 10240},
    {"Case3: Equal priority 32x32", 32, 32, 20480},
    {"Case4: Equal priority 64x32", 64, 32, 40960},
    {"Case5: Equal priority 128x64", 128, 64, 81920},
  };

  int passed = 0;
  int failed = 0;

  for (const auto &tc : test_cases) {
    case0TilingData tilingData;
    tilingData.set_ub_size(tc.ub_size);
    tilingData.set_S1(tc.S1);
    tilingData.set_S0(tc.S0);
    tilingData.set_block_dim(1);

    std::cout << "\n========== Testing: " << tc.name << " ==========" << std::endl;
    std::cout << "Input: S0=" << tc.S0 << ", S1=" << tc.S1 << ", ub_size=" << tc.ub_size << std::endl;

    if (GetTiling(tilingData, -1, nullptr)) {
      std::cout << "S0: " << tilingData.get_s0t_size() << std::endl;
      std::cout << "S1: " << tilingData.get_s1t_size() << std::endl;
      std::cout << "q0_size: " << tilingData.get_q0_size() << std::endl;

      // 验证切分结果
      if (tilingData.get_s0t_size() > 0 && tilingData.get_s1t_size() > 0) {
        std::cout << "PASSED - Both axes are tiled" << std::endl;
        passed++;
      } else {
        std::cout << "FAILED - One or both axes not tiled" << std::endl;
        failed++;
      }
    } else {
      std::cout << "FAILED - GetTiling returned false" << std::endl;
      failed++;
    }
  }

  std::cout << "\n==================== Test Summary ====================" << std::endl;
  std::cout << "Total: " << (passed + failed) << std::endl;
  std::cout << "Passed: " << passed << std::endl;
  std::cout << "Failed: " << failed << std::endl;
  std::cout << "====================================================" << std::endl;

  return (failed == 0) ? 0 : -1;
}
)";

// fused_schedule_result_reuse_schedule_group的raw string常量
const std::string kFusedReuseScheduleGroupMain = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_graph0_tiling_key = " << tilingData.get_graph0_tiling_key() << std::endl;
  std::cout << "get_graph1_tiling_key = " << tilingData.get_graph1_tiling_key() << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.graph0_result1_g0_tiling_data.get_nd2bt_size() << std::endl;
  std::cout << "get_s0bt_size = " << tilingData.graph1_result0_g0_tiling_data.get_s0bt_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  tilingData.graph1_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph1_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";

const std::string kDiffSearchAxisMain = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_ND2(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";

const std::string kScoreSameScheduleMain = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_tiling_key = " << tilingData.get_tiling_key() << std::endl;
  std::cout << "get_nd2bt_size = " << tilingData.get_nd2bt_size() << std::endl;
  std::cout << "get_nd2t_size = " << tilingData.get_nd2t_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_ND2(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";

const std::string kPromptAlignedMain = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_block_dim = " << tilingData.get_block_dim() << std::endl;
  std::cout << "get_nd2t_size = " << tilingData.get_nd2t_size() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_ND2(1025);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";

// 空tensor测试用例的辅助函数
namespace {
void BuildEmptyTensorGraph(const std::string &graph_name, ascir::ScheduledResult &schedule_result) {
  ascir::ScheduleGroup schedule_group1;
  ascir::AscGraph graph(graph_name.c_str());
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph), af::SUCCESS);
  graph.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result.schedule_groups.emplace_back(schedule_group1);
}

void GenerateTilingCodeForEmptyTensor(const std::string &op_name,
                                      const std::vector<ascir::ScheduledResult> &schedule_results,
                                      const std::string &graph_name) {
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  SaveConcatTilingApi(tiling_funcs);
  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  EXPECT_EQ(res, true);

  auto tiling_res =
      GenerateTilingCodeCommon(op_name, fused_scheduled_result, options, tiling_func, "Concat_tiling_data.h", true);

  std::ofstream oss;
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
  auto ret = autofuse::test::CopyOpLog(TOP_DIR);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
}

void CompileAndRunEmptyTensorTest(const std::string &graph_name) {
  std::ofstream oss;
  oss.open("tiling_func_main_concat_empty.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

int main() {
)" + graph_name + R"(TilingData tilingData;
  tilingData.set_block_dim(1);
  tilingData.set_ub_size(1024);
  tilingData.set_ND(0);

  if (GetTiling(tilingData, -1, nullptr)) {
    std::cout << "GetTiling success with empty tensor ND=0" << std::endl;
    return 0;
  } else {
    std::cout << "GetTiling failed with empty tensor" << std::endl;
    return -1;
  }
}
)";
  oss << kRunTilingFuncMainLocal;
  oss.close();

  auto ret = std::system(kCompileEmptyCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat_empty > ./info_empty.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info_empty.log", "GetTiling success with empty tensor"), true);
}
}  // namespace

af::Status ConstructTQueTBufScheduleResults(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1(kSecondGraphName.c_str());
  GE_ASSERT_EQ(af::ascir::cg::BuildTqueTbufAscendGraphMultiCaseG0(graph_0), af::SUCCESS);
  graph_0.SetTilingKey(0U);
  GE_ASSERT_EQ(af::ascir::cg::BuildTqueTbufAscendGraphMultiCaseG1(graph_1), af::SUCCESS);
  graph_1.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);

  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  return af::SUCCESS;
}

af::Status ConstructAutoTuneResults(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::string json_info;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  ascir::AscGraph graph_1(kSecondGraphName.c_str());
  GE_ASSERT_EQ(af::ascir::cg::BuildMultiCaseG0(graph_0), af::SUCCESS);
  graph_0.SetTilingKey(0U);
  GE_ASSERT_EQ(af::ascir::cg::BuildMultiCaseG1(graph_1), af::SUCCESS);
  graph_1.SetTilingKey(1U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  schedule_group2.impl_graphs.emplace_back(graph_1);

  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func =
      ("int32_t CalcScore(" + kFirstGraphName + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  return af::SUCCESS;
}

af::Status ConstructSingleCaseForMultiTileScheduleResult(std::vector<ascir::ScheduledResult> &schedule_results) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  std::vector<att::ModelInfo> model_info_list;
  ascir::AscGraph graph_0(kFirstGraphName.c_str());
  GE_ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphS0S1MultiTiling(graph_0), af::SUCCESS);
  graph_0.SetTilingKey(0U);
  schedule_group1.impl_graphs.emplace_back(graph_0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);
  return af::SUCCESS;
}

af::Status GenTilingImpl(std::vector<ascir::ScheduledResult> &schedule_results) {
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  SaveConcatTilingApi(tiling_funcs);
  GE_ASSERT_EQ(res, true);
  GE_ASSERT_EQ(att::test_common::GenTilingCodeToFile(op_name, tiling_funcs, fused_scheduled_result, options,
                                                     kFirstGraphName + "TilingData"),
               af::SUCCESS);
  auto ret = autofuse::test::CopyOpLog(TOP_DIR);
  GE_ASSERT_EQ(ret, 0);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  GE_ASSERT_EQ(ret, 0);
  return af::SUCCESS;
}

af::Status ConstructTQueTBufMultiCaseGroup() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructTQueTBufScheduleResults(schedule_results), af::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), af::SUCCESS);
  return af::SUCCESS;
}

af::Status ConstructAutoTuneCaseGroup() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructAutoTuneResults(schedule_results), af::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), af::SUCCESS);
  return af::SUCCESS;
}

af::Status ConstructSingleCaseForMultiTile() {
  std::vector<ascir::ScheduledResult> schedule_results;
  GE_ASSERT_EQ(ConstructSingleCaseForMultiTileScheduleResult(schedule_results), af::SUCCESS);
  GE_ASSERT_EQ(GenTilingImpl(schedule_results), af::SUCCESS);
  return af::SUCCESS;
}

TEST_F(STestGenConcat, construct_single_case_for_multi_tile) {
  EXPECT_EQ(ConstructSingleCaseForMultiTile(), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_api.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  // FP32 buffers aligned to 32 bytes: q0 = 32*ceil(s1/8), q1 = 32*ceil(s0*s1/8).
  // (16,16) needs 64+1024=1088 bytes; (16,15) needs 64+960=1024 bytes.
  MY_ASSERT_EQ(tilingData.get_s0t_size(), 16);
  MY_ASSERT_EQ(tilingData.get_s1t_size(), 15);
  MY_ASSERT_EQ(tilingData.get_q0_size(), 64);
  MY_ASSERT_EQ(tilingData.get_q1_size(), 960);
  MY_ASSERT_EQ(tilingData.get_q0_size() + tilingData.get_q1_size(), tilingData.get_ub_size());
  MY_ASSERT_EQ((32 * ((16 + 7) / 8) + 32 * ((16 * 16 + 7) / 8)) > tilingData.get_ub_size(), true);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(1);
  tilingData.set_ub_size(1024);
  tilingData.set_S1(1024);
  tilingData.set_S0(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(kCompileDebugCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：验证同等优先级轴的切分功能
// 构造2个order相同(=1)的切分轴S0和S1
// 预期结果：
// 1. Tiling求解成功
// 2. S0和S1被正确切分
// 3. UB利用率符合预期
// 用例描述：验证同等优先级轴的切分功能
TEST_F(STestGenConcat, equal_priority_tiling) {
  EXPECT_EQ(ConstructSingleCaseForMultiTile(), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat_equal_priority.cpp", std::ios::out);
  oss << ResultCheckerUtils::DefineCheckerFunction() << kEqualPriorityTilingMain;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat_equal_priority.cpp "
      "Concat_tiling_func.cpp -o tiling_func_main_concat_equal_priority -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat_equal_priority");
  EXPECT_EQ(ret, 0);
}

// 用例描述：验证同等优先级轴的切分功能(使用多核UB tradeoff)
// 构造2个order相同(=1)的切分轴S0和S1
// 预期结果：
// 1. Tiling求解成功
// 2. S0和S1被正确切分
// 3. UB利用率符合预期
TEST_F(STestGenConcat, equal_priority_tiling_trade_off) {
  setenv("AUTOFUSE_DFX_FLAGS",
         "--att_enable_multicore_ub_tradeoff=true;--att_corenum_threshold=100;--att_ub_threshold=10", 1);
  EXPECT_EQ(ConstructSingleCaseForMultiTile(), af::SUCCESS);
  std::ofstream oss("tiling_func_main_concat_equal_priority.cpp", std::ios::out);
  oss << ResultCheckerUtils::DefineCheckerFunction() << kEqualPriorityTradeoffMain;
  oss.close();
  auto ret = std::system(
      "g++ -ggdb3 -O0 tiling_func_main_concat_equal_priority.cpp "
      "Concat_tiling_func.cpp -o tiling_func_main_concat_equal_priority -I ./ -DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat_equal_priority");
  EXPECT_EQ(ret, 0);
}

TEST_F(STestGenConcat, tque_tbuf_case1) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(ConstructTQueTBufMultiCaseGroup(), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_api.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim(), 1);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b1_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b2_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b3_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_q0_size(), 0);
  MY_ASSERT_EQ(tilingData.graph0_result0_g0_tiling_data.get_b0_size(), 0);

  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_b3_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q0_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q1_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_q2_size(), 4096);
  MY_ASSERT_EQ(tilingData.graph0_result1_g0_tiling_data.get_b0_size(), 16384);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(kCompileDebugCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：复用tque_tbuf_case1的构图，开启DFX --att_accuracy_level=0，使能多核调优，增加使用的核数
// 用例期望结果：
// 1.Tiling求解成功；
// 2.使用核数未用满
TEST_F(STestGenConcat, auto_tuning_accuracy_level_0) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=0", 1);
  EXPECT_EQ(ConstructAutoTuneCaseGroup(), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_api.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim() < 64, true);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(256 * 1024);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024 * 1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024 * 1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(kCompileDebugCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

// 用例描述：复用tque_tbuf_case1的构图，开启DFX --att_accuracy_level=1，使能多核调优，增加使用的核数
// 用例期望结果：
// 1.Tiling求解成功；
// 2.使用核数相较level0模式上升；
TEST_F(STestGenConcat, auto_tuning_accuracy_level_1) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=1", 1);
  EXPECT_EQ(ConstructAutoTuneCaseGroup(), af::SUCCESS);
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  const std::string kRunTilingFuncMainLocal = R"(
#include "Concat_tiling_api.h"
using namespace optiling;
void PrintResult(case0TilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  MY_ASSERT_EQ(tilingData.get_block_dim() == 64, true);
  MY_ASSERT_EQ(tilingData.get_graph0_tiling_key(), 1);
  std::cout << "====================================================" << std::endl;
}

int main() {
  case0TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(256 * 1024);
  tilingData.graph0_result0_g0_tiling_data.set_ND(1024 * 1024);
  tilingData.graph0_result1_g0_tiling_data.set_S0(1024 * 1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss << ResultCheckerUtils::DefineCheckerFunction() << kRunTilingFuncMainLocal;
  oss.close();
  auto ret = std::system(kCompileDebugCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

af::Status ConstructConcatTwoTilingCaseS0S1() {
  const std::string kFirstName = "Concat";
  ascir::AscGraph graph_nd(kFirstName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  GE_ASSERT_SUCCESS(af::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd));
  graph_nd.SetTilingKey(0U);
  GE_ASSERT_SUCCESS(af::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0));
  graph_s0.SetTilingKey(1U);

  ascir::ScheduleGroup schedule_group;
  ascir::ScheduledResult schedule_result;
  schedule_group.impl_graphs.emplace_back(graph_nd);
  schedule_group.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group.impl_graphs);
  schedule_result.schedule_groups.emplace_back(schedule_group);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result});
  GE_ASSERT_EQ(GenTilingImplAutoFuseV3("Concat", fused_scheduled_result, options, tiling_funcs, true), true);
  SaveConcatTilingApi(tiling_funcs);
  GE_ASSERT_EQ(att::test_common::GenTilingCodeToFile("Concat", tiling_funcs, fused_scheduled_result, options,
                                                     kFirstName + "TilingData"),
               af::SUCCESS);
  auto ret = autofuse::test::CopyOpLog(TOP_DIR);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  GE_ASSERT_TRUE(ret == 0);
  std::ofstream oss("tiling_func_main_concat.cpp", std::ios::out);
  oss << R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;
int main() {
  ConcatTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_S0(1024);
  tilingData.set_S1(1024);
  if (!GetTiling(tilingData, -1, nullptr)) {
    return -1;
  }
  std::cout << "get_tiling_key = " << tilingData.get_tiling_key() << std::endl;
  return 0;
}
)";
  return af::SUCCESS;
}

const std::string kS0S1TilingTestMain = R"(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(graph_ndTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_graph0_tiling_key();
  std::cout << "get_tiling_key"<< " = " << tiling_key << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_ndTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_S0(1024);
  tilingData.graph0_result1_g0_tiling_data.set_S1(1024);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";

af::Status ConstructTwoScheduleResultS0S1() {
  const std::string graph_name = "graph_nd";
  ascir::AscGraph graph_nd(graph_name.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  GE_ASSERT_SUCCESS(af::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd));
  graph_nd.SetTilingKey(0U);
  GE_ASSERT_SUCCESS(af::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0));
  graph_s0.SetTilingKey(1U);

  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  schedule_group2.impl_graphs.emplace_back(graph_s0);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func = ("int32_t CalcScore(" + graph_name + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func = ("int32_t CalcScore(" + graph_name + "TilingData &tiling_data) { return 2;}").c_str();

  std::vector<ascir::ScheduledResult> schedule_results = {schedule_result1, schedule_result2};
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  GE_ASSERT_TRUE(GenTilingImplAutoFuseV3("Concat", fused_scheduled_result, options, tiling_funcs, true));
  SaveConcatTilingApi(tiling_funcs);
  GE_ASSERT_EQ(att::test_common::GenTilingCodeToFile("Concat", tiling_funcs, fused_scheduled_result, options,
                                                     graph_name + "TilingData"),
               af::SUCCESS);
  auto ret = autofuse::test::CopyOpLog(TOP_DIR);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  GE_ASSERT_TRUE(ret == 0);
  std::ofstream oss("tiling_func_main_concat.cpp", std::ios::out);
  oss << kS0S1TilingTestMain;
  return af::SUCCESS;
}

TEST_F(STestGenConcat, case_axes_reorder) {
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
  Concat_Normal_AfterQueBufAlloc(graph_normal);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["gen_extra_info"] = "1";
  options["solver_type"] = "AxesReorder";
  EXPECT_EQ(GenTilingImpl("Concat", graphs, options), true);
  AddHeaderGuardToFile("autofuse_tiling_func_common.h", "__AUTOFUSE_TILING_FUNC_COMMON_H__");
  auto ret =
      std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_concat.cpp ./ -f").c_str());
  ret = autofuse::test::CopyOpLog(TOP_DIR);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);

  ret = std::system(kCompileAllCmd.c_str());
  EXPECT_EQ(ret, 0);

  ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ret, 0);
}

TEST_F(STestGenConcat, case_axes_reorder_got_static_shape) {
  const std::string kGraphName = "graph_static";
  ascir::AscGraph graph_static(kGraphName.c_str());
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphStatic(graph_static), af::SUCCESS);
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  BuildSingleGraphToScheduleGroup(graph_static, schedule_group, 0U);
  schedule_result.schedule_groups.emplace_back(schedule_group);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result});
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kIsStaticShapeTestMain, "Got static graph");
}

TEST_F(STestGenConcat, case_axes_reorder_got_dynamic_shape) {
  const std::string kGraphName = "graph_dynamic";
  ascir::AscGraph graph_dynamic(kGraphName.c_str());
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_dynamic), af::SUCCESS);
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  BuildSingleGraphToScheduleGroup(graph_dynamic, schedule_group, 0U);
  schedule_result.schedule_groups.emplace_back(schedule_group);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(
      std::vector<ascir::ScheduledResult>{schedule_result});
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kIsStaticShapeTestMain, "Got dynamic graph");
}

void STestGenConcat::VerifyGroupCacheBehavior() {
  // ========== 缓存校验 ==========
  // Test 1: ND=1024 Group Cache MISS -> SAVE
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] MISS! key=[1024]"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] SAVE SUCCESS: key=[1024]"), true);

  // Test 2: ND=1024 Group Cache HIT
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] HIT"), true);

  // Test 3: ND=2048 Group Cache MISS -> SAVE
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] MISS! key=[2048]"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Group Cache] SAVE SUCCESS: key=[2048]"), true);

  // Test 4: ND=2048 Group Cache HIT (已在上面通过 HIT 校验覆盖)

  // Test 5: ND=4096 Operator Cache MISS -> SAVE
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] MISS! key=[4096]"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] SAVE SUCCESS: key=[4096]"), true);

  // 新格式校验：多维shape key应带分隔逻辑
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "if (i != 0) {"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "out.append(\",\");"), true);

  // Test 6: ND=4096 Operator Cache HIT
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] HIT! key=[4096]"), true);
  // The current cache-hit event is checked above; the removed legacy message is no longer emitted.
}

void STestGenConcat::VerifyStaticShapeCacheBehavior() {
  // 用例描述：验证静态shape场景下的缓存功能
  // 用例期望结果：
  // 1. 生成的TilingFunc包含算子级缓存代码（即使kInputShapeSize=0）
  // 2. 使用空key进行缓存查询和保存
  // 3. 编译通过且运行时缓存正常工作

  // 验证生成的代码包含kInputShapeSize = 0（静态shape）
  EXPECT_TRUE(
      ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "constexpr size_t kInputShapeSize = 0;"));

  // 验证生成的代码包含静态shape的缓存处理（空key）
  EXPECT_TRUE(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
                                                       "std::array<uint32_t, kInputShapeSize> input_shapes = {};"));

  // 验证生成的代码包含算子级缓存查询（静态shape）
  EXPECT_TRUE(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "if (cache_hit) {"));
}

void STestGenConcat::BuildSingleGraphToScheduleGroup(ascir::AscGraph &graph, ascir::ScheduleGroup &schedule_group,
                                                     uint32_t tiling_key) {
  graph.SetTilingKey(tiling_key);
  schedule_group.impl_graphs.emplace_back(graph);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group.impl_graphs);
}

void STestGenConcat::ConstructTwoGraphTwoResult(const std::string &graph_name, ascir::AscGraph &graph1,
                                                ascir::AscGraph &graph2,
                                                ascir::FusedScheduledResult &fused_scheduled_result) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_group1.impl_graphs.emplace_back(graph1);
  schedule_group2.impl_graphs.emplace_back(graph2);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func = ("int32_t CalcScore(" + graph_name + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func = ("int32_t CalcScore(" + graph_name + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
}

void STestGenConcat::GenerateTilingFunctionAndWriteToFile(const std::string &op_name,
                                                          const ascir::FusedScheduledResult &fused_scheduled_result,
                                                          std::map<std::string, std::string> &options) {
  std::map<std::string, std::string> tiling_funcs;
  auto res = GenTilingImplAutoFuseV3(op_name, fused_scheduled_result, options, tiling_funcs, true);
  SaveConcatTilingApi(tiling_funcs);
  EXPECT_EQ(res, true);

  std::string tiling_func;
  CombineTilings(tiling_funcs, tiling_func);
  std::ofstream oss;
  oss.open("Concat_tiling_func.cpp", std::ios::out);
  oss << "#include \"Concat_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
}

void STestGenConcat::GenerateTilingDataAndHeader(const std::string &op_name, const std::string &graph_name,
                                                 const ascir::FusedScheduledResult &fused_scheduled_result,
                                                 std::map<std::string, std::string> &options) {
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_scheduled_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options.at(kTilingDataTypeName);
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), af::SUCCESS);

  std::ofstream oss;
  oss.open("Concat_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
}

void STestGenConcat::PrepareTestEnvironmentFiles(const std::string &test_header_content) {
  auto ret = autofuse::test::CopyOpLog(TOP_DIR);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);

  EXPECT_EQ(ResultCheckerUtils::ReplaceLogMacrosGeneric("Concat_tiling_func.cpp"), true);

  if (!test_header_content.empty()) {
    std::ofstream oss;
    oss.open("Concat_tiling_test.h", std::ios::out);
    oss << test_header_content;
    oss.close();
  }
}

void STestGenConcat::CompileGeneratedTilingCode() {
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
}

void STestGenConcat::CompileAndRunTilingTest(const std::string &output_file) {
  auto ret = std::system((kCompileCmd + " 2>&1").c_str());
  EXPECT_EQ(ret, 0);

  ret = std::system(("./tiling_func_main_concat > " + output_file + " 2>&1").c_str());
  EXPECT_EQ(ret, 0);
}

const std::string STestGenConcat::kStaticShapeCacheTestMain = R"RAW(
#include <iostream>
#include "Concat_tiling_api.h"
using namespace optiling;

void PrintResult(graph_static_cacheTilingData& tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "Static shape cache test" << std::endl;
  std::cout << "get_block_dim = " << tilingData.get_block_dim() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_static_cacheTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);

  // First call: cache miss, normal solve
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "First tiling func execute failed." << std::endl;
    return -1;
  }

  // Restore the original resource request: the first solve overwrites block_dim.
  const auto expected_block_dim = tilingData.get_block_dim();
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  // Second call: identical shape and resources must hit the cache.
  if (GetTiling(tilingData, -1, nullptr)) {
    if (tilingData.get_block_dim() != expected_block_dim || tilingData.get_ub_size() != 245760) {
      return -1;
    }
    std::cout << "Second call success - cache hit" << std::endl;
  } else {
    std::cout << "Second tiling func execute failed." << std::endl;
    return -1;
  }

  return 0;
}
)RAW";

TEST_F(STestGenConcat, reuse_schedule_group_with_same_input_axis_name) {
  const std::string kGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), af::SUCCESS);
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_s0), af::SUCCESS);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  ConstructTwoGraphTwoResult(kGraphName, graph_nd, graph_s0, fused_scheduled_result);
  GenerateTilingFunctionAndWriteToFile("Concat", fused_scheduled_result, options);
  GenerateTilingDataAndHeader("Concat", kGraphName, fused_scheduled_result, options);
  PrepareTestEnvironmentFiles(kConcatTilingTestHead);

  // 验证生成的Group缓存代码
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "namespace AscGraph0ScheduleResult0G0"),
            true);
  EXPECT_EQ(
      ResultCheckerUtils::IsFileContainsString(
          "Concat_tiling_func.cpp", "using GroupLevelCache = FixedSizeHashMap<kInputShapeSize, 4, TilingDataCopy>"),
      true);
  EXPECT_TRUE(
      ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
                                               "bool GetTiling(AscGraph0ScheduleResult0G0TilingData &tiling_data, "
                                               "std::unordered_map<int64_t, uint64_t> &workspace_map, "
                                               "int32_t tiling_case_id, GroupLevelCache *cache"));

  std::ofstream oss("tiling_func_main_concat.cpp", std::ios::out);
  oss << kRunTilingFuncMainSameND;
  oss.close();
  CompileGeneratedTilingCode();
  auto ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  VerifyGroupCacheBehavior();
  (void)ret;
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_name) {
  const std::string kGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), af::SUCCESS);
  graph_nd.SetTilingKey(0U);
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphS0(graph_s0), af::SUCCESS);
  graph_s0.SetTilingKey(1U);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  ConstructTwoGraphTwoResult(kGraphName, graph_nd, graph_s0, fused_scheduled_result);
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kRunTilingFuncMain, "get_tiling_key = 1");
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_search_axis_name) {
  const std::string kGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), af::SUCCESS);
  graph_nd.SetTilingKey(0U);
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND2(graph_s0), af::SUCCESS);
  graph_s0.SetTilingKey(1U);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  ConstructTwoGraphTwoResult(kGraphName, graph_nd, graph_s0, fused_scheduled_result);
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kDiffSearchAxisMain, "get_tiling_key = 1");
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order) {
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
}

TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_schedule_result) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_schedule_result=0", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 默认选择tiling key 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_normal_tiling_case) {
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
}

// 强制选择tiling case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_tiling_case) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定op name,选择tiling case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定错误的op name,按照默认选择tiling case 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_error_op_name) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Conct;--force_tiling_case=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定op name,强制选择group0的tiling case为case 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name_group0_1) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_tiling_case=g0_1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定错误的op name,按照默认选择schedule result 0
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_error_op_name2) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Conct;--force_schedule_result=1", 1);
  EXPECT_EQ(ConstructConcatTwoTilingCaseS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 强制指定正确的的op name,选择schedule result 1
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_force_op_name2) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_template_op_name=Concat;--force_schedule_result=1", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

// 测试att_accuracy_level=1场景，使能自动选择最优核数
TEST_F(STestGenConcat, reuse_schedule_group_with_different_input_axis_order_auto_tuning) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_accuracy_level=1", 1);
  EXPECT_EQ(ConstructTwoScheduleResultS0S1(), af::SUCCESS);
  auto ret = std::system(kCompileCmd.c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_tiling_key = 1"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(STestGenConcat, fused_schedule_result_reuse_schedule_group) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kGraphName = "graph_nd";
  {
    ascir::AscGraph graph_nd(kGraphName.c_str());
    ascir::AscGraph graph_s0("graph_s0");
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), af::SUCCESS);
    graph_nd.SetTilingKey(0U);
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND2(graph_s0), af::SUCCESS);
    graph_s0.SetTilingKey(1U);
    ConstructTwoGraphTwoResult(kGraphName, graph_nd, graph_s0, fused_scheduled_result);
  }
  {
    ascir::AscGraph graph_nd("graph_s0s1");
    ascir::AscGraph graph_s0("graph_s1s0");
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd), af::SUCCESS);
    graph_nd.SetTilingKey(2U);
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0), af::SUCCESS);
    graph_s0.SetTilingKey(3U);
    ascir::ScheduleGroup sg1;
    ascir::ScheduleGroup sg2;
    ascir::ScheduledResult sr1;
    ascir::ScheduledResult sr2;
    std::vector<ascir::ScheduledResult> srs;
    sg1.impl_graphs.emplace_back(graph_nd);
    sg2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(sg1.impl_graphs);
    GraphConstructUtils::UpdateGraphsVectorizedStride(sg2.impl_graphs);
    sr1.schedule_groups.emplace_back(sg1);
    sr1.score_func = ("int32_t CalcScore(" + kGraphName + "TilingData &tiling_data) { return 3;}").c_str();
    sr2.schedule_groups.emplace_back(sg2);
    sr2.score_func = ("int32_t CalcScore(" + kGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    srs.emplace_back(sr1);
    srs.emplace_back(sr2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(srs);
  }

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kFusedReuseScheduleGroupMain, "");
  auto ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_graph0_tiling_key = 1"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_graph1_tiling_key = 0"), true);
}

namespace {
void BuildScoreCaseFirstBlock(const std::string &graph_name, ascir::FusedScheduledResult &fused_scheduled_result) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  ascir::ScheduledResult schedule_result2;
  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::AscGraph graph_nd(graph_name.c_str());
  const char *kSr0Names[] = {"sr0_graph_g0", "sr0_graph_g1", "sr0_graph_g2", "sr0_graph_g3", "sr0_graph_g4"};
  std::vector<ascir::AscGraph> sr0_graphs = {ascir::AscGraph(kSr0Names[0]), ascir::AscGraph(kSr0Names[1]),
                                             ascir::AscGraph(kSr0Names[2]), ascir::AscGraph(kSr0Names[3]),
                                             ascir::AscGraph(kSr0Names[4])};
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), af::SUCCESS);
  graph_nd.SetTilingKey(0U);
  for (size_t i = 0; i < sr0_graphs.size(); i++) {
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_graphs[i]), af::SUCCESS);
    sr0_graphs[i].SetTilingKey(static_cast<uint32_t>(i + 1));
  }
  schedule_group1.impl_graphs.emplace_back(graph_nd);
  for (auto &g : sr0_graphs) {
    schedule_group2.impl_graphs.emplace_back(g);
  }
  const std::string kScoreTpl = "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return %d;}";
  schedule_group1.graph_name_to_score_funcs["graph_nd"] =
      "int32_t CalcScore(AscGraph0ScheduleResult0G0TilingData &tiling_data) {return -1;}";
  schedule_group2.graph_name_to_score_funcs[kSr0Names[0]] =
      "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 100;}";
  schedule_group2.graph_name_to_score_funcs[kSr0Names[1]] =
      "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 1;}";
  schedule_group2.graph_name_to_score_funcs[kSr0Names[3]] =
      "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return 100;}";
  schedule_group2.graph_name_to_score_funcs[kSr0Names[4]] =
      "int32_t CalcScore(AscGraph0ScheduleResult1G0TilingData &tiling_data) {return -1;}";
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group1.impl_graphs);
  GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.score_func = ("int32_t CalcScore(" + graph_name + "TilingData &tiling_data) { return 1;}").c_str();
  schedule_result2.schedule_groups.emplace_back(schedule_group2);
  schedule_result2.score_func = ("int32_t CalcScore(" + graph_name + "TilingData &tiling_data) { return 2;}").c_str();
  schedule_results.emplace_back(schedule_result1);
  schedule_results.emplace_back(schedule_result2);
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
}
}  // namespace

TEST_F(STestGenConcat, fused_schedule_result_tiling_case_score) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kGraphName = "graph_nd";
  BuildScoreCaseFirstBlock(kGraphName, fused_scheduled_result);
  {
    ascir::AscGraph graph_nd("graph_s0s1");
    ascir::AscGraph graph_s0("graph_s1s0");
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphS0S1_Reorder(graph_nd), af::SUCCESS);
    graph_nd.SetTilingKey(6U);
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphS1S0_Reorder(graph_s0), af::SUCCESS);
    graph_s0.SetTilingKey(7U);
    ConstructTwoGraphTwoResult(kGraphName, graph_nd, graph_s0, fused_scheduled_result);
  }

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kFusedReuseScheduleGroupMain, "get_graph0_tiling_key = 1");
}

// 场景：
// graph0: 先切Tile再切Block（有打分函数）
// graph1: 先切Block再切Tile（有打分函数）
// graph2: 先切Block再切Tile（与graph1相同Schedule，有打分函数）
// 预期：
// 选择graph1，graph0虽然打分高，但不可与graph1,2进行打分比较
TEST_F(STestGenConcat, fused_schedule_result_tiling_case_score_same_schedule) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kGraphName = "graph_nd";
  {
    ascir::AscGraph sr0_g0_0(kGraphName.c_str());
    ascir::AscGraph sr0_g0_1("graph_nd_1");
    ascir::AscGraph sr0_g0_2("graph_nd_2");
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND2TB(sr0_g0_0), af::SUCCESS);
    sr0_g0_0.SetTilingKey(0U);
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND2WithAbs(sr0_g0_1), af::SUCCESS);
    sr0_g0_1.SetTilingKey(1U);
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND2(sr0_g0_2), af::SUCCESS);
    sr0_g0_2.SetTilingKey(2U);

    ascir::ScheduleGroup schedule_group2;
    schedule_group2.impl_graphs.emplace_back(sr0_g0_0);
    schedule_group2.impl_graphs.emplace_back(sr0_g0_1);
    schedule_group2.impl_graphs.emplace_back(sr0_g0_2);
    schedule_group2.graph_name_to_score_funcs["graph_nd_0"] =
        "int32_t CalcScore(graph_ndTilingData &tiling_data) {return 101;}";
    schedule_group2.graph_name_to_score_funcs["graph_nd_1"] =
        "int32_t CalcScore(graph_ndTilingData &tiling_data) {return 100;}";
    schedule_group2.graph_name_to_score_funcs["graph_nd_2"] =
        "int32_t CalcScore(graph_ndTilingData &tiling_data) {return -1;}";
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);

    ascir::ScheduledResult schedule_result2;
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_result2.score_func = ("int32_t CalcScore(" + kGraphName + "TilingData &tiling_data) { return 2;}").c_str();
    std::vector<ascir::ScheduledResult> schedule_results;
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kScoreSameScheduleMain, "get_tiling_key = ");
}

TEST_F(STestGenConcat, fused_schedule_result_prompt_aligned) {
  ascir::FusedScheduledResult fused_scheduled_result;
  const std::string kGraphName = "graph_nd";
  {
    ascir::ScheduleGroup schedule_group2;
    ascir::ScheduledResult schedule_result2;
    std::vector<ascir::ScheduledResult> schedule_results;
    ascir::AscGraph graph_s0(kGraphName.c_str());
    ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND2TB(graph_s0), af::SUCCESS);
    graph_s0.SetTilingKey(1U);
    schedule_group2.impl_graphs.emplace_back(graph_s0);
    GraphConstructUtils::UpdateGraphsVectorizedStride(schedule_group2.impl_graphs);
    schedule_result2.schedule_groups.emplace_back(schedule_group2);
    schedule_results.emplace_back(schedule_result2);
    fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  }

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  options.emplace("enable_score_func", "1");
  GenConcatTilingFullPipeline("Concat", fused_scheduled_result, options, kGraphName);
  WriteMainCompileRunAndCheck(kPromptAlignedMain, "");
  auto ret = std::system("./tiling_func_main_concat");
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "get_nd2t_size = 17"), true);
  (void)ret;
}

// 测试ND为0时动态shape场景的空tensor处理
TEST_F(STestGenConcat, empty_tensor_nd_zero_dynamic_shape) {
  const std::string kFirstGraphName = "graph_empty_tensor";
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;

  // 构建图结构
  BuildEmptyTensorGraph(kFirstGraphName, schedule_result1);
  schedule_results.emplace_back(schedule_result1);

  // 生成tiling代码
  GenerateTilingCodeForEmptyTensor("Concat", schedule_results, kFirstGraphName);

  // 编译并运行测试
  CompileAndRunEmptyTensorTest(kFirstGraphName);
}
// 用例描述：验证静态shape场景下的缓存功能
// 用例期望结果：
// 1. 生成的TilingFunc包含算子级缓存代码（即使kInputShapeSize=0）
// 2. 使用空key进行缓存查询和保存
// 3. 编译通过且运行时缓存正常工作
TEST_F(STestGenConcat, static_shape_cache_support) {
  // 构建图
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;

  const std::string kFirstGraphName = "graph_static_cache";
  ascir::AscGraph graph_static(kFirstGraphName.c_str());
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphStatic(graph_static), af::SUCCESS);

  // 使用辅助函数构建schedule_group
  BuildSingleGraphToScheduleGroup(graph_static, schedule_group1, 0U);

  // 组装schedule_result
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);

  // 配置options
  std::map<std::string, std::string> options;
  std::string op_name = "Concat";
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);

  // 使用辅助函数生成tiling函数和数据
  GenerateTilingFunctionAndWriteToFile(op_name, fused_scheduled_result, options);
  GenerateTilingDataAndHeader(op_name, kFirstGraphName, fused_scheduled_result, options);

  // 准备测试环境并验证生成的代码
  PrepareTestEnvironmentFiles();
  VerifyStaticShapeCacheBehavior();

  // 生成静态shape缓存测试主函数
  std::ofstream oss;
  oss.open("tiling_func_main_concat.cpp", std::ios::out);
  oss << kStaticShapeCacheTestMain;
  oss.close();

  // 编译并运行测试，验证输出
  CompileAndRunTilingTest();
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Static shape cache test"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("./info.log", "Second call success"), true);
  EXPECT_TRUE(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] MISS! key=[]"));
  EXPECT_TRUE(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] SAVE SUCCESS: key=[]"));
  EXPECT_TRUE(ResultCheckerUtils::IsFileContainsString("./info.log", "[Operator Cache] HIT! key=[]"));
}

// 用例描述：验证多group场景下enable_group_parallel_optimize=true时的代码生成
// 用例期望结果：
// 1. 生成的FindPerfBetterTilingbyCaseId函数包含core_num参数
// 2. 生成的GetTilingCaseScoreFunc函数包含core_num参数
// 3. 函数调用链正确传递corenum参数
// 4. 包含group_num常量和性能调整代码
TEST_F(STestGenConcat, multi_group_with_enable_group_parallel_optimize) {
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduleGroup schedule_group2;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;
  const std::string kGraphName = "graph_nd";
  ascir::AscGraph graph_nd(kGraphName.c_str());
  ascir::AscGraph graph_s0("graph_s0");
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_nd), af::SUCCESS);
  ASSERT_EQ(af::ascir::cg::BuildConcatGroupAscendGraphND(graph_s0), af::SUCCESS);
  BuildSingleGraphToScheduleGroup(graph_nd, schedule_group1, 0U);
  BuildSingleGraphToScheduleGroup(graph_s0, schedule_group2, 1U);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_result1.schedule_groups.emplace_back(schedule_group2);
  schedule_result1.enable_group_parallel = true;
  schedule_results.emplace_back(schedule_result1);

  std::map<std::string, std::string> options;
  options.emplace(kGenConfigType, "AxesReorder");
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  GenerateTilingFunctionAndWriteToFile("Concat", fused_scheduled_result, options);
  GenerateTilingDataAndHeader("Concat", kGraphName, fused_scheduled_result, options);
  PrepareTestEnvironmentFiles(kConcatTilingTestHead);

  std::ofstream oss("tiling_func_main_concat.cpp", std::ios::out);
  oss << kRunTilingFuncMainSameND;
  oss.close();
  CompileGeneratedTilingCode();

  // 验证生成的代码包含core_num参数传递
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "bool FindPerfBetterTilingbyCaseId"),
            true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString(
                "Concat_tiling_func.cpp",
                ", uint32_t core_num, const char *sub_case_tag, const char *&selected_sub_case_tag) {"),
            true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", "bool GetTilingCaseScoreFunc"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
                                                     "uint32_t corenum = tiling_data.get_block_dim()"),
            true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp", ", corenum);"), true);
  EXPECT_EQ(ResultCheckerUtils::IsFileContainsString("Concat_tiling_func.cpp",
                                                     ", core_num, tiling.sub_case_tag, selected_sub_case_tag);"),
            true);

  auto ret = std::system("./tiling_func_main_concat > ./info.log");
  EXPECT_EQ(ret, 0);
}
