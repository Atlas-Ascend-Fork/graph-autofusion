/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_graph_builder.h"
#include "gtest/gtest.h"
#include "ascendc_ir.h"
#include "ascir_ops_utils.h"
#include "asc_graph_utils.h"
#include "task_generator/concat_inputs_unification_pass.h"
#include "platform/platform_factory.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

class ConcatInputUnificationPassTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  static af::AscGraph BuildGraph(const af::Expression &s0, const af::Expression &s1,
                                 const std::vector<bool> &inputs_is_load, DataType dtype = af::DT_FLOAT16) {
    auto s2 = s1 * af::Symbol(inputs_is_load.size());
    af::testing::AscGraphBuilder graph_builder("test_graph");
    graph_builder.Loops({s0, s2});
    std::vector<std::string> concat_inputs;
    for (size_t i = 0; i < inputs_is_load.size(); ++i) {
      const auto is_load = inputs_is_load[i];
      const auto data_node_name = "data" + std::to_string(i);
      const auto load_node_name = "load" + std::to_string(i);
      graph_builder.Data(data_node_name, static_cast<int64_t>(i), dtype)
          .Load(load_node_name, data_node_name, {s0, s1}, {s1, af::sym::kSymbolOne});
      if (is_load) {
        concat_inputs.push_back(load_node_name);
      } else {
        const auto relu_node_name = "relu" + std::to_string(i);
        graph_builder.Relu(relu_node_name, load_node_name);
        concat_inputs.push_back(relu_node_name);
      }
    }
    graph_builder.Concat("concat", concat_inputs).Store("store", "concat").Output("out", "store");
    return graph_builder.Build();
  }

  static AscGraph BuildGraphShareInput() {
    const af::Expression s0 = af::Symbol(16);
    const af::Expression s1 = af::Symbol(17);
    auto s2 = s1 * Symbol(4);
    af::testing::AscGraphBuilder graph_builder("test_graph");
    std::vector<std::string> concat_inputs;
    graph_builder.Loops({s0, s2})
        .Data("data0", static_cast<int64_t>(0), ge::DT_FLOAT16)
        .Data("data1", static_cast<int64_t>(1), ge::DT_FLOAT16)
        .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
        .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
        .Abs("abs0", "load1")
        .Concat("concat", {"load0", "abs0", "load0", "abs0"})
        .Store("store", "concat")
        .Output("out", "store");
    return graph_builder.Build();
  }

  static AscGraph Build2InputGraph(const af::Expression &s0, const af::Expression &s1,
                                   DataType dtype = ge::DT_FLOAT16) {
    const std::vector<bool> inputs_is_load{true, false};
    return BuildGraph(s0, s1, inputs_is_load, dtype);
  }
};

TEST_F(ConcatInputUnificationPassTest, RunSuccess) {
  const auto graph = Build2InputGraph(af::Symbol("s0"), af::Symbol(15));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), af::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_0") != nullptr);
}

TEST_F(ConcatInputUnificationPassTest, DstColSizeOverLimit) {
  const auto graph = Build2InputGraph(af::Symbol("s0"), af::Symbol(34));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), af::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_0") == nullptr);
}

TEST_F(ConcatInputUnificationPassTest, SrcColSizeIsAligned) {
  const auto graph = Build2InputGraph(af::Symbol("s0"), af::Symbol(2));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), af::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_0") == nullptr);
}

TEST_F(ConcatInputUnificationPassTest, AllInputsIsLoad) {
  const std::vector<bool> inputs_is_load{true, true};
  const auto graph = BuildGraph(af::Symbol("s0"), af::Symbol(15), inputs_is_load);
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), af::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_0") == nullptr);
}
// src_col_size * dtype_size 超过 256/2 = 128 时才会走 IsSrcColSizeOverLimit 置位分支，
// 上面的 DstColSizeOverLimit(s1=34, fp16 -> 68) 并未真正越限，这里显式构造越限输入
// （fp16: s1=65 -> 130 > 128），覆盖 LogSrcColSizeOverLimit 的常量表达式路径。
TEST_F(ConcatInputUnificationPassTest, DstColSizeReallyOverLimit) {
  const auto graph = Build2InputGraph(af::Symbol("s0"), af::Symbol(65));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), af::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_0") == nullptr);
}

// concat_dim 越界时 GetColSize 访问 repeats 越界，这里只验证非常量表达式路径：
// 当 src_col_size 为符号量时 CanOptimize 直接返回 false（不进入越限判定）。
TEST_F(ConcatInputUnificationPassTest, SrcColSizeSymbolicNotOverLimit) {
  const auto graph = Build2InputGraph(af::Symbol("s0"), af::Symbol("s1"));
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), af::SUCCESS);
}

TEST_F(ConcatInputUnificationPassTest, InputShared) {
  const auto graph = BuildGraphShareInput();
  std::vector<::ascir::ImplGraph> graphs{graph};
  EXPECT_EQ(ConcatInputUnificationPass::Run(graphs), af::SUCCESS);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_0") != nullptr);
  EXPECT_TRUE(graphs[0].FindNode("load0_ub_cpy_input_2") != nullptr);
  EXPECT_TRUE(graphs[0].FindNode("abs0_ub_cpy_input_3") != nullptr);
}
}  // namespace schedule
