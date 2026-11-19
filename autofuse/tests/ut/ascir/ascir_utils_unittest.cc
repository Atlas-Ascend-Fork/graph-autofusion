/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <utility>
#include <vector>
#include "graph/symbolizer/symbolic.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_registry.h"
#include "ascir_utils.h"
#define private public
#include "asc_graph_dumper_context.h"
#undef private

class AscendGraphDumpUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    unsetenv("AUTOFUSE_DFX_FLAGS");
    ::ascir::utils::ResetDumpConfig();
    system("rm -rf ./ge_onnx* ./autofuse_compile_debug");
  }
};

namespace af {
TEST(UnsupportedAscIrRegistration, HasExpectedDefinitionAndImplementations) {
  const auto &registry = ascir::AscirRegistry::GetInstance().GetAll();
  const auto iter = registry.find("Unsupported");
  ASSERT_NE(iter, registry.end());

  auto definition = iter->second;
  EXPECT_TRUE(definition.GetInputDefs().empty());
  ASSERT_EQ(definition.GetOutputDefs().size(), 1U);
  EXPECT_EQ(definition.GetOutputDefs()[0].first, "y");
  EXPECT_TRUE(definition.IsStartNode());
  ASSERT_EQ(definition.GetAttrDefs().size(), 1U);
  EXPECT_EQ(definition.GetAttrDefs()[0].name, "error_msg");
  EXPECT_EQ(definition.GetComputeType(), ComputeType::kComputeInvalid);

  auto codegen = definition.GetAscIrCodegenImpl("2201");
  ASSERT_NE(codegen, nullptr);
  EXPECT_TRUE(codegen->GetApiCallName().empty());
  EXPECT_EQ(codegen->GetApiName(), "Unsupported");

  auto att = definition.GetAscIrAttImpl("2201");
  ASSERT_NE(att, nullptr);
  EXPECT_EQ(att->GetApiPerf(), nullptr);
  EXPECT_EQ(att->GetAscendCApiPerfTable(), nullptr);
}

TEST(ReduceAscIrRegistration, SupportsExpectedV35Dtypes) {
  const std::vector<af::DataType> expected_dtypes = {af::DT_BOOL,  af::DT_UINT8, af::DT_INT8, af::DT_INT16,
                                                     af::DT_INT32, af::DT_INT64, af::DT_FLOAT};
  const std::vector<std::pair<std::string, std::vector<af::DataType>>> reduce_dtypes = {
      {"Prod", {af::DT_INT8, af::DT_INT16, af::DT_INT32, af::DT_INT64, af::DT_FLOAT}},
      {"Any", expected_dtypes},
      {"All", expected_dtypes}};
  const std::map<std::string,
                 af::Status (*)(const std::vector<af::DataType> &, std::vector<af::DataType> &, const std::string &)>
      infer_dtype_functions = {{"Prod", af::ascir_op::Prod::InferDataType},
                               {"Any", af::ascir_op::Any::InferDataType},
                               {"All", af::ascir_op::All::InferDataType}};

  const auto &registry = ascir::AscirRegistry::GetInstance().GetAll();
  for (const auto &expected : reduce_dtypes) {
    const auto iter = registry.find(expected.first);
    ASSERT_NE(iter, registry.end()) << expected.first;

    const auto &soc_to_store = iter->second.GetSocToDataTypeSymbolStore();
    ASSERT_FALSE(soc_to_store.empty()) << expected.first;
    const auto &named_symbols = soc_to_store.begin()->second.GetNamedSymbols();
    const auto symbol_iter = named_symbols.find("T");
    ASSERT_NE(symbol_iter, named_symbols.end()) << expected.first;
    ASSERT_NE(symbol_iter->second, nullptr) << expected.first;

    for (const auto dtype : expected.second) {
      std::vector<af::DataType> output_dtypes;
      const auto input_dtypes = std::vector<af::DataType>{dtype};
      const auto infer_dtype = infer_dtype_functions.at(expected.first);
      EXPECT_EQ(infer_dtype(input_dtypes, output_dtypes, "3510"), af::SUCCESS) << expected.first;
      ASSERT_EQ(output_dtypes.size(), 1U);
      EXPECT_EQ(output_dtypes[0], dtype) << expected.first;
    }
  }
}

TEST_F(AscendGraphDumpUT, test_dump_when_env_not_set) {
  AscGraph graph("test");
  ::ascir::utils::DumpGraph(graph, "empty_stage0");

  AscGraph graph1("test1");
  ::ascir::utils::DumpGraph(graph1, "empty1_stage0");
  ::ascir::utils::DumpGraph(graph, "empty_stage1");
  ::ascir::utils::DumpGraph(graph1, "empty1_stage1");
  ::ascir::utils::DumpGraph(graph, "empty_stage2");

  auto &watched_graphs = ::ascir::AscGraphDumperContext::GetThreadLocalCtx().orderd_graphs_;
  ASSERT_EQ(watched_graphs.size(), 2U);

  EXPECT_EQ(watched_graphs.front().first, "empty1_stage1");
  EXPECT_EQ(watched_graphs.front().second.GetName(), graph1.GetName());
  EXPECT_EQ(watched_graphs.back().first, "empty_stage2");
  EXPECT_EQ(watched_graphs.back().second.GetName(), graph.GetName());
  ::ascir::AscGraphDumperContext::GetThreadLocalCtx().DumpWatchedGraphs();
  EXPECT_EQ(watched_graphs.size(), 2U);
  ::ascir::AscGraphDumperContext::GetThreadLocalCtx().ClearAllWatchGraphs();
  EXPECT_TRUE(watched_graphs.empty());
}

TEST_F(AscendGraphDumpUT, test_not_watch_when_env_set) {
  ::ascir::utils::ResetDumpConfig();
  setenv("AUTOFUSE_DFX_FLAGS", "codegen_compile_debug=true", 1);
  AscGraph graph("test");
  ::ascir::utils::DumpGraph(graph, "empty");

  AscGraph graph1("test1");
  ::ascir::utils::DumpGraph(graph, "empty1");

  auto &watched_graphs = ::ascir::AscGraphDumperContext::GetThreadLocalCtx().orderd_graphs_;
  EXPECT_TRUE(watched_graphs.empty());

  AscGraph sub("sub");
  graph.AddSubGraph(sub);
  ::ascir::utils::AlwaysDumpGraph(graph, "AutoFuseBeforeOptimize");
  EXPECT_TRUE(watched_graphs.empty());

  ::ascir::utils::DumpGraph(graph, "AutoFuseBeforeOptimize");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}
}  // namespace af
