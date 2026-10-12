/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#define private public
#define protected public
#include "tiling_code_generator.h"
#include "high_perf_tiling_code_gen_impl.h"
#include "tiling_code_gen_impl.h"
#include "cache/operator_level_cache_gen.h"
#undef private
#undef protected
#include "generator_utils/tilingdata_gen_utils.h"
#include "common/tiling_source_dependencies.h"

#include <symengine/symengine_rcp.h>
#include <symengine/basic.h>
#include <symengine/symbol.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/integer.h>
#include "stub_solver_model_info.h"
#include "reuse_group_utils/reuse_group_utils.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "base/att_const_values.h"
#include "base/base_types.h"
#include "autofuse_config/auto_fuse_config.h"
#include "util/base_types_printer.h"
#include "util/duration.h"
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"

const std::string op_name = "OpTest";

namespace att {
namespace {

size_t CountSubstr(const std::string &text, const std::string &pattern) {
  size_t count = 0U;
  size_t pos = text.find(pattern);
  while (pos != std::string::npos) {
    ++count;
    pos = text.find(pattern, pos + pattern.size());
  }
  return count;
}

std::string ExtractGuardBody(const std::string &source, const std::string &guard) {
  const auto guard_pos = source.find(guard);
  if (guard_pos == std::string::npos) return {};
  const auto open_pos = source.find('{', guard_pos);
  if (open_pos == std::string::npos) return {};
  size_t depth = 1U;
  for (size_t pos = open_pos + 1U; pos < source.size(); ++pos) {
    if (source[pos] == '{') {
      ++depth;
    } else if (source[pos] == '}' && --depth == 0U) {
      return source.substr(open_pos + 1U, pos - open_pos - 1U);
    }
  }
  return {};
}

void ExpectSystemHeaders(const std::string &source, const std::vector<std::string> &required,
                         const std::vector<std::string> &forbidden) {
  for (const auto &header : required) {
    EXPECT_NE(source.find("#include <" + header + ">"), std::string::npos) << header;
  }
  for (const auto &header : forbidden) {
    EXPECT_EQ(source.find("#include <" + header + ">"), std::string::npos) << header;
  }
}
}  // namespace

class MockHighPerfTilingCodeGenImpl : public HighPerfTilingCodeGenImpl {
 public:
  MockHighPerfTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                const bool is_uniq_group)
      : HighPerfTilingCodeGenImpl(mock_op_name, config, model_infos, score_funcs, is_uniq_group) {}
};

class MockTilingCodeGenerator : public TilingCodeGenerator {
 protected:
  TilingCodeGenImplPtr CreateTilingCodeGenImpl(const std::string &mock_op_name, const TilingCodeGenConfig &config,
                                               const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                               const bool is_uniq_group) override {
    std::shared_ptr<MockHighPerfTilingCodeGenImpl> impl =
        std::make_shared<MockHighPerfTilingCodeGenImpl>(mock_op_name, config, model_infos, score_funcs, is_uniq_group);
    return impl;
  }
};

class GeneratorUT : public testing::Test {};

TEST(GeneratorUT, SerializedEdgeDiscoveryIsDeterministicAndDeduplicated) {
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, model_infos, score_funcs, false);
  const std::map<size_t, std::pair<std::string, std::string>> graph_info = {
      {0U, {"ScheduleResult0", "group0"}}, {1U, {"ScheduleResult1", "group1"}}, {2U, {"ScheduleResult2", "group2"}}};
  std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> var_relation;
  var_relation[1U][0U]["dep"] = CreateExpr(1);
  var_relation[1U][0U]["duplicate"] = CreateExpr(2);
  var_relation[1U][1U]["self"] = CreateExpr(3);
  var_relation[1U][99U]["unknown"] = CreateExpr(4);
  var_relation[0U][1U]["reverse"] = CreateExpr(5);

  EXPECT_EQ(gen_impl.GetSerializedGroupEdges(graph_info, var_relation, false),
            (std::vector<std::pair<size_t, size_t>>{{0U, 1U}, {1U, 0U}}));
  EXPECT_TRUE(gen_impl.GetSerializedGroupEdges(graph_info, var_relation, true).empty());
}

TEST(GeneratorUT, SerializedEdgeDiscoveryIgnoresEmptyInnerRelation) {
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, model_infos, score_funcs, false);
  const std::map<size_t, std::pair<std::string, std::string>> graph_info = {{0U, {"ScheduleResult0", "group0"}},
                                                                            {1U, {"ScheduleResult1", "group1"}}};
  std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> var_relation;
  var_relation[1U][0U];

  EXPECT_TRUE(gen_impl.GetSerializedGroupEdges(graph_info, var_relation, false).empty());

  var_relation[1U][0U]["dep"] = CreateExpr(1);
  EXPECT_EQ(gen_impl.GetSerializedGroupEdges(graph_info, var_relation, false),
            (std::vector<std::pair<size_t, size_t>>{{0U, 1U}}));
}

TEST(GeneratorUT, SerializedEdgePenaltyEmitsCappedExpressionAndModelLog) {
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, model_infos, score_funcs, false);
  const std::string source = gen_impl.GenSerializedEdgePenalty({"g0_perf", "g1_perf"}, {{0U, 1U}}, "  ");
  EXPECT_NE(source.find("serialized_edge_base"), std::string::npos);
  EXPECT_NE(source.find("min(8000.0"), std::string::npos);
  EXPECT_NE(source.find("0.10"), std::string::npos);
  EXPECT_NE(source.find("max(g0_perf, g1_perf)"), std::string::npos);
  EXPECT_NE(source.find("serialized_edge_base += std::max(g0_perf, g1_perf)"), std::string::npos);
  EXPECT_NE(source.find("std::min(8000.000000, 0.100000 * serialized_edge_base)"), std::string::npos);
  EXPECT_NE(source.find("serialized_edge_model=serialized_edge_v1"), std::string::npos);
  EXPECT_TRUE(gen_impl.GenSerializedEdgePenalty({"g0_perf"}, {{0U, 1U}}, "  ").empty());
}

namespace {
struct ScheduleResultSourceCase {
  TilingModelInfo model_infos;
  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  std::map<std::string, std::set<std::string>> hardware_map;
};

ScheduleResultSourceCase MakeTwoGroupScheduleResultSourceCase(const bool enable_group_parallel) {
  ScheduleResultSourceCase test_case;
  auto group0 = CreateModelInfo();
  group0.schedule_group_ident = {0U, 0U, 0U};
  group0.enable_group_parallel = enable_group_parallel;
  auto group1 = CreateModelInfo();
  group1.schedule_group_ident = {0U, 0U, 1U};
  group1.enable_group_parallel = enable_group_parallel;
  test_case.model_infos = {group0, group1};
  test_case.graph_info = {{0U, {"ScheduleResult0", "group0"}}, {1U, {"ScheduleResult1", "group1"}}};
  test_case.hardware_map = {{"group0", {"block_dim"}}, {"group1", {"block_dim"}}};
  return test_case;
}
}  // namespace

TEST(GeneratorUT, GenUpdatePerfAddsSerializedEdgeToNonParallelObjective) {
  const auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);
  gen_impl.var_relations_[0U][0U][1U][0U]["dep"] = CreateExpr(1);

  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_NE(source.find("group_sum"), std::string::npos);
  EXPECT_NE(source.find("serialized_edge_penalty"), std::string::npos);
  EXPECT_NE(source.find("objective=group_sum"), std::string::npos);
  EXPECT_NE(source.find("The value of graph0_result0 is %lf objective(group_sum(group0 + group1)=%lf + edge_model("),
            std::string::npos);
  EXPECT_NE(source.find("edge_model(serialized_edge_v1"), std::string::npos);
  EXPECT_NE(source.find("base=max(group0, group1)=%lf, penalty=min(8000.000000, 0.100000*base)"), std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfSingleGroupHasNoSerializedEdgePenalty) {
  auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  test_case.graph_info.erase(1U);
  test_case.hardware_map.erase("group1");
  test_case.model_infos.resize(1U);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);
  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_EQ(source.find("serialized_edge_penalty"), std::string::npos);
  EXPECT_NE(source.find("edge_model(serialized_edge_v1, base=0=%lf, penalty=min(8000.000000, 0.100000*base)=%lf)=%lf"),
            std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfParallelHasNoSerializedEdgePenalty) {
  const auto test_case = MakeTwoGroupScheduleResultSourceCase(true);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);
  gen_impl.var_relations_[0U][0U][1U][0U]["dep"] = CreateExpr(1);
  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  EXPECT_EQ(gen_impl.tiling_func_.GetOutputStr().find("serialized_edge_penalty"), std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfEmptyRelationKeepsGroupSumCode) {
  const auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);
  gen_impl.var_relations_[0U][0U][1U][0U];
  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_NE(source.find("cur_perf = ScheduleResult0::GetPerf(group0_tiling_data);"), std::string::npos);
  EXPECT_NE(source.find("cur_perf += ScheduleResult1::GetPerf(group1_tiling_data);"), std::string::npos);
  EXPECT_EQ(source.find("serialized_edge_penalty"), std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfMapsNonContiguousGroupIdsToPerfPositions) {
  auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  test_case.graph_info = {{2U, {"ScheduleResult0", "group0"}}, {5U, {"ScheduleResult1", "group1"}}};
  test_case.model_infos[0].schedule_group_ident.group_id = 2U;
  test_case.model_infos[1].schedule_group_ident.group_id = 5U;
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);
  gen_impl.var_relations_[0U][0U][5U][2U]["dep"] = CreateExpr(1);

  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_NE(source.find("serialized_edge_base += std::max(ScheduleResult0::GetPerf(group0_tiling_data), "),
            std::string::npos);
  EXPECT_NE(source.find("ScheduleResult1::GetPerf(group1_tiling_data));"), std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfAddsSerializedEdgeFromSharedWorkspace) {
  auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  test_case.model_infos[0].workspace_size_map.emplace(5, CreateExpr(1024));
  test_case.model_infos[1].workspace_size_map.emplace(5, CreateExpr(1024));
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);

  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_NE(source.find("serialized_edge_penalty"), std::string::npos);
  EXPECT_NE(source.find("objective=group_sum"), std::string::npos);
  EXPECT_NE(source.find("serialized_edge_base += std::max(ScheduleResult0::GetPerf(group0_tiling_data), "),
            std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfDistinctWorkspaceHasNoSerializedEdgePenalty) {
  auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  test_case.model_infos[0].workspace_size_map.emplace(5, CreateExpr(1024));
  test_case.model_infos[1].workspace_size_map.emplace(7, CreateExpr(2048));
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);

  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_NE(source.find("cur_perf = ScheduleResult0::GetPerf(group0_tiling_data);"), std::string::npos);
  EXPECT_NE(source.find("cur_perf += ScheduleResult1::GetPerf(group1_tiling_data);"), std::string::npos);
  EXPECT_EQ(source.find("serialized_edge_penalty"), std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfParallelSharedWorkspaceHasNoSerializedEdgePenalty) {
  auto test_case = MakeTwoGroupScheduleResultSourceCase(true);
  test_case.model_infos[0].workspace_size_map.emplace(5, CreateExpr(1024));
  test_case.model_infos[1].workspace_size_map.emplace(5, CreateExpr(1024));
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);

  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  EXPECT_EQ(gen_impl.tiling_func_.GetOutputStr().find("serialized_edge_penalty"), std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfWorkspaceEdgeDeduplicatedWithVarRelationEdge) {
  auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  test_case.model_infos[0].workspace_size_map.emplace(5, CreateExpr(1024));
  test_case.model_infos[1].workspace_size_map.emplace(5, CreateExpr(1024));
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);
  gen_impl.var_relations_[0U][0U][1U][0U]["dep"] = CreateExpr(1);

  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_EQ(CountSubstr(source, "serialized_edge_base += std::max("), 1U);
}

TEST(GeneratorUT, GenUpdatePerfInvalidEdgeDoesNotEmitUndefinedPenaltyLog) {
  auto model_info = CreateModelInfo();
  model_info.schedule_group_ident = {0U, 0U, 0U};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), {model_info}, score_funcs, false);

  ASSERT_EQ(gen_impl.GenUpdatePerf(0U, 0U, {"g0_perf"}, {{0U, 1U}}, {"g0_block"}, {}), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_EQ(source.find("serialized_edge_penalty"), std::string::npos);
  EXPECT_EQ(source.find("objective=group_sum"), std::string::npos);
}

TEST(GeneratorUT, GenUpdatePerfMissingHardwareGroupSkipsSerializedEdge) {
  auto test_case = MakeTwoGroupScheduleResultSourceCase(false);
  test_case.graph_info = {{2U, {"ScheduleResult0", "group0"}}, {5U, {"ScheduleResult1", "group1"}}};
  test_case.model_infos[0].schedule_group_ident.group_id = 2U;
  test_case.model_infos[1].schedule_group_ident.group_id = 5U;
  test_case.hardware_map.erase("group1");
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", TilingCodeGenConfig(), test_case.model_infos, score_funcs, false);
  gen_impl.var_relations_[0U][0U][5U][2U]["dep"] = CreateExpr(1);

  ASSERT_EQ(gen_impl.GenSingleGroupScheduleResult(0U, 0U, test_case.graph_info, test_case.hardware_map), af::SUCCESS);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_EQ(source.find("serialized_edge_penalty"), std::string::npos);
  EXPECT_EQ(source.find("objective=group_sum"), std::string::npos);
}

TEST(GeneratorUT, SourceDependenciesMergeAndRenderIncludes) {
  autofuse::GeneratedCode target;
  autofuse::RequireSystemHeader(target.dependencies, "vector");
  autofuse::RequireExternalHeader(target.dependencies, "acl/acl.h");
  autofuse::RequireGeneratedHeader(target.dependencies, autofuse::GeneratedHeaderId::kState);

  autofuse::GeneratedCode fragment;
  autofuse::RequireSystemHeader(fragment.dependencies, "cstdint");
  autofuse::RequireSystemHeader(fragment.dependencies, "vector");
  autofuse::RequireExternalHeader(fragment.dependencies, "acl/acl.h");
  autofuse::RequireExternalHeaderUnlessCceKtTest(fragment.dependencies, "platform_ascendc.h");
  autofuse::RequireGeneratedHeader(fragment.dependencies, autofuse::GeneratedHeaderId::kTilingData);
  autofuse::RequireGeneratedHeader(fragment.dependencies, autofuse::GeneratedHeaderId::kSolver);
  autofuse::AppendGeneratedCode(target, fragment);

  EXPECT_EQ(target.dependencies.system_headers.size(), 2U);
  EXPECT_EQ(target.dependencies.external_headers.size(), 1U);
  EXPECT_EQ(target.dependencies.generated_headers.size(), 3U);
  std::string includes;
  ASSERT_EQ(autofuse::RenderIncludes(target.dependencies, includes), af::SUCCESS);
  EXPECT_EQ(includes,
            "#include <cstdint>\n#include <vector>\n#include \"acl/acl.h\"\n"
            "#include \"autofuse_tiling_data.h\"\n"
            "#include \"autofuse_tiling_func_state.h\"\n"
            "#include \"autofuse_tiling_func_solver.h\"\n"
            "#ifndef __CCE_KT_TEST__\n#include \"platform_ascendc.h\"\n#endif\n");
}

TEST(GeneratorUT, GeneratedHeaderRejectsGeneratedHeaderDependency) {
  autofuse::GeneratedCode header;
  header.body = "struct HeaderValue {};\n";
  autofuse::RequireGeneratedHeader(header.dependencies, autofuse::GeneratedHeaderId::kSolver);
  std::string output;
  EXPECT_EQ(autofuse::RenderGeneratedHeader(header, "AUTOFUSE_TEST_HEADER_H_", output), af::FAILED);

  header.dependencies.generated_headers.clear();
  autofuse::RequireSystemHeader(header.dependencies, "vector");
  ASSERT_EQ(autofuse::RenderGeneratedHeader(header, "AUTOFUSE_TEST_HEADER_H_", output), af::SUCCESS);
  EXPECT_EQ(output,
            "#ifndef AUTOFUSE_TEST_HEADER_H_\n#define AUTOFUSE_TEST_HEADER_H_\n\n#include <vector>\n\n"
            "struct HeaderValue {};\n\n#endif  // AUTOFUSE_TEST_HEADER_H_\n");
}

TEST(GeneratorUT, AtomicHeaderKeysReplaceHistoricalSplitHeaders) {
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.enable_autofuse_pgo = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  EXPECT_EQ(tiling_res.count("TilingBaseHeader"), 0U);
  EXPECT_EQ(tiling_res.count("TilingEntryHeader"), 0U);
  EXPECT_EQ(tiling_res.count("TilingTailHeader"), 0U);
  EXPECT_EQ(tiling_res.count(kTilingStateHeaderIdentify), 1U);
  EXPECT_EQ(tiling_res.count(kTilingLogHeaderIdentify), 1U);
  EXPECT_EQ(tiling_res.count(kTilingPgoHeaderIdentify), 1U);
  EXPECT_EQ(tiling_res.count(kTilingSolverHeaderIdentify), 1U);
  EXPECT_EQ(tiling_res.count(kTilingApiHeaderIdentify), 1U);

  const auto &solver_header = tiling_res.at(kTilingSolverHeaderIdentify);
  EXPECT_NE(solver_header.find("GetTemp(size_t idx)"), std::string::npos);
  EXPECT_NE(solver_header.find("#include <cstddef>"), std::string::npos);
  EXPECT_EQ(solver_header.find("#define Max(a, b)"), std::string::npos);
  EXPECT_NE(solver_header.find("inline auto Max"), std::string::npos);
  EXPECT_EQ(solver_header.find("#define Min(a, b)"), std::string::npos);
  EXPECT_NE(solver_header.find("inline auto Min"), std::string::npos);
  EXPECT_EQ(solver_header.find("#define Abs(a)"), std::string::npos);
  EXPECT_NE(solver_header.find("inline auto Abs"), std::string::npos);

  const auto &solver_source = tiling_res.at(kTilingSolverIdentify);
  ExpectSystemHeaders(solver_source, {"algorithm", "cmath", "cstddef", "cstdint", "functional", "utility", "vector"},
                      {"array", "cfloat", "cstring", "map", "memory", "string", "unordered_map"});
  EXPECT_NE(solver_source.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
  EXPECT_NE(solver_source.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
  EXPECT_EQ(solver_source.find("#include \"autofuse_tiling_data.h\""), std::string::npos);
  EXPECT_EQ(solver_source.find("#include \"autofuse_tiling_func_pgo.h\""), std::string::npos);
  EXPECT_EQ(solver_source.find("#include \"autofuse_tiling_func_api.h\""), std::string::npos);

  const auto &group_source = tiling_res.at("asc_graph0_schedule_result0_g0");
  EXPECT_EQ(group_source.find("autofuse_tiling_func_common.h"), std::string::npos);
  EXPECT_EQ(group_source.find("autofuse_tiling_func_base.h"), std::string::npos);
  EXPECT_EQ(group_source.find("autofuse_tiling_func_entry.h"), std::string::npos);
  EXPECT_EQ(group_source.find("autofuse_tiling_func_tail.h"), std::string::npos);
}

TEST(GeneratorUT, TensorFlowFinalTilingIncludesCompleteTilingRepresentation) {
  auto model_info = CreateModelInfo();
  model_info.schedule_group_ident = {0UL, 0UL, 0UL};
  TilingModelInfo model_infos{model_info};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);

  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.is_inductor_scene = false;
  config.gen_tiling_data = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    (void)key;
    all_code += value;
  }
  EXPECT_NE(all_code.find("GetTilingDataRepr("), std::string::npos);
  EXPECT_NE(all_code.find("final_repr_kind = final_tiling_repr.empty() ? \"unavailable\" : \"full_json\""),
            std::string::npos);
}

TEST(GeneratorUT, TensorFlowMultiGroupFinalTilingUsesGroupRepresentation) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0UL][0UL];
  schedule_result.asc_graph_id = 0UL;
  schedule_result.impl_graph_id = 0UL;
  auto group0 = CreateModelInfo();
  group0.schedule_group_ident = {0UL, 0UL, 0UL};
  auto group1 = CreateModelInfo();
  group1.schedule_group_ident = {0UL, 0UL, 1UL};
  schedule_result.groups_tiling_model_info[0UL] = {group0};
  schedule_result.groups_tiling_model_info[1UL] = {group1};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, schedule_result.groups_tiling_model_info[0UL]),
            af::SUCCESS);
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 1UL}, schedule_result.groups_tiling_model_info[1UL]),
            af::SUCCESS);

  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    (void)key;
    all_code += value;
  }
  EXPECT_GE(CountSubstr(all_code, "GetTilingDataRepr(const AscGraph0ScheduleResult0G"), 2U);
  EXPECT_GE(CountSubstr(all_code, "final_repr_kind = final_tiling_repr.empty()"), 2U);
}

TEST(GeneratorUT, DurationSplitSourcesIncludeDirectDependencies) {
  DurationInitGuard duration_guard(1U);
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  const auto &log_header = tiling_res.at(kTilingLogHeaderIdentify);
  ExpectSystemHeaders(log_header, {"cstdint", "memory", "string"}, {"chrono", "new"});
  const auto &solver_source = tiling_res.at(kTilingSolverIdentify);
  ExpectSystemHeaders(solver_source, {"chrono", "memory", "new", "string"}, {});
}

TEST(GeneratorUT, NormalGroupRegistersOnlyDirectStandardHeaders) {
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  const auto &group_source = tiling_res.at("asc_graph0_schedule_result0_g0");
  ExpectSystemHeaders(group_source,
                      {"algorithm", "cmath", "cstdint", "map", "memory", "string", "unordered_map", "vector"},
                      {"array", "cfloat", "cstddef", "cstdlib", "cstring", "new"});
  EXPECT_NE(group_source.find("#include \"autofuse_tiling_func_api.h\""), std::string::npos);
}

TEST(GeneratorUT, PgoGroupRegistersPgoOnlyStandardHeaders) {
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.enable_autofuse_pgo = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  const auto &group_source = tiling_res.at("asc_graph0_schedule_result0_g0");
  ExpectSystemHeaders(group_source,
                      {"algorithm", "cfloat", "cmath", "cstddef", "cstdint", "cstdlib", "map", "memory", "new",
                       "string", "unordered_map", "vector"},
                      {"array", "cstring"});
}

TEST(GeneratorUT, PgoOwnershipKeepsCompleteTilingDataOutOfPgoHeader) {
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.enable_autofuse_pgo = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  const auto &data_header = tiling_res.at(config.tiling_data_type_name);
  const auto &pgo_header = tiling_res.at(kTilingPgoHeaderIdentify);
  const auto &api_header = tiling_res.at(kTilingApiHeaderIdentify);
  EXPECT_NE(data_header.find("struct AutofuseTilingDataPerf"), std::string::npos);
  EXPECT_NE(data_header.find("AutofuseTilingData tiling_data;"), std::string::npos);
  EXPECT_NE(pgo_header.find("struct PgoTensorArgs"), std::string::npos);
  EXPECT_NE(pgo_header.find("class PgoConfig"), std::string::npos);
  EXPECT_NE(pgo_header.find("struct SearchConfig"), std::string::npos);
  EXPECT_NE(pgo_header.find("struct OpTestTilingData;"), std::string::npos);
  EXPECT_NE(pgo_header.find("struct AutofuseTilingDataPerf;"), std::string::npos);
  EXPECT_EQ(pgo_header.find("AutofuseTilingData tiling_data;"), std::string::npos);
  EXPECT_EQ(pgo_header.find("#include \"autofuse_tiling_data.h\""), std::string::npos);
  EXPECT_NE(api_header.find("struct AutofuseTilingDataPerf;\nnamespace optiling {\nstruct OpTestTilingData;"),
            std::string::npos);
  EXPECT_NE(api_header.find("struct PgoTensorArgs;\nstruct SearchConfig;"), std::string::npos);
  EXPECT_NE(api_header.find("struct FinalTilingGroupSelection;"), std::string::npos);
  EXPECT_EQ(api_header.find("namespace optiling {\nstruct AutofuseTilingDataPerf;"), std::string::npos);
}

TEST(GeneratorUT, AutofuseAtomicHeadersOwnPgoConfigAndGlobalApiTypes) {
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.is_autofuse = true;
  config.is_inductor_scene = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  const auto &pgo_header = tiling_res.at(kTilingPgoHeaderIdentify);
  const auto &api_header = tiling_res.at(kTilingApiHeaderIdentify);
  EXPECT_NE(pgo_header.find("class PgoConfig"), std::string::npos);
  EXPECT_NE(pgo_header.find("class PgoConfigRuntimeGuard"), std::string::npos);
  EXPECT_NE(pgo_header.find("void *stream = nullptr;"), std::string::npos);
  EXPECT_NE(api_header.find("struct AutofuseTilingData;\nstruct AutofuseTilingDataPerf;"), std::string::npos);
  EXPECT_NE(api_header.find("struct FinalTilingGroupSelection;\nstruct AutofuseTilingData;"), std::string::npos);
  EXPECT_NE(api_header.find("bool GetWorkspaceSize(const AutofuseTilingData &tiling_data, uint64_t &workspace_size);"),
            std::string::npos);
  EXPECT_EQ(api_header.find("uint32_t GetWorkspaceSize"), std::string::npos);
  EXPECT_EQ(api_header.find("namespace optiling {\nstruct AutofuseTilingData;"), std::string::npos);
}

TEST(GeneratorUT, MultiGroupApiHeaderCollectsBodyDeclarations) {
  ModelInfo model_info = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.asc_graph_id = 0U;
  schedule_result.impl_graph_id = 0U;
  schedule_result.groups_tiling_model_info[0] = {model_info};
  schedule_result.groups_tiling_model_info[1] = {model_info};
  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    infos[0].schedule_group_ident.group_id = group_id;
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
  const auto &api_header = tiling_res.at(kTilingApiHeaderIdentify);
  EXPECT_NE(api_header.find("namespace AscGraph0ScheduleResult0G0"), std::string::npos);
  EXPECT_NE(api_header.find("namespace AscGraph0ScheduleResult0G1"), std::string::npos);
  EXPECT_EQ(api_header.find("struct AutofuseTilingData;"), std::string::npos);
  EXPECT_GE(CountSubstr(api_header, "double GetPerf("), 2U);
}

void CheckStateHeaderCacheGeneration(const std::map<std::string, std::string> &tiling_res) {
  const auto &state_header = tiling_res.at(kTilingStateHeaderIdentify);
  EXPECT_NE(state_header.find("template <typename TilingData>"), std::string::npos);
  EXPECT_NE(state_header.find("inline static thread_local std::unique_ptr<OperatorLevelCache<TilingData>>"),
            std::string::npos);
  EXPECT_EQ(state_header.find("kOperatorCacheCapacity, OpTestTilingData>"), std::string::npos);
  EXPECT_EQ(state_header.find("#include \"autofuse_tiling_data.h\""), std::string::npos);
  for (const auto &token :
       {"enum class OperatorCacheSaveResult", "kSaved", "kClearedAndSaved", "kFailed",
        "uint64_t min_count = access_counts_[0]", "last_aged_min_count_ = min_count", "GetLastAgedMinCount()"}) {
    EXPECT_NE(state_header.find(token), std::string::npos);
  }
  for (const auto &token : {"OP_LOGD", "OP_LOGI", "OP_NAME"}) {
    EXPECT_EQ(state_header.find(token), std::string::npos);
  }
  const auto &group_source = tiling_res.at("asc_graph0_schedule_result0_g0");
  for (const auto &token : {"#include <array>", "#include <cstring>", "const uint32_t request_block_dim =",
                            "const uint32_t request_ub_size =", "OperatorCacheKey operator_cache_key",
                            "[Operator Cache] HIT!", "[Operator Cache] MISS!", "OperatorCacheSaveResult::kSaved",
                            "OperatorCacheSaveResult::kClearedAndSaved", "OperatorCacheSaveResult::kFailed",
                            "[Operator Cache] SAVE SUCCESS", "[Operator Cache] CACHE CLEARED AND SAVE SUCCESS",
                            "min_count=%lu", "GetLastAgedMinCount()", "[Operator Cache] SAVE FAILED"}) {
    EXPECT_NE(group_source.find(token), std::string::npos);
  }
  EXPECT_EQ(CountSubstr(group_source, "FindOperatorCache(operator_cache_key)"), 1U);
  const auto &tail = tiling_res.at(kTilingScheduleGroupTailIdentify);
  EXPECT_EQ(tail.find("TilingCacheContext<TilingData>::"), std::string::npos);
  EXPECT_EQ(tail.find("#include"), std::string::npos);
}

TEST(GeneratorUT, StateHeaderCacheUsesUninstantiatedTilingDataTemplate) {
  unsetenv("AUTOFUSE_FLAGS");
  AutoFuseConfig::MutablePgoStrategyConfig() = PgoStrategyConfig();
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  AutoFuseConfig::MutableAttStrategyConfig().Reset();
  ASSERT_EQ(AutoFuseConfig::MutableAttStrategyConfig().Init(), af::SUCCESS);
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.cache_enabled_at_compile_time = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  CheckStateHeaderCacheGeneration(tiling_res);
  unsetenv("AUTOFUSE_DFX_FLAGS");
  AutoFuseConfig::MutableAttStrategyConfig().Reset();
}

TEST(GeneratorUT, PgoDisablesOperatorCacheEvenWhenCacheFlagIsEnabled) {
  struct ConfigGuard {
    const AttStrategyConfig original_att = AutoFuseConfig::GetAttStrategyConfig();
    const PgoStrategyConfig original_pgo = AutoFuseConfig::GetPgoStrategyConfig();
    ~ConfigGuard() {
      unsetenv("AUTOFUSE_DFX_FLAGS");
      unsetenv("AUTOFUSE_FLAGS");
      AutoFuseConfig::MutableAttStrategyConfig() = original_att;
      AutoFuseConfig::MutablePgoStrategyConfig() = original_pgo;
    }
  } config_guard;
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  AutoFuseConfig::MutableAttStrategyConfig().Reset();
  ASSERT_EQ(AutoFuseConfig::MutableAttStrategyConfig().Init(), af::SUCCESS);
  AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  ASSERT_EQ(AutoFuseConfig::MutablePgoStrategyConfig().Init(), af::SUCCESS);

  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.cache_enabled_at_compile_time = true;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl(op_name, config, model_infos, score_funcs, true);

  EXPECT_FALSE(gen_impl.config_.cache_enabled_at_compile_time);
}

TEST(GeneratorUT, InductorEnablesOperatorCacheByDefault) {
  struct ConfigGuard {
    const AttStrategyConfig original_att = AutoFuseConfig::GetAttStrategyConfig();
    const PgoStrategyConfig original_pgo = AutoFuseConfig::GetPgoStrategyConfig();
    ~ConfigGuard() {
      unsetenv("AUTOFUSE_FLAGS");
      AutoFuseConfig::MutableAttStrategyConfig() = original_att;
      AutoFuseConfig::MutablePgoStrategyConfig() = original_pgo;
    }
  } config_guard;
  unsetenv("AUTOFUSE_DFX_FLAGS");
  unsetenv("AUTOFUSE_FLAGS");
  AutoFuseConfig::MutableAttStrategyConfig().Reset();
  AutoFuseConfig::MutablePgoStrategyConfig() = PgoStrategyConfig();

  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.is_inductor_scene = true;
  config.cache_enabled_at_compile_time = false;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl(op_name, config, model_infos, score_funcs, true);

  EXPECT_TRUE(gen_impl.config_.cache_enabled_at_compile_time);
}

TEST(GeneratorUT, OperatorCacheKeyIncludesShapeAndResources) {
  ge::CodePrinter code_printer;
  cache::OperatorLevelCacheGen cache_gen;
  cache::OperatorLevelCacheGen::GenConstantDefs(code_printer, 2U);
  ASSERT_EQ(cache_gen.GenFixedSizeHashMapDef(code_printer), af::SUCCESS);
  ASSERT_EQ(cache::OperatorLevelCacheGen::GenOperatorCacheTypes(code_printer), af::SUCCESS);
  ASSERT_EQ(cache::OperatorLevelCacheGen::GenTilingCacheContext(code_printer), af::SUCCESS);
  const std::string generated = code_printer.GetOutputStr();

  EXPECT_NE(generated.find("struct OperatorCacheKey"), std::string::npos);
  EXPECT_NE(generated.find("#pragma pack(push, 1)"), std::string::npos);
  EXPECT_NE(generated.find("#pragma pack(pop)"), std::string::npos);
  EXPECT_NE(generated.find("std::array<uint32_t, kInputShapeSize> input_shapes;"), std::string::npos);
  EXPECT_NE(generated.find("uint32_t request_block_dim;"), std::string::npos);
  EXPECT_NE(generated.find("uint32_t request_ub_size;"), std::string::npos);
  EXPECT_NE(generated.find("key.input_shapes"), std::string::npos);
  EXPECT_NE(generated.find("key.request_block_dim"), std::string::npos);
  EXPECT_NE(generated.find("key.request_ub_size"), std::string::npos);
  EXPECT_NE(generated.find("FixedSizeHashMap<kInputShapeSize, kOperatorCacheCapacity, TilingData, OperatorCacheKey>"),
            std::string::npos);
}

TEST(GeneratorUT, OperatorCacheFunctionsHaveValidGeneratedSignatures) {
  ge::CodePrinter code_printer;
  ASSERT_EQ(cache::OperatorLevelCacheGen::GenOperatorCacheFunctions(code_printer, "OpTestTilingData"), af::SUCCESS);
  const std::string generated = code_printer.GetOutputStr();
  EXPECT_EQ(generated.find("OperatorCacheKey& key, )"), std::string::npos);
  EXPECT_NE(generated.find("FindOperatorCache(const OperatorCacheKey& key, OpTestTilingData& tiling_data"),
            std::string::npos);
  EXPECT_NE(generated.find("SaveOperatorCache(const OperatorCacheKey& key, const OpTestTilingData& tiling_data"),
            std::string::npos);
}

TEST(GeneratorUT, StaticShapeOperatorCacheQueriesOnceAndKeepsLogsInGroupSource) {
  TilingModelInfo model_infos(1U);
  model_infos[0].graph_name = op_name;
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "OpTestTilingData";
  config.cache_enabled_at_compile_time = true;
  ge::CodePrinter code_printer;

  ASSERT_EQ(cache::OperatorLevelCacheGen::GenInitAndQueryCacheCode(code_printer, model_infos, config), af::SUCCESS);
  ASSERT_EQ(cache::OperatorLevelCacheGen::GenSaveCacheCalls(code_printer, model_infos, config), af::SUCCESS);
  const auto group_source = code_printer.GetOutputStr();
  EXPECT_EQ(CountSubstr(group_source, "FindOperatorCache(operator_cache_key)"), 1U);
  EXPECT_NE(group_source.find("[Operator Cache] HIT! key=[]"), std::string::npos);
  EXPECT_NE(group_source.find("[Operator Cache] MISS! key=[]"), std::string::npos);
  EXPECT_NE(group_source.find("[Operator Cache] SAVE SUCCESS: key=[]"), std::string::npos);
}

TEST(GeneratorUT, OperatorCacheQueryCanContinueAfterHit) {
  TilingModelInfo model_infos(1U);
  model_infos[0].graph_name = op_name;
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "OpTestTilingData";
  config.cache_enabled_at_compile_time = true;
  ge::CodePrinter code_printer;

  ASSERT_EQ(cache::OperatorLevelCacheGen::GenInitAndQueryCacheCode(code_printer, model_infos, config, false),
            af::SUCCESS);
  const auto generated = code_printer.GetOutputStr();
  EXPECT_NE(generated.find("cache_hit = true;"), std::string::npos);
  EXPECT_EQ(generated.find("return true;"), std::string::npos);
  EXPECT_NE(generated.find("} else {"), std::string::npos);
}

TEST(GeneratorUT, GroupCacheHitFallsBackToTilingOnMiss) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.cache_enabled_at_compile_time = true;
  TilingModelInfo model_infos{CreateModelInfo(1U, ge::ExprType::kExprVariable)};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl(op_name, config, model_infos, score_funcs, false);
  gen_impl.with_reuse_info_ = true;

  ASSERT_EQ(gen_impl.GenGroupCacheLookupCode(), af::SUCCESS);
  const std::string generated = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_NE(generated.find("std::array<uint32_t, kInputShapeSize> input_shapes"), std::string::npos);
  EXPECT_NE(generated.find("if (FindGroupCache(input_shapes, tiling_data, *cache))"), std::string::npos);
  EXPECT_NE(generated.find("return true;"), std::string::npos);
  EXPECT_NE(generated.find("find no cache, turn to main tiling procedure"), std::string::npos);
}

TEST(GeneratorUT, FusedOperatorCacheSkipsExplicitTilingCase) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.cache_enabled_at_compile_time = true;
  TilingModelInfo model_infos(1U);
  model_infos[0].graph_name = op_name;
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl(op_name, config, model_infos, score_funcs, false);
  gen_impl.config_.cache_enabled_at_compile_time = true;
  std::map<size_t, std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>> namespace_map;
  namespace_map[0][0][0] = {"ScheduleResult0", "group0"};

  ASSERT_EQ(gen_impl.GenFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  const std::string generated = gen_impl.tiling_func_.GetOutputStr();
  const auto query_pos = generated.find("FindOperatorCache(operator_cache_key)");
  ASSERT_NE(query_pos, std::string::npos);
  const auto case_guard_pos = generated.find("if (tiling_case_id == -1)");
  ASSERT_NE(case_guard_pos, std::string::npos);
  EXPECT_LT(case_guard_pos, query_pos);
  const auto save_pos = generated.find("SaveOperatorCache(operator_cache_key");
  ASSERT_NE(save_pos, std::string::npos);
  EXPECT_LT(case_guard_pos, save_pos);
}

TEST(GeneratorUT, ForcedTilingCaseBypassesOperatorCache) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.cache_enabled_at_compile_time = true;
  config.force_tiling_case.is_single_mode = true;
  config.force_tiling_case.single_case = 0;
  TilingModelInfo model_infos(1U);
  model_infos[0].graph_name = op_name;
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl(op_name, config, model_infos, score_funcs, true);
  std::map<size_t, std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>> namespace_map;
  namespace_map[0][0][0] = {"ScheduleResult0", "group0"};

  ASSERT_EQ(gen_impl.GenFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  const std::string generated = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_EQ(generated.find("FindOperatorCache(operator_cache_key)"), std::string::npos);
  EXPECT_EQ(generated.find("SaveOperatorCache(operator_cache_key"), std::string::npos);
}

TEST(GeneratorUT, NormalGenerationOmitsUnusedPgoHeader) {
  TilingModelInfo model_infos{CreateModelInfo()};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  EXPECT_EQ(tiling_res.count(kTilingPgoHeaderIdentify), 0U);
}

TEST(GeneratorUT, ReuseGroupStateHeaderKeepsGroupNamespacesAndForwardDeclarations) {
  TilingModelInfo primary_group{CreateModelInfo()};
  TilingModelInfo reuse_group{CreateModelInfo()};
  primary_group[0].schedule_group_ident = {0UL, 0UL, 0UL};
  reuse_group[0].schedule_group_ident = {0UL, 0UL, 1UL};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, primary_group), af::SUCCESS);
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 1UL}, reuse_group), af::SUCCESS);
  const auto reuse_info = reuse_group[0].reuse_schedule_group->info;
  auto shared_reuse_group = primary_group[0].reuse_schedule_group;
  shared_reuse_group->schedule_group_to_info[{0UL, 0UL, 1UL}] = reuse_info;
  reuse_group[0].reuse_schedule_group = shared_reuse_group;

  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0UL][0UL];
  schedule_result.asc_graph_id = 0UL;
  schedule_result.impl_graph_id = 0UL;
  schedule_result.groups_tiling_model_info[0UL] = primary_group;
  schedule_result.groups_tiling_model_info[1UL] = reuse_group;
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.is_autofuse = true;
  config.is_inductor_scene = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
  const auto &state_header = tiling_res.at(kTilingStateHeaderIdentify);
  const auto &api_header = tiling_res.at(kTilingApiHeaderIdentify);
  EXPECT_NE(state_header.find("struct AscGraph0ScheduleResult0G0TilingData;\nnamespace optiling {\nnamespace "
                              "AscGraph0ScheduleResult0G0 {"),
            std::string::npos);
  EXPECT_NE(state_header.find("struct AscGraph0ScheduleResult0G1TilingData;\nnamespace optiling {\nnamespace "
                              "AscGraph0ScheduleResult0G1 {"),
            std::string::npos);
  EXPECT_NE(api_header.find("struct AscGraph0ScheduleResult0G0TilingData;"), std::string::npos);
  EXPECT_NE(api_header.find("struct AscGraph0ScheduleResult0G1TilingData;"), std::string::npos);
  EXPECT_NE(api_header.find("int32_t tiling_case_id, AutofuseTilingData* output_tiling_data"), std::string::npos);
  EXPECT_NE(state_header.find("#include <unordered_map>"), std::string::npos);
  EXPECT_NE(state_header.find("AscGraph0ScheduleResult0G0::GroupLevelCache* cache"), std::string::npos);
  EXPECT_EQ(state_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_NE(tiling_res.at(kTilingScheduleGroupTailIdentify).find("#include <cfloat>"), std::string::npos);
  const auto &reuse_source = tiling_res.at("asc_graph0_schedule_result0_g1");
  ExpectSystemHeaders(reuse_source, {"cstdint", "unordered_map"},
                      {"algorithm", "array", "cfloat", "cmath", "cstddef", "cstdlib", "cstring", "map", "memory", "new",
                       "string", "vector"});
  EXPECT_NE(reuse_source.find("#include \"autofuse_tiling_data.h\""), std::string::npos);
  EXPECT_NE(reuse_source.find("#include \"autofuse_tiling_func_state.h\""), std::string::npos);
  EXPECT_NE(reuse_source.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
  EXPECT_NE(reuse_source.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
  EXPECT_NE(reuse_source.find("#include \"autofuse_tiling_func_api.h\""), std::string::npos);
  EXPECT_EQ(reuse_source.find("#include \"autofuse_tiling_func_pgo.h\""), std::string::npos);
}

TEST(GeneratorUT, ReuseGroupFinalTilingSelectionIncludesCompleteType) {
  TilingModelInfo primary_group{CreateModelInfo()};
  TilingModelInfo reuse_group{CreateModelInfo()};
  primary_group[0].schedule_group_ident = {0UL, 0UL, 0UL};
  reuse_group[0].schedule_group_ident = {0UL, 0UL, 1UL};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, primary_group), af::SUCCESS);
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 1UL}, reuse_group), af::SUCCESS);
  auto shared_reuse_group = primary_group[0].reuse_schedule_group;
  shared_reuse_group->schedule_group_to_info[{0UL, 0UL, 1UL}] = reuse_group[0].reuse_schedule_group->info;
  reuse_group[0].reuse_schedule_group = shared_reuse_group;

  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0UL][0UL];
  schedule_result.asc_graph_id = 0UL;
  schedule_result.impl_graph_id = 0UL;
  schedule_result.groups_tiling_model_info[0UL] = primary_group;
  schedule_result.groups_tiling_model_info[1UL] = reuse_group;
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.is_autofuse = true;
  config.is_inductor_scene = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
  const auto &reuse_source = tiling_res.at("asc_graph0_schedule_result0_g1");
  // The reuse wrapper assigns FinalTilingGroupSelection members, so its split translation unit must see
  // the complete definition from the log header instead of the api-header forward declaration.
  EXPECT_NE(reuse_source.find("bool GetFinalTilingSelection("), std::string::npos);
  EXPECT_NE(reuse_source.find("selection.group = 1U;"), std::string::npos);
  EXPECT_NE(reuse_source.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
}

TEST(GeneratorUT, WorkspaceReuseRelationIsOmittedFromGroupCache) {
  TilingModelInfo primary_group{CreateModelInfo()};
  TilingModelInfo reuse_group{CreateModelInfo()};
  primary_group[0].schedule_group_ident = {0UL, 0UL, 0UL};
  reuse_group[0].schedule_group_ident = {0UL, 0UL, 1UL};
  primary_group[0].workspace_size_map[1] = CreateExpr(1);
  reuse_group[0].workspace_size_map[0] = CreateExpr(1);
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, primary_group), af::SUCCESS);
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 1UL}, reuse_group), af::SUCCESS);
  auto shared_reuse_group = primary_group[0].reuse_schedule_group;
  shared_reuse_group->schedule_group_to_info[{0UL, 0UL, 1UL}] = reuse_group[0].reuse_schedule_group->info;
  reuse_group[0].reuse_schedule_group = shared_reuse_group;

  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0UL][0UL];
  schedule_result.asc_graph_id = 0UL;
  schedule_result.impl_graph_id = 0UL;
  schedule_result.groups_tiling_model_info[0UL] = primary_group;
  schedule_result.groups_tiling_model_info[1UL] = reuse_group;
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.is_autofuse = true;
  config.is_inductor_scene = true;
  std::map<std::string, std::string> tiling_res;

  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  const auto &primary_source = tiling_res.at("asc_graph0_schedule_result0_g0");
  const auto &reuse_source = tiling_res.at("asc_graph0_schedule_result0_g1");
  EXPECT_EQ(primary_source.find("GroupLevelCache"), std::string::npos);
  EXPECT_EQ(reuse_source.find("GroupLevelCache"), std::string::npos);
  EXPECT_EQ(primary_source.find("* cache"), std::string::npos);
  EXPECT_EQ(reuse_source.find("* cache"), std::string::npos);

  const auto &tail_source = tiling_res.at(kTilingScheduleGroupTailIdentify);
  EXPECT_EQ(tail_source.find("AscGraph0ScheduleResult0G0::GroupLevelCache"), std::string::npos);
  EXPECT_EQ(tail_source.find("AscGraph0ScheduleResult0G0_Cache"), std::string::npos);
  EXPECT_EQ(tail_source.find("AscGraph0ScheduleResult0G1::GroupLevelCache"), std::string::npos);
  EXPECT_EQ(tail_source.find("AscGraph0ScheduleResult0G1_Cache"), std::string::npos);
}

TEST(GeneratorUT, ReuseGroupPgoCopiesPrimaryGroupTiling) {
  TilingModelInfo primary_group{CreateModelInfo()};
  TilingModelInfo reuse_group{CreateModelInfo()};
  primary_group[0].schedule_group_ident = {0UL, 0UL, 0UL};
  reuse_group[0].schedule_group_ident = {0UL, 0UL, 1UL};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, primary_group), af::SUCCESS);
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 1UL}, reuse_group), af::SUCCESS);
  auto shared_reuse_group = primary_group[0].reuse_schedule_group;
  shared_reuse_group->schedule_group_to_info[{0UL, 0UL, 1UL}] = reuse_group[0].reuse_schedule_group->info;
  reuse_group[0].reuse_schedule_group = shared_reuse_group;

  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0UL][0UL];
  schedule_result.asc_graph_id = 0UL;
  schedule_result.impl_graph_id = 0UL;
  schedule_result.groups_tiling_model_info[0UL] = primary_group;
  schedule_result.groups_tiling_model_info[1UL] = reuse_group;
  TilingCodeGenConfig config;
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.is_autofuse = true;
  config.is_inductor_scene = true;
  config.enable_autofuse_pgo = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
  const auto &api_header = tiling_res.at(kTilingApiHeaderIdentify);
  const auto &reuse_source = tiling_res.at("asc_graph0_schedule_result0_g1");
  EXPECT_NE(api_header.find("AutofuseTilingData* output_tiling_data"), std::string::npos);
  EXPECT_EQ(api_header.find("AscGraph0ScheduleResult0G1TilingData* output_tiling_data"), std::string::npos);
  EXPECT_NE(reuse_source.find("AutofuseTilingData* output_tiling_data"), std::string::npos);
  const auto &tail_source = tiling_res.at(kTilingScheduleGroupTailIdentify);
  EXPECT_NE(tail_source.find("RefToRef<AscGraph0ScheduleResult0G0TilingData, "
                             "AscGraph0ScheduleResult0G1TilingData>("
                             "tiling_data.graph0_result0_g0_tiling_data)"),
            std::string::npos);
  const auto pgo_by_core_num_begin = tail_source.find("bool GetScheduleResult0PGOByCoreNum");
  const auto pgo_by_core_num_end = tail_source.find("using ScheduleResultFunctionPGOByCoreNum", pgo_by_core_num_begin);
  ASSERT_NE(pgo_by_core_num_begin, std::string::npos);
  ASSERT_NE(pgo_by_core_num_end, std::string::npos);
  EXPECT_EQ(tail_source.substr(pgo_by_core_num_begin, pgo_by_core_num_end - pgo_by_core_num_begin)
                .find("AscGraph0ScheduleResult0G1::GetTiling"),
            std::string::npos);
}

TEST(GeneratorUT, Normal) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, NormalStaticUint32Shape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, af::ExprType::kExprConstantInteger);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  EXPECT_EQ(tiling_res.find("TilingTailHeader"), tiling_res.end());
  const auto &tail = tiling_res.at(kTilingScheduleGroupTailIdentify);
  EXPECT_EQ(tail.find("#include"), std::string::npos);
}

TEST(GeneratorUT, NormalStaticRationShape) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo(1, af::ExprType::kExprConstantRation);
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  TilingCodeGenerator generator;
  std::map<size_t, std::map<size_t, std::vector<ModelInfo>>> model_infos_new;
  model_infos_new[0][0] = model_infos;
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  ASSERT_EQ(tiling_res.size(), 8);
}

TEST(GeneratorUT, GenTilingSolverSuccess) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo = CreateModelInfo();
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, InvalidConfig) {
  TilingModelInfo model_infos;
  ModelInfo modelInfo;
  model_infos.emplace_back(modelInfo);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::MAX;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_NE(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, TestSymengine) {
  using namespace SymEngine;
  using SymEngine::Basic;
  using SymEngine::make_rcp;
  using SymEngine::RCP;
  using SymEngine::Symbol;
  const RCP<const Basic> x = make_rcp<SymEngine::Symbol>("x");
  EXPECT_EQ(x->__str__(), "x");
}

TEST(GeneratorUT, TestSymengine2) {
  using namespace SymEngine;
  RCP<const Basic> x1 = symbol("x1");
  RCP<const Basic> x2 = symbol("x2");
  RCP<const Basic> int1 = integer(1);
  RCP<const Basic> int2 = integer(2);
  RCP<const Basic> y = mul(x2, add(x1, int1));
  RCP<const Basic> z = mul(add(int1, x1), x2);
  EXPECT_EQ(x1->__str__(), "x1");
  EXPECT_EQ(x2->__str__(), "x2");
  EXPECT_EQ(y->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(z->__str__(), "x2*(1 + x1)");
  EXPECT_EQ(is_a<Symbol>(*x1), true);
  EXPECT_EQ(is_a<Symbol>(*x2), true);
  EXPECT_EQ(is_a<Symbol>(*y), false);
  EXPECT_EQ(is_a<Symbol>(*z), false);
  EXPECT_EQ(is_a<Integer>(*int1), true);
  RCP<const Basic> multi_add = add(add(int1, x1), int2);
  EXPECT_EQ(multi_add->__str__(), "3 + x1");
  RCP<const Basic> m = add(mul(add(int1, x1), x2), int2);
  EXPECT_EQ(m->get_args()[0]->__str__(), "2");
  EXPECT_EQ(m->get_args()[1]->__str__(), "x2*(1 + x1)");
}

TEST(GeneratorUT, AddElementInTilingData) {
  ge::CodePrinter dumper;
  TilingDataGenUtils::AddStructElementDefinition(dumper, "TCubeTiling", "mm_tiling");
  EXPECT_TRUE(dumper.GetOutputStr().find("TCubeTiling, mm_tiling") != std::string::npos);
}

TEST(GeneratorUT, TestSchedGroup) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), af::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  all_model_infos[0].var_relations[1][0]["m_size"] = CreateExpr("m_size") + CreateExpr(1);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
  const auto &tail = tiling_res.at(kTilingScheduleGroupTailIdentify);
  EXPECT_NE(tail.find("#include \"OpTest_tiling_data.h\""), std::string::npos);
  EXPECT_NE(tail.find("#include \"autofuse_tiling_func_state.h\""), std::string::npos);
  EXPECT_NE(tail.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
  EXPECT_NE(tail.find("#include \"autofuse_tiling_func_api.h\""), std::string::npos);
  EXPECT_NE(tail.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
  EXPECT_NE(tail.find("#include <algorithm>"), std::string::npos);
  EXPECT_NE(tail.find("#include <array>"), std::string::npos);
  EXPECT_NE(tail.find("#include <cmath>"), std::string::npos);
  EXPECT_NE(tail.find("#include <functional>"), std::string::npos);
  EXPECT_NE(tail.find("#include <limits>"), std::string::npos);
  EXPECT_NE(tail.find("#include <utility>"), std::string::npos);
  EXPECT_NE(tail.find("!std::isfinite("), std::string::npos);
}

TEST(GeneratorUT, TestSchedGroupEnableGroupParallel) {
  ModelInfo modelInfo = CreateModelInfo();
  FusedParsedScheduleResult fused_schedule_result;
  auto &all_model_infos = fused_schedule_result[0];
  std::map<size_t, std::vector<ModelInfo>> model_infos1;

  model_infos1[0] = {modelInfo, modelInfo};
  model_infos1[0][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][0].schedule_group_ident.group_id = 0;
  model_infos1[0][0].tiling_case_id = 0;
  model_infos1[0][0].enable_group_parallel = true;
  model_infos1[0][1].schedule_group_ident.impl_graph_id = 0;
  model_infos1[0][1].schedule_group_ident.group_id = 0;
  model_infos1[0][1].tiling_case_id = 1;
  model_infos1[0][1].enable_group_parallel = true;

  model_infos1[1] = {modelInfo};
  model_infos1[1][0].schedule_group_ident.impl_graph_id = 0;
  model_infos1[1][0].schedule_group_ident.group_id = 1;
  model_infos1[1][0].tiling_case_id = 2;
  model_infos1[1][0].enable_group_parallel = true;
  for (auto &model_info : model_infos1) {
    EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_info.second), af::SUCCESS);
  }
  all_model_infos[0].groups_tiling_model_info = model_infos1;
  all_model_infos[0].impl_graph_id = 0;
  all_model_infos[0].enable_group_parallel = true;

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = true;
  config.gen_extra_infos = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  EXPECT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);
  const auto &tail = tiling_res.at(kTilingScheduleGroupTailIdentify);
  EXPECT_EQ(tail.find("#include <cmath>"), std::string::npos);
  EXPECT_EQ(tail.find("#include <limits>"), std::string::npos);
  bool flag_arrange = false;
  bool flag_parallel = false;
  for (const auto &[key, value] : tiling_res) {
    if (value.find("  ArrangeBlockOffsetsAscGraph0Result0(") != std::string::npos) {
      flag_arrange = true;
      EXPECT_NE(value.find("  uint32_t actual_max_block_dim = 0U;"), std::string::npos);
    }
    if (value.find("UpdateCurPerfAndBlockByGroup(") != std::string::npos) {
      flag_parallel = true;
    }
  }
  EXPECT_EQ(flag_arrange && flag_parallel, true);
}

TEST(GeneratorUT, CreateAxesReorderTilingCodeGenImplSuccess) {
  TilingModelInfo model_infos;
  model_infos.emplace_back(CreateModelInfo());
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::AXES_REORDER;
  config.gen_extra_infos = true;
  TilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);
}

TEST(GeneratorUT, AxesReorderSolverHeaderRegistersOnlyDirectStandardHeaders) {
  ModelInfo model_info = CreateModelInfo();
  size_t order = 1U;
  for (const auto &arg : model_info.arg_list) {
    if (arg->name == "stepm" || arg->name == "stepn") {
      // R2 剥离后 INNER 轴仅剩 stepm/stepn；置 bind_multicore=false + 同 order 构造 equal-order 触发条件
      arg->bind_multicore = false;
      arg->order = 0U;
    } else {
      arg->order = order++;
    }
  }
  TilingModelInfo model_infos{model_info};
  ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  TilingCodeGenConfig config;
  config.type = TilingImplType::AXES_REORDER;
  config.is_inductor_scene = true;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;

  ASSERT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);
  const auto &solver_header = tiling_res.at(kTilingSolverHeaderIdentify);
  ExpectSystemHeaders(
      solver_header,
      {"cmath", "cstddef", "cstdint", "functional", "sstream", "string", "type_traits", "utility", "vector"},
      {"algorithm", "array", "map", "memory", "set", "unordered_map"});
  const auto &solver_source = tiling_res.at(kTilingSolverIdentify);
  EXPECT_NE(solver_source.find("#include <limits>"), std::string::npos);
  EXPECT_NE(solver_source.find("#include <map>"), std::string::npos);
  EXPECT_NE(solver_source.find("lcm(info.var_a->align, info.var_b->align)"), std::string::npos);
  EXPECT_EQ(solver_source.find("::lcm(info.var_a->align, info.var_b->align)"), std::string::npos);
}

TEST(GeneratorUT, TilingCodeGenImplConstruct) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  MockHighPerfTilingCodeGenImpl impl("test", config, tiling_model_info, score_funcs, true);
  EXPECT_EQ(config.force_template_op_name, "test");
  impl.GenGetAllSchedulesResults({});
  EXPECT_EQ(impl.tiling_func_.GetOutputStr().empty(), true);
}

TEST(GeneratorUT, GenVariableAnnotationShowsReduceBreakdownAndContribSemantics) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  ModelInfo model_info = CreateModelInfo();
  Expr contrib_var = CreateExpr("Min_AIV_VEC_core_contrib");
  TernaryOp contrib_op(CreateExpr("reduce_total_perf") * CreateExpr("core_exe_time"));
  contrib_op.SetVariable(contrib_var);
  contrib_op.SetDescription("AIV_VEC core contribution = node API perf * core exe time");
  model_info.ternary_op_map[contrib_var] = contrib_op;
  model_info.perf_breakdowns = {
      {"Min Reduce API",
       {{"reduce_body_perf", CreateExpr("reduce_body_perf"), "Reduce API body perf", 0U},
        {"reduce_total_perf", CreateExpr("reduce_total_perf"), "Reduce API total perf = body + merge", 0U}}}};
  tiling_model_info.push_back(model_info);
  MockHighPerfTilingCodeGenImpl impl("test", config, tiling_model_info, score_funcs, true);
  ArgsManager args_manager(model_info);
  ASSERT_TRUE(args_manager.Process(false));
  EXPECT_EQ(impl.GenVariableAnnotation(args_manager), af::SUCCESS);

  const std::string tiling_func_output = impl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("Reduce perf breakdown used for tiling case 0"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("reduce_total_perf: Reduce API total perf = body + merge"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("AIV_VEC core contribution = node API perf * core exe time"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("core_exe_time * reduce_total_perf"), std::string::npos);
}

TEST(GeneratorUT, TilingCodeGenImplPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  EXPECT_EQ(genImpl.GenTilingImplPublicFunc(), af::SUCCESS);

  std::string expectCode = R"rawliteral(  bool GetTiling(TilingData &tiling_data) {
    OP_LOGD(OP_NAME, "Execute DoTiling.");
    if (!DoTiling(tiling_data)) {
      OP_LOGW(OP_NAME, "Failed to do tiling.");
      return false;
    }
    if (is_empty_tensor_) {
      OP_LOGW(OP_NAME, "Empty tensor, skip DoApiTiling and GeneralTiling.");
      return true;
    }
    DoApiTiling(tiling_data);
    GeneralTiling(tiling_data);
    if (tiling_data.get_block_dim() > corenum_) {
      OP_LOGW(OP_NAME, "Generated block_dim %u exceeds core budget %u.", tiling_data.get_block_dim(), corenum_);
      return false;
    }
    TilingSummary(tiling_data);
    return true;
  }
  virtual double GetPerf(TilingData &tiling_data) { (void)tiling_data; return 0.0; }
  virtual const char* GetScheduleName() { return ""; }
  virtual void TilingSummary(TilingData &tiling_data) = 0;
  virtual bool ExecutePGOSolver(TilingData &tiling_data, std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData* autofuse_tiling_data, void* stream, std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {
    (void)tiling_data; (void)tiling_data_list; (void)autofuse_tiling_data; (void)stream; (void)workspace_map; (void)block_dim_vec; (void)search_cfg;
    return false;
  }
  virtual int32_t CalcScore(const TilingData &tiling_data) { (void)tiling_data; return 0;}
  virtual void GetTilingData(TilingDataCopy &from_tiling, TilingData &to_tiling) { (void)from_tiling; (void)to_tiling; }
  virtual void SetTilingData(TilingData &from_tiling, TilingDataCopy &to_tiling) { (void)from_tiling; (void)to_tiling; }
  virtual bool SetWorkspaceSize(TilingData &tiling_data, std::unordered_map<int64_t, uint64_t> &workspace_map) { (void)tiling_data; (void)workspace_map; return true; }
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);
}

TEST(GeneratorUT, GenTilingHeadPGO) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);

  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.GenTilingHead(tiling_res, enable_group_parallels);
  std::string expectCode = R"rawliteral(namespace optiling {

} // namespace optiling
)rawliteral";
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), expectCode);
}

TEST(GeneratorUT, GenScheduleGroupTilingTailPGOSuccess) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.gen_tiling_data = false;
  genImpl.enable_group_parallels_ = enable_group_parallels;
  EXPECT_EQ(genImpl.GenScheduleGroupTilingTail(), af::SUCCESS);

  EXPECT_EQ(genImpl.tiling_func_.GetOutputStr().empty(), false);
}

TEST(GeneratorUT, GenTilingPGOSuccess) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  ReuseScheduleGroup reuse_schedule_group;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.gen_tiling_data = false;
  EXPECT_EQ(genImpl.GenTiling(tiling_res, {}, 0, enable_group_parallels), af::SUCCESS);

  EXPECT_EQ(genImpl.tiling_func_.GetOutputStr().empty(), false);
}

TEST(GeneratorUT, PGOByCoreNumSingleGroupUsesGetTilingKey) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.enable_autofuse_pgo = true;

  ASSERT_EQ(genImpl.GenPGOByCoreNumSearchTilingKey(), af::SUCCESS);
  const std::string output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(output.find("GetTilingKey(*tiling_data, tiling_case)"), std::string::npos);
  EXPECT_EQ(output.find("GetTilingCore(*tiling_data, tiling_case"), std::string::npos);
}

TEST(GeneratorUT, RootGetTilingFailuresUseWarningLogOnlyForPGOPath) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ModelInfo info;
  tiling_model_info.push_back(info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  genImpl.config_.cache_enabled_at_compile_time = false;
  std::map<size_t, std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>> namespace_map;
  namespace_map[0][0] = {};

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("OP_LOGE(OP_NAME, \"Aggregated block_dim %u exceeds core budget %u.\""),
            std::string::npos);

  genImpl.config_.is_inductor_scene = true;
  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"Aggregated block_dim %u exceeds core budget %u.\""),
            std::string::npos);

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("tiling_data->set_block_dim(block_dim_i);"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"Failed to get tiling of AscGraph0.\");"), std::string::npos);
}

TEST(GeneratorUT, PGOByCoreNumNormalizesSharedGroupBlockDim) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  std::map<size_t, std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>> namespace_map;
  namespace_map[0][0][0] = {"ScheduleResult0", "group0"};
  namespace_map[0][0][1] = {"ScheduleResult0", "group1"};
  genImpl.enable_group_parallels_[0][0] = true;

  EXPECT_EQ(genImpl.GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  const std::string output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(output.find("tiling_data->set_block_dim(block_dim_i);"), std::string::npos);
  EXPECT_NE(output.find("uint32_t result_block_dim = 0U;"), std::string::npos);
  EXPECT_NE(output.find("result_block_dim = std::min(result_block_dim, block_dim_i);"), std::string::npos);
  EXPECT_NE(output.find("total_block_dim = std::max(total_block_dim, result_block_dim);"), std::string::npos);
}

TEST(GeneratorUT, PGOByCoreNumSerialGroupsUseMaximumBlockDim) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  std::map<size_t, std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>> namespace_map;
  namespace_map[0][0][0] = {"ScheduleResult0", "group0"};
  namespace_map[0][0][1] = {"ScheduleResult0", "group1"};

  EXPECT_EQ(genImpl.GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(namespace_map), af::SUCCESS);
  const std::string output = genImpl.tiling_func_.GetOutputStr();
  // This entry point emits the root candidate collection; schedule-result block_dim updates are generated separately.
  EXPECT_NE(
      output.find("result_block_dim = std::max(result_block_dim, tiling_data_tmp.group1_tiling_data.get_block_dim());"),
      std::string::npos);
  EXPECT_EQ(output.find("result_block_dim = std::min(result_block_dim, block_dim_i);"), std::string::npos);
  EXPECT_EQ(output.find("result_block_dim += tiling_data_tmp.group1_tiling_data.get_block_dim();"), std::string::npos);
  EXPECT_NE(output.find("ret = !tiling_data_list.empty();"), std::string::npos);
}

TEST(GeneratorUT, GetTilingRejectsBlockDimAboveCoreBudget) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  ModelInfo model_info;
  model_info.hardware_cons[HardwareDef::CORENUM] = CreateExpr("128");
  tiling_model_info.push_back(model_info);

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  ASSERT_EQ(genImpl.GenGetTiling(), af::SUCCESS);
  const std::string output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(output.find("tiling_data.get_block_dim() > corenum_"), std::string::npos);
}

TEST(GeneratorUT, GetTilingPropagatesScheduleSummaryFailure) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ModelInfo model_info;
  tiling_model_info.push_back(model_info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> namespace_map;
  namespace_map[0] = {};

  ASSERT_EQ(genImpl.GenGetTilingForAllSchedulesResults(0, namespace_map), af::SUCCESS);
  const std::string output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(output.find("const bool result = GetResultSummary(best_perf, tiling_data);"), std::string::npos);
  EXPECT_NE(output.find("return result;"), std::string::npos);
}

TEST(GeneratorUT, PGOGetTilingKeyFailureUsesWarningLog) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  ModelInfo info;
  info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(info);

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.config_.is_inductor_scene = true;
  genImpl.config_.cache_enabled_at_compile_time = false;
  EXPECT_EQ(genImpl.GenGetTilingKeyCall(""), af::SUCCESS);

  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
}

void AssertOperatorCacheSaveAndPerfGuards(const std::string &generated) {
  const auto save_guard_body = ExtractGuardBody(generated, "if (ret) {");
  EXPECT_NE(save_guard_body.find("const auto cache_save_result ="), std::string::npos);
  const auto perf_guard_body = ExtractGuardBody(generated, "if (ret && perf != nullptr) {");
  EXPECT_NE(perf_guard_body.find("*perf = GetPerf(tiling_data);"), std::string::npos);

  EXPECT_NE(generated.find("bool cache_hit = false;"), std::string::npos);
  EXPECT_NE(generated.find("cache_hit = true;"), std::string::npos);
  EXPECT_NE(generated.find("if (tiling_case_id == -1)"), std::string::npos);
  EXPECT_NE(generated.find("if (tiling_case_id == -1 && !cache_hit)"), std::string::npos);
  const auto return_pos = generated.find("return ret;");
  ASSERT_NE(return_pos, std::string::npos);
  for (const auto &duration_code :
       {"DurationEnd", "DurationManager::GetInstance().Print()", "DurationManager::GetInstance().Clear()"}) {
    const auto duration_pos = generated.find(duration_code);
    ASSERT_NE(duration_pos, std::string::npos);
    EXPECT_LT(duration_pos, return_pos);
  }
}

TEST(GeneratorUT, OperatorCacheSaveAndPerfAreGuardedByTilingSuccess) {
  unsetenv("AUTOFUSE_FLAGS");
  AutoFuseConfig::MutablePgoStrategyConfig() = PgoStrategyConfig();
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  struct CacheConfigGuard {
    ~CacheConfigGuard() {
      unsetenv("AUTOFUSE_DFX_FLAGS");
      AutoFuseConfig::MutableAttStrategyConfig().Reset();
    }
  } cache_config_guard;
  AutoFuseConfig::MutableAttStrategyConfig().Reset();
  ASSERT_EQ(AutoFuseConfig::MutableAttStrategyConfig().Init(), af::SUCCESS);
  AutoFuseConfig::MutableAttStrategyConfig().enable_tiling_cache = "true";
  AutoFuseConfig::MutableAttStrategyConfig().set_env_enable_tiling_cache = true;
  struct DurationLevelGuard {
    decltype(kg_duration_level) &level;
    decltype(kg_duration_level) old_level;
    ~DurationLevelGuard() {
      level = old_level;
    }
  } duration_level_guard{kg_duration_level, kg_duration_level};
  kg_duration_level = 2U;
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "OpTestTilingData";
  config.cache_enabled_at_compile_time = true;
  TilingModelInfo tiling_model_info;
  ModelInfo info;
  info.graph_name = op_name;
  info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl(op_name, config, tiling_model_info, score_funcs, true);
  ASSERT_TRUE(gen_impl.config_.cache_enabled_at_compile_time);

  ASSERT_EQ(gen_impl.GenGetTilingFunctionBody(false, false, ""), af::SUCCESS);
  const std::string generated = gen_impl.tiling_func_.GetOutputStr();
  AssertOperatorCacheSaveAndPerfGuards(generated);
}

TEST(GeneratorUT, GetResultSummaryFailureUsesWarningLogForPGOPath) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ModelInfo info;
  tiling_model_info.push_back(info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);

  EXPECT_EQ(genImpl.GenGetResultSummary(0), af::SUCCESS);
  std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);

  genImpl.config_.enable_autofuse_pgo = true;
  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenGetResultSummary(0), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);

  genImpl.config_.enable_autofuse_pgo = false;
  genImpl.config_.is_inductor_scene = true;
  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenGetResultSummary(0), af::SUCCESS);
  tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("OP_LOGW(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("OP_LOGE(OP_NAME, \"GetTiling Failed.\");"), std::string::npos);
}

TEST(GeneratorUT, PGOGetAllSchedulesResultsDoesNotPushGraphTilingTmpOutsideScheduleResult) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.enable_autofuse_pgo = true;
  config.is_inductor_scene = true;
  TilingModelInfo tiling_model_info;
  ModelInfo info;
  tiling_model_info.push_back(info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> namespace_map;
  namespace_map[0] = {};
  namespace_map[1] = {};

  genImpl.GenPGOGetAllSchedulesResults(0, namespace_map);

  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(tiling_func_output.find("if (!AscGraph0::GetTiling(tilingTmp, index, nullptr)) {"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("cur_perf = DBL_MAX;"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("continue;"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("tiling_perf.tiling_data = tilingTmp;"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("tiling_data_list.push_back(tiling_perf);"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("PgoConfig::Instance().single_callback("), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("*tilingData = tilingTmp;"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("bool has_valid_tiling = false;"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("has_valid_tiling = true;"), std::string::npos);
}

TEST(GeneratorUT, PGOCallbacksUseUint64WorkspaceSize) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.enable_autofuse_pgo = true;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, false);
  ge::CodePrinter pgo_header;

  genImpl.GenPgoCallbackDefs(pgo_header);

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult0", "group1");
  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");
  genImpl.tiling_func_.Reset();
  ASSERT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);

  const std::string callback_output = pgo_header.GetOutputStr();
  const std::string schedule_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_NE(callback_output.find("void* stream, uint64_t workspaceSize"), std::string::npos);
  EXPECT_EQ(callback_output.find("void* stream, uint32_t workspaceSize"), std::string::npos);
  EXPECT_NE(schedule_output.find("void* stream, uint64_t workspaceSize"), std::string::npos);
  EXPECT_EQ(schedule_output.find("void* stream, uint32_t workspaceSize"), std::string::npos);
}

static const std::string kExpectPGOCode =
    R"rawliteral(inline bool GetScheduleResult0PGO(std::vector<AutofuseTilingDataPerf>& tiling_data_list, const uint32_t ori_block_dim, const int32_t tiling_case_id,AutofuseTilingData &tiling_data, double &cur_perf, double &best_perf, uint32_t &cur_block_dim,void* stream, uint64_t workspaceSize, std::vector<uint32_t*> multi_group_block_dim_list = {}, const SearchConfig *search_cfg=nullptr) {
  (void)cur_perf; (void)cur_block_dim;
  std::vector<AutofuseTilingDataPerf> tiling_data_list_tmp{};
  workspaceSize = 0;
  std::unordered_map<int64_t, uint64_t> workspace_map_filter_use{};
  tiling_data.set_graph0_tiling_key(0);
  auto &group0_tiling_data = tiling_data.group0_tiling_data;
  group0_tiling_data.set_block_dim(ori_block_dim);
  size_t candidate_begin_index0 = tiling_data_list_tmp.size();
  auto result0 = ScheduleResult0::PGOSearchTilingKey(tiling_data_list_tmp, group0_tiling_data, tiling_case_id, &tiling_data, PgoConfig::Instance().tensor_args, stream, workspaceSize, best_perf, workspace_map_filter_use, multi_group_block_dim_list, search_cfg);
  if (result0) {
    bool has_solution = true;
    std::vector<bool> valid_candidates(tiling_data_list_tmp.size() - candidate_begin_index0, true);
    for (size_t candidate_index = candidate_begin_index0; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      auto &tiling_data = tiling_data_perf.tiling_data;
      std::unordered_map<int64_t, uint64_t> workspace_map;
      workspace_map.reserve(workspace_map_filter_use.size());
      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());
      tiling_data.group1_tiling_data.set_block_dim(ori_block_dim);
      has_solution = ScheduleResult0::GetTiling(tiling_data.group1_tiling_data, workspace_map, -1);
      if (!has_solution) {
        OP_LOGI(OP_NAME, "No solution for group0 at group1");
        valid_candidates[candidate_index - candidate_begin_index0] = false;
        continue;
      }
      uint32_t max_block_dim = tiling_data.group0_tiling_data.get_block_dim();
      max_block_dim = Max(max_block_dim, tiling_data.group1_tiling_data.get_block_dim());
      tiling_data.set_block_dim(max_block_dim);
      uint64_t workspaceSizeTmp = 0U;
      if (!GetWorkspaceSize(tiling_data, workspaceSizeTmp)) {
        return false;
      }
      if (workspaceSizeTmp > workspaceSize) {
        workspaceSize = workspaceSizeTmp;
      }
    }
    if (__builtin_add_overflow(workspaceSize, 16ULL * 1024ULL * 1024ULL, &workspaceSize)) {
      OP_LOGE(OP_NAME, "Workspace size overflow in PGO group profiling.");
      return false;
    }
    if (PgoConfig::Instance().batch_callback) {
      if (PgoConfig::Instance().batch_callback(PgoConfig::Instance().tensor_args, stream, workspaceSize, &tiling_data_list_tmp) != 0) {
        return false;
      }
    }
    for (size_t candidate_index = candidate_begin_index0; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      const size_t candidate_offset = candidate_index - candidate_begin_index0;
      if (candidate_offset >= valid_candidates.size() || !valid_candidates[candidate_offset]) {
        continue;
      }
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      tiling_data_list.push_back(tiling_data_perf);
      if (tiling_data_perf.best_perf < best_perf) {
        tiling_data = tiling_data_perf.tiling_data;
        best_perf = tiling_data_perf.best_perf;
      }
    }
  }
  auto &group1_tiling_data = tiling_data.group1_tiling_data;
  group1_tiling_data.set_block_dim(ori_block_dim);
  size_t candidate_begin_index1 = tiling_data_list_tmp.size();
  auto result1 = ScheduleResult0::PGOSearchTilingKey(tiling_data_list_tmp, group1_tiling_data, tiling_case_id, &tiling_data, PgoConfig::Instance().tensor_args, stream, workspaceSize, best_perf, workspace_map_filter_use, multi_group_block_dim_list, search_cfg);
  if (result1) {
    bool has_solution = true;
    std::vector<bool> valid_candidates(tiling_data_list_tmp.size() - candidate_begin_index1, true);
    for (size_t candidate_index = candidate_begin_index1; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      auto &tiling_data = tiling_data_perf.tiling_data;
      std::unordered_map<int64_t, uint64_t> workspace_map;
      workspace_map.reserve(workspace_map_filter_use.size());
      workspace_map.insert(workspace_map_filter_use.begin(), workspace_map_filter_use.end());
      uint32_t max_block_dim = tiling_data.group0_tiling_data.get_block_dim();
      max_block_dim = Max(max_block_dim, tiling_data.group1_tiling_data.get_block_dim());
      tiling_data.set_block_dim(max_block_dim);
      uint64_t workspaceSizeTmp = 0U;
      if (!GetWorkspaceSize(tiling_data, workspaceSizeTmp)) {
        return false;
      }
      if (workspaceSizeTmp > workspaceSize) {
        workspaceSize = workspaceSizeTmp;
      }
    }
    if (__builtin_add_overflow(workspaceSize, 16ULL * 1024ULL * 1024ULL, &workspaceSize)) {
      OP_LOGE(OP_NAME, "Workspace size overflow in PGO group profiling.");
      return false;
    }
    if (PgoConfig::Instance().batch_callback) {
      if (PgoConfig::Instance().batch_callback(PgoConfig::Instance().tensor_args, stream, workspaceSize, &tiling_data_list_tmp) != 0) {
        return false;
      }
    }
    for (size_t candidate_index = candidate_begin_index1; candidate_index < tiling_data_list_tmp.size(); ++candidate_index) {
      const size_t candidate_offset = candidate_index - candidate_begin_index1;
      if (candidate_offset >= valid_candidates.size() || !valid_candidates[candidate_offset]) {
        continue;
      }
      auto &tiling_data_perf = tiling_data_list_tmp[candidate_index];
      tiling_data_list.push_back(tiling_data_perf);
      if (tiling_data_perf.best_perf < best_perf) {
        tiling_data = tiling_data_perf.tiling_data;
        best_perf = tiling_data_perf.best_perf;
      }
    }
  }
  return true;
}
)rawliteral";

TEST(GeneratorUT, DISABLED_GenGetScheduleResultPGOSuccess) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);

  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult0", "group1");

  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  EXPECT_EQ(genImpl.tiling_func_.output_.str(), kExpectPGOCode);

  EnableGroupParallels enable_group_parallels;
  enable_group_parallels[0][0] = true;
  genImpl.tiling_func_.Reset();
  genImpl.enable_group_parallels_ = enable_group_parallels;
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_EQ(tiling_func_output.empty(), false);
  EXPECT_NE(tiling_func_output.find("uint32_t max_block_dim = tiling_data.group0_tiling_data.get_block_dim();"),
            std::string::npos);
  EXPECT_NE(tiling_func_output.find("max_block_dim = Max(max_block_dim, "
                                    "tiling_data.group1_tiling_data.get_block_dim());"),
            std::string::npos);
  EXPECT_NE(tiling_func_output.find("tiling_data.set_block_dim(max_block_dim);"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("ArrangeBlockOffsetsAscGraph0Result0(tiling_data, ori_block_dim);"),
            std::string::npos);
  EXPECT_EQ(tiling_func_output.find("ArrangeBlockOffsetsAscGraph0Result0(tiling_data, tiling_data.get_block_dim());"),
            std::string::npos);
}

TEST(GeneratorUT, GenPGOGetScheduleResultSetsCurrentGroupVarRelationsBeforeSearch) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);

  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.var_relations_[0][0][1][0]["dep_size"] = CreateExpr("src_size") + CreateExpr(1);

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult1", "group1");

  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  const auto search_pos =
      tiling_func_output.find("ScheduleResult1::PGOSearchTilingKey(tiling_data_list_tmp, group1_tiling_data");
  const auto group1_entry_pos =
      tiling_func_output.rfind("auto &group1_tiling_data = tiling_data.group1_tiling_data;", search_pos);
  const auto guard_pos =
      tiling_func_output.find("tiling_data.group0_tiling_data.get_src_size() == 0U", group1_entry_pos);
  const auto value_pos = tiling_func_output.find("double group1_dep_size_var_relation_value =", group1_entry_pos);
  const auto set_pos = tiling_func_output.find(
      "tiling_data.group1_tiling_data.set_dep_size(static_cast<uint32_t>(group1_dep_size_var_relation_value))",
      group1_entry_pos);

  ASSERT_NE(search_pos, std::string::npos);
  ASSERT_NE(group1_entry_pos, std::string::npos);
  ASSERT_NE(guard_pos, std::string::npos);
  ASSERT_NE(value_pos, std::string::npos);
  ASSERT_NE(set_pos, std::string::npos);
  EXPECT_LT(group1_entry_pos, set_pos);
  EXPECT_LT(guard_pos, value_pos);
  EXPECT_LT(value_pos, set_pos);
  EXPECT_LT(set_pos, search_pos);
}

TEST(GeneratorUT, InductorPgoOnlyFillsGroupsAfterCurrentGroup) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.is_inductor_scene = true;
  TilingModelInfo tiling_model_info{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, tiling_model_info, score_funcs, false);
  const std::map<size_t, std::pair<std::string, std::string>> graph_info = {
      {0U, {"ScheduleResult0", "group0"}},
      {1U, {"ScheduleResult1", "group1"}},
      {2U, {"ScheduleResult2", "group2"}},
  };
  const std::map<std::string, std::set<std::string>> hardware_map = {
      {"group0", {"block_dim"}},
      {"group1", {"block_dim"}},
      {"group2", {"block_dim"}},
  };

  gen_impl.GenFillOtherGroupsGetTiling(0U, 0U, graph_info, *graph_info.find(1U), hardware_map);
  const std::string source = gen_impl.tiling_func_.GetOutputStr();

  EXPECT_EQ(source.find("ScheduleResult0::GetTiling"), std::string::npos);
  EXPECT_EQ(source.find("ScheduleResult1::GetTiling"), std::string::npos);
  EXPECT_NE(source.find("ScheduleResult2::GetTiling"), std::string::npos);
}

TEST(GeneratorUT, GenPGOGetScheduleResultGuardsInvalidVarRelationBeforeSet) {
  TilingCodeGenConfig config;
  config.tiling_data_type_name = "AutofuseTilingData";
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;
  config.is_inductor_scene = true;

  TilingModelInfo tiling_model_info;
  ModelInfo group0_info;
  group0_info.schedule_group_ident.asc_graph_id = 0;
  group0_info.schedule_group_ident.impl_graph_id = 0;
  group0_info.schedule_group_ident.group_id = 0;
  tiling_model_info.push_back(group0_info);
  ModelInfo group1_info;
  group1_info.schedule_group_ident.asc_graph_id = 0;
  group1_info.schedule_group_ident.impl_graph_id = 0;
  group1_info.schedule_group_ident.group_id = 1;
  tiling_model_info.push_back(group1_info);

  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.var_relations_[0][0][1][0]["Rm_org_size"] =
      af::sym::Ceiling(af::sym::Ceiling(CreateExpr(7) / CreateExpr("z2t_size")) / CreateExpr("z2Tt_size"));

  std::map<size_t, std::pair<std::string, std::string>> graph_info;
  graph_info[0] = std::make_pair("ScheduleResult0", "group0");
  graph_info[1] = std::make_pair("ScheduleResult1", "group1");

  std::map<std::string, std::set<std::string>> hardware_map;
  hardware_map["group0"].insert("block_dim");
  hardware_map["group1"].insert("block_dim");

  genImpl.tiling_func_.Reset();
  EXPECT_EQ(genImpl.GenPGOGetScheduleResult(0, 0, graph_info, hardware_map), af::SUCCESS);
  const std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  const auto guard_pos = tiling_func_output.find("get_z2t_size() == 0U");
  const auto value_pos = tiling_func_output.find("double group1_Rm_org_size_var_relation_value =");
  const auto finite_pos = tiling_func_output.find("!std::isfinite(group1_Rm_org_size_var_relation_value)");
  const auto set_pos = tiling_func_output.find("set_Rm_org_size(static_cast<uint32_t>(");
  const auto invalid_pos =
      tiling_func_output.find("valid_candidates[candidate_index - candidate_begin_index0] = false;");
  const auto search_pos =
      tiling_func_output.find("ScheduleResult1::GetTiling(tiling_data.group1_tiling_data, workspace_map, -1)");

  ASSERT_NE(guard_pos, std::string::npos);
  EXPECT_NE(tiling_func_output.find("get_z2Tt_size() == 0U"), std::string::npos);
  ASSERT_NE(value_pos, std::string::npos);
  ASSERT_NE(finite_pos, std::string::npos);
  ASSERT_NE(set_pos, std::string::npos);
  ASSERT_NE(invalid_pos, std::string::npos);
  ASSERT_NE(search_pos, std::string::npos);
  EXPECT_NE(tiling_func_output.find("const size_t candidate_count_before = tiling_data_list.size();"),
            std::string::npos);
  EXPECT_NE(tiling_func_output.find("return tiling_data_list.size() > candidate_count_before;"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("std::vector<AutofuseTilingDataPerf> valid_tiling_data_list;"), std::string::npos);
  EXPECT_NE(tiling_func_output.find("&valid_tiling_data_list) != 0"), std::string::npos);
  EXPECT_EQ(tiling_func_output.find("workspaceSize, &tiling_data_list_tmp) != 0"), std::string::npos);
  EXPECT_LT(guard_pos, value_pos);
  EXPECT_LT(value_pos, finite_pos);
  EXPECT_LT(finite_pos, set_pos);
  EXPECT_LT(set_pos, search_pos);
}

TEST(GeneratorUT, GenWorkspaceRelatedVarsGuardsDynamicDenominator) {
  std::map<int64_t, Expr> workspace_size_map;
  workspace_size_map[0] = af::sym::Ceiling(CreateExpr(512) / CreateExpr("a1t_size"));

  const auto code = GenWorkspaceRelatedVars(workspace_size_map, {});

  EXPECT_NE(code.find("double a1t_size = tiling_data.get_a1t_size();"), std::string::npos);
  EXPECT_NE(code.find("if (a1t_size <= 0) {"), std::string::npos);
  EXPECT_NE(code.find("return false;"), std::string::npos);
  EXPECT_LT(code.find("if (a1t_size <= 0) {"), code.find("static_cast<uint64_t>(Ceiling(512/a1t_size))"));
}

TEST(GeneratorUT, WorkspaceOffsetHelpersUseUint64AndCheckedAdd) {
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, model_infos, score_funcs, false);
  gen_impl.workspace_tensor_id_set_[0][0] = {1, 2};

  gen_impl.tiling_func_.Reset();
  gen_impl.GenWorkspaceOffsetHelpers();
  const auto code = gen_impl.tiling_func_.GetOutputStr();

  EXPECT_NE(code.find("inline bool FinalizeWorkspaceOffsets("), std::string::npos);
  EXPECT_NE(code.find("uint64_t workspace_offset = 0U;"), std::string::npos);
  EXPECT_NE(code.find("__builtin_add_overflow(workspace_offset, workspace_size_1, &workspace_offset)"),
            std::string::npos);
  EXPECT_NE(code.find("return false;"), std::string::npos);
  EXPECT_EQ(code.find("uint32_t workspace_offset"), std::string::npos);
}

TEST(GeneratorUT, GenUpdateWorkspaceDoesNotNarrowWorkspaceMapValue) {
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, model_infos, score_funcs, false);
  gen_impl.workspace_tensor_id_set_[0][0] = {1};

  gen_impl.tiling_func_.Reset();
  gen_impl.GenUpdateWorkspace(0, 0);
  const auto code = gen_impl.tiling_func_.GetOutputStr();

  EXPECT_NE(code.find("tiling_data.set_workspace1(it1->second);"), std::string::npos);
  EXPECT_EQ(code.find("static_cast<uint32_t>(it1->second)"), std::string::npos);
}

TEST(GeneratorUT, WorkspaceSetFailurePropagatesThroughUpdateBetterTiling) {
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{CreateModelInfo()};
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, model_infos, score_funcs, false);

  gen_impl.tiling_func_.Reset();
  gen_impl.GenVirtualDataTransferFuncs();
  gen_impl.GenUpdateBetterTiling();
  const auto code = gen_impl.tiling_func_.GetOutputStr();

  EXPECT_NE(code.find("virtual bool SetWorkspaceSize("), std::string::npos);
  EXPECT_EQ(code.find("virtual void SetWorkspaceSize("), std::string::npos);
  EXPECT_NE(code.find("bool UpdateBetterTiling("), std::string::npos);
  const auto check_pos = code.find("if (!tilingCaseImplPtr->SetWorkspaceSize(tiling_data, workspace_map)) {");
  ASSERT_NE(check_pos, std::string::npos);
  EXPECT_NE(code.find("return false;", check_pos), std::string::npos);
}

TEST(GeneratorUT, PgoCandidateSkipsWhenSetWorkspaceSizeFails) {
  AxesReorderSolverGen solver_gen("0", "AutofuseTilingData");

  const auto code = solver_gen.GenPGOSolverFuncImpl();
  const auto check_pos = code.find("if (!SetWorkspaceSize(new_auto_tiling, workspace_map)) {");

  ASSERT_NE(check_pos, std::string::npos);
  EXPECT_NE(code.find("continue;", check_pos), std::string::npos);
  EXPECT_EQ(code.find("SetWorkspaceSize(new_auto_tiling, workspace_map);\n"), std::string::npos);
}

// UT测试：验证tiling_data.set参数溢出修复
// 测试用例1: 验证 MemoryTilingDataGen::GenFuncImpl 使用 static_cast<uint32_t>()
TEST(GeneratorUT, MemoryTilingDataGen_GenFuncImpl_UseStaticCast) {
  ModelInfo model_info;
  // 创建大数值表达式: 70000 * 70000 * 4 > UINT32_MAX(4294967295)
  // 70000 * 70000 = 4900000000 > UINT32_MAX
  Expr large_expr = af::Symbol(70000, "tmp") * af::Symbol(70000, "tmp") * af::Symbol(4, "tmp");
  model_info.container_exprs["LargeContainer"] = large_expr;

  // 创建 MemoryTilingDataGen 对象
  auto memory_gen = att::MemoryTilingDataGen(model_info);
  EXPECT_EQ(memory_gen.Init(), af::SUCCESS);

  // 获取生成的函数实现代码
  const std::vector<std::string> func_impls = memory_gen.GetTilingFuncImpl("TestTilingData");

  // 验证生成的代码包含 "static_cast<uint32_t>("
  bool found_static_cast = false;
  for (const auto &code : func_impls) {
    if (code.find("static_cast<uint32_t>(") != std::string::npos) {
      found_static_cast = true;
      break;
    }
  }
  EXPECT_TRUE(found_static_cast) << "Generated code should contain 'static_cast<uint32_t>(' to prevent overflow";
}

// 测试用例2: 验证硬件约束代码生成使用 double 类型
TEST(GeneratorUT, GenHardwareCheckCode_UseDoubleType) {
  ModelInfo model_info;
  // 创建大数值硬件约束: 102400 * 102400 = 10485760000，可能导致 uint32_t 溢出
  Expr large_hardware_expr = af::Symbol(102400, "tmp") * af::Symbol(102400, "tmp");
  model_info.hardware_cons[HardwareDef::UB] = large_hardware_expr;

  TilingModelInfo model_infos;
  model_infos.emplace_back(model_info);
  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.gen_extra_infos = false;
  config.gen_tiling_data = false;
  MockTilingCodeGenerator generator;
  EXPECT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, 0UL}, model_infos), af::SUCCESS);
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config), af::SUCCESS);

  // 获取生成的代码
  std::map<std::string, std::string> tiling_res;
  EXPECT_EQ(generator.GenTilingCode(op_name, model_infos, config, tiling_res), af::SUCCESS);

  // 验证生成的代码包含 "double " 类型声明
  bool found_double_type = false;
  for (const auto &[key, code] : tiling_res) {
    if (code.find("double ") != std::string::npos) {
      found_double_type = true;
      break;
    }
  }
  EXPECT_TRUE(found_double_type) << "Generated hardware check code should contain 'double ' type to prevent overflow";
}

TEST(GeneratorUT, GenHardwareJudgeSkipsRelatedCoreNumExpression) {
  TilingModelInfo model_infos{CreateModelInfo(1U, ge::ExprType::kExprVariable)};
  TilingCodeGenConfig config;
  ScoreFuncs score_funcs;
  MockHighPerfTilingCodeGenImpl gen_impl("test", config, model_infos, score_funcs, true);

  ASSERT_EQ(gen_impl.GenHardwareJudge(model_infos.front()), af::SUCCESS);
  const std::string output = gen_impl.tiling_func_.GetOutputStr();
  EXPECT_EQ(output.find("block_dim expr"), std::string::npos);
}

TEST(GeneratorUT, InductorTilingCasesGenerateSearchOverrideForSingleAndMultipleGroups) {
  for (const bool is_uniq_group : {true, false}) {
    auto model_info = CreateModelInfo();
    model_info.schedule_group_ident = {0UL, 0UL, 0UL};
    TilingCodeGenConfig config;
    config.is_inductor_scene = true;
    config.tiling_data_type_name = is_uniq_group ? "AutofuseTilingData" : "AscGraph0ScheduleResult0G0TilingData";
    MockHighPerfTilingCodeGenImpl gen_impl(op_name, config, {model_info}, {}, is_uniq_group);

    ASSERT_EQ(gen_impl.GenTilingCaseImpl(model_info), af::SUCCESS);
    const std::string output = gen_impl.tiling_func_.GetOutputStr();
    const std::string solver = ExtractGuardBody(output, "const SearchConfig *search_cfg=nullptr) override");
    ASSERT_FALSE(solver.empty()) << "is_uniq_group=" << is_uniq_group;
    EXPECT_NE(output.find("const SearchConfig *search_cfg=nullptr) override"), std::string::npos);
    EXPECT_NE(solver.find("pending_search_cfg_ = search_cfg;"), std::string::npos);
    EXPECT_NE(solver.find("tiling_data_list.push_back(tiling_perf);"), std::string::npos);
    EXPECT_NE(solver.find("pending_search_cfg_ = nullptr;"), std::string::npos);
    EXPECT_EQ(solver.find("EmitFinalTiling"), std::string::npos);
    EXPECT_EQ(solver.find("GetTilingDataRepr"), std::string::npos);
    if (is_uniq_group) {
      EXPECT_NE(solver.find("tiling_perf.tiling_data = tiling_data;"), std::string::npos);
    } else {
      EXPECT_NE(solver.find("autofuse_tiling_data->graph0_result0_g0_tiling_data = tiling_data;"), std::string::npos);
      EXPECT_NE(solver.find("tiling_perf.tiling_data = *autofuse_tiling_data;"), std::string::npos);
    }

    gen_impl.config_.is_inductor_scene = false;
    gen_impl.tiling_func_.Reset();
    ASSERT_EQ(gen_impl.GenTilingCaseImpl(model_info), af::SUCCESS);
    EXPECT_EQ(gen_impl.tiling_func_.GetOutputStr().find("bool ExecutePGOSolver("), std::string::npos);
  }
}

// Task 3: Inductor scene triggers ATT PGO main search skeleton, PGOSearchTilingKey and perf extraction

TEST(GeneratorUT, InductorSceneTriggersPGOSkeletonAndSearchTilingKey) {
  TilingCodeGenConfig config;
  TilingModelInfo tiling_model_info;
  ScoreFuncs score_funcs;
  EnableGroupParallels enable_group_parallels;
  std::map<std::string, std::string> tiling_res;
  config.force_template_op_name = "test";
  config.force_schedule_result = 0L;

  ModelInfo info;
  info.reuse_schedule_group = std::make_shared<ReuseScheduleGroup>();
  tiling_model_info.push_back(info);
  enable_group_parallels[0][0] = true;

  MockHighPerfTilingCodeGenImpl genImpl("test", config, tiling_model_info, score_funcs, true);
  genImpl.config_.is_inductor_scene = true;
  genImpl.config_.gen_tiling_data = false;
  EXPECT_EQ(genImpl.GenTiling(tiling_res, {}, 0, enable_group_parallels), af::SUCCESS);

  std::string tiling_func_output = genImpl.tiling_func_.GetOutputStr();
  EXPECT_FALSE(tiling_func_output.empty());
  // Inductor scene must generate ExecutePGOSolver override for single-group
  EXPECT_NE(tiling_func_output.find("bool ExecutePGOSolver("), std::string::npos);
  // Inductor scene must generate SearchAllTilingbyCaseId
  EXPECT_NE(tiling_func_output.find("SearchAllTilingbyCaseId("), std::string::npos);
  // PGOSearchTilingKey must be generated
  EXPECT_NE(tiling_func_output.find("PGOSearchTilingKey("), std::string::npos);
  const auto callback_check = tiling_func_output.find(
      "if (PgoConfig::Instance().batch_callback(PgoConfig::Instance().tensor_args, stream, workspaceSize");
  ASSERT_NE(callback_check, std::string::npos);
  EXPECT_NE(tiling_func_output.find("return false;", callback_check), std::string::npos);
  // GetPerf must be called inside ExecutePGOSolver override
  EXPECT_NE(tiling_func_output.find("GetPerf(tiling_data)"), std::string::npos);
}

// ============================================================================
// Cache line conflict detection tests (Task 1 - expected to fail until feature is implemented)
// ============================================================================

// Test 1: Both groups have cache line conflict → should use sum aggregation
TEST(GeneratorUT, GroupParallelCacheLine_AllConflict_UseSumAggregation) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: conflict (expr=4/8, small value → not aligned to cache line)
  auto info0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  // Group 1: conflict (expr=4/8, same conflict)
  auto info1 =
      CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_0_0"));
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_1_1"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum + normal_perf_merged"));
}

// Test 2: Boundary expression exactly equals cache_line_size → stays normal
TEST(GeneratorUT, GroupParallelCacheLine_BoundaryEqualCacheLine_StaysNormal) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  auto info0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(128) / CreateExpr(256), 128, CacheLineDirection::kUbToGm);
  auto info1 =
      CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(128) / CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("< 128"));
  EXPECT_THAT(all_code, HasSubstr("return false"));
}

// Test 3: Group0 conflict, Group1 normal → init from first normal group
TEST(GeneratorUT, GroupParallelCacheLine_FirstConflictSecondNormal_InitFromFirstNormal) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: conflict (expr=4, small value)
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4), 128, CacheLineDirection::kUbToGm);
  // Group 1: normal (expr=256, large aligned value)
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("has_normal_group"));
  EXPECT_THAT(all_code, HasSubstr("normal_perf_merged += cur_tmp_perf;"));
  EXPECT_THAT(all_code, HasSubstr("Final normal perf"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum +="));
}

// Test 4: Multi-case final tiling key dispatch uses case helper
TEST(GeneratorUT, GroupParallelCacheLine_FinalTilingKeyDispatch_UsesFinalCaseHelper) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: 2 cases (case_id 0 and 1), both conflict
  auto info0_case0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  auto info0_case1 =
      CreateGroupParallelCacheLineModelInfo(0, 1, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  // Group 1: 1 case (case_id 2), normal
  auto info1_case2 = CreateGroupParallelCacheLineModelInfo(1, 2, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0_case0, info0_case1};
  schedule_result.groups_tiling_model_info[1] = {info1_case2};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("get_tiling_key())"));
  EXPECT_THAT(all_code, HasSubstr("case 0: return IsConflictGroup_0_0_0_0()"));
  EXPECT_THAT(all_code, HasSubstr("case 1: return IsConflictGroup_0_0_0_1()"));
}

// Test 5: Byte expression does not multiply dtype_size again
TEST(GeneratorUT, GroupParallelCacheLine_ByteExprDoesNotMultiplyDtypeAgain) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Composite expression: CreateExpr(64) * CreateExpr(2) = 128 bytes (already in bytes)
  Expr byte_expr = CreateExpr(64) * CreateExpr(2);
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, byte_expr, 128, CacheLineDirection::kUbToGm);
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, byte_expr, 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  using testing::Not;
  EXPECT_THAT(all_code, Not(HasSubstr("dtype_size")));
  EXPECT_THAT(all_code, HasSubstr("< 128"));
}

// Test 6: Missing schedule table → fallback to normal with log
TEST(GeneratorUT, GroupParallelCacheLine_MissingScheduleTable_FallbackToNormalWithLog) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: no schedule table (nullptr), manually created
  ModelInfo info0 = CreateModelInfo();
  info0.schedule_group_ident.asc_graph_id = 0;
  info0.schedule_group_ident.impl_graph_id = 0;
  info0.schedule_group_ident.group_id = 0;
  info0.tiling_case_id = 0;
  info0.enable_group_parallel = true;
  info0.tiling_schedule_config_table = nullptr;
  CacheLineConfig cfg0;
  cfg0.node_name = "test_cache_line_node";
  cfg0.cache_line_expr = CreateExpr(4);
  cfg0.cache_line_size = 128;
  cfg0.direction = CacheLineDirection::kUbToGm;
  info0.cache_line_config = {cfg0};

  // Group 1: normal
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("cache line size is unavailable, fallback to normal group"));
}

// Test 7: Direction is kGmToUb (read) → should also use conflict aggregation
TEST(GeneratorUT, GroupParallelCacheLine_GmToUbConflict_UseSumAggregation) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: kGmToUb direction (read)
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4), 128, CacheLineDirection::kGmToUb);
  // Group 1: kUbToGm direction (write, valid)
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(4), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_0_0"));
  EXPECT_THAT(all_code, HasSubstr("IsConflictGroup_0_0_1_1"));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum +="));
  EXPECT_THAT(all_code, HasSubstr("conflict_perf_sum + normal_perf_merged"));
  EXPECT_THAT(all_code, Not(HasSubstr("no valid gm<->ub cache line expr, fallback to normal group")));
}

// Test 8: Duplicate final tiling key → fallback to normal with log
TEST(GeneratorUT, GroupParallelCacheLine_DuplicateFinalKey_FallbackToNormalWithLog) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  // Group 0: 2 cases with SAME tiling_case_id=0 (duplicate key)
  auto info0_case0 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);
  auto info0_case1 =
      CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4) / CreateExpr(8), 128, CacheLineDirection::kUbToGm);

  // Group 1: normal
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0_case0, info0_case1};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("duplicate final tiling key mapping, fallback to normal group"));
  EXPECT_EQ(CountSubstr(all_code, "auto IsConflictGroup_0_0_0_0 ="), 1U);
}

TEST(GeneratorUT, GroupParallelCacheLine_DynamicInputSizeSymbols_GenerateContext) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  Expr s1 = CreateExpr("s1");
  Expr s20 = CreateExpr("s20");
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, s1 * s20 * CreateExpr(4), 128, CacheLineDirection::kUbToGm);
  info0.sizes = {s1, s20};
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};

  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  using testing::Not;
  EXPECT_THAT(all_code, HasSubstr("auto s1 = group_tiling_data.get_s1();"));
  EXPECT_THAT(all_code, HasSubstr("auto s20 = group_tiling_data.get_s20();"));
  EXPECT_THAT(all_code, Not(HasSubstr("cache line expr is not codegenable, fallback to normal group")));
}

TEST(GeneratorUT, GroupParallelCacheLine_UnknownDirection_FallbackToNormal) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, CreateExpr(4), 128, CacheLineDirection::kUnknown);
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};
  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  using testing::HasSubstr;
  EXPECT_THAT(all_code, HasSubstr("no valid gm<->ub cache line expr, fallback to normal group"));
}

TEST(GeneratorUT, GroupParallelCacheLine_MultiWriteExprs_DeduplicateContext) {
  FusedParsedScheduleResult fused_schedule_result;
  auto &schedule_result = fused_schedule_result[0][0];
  schedule_result.impl_graph_id = 0;
  schedule_result.enable_group_parallel = true;

  Expr s1 = CreateExpr("s1");
  Expr s20 = CreateExpr("s20");
  auto info0 = CreateGroupParallelCacheLineModelInfo(0, 0, s1 * s20 * CreateExpr(4), 128, CacheLineDirection::kUbToGm);
  info0.sizes = {s1, s20};
  CacheLineConfig second_cfg = info0.cache_line_config[0];
  second_cfg.node_name = "test_cache_line_node2";
  info0.cache_line_config.push_back(second_cfg);
  auto info1 = CreateGroupParallelCacheLineModelInfo(1, 1, CreateExpr(256), 128, CacheLineDirection::kUbToGm);

  schedule_result.groups_tiling_model_info[0] = {info0};
  schedule_result.groups_tiling_model_info[1] = {info1};
  for (auto &[group_id, infos] : schedule_result.groups_tiling_model_info) {
    ASSERT_EQ(ReuseGroupUtils::InitReuseScheduleGroup({0UL, 0UL, group_id}, infos), af::SUCCESS);
  }

  TilingCodeGenConfig config;
  config.path = "./";
  config.type = TilingImplType::HIGH_PERF;
  config.tiling_data_type_name = "OpTestTilingData";
  config.gen_tiling_data = false;
  config.gen_extra_infos = false;
  std::map<std::string, std::string> tiling_res;
  TilingCodeGenerator generator;
  ASSERT_EQ(generator.GenTilingCode(op_name, fused_schedule_result, config, tiling_res), af::SUCCESS);

  std::string all_code;
  for (const auto &[key, value] : tiling_res) {
    all_code += value;
  }

  EXPECT_EQ(CountSubstr(all_code, "auto s1 = group_tiling_data.get_s1();"), 1U);
  EXPECT_EQ(CountSubstr(all_code, "auto s20 = group_tiling_data.get_s20();"), 1U);
}

}  // namespace att
