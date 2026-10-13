/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "codegen_tiling_data.h"
#include "tests/common/codegen_workspace_test_utils.h"

class TestCodegenTilingData : public testing::Test, public codegen::TilingData {
 protected:
  TestCodegenTilingData() : codegen::TilingData("TestKernel", "TilingData") {}
};

TEST_F(TestCodegenTilingData, ClassBegin) {
  EXPECT_EQ(this->ClassBegin("TestKernelTilingData", ""), "BEGIN_TILING_DATA_DEF_T(TestKernelTilingData)");
}

TEST_F(TestCodegenTilingData, ClassEnd) {
  EXPECT_EQ(this->ClassEnd(), "END_TILING_DATA_DEF_T;");
}

TEST_F(TestCodegenTilingData, DataFieldDefine) {
  af::SizeVar s0(af::Symbol("s0"));
  EXPECT_EQ(this->DataFieldDefine(s0), "TILING_DATA_FIELD_DEF_T(uint32_t, s0);");
}

TEST_F(TestCodegenTilingData, ClassRegister) {
  EXPECT_EQ(this->ClassRegister(), "REGISTER_TILING_DATA_CLASS(TestKernel, TilingData)");
}

TEST_F(TestCodegenTilingData, SingleGroupGenerateTilingData) {
  af::AscGraph graph0("test_graph0");
  graph0.CreateSizeVar("s0");
  graph0.CreateSizeVar("s1");

  af::AscGraph graph1("test_graph1");
  graph1.CreateSizeVar("s1");
  graph1.CreateSizeVar("s2");

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);
  impl_graphs.push_back(graph1);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const std::string test_res = R"rawliteral(#ifndef __TestKernel_Tiling_Data_H__
#define __TestKernel_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() const { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(TestKernelTilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);
  TILING_DATA_FIELD_DEF_T(uint32_t, s0);
  TILING_DATA_FIELD_DEF_T(uint32_t, s1);
  TILING_DATA_FIELD_DEF_T(uint32_t, s2);
END_TILING_DATA_DEF_T;

using AutofuseTilingData = TestKernelTilingData;
struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)rawliteral";
  EXPECT_EQ(this->Generate(fused_schedule_result), test_res);
}

TEST_F(TestCodegenTilingData, GenerateWorkspaceFieldsShouldUseUint64) {
  auto fused_schedule_result = autofuse::tests::MakeWorkspaceFusedScheduleResult(2UL);
  const auto result = this->Generate(fused_schedule_result);

  EXPECT_NE(result.find("TILING_DATA_FIELD_DEF_T(uint64_t, workspace1);"), std::string::npos);
  EXPECT_NE(result.find("TILING_DATA_FIELD_DEF_T(uint64_t, workspace2);"), std::string::npos);
  EXPECT_EQ(result.find("TILING_DATA_FIELD_DEF_T(uint32_t, workspace"), std::string::npos);
}

TEST_F(TestCodegenTilingData, GenerateConstWorkspaceSizeShouldUseUint64) {
  auto fused_schedule_result = autofuse::tests::MakeWorkspaceFusedScheduleResult(2UL);
  const auto result = this->GenerateConst(fused_schedule_result);

  EXPECT_NE(result.find("uint64_t workspace_size;"), std::string::npos);
  EXPECT_EQ(result.find("uint32_t workspace_size;"), std::string::npos);
}

TEST_F(TestCodegenTilingData, SingleGroupGenerateTilingDataWithTranspose) {
  af::AscGraph graph0("test_graph0");
  graph0.CreateSizeVar("s0");
  graph0.CreateSizeVar("s1");

  af::AscGraph graph1("test_graph1");
  graph1.CreateSizeVar("s1");
  graph1.CreateSizeVar("s2");

  af::ascir_op::Transpose transpose_op("Transpose");
  graph1.AddNode(transpose_op);

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);
  impl_graphs.push_back(graph1);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const std::string test_res = R"rawliteral(#ifndef __TestKernel_Tiling_Data_H__
#define __TestKernel_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() const { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(TestKernelTilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);
  TILING_DATA_FIELD_DEF_T(uint32_t, s0);
  TILING_DATA_FIELD_DEF_T(uint32_t, s1);
  TILING_DATA_FIELD_DEF_T(uint32_t, s2);
  TILING_DATA_FIELD_DEF_T_STRUCT(ConfusionTransposeTiling, Transpose_tilingData_1);
END_TILING_DATA_DEF_T;

using AutofuseTilingData = TestKernelTilingData;
struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)rawliteral";
  EXPECT_EQ(this->Generate(fused_schedule_result), test_res);
}

TEST_F(TestCodegenTilingData, SingleGroupGenerateConstTilingDataWithTranspose) {
  af::AscGraph graph0("test_graph0");
  graph0.CreateSizeVar("s0");
  graph0.CreateSizeVar("s1");

  af::AscGraph graph1("test_graph1");
  graph1.CreateSizeVar("s1");
  graph1.CreateSizeVar("s2");

  af::ascir_op::Transpose transpose_op("Transpose");
  graph1.AddNode(transpose_op);

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);
  impl_graphs.push_back(graph1);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const std::string test_res = R"rawliteral(std::string tiling_data_const_gen_result;
AutofuseTilingData TilingDataValue;

void replaceSubstring(std::string& ori_str, const std::string& old_sub_str, const std::string& new_sub_str) {
  size_t pos = ori_str.find(old_sub_str);
  if (pos != std::string::npos) {
    ori_str.replace(pos, old_sub_str.length(), new_sub_str);
  }
}

std::string GenTilingDataFieldConstDefFunc(const std::string &f_name, uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << "const uint32_t ";
  ss_mid << f_name << " = " << std::to_string(value) << ";" << std::endl;
  return ss_mid.str();
}
std::string GenTilingDataFieldConstDefFunc(const std::string &f_name, uint64_t value) {
  std::stringstream ss_mid;
  ss_mid << "const uint64_t ";
  ss_mid << f_name << " = " << std::to_string(value) << ";" << std::endl;
  return ss_mid.str();
}

std::string GenTilingDataFieldConstValueFunc(uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << std::to_string(value) << std::endl;
  return ss_mid.str();
}


extern "C" const char* GenConstTilingData(char* config_file, int aiv_num, int ub_size) {
  uint64_t workspace_size;
  uint32_t block_dim;
  ResLimit limit;
  limit.aiv_num = aiv_num;
  limit.ub_size = ub_size - 256;
  auto ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, nullptr);
  if (ret != 0) {
    OP_LOGE(OP_NAME, "AutofuseTilingWithConfig failed: %ld", ret);
    return "";
  }
  std::string GenTilingDataValue_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.block_dim);
  std::string GenTilingDataValue_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.corenum);
  std::string GenTilingDataValue_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.ub_size);
  std::string GenTilingDataValue_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.hbm_size);
  std::string GenTilingDataValue_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("tiling_key", TilingDataValue.tiling_key);
  std::string GenTilingDataValue_s0_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s0", TilingDataValue.s0);
  std::string GenTilingDataValue_s1_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s1", TilingDataValue.s1);
  std::string GenTilingDataValue_s2_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s2", TilingDataValue.s2);
  std::string GenTilingDataValue_Transpose_tilingData_1_param0_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param0);
  std::string GenTilingDataValue_Transpose_tilingData_1_param1_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param1);
  std::string GenTilingDataValue_Transpose_tilingData_1_param2_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param2);
  std::string GenTilingDataValue_Transpose_tilingData_1_param3_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param3);
  std::string GenTilingDataValue_Transpose_tilingData_1_param4_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param4);
  std::string GenTilingDataValue_Transpose_tilingData_1_param5_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param5);
  std::string GenTilingDataValue_Transpose_tilingData_1_param6_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param6);
  std::string GenTilingDataValue_Transpose_tilingData_1_param7_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param7);
  std::string GenTilingDataValue_Transpose_tilingData_1_param8_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param8);
  std::string GenTilingDataValue_Transpose_tilingData_1_param9_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param9);
  std::string GenTilingDataValue_Transpose_tilingData_1_param10_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param10);
  std::string GenTilingDataValue_Transpose_tilingData_1_param11_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param11);
  std::string GenTilingDataValue_Transpose_tilingData_1_param12_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param12);
  std::string GenTilingDataValue_Transpose_tilingData_1_param13_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param13);
  std::string GenTilingDataValue_Transpose_tilingData_1_param14_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param14);
  std::string GenTilingDataValue_Transpose_tilingData_1_param15_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param15);
  std::string GenTilingDataValue_Transpose_tilingData_1_param16_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param16);
  std::string GenTilingDataValue_Transpose_tilingData_1_param17_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.Transpose_tilingData_1.param17);

  tiling_data_const_gen_result = R"(#ifndef __TestKernel_Tiling_Data_H__
#define __TestKernel_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() const { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(TestKernelTilingData)
  GenTilingDataValue_block_dim_field_DeclareFunc_def
  GenTilingDataValue_corenum_field_DeclareFunc_def
  GenTilingDataValue_ub_size_field_DeclareFunc_def
  GenTilingDataValue_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_tiling_key_field_DeclareFunc_def
  GenTilingDataValue_s0_field_DeclareFunc_def
  GenTilingDataValue_s1_field_DeclareFunc_def
  GenTilingDataValue_s2_field_DeclareFunc_def
  const ConfusionTransposeTiling Transpose_tilingData_1 = {GenTilingDataValue_Transpose_tilingData_1_param0_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param1_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param2_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param3_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param4_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param5_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param6_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param7_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param8_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param9_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param10_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param11_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param12_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param13_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param14_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param15_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param16_field_DeclareFunc_field_def, GenTilingDataValue_Transpose_tilingData_1_param17_field_DeclareFunc_field_def};

END_TILING_DATA_DEF_T;

using AutofuseTilingData = TestKernelTilingData;
struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)";
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_block_dim_field_DeclareFunc_def",GenTilingDataValue_block_dim_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_corenum_field_DeclareFunc_def",GenTilingDataValue_corenum_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_ub_size_field_DeclareFunc_def",GenTilingDataValue_ub_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_hbm_size_field_DeclareFunc_def",GenTilingDataValue_hbm_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_tiling_key_field_DeclareFunc_def",GenTilingDataValue_tiling_key_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_s0_field_DeclareFunc_def",GenTilingDataValue_s0_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_s1_field_DeclareFunc_def",GenTilingDataValue_s1_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_s2_field_DeclareFunc_def",GenTilingDataValue_s2_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param0_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param0_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param1_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param1_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param2_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param2_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param3_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param3_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param4_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param4_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param5_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param5_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param6_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param6_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param7_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param7_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param8_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param8_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param9_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param9_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param10_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param10_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param11_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param11_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param12_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param12_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param13_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param13_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param14_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param14_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param15_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param15_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param16_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param16_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_Transpose_tilingData_1_param17_field_DeclareFunc_field_def",GenTilingDataValue_Transpose_tilingData_1_param17_field_DeclareFunc_field_def);

  return tiling_data_const_gen_result.c_str();
}

)rawliteral";
  EXPECT_EQ(this->GenerateConst(fused_schedule_result), test_res);
}

TEST_F(TestCodegenTilingData, multiGroupGenerateConstTilingDataWithTranspose) {
  af::AscGraph graph0("test_graph0");
  graph0.CreateSizeVar("s0");
  graph0.CreateSizeVar("s1");

  af::AscGraph graph1("test_graph1");
  graph1.CreateSizeVar("s1");
  graph1.CreateSizeVar("s2");

  af::ascir_op::Transpose transpose_op("Transpose");
  graph1.AddNode(transpose_op);

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);
  impl_graphs.push_back(graph1);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_result.schedule_groups.push_back(schedule_group);

  schedule_results.push_back(schedule_result);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  const std::string test_res = R"rawliteral(std::string tiling_data_const_gen_result;
AutofuseTilingData TilingDataValue;

void replaceSubstring(std::string& ori_str, const std::string& old_sub_str, const std::string& new_sub_str) {
  size_t pos = ori_str.find(old_sub_str);
  if (pos != std::string::npos) {
    ori_str.replace(pos, old_sub_str.length(), new_sub_str);
  }
}

std::string GenTilingDataFieldConstDefFunc(const std::string &f_name, uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << "const uint32_t ";
  ss_mid << f_name << " = " << std::to_string(value) << ";" << std::endl;
  return ss_mid.str();
}
std::string GenTilingDataFieldConstDefFunc(const std::string &f_name, uint64_t value) {
  std::stringstream ss_mid;
  ss_mid << "const uint64_t ";
  ss_mid << f_name << " = " << std::to_string(value) << ";" << std::endl;
  return ss_mid.str();
}

std::string GenTilingDataFieldConstValueFunc(uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << std::to_string(value) << std::endl;
  return ss_mid.str();
}


extern "C" const char* GenConstTilingData(char* config_file, int aiv_num, int ub_size) {
  uint64_t workspace_size;
  uint32_t block_dim;
  ResLimit limit;
  limit.aiv_num = aiv_num;
  limit.ub_size = ub_size - 256;
  auto ret = AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, nullptr);
  if (ret != 0) {
    OP_LOGE(OP_NAME, "AutofuseTilingWithConfig failed: %ld", ret);
    return "";
  }
  std::string GenTilingDataValue_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.block_dim);
  std::string GenTilingDataValue_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.corenum);
  std::string GenTilingDataValue_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.ub_size);
  std::string GenTilingDataValue_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.hbm_size);
  std::string GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("graph0_tiling_key", TilingDataValue.graph0_tiling_key);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.graph0_result0_g0_tiling_data.block_dim);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.graph0_result0_g0_tiling_data.corenum);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.graph0_result0_g0_tiling_data.ub_size);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.graph0_result0_g0_tiling_data.hbm_size);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("tiling_key", TilingDataValue.graph0_result0_g0_tiling_data.tiling_key);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_s0_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s0", TilingDataValue.graph0_result0_g0_tiling_data.s0);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_s1_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s1", TilingDataValue.graph0_result0_g0_tiling_data.s1);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_s2_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s2", TilingDataValue.graph0_result0_g0_tiling_data.s2);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param0);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param1);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param2);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param3);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param4);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param5);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param6);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param7);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param8);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param9);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param10);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param11);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param12);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param13);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param14);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param15);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param16);
  std::string GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g0_tiling_data.Transpose_tilingData_1.param17);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.graph0_result0_g1_tiling_data.block_dim);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.graph0_result0_g1_tiling_data.corenum);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.graph0_result0_g1_tiling_data.ub_size);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.graph0_result0_g1_tiling_data.hbm_size);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("tiling_key", TilingDataValue.graph0_result0_g1_tiling_data.tiling_key);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_s0_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s0", TilingDataValue.graph0_result0_g1_tiling_data.s0);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_s1_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s1", TilingDataValue.graph0_result0_g1_tiling_data.s1);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_s2_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("s2", TilingDataValue.graph0_result0_g1_tiling_data.s2);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param0);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param1);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param2);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param3);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param4);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param5);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param6);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param7);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param8);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param9);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param10);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param11);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param12);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param13);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param14);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param15);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param16);
  std::string GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def = GenTilingDataFieldConstValueFunc(TilingDataValue.graph0_result0_g1_tiling_data.Transpose_tilingData_1.param17);

  tiling_data_const_gen_result = R"(#ifndef __TestKernel_Tiling_Data_H__
#define __TestKernel_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() const { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(AscGraph0ScheduleResult0G0TilingData)
  GenTilingDataValue_graph0_result0_g0_tiling_data_block_dim_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g0_tiling_data_corenum_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g0_tiling_data_ub_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g0_tiling_data_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g0_tiling_data_tiling_key_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g0_tiling_data_s0_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g0_tiling_data_s1_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g0_tiling_data_s2_field_DeclareFunc_def
  const ConfusionTransposeTiling Transpose_tilingData_1 = {GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def};

END_TILING_DATA_DEF_T;

BEGIN_TILING_DATA_DEF_T(AscGraph0ScheduleResult0G1TilingData)
  GenTilingDataValue_graph0_result0_g1_tiling_data_block_dim_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g1_tiling_data_corenum_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g1_tiling_data_ub_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g1_tiling_data_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g1_tiling_data_tiling_key_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g1_tiling_data_s0_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g1_tiling_data_s1_field_DeclareFunc_def
  GenTilingDataValue_graph0_result0_g1_tiling_data_s2_field_DeclareFunc_def
  const ConfusionTransposeTiling Transpose_tilingData_1 = {GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def, GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def};

END_TILING_DATA_DEF_T;

BEGIN_TILING_DATA_DEF_T(TestKernelTilingData)
  GenTilingDataValue_block_dim_field_DeclareFunc_def
  GenTilingDataValue_corenum_field_DeclareFunc_def
  GenTilingDataValue_ub_size_field_DeclareFunc_def
  GenTilingDataValue_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def
  TILING_DATA_FIELD_DEF_T_STRUCT(AscGraph0ScheduleResult0G0TilingData, graph0_result0_g0_tiling_data);
  TILING_DATA_FIELD_DEF_T_STRUCT(AscGraph0ScheduleResult0G1TilingData, graph0_result0_g1_tiling_data);
END_TILING_DATA_DEF_T;

using AutofuseTilingData = TestKernelTilingData;
struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)";
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_block_dim_field_DeclareFunc_def",GenTilingDataValue_block_dim_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_corenum_field_DeclareFunc_def",GenTilingDataValue_corenum_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_ub_size_field_DeclareFunc_def",GenTilingDataValue_ub_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_hbm_size_field_DeclareFunc_def",GenTilingDataValue_hbm_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def",GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_block_dim_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_block_dim_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_corenum_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_corenum_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_ub_size_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_ub_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_hbm_size_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_hbm_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_tiling_key_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_tiling_key_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_s0_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_s0_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_s1_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_s1_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_s2_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g0_tiling_data_s2_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g0_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_block_dim_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_block_dim_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_corenum_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_corenum_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_ub_size_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_ub_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_hbm_size_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_hbm_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_tiling_key_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_tiling_key_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_s0_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_s0_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_s1_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_s1_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_s2_field_DeclareFunc_def",GenTilingDataValue_graph0_result0_g1_tiling_data_s2_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param0_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param1_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param2_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param3_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param4_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param5_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param6_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param7_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param8_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param9_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param10_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param11_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param12_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param13_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param14_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param15_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param16_field_DeclareFunc_field_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def",GenTilingDataValue_graph0_result0_g1_tiling_data_Transpose_tilingData_1_param17_field_DeclareFunc_field_def);

  return tiling_data_const_gen_result.c_str();
}

)rawliteral";
  EXPECT_EQ(this->GenerateConst(fused_schedule_result), test_res);
}

TEST_F(TestCodegenTilingData, MultiGroupGenerateTilingData) {
  af::AscGraph graph0("test_graph0");
  graph0.CreateSizeVar("s0");
  graph0.CreateSizeVar("s1");

  af::AscGraph graph1("test_graph1");
  graph1.CreateSizeVar("s1");
  graph1.CreateSizeVar("s2");

  af::AscGraph graph2("test_graph2");
  graph2.CreateSizeVar("s0");
  graph2.CreateSizeVar("s1");

  af::AscGraph graph3("test_graph3");
  graph3.CreateSizeVar("s1");
  graph3.CreateSizeVar("s2");

  af::AscGraph graph4("test_graph4");
  graph4.CreateSizeVar("s0");
  graph4.CreateSizeVar("s1");

  af::AscGraph graph5("test_graph5");
  graph5.CreateSizeVar("s1");
  graph5.CreateSizeVar("s2");

  // schedule_result0
  std::vector<ascir::ImplGraph> impl_graphs0;
  impl_graphs0.push_back(graph0);
  impl_graphs0.push_back(graph1);

  // schedule_result1 schedule_group0
  std::vector<ascir::ImplGraph> impl_graphs1;
  impl_graphs1.push_back(graph2);
  impl_graphs1.push_back(graph3);

  // schedule_result1 schedule_group1
  std::vector<ascir::ImplGraph> impl_graphs2;
  impl_graphs2.push_back(graph4);
  impl_graphs2.push_back(graph5);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result0;
  ascir::ScheduleGroup schedule_group0;
  schedule_group0.impl_graphs = impl_graphs0;
  schedule_result0.schedule_groups.push_back(schedule_group0);
  schedule_results.push_back(schedule_result0);

  ascir::ScheduledResult schedule_result1;
  ascir::ScheduleGroup schedule_group10;
  schedule_group10.impl_graphs = impl_graphs1;
  ascir::ScheduleGroup schedule_group11;
  schedule_group11.impl_graphs = impl_graphs2;
  schedule_result1.schedule_groups.push_back(schedule_group10);
  schedule_result1.schedule_groups.push_back(schedule_group11);
  schedule_results.push_back(schedule_result1);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const std::string test_res = R"rawliteral(#ifndef __TestKernel_Tiling_Data_H__
#define __TestKernel_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() const { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(AscGraph0ScheduleResult0G0TilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);
  TILING_DATA_FIELD_DEF_T(uint32_t, s0);
  TILING_DATA_FIELD_DEF_T(uint32_t, s1);
  TILING_DATA_FIELD_DEF_T(uint32_t, s2);
END_TILING_DATA_DEF_T;

BEGIN_TILING_DATA_DEF_T(AscGraph0ScheduleResult1G0TilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);
  TILING_DATA_FIELD_DEF_T(uint32_t, s0);
  TILING_DATA_FIELD_DEF_T(uint32_t, s1);
  TILING_DATA_FIELD_DEF_T(uint32_t, s2);
END_TILING_DATA_DEF_T;

BEGIN_TILING_DATA_DEF_T(AscGraph0ScheduleResult1G1TilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, tiling_key);
  TILING_DATA_FIELD_DEF_T(uint32_t, s0);
  TILING_DATA_FIELD_DEF_T(uint32_t, s1);
  TILING_DATA_FIELD_DEF_T(uint32_t, s2);
END_TILING_DATA_DEF_T;

BEGIN_TILING_DATA_DEF_T(TestKernelTilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, graph0_tiling_key);
  TILING_DATA_FIELD_DEF_T_STRUCT(AscGraph0ScheduleResult0G0TilingData, graph0_result0_g0_tiling_data);
  TILING_DATA_FIELD_DEF_T_STRUCT(AscGraph0ScheduleResult1G0TilingData, graph0_result1_g0_tiling_data);
  TILING_DATA_FIELD_DEF_T_STRUCT(AscGraph0ScheduleResult1G1TilingData, graph0_result1_g1_tiling_data);
END_TILING_DATA_DEF_T;

using AutofuseTilingData = TestKernelTilingData;
struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)rawliteral";
  EXPECT_EQ(this->Generate(fused_schedule_result), test_res);
}

TEST_F(TestCodegenTilingData, AddApiTilingData) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  /* x是否需要指定位宽，调度API函数需要指定 */
  af::ascir_op::Transpose transpose_op("Transpose");
  graph.AddNode(transpose_op);

  /* TODO：perm成员赋值方式 */
  // transpose_op.perm = std::vector<int32>{0, 2, 1};

  *transpose_op.y.axis = {z1.id, z0.id};

  auto transpose = graph.FindNode("Transpose");
  transpose->attr.api.compute_type = af::ComputeType::kComputeTranspose;
  transpose->attr.api.type = af::ApiType::kAPITypeCompute;
  transpose->attr.api.unit = af::ComputeUnit::kUnitVector;

  /* TODO:不配置循环轴 */
  transpose->attr.sched.loop_axis = z0.id;

  /* TODO：新增perm节点 ？？ */
  transpose->outputs[0].attr.vectorized_axis = {z1.id, z0.id};
  transpose->outputs[0].attr.dtype = ge::DT_FLOAT;
  transpose->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  transpose->outputs[0].attr.mem.tensor_id = 1;
  transpose->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  transpose->outputs[0].attr.que.id = 2;
  transpose->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  /*  TODO：AddTensor只需要add输出吗？ Perm是否需要体现 */
  tpipe.AddTensor(transpose->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.SetTilingCaseId(0);
  codegen::ApiTensor x;

  std::stringstream ss;
  uint32_t tiling_case_id = 0;
  std::string tilingString;
  codegen::TilingData::AddApiTilingData(graph, ss, tiling_case_id);
  tilingString = ss.str();

  EXPECT_EQ(tilingString,
            std::string{"  TILING_DATA_FIELD_DEF_T_STRUCT(ConfusionTransposeTiling, Transpose_tilingData_0);\n"});
}
