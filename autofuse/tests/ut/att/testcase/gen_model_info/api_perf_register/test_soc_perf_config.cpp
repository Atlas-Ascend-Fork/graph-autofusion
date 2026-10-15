/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <string>

#include "common/platform_context.h"
#include "common_utils.h"
#include "gen_model_info/api_perf_register/api_perf_factory.h"
#include "gen_model_info/api_perf_register/ascendc_api_perf.h"
#include "gen_model_info/api_perf_register/soc_perf_config.h"
#include "graph/ascendc_ir/ascir_register.h"
#define private public
#include "expr_gen/pipe_perf_expr.h"
#undef private
#include "gen_model_info.h"
#include "gtest/gtest.h"
#include "core/test_fa_ascir_graph.h"
#include "graph_construct_utils.h"
#include "soc_perf_test_config.h"
#include "v35/att/api_perf_register/perf_param_v2.h"

namespace att {
namespace {
using soc_perf_test::kSocA;
using soc_perf_test::kSocB;
using soc_perf_test::kV35NpuArch;

constexpr char kDefaultAscIrKey[] = "SocPerfTestDefaultAscIr";
constexpr char kReplacementAscIrKey[] = "SocPerfTestReplacementAscIr";
constexpr char kDefaultAscendCKey[] = "SocPerfTestDefaultAscendC";
constexpr char kReplacementAscendCKey[] = "SocPerfTestReplacementAscendC";
constexpr char kUnknownNpuArch[] = "SocPerfTestUnknownArch";

class UnknownArchAscIrAtt : public af::ascir::AscIrAtt {
 public:
  void *GetApiPerf() const override {
    static ge::char_t api_perf_name[] = "Tanh";
    return ge::PtrToPtr<ge::char_t, void>(api_perf_name);
  }

  void *GetAscendCApiPerfTable() const override {
    return nullptr;
  }
};

REG_ASC_IR(SocPerfUnknownArchApi)
    .Input("x", "T")
    .Output("y", "T")
    .Impl({kUnknownNpuArch}, {af::ascir::AscIrImplCreator<UnknownArchAscIrAtt>(),
                              af::ascir::AscIrImplCreator<af::ascir::AscIrCodegen>(),
                              {{"T", af::TensorType{ge::DT_FLOAT}}}});

int g_default_ascir_calls = 0;
int g_replacement_ascir_calls = 0;
int g_default_ascendc_calls = 0;
int g_replacement_ascendc_calls = 0;

af::Status CountDefaultAscIr(const std::vector<TensorShapeInfo> &, const std::vector<TensorShapeInfo> &,
                             const NodeInfo &, PerfOutputInfo &) {
  ++g_default_ascir_calls;
  return af::SUCCESS;
}

af::Status CountReplacementAscIr(const std::vector<TensorShapeInfo> &, const std::vector<TensorShapeInfo> &,
                                 const NodeInfo &, PerfOutputInfo &) {
  ++g_replacement_ascir_calls;
  return af::SUCCESS;
}

af::Status CountDefaultAscendC(const NodeDetail &, PerfOutputInfo &) {
  ++g_default_ascendc_calls;
  return af::SUCCESS;
}

af::Status CountReplacementAscendC(const NodeDetail &, PerfOutputInfo &) {
  ++g_replacement_ascendc_calls;
  return af::SUCCESS;
}

af::Status FailingSocAscIrPerf(const std::vector<TensorShapeInfo> &, const std::vector<TensorShapeInfo> &,
                               const NodeInfo &, PerfOutputInfo &perf) {
  perf.pipe_res[PipeType::AIV_VEC] = CreateExpr(123);
  return af::FAILED;
}

af::Status CallAscIrPerf(Perf perf_func) {
  NodeInfo node;
  PerfOutputInfo perf;
  return perf_func({}, {}, node, perf);
}

af::Status CallAscendCPerf(AscendCPerf perf_func) {
  NodeDetail node;
  PerfOutputInfo perf;
  return perf_func(node, perf);
}

af::Status SetV35Platform(const std::string &soc_version) {
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = kV35NpuArch;
  platform_info.aiv_num = 1;
  platform_info.ub_size = 262144;
  return ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, soc_version);
}

NodePerfInfo MakeAbsPerfInfo() {
  NodePerfInfo node;
  node.optype = "Abs";
  node.input_dtype = "float16";
  node.output_dtype = "float16";
  node.dims = {CreateExpr(64)};
  return node;
}

NodeInfo MakeAbsNodeInfo() {
  NodeInfo node;
  node.name = "soc_abs";
  node.node_type = "Abs";
  node.node_unit = "Abs";
  auto tensor = std::make_shared<Tensor>();
  tensor->data_type = "float16";
  tensor->data_type_size = 2U;
  tensor->repeat = {CreateExpr(64)};
  tensor->stride = {af::sym::kSymbolOne};
  node.inputs.emplace_back(tensor);
  node.outputs.emplace_back(tensor);
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Abs", 1, 1);
  return node;
}

::ascir::FusedScheduledResult MakeSocScheduleResult() {
  af::AscGraph graph("soc_target_graph");
  if (GraphConstructUtils::BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(graph) != af::SUCCESS) {
    return {};
  }
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);

  ::ascir::ScheduledResult schedule_result;
  ::ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.emplace_back(graph);
  schedule_result.schedule_groups.emplace_back(schedule_group);
  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::vector{schedule_result});
  return fused_schedule_result;
}

SocPerfConfigs MakeSingleConfig(const std::string &soc_version, const Json &api_override) {
  SocPerfConfigs configs;
  SocPerfConfig config;
  config.api_param_overrides = api_override;
  configs.emplace(SocPerfKey{kV35NpuArch, soc_version}, config);
  return configs;
}

class ScopedPlacementConfigs {
 public:
  ScopedPlacementConfigs(const SocPerfConfigs &configs, void *storage)
      : configs_(new(storage) SocPerfConfigs(configs)) {}
  ~ScopedPlacementConfigs() {
    configs_->~SocPerfConfigs();
  }
  const SocPerfConfigs &Get() const {
    return *configs_;
  }
  const void *Address() const {
    return configs_;
  }

 private:
  SocPerfConfigs *configs_;
};

Expr MakeLinearResult(const Json &params) {
  const auto &record = params.at("Abs").at("model_params").at("float16tofloat16");
  return CreateExpr(record.at("k").get<float>()) * CreateExpr(64) + CreateExpr(record.at("b").get<float>());
}

class SocPerfConfigTest : public testing::Test {
 public:
  void SetUp() override {
    g_default_ascir_calls = 0;
    g_replacement_ascir_calls = 0;
    g_default_ascendc_calls = 0;
    g_replacement_ascendc_calls = 0;
    EvalCosts::Instance().RegisterFunc(kDefaultAscIrKey, &CountDefaultAscIr);
    EvalCosts::Instance().RegisterFunc(kReplacementAscIrKey, &CountReplacementAscIr);
    EvalCosts::Instance().RegisterAscendCFunc(kDefaultAscendCKey, &CountDefaultAscendC);
    EvalCosts::Instance().RegisterAscendCFunc(kReplacementAscendCKey, &CountReplacementAscendC);
  }

  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(SocPerfConfigTest, ExplicitConfigsRejectUnknownArchAndKeepBaselineWithoutSocVersion) {
  const SocPerfConfigs &configs = soc_perf_test::GetSocPerfTestConfigs();
  const SocPerfTarget unknown_arch{"UnknownArch", kSocA};
  EXPECT_NE(ValidateSocPerfConfig(unknown_arch, configs), af::SUCCESS);

  const SocPerfTarget missing_soc{kV35NpuArch, ""};
  EXPECT_EQ(ValidateSocPerfConfig(missing_soc, configs), af::SUCCESS);
}

TEST_F(SocPerfConfigTest, ExplicitConfigsKeepBaselineForUnregisteredSoc) {
  const SocPerfTarget target{kV35NpuArch, "SOC_NOT_BOUND"};
  EXPECT_EQ(ValidateSocPerfConfig(target, soc_perf_test::GetSocPerfTestConfigs()), af::SUCCESS);
  std::shared_ptr<const Json> params;
  ASSERT_EQ(GetSocApiParams(target, kParamV2Info, soc_perf_test::GetSocPerfTestConfigs(), params), af::SUCCESS);
  EXPECT_EQ(*params, Json::parse(kParamV2Info));
}

TEST_F(SocPerfConfigTest, PublicWrapperUsesTestResource) {
  const SocPerfTarget target{kV35NpuArch, kSocA};
  EXPECT_EQ(ValidateSocPerfConfig(target), af::SUCCESS);
}

TEST_F(SocPerfConfigTest, RegisteredV2ArchitectureUsesBaselineWithoutConfigEntry) {
  soc_perf_test::RegisterSocPerfTestArchitecture();
  std::shared_ptr<const Json> first_params;
  for (const std::string &soc : {std::string("SOC_UNREGISTERED"), std::string()}) {
    const SocPerfTarget target{soc_perf_test::kRegisteredNpuArch, soc};
    ASSERT_EQ(ValidateSocPerfConfig(target), af::SUCCESS);
    std::shared_ptr<const Json> params;
    ASSERT_EQ(GetSocApiParams(target, kParamV2Info, params), af::SUCCESS);
    EXPECT_EQ(*params, Json::parse(kParamV2Info));
    if (first_params == nullptr) {
      first_params = params;
    }
    EXPECT_EQ(params, first_params);
    ASSERT_EQ(soc_perf_test::SetSocPerfTestPlatform(target.npu_arch, soc), af::SUCCESS);
    EXPECT_EQ(ValidateSocPerfEntry(), af::SUCCESS);
  }
}

TEST_F(SocPerfConfigTest, V2ArchitecturesUnknownSocUseOriginalBaseline) {
  const Json baseline = Json::parse(kParamV2Info);
  const auto default_api = ApiPerfFactory::Instance().Create("AbsV2");
  ASSERT_NE(default_api, nullptr);
  for (const std::string &npu_arch : {"3510", "5102", "9201", "9202"}) {
    std::shared_ptr<const Json> first_params;
    for (const std::string &soc : {std::string("Ascend950PR_9589"), std::string("SOC_UNREGISTERED"), std::string()}) {
      const SocPerfTarget target{npu_arch, soc};
      ASSERT_EQ(ValidateSocPerfConfig(target), af::SUCCESS);
      std::shared_ptr<const Json> params;
      ASSERT_EQ(GetSocApiParams(target, kParamV2Info, params), af::SUCCESS);
      EXPECT_EQ(*params, baseline);
      if (first_params == nullptr) {
        first_params = params;
      }
      EXPECT_EQ(params, first_params);
      ASSERT_EQ(soc_perf_test::SetSocPerfTestPlatform(npu_arch, soc), af::SUCCESS);
      if (soc.empty()) {
        ge::PlatformContext::GetInstance().SetPlatform(npu_arch);
      }
      EXPECT_EQ(ValidateSocPerfEntry(), af::SUCCESS);
      Expr perf;
      ASSERT_EQ(GetPerf(MakeAbsPerfInfo(), perf), af::SUCCESS);
      EXPECT_EQ(Str(perf), Str(MakeLinearResult(baseline)));
      const auto api = GetApiPerf("Abs");
      ASSERT_NE(api, nullptr);
      EXPECT_EQ(api->GetPerfFunc(), default_api->GetPerfFunc());
      EXPECT_EQ(GetAscendCPerfFunc("LoadV2"), EvalCosts::Instance().GetAscendCFunc("LoadV2"));
    }
  }
}

TEST_F(SocPerfConfigTest, NpuArch3510RegisteredSocOverridesBaseline) {
  const SocPerfTarget target{soc_perf_test::kV35AltNpuArch, "SOC_3510_REGISTERED"};
  SocPerfConfigs configs;
  configs.emplace(SocPerfKey{target.npu_arch, ""}, SocPerfConfig{});
  SocPerfConfig config;
  config.api_param_overrides = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  config.ascir_model_overrides = {{"AbsV2", kReplacementAscIrKey}};
  configs.emplace(SocPerfKey{target.npu_arch, target.soc_version}, config);
  ASSERT_EQ(ValidateSocPerfConfig(target, configs), af::SUCCESS);
  std::shared_ptr<const Json> params;
  ASSERT_EQ(GetSocApiParams(target, kParamV2Info, configs, params), af::SUCCESS);
  EXPECT_EQ(Str(MakeLinearResult(*params)), "74.0");
  Perf selected_func = nullptr;
  ASSERT_EQ(ResolveSocAscIrPerf(target, "AbsV2", &CountDefaultAscIr, selected_func, configs), af::SUCCESS);
  EXPECT_EQ(selected_func, &CountReplacementAscIr);

  std::shared_ptr<const Json> default_params;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{target.npu_arch, "SOC_UNREGISTERED"}, kParamV2Info, default_params),
            af::SUCCESS);
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{target.npu_arch, kSocA}, kParamV2Info, params), af::SUCCESS);
  EXPECT_EQ(Str(MakeLinearResult(*params)), "74.0");
  EXPECT_NE(params, default_params);
  ASSERT_EQ(soc_perf_test::SetSocPerfTestPlatform(target.npu_arch, kSocA), af::SUCCESS);
  const auto api = GetApiPerf("Abs");
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->GetPerfFunc(), EvalCosts::Instance().GetFunc("ExpV2"));
}

TEST_F(SocPerfConfigTest, NpuArch3510InvalidRegisteredConfigDoesNotFallback) {
  const SocPerfTarget target{soc_perf_test::kV35AltNpuArch, "SOC_3510_INVALID"};
  SocPerfConfigs configs;
  configs.emplace(SocPerfKey{target.npu_arch, ""}, SocPerfConfig{});
  SocPerfConfig config;
  config.api_param_overrides = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  config.api_param_overrides["Abs"]["model_params"]["float16tofloat16"].erase("b");
  configs.emplace(SocPerfKey{target.npu_arch, target.soc_version}, config);
  EXPECT_NE(ValidateSocPerfConfig(target, configs), af::SUCCESS);
  std::shared_ptr<const Json> params;
  EXPECT_NE(GetSocApiParams(target, kParamV2Info, configs, params), af::SUCCESS);
  EXPECT_EQ(params, nullptr);
  config.api_param_overrides = Json::object();
  config.ascir_model_overrides = {{"AbsV2", "MissingSocModel"}};
  configs[SocPerfKey{target.npu_arch, target.soc_version}] = config;
  EXPECT_NE(ValidateSocPerfConfig(target, configs), af::SUCCESS);
}

TEST_F(SocPerfConfigTest, GetSocApiParamsSelectsCoefficientsBySoc) {
  const SocPerfConfigs &configs = soc_perf_test::GetSocPerfTestConfigs();
  std::shared_ptr<const Json> params_a;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, kSocA}, kParamV2Info, configs, params_a), af::SUCCESS);
  EXPECT_EQ(Str(MakeLinearResult(*params_a)), "74.0");

  std::shared_ptr<const Json> params_b;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, kSocB}, kParamV2Info, configs, params_b), af::SUCCESS);
  EXPECT_EQ(Str(MakeLinearResult(*params_b)), "148.0");
}

TEST_F(SocPerfConfigTest, GetSocApiParamsKeepsBaselineForEmptyOverride) {
  const SocPerfConfigs configs = MakeSingleConfig("SOC_EMPTY", Json::object());
  std::shared_ptr<const Json> params;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_EMPTY"}, kParamV2Info, configs, params), af::SUCCESS);
  EXPECT_EQ(*params, Json::parse(kParamV2Info));
}

TEST_F(SocPerfConfigTest, EmptyOverrideTargetsShareParsedBaseline) {
  SocPerfConfigs configs = MakeSingleConfig("SOC_EMPTY_A", Json::object());
  SocPerfConfig empty_config;
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_EMPTY_B"}, empty_config);
  ASSERT_EQ(configs.size(), 2U);
  ASSERT_NE(configs.find(SocPerfKey{kV35NpuArch, "SOC_EMPTY_B"}), configs.end());

  std::shared_ptr<const Json> params_a;
  std::shared_ptr<const Json> params_b;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_EMPTY_A"}, kParamV2Info, configs, params_a), af::SUCCESS);
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_EMPTY_B"}, kParamV2Info, configs, params_b), af::SUCCESS);
  EXPECT_EQ(params_a, params_b);
}

TEST_F(SocPerfConfigTest, BuiltinGetSocApiParamsCachesByTarget) {
  std::shared_ptr<const Json> first_a;
  std::shared_ptr<const Json> second_a;
  std::shared_ptr<const Json> first_b;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, kSocA}, kParamV2Info, first_a), af::SUCCESS);
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, kSocB}, kParamV2Info, first_b), af::SUCCESS);
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, kSocA}, kParamV2Info, second_a), af::SUCCESS);
  EXPECT_EQ(first_a, second_a);
  EXPECT_NE(first_a, first_b);
}

TEST_F(SocPerfConfigTest, GetSocApiParamsDoesNotReuseDestroyedExplicitConfigAddress) {
  const SocPerfTarget target{kV35NpuArch, "SOC_EXPLICIT_CONFIG_ADDRESS_REUSE"};
  alignas(SocPerfConfigs) std::byte config_storage[sizeof(SocPerfConfigs)];
  const void *config_address = &config_storage;
  std::shared_ptr<const Json> params_a;
  std::shared_ptr<const Json> params_b;
  {
    ScopedPlacementConfigs configs_a(MakeSingleConfig(target.soc_version, soc_perf_test::MakeApiOverride(1.0F, 10.0F)),
                                     &config_storage);
    ASSERT_EQ(GetSocApiParams(target, kParamV2Info, configs_a.Get(), params_a), af::SUCCESS);
  }
  {
    ScopedPlacementConfigs configs_b(MakeSingleConfig(target.soc_version, soc_perf_test::MakeApiOverride(2.0F, 20.0F)),
                                     &config_storage);
    ASSERT_EQ(configs_b.Address(), config_address);
    ASSERT_EQ(GetSocApiParams(target, kParamV2Info, configs_b.Get(), params_b), af::SUCCESS);
  }
  EXPECT_EQ(Str(MakeLinearResult(*params_a)), "74.0");
  EXPECT_EQ(Str(MakeLinearResult(*params_b)), "148.0");
}

TEST_F(SocPerfConfigTest, GetSocApiParamsCacheSeparatesExplicitConfigs) {
  const SocPerfTarget target{kV35NpuArch, "SOC_EXPLICIT_CONFIG_CACHE"};
  const SocPerfConfigs configs_a = MakeSingleConfig(target.soc_version, soc_perf_test::MakeApiOverride(1.0F, 10.0F));
  const SocPerfConfigs configs_b = MakeSingleConfig(target.soc_version, soc_perf_test::MakeApiOverride(2.0F, 20.0F));

  std::shared_ptr<const Json> params_a;
  std::shared_ptr<const Json> params_b;
  std::shared_ptr<const Json> params_a_again;
  ASSERT_EQ(GetSocApiParams(target, kParamV2Info, configs_a, params_a), af::SUCCESS);
  ASSERT_EQ(GetSocApiParams(target, kParamV2Info, configs_b, params_b), af::SUCCESS);
  ASSERT_EQ(GetSocApiParams(target, kParamV2Info, configs_a, params_a_again), af::SUCCESS);
  EXPECT_EQ(Str(MakeLinearResult(*params_a)), "74.0");
  EXPECT_EQ(Str(MakeLinearResult(*params_b)), "148.0");
  EXPECT_EQ(Str(MakeLinearResult(*params_a_again)), "74.0");
}

TEST_F(SocPerfConfigTest, ValidateRejectsNonObjectApiOverrides) {
  const SocPerfConfigs configs = MakeSingleConfig("SOC_NON_OBJECT_OVERRIDES", Json::array());
  EXPECT_NE(ValidateSocPerfConfig(SocPerfTarget{kV35NpuArch, "SOC_NON_OBJECT_OVERRIDES"}, configs), af::SUCCESS);
}

TEST_F(SocPerfConfigTest, GetSocApiParamsRejectsUnknownApiAndDtype) {
  Json api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  api_override["UnknownApi"] = api_override["Abs"];
  SocPerfConfigs configs = MakeSingleConfig("SOC_UNKNOWN_API", api_override);
  std::shared_ptr<const Json> params;
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_UNKNOWN_API"}, kParamV2Info, configs, params), af::SUCCESS);

  api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  api_override["Abs"]["model_params"]["unknown_dtype"] = {{"k", 1.0F}, {"b", 10.0F}};
  configs = MakeSingleConfig("SOC_UNKNOWN_DTYPE", api_override);
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_UNKNOWN_DTYPE"}, kParamV2Info, configs, params),
            af::SUCCESS);
  EXPECT_EQ(params, nullptr);
}

TEST_F(SocPerfConfigTest, GetSocApiParamsRejectsModelTypeAndInputFieldOverride) {
  Json api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  api_override["Abs"]["model_type"] = "AnotherModel";
  SocPerfConfigs configs = MakeSingleConfig("SOC_MODEL_TYPE", api_override);
  std::shared_ptr<const Json> params;
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_MODEL_TYPE"}, kParamV2Info, configs, params), af::SUCCESS);

  for (const std::string &field : {"data_type_size", "block_count_idx"}) {
    api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
    api_override["Abs"]["model_params"]["float16tofloat16"][field] = 2.0F;
    configs = MakeSingleConfig("SOC_INPUT_FIELD", api_override);
    EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_INPUT_FIELD"}, kParamV2Info, configs, params),
              af::SUCCESS);
  }
  EXPECT_EQ(params, nullptr);
}

TEST_F(SocPerfConfigTest, GetSocApiParamsRejectsIncompleteInvalidAndZeroDivisorRecords) {
  Json api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  api_override["Abs"]["model_params"]["float16tofloat16"].erase("b");
  SocPerfConfigs configs = MakeSingleConfig("SOC_MISSING_COEFF", api_override);
  std::shared_ptr<const Json> params;
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_MISSING_COEFF"}, kParamV2Info, configs, params),
            af::SUCCESS);

  api_override = soc_perf_test::MakeApiOverride(std::numeric_limits<float>::quiet_NaN(), 10.0F);
  configs = MakeSingleConfig("SOC_NOT_FINITE", api_override);
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_NOT_FINITE"}, kParamV2Info, configs, params), af::SUCCESS);

  api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  api_override["Abs"]["model_params"]["float16tofloat16"]["k"] = "invalid";
  configs = MakeSingleConfig("SOC_NOT_NUMBER", api_override);
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_NOT_NUMBER"}, kParamV2Info, configs, params), af::SUCCESS);

  api_override["Abs"]["model_params"]["float16tofloat16"] = {{"h", 1.0F}, {"a", 0.0F}, {"b", 1.0F}, {"hl", 0.0F}};
  api_override["Load"]["model_params"]["float16tofloat16"] = api_override["Abs"]["model_params"]["float16tofloat16"];
  api_override["Abs"] = Json::object();
  configs = MakeSingleConfig("SOC_ZERO_DIVISOR", api_override);
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_ZERO_DIVISOR"}, kParamV2Info, configs, params),
            af::SUCCESS);
  EXPECT_EQ(params, nullptr);
}

TEST_F(SocPerfConfigTest, GetSocApiParamsRejectsFloatOverflowAndUnderflowDivisor) {
  Json api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  api_override["Abs"]["model_params"]["float16tofloat16"]["k"] = 1e100;
  SocPerfConfigs configs = MakeSingleConfig("SOC_FLOAT_OVERFLOW", api_override);
  std::shared_ptr<const Json> params;
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_FLOAT_OVERFLOW"}, kParamV2Info, configs, params),
            af::SUCCESS);

  api_override = soc_perf_test::MakeApiOverride(1.0F, 10.0F);
  api_override["Load"]["model_params"]["float16tofloat16"] = {{"h", 1.0F}, {"a", 1e-300}, {"b", 1.0F}, {"hl", 0.0F}};
  configs = MakeSingleConfig("SOC_FLOAT_UNDERFLOW_DIVISOR", api_override);
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_FLOAT_UNDERFLOW_DIVISOR"}, kParamV2Info, configs, params),
            af::SUCCESS);
  EXPECT_EQ(params, nullptr);
}

TEST_F(SocPerfConfigTest, GetSocApiParamsAllowsNegativeCoefficients) {
  const SocPerfConfigs configs = MakeSingleConfig("SOC_NEGATIVE", soc_perf_test::MakeApiOverride(-1.0F, -10.0F));
  std::shared_ptr<const Json> params;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_NEGATIVE"}, kParamV2Info, configs, params), af::SUCCESS);
  EXPECT_EQ(Str(MakeLinearResult(*params)), "-74.0");
}

TEST_F(SocPerfConfigTest, GetSocApiParamsRejectsForeignBaseline) {
  const SocPerfConfigs configs = MakeSingleConfig("SOC_BASELINE", soc_perf_test::MakeApiOverride(1.0F, 10.0F));
  std::shared_ptr<const Json> params;
  EXPECT_NE(GetSocApiParams(SocPerfTarget{kV35NpuArch, "SOC_BASELINE"}, kParamV2Info + "\n", configs, params),
            af::SUCCESS);
}

TEST_F(SocPerfConfigTest, ExplicitTargetIsIndependentFromPlatformState) {
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = kV35NpuArch;
  platform_info.aiv_num = 1;
  platform_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, kSocB), af::SUCCESS);

  const SocPerfConfigs &configs = soc_perf_test::GetSocPerfTestConfigs();
  std::shared_ptr<const Json> params;
  ASSERT_EQ(GetSocApiParams(SocPerfTarget{kV35NpuArch, kSocA}, kParamV2Info, configs, params), af::SUCCESS);
  EXPECT_EQ(Str(MakeLinearResult(*params)), "74.0");
}

TEST_F(SocPerfConfigTest, HistoricalGetterContentIsUnchanged) {
  const PerfParamTableV2 param_table;
  ASSERT_NE(param_table.GetAscendCApiPerfTable(), nullptr);
  EXPECT_EQ(*param_table.GetAscendCApiPerfTable(), kParamV2Info);
}

TEST_F(SocPerfConfigTest, ResolveSocAscIrPerfUsesReplacementOnce) {
  SocPerfConfigs configs;
  SocPerfConfig config;
  config.ascir_model_overrides = {{"AbsV2", kReplacementAscIrKey}};
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_MODEL_ASCIR"}, config);

  Perf selected_func = nullptr;
  ASSERT_EQ(ResolveSocAscIrPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_ASCIR"}, "AbsV2", &CountDefaultAscIr,
                                selected_func, configs),
            af::SUCCESS);
  EXPECT_EQ(selected_func, &CountReplacementAscIr);
  EXPECT_EQ(CallAscIrPerf(selected_func), af::SUCCESS);
  EXPECT_EQ(g_default_ascir_calls, 0);
  EXPECT_EQ(g_replacement_ascir_calls, 1);
}

TEST_F(SocPerfConfigTest, ResolveSocAscIrPerfKeepsDefaultForEmptyOverride) {
  const SocPerfConfigs configs = MakeSingleConfig("SOC_MODEL_ASCIR_EMPTY", Json::object());
  Perf selected_func = nullptr;
  ASSERT_EQ(ResolveSocAscIrPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_ASCIR_EMPTY"}, "Abs", &CountDefaultAscIr,
                                selected_func, configs),
            af::SUCCESS);
  EXPECT_EQ(selected_func, &CountDefaultAscIr);
  EXPECT_EQ(CallAscIrPerf(selected_func), af::SUCCESS);
  EXPECT_EQ(g_default_ascir_calls, 1);
  EXPECT_EQ(g_replacement_ascir_calls, 0);
}

TEST_F(SocPerfConfigTest, ResolveSocAscendCPerfUsesReplacementOnce) {
  SocPerfConfigs configs;
  SocPerfConfig config;
  config.ascendc_model_overrides = {{"LoadV2", kReplacementAscendCKey}};
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_MODEL_ASCENDC"}, config);

  AscendCPerf selected_func = nullptr;
  ASSERT_EQ(ResolveSocAscendCPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_ASCENDC"}, "LoadV2", &CountDefaultAscendC,
                                  selected_func, configs),
            af::SUCCESS);
  EXPECT_EQ(selected_func, &CountReplacementAscendC);
  EXPECT_EQ(CallAscendCPerf(selected_func), af::SUCCESS);
  EXPECT_EQ(g_default_ascendc_calls, 0);
  EXPECT_EQ(g_replacement_ascendc_calls, 1);
}

TEST_F(SocPerfConfigTest, ResolveSocAscendCPerfKeepsDefaultForEmptyOverride) {
  const SocPerfConfigs configs = MakeSingleConfig("SOC_MODEL_ASCENDC_EMPTY", Json::object());
  AscendCPerf selected_func = nullptr;
  ASSERT_EQ(ResolveSocAscendCPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_ASCENDC_EMPTY"}, "LoadV2", &CountDefaultAscendC,
                                  selected_func, configs),
            af::SUCCESS);
  EXPECT_EQ(selected_func, &CountDefaultAscendC);
  EXPECT_EQ(CallAscendCPerf(selected_func), af::SUCCESS);
  EXPECT_EQ(g_default_ascendc_calls, 1);
  EXPECT_EQ(g_replacement_ascendc_calls, 0);
}

TEST_F(SocPerfConfigTest, ResolveRejectsMissingSourceOrDestination) {
  SocPerfConfigs configs;
  SocPerfConfig config;
  config.ascir_model_overrides = {{"UnknownApi", kReplacementAscIrKey}};
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_MODEL_SOURCE_MISSING"}, config);
  Perf selected_func = nullptr;
  EXPECT_NE(ResolveSocAscIrPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_SOURCE_MISSING"}, "UnknownApi",
                                &CountDefaultAscIr, selected_func, configs),
            af::SUCCESS);

  configs.clear();
  config.ascir_model_overrides = {{"AbsV2", "UnknownAscIrPerf"}};
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_MODEL_DEST_MISSING"}, config);
  EXPECT_NE(ResolveSocAscIrPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_DEST_MISSING"}, "AbsV2", &CountDefaultAscIr,
                                selected_func, configs),
            af::SUCCESS);

  configs.clear();
  config.ascir_model_overrides.clear();
  config.ascendc_model_overrides = {{"UnknownAscendC", kReplacementAscendCKey}};
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_MODEL_ASCENDC_SOURCE_MISSING"}, config);
  AscendCPerf selected_ascendc_func = nullptr;
  EXPECT_NE(ResolveSocAscendCPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_ASCENDC_SOURCE_MISSING"}, "UnknownAscendC",
                                  &CountDefaultAscendC, selected_ascendc_func, configs),
            af::SUCCESS);

  configs.clear();
  config.ascendc_model_overrides = {{"LoadV2", "UnknownAscendCPerf"}};
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_MODEL_ASCENDC_DEST_MISSING"}, config);
  EXPECT_NE(ResolveSocAscendCPerf(SocPerfTarget{kV35NpuArch, "SOC_MODEL_ASCENDC_DEST_MISSING"}, "LoadV2",
                                  &CountDefaultAscendC, selected_ascendc_func, configs),
            af::SUCCESS);
  EXPECT_EQ(selected_func, nullptr);
  EXPECT_EQ(selected_ascendc_func, nullptr);
}

TEST_F(SocPerfConfigTest, ResolveQueriesOnlyRequestedApiOverride) {
  const SocPerfTarget target{kV35NpuArch, "SOC_MODEL_UNRELATED_OVERRIDE"};
  SocPerfConfigs configs;
  SocPerfConfig config;
  config.ascir_model_overrides = {{"UnknownApi", kReplacementAscIrKey}};
  config.ascendc_model_overrides = {{"UnknownAscendC", kReplacementAscendCKey}};
  configs.emplace(SocPerfKey{target.npu_arch, target.soc_version}, config);
  EXPECT_NE(ValidateSocPerfConfig(target, configs), af::SUCCESS);

  Perf selected_func = nullptr;
  ASSERT_EQ(ResolveSocAscIrPerf(target, "AbsV2", &CountDefaultAscIr, selected_func, configs), af::SUCCESS);
  EXPECT_EQ(selected_func, &CountDefaultAscIr);

  AscendCPerf selected_ascendc_func = nullptr;
  ASSERT_EQ(ResolveSocAscendCPerf(target, "LoadV2", &CountDefaultAscendC, selected_ascendc_func, configs), af::SUCCESS);
  EXPECT_EQ(selected_ascendc_func, &CountDefaultAscendC);
}

TEST_F(SocPerfConfigTest, ResolveUsesExplicitTargetAndStaticRegistrationDoesNotReadPlatform) {
  EXPECT_NE(GetPerfFunc("AbsV2"), nullptr);

  ge::PlatformInfo platform_info;
  platform_info.soc_ver = kV35NpuArch;
  platform_info.aiv_num = 1;
  platform_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, kSocB), af::SUCCESS);

  SocPerfConfigs configs;
  SocPerfConfig config;
  config.ascir_model_overrides = {{"AbsV2", kReplacementAscIrKey}};
  configs.emplace(SocPerfKey{kV35NpuArch, kSocA}, config);

  Perf selected_func = nullptr;
  ASSERT_EQ(ResolveSocAscIrPerf(SocPerfTarget{kV35NpuArch, kSocA}, "AbsV2", &CountDefaultAscIr, selected_func, configs),
            af::SUCCESS);
  EXPECT_EQ(selected_func, &CountReplacementAscIr);
  EXPECT_NE(GetPerfFunc("AbsV2"), nullptr);
}

TEST_F(SocPerfConfigTest, GetPerfSelectsParamsByPlatformTarget) {
  SetV35Platform(kSocA);
  Expr res;
  ASSERT_EQ(GetPerf(MakeAbsPerfInfo(), res), af::SUCCESS);
  EXPECT_EQ(Str(res), "74.0");

  SetV35Platform(kSocB);
  ASSERT_EQ(GetPerf(MakeAbsPerfInfo(), res), af::SUCCESS);
  EXPECT_EQ(Str(res), "148.0");
}

TEST_F(SocPerfConfigTest, GetApiPerfReplacesPerfAndKeepsMetadata) {
  SetV35Platform(kSocA);
  const auto default_api = ApiPerfFactory::Instance().Create("AbsV2");
  const auto soc_api = GetApiPerf("Abs");
  ASSERT_NE(default_api, nullptr);
  ASSERT_NE(soc_api, nullptr);
  EXPECT_EQ(soc_api->GetApiName(), default_api->GetApiName());
  EXPECT_EQ(soc_api->GetPerfParam(), default_api->GetPerfParam());
  EXPECT_EQ(soc_api->GetMicroPerfFunc(), default_api->GetMicroPerfFunc());
  EXPECT_EQ(soc_api->GetTilingScheduleConfigTable(), default_api->GetTilingScheduleConfigTable());
  EXPECT_EQ(soc_api->GetPerfFunc(), EvalCosts::Instance().GetFunc("ExpV2"));
}

TEST_F(SocPerfConfigTest, ValidateRejectsAscIrTypeKeyForModelOverride) {
  SocPerfConfigs configs;
  SocPerfConfig config;
  config.ascir_model_overrides = {{"Abs", kReplacementAscIrKey}};
  configs.emplace(SocPerfKey{kV35NpuArch, "SOC_MODEL_ASCIR_TYPE_KEY"}, config);
  EXPECT_NE(ValidateSocPerfConfig(SocPerfTarget{kV35NpuArch, "SOC_MODEL_ASCIR_TYPE_KEY"}, configs), af::SUCCESS);
}

TEST_F(SocPerfConfigTest, GetApiPerfAppliesOverrideByRegistrationNameAndAlias) {
  SetV35Platform(kSocA);

  const auto abs_api = GetApiPerf("Abs");
  ASSERT_NE(abs_api, nullptr);
  EXPECT_EQ(abs_api->GetPerfFunc(), EvalCosts::Instance().GetFunc("ExpV2"));

  const auto select_api = GetApiPerf("Select");
  ASSERT_NE(select_api, nullptr);
  EXPECT_EQ(select_api->GetPerfFunc(), EvalCosts::Instance().GetFunc("ExpV2"));
}

TEST_F(SocPerfConfigTest, GetApiPerfAppliesOverrideWhenRegistryApiPerfNameIsEmpty) {
  SetV35Platform(kSocA);
  const auto soc_api = GetApiPerf("Tanh");
  ASSERT_NE(soc_api, nullptr);
  EXPECT_EQ(soc_api->GetPerfFunc(), EvalCosts::Instance().GetFunc("ExpV2"));
}

TEST_F(SocPerfConfigTest, GetApiPerfRejectsUnknownPlatformTarget) {
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = kUnknownNpuArch;
  platform_info.aiv_num = 1;
  platform_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, kSocA), af::SUCCESS);
  EXPECT_EQ(GetApiPerf("SocPerfUnknownArchApi"), nullptr);
}

TEST_F(SocPerfConfigTest, GetAscendCPerfFuncAppliesSocOverride) {
  SetV35Platform(kSocA);
  EXPECT_EQ(GetAscendCPerfFunc("LoadV2"), EvalCosts::Instance().GetAscendCFunc("StoreV2"));

  SetV35Platform(kSocB);
  EXPECT_EQ(GetAscendCPerfFunc("LoadV2"), EvalCosts::Instance().GetAscendCFunc("LoadV2"));
}

TEST_F(SocPerfConfigTest, V2GetPerfKeepsOriginalPath) {
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "2201";
  platform_info.aiv_num = 1;
  platform_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, ""), af::SUCCESS);

  Expr res;
  ASSERT_EQ(GetPerf(MakeAbsPerfInfo(), res), af::SUCCESS);
  EXPECT_FALSE(Str(res).empty());
}

TEST_F(SocPerfConfigTest, GetModelInfoMapRejectsInvalidSocTargetBeforeModelGeneration) {
  ge::PlatformContext::GetInstance().SetPlatform("UnknownArch");
  const auto fused_schedule_result = MakeSocScheduleResult();
  FusedParsedScheduleResult model_infos;
  EXPECT_NE(GetModelInfoMap(fused_schedule_result, {}, model_infos), af::SUCCESS);
  EXPECT_TRUE(model_infos.empty());

  model_infos.clear();
  ASSERT_EQ(SetV35Platform("SOC_NOT_BOUND"), af::SUCCESS);
  EXPECT_EQ(GetModelInfoMap(fused_schedule_result, {}, model_infos), af::SUCCESS);
  EXPECT_FALSE(model_infos.empty());

  model_infos.clear();
  ASSERT_EQ(SetV35Platform(soc_perf_test::kSocEmpty), af::SUCCESS);
  EXPECT_EQ(GetModelInfoMap(fused_schedule_result, {}, model_infos), af::SUCCESS);
  EXPECT_FALSE(model_infos.empty());
}

TEST_F(SocPerfConfigTest, V35PipeFormulaFailureDoesNotCommitNodePerf) {
  SetV35Platform(kSocB);
  const Perf saved_func = EvalCosts::Instance().GetFunc("AbsV2");
  ASSERT_NE(saved_func, nullptr);
  EvalCosts::Instance().RegisterFunc("AbsV2", &FailingSocAscIrPerf);

  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  PipePerfExpr pipe_perf(tuning_space);
  std::map<PipeType, Expr> node_perf;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  std::vector<PerfBreakdownGroup> perf_breakdowns;
  EXPECT_NE(pipe_perf.GetNodePerf(MakeAbsNodeInfo(), node_perf, ternary_ops, perf_breakdowns), af::SUCCESS);
  EXPECT_TRUE(node_perf.empty());
  EXPECT_TRUE(perf_breakdowns.empty());
  EvalCosts::Instance().RegisterFunc("AbsV2", saved_func);
}

TEST_F(SocPerfConfigTest, V2PipeFormulaFailureKeepsWarningOnlyBehavior) {
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "2201";
  platform_info.aiv_num = 1;
  platform_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, ""), af::SUCCESS);
  const Perf saved_func = EvalCosts::Instance().GetFunc("Abs");
  ASSERT_NE(saved_func, nullptr);
  EvalCosts::Instance().RegisterFunc("Abs", &FailingSocAscIrPerf);

  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  PipePerfExpr pipe_perf(tuning_space);
  std::map<PipeType, Expr> node_perf;
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;
  std::vector<PerfBreakdownGroup> perf_breakdowns;
  EXPECT_EQ(pipe_perf.GetNodePerf(MakeAbsNodeInfo(), node_perf, ternary_ops, perf_breakdowns), af::SUCCESS);
  EXPECT_FALSE(node_perf.empty());
  EvalCosts::Instance().RegisterFunc("Abs", saved_func);
}
}  // namespace
}  // namespace att
