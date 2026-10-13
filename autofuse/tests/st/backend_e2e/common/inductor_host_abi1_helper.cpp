/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "autofuse_tiling_data.h"
#include "inductor_host_helper_common.h"

namespace {

using autofuse::tests::HostCaseRunner;
using autofuse::tests::HostHelperOptions;
using autofuse::tests::InputConfigs;
using autofuse::tests::ParseHostHelperOptions;
using autofuse::tests::ResLimit;
using autofuse::tests::ResolveSymbol;
using autofuse::tests::RunHostCheck;

using GetTilingDataReprFn = std::string (*)(const AutofuseTilingData *);

#ifndef INDUCTOR_HOST_HELPER_DYNAMIC_ARG_COUNT
#define INDUCTOR_HOST_HELPER_DYNAMIC_ARG_COUNT 0
#endif

constexpr int kDynamicShapeArgCountValue = INDUCTOR_HOST_HELPER_DYNAMIC_ARG_COUNT;
static_assert(kDynamicShapeArgCountValue >= 0, "dynamic shape arg count should not be negative");
constexpr size_t kDynamicShapeArgCount = static_cast<size_t>(kDynamicShapeArgCountValue);

template <size_t>
using GenerateDynamicArg = int64_t;

template <size_t>
using TilingDynamicArg = uint32_t;

template <typename DynamicIndexSequence>
struct GenerateTopnSolutionsSignature;

template <size_t... Indexes>
struct GenerateTopnSolutionsSignature<std::index_sequence<Indexes...>> {
  using Type = int64_t (*)(GenerateDynamicArg<Indexes>..., const InputConfigs &, int64_t,
                           std::vector<AutofuseTilingData> &, std::vector<int64_t> &, std::vector<int64_t> &,
                           ResLimit *);
};

template <typename DynamicIndexSequence>
struct AutofuseTilingSignature;

template <size_t... Indexes>
struct AutofuseTilingSignature<std::index_sequence<Indexes...>> {
  using Type = int64_t (*)(TilingDynamicArg<Indexes>..., AutofuseTilingData *, uint64_t *, uint32_t *, ResLimit *);
};

struct TopnResult {
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;
};

template <size_t DynamicShapeArgCount>
class TypedHostRunner : public HostCaseRunner {
 public:
  explicit TypedHostRunner(const std::vector<int64_t> &dynamic_shape_args) {
    for (size_t i = 0; i < DynamicShapeArgCount; ++i) {
      dynamic_shape_args_[i] = dynamic_shape_args[i];
    }
  }

  bool Resolve(void *handle) override {
    repr_ = ResolveSymbol<GetTilingDataReprFn>(handle, "GetTilingDataRepr");
    gen_ = ResolveSymbol<GenerateTopnSolutionsFn>(handle, "GenerateTopnSolutions");
    autofuse_tiling_ = ResolveSymbol<AutofuseTilingFn>(handle, "AutofuseTiling");
    return repr_ != nullptr && gen_ != nullptr && autofuse_tiling_ != nullptr;
  }

  int64_t GenerateTopn(const InputConfigs &input_configs, int64_t topn) override {
    topn_result_ = {};
    return GenerateTopnImpl(input_configs, topn, DynamicIndexSequence{});
  }

  int64_t RunAutofuseTiling() override {
    default_tiling_data_ = {};
    default_workspace_ = 0;
    default_block_dim_ = 0;
    return RunAutofuseTilingImpl(DynamicIndexSequence{});
  }

  size_t ResultSize() const override {
    return topn_result_.tiling_datas.size();
  }
  int64_t ResultWorkspace(size_t index) const override {
    return topn_result_.workspaces[index];
  }
  int64_t ResultBlockDim(size_t index) const override {
    return topn_result_.block_dims[index];
  }
  std::string ResultRepr(size_t index) const override {
    return repr_(&topn_result_.tiling_datas[index]);
  }
  bool VerifyExtraTopnResult() const override {
#ifdef INDUCTOR_HOST_HELPER_CHECK_Z0T_POSITIVE
    for (const auto &tiling_data : topn_result_.tiling_datas) {
      if (tiling_data.get_z0t_size() == 0U) {
        std::cerr << "topn z0t_size should be positive" << std::endl;
        return false;
      }
    }
#endif
    return true;
  }
  std::string DefaultRepr() const override {
    return repr_(&default_tiling_data_);
  }
  uint64_t DefaultWorkspace() const override {
    return default_workspace_;
  }
  uint32_t DefaultBlockDim() const override {
    return default_block_dim_;
  }

 private:
  using DynamicIndexSequence = std::make_index_sequence<DynamicShapeArgCount>;
  using GenerateTopnSolutionsFn = typename GenerateTopnSolutionsSignature<DynamicIndexSequence>::Type;
  using AutofuseTilingFn = typename AutofuseTilingSignature<DynamicIndexSequence>::Type;

  template <size_t... Indexes>
  int64_t GenerateTopnImpl(const InputConfigs &input_configs, int64_t topn, std::index_sequence<Indexes...>) {
    return gen_(dynamic_shape_args_[Indexes]..., input_configs, topn, topn_result_.tiling_datas,
                topn_result_.workspaces, topn_result_.block_dims, &res_limit_);
  }

  template <size_t... Indexes>
  int64_t RunAutofuseTilingImpl(std::index_sequence<Indexes...>) {
    return autofuse_tiling_(static_cast<uint32_t>(dynamic_shape_args_[Indexes])..., &default_tiling_data_,
                            &default_workspace_, &default_block_dim_, &res_limit_);
  }

  std::array<int64_t, DynamicShapeArgCount> dynamic_shape_args_ = {};
  ResLimit res_limit_ = {1, 48, 0, 192 * 1024, {0}};
  TopnResult topn_result_;
  AutofuseTilingData default_tiling_data_ = {};
  uint64_t default_workspace_ = 0;
  uint32_t default_block_dim_ = 0;
  GenerateTopnSolutionsFn gen_ = nullptr;
  AutofuseTilingFn autofuse_tiling_ = nullptr;
  GetTilingDataReprFn repr_ = nullptr;
};

}  // namespace

int main(int argc, char **argv) {
  HostHelperOptions options;
  if (!ParseHostHelperOptions(argc, argv, &options)) {
    return 1;
  }
  if (options.dynamic_shape_args.size() != kDynamicShapeArgCount) {
    std::cerr << "dynamic shape arg count mismatch: helper expects " << kDynamicShapeArgCount << ", got "
              << options.dynamic_shape_args.size() << std::endl;
    return 1;
  }
  TypedHostRunner<kDynamicShapeArgCount> runner(options.dynamic_shape_args);
  return RunHostCheck(options, &runner);
}
