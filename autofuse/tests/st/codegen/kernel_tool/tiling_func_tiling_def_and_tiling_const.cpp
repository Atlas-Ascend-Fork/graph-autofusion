/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autofuse_tiling_func_common.h"
#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/tiling_context.h"
#endif
extern "C" size_t GetTilingDataSize() {
  return sizeof(AutofuseTilingData);
}

uint64_t GetWorkspaceSize(const AutofuseTilingData &t) {
  using namespace optiling;
  uint64_t ws_size = 0;
  if (t.tiling_key == 0) {
    ws_size += 0;
  }

  ws_size = (ws_size + 512 - 1) / 512 * 512;
  return ws_size;
}

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};
constexpr ResLimit g_no_limit_res = {1, 48, 0, 192 * 1024, {}};
extern "C" int64_t AutofuseTiling(uint32_t s2, uint32_t s3, AutofuseTilingData *tiling, uint64_t *workspaceSize,
                                  uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size) {
  tiling->set_s2(s2);
  tiling->set_s3(s3);
  tiling->set_block_dim(aiv_num);
  tiling->set_ub_size(ub_size);
  if (!optiling::GetTiling(*tiling, -1)) {
    return -1;
  }
  *blockDim = tiling->get_block_dim();
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}
extern "C" int64_t AutofuseTilingWithConfig(const char *config_file, uint32_t s2, uint32_t s3,
                                            AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim,
                                            ResLimit *res_limit = nullptr) {
  const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  tiling->set_s2(s2);
  tiling->set_s3(s3);
  tiling->set_block_dim(limit->aiv_num);
  tiling->set_ub_size(limit->ub_size);
  if (!optiling::GetTiling(*tiling, -1)) {
    return -1;
  }
  *blockDim = tiling->get_block_dim();
  using namespace optiling;
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}

#ifndef __CCE_KT_TEST__
extern "C" bool AutofuseIsStaticShape() {
  return false;
}
extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t) {
  if (t.tiling_key == 0) {
    return 0;
  }
  return -1;
}

namespace gert {
class TilingSymbolEvalContext : public TilingContext {
 public:
  const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
    auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
    if (tensor == nullptr) {
      return nullptr;
    }
    return tensor;
  }
};

class SymbolTilingParseContext : public KernelContext {
 public:
  fe::PlatFormInfos *GetPlatFormInfos() const {
    auto platform = GetInputValue<fe::PlatFormInfos *>(0);
    if (platform == nullptr) {
      return nullptr;
    }
    return platform;
  }
};
}  // namespace gert
struct AfTilingParseData {
  uint32_t aiv_num;
  uint64_t ub_size;
};
extern "C" ge::graphStatus TilingParse(gert::SymbolTilingParseContext *context) {
  auto platform = context->GetPlatFormInfos();
  if (platform == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto ascendc_platform = platform_ascendc::PlatformAscendC(platform);
  uint32_t platform_core_num = ascendc_platform.GetCoreNumAiv();
  uint32_t aiv_num = 0;
  uint64_t ub_size = (184 * 1024);
  aiv_num = platform_core_num;
  ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto extend_context = reinterpret_cast<gert::KernelContext *>(context);
  auto tiling_parse_data_av = extend_context->GetOutput(0);
  if (tiling_parse_data_av == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto tiling_parse_data_ptr = new (std::nothrow) uint8_t[sizeof(AfTilingParseData)];
  if (tiling_parse_data_ptr == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling_parse_data_av->SetWithDefaultDeleter<uint8_t[]>(tiling_parse_data_ptr);
  auto tiling_parse_data = extend_context->GetOutputPointer<AfTilingParseData *>(0);
  (*tiling_parse_data)->aiv_num = aiv_num;
  ub_size -=
      (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950 && ub_size % 1024 == 0) ? 256 : 0;
  (*tiling_parse_data)->ub_size = ub_size;
  return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context) {
  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);
  auto input_data_num = extend_context->GetInputValue<size_t>(0U);
  auto parse = extend_context->GetInputValue<AfTilingParseData *>(input_data_num + 1);
  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  auto s3 = [&]() -> int64_t {
    const auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  uint64_t workspace_size;
  uint32_t block_dim;
  static const char *config_file = "./autofuse_pointwise_0__abs__add_config.txt";
  auto tiling_data = context->GetTilingData<AutofuseTilingData>();
  ResLimit limit;
  limit.aiv_num = parse->aiv_num;
  limit.ub_size = (uint32_t)parse->ub_size;
  auto ret = AutofuseTilingWithConfig(config_file, s2, s3, tiling_data, &workspace_size, &block_dim, &limit);
  context->SetBlockDim(block_dim);
  *context->GetWorkspaceSizes(1) = workspace_size;

  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
  return ret;
}

extern "C" ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context) {
  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);
  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);
  if (symbol_src_vec == nullptr) {
    return ge::GRAPH_FAILED;
  }

  if (symbol_src_vec->GetCapacity() < 2) {
    return ge::GRAPH_FAILED;
  }

  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_src_vec->MutableData()[0] = s2;

  auto s3 = [&]() -> int64_t {
    const auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  symbol_src_vec->MutableData()[1] = s3;

  symbol_src_vec->SetSize(2);
  return ge::GRAPH_SUCCESS;
}
extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info,
                                              size_t size) {
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;
  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_info += ("s2: " + std::to_string(s2));

  auto s3 = [&]() -> int64_t {
    const auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  symbol_info += (", s3: " + std::to_string(s3));

  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
#endif

#ifndef __CCE_KT_TEST__
#endif
