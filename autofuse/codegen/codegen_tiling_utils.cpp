/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling_utils.h"
#include "common_utils.h"
#include "common/platform_context.h"

#include <cstdint>
#include <algorithm>
#include <limits>
#include "common_utils.h"

namespace codegen {
std::string GenCvStageSizeExpr(const af::Expression &size, const std::string &tiling_data) {
  // Match ProcessSingleGroup/ProcessMultiGroup: constants are not serialized.
  if (size.IsConstExpr()) {
    return af::SymbolicUtils::AsNumerDenomToString(size);
  }
  return tiling_data + "." + size.Str().get();
}

// Preserve the existing first-inner (Inductor) / last-inner (GE) selection.
void AppendCvStageSizeAccessor(const af::AscGraph &graph, bool first_inner, std::stringstream &ss) {
  af::Expression stage_size = af::Symbol(0);
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->type == ascir::Axis::kAxisTypeTileInner) {
      stage_size = axis->size;
      if (first_inner) {
        break;
      }
    }
  }
  ss << "#define GET_CV_STAGE_SIZE(tiling_data) ";
  if (stage_size.IsConstExpr()) {
    // Keep the argument used even when the budget is folded to a constant.
    ss << "(static_cast<void>(tiling_data), " << GenCvStageSizeExpr(stage_size, "") << ")\n";
  } else {
    ss << GenCvStageSizeExpr(stage_size, "(tiling_data)") << "\n";
  }
}

ge::AscNodePtr GetBmmInputView(const ge::AscNodePtr &node) {
  if (node == nullptr || node->GetType() != "BatchMatMul" || node->GetInDataNodesSize() != 2U) {
    return nullptr;
  }
  const auto input = node->GetInDataAnchor(1);
  if (input == nullptr || input->GetPeerOutAnchor() == nullptr) {
    return nullptr;
  }
  const auto load = std::dynamic_pointer_cast<ge::AscNode>(input->GetPeerOutAnchor()->GetOwnerNode());
  if (load == nullptr || load->GetType() != "Load" || load->GetInDataNodesSize() != 1U ||
      (*load->GetInDataNodes().begin())->GetType() != "Data") {
    return nullptr;
  }
  const auto &attr = load->outputs[0].attr;
  if (attr.repeats.size() != 3U || attr.strides.size() != 3U ||
      !std::all_of(attr.repeats.begin(), attr.repeats.end() - 1, [](const af::Expression &dim) {
        int64_t value = 0;
        return !dim.GetConstValue(value) || value > 1;
      })) {
    return nullptr;
  }
  int64_t batch = 0;
  int64_t n = 0;
  (void)attr.repeats[0].GetConstValue(batch);
  if (attr.repeats[2].GetConstValue(n) && n <= 0) {
    return nullptr;
  }
  const auto type_bytes = af::GetSizeByDataType(attr.dtype);
  // Leave at least two rows in each 64 KiB prologue buffer.
  constexpr int64_t kMaxInputRowBytes = 32 * 1024;
  if (type_bytes <= 0 ||
      (n > 0 && (n > kMaxInputRowBytes / type_bytes ||
                 (batch > 0 && (batch - 1) > std::numeric_limits<uint32_t>::max() / (n * type_bytes))))) {
    return nullptr;
  }
  if ((n != 1 && af::SymbolicUtils::StaticCheckEq(attr.strides[2], af::sym::kSymbolOne) != af::TriBool::kTrue) ||
      af::SymbolicUtils::StaticCheckEq(attr.strides[0], attr.repeats[2]) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(attr.strides[1], attr.repeats[0] * attr.repeats[2]) != af::TriBool::kTrue) {
    return nullptr;
  }
  return load;
}

ge::AscNodePtr FindBmmInputView(const ascir::FusedScheduledResult &fused) {
  if (!ascgen_utils::IsCubeFusedScheduled(fused) || ascgen_utils::IsConv2DFusedScheduled(fused)) {
    return nullptr;
  }
  for (const auto &results : fused.node_idx_to_scheduled_results) {
    for (const auto &result : results) {
      for (const auto &group : result.schedule_groups) {
        for (const auto &graph : group.impl_graphs) {
          for (const auto &node : graph.GetAllNodes()) {
            const auto load = GetBmmInputView(node);
            if (load != nullptr) {
              return load;
            }
          }
        }
      }
    }
  }
  return nullptr;
}

bool IsRegisterBmmInputPackSupported() {
  const auto impl = ascgen_utils::GetAscIrCodegenImpl("Transpose");
  return impl != nullptr && impl->GetApiCallName() == "TransposeRegApiCall";
}

bool CanPackBmmInput(const ge::AscNodePtr &view) {
  if (view == nullptr || !IsRegisterBmmInputPackSupported()) {
    return false;
  }
  const auto &attr = view->outputs[0].attr;
  int64_t n = 0;
  const auto bytes = af::GetSizeByDataType(attr.dtype);
  return attr.repeats.size() == 3U && (!attr.repeats.back().GetConstValue(n) || n == 1) &&
         (bytes == sizeof(int16_t) || bytes == sizeof(int32_t));
}

using namespace ascgen_utils;

std::string GetKernelTaskType(const ascir::FusedScheduledResult &schedule_results) {
  if (ascgen_utils::IsJustCubeFixpip(schedule_results)) {
    return kKernelTaskTypeAICOnly;
  } else if (ascgen_utils::IsCubeFusedScheduled(schedule_results)) {
    if (ascgen_utils::IsConv2DFusedScheduled(schedule_results)) {
      std::string npu_arch;
      GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
      if (npu_arch == "5102") {
        return kKernelTaskTypeMixAICOneOne;
      } else {
        return kKernelTaskTypeMixAICOneTwo;
      }
    } else {
      return kKernelTaskTypeMixAICOneTwo;
    }
  }
  return schedule_results.workspace_nodes.size() != 0 ? kKernelTaskTypeMixAIVOneZero : kKernelTaskTypeAIVOnly;
}

void AppendCvCoreLimitHelperDefs(std::stringstream &ss, const ascir::FusedScheduledResult &result) {
  const auto kernel_type = GetKernelTaskType(result);
  const uint32_t task_ratio = kernel_type == kKernelTaskTypeMixAICOneOne ? 1U : 2U;
  ss << "#include <mutex>\n";
  ss << "constexpr uint32_t kCvTaskRatio = " << task_ratio << "U;\n";
  ss << R"(
// Resource limits are resolved before either tiler assigns work.
inline bool NormalizeCvCoreLimit(uint32_t &aic_num, uint32_t &aiv_num,
                                 uint32_t platform_aic, uint32_t platform_aiv) {
  const uint32_t aic_limit = aic_num == 0U ? platform_aic : std::min(aic_num, platform_aic);
  const uint32_t aiv_limit = aiv_num == 0U ? platform_aiv : std::min(aiv_num, platform_aiv);
)";
  if (kernel_type == kKernelTaskTypeAICOnly) {
    ss << "  aic_num = aic_limit;\n  aiv_num = aiv_limit;\n  return aic_num != 0U;\n}\n";
    return;
  }
  ss << R"(
  const uint32_t groups = std::min(aic_limit, aiv_limit / kCvTaskRatio);
  if (groups == 0U) {
    return false;
  }
  aic_num = groups;
  aiv_num = groups * kCvTaskRatio;
  return true;
}

// GE also tiles static CV nodes after compilation. Export the actual kernel ratio
// so that its Cube and Vector tilers use the same resource budget.
extern "C" bool GetCvCoreLimits(uint32_t *aic_num, uint32_t *aiv_num, uint32_t *task_ratio) {
  if (aic_num == nullptr || aiv_num == nullptr || task_ratio == nullptr ||
      !NormalizeCvCoreLimit(*aic_num, *aiv_num, *aic_num, *aiv_num)) {
    return false;
  }
  *task_ratio = kCvTaskRatio;
  return true;
}
)";
}

namespace {
// Native key: Tensor API (2), BroadcastBatch (5), without transpose bits.
constexpr int64_t kTensorBroadcastBatchTilingKey = 1282;

void AppendCvSafetyAivOnlyModeDef(std::stringstream &ss, bool is_batch, bool is_device) {
  ss << "static inline " << (is_device ? "__aicore__ " : "")
     << "constexpr bool is_cv_safety_aiv_only_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key : {513, 8193}) {
      ss << "      case " << key << ":" << std::endl;
    }
  } else {
    for (auto key : {8193, 8209, 8257, 8273, 12289, 12305, 12353, 12369, 16449}) {
      ss << "      case " << key << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyBlockIdxScheduledModeDef(std::stringstream &ss, bool is_batch, bool is_device) {
  ss << "static inline " << (is_device ? "__aicore__ " : "")
     << "constexpr bool is_cv_safety_blockidx_scheduled_mode(int64_t tiling_key) {" << std::endl;
  // Native Stream-K must notify paired AIVs even on AICs with no MM task.
  // Keep every launched AIC in the native mixed kernel without changing its usedCoreNum.
  ss << "    const int64_t model = (tiling_key >> 12) & 0xF;" << std::endl;
  ss << "    if ((model == 1 || model == 7) && is_cv_safety_mix_mode(tiling_key)) { return true; }" << std::endl;
  if (is_batch) {
    ss << "    if ((tiling_key & ~0xF0LL) == " << kTensorBroadcastBatchTilingKey << "LL) {" << std::endl;
    ss << "      return true;" << std::endl;
    ss << "    }" << std::endl;
  }
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key : {257, 273, 321, 337, 769, 785, 833, 849, 2097409, 2097425, 2097473, 2097489}) {
      ss << "      case " << key << ":" << std::endl;
      ss << "      case " << (key + 1) << ":" << std::endl;  // Tensor API uses the same block-index scheduling.
    }
    ss << "        return true;" << std::endl;
  }
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvSafetyMixModeDef(std::stringstream &ss, bool is_batch, bool is_device) {
  ss << "static inline " << (is_device ? "__aicore__ " : "")
     << "constexpr bool is_cv_safety_mix_mode(int64_t tiling_key) {" << std::endl;
  ss << "    switch (tiling_key) {" << std::endl;
  if (is_batch) {
    for (auto key :
         {513, 4097, 4113, 4161, 4177, 8193, 2097409, 2097425, 2097473, 2097489, 2101249, 2101265, 2101313, 2101329}) {
      ss << "      case " << key << ":" << std::endl;
      ss << "      case " << (key + 1) << ":" << std::endl;
    }
  } else {
    for (auto key : {4097,    4113,    4161,    4177,    8193,    8209,    8257,    8273,    12289,   12305,   12353,
                     12369,   16449,   28673,   28689,   28737,   28753,   1048577, 1048593, 1048641, 1048657, 1179649,
                     1179665, 1179713, 1179729, 2097153, 2097169, 2097217, 2097233, 2101249, 2101265, 2101313, 2101329,
                     2125825, 2125841, 2125889, 2125905, 2228225, 2228241, 2228289, 2228305}) {
      ss << "      case " << key << ":" << std::endl;
      ss << "      case " << (key + 1) << ":" << std::endl;
    }
  }
  ss << "        return true;" << std::endl;
  ss << "      default:" << std::endl;
  ss << "        return false;" << std::endl;
  ss << "    }" << std::endl;
  ss << "}" << std::endl;
}
}  // namespace

void AppendCvBaseAlignHelperDefs(std::stringstream &ss) {
  ss << std::endl;
  ss << "static int32_t g_basen_basem_align = 0;" << std::endl << std::endl;
  ss << "int32_t get_g_basen_basem_align() {" << std::endl;
  ss << "  return g_basen_basem_align;" << std::endl;
  ss << "}" << std::endl << std::endl;
  ss << "void set_g_basen_basem_align(int32_t value) {" << std::endl;
  ss << "  g_basen_basem_align = value;" << std::endl;
  ss << "}" << std::endl;
}

void AppendCvPlatformSelectionDefs(std::stringstream &ss, bool avoid_tensor_api, bool a_trans, bool b_trans) {
  // Match native MM/BMM capabilities in both host and device compilation.
  ss << "#if defined(__FIXED_POINT_ONLY_CUBE_TO_L0C__) && __FIXED_POINT_ONLY_CUBE_TO_L0C__\n"
     << "#define CV_TENSOR_API_SUPPORTED 0\n#else\n#define CV_TENSOR_API_SUPPORTED 1\n#endif\n"
     << "#define CV_AVOID_TENSOR_API " << (avoid_tensor_api ? 1 : 0) << "\n"
     << "#define CV_MM_DISPATCH_SUPPORTED(API, MD, FL, LO) \\\n"
     << "  (CV_TENSOR_API_SUPPORTED ? \\\n"
     << "   ((API) != 1 || CV_AVOID_TENSOR_API || (MD) != 0) : \\\n"
     << "   ((API) == 1 && (MD) == 0 && (LO) == 0))\n"
     << "#define CV_BMM_DISPATCH_SUPPORTED(API, MD, FL, LO, ITER) \\\n"
     << "  (CV_TENSOR_API_SUPPORTED ? \\\n"
     << "   ((API) != 1 || CV_AVOID_TENSOR_API || (FL) == 0) : \\\n"
     << "   ((MD) == 0 && (LO) == 0 && (((API) == 0 && (ITER) == 0) || \\\n"
     << "    ((API) == 1 && ((ITER) == 0 || (ITER) == 1)))))\n";
  // Choices include native normalization: K=0/ToMul/MergeBatch may clear transpose; unit-N B may set it.
  ss << "#if defined(INDUCTOR_CV_FUSION) && CV_TENSOR_API_SUPPORTED && !CV_AVOID_TENSOR_API\n"
     << "#define CV_A_TRANS_CHOICES MAT_MUL_NO_TRANS" << (a_trans ? ", MAT_MUL_TRANS" : "") << "\n"
     << "#define CV_B_TRANS_CHOICES MAT_MUL_NO_TRANS" << (b_trans ? ", MAT_MUL_TRANS" : "") << "\n"
     << "#define CV_TRANSPOSE_SUPPORTED(AT, BT) (((AT) == 0 || " << (a_trans ? 1 : 0) << ") && ((BT) == 0 || "
     << (b_trans ? 1 : 0) << "))\n"
     << "#else\n"
     << "#define CV_A_TRANS_CHOICES MAT_MUL_NO_TRANS, MAT_MUL_TRANS\n"
     << "#define CV_B_TRANS_CHOICES MAT_MUL_NO_TRANS, MAT_MUL_TRANS\n"
     << "#define CV_TRANSPOSE_SUPPORTED(AT, BT) 1\n"
     << "#endif\n";
}

void AppendCvSafetyMixModeHelperDefs(std::stringstream &ss, bool is_batch, bool is_device) {
  AppendCvSafetyAivOnlyModeDef(ss, is_batch, is_device);
  AppendCvSafetyMixModeDef(ss, is_batch, is_device);
  AppendCvSafetyBlockIdxScheduledModeDef(ss, is_batch, is_device);
  ss << "static inline " << (is_device ? "__aicore__ " : "")
     << "constexpr int is_cv_safety_fusion_mode(int64_t tiling_key) {\n"
        "  return is_cv_safety_aiv_only_mode(tiling_key) ? 2 : (is_cv_safety_mix_mode(tiling_key) ? 1 : 0);\n"
        "}\n";
}

void AppendCvUbSupportedKeyHelperDef(std::stringstream &ss, const std::string &tensor_b_full_load, bool is_device) {
  ss << "static inline " << (is_device ? "__aicore__ " : "")
     << "constexpr bool is_cv_ub_supported_key(int64_t tiling_key) {\n"
        "  const int64_t key = tiling_key & ~0xF0LL;\n"
        "  return (key == 1) || (key == 2) || (key == 0x20001) || (("
     << tensor_b_full_load << ") && (key == 0x20002));\n}\n";
}

}  // namespace codegen
