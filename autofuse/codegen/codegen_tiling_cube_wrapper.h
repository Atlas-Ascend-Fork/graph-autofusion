#include <string>

inline const std::string kCubeKernelTilingWrapperHppValue = R"(
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUBE_KERNEL_TILING_WRAPPER_H
#define CUBE_KERNEL_TILING_WRAPPER_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/types.h"
#include "arch35/mat_mul_tiling_data.h"

namespace ge {
namespace autofuse {

struct TensorInfo {
    std::string param_name;
    std::vector<int64_t> shape;
    std::vector<int64_t> ori_shape;
    std::string dtype;
    std::string format;
    std::string name;
    int64_t range_start = 0;
    int64_t range_end = 0;
};

struct AttrInfo {
    std::string name;
    std::string dtype;
    std::string value_str;
    bool value_bool = false;
    int64_t value_int = 0;
    double value_float = 0.0;
    std::vector<int64_t> value_list_int;
    std::vector<double> value_list_float;
    std::vector<std::string> value_list_str;
    bool is_list = false;
};

struct CompileInfo {
    std::string soc_version;
    std::string core_type;
    std::string device_id;
    std::string op_kernel_lib;
    std::string op_impl_mode;
    int64_t aicore_num = 0;
    int64_t aiv_num = 0;
    std::map<std::string, std::string> extra_info;
};

struct TilingResult {
    std::vector<uint8_t> tiling_data;
    int64_t tiling_key = 0;
    int64_t block_dim = 0;
    int64_t workspace_size = 0;
    uint32_t cube_used_core_num = 1;
    uint32_t cube_base_m = 16;
    uint32_t cube_base_n = 16;
    bool atomic_flag = false;
    std::string error_msg;
    bool success = false;

    BatchMatMulV3BasicTilingData batch_matmul_tiling_data;
    MatMulV3BasicTilingData matmul_basic_tiling_data;
};

extern "C" bool AutofuseDoCubeMatMulTiling(const CompileInfo* compile_info,
                                           const std::vector<TensorInfo>* inputs,
                                           const std::vector<TensorInfo>* outputs,
                                           const std::vector<AttrInfo>* attrs,
                                           bool is_batch,
                                           TilingResult* result);

class CubeKernelTilingWrapper {
public:
    CubeKernelTilingWrapper();
    ~CubeKernelTilingWrapper();

    TilingResult DoMatMulTiling(const CompileInfo& compile_info,
                                const std::vector<TensorInfo>& inputs,
                                const std::vector<TensorInfo>& outputs,
                                const std::vector<AttrInfo>& attrs,
                                bool is_batch = false);

    static void BuildMatMulArgs(const std::vector<TensorInfo>& args_list,
                                int input_num,
                                bool transpose_a,
                                bool transpose_b,
                                std::vector<TensorInfo>& origin_inputs,
                                std::vector<TensorInfo>& origin_outputs,
                                std::vector<TensorInfo>& inputs);
};

} // namespace autofuse
} // namespace ge

#endif // CUBE_KERNEL_TILING_WRAPPER_H
)";

inline const std::string kCubeKernelTilingWrapperCppValue = R"(
#include "autofuse_tiling_func_log.h"
#include "registry/op_impl_space_registry_v2.h"

#include "context_builder/op_tiling_context_builder.h"
#include "context_builder/op_tiling_parse_context_builder.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_kernel_registry.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ge {
namespace autofuse {

namespace {
constexpr size_t kMaxTilingDataSize = 64 * 1024;
constexpr size_t kWorkspaceCapacity = 4096;
const std::vector<uint32_t> kSingleOutputInstanceNum = {1U};

struct OpHostFuncs {
  gert::OpImplRegisterV2::TilingKernelFunc tiling = nullptr;
  gert::OpImplRegisterV2::KernelFunc tiling_parse = nullptr;
  gert::OpImplRegisterV2::CompileInfoCreatorFunc compile_info_creator = nullptr;
  size_t max_tiling_data_size = 0UL;
};

struct CachedOpHostFuncs {
  OpHostFuncs funcs;
  bool loaded = false;
  std::once_flag once;
};

struct OpHostSchema {
  const char *op_type;
};

struct MatMulAttrs {
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  int64_t offset_x = 0;
  int64_t op_impl_mode = 0;
  bool enable_hf32 = false;
  bool has_bias = false;
  bool has_offset_w = false;
  bool has_optional_input_markers = false;
};

struct RuntimeTilingKey {
  std::string soc_version;
  std::string device_id;
  std::string op_type;
  std::string dtype;
  std::string format;
  std::vector<int64_t> input0_shape;
  std::vector<int64_t> input1_shape;
  std::vector<int64_t> input2_shape;
  std::vector<int64_t> input3_shape;
  std::string input2_dtype;
  std::string input3_dtype;
  std::string input2_format;
  std::string input3_format;
  std::vector<int64_t> output_shape;
  bool is_batch = false;
  size_t input_num = 0U;
  bool has_bias = false;
  bool has_offset_w = false;
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  int64_t offset_x = 0;
  int64_t op_impl_mode = 0;
  bool enable_hf32 = false;
  int64_t aicore_num = 0;
  int64_t aiv_num = 0;

  bool operator<(const RuntimeTilingKey &other) const {
    return std::tie(soc_version, device_id, op_type, dtype, format, input0_shape, input1_shape, input2_shape,
                    input3_shape, input2_dtype, input3_dtype, input2_format, input3_format, output_shape, is_batch,
                    input_num, has_bias, has_offset_w, transpose_x1, transpose_x2, offset_x, op_impl_mode, enable_hf32,
                    aicore_num, aiv_num) <
           std::tie(other.soc_version, other.device_id, other.op_type, other.dtype, other.format, other.input0_shape,
                    other.input1_shape, other.input2_shape, other.input3_shape, other.input2_dtype, other.input3_dtype,
                    other.input2_format, other.input3_format, other.output_shape, other.is_batch, other.input_num,
                    other.has_bias, other.has_offset_w, other.transpose_x1, other.transpose_x2, other.offset_x,
                    other.op_impl_mode, other.enable_hf32, other.aicore_num, other.aiv_num);
  }
};

struct CompileState;

struct CompileStateKey {
  std::string soc_version;
  std::string device_id;
  std::string dtype;
  std::string format;
  std::string input2_dtype;
  std::string input3_dtype;
  std::string input2_format;
  std::string input3_format;
  bool is_batch = false;
  size_t input_num = 0U;
  bool has_bias = false;
  bool has_offset_w = false;
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  int64_t offset_x = 0;
  int64_t op_impl_mode = 0;
  bool enable_hf32 = false;
  int64_t aicore_num = 0;
  int64_t aiv_num = 0;

  bool operator<(const CompileStateKey &other) const {
    return std::tie(soc_version, device_id, dtype, format, input2_dtype, input3_dtype, input2_format, input3_format,
                    is_batch, input_num, transpose_x1, transpose_x2, offset_x, op_impl_mode, enable_hf32, aicore_num,
                    aiv_num, has_bias, has_offset_w) <
           std::tie(other.soc_version, other.device_id, other.dtype, other.format, other.input2_dtype,
                    other.input3_dtype, other.input2_format, other.input3_format, other.is_batch, other.input_num,
                    other.transpose_x1, other.transpose_x2, other.offset_x, other.op_impl_mode, other.enable_hf32,
                    other.aicore_num, other.aiv_num, other.has_bias, other.has_offset_w);
  }
};

struct CompileState {
  std::string compile_json;
  fe::PlatFormInfos platform_info;
  void *compile_info_ptr = nullptr;
};

struct TilingRequest {
  const CompileInfo &compile_info;
  const std::vector<TensorInfo> &inputs;
  const std::vector<TensorInfo> &outputs;
  bool is_batch = false;
  const OpHostSchema &schema;
  MatMulAttrs matmul_attrs;
  ge::DataType data_type = ge::DT_UNDEFINED;
  ge::Format format = ge::FORMAT_RESERVED;
};

struct TilingScratch {
  std::unique_ptr<uint8_t[]> tiling_data_holder;
  std::unique_ptr<uint8_t[]> workspace_holder;
  size_t tiling_data_capacity = 0UL;
  std::vector<gert::Tensor *> input_tensors;
  std::vector<gert::Tensor *> output_tensors;

  bool EnsureCapacity(size_t required_tiling_data_capacity, std::string &error_msg, const char *op_type) {
    if (tiling_data_holder == nullptr || required_tiling_data_capacity > tiling_data_capacity) {
      auto new_tiling_data_holder = gert::TilingData::CreateCap(required_tiling_data_capacity);
      if (new_tiling_data_holder == nullptr) {
        error_msg = std::string(op_type) + " tiling data allocation failed";
        return false;
      }
      tiling_data_holder = std::move(new_tiling_data_holder);
      tiling_data_capacity = required_tiling_data_capacity;
    }
    if (workspace_holder == nullptr) {
      workspace_holder = gert::ContinuousVector::Create<size_t>(kWorkspaceCapacity);
      if (workspace_holder == nullptr) {
        error_msg = std::string(op_type) + " workspace allocation failed";
        return false;
      }
    }
    input_tensors.reserve(4U);
    output_tensors.reserve(1U);
    return true;
  }

  gert::TilingData *MutableTilingData() const {
    return reinterpret_cast<gert::TilingData *>(tiling_data_holder.get());
  }

  gert::ContinuousVector *MutableWorkspace() const {
    return reinterpret_cast<gert::ContinuousVector *>(workspace_holder.get());
  }
};

TilingScratch &GetTilingScratch() {
  thread_local TilingScratch scratch;
  return scratch;
}

CachedOpHostFuncs &GetMatMulFuncsCache() {
  static auto *cache = new CachedOpHostFuncs();
  return *cache;
}

CachedOpHostFuncs &GetBatchMatMulFuncsCache() {
  static auto *cache = new CachedOpHostFuncs();
  return *cache;
}

std::mutex &GetCompileStateMutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::map<CompileStateKey, std::shared_ptr<const CompileState>> &GetCompileStateCache() {
  static auto cache = std::make_shared<std::map<CompileStateKey, std::shared_ptr<const CompileState>>>();
  return *cache;
}

std::mutex &GetTilingResultCacheMutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::map<RuntimeTilingKey, TilingResult> &GetTilingResultCache() {
  static auto *cache = new std::map<RuntimeTilingKey, TilingResult>();
  return *cache;
}

const OpHostSchema &GetOpHostSchema(bool is_batch) {
  static const OpHostSchema kMatMulV3Schema{"MatMulV3"};
  static const OpHostSchema kBatchMatMulV3Schema{"BatchMatMulV3"};
  return is_batch ? kBatchMatMulV3Schema : kMatMulV3Schema;
}

ge::DataType DtypeToGeDataType(const std::string &dtype) {
  if (dtype == "float" || dtype == "float32" || dtype == "DT_FLOAT" || dtype == "torch.float32") {
    return ge::DT_FLOAT;
  }
  if (dtype == "float16" || dtype == "half" || dtype == "DT_FLOAT16" || dtype == "torch.float16") {
    return ge::DT_FLOAT16;
  }
  if (dtype == "bfloat16" || dtype == "bf16" || dtype == "DT_BF16" || dtype == "torch.bfloat16") {
    return ge::DT_BF16;
  }
  return ge::DT_UNDEFINED;
}

ge::Format FormatToGeFormat(const std::string &format) {
  if (format.empty() || format == "ND" || format == "FORMAT_ND" || format == "ACL_FORMAT_ND") {
    return ge::FORMAT_ND;
  }
  return ge::FORMAT_RESERVED;
}

bool AttrAsBool(const AttrInfo &attr) {
  if (attr.dtype == "bool") {
    return attr.value_bool;
  }
  if (attr.dtype == "int" || attr.dtype == "int32" || attr.dtype == "int64") {
    return attr.value_int != 0;
  }
  return false;
}

int64_t AttrAsInt(const AttrInfo &attr) {
  if (attr.dtype == "bool") {
    return attr.value_bool ? 1 : 0;
  }
  if (attr.dtype == "int" || attr.dtype == "int32" || attr.dtype == "int64") {
    return attr.value_int;
  }
  return 0;
}

MatMulAttrs ReadMatMulAttrs(const std::vector<AttrInfo> &attrs, bool is_batch) {
  MatMulAttrs result;
  const char *transpose_x1_name = is_batch ? "adj_x1" : "transpose_x1";
  const char *transpose_x2_name = is_batch ? "adj_x2" : "transpose_x2";
  for (const auto &attr : attrs) {
    if (attr.name == transpose_x1_name) {
      result.transpose_x1 = AttrAsBool(attr);
    } else if (attr.name == transpose_x2_name) {
      result.transpose_x2 = AttrAsBool(attr);
    } else if (attr.name == "offset_x") {
      result.offset_x = AttrAsInt(attr);
    } else if (attr.name == "opImplMode") {
      result.op_impl_mode = AttrAsInt(attr);
    } else if (attr.name == "enable_hf32") {
      result.enable_hf32 = AttrAsBool(attr);
    } else if (attr.name == "autofuse_has_bias") {
      result.has_bias = AttrAsBool(attr);
      result.has_optional_input_markers = true;
    } else if (attr.name == "autofuse_has_offset_w") {
      result.has_offset_w = AttrAsBool(attr);
      result.has_optional_input_markers = true;
    }
  }
  result.enable_hf32 = result.enable_hf32 || result.op_impl_mode != 0;
  return result;
}

bool LoadOpHostFuncs(const char *op_type, OpHostFuncs &funcs) {
  auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  if (registry == nullptr) {
    return false;
  }
  const auto *op_impl = registry->GetOpImpl(op_type);
  if (op_impl == nullptr) {
    return false;
  }
  funcs.tiling = op_impl->tiling;
  funcs.tiling_parse = op_impl->tiling_parse;
  funcs.compile_info_creator = op_impl->compile_info_creator;
  funcs.max_tiling_data_size = op_impl->max_tiling_data_size;
  return funcs.tiling != nullptr && funcs.tiling_parse != nullptr && funcs.compile_info_creator != nullptr;
}

const CachedOpHostFuncs &GetOpHostFuncs(bool is_batch, const char *op_type) {
  CachedOpHostFuncs &cache = is_batch ? GetBatchMatMulFuncsCache() : GetMatMulFuncsCache();
  std::call_once(cache.once, [&cache, op_type]() { cache.loaded = LoadOpHostFuncs(op_type, cache.funcs); });
  return cache;
}

bool ParseUint32(const std::string &value, uint32_t &result) {
  if (value.empty()) {
    return false;
  }
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  result = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseUint64(const std::string &value, uint64_t &result) {
  if (value.empty()) {
    return false;
  }
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0') {
    return false;
  }
  result = static_cast<uint64_t>(parsed);
  return true;
}

uint32_t GetCompileDeviceId(const CompileInfo &compile_info) {
  uint32_t device_id = 0;
  (void)ParseUint32(compile_info.device_id, device_id);
  return device_id;
}

std::string GetPlatformString(fe::PlatFormInfos &platform_info, const std::string &label, const std::string &key,
                              const std::string &fallback = "") {
  std::string value;
  if (platform_info.GetPlatformResWithLock(label, key, value) && !value.empty()) {
    return value;
  }
  return fallback;
}

uint64_t GetPlatformUint64(fe::PlatFormInfos &platform_info, const std::string &label, const std::string &key,
                           uint64_t fallback = 0U) {
  uint64_t value = 0U;
  if (ParseUint64(GetPlatformString(platform_info, label, key), value)) {
    return value;
  }
  return fallback;
}

uint64_t GetLocalMemSize(fe::PlatFormInfos &platform_info, fe::LocalMemType mem_type, const std::string &label,
                         const std::string &key) {
  uint64_t size = 0U;
  platform_info.GetLocalMemSize(mem_type, size);
  if (size != 0U) {
    return size;
  }
  return GetPlatformUint64(platform_info, label, key);
}

uint32_t GetPlatformCoreNum(fe::PlatFormInfos &platform_info, const std::string &core_type,
                            const std::string &soc_info_key) {
  uint32_t core_num = platform_info.GetCoreNumByType(core_type);
  if (core_num != 0U) {
    return core_num;
  }
  return static_cast<uint32_t>(GetPlatformUint64(platform_info, "SoCInfo", soc_info_key));
}

void UpdatePlatformCoreNum(const CompileInfo &compile_info, fe::PlatFormInfos &platform_info) {
  uint32_t aic_num = compile_info.aicore_num > 0 ? static_cast<uint32_t>(compile_info.aicore_num)
                                                 : GetPlatformCoreNum(platform_info, "AiCore", "cube_core_cnt");
  uint32_t aiv_num = compile_info.aiv_num > 0 ? static_cast<uint32_t>(compile_info.aiv_num)
                                              : GetPlatformCoreNum(platform_info, "VectorCore", "vector_core_cnt");
  std::map<std::string, std::string> soc_info;
  if (platform_info.GetPlatformResWithLock("SoCInfo", soc_info)) {
    if (aic_num != 0U) {
      soc_info["cube_core_cnt"] = std::to_string(aic_num);
      soc_info["ai_core_cnt"] = std::to_string(aic_num);
    }
    if (aiv_num != 0U) {
      soc_info["vector_core_cnt"] = std::to_string(aiv_num);
    }
    platform_info.SetPlatformResWithLock("SoCInfo", soc_info);
  }
  platform_info.SetCoreNumByCoreType("AiCore");
}

bool FillRuntimePlatformInfo(const CompileInfo &compile_info, fe::PlatFormInfos &platform_info) {
  if (fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(GetCompileDeviceId(compile_info),
                                                                            platform_info, true) != 0U) {
    return false;
  }
  UpdatePlatformCoreNum(compile_info, platform_info);
  return true;
}

std::string JsonEscape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char ch : value) {
    if (ch == '\\' || ch == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(ch);
  }
  return escaped;
}

const char *BoolLiteral(bool value) {
  return value ? "true" : "false";
}

bool HasIntrinsic(fe::PlatFormInfos &platform_info, const std::string &intrinsic_name) {
  std::map<std::string, std::string> intrinsic_res;
  if (platform_info.GetPlatformResWithLock("AICoreintrinsicDtypeMap", intrinsic_res) &&
      intrinsic_res.find(intrinsic_name) != intrinsic_res.end()) {
    return true;
  }
  auto intrinsic_map = platform_info.GetAICoreIntrinsicDtype();
  return intrinsic_map.find(intrinsic_name) != intrinsic_map.end();
}

std::string MakeCubeCompileJson(const CompileInfo &compile_info, fe::PlatFormInfos &platform_info, bool is_batch,
                                bool transpose_x1, bool transpose_x2, int64_t offset_x, int64_t op_impl_mode,
                                bool enable_hf32) {
  const std::string soc_version =
      GetPlatformString(platform_info, "version", "Short_SoC_version", compile_info.soc_version);
  const uint32_t core_num = GetPlatformCoreNum(platform_info, "AiCore", "cube_core_cnt");
  const uint32_t vector_core_num = GetPlatformCoreNum(platform_info, "VectorCore", "vector_core_cnt");
  const uint64_t bt_size = GetPlatformUint64(platform_info, "AICoreSpec", "bt_size");
  const uint64_t ub_size = GetLocalMemSize(platform_info, fe::LocalMemType::UB, "AICoreSpec", "ub_size");
  const uint64_t l2_size = GetLocalMemSize(platform_info, fe::LocalMemType::L2, "SoCInfo", "l2_size");
  const uint64_t l1_size = GetLocalMemSize(platform_info, fe::LocalMemType::L1, "AICoreSpec", "l1_size");
  const uint64_t l0a_size = GetLocalMemSize(platform_info, fe::LocalMemType::L0_A, "AICoreSpec", "l0_a_size");
  const uint64_t l0b_size = GetLocalMemSize(platform_info, fe::LocalMemType::L0_B, "AICoreSpec", "l0_b_size");
  const uint64_t l0c_size = GetLocalMemSize(platform_info, fe::LocalMemType::L0_C, "AICoreSpec", "l0_c_size");
  const std::string load3d_constraints =
      GetPlatformString(platform_info, "AICoreSpec", "load3d_constraints", "unknown");
  std::ostringstream ss;
  ss << "{\"_pattern\":\"MatMul\",\"attrs\":{";
  ss << "\"transpose_a\":" << BoolLiteral(transpose_x1) << ",";
  ss << "\"transpose_b\":" << BoolLiteral(transpose_x2) << ",";
  ss << "\"offset_x\":" << offset_x << ",";
  ss << (is_batch ? "\"enable_hf32\":" : "\"opImplMode\":") << (is_batch ? (enable_hf32 ? 1 : 0) : op_impl_mode);
  ss << "},\"binary_attrs\":{\"bias_flag\":false,\"nd_flag\":true,\"split_k_flag\":false,";
  ss << "\"zero_flag\":false,\"weight_nz\":false,\"l2_size\":" << l2_size << "},\"binary_mode_flag\":true,";
  ss << "\"block_dim\":{\"CORE_NUM\":" << core_num << ",\"vector_core_cnt\":" << vector_core_num << "},";
  ss << "\"corerect_range_flag\":null,\"dynamic_mode\":\"dynamic_mkn\",\"fused_double_operand_num\":0,";
  ss << "\"hardware_info\":{\"BT_SIZE\":" << bt_size << ",\"load3d_constraints\":\"" << JsonEscape(load3d_constraints)
     << "\",";
  ss << "\"Intrinsic_fix_pipe_l0c2out\":" << BoolLiteral(HasIntrinsic(platform_info, "Intrinsic_fix_pipe_l0c2out"))
     << ",";
  ss << "\"Intrinsic_data_move_l12ub\":" << BoolLiteral(HasIntrinsic(platform_info, "Intrinsic_data_move_l12ub"))
     << ",";
  ss << "\"Intrinsic_data_move_l0c2ub\":" << BoolLiteral(HasIntrinsic(platform_info, "Intrinsic_data_move_l0c2ub"))
     << ",";
  ss << "\"Intrinsic_data_move_out2l1_nd2nz\":"
     << BoolLiteral(HasIntrinsic(platform_info, "Intrinsic_data_move_out2l1_nd2nz")) << ",";
  ss << "\"Intrinsic_data_move_l12bt\":" << BoolLiteral(HasIntrinsic(platform_info, "Intrinsic_data_move_l12bt"))
     << ",";
  ss << "\"UB_SIZE\":" << ub_size << ",\"L2_SIZE\":" << l2_size << ",\"L1_SIZE\":" << l1_size << ",";
  ss << "\"L0A_SIZE\":" << l0a_size << ",\"L0B_SIZE\":" << l0b_size << ",\"L0C_SIZE\":" << l0c_size << ",";
  ss << "\"CORE_NUM\":" << core_num << ",\"vector_core_cnt\":" << vector_core_num << ",";
  ss << "\"socVersion\":\"" << JsonEscape(soc_version) << "\"},\"format_a\":\"ND\",\"format_b\":\"ND\",";
  ss << "\"repo_range\":{},\"repo_seeds\":{}}";
  return ss.str();
}

std::vector<uint32_t> MakeMatMulInputInstanceNum(bool has_bias, bool has_offset_w) {
  return {1U, 1U, has_bias ? 1U : 0U, has_offset_w ? 1U : 0U};
}

std::vector<const TensorInfo *> BuildMatMulInputSlots(const std::vector<TensorInfo> &inputs, const MatMulAttrs &attrs) {
  std::vector<const TensorInfo *> slots = {&inputs[0], &inputs[1], nullptr, nullptr};
  size_t input_index = 2U;
  if (attrs.has_bias && input_index < inputs.size()) {
    slots[2U] = &inputs[input_index++];
  }
  if (attrs.has_offset_w && input_index < inputs.size()) {
    slots[3U] = &inputs[input_index++];
  }
  return slots;
}

std::unique_ptr<CompileState> BuildCompileState(const CompileInfo &compile_info, const OpHostSchema &schema,
                                                 const OpHostFuncs &funcs, bool is_batch, ge::DataType data_type,
                                                 ge::Format format, const MatMulAttrs &attrs,
                                                 const std::vector<TensorInfo> &inputs, std::string &error_msg) {
  auto state = std::make_unique<CompileState>();
  if (!FillRuntimePlatformInfo(compile_info, state->platform_info)) {
    error_msg = std::string(schema.op_type) + " platform info setup failed";
    return nullptr;
  }
  state->compile_info_ptr = funcs.compile_info_creator();
  if (state->compile_info_ptr == nullptr) {
    error_msg = std::string(schema.op_type) + " compile info creation failed";
    return nullptr;
  }
  state->compile_json = MakeCubeCompileJson(compile_info, state->platform_info, is_batch, attrs.transpose_x1,
                                            attrs.transpose_x2, attrs.offset_x, attrs.op_impl_mode, attrs.enable_hf32);
  gert::OpTilingParseContextBuilder parse_builder;
  const auto input_slots = BuildMatMulInputSlots(inputs, attrs);
  parse_builder.OpType(schema.op_type)
      .OpName(schema.op_type)
      .IOInstanceNum(MakeMatMulInputInstanceNum(attrs.has_bias, attrs.has_offset_w), kSingleOutputInstanceNum)
      .InputTensorDesc(0, data_type, format, format)
      .InputTensorDesc(1, data_type, format, format);
  size_t input_desc_index = 2U;
  if (input_slots[2U] != nullptr) {
    const auto input_format = FormatToGeFormat(input_slots[2U]->format);
    parse_builder.InputTensorDesc(input_desc_index++, DtypeToGeDataType(input_slots[2U]->dtype), input_format,
                                  input_format);
  }
  if (input_slots[3U] != nullptr) {
    const auto input_format = FormatToGeFormat(input_slots[3U]->format);
    parse_builder.InputTensorDesc(input_desc_index++, DtypeToGeDataType(input_slots[3U]->dtype), input_format,
                                  input_format);
  }
  auto parse_holder = parse_builder.OutputTensorDesc(0, data_type, format, format)
                          .CompiledJson(state->compile_json.c_str())
                          .CompiledInfo(state->compile_info_ptr)
                          .PlatformInfo(const_cast<fe::PlatFormInfos *>(&state->platform_info))
                          .Build();
  auto *parse_ctx = reinterpret_cast<gert::KernelContext *>(parse_holder.GetContext());
  const auto parse_ret = parse_ctx == nullptr ? ge::GRAPH_FAILED : funcs.tiling_parse(parse_ctx);
  if (parse_ctx == nullptr || parse_ret != ge::GRAPH_SUCCESS) {
    error_msg = std::string(schema.op_type) + " tiling parse failed";
    return nullptr;
  }
  return state;
}

CompileStateKey MakeCompileStateKey(const CompileInfo &compile_info, bool is_batch, size_t input_num,
                                     const std::vector<TensorInfo> &inputs, const std::string &dtype,
                                     const std::string &format, const MatMulAttrs &attrs) {
  std::string input2_dtype;
  std::string input3_dtype;
  std::string input2_format;
  std::string input3_format;
  if (inputs.size() > 2U) {
    input2_dtype = inputs[2U].dtype;
    input2_format = inputs[2U].format;
  }
  if (inputs.size() > 3U) {
    input3_dtype = inputs[3U].dtype;
    input3_format = inputs[3U].format;
  }
  return CompileStateKey{compile_info.soc_version,
                         compile_info.device_id,
                          dtype,
                          format,
                          input2_dtype,
                          input3_dtype,
                          input2_format,
                          input3_format,
                          is_batch,
                          input_num,
                          attrs.has_bias,
                          attrs.has_offset_w,
                          attrs.transpose_x1,
                         attrs.transpose_x2,
                         attrs.offset_x,
                         attrs.op_impl_mode,
                         attrs.enable_hf32,
                         compile_info.aicore_num,
                         compile_info.aiv_num};
}

std::shared_ptr<const CompileState> GetCompileState(const CompileInfo &compile_info, const OpHostSchema &schema,
                                                    const OpHostFuncs &funcs, bool is_batch, ge::DataType data_type,
                                                    ge::Format format, const MatMulAttrs &attrs,
                                                    const std::string &dtype, const std::string &format_name,
                                                    const std::vector<TensorInfo> &inputs,
                                                    std::string &error_msg) {
  const CompileStateKey key = MakeCompileStateKey(compile_info, is_batch, inputs.size(), inputs, dtype, format_name,
                                                  attrs);
  {
    std::lock_guard<std::mutex> lock(GetCompileStateMutex());
    auto &cache = GetCompileStateCache();
    const auto it = cache.find(key);
    if (it != cache.end() && it->second != nullptr) {
      return it->second;
    }
  }

  auto state = BuildCompileState(compile_info, schema, funcs, is_batch, data_type, format, attrs, inputs, error_msg);
  if (state == nullptr) {
    return nullptr;
  }

  auto cached_state = std::shared_ptr<const CompileState>(std::move(state));
  std::lock_guard<std::mutex> lock(GetCompileStateMutex());
  auto &cache_state = GetCompileStateCache()[key];
  if (cache_state == nullptr) {
    cache_state = std::move(cached_state);
  }
  return cache_state;
}

void FillShape(gert::Shape &shape, const std::vector<int64_t> &dims) {
  shape.SetScalar();
  for (const auto dim : dims) {
    shape.AppendDim(dim);
  }
}

std::vector<int64_t> GetRuntimeShape(const TensorInfo &tensor) {
  return tensor.shape.empty() ? tensor.ori_shape : tensor.shape;
}

TilingRequest MakeTilingRequest(const CompileInfo &compile_info, const std::vector<TensorInfo> &inputs,
                                const std::vector<TensorInfo> &outputs, const std::vector<AttrInfo> &attrs,
                                bool is_batch) {
  const auto &schema = GetOpHostSchema(is_batch);
  MatMulAttrs matmul_attrs = ReadMatMulAttrs(attrs, is_batch);
  return TilingRequest{compile_info, inputs, outputs, is_batch, schema, matmul_attrs,
                       DtypeToGeDataType(inputs[0].dtype), FormatToGeFormat(inputs[0].format)};
}

RuntimeTilingKey MakeRuntimeTilingKey(const TilingRequest &request) {
  RuntimeTilingKey key;
  key.soc_version = request.compile_info.soc_version;
  key.device_id = request.compile_info.device_id;
  key.op_type = request.schema.op_type == nullptr ? std::string() : request.schema.op_type;
  key.input_num = request.inputs.size();
  key.has_bias = request.matmul_attrs.has_bias;
  key.has_offset_w = request.matmul_attrs.has_offset_w;
  if (!request.inputs.empty()) {
    key.dtype = request.inputs[0].dtype;
    key.format = request.inputs[0].format;
    key.input0_shape = GetRuntimeShape(request.inputs[0]);
  }
  if (request.inputs.size() > 1) {
    key.input1_shape = GetRuntimeShape(request.inputs[1]);
  }
  if (request.inputs.size() > 2) {
    key.input2_shape = GetRuntimeShape(request.inputs[2]);
    key.input2_dtype = request.inputs[2].dtype;
    key.input2_format = request.inputs[2].format;
  }
  if (request.inputs.size() > 3) {
    key.input3_shape = GetRuntimeShape(request.inputs[3]);
    key.input3_dtype = request.inputs[3].dtype;
    key.input3_format = request.inputs[3].format;
  }
  if (!request.outputs.empty()) {
    key.output_shape = GetRuntimeShape(request.outputs[0]);
  }
  key.is_batch = request.is_batch;
  key.transpose_x1 = request.matmul_attrs.transpose_x1;
  key.transpose_x2 = request.matmul_attrs.transpose_x2;
  key.offset_x = request.matmul_attrs.offset_x;
  key.op_impl_mode = request.matmul_attrs.op_impl_mode;
  key.enable_hf32 = request.matmul_attrs.enable_hf32;
  key.aicore_num = request.compile_info.aicore_num;
  key.aiv_num = request.compile_info.aiv_num;
  return key;
}

bool TryGetCachedTilingResult(const RuntimeTilingKey &key, TilingResult *result) {
  if (result == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(GetTilingResultCacheMutex());
  const auto &cache = GetTilingResultCache();
  const auto it = cache.find(key);
  if (it == cache.end() || !it->second.success) {
    return false;
  }
  *result = it->second;
  return true;
}

void CacheTilingResult(const RuntimeTilingKey &key, const TilingResult &result) {
  if (!result.success) {
    return;
  }
  std::lock_guard<std::mutex> lock(GetTilingResultCacheMutex());
  GetTilingResultCache()[key] = result;
}

gert::StorageShape MakeStorageShape(const TensorInfo &tensor) {
  gert::StorageShape storage_shape;
  const auto runtime_shape = GetRuntimeShape(tensor);
  FillShape(storage_shape.MutableOriginShape(), tensor.ori_shape.empty() ? runtime_shape : tensor.ori_shape);
  FillShape(storage_shape.MutableStorageShape(), runtime_shape);
  return storage_shape;
}

bool ValidateTilingRequest(const CompileInfo *compile_info, const std::vector<TensorInfo> *inputs,
                           const std::vector<TensorInfo> *outputs, const std::vector<AttrInfo> *attrs,
                           TilingResult *result) {
  return compile_info != nullptr && inputs != nullptr && outputs != nullptr && attrs != nullptr && result != nullptr &&
         inputs->size() >= 2 && inputs->size() <= 4 && !outputs->empty();
}

bool IsSupportedTilingTensorDesc(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                 ge::DataType data_type, ge::Format format) {
  if (data_type == ge::DT_UNDEFINED || format == ge::FORMAT_RESERVED || DtypeToGeDataType(outputs[0].dtype) != data_type ||
      FormatToGeFormat(outputs[0].format) != format) {
    return false;
  }
  for (const auto &input : inputs) {
    if (DtypeToGeDataType(input.dtype) == ge::DT_UNDEFINED || FormatToGeFormat(input.format) == ge::FORMAT_RESERVED) {
      return false;
    }
  }
  return DtypeToGeDataType(inputs[1].dtype) == data_type && FormatToGeFormat(inputs[1].format) == format;
}

bool IsOptionalInputSlotsValid(const TilingRequest &request) {
  const size_t expected_input_num = 2U + (request.matmul_attrs.has_bias ? 1U : 0U) +
                                    (request.matmul_attrs.has_offset_w ? 1U : 0U);
  return request.inputs.size() == expected_input_num;
}

template <typename TilingContext>
void FillTilingResultFromContext(TilingContext *tiling_ctx, TilingResult &result) {
  auto *raw_tiling_data = tiling_ctx->GetRawTilingData();
  const size_t tiling_data_len = raw_tiling_data->GetDataSize();
  result.tiling_data.assign(reinterpret_cast<const uint8_t *>(raw_tiling_data->GetData()),
                            reinterpret_cast<const uint8_t *>(raw_tiling_data->GetData()) + tiling_data_len);
  result.tiling_key = static_cast<int64_t>(tiling_ctx->GetTilingKey());
  result.block_dim = static_cast<int64_t>(tiling_ctx->GetBlockDim());
  const size_t workspace_num = tiling_ctx->GetWorkspaceNum();
  auto *workspace_sizes = workspace_num > 0 ? tiling_ctx->GetWorkspaceSizes(workspace_num) : nullptr;
  result.workspace_size = workspace_sizes == nullptr ? 0 : static_cast<int64_t>(workspace_sizes[0]);
  result.success = true;
}

uint32_t PositiveOrDefault(int64_t value, uint32_t default_value) {
  return value > 0 ? static_cast<uint32_t>(value) : default_value;
}

template <typename T>
bool CopyExactTilingData(const TilingResult &result, const size_t tiling_data_len, T &tiling_data) {
  if (tiling_data_len != sizeof(T)) {
    return false;
  }
  std::memcpy(&tiling_data, result.tiling_data.data(), sizeof(T));
  return true;
}

void FillMetaFromBasicTiling(TilingResult &result, const MatMulV3BasicTilingData &tiling_data) {
  result.matmul_basic_tiling_data = tiling_data;
  result.cube_used_core_num = std::max(tiling_data.usedCoreNum, 1U);
  result.cube_base_m = std::max(tiling_data.baseM, 1U);
  result.cube_base_n = std::max(tiling_data.baseN, 1U);
}

void FillMetaFromBatchBasicTiling(TilingResult &result, const BatchMatMulV3BasicTilingData &tiling_data) {
  result.batch_matmul_tiling_data = tiling_data;
  FillMetaFromBasicTiling(result, tiling_data.matMulTilingData);
}

void FillMetaFromTCubeTiling(TilingResult &result, const TCubeTiling &tiling_data) {
  result.cube_used_core_num = PositiveOrDefault(tiling_data.usedCoreNum, 1U);
  result.cube_base_m = PositiveOrDefault(tiling_data.baseM, 1U);
  result.cube_base_n = PositiveOrDefault(tiling_data.baseN, 1U);
}

void FillBatchTilingMeta(TilingResult &result, const size_t tiling_data_len) {
  BatchMatMulV3BasicTilingData batch_basic_tiling_data = {};
  if (CopyExactTilingData(result, tiling_data_len, batch_basic_tiling_data)) {
    FillMetaFromBatchBasicTiling(result, batch_basic_tiling_data);
    return;
  }
  BatchMatMulV3TilingData batch_tiling_data = {};
  if (CopyExactTilingData(result, tiling_data_len, batch_tiling_data)) {
    FillMetaFromTCubeTiling(result, batch_tiling_data.matMulTilingData.tCubeTiling);
    return;
  }
  BatchMatMulV3IterBatchBasicTilingData iter_batch_tiling_data = {};
  if (CopyExactTilingData(result, tiling_data_len, iter_batch_tiling_data)) {
    result.cube_base_m = std::max(iter_batch_tiling_data.baseM, 1U);
    result.cube_base_n = std::max(iter_batch_tiling_data.baseN, 1U);
    return;
  }
  BatchMatMulToMulBasicTilingData batch_to_mul_tiling_data = {};
  if (CopyExactTilingData(result, tiling_data_len, batch_to_mul_tiling_data)) {
    result.cube_used_core_num = std::max(batch_to_mul_tiling_data.usedCoreNum, 1U);
    return;
  }
  BatchMatMulV3MergeBatchBasicTilingData merge_batch_tiling_data = {};
  (void)CopyExactTilingData(result, tiling_data_len, merge_batch_tiling_data);
}

void FillMatMulTilingMeta(TilingResult &result, const size_t tiling_data_len) {
  MatMulV3BasicTilingData basic_tiling_data = {};
  if (CopyExactTilingData(result, tiling_data_len, basic_tiling_data)) {
    FillMetaFromBasicTiling(result, basic_tiling_data);
    return;
  }
  MatMulV3TilingDataCopy tiling_data_copy = {};
  if (CopyExactTilingData(result, tiling_data_len, tiling_data_copy)) {
    FillMetaFromTCubeTiling(result, tiling_data_copy.matMulTilingData.tCubeTiling);
    return;
  }
  MatMulV3TilingData tiling_data = {};
  if (CopyExactTilingData(result, tiling_data_len, tiling_data)) {
    FillMetaFromTCubeTiling(result, tiling_data.tCubeTiling);
    return;
  }
  MatMulToMulBasicTilingData to_mul_tiling_data = {};
  if (CopyExactTilingData(result, tiling_data_len, to_mul_tiling_data)) {
    result.cube_used_core_num = std::max(to_mul_tiling_data.usedCoreNum, 1U);
    if (to_mul_tiling_data.baseMN > 0) {
      result.cube_base_m = 1U;
      result.cube_base_n = to_mul_tiling_data.baseMN;
    }
    return;
  }
  MatMulV3KEqZeroBasicTilingData k_eq_zero_tiling_data = {};
  (void)CopyExactTilingData(result, tiling_data_len, k_eq_zero_tiling_data);
}

void FillTilingMeta(TilingResult &result, bool is_batch) {
  const size_t tiling_data_len = result.tiling_data.size();
  if (is_batch) {
    FillBatchTilingMeta(result, tiling_data_len);
    return;
  }
  FillMatMulTilingMeta(result, tiling_data_len);
}

template <typename RunTiling>
bool BuildTilingContext(const TilingRequest &request, const CompileState &compile_state, gert::TilingData *tiling_data,
                        gert::ContinuousVector *workspace, const std::vector<gert::Tensor *> &input_tensors,
                        const std::vector<gert::Tensor *> &output_tensors, RunTiling run_tiling) {
  gert::OpTilingContextBuilder tiling_builder;
  auto tiling_holder =
      request.is_batch ? tiling_builder.OpType(request.schema.op_type)
                              .OpName(request.schema.op_type)
                              .IOInstanceNum(MakeMatMulInputInstanceNum(request.matmul_attrs.has_bias,
                                                                        request.matmul_attrs.has_offset_w),
                                             kSingleOutputInstanceNum)
                             .AppendAttr(request.matmul_attrs.transpose_x1)
                             .AppendAttr(request.matmul_attrs.transpose_x2)
                             .AppendAttr(request.matmul_attrs.offset_x)
                             .AppendAttr(request.matmul_attrs.enable_hf32)
                             .CompileInfo(compile_state.compile_info_ptr)
                             .PlatformInfo(const_cast<fe::PlatFormInfos *>(&compile_state.platform_info))
                             .TilingData(tiling_data)
                             .Workspace(workspace)
                             .InputTensors(input_tensors)
                             .OutputTensors(output_tensors)
                             .Build()
                        : tiling_builder.OpType(request.schema.op_type)
                              .OpName(request.schema.op_type)
                              .IOInstanceNum(MakeMatMulInputInstanceNum(request.matmul_attrs.has_bias,
                                                                        request.matmul_attrs.has_offset_w),
                                             kSingleOutputInstanceNum)
                             .AppendAttr(request.matmul_attrs.transpose_x1)
                             .AppendAttr(request.matmul_attrs.transpose_x2)
                             .AppendAttr(request.matmul_attrs.offset_x)
                             .AppendAttr(request.matmul_attrs.op_impl_mode)
                             .CompileInfo(compile_state.compile_info_ptr)
                             .PlatformInfo(const_cast<fe::PlatFormInfos *>(&compile_state.platform_info))
                             .TilingData(tiling_data)
                             .Workspace(workspace)
                             .InputTensors(input_tensors)
                             .OutputTensors(output_tensors)
                             .Build();
  return run_tiling(tiling_holder.GetContext());
}

bool RunSharedCubeTiling(const TilingRequest &request, TilingResult &result) {
  const RuntimeTilingKey runtime_key = MakeRuntimeTilingKey(request);
  if (TryGetCachedTilingResult(runtime_key, &result)) {
    return true;
  }

  const char *op_type = request.schema.op_type;
  const auto &cached_funcs = GetOpHostFuncs(request.is_batch, op_type);
  if (!cached_funcs.loaded) {
    result.error_msg = std::string(op_type) + " shared op_host registry lookup failed";
    return false;
  }

  const auto &funcs = cached_funcs.funcs;
  auto compile_state = GetCompileState(request.compile_info, request.schema, funcs, request.is_batch, request.data_type,
                                        request.format, request.matmul_attrs, request.inputs[0].dtype,
                                        request.inputs[0].format, request.inputs, result.error_msg);
  if (compile_state == nullptr) {
    return false;
  }

  const size_t tiling_data_capacity = std::max(funcs.max_tiling_data_size, kMaxTilingDataSize);
  auto &scratch = GetTilingScratch();
  if (!scratch.EnsureCapacity(tiling_data_capacity, result.error_msg, op_type)) {
    return false;
  }
  auto *tiling_data = scratch.MutableTilingData();
  auto *workspace = scratch.MutableWorkspace();
  tiling_data->SetDataSize(0UL);
  (void)workspace->SetSize(0UL);

  gert::StorageFormat storage_format(request.format, request.format, {});
  std::vector<gert::Tensor> input_tensors_storage;
  const auto input_slots = BuildMatMulInputSlots(request.inputs, request.matmul_attrs);
  input_tensors_storage.reserve(input_slots.size());
  scratch.input_tensors.clear();
  for (const auto *input : input_slots) {
    if (input == nullptr) {
      continue;
    }
    ge::Format input_format = FormatToGeFormat(input->format);
    gert::StorageFormat input_storage_format(input_format, input_format, {});
    input_tensors_storage.emplace_back(MakeStorageShape(*input), input_storage_format, DtypeToGeDataType(input->dtype));
    scratch.input_tensors.push_back(&input_tensors_storage.back());
  }
  std::array<gert::Tensor, 1> output_tensors_storage = {
      gert::Tensor(MakeStorageShape(request.outputs[0]), storage_format, request.data_type)};
  scratch.output_tensors.clear();
  scratch.output_tensors.push_back(&output_tensors_storage[0]);

  const bool tiling_ok = BuildTilingContext(
      request, *compile_state, tiling_data, workspace, scratch.input_tensors, scratch.output_tensors,
      [&](auto *tiling_ctx) {
        if (tiling_ctx == nullptr) {
          result.error_msg = std::string(op_type) + " shared tiling context build failed";
          return false;
        }
        if (funcs.tiling(tiling_ctx) != ge::GRAPH_SUCCESS) {
          result.error_msg = std::string(op_type) + " shared tiling call failed";
          return false;
        }
        if (tiling_ctx->GetRawTilingData() == nullptr || tiling_ctx->GetRawTilingData()->GetDataSize() == 0) {
          result.error_msg = std::string(op_type) + " shared tiling returned empty data";
          return false;
        }
        FillTilingResultFromContext(tiling_ctx, result);
        return true;
      });
  if (!tiling_ok) {
    return false;
  }
  CacheTilingResult(runtime_key, result);
  return true;
}

}  // namespace

extern "C" bool AutofuseDoCubeMatMulTiling(const ge::autofuse::CompileInfo *compile_info,
                                           const std::vector<ge::autofuse::TensorInfo> *inputs,
                                           const std::vector<ge::autofuse::TensorInfo> *outputs,
                                           const std::vector<ge::autofuse::AttrInfo> *attrs, bool is_batch,
                                           ge::autofuse::TilingResult *result) {
  using namespace ge::autofuse;
  if (!ValidateTilingRequest(compile_info, inputs, outputs, attrs, result)) {
    return false;
  }
  const TilingRequest request = MakeTilingRequest(*compile_info, *inputs, *outputs, *attrs, is_batch);
  if (!IsOptionalInputSlotsValid(request)) {
    return false;
  }
  if (!IsSupportedTilingTensorDesc(*inputs, *outputs, request.data_type, request.format)) {
    return false;
  }
  return RunSharedCubeTiling(request, *result);
}

CubeKernelTilingWrapper::CubeKernelTilingWrapper() {}

CubeKernelTilingWrapper::~CubeKernelTilingWrapper() {}

void CubeKernelTilingWrapper::BuildMatMulArgs(const std::vector<TensorInfo> &args_list, int input_num,
                                              bool transpose_a, bool transpose_b,
                                              std::vector<TensorInfo> &origin_inputs,
                                              std::vector<TensorInfo> &origin_outputs,
                                              std::vector<TensorInfo> &inputs) {
  origin_inputs.clear();
  origin_outputs.clear();
  inputs.clear();

  int64_t m = 0;
  int64_t n = 0;
  std::vector<int64_t> write_shape;

  for (int i = 0; i < input_num && i < static_cast<int>(args_list.size()); ++i) {
    TensorInfo input = args_list[i];
    input.param_name = "input" + std::to_string(i);
    input.ori_shape = input.shape;

    origin_inputs.push_back(input);
    inputs.push_back(input);

    if (i == 0) {
      write_shape = input.shape;
      m = transpose_a ? input.shape[input.shape.size() - 1] : input.shape[input.shape.size() - 2];
    } else if (i == 1) {
      n = transpose_b ? input.shape[input.shape.size() - 2] : input.shape[input.shape.size() - 1];
    }
  }

  if (args_list.size() >= 2) {
    TensorInfo output = args_list[args_list.size() - 2];
    output.param_name = "output0";
    if (!write_shape.empty()) {
      write_shape[write_shape.size() - 1] = n;
      write_shape[write_shape.size() - 2] = m;
      output.shape = write_shape;
      output.ori_shape = write_shape;
    }
    if (!inputs.empty()) {
      output.dtype = inputs.back().dtype;
    }
    origin_outputs.push_back(output);
  }
}

TilingResult CubeKernelTilingWrapper::DoMatMulTiling(const CompileInfo &compile_info,
                                                     const std::vector<TensorInfo> &inputs,
                                                     const std::vector<TensorInfo> &outputs,
                                                     const std::vector<AttrInfo> &attrs, bool is_batch) {
  TilingResult result;
  if (AutofuseDoCubeMatMulTiling(&compile_info, &inputs, &outputs, &attrs, is_batch, &result)) {
    FillTilingMeta(result, is_batch);
  } else {
    result.success = false;
    if (result.error_msg.empty()) {
      result.error_msg = "codegen shared MatMulV3 tiling failed";
    }
  }
  return result;
}

}  // namespace autofuse
}  // namespace ge

)";
