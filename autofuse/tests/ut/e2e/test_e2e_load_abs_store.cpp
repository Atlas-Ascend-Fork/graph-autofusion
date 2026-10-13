/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "e2e_load_abs_store.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "ascgraph_info_complete.h"
#include "ascir_utils.h"
#define private public
#include "optimize.h"
#undef private
#include "codegen.h"
#include "e2e_common.h"
#include "platform_context.h"
#include "autofuse_config/auto_fuse_config.h"
#include "../../common/att/common_gen_utils.h"

using namespace ascir;

class E2E_LoadAbsStore : public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;
  codegen::Codegen codegen;

  E2E_LoadAbsStore()
      : optimizer(optimize::OptimizerOptions{}),
        codegen(codegen::CodegenOptions{.tiling_lib_path = "asdf", .tiling_lib_codegen_symbol = "as"}) {}

  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

std::map<std::string, std::string> MakeDefaultShapeInfo() {
  std::string source_tmpl = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(IDX);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(DIM);
  }())";
  auto replace_src = [&source_tmpl](const std::string &idx, const std::string &dim) {
    std::string s = source_tmpl;
    s.replace(s.find("IDX"), 3, idx);
    s.replace(s.find("DIM"), 3, dim);
    return s;
  };
  return {{"s0", replace_src("0", "1")}, {"s1", replace_src("1", "1")}, {"s2", replace_src("2", "1")}};
}

static std::string GetCodegenTilingWithLambdaExpectCode() {
  return R"rawliteral(#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cstdint>
#include "autofuse_tiling_data.h"
#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "platform/platform_infos_def.h"
#include "platform_ascendc.h"
#include "acl/acl.h"
#endif

#include <iostream>
#include <fstream>
#include <cinttypes>
#include <sys/syscall.h>
#include <unistd.h>
#include <securec.h>
#include "dlog_pub.h"
#define OP_LOGD(name, fmt, ...)
#define OP_LOGI(name, fmt, ...)
#define GE_MODULE_NAME static_cast<int32_t>(45)
inline uint64_t GetTid() {
     return static_cast<uint64_t>(syscall(__NR_gettid));
}
#define GELOGE(ERROR_CODE, fmt, ...)
#define OP_LOGE(name, fmt, ...)
#define OP_NAME "asc0000_autofused_abs"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define Pow(a, b) pow(a, b)
#define Rational(a, b) ((double)(a) / (double)(b))

namespace optiling {
extern "C" bool GetTiling(AutofuseTilingData& tiling_data, int32_t tilingCaseId=-1) {
  return true;
}
inline bool IsEqual(double a, double b) {
  return true;
}
}

#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/tiling_context.h"
#endif
extern "C" size_t GetTilingDataSize()
{
  return sizeof(AutofuseTilingData);
}

uint64_t GetWorkspaceSize(const AutofuseTilingData &t) {
  using namespace optiling;
  uint64_t ws_size = 0;

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
constexpr ResLimit g_no_limit_res = {1, 2201, 0, 2201, {}};
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint64_t* workspaceSize, uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size)
{
  tiling->set_block_dim(aiv_num);
  tiling->set_ub_size(ub_size - 256);
  if (!optiling::GetTiling(*tiling, -1)) {
      return -1;
  }
  *blockDim = tiling->get_block_dim();
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}
extern "C" int64_t AutofuseTilingWithConfig(const char *config_file, AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, int32_t tiling_case_id = -1)
{
 const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  tiling->set_block_dim(limit->aiv_num);
  tiling->set_ub_size(limit->ub_size);
  (void)config_file;
  if (!optiling::GetTiling(*tiling, tiling_case_id)) {
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
  return true;
}
extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{

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
}
bool version_is_ASCEND950 = false;
struct AfTilingParseData{
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
 if (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
 version_is_ASCEND950 = true;
 }
 ub_size -= (ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910 && ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910B && ub_size % 1024 == 0) ? 256 : 0;
 (*tiling_parse_data)->ub_size = ub_size;
 return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)
{
  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);
  auto input_data_num =  extend_context->GetInputValue<size_t>(0U);
  auto parse = extend_context->GetInputValue<AfTilingParseData*>(input_data_num + 1);
  uint64_t workspace_size;
  uint32_t block_dim;
  static const char* config_file = nullptr;
  auto tiling_data =  context->GetTilingData<AutofuseTilingData>();
  ResLimit limit;
  limit.aiv_num = parse->aiv_num;
  limit.ub_size = (uint32_t)parse->ub_size;
  auto ret = AutofuseTilingWithConfig(config_file, tiling_data, &workspace_size, &block_dim, &limit);
  context->SetBlockDim(block_dim);
  *context->GetWorkspaceSizes(1) = workspace_size;

  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
  return ret;
}

extern "C" ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context)
{
  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);
  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);
  if (symbol_src_vec == nullptr) {
    return ge::GRAPH_FAILED;
  }

  symbol_src_vec->SetSize(0);
  return ge::GRAPH_SUCCESS;
}
extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;

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

std::string tiling_data_const_gen_result;
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
  (void)AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit);
  std::string GenTilingDataValue_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.block_dim);
  std::string GenTilingDataValue_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.corenum);
  std::string GenTilingDataValue_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.ub_size);
  std::string GenTilingDataValue_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.hbm_size);
  std::string GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("graph0_tiling_key", TilingDataValue.graph0_tiling_key);

  tiling_data_const_gen_result = R"(#ifndef __Autofuse_Tiling_Data_H__
#define __Autofuse_Tiling_Data_H__
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

BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
  GenTilingDataValue_block_dim_field_DeclareFunc_def
  GenTilingDataValue_corenum_field_DeclareFunc_def
  GenTilingDataValue_ub_size_field_DeclareFunc_def
  GenTilingDataValue_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def
END_TILING_DATA_DEF_T;

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

  return tiling_data_const_gen_result.c_str();
}


#ifndef __CCE_KT_TEST__
extern "C" int64_t GetTilingKeyForStatic()
{
  return FindBestTilingKey(TilingDataValue);
}
std::string kernel_type;
extern "C" const char* GetTilingKeyKernelTypeForStatic()
{
  const std::map<int64_t, std::string> kernel_type_map = {
  };

  auto tiling_key = FindBestTilingKey(TilingDataValue);
  auto it = kernel_type_map.find(tiling_key);
  if (it != kernel_type_map.end()) {
    kernel_type = it->second;
  }
  return kernel_type.c_str();
}
#endif
)rawliteral";
}

static std::string GetCodegenPGOCodeExpectCode() {
  return R"rawliteral(#include <cinttypes>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <securec.h>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"
#include "dlog_pub.h"
#include "mspti.h"
#include "tiling/platform/platform_ascendc.h"

#include "autofuse_tiling_data.h"

#ifndef AUTOFUSE_PGO_TENSOR_ARGS_DEFINED
#define AUTOFUSE_PGO_TENSOR_ARGS_DEFINED
struct PgoTensorArgs {
  void **inputs = nullptr;
  uint32_t input_num = 0;
  void **outputs = nullptr;
  uint32_t output_num = 0;
};
#endif
namespace {
constexpr bool g_is_mix_operator = false;
static bool g_is_static_kernel = false;
std::vector<uint32_t> g_mix_graph0_tiling_keys = {
};
bool IsMixTiling(const AutofuseTilingData &t) {
  if constexpr (!g_is_mix_operator) {
    return false;
  }
  if (!g_is_static_kernel) {
    return true;
  }
  if (!g_mix_graph0_tiling_keys.empty() && std::find(g_mix_graph0_tiling_keys.begin(), g_mix_graph0_tiling_keys.end(), t.graph0_tiling_key) != g_mix_graph0_tiling_keys.end()) {
    return true;
  }
  return false;
}
static std::string g_kernel_name;
static std::string g_kernel_o_file;
static std::string g_npu_lock_file;
#define PGO_GRAPH_NAME "test_graph"
const char *pgo_dir = "";
const char *config_file = "/test_graph_config.txt";
const char *search_file = "/test_graph_search.txt";
const char *kernel_file = "/libtest_graph.so";
#define SUCCESS 0
#define FAILED 1
inline uint64_t PgoGetTid() {
  return static_cast<uint64_t>(syscall(__NR_gettid));
}
constexpr int32_t PGO_MODULE_NAME = static_cast<int32_t>(45);
#define PGO_LOG_PREFIX "%" PRIu64 " %s:[PGO][" PGO_GRAPH_NAME "] "
#define DLOGD(fmt, ...) do { dlog_debug(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); } while (false)
#define DLOGI(fmt, ...) do { dlog_info(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); } while (false)
#define DLOGW(fmt, ...) do { dlog_warn(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); } while (false)
#define DLOGE(fmt, ...) do { dlog_error(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); } while (false)

class CardLock {
public:
  CardLock(const char *path) {
    fd_ = open(path, O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
      DLOGE("open lock file: %s", std::strerror(errno));
      std::exit(1);
    }
    if (flock(fd_, LOCK_EX) == -1) {
      DLOGE("flock LOCK_EX: %s", std::strerror(errno));
      std::exit(1);
    }
  }

  ~CardLock() {
    if (fd_ != -1) {
      if (flock(fd_, LOCK_UN) == -1) {
        DLOGW("flock LOCK_UN: %s", std::strerror(errno));
      }
      close(fd_);
    }
  }

  CardLock(const CardLock&) = delete;
  CardLock& operator=(const CardLock&) = delete;

private:
  int fd_{-1};
};

void PgoSaveTilingKey(const AutofuseTilingData &tiling_data, double best_perf, std::ofstream &out_file) {
  const size_t tiling_bytes = sizeof(tiling_data);
  const size_t tiling_bytes_align = (tiling_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  std::vector<int32_t> tiling_i32(tiling_bytes_align, 0);
  memcpy_s(tiling_i32.data(), tiling_i32.size() * sizeof(int32_t), &tiling_data, tiling_bytes);
  for (size_t idx = 0; idx < tiling_i32.size(); ++idx) {
    out_file << tiling_i32[idx] << " ";
  }
  out_file << "# " << best_perf << std::endl;
}
void AppendPgoSearchTilingData(const AutofuseTilingData &tiling_data, double best_perf, std::ios::openmode mode = std::ios::app) {
  DLOGD("AppendPgoSearchTilingData to file: %s", search_file);
  std::ofstream out_file(search_file, mode);
  if (!out_file.is_open()) {
    DLOGE("Failed to open file:%s", search_file);
    return;
  }
  PgoSaveTilingKey(tiling_data, best_perf, out_file);
  out_file.close();

  int fd = ::open(search_file, O_WRONLY);
  if (fd < 0) {
    DLOGE("Failed to open file:%s", search_file);
    return;
  }
  if (::fsync(fd) < 0) {
    DLOGW("Failed to fsync file:%s", search_file);
  }
  ::close(fd);

  return;
}
struct AivKernelLaunchOpArgs {
  uint64_t workspace_addr;
  uint64_t tiling_addr;
};
struct MixKernelLaunchOpArgs {
  uint64_t ffts;
  uint64_t workspace_addr;
  uint64_t tiling_addr;
};
void *g_workspace = nullptr;
static void *handle = nullptr;
static bool initialized = false;

__attribute__((constructor)) void Init() {
  if (initialized) return;
  handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    DLOGE("Failed to load %s: %s", kernel_file, dlerror());
    return;
  }
  DLOGD("Kernel api lib %s load succeed", kernel_file);
  initialized = true;
}

__attribute__((destructor)) void DeInit() {
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
  initialized = false;
}

inline void *GetFunc(const char *func_name) {
  if (handle == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle, func_name);
  if (func == nullptr) {
    DLOGE("Failed to load wrapper api func: %s", dlerror());
  }
  return func;
}
aclrtStream g_stream;
uint64_t ffts;
void *g_pgo_inputs[1] = {nullptr};
void *g_pgo_outputs[1] = {nullptr};
PgoTensorArgs g_pgo_tensor_args = {g_pgo_inputs, 0U, g_pgo_outputs, 0U};

void *g_tiling_device_addr = nullptr;
struct LaunchParams {
  AivKernelLaunchOpArgs aiv_args;
  void *aiv_args_device;
  MixKernelLaunchOpArgs mix_args;
  void *mix_args_device;
} g_launch_params;
aclError LaunchParamsInit(PgoTensorArgs *tensor_args) {
  static void *ffts = nullptr;
  aclError ret = ACL_SUCCESS;
  constexpr uint32_t kPgoInputCount = 0U;
  constexpr uint32_t kPgoOutputCount = 0U;
  if (tensor_args == nullptr || tensor_args->input_num < kPgoInputCount ||
      tensor_args->output_num < kPgoOutputCount ||
      (kPgoInputCount > 0U && tensor_args->inputs == nullptr) ||
      (kPgoOutputCount > 0U && tensor_args->outputs == nullptr)) {
    DLOGE("invalid pgo tensor args");
    return FAILED;
  }
  g_launch_params.aiv_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);
  ret = aclrtGetHardwareSyncAddr(&ffts);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl get hardware sync addr failed, ERROR: %d", ret);
    return FAILED;
  }
  g_launch_params.mix_args.ffts = reinterpret_cast<uint64_t>(ffts);
  g_launch_params.mix_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);
  ret = aclrtMalloc(&g_launch_params.aiv_args_device, sizeof(AivKernelLaunchOpArgs), ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl malloc aiv args device failed, ERROR: %d", ret);
    return FAILED;
  }
  ret = aclrtMalloc(&g_launch_params.mix_args_device, sizeof(MixKernelLaunchOpArgs), ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl malloc mix args device failed, ERROR: %d", ret);
    return FAILED;
  }
  return ACL_SUCCESS;
}
void LaunchParamsDeInit() {
  if (g_launch_params.aiv_args_device != nullptr) {
    auto ret = aclrtFree(g_launch_params.aiv_args_device);
    if (ret != ACL_SUCCESS) {
      DLOGW("acl free aiv args device failed, ERROR: %d", ret);
    }
    g_launch_params.aiv_args_device = nullptr;
  }
  if (g_launch_params.mix_args_device != nullptr) {
    auto ret = aclrtFree(g_launch_params.mix_args_device);
    if (ret != ACL_SUCCESS) {
      DLOGW("acl free mix args device failed, ERROR: %d", ret);
    }
    g_launch_params.mix_args_device = nullptr;
  }
}
aclError UpdateLaunchParam(const AutofuseTilingData &tiling_data) {
  if (IsMixTiling(tiling_data)) {
    auto ret = aclrtMemcpy((void *)g_launch_params.mix_args.tiling_addr, sizeof(AutofuseTilingData), (void *)&tiling_data, sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      DLOGE("memcpy tiling data to device failed, ERROR: %d", ret);
      return FAILED;
    }
    g_launch_params.mix_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);
    ret = aclrtMemcpy(g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), (void *)&g_launch_params.mix_args, sizeof(g_launch_params.mix_args), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      DLOGE("memcpy mix_args to device failed, ERROR: %d", ret);
      return FAILED;
    }
  } else {
    auto ret = aclrtMemcpy((void *)g_launch_params.aiv_args.tiling_addr, sizeof(AutofuseTilingData), (void *)&tiling_data, sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      DLOGE("memcpy tiling data to device failed, ERROR: %d", ret);
      return FAILED;
    }
    g_launch_params.aiv_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);
    ret = aclrtMemcpy(g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), (void *)&g_launch_params.aiv_args, sizeof(g_launch_params.aiv_args), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      DLOGE("memcpy aiv_args to device failed, ERROR: %d", ret);
      return FAILED;
    }
  }
  return ACL_SUCCESS;
}
struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};
ResLimit g_res_limit = {1, {}};
inline bool IsEqual(double a, double b) {
  const double epsilon = 1e-8;
  double abs = (a > b) ? (a - b) : (b - a);
  return abs < epsilon;
}
} // namespace
typedef uint64_t (*GetTilingKeyCountType)(void);
GetTilingKeyCountType get_tiling_key_count_fn = reinterpret_cast<GetTilingKeyCountType>(GetFunc("GetTilingKeyCount"));
typedef int64_t (*FindBestTilingKeyType)(AutofuseTilingData &t);
FindBestTilingKeyType find_best_tiling_key_fn = reinterpret_cast<FindBestTilingKeyType>(GetFunc("FindBestTilingKey"));
int WrapperOnlyLaunch(uint64_t workspace_size, AutofuseTilingData *tiling_data) {
  static bool inited = false;
  static aclrtBinHandle bin_handle = nullptr;
  if (get_tiling_key_count_fn == nullptr) {
    DLOGE("get_tiling_key_count_fn is nullptr");
    return FAILED;
  }
  static uint64_t tiling_key_count = get_tiling_key_count_fn();
  static std::vector<aclrtFuncHandle> func_handles(tiling_key_count);
  if (tiling_data == nullptr) {
    DLOGE("tiling_data is null");
    return -1;
  }
  uint32_t block_dim = tiling_data->block_dim;
  aclError ret = ACL_SUCCESS;
  int64_t tiling_key = 0;
  if (find_best_tiling_key_fn != nullptr) {
    tiling_key = find_best_tiling_key_fn(*tiling_data);
    if (tiling_key < 0 || static_cast<uint64_t>(tiling_key) >= tiling_key_count) {
      DLOGE("find best tiling key failed, tiling_key=%ld, valid range=[0,%lu)", tiling_key, tiling_key_count);
      return FAILED;
    }
  } else {
    DLOGE("find best tiling key func is null");
    return FAILED;
  }
  if (!inited) {
    auto ret = aclrtBinaryLoadFromFile(g_kernel_o_file.c_str(), nullptr, &bin_handle);
    if (ret != ACL_SUCCESS) {
      DLOGE("acl load binary from file failed, ERROR: %d", ret);
      return FAILED;
    }
    if (g_is_static_kernel) {
      aclrtFuncHandle func_handle = nullptr;
      ret = aclrtBinaryGetFunction(bin_handle, (g_kernel_name + "_" + std::to_string(tiling_key)).c_str(), &func_handle);
      if (ret != ACL_SUCCESS) {
        DLOGE("acl get function failed, ERROR: %d", ret);
        return FAILED;
      }
      func_handles[tiling_key] = func_handle;
    } else {
      for (uint64_t i = 0; i < tiling_key_count; ++i) {
        aclrtFuncHandle func_handle = nullptr;
        ret = aclrtBinaryGetFunction(bin_handle, (g_kernel_name + "_" + std::to_string(i)).c_str(), &func_handle);
        if (ret != ACL_SUCCESS) {
          DLOGE("acl get function failed, ERROR: %d", ret);
          return FAILED;
        }
        func_handles[i] = func_handle;
      }
    }
    inited = true;
  }
  if (IsMixTiling(*tiling_data)) {
    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), nullptr, g_stream);
  } else {
    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), nullptr, g_stream);
  }
  auto ret_async = aclrtSynchronizeStream(g_stream);
  if (ret != ACL_SUCCESS) {
    DLOGE("aclrtLaunchKernelV2 failed, ERROR: %d", ret);
    return FAILED;
  }
  if (ret_async != ACL_SUCCESS) {
    DLOGE("aclrtSynchronizeStream failed, ERROR: %d", ret_async);
    return FAILED;
  }
  return ret;
}

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))
constexpr size_t group_size = 1000ULL;
static std::map<uint64_t, msptiActivity*> g_profiling_map;
constexpr uint64_t loop = 20;
constexpr int max_flush_times = 5;
constexpr size_t mspti_buffer_size = 16ULL * 1024 * 1024;
static double best_perf = DBL_MAX;

static const char* GetActivityKindString(msptiActivityKind kind) {
  static const std::unordered_map<msptiActivityKind, const char*> STRING_MAP = {
    {MSPTI_ACTIVITY_KIND_INVALID, "INVALID"},
    {MSPTI_ACTIVITY_KIND_MARKER, "MARKER"},
    {MSPTI_ACTIVITY_KIND_KERNEL, "KERNEL"},
    {MSPTI_ACTIVITY_KIND_API, "API"},
    {MSPTI_ACTIVITY_KIND_MEMORY, "MEMORY"},
    {MSPTI_ACTIVITY_KIND_MEMSET, "MEMSET"},
    {MSPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY"},
    {MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, "CORRELATION"}
  };
  auto it = STRING_MAP.find(kind);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
}

static const char* GetResultCodeString(msptiResult result) {
  static const std::unordered_map<msptiResult, const char*> STRING_MAP = {
    {MSPTI_SUCCESS, "SUCCESS"},
    {MSPTI_ERROR_INVALID_PARAMETER, "ERROR_INVALID_PARAMETER"},
    {MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, "MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED"},
    {MSPTI_ERROR_DEVICE_OFFLINE, "DEVICE_OFFLINE"},
    {MSPTI_ERROR_QUEUE_EMPTY, "QUEUE_EMPTY"},
    {MSPTI_ERROR_INNER, "ERROR_INNER"}
  };

  auto it = STRING_MAP.find(result);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
}

void UserBufferRequest(uint8_t **buffer, size_t *size, size_t *records_num) {
  DLOGD("[mspti] UserBufferRequest...");
  uint8_t *mspti_buffer = reinterpret_cast<uint8_t *>(malloc(mspti_buffer_size + ALIGN_SIZE));
  if (mspti_buffer == nullptr) {
    DLOGE("[mspti] malloc mspti_buffer failed");
    *buffer = nullptr;
    *size = 0;
    *records_num = 0;
    return;
  }
  *buffer = ALIGN_BUFFER(mspti_buffer, ALIGN_SIZE);
  *size = mspti_buffer_size;
  *records_num = 0;
}

void UserBufferComplete(uint8_t *buffer, size_t size, size_t valid_size) {
  DLOGD("[mspti] UserBufferComplete, buf addr: %" PRIuPTR ", size: %zu, valid size: %zu", (uintptr_t)buffer, size, valid_size);
  if (valid_size > 0) {
    msptiActivity *mspti_record = NULL;
    msptiResult status = MSPTI_SUCCESS;
    do {
      status = msptiActivityGetNextRecord(buffer, valid_size, &mspti_record);
      if (status == MSPTI_SUCCESS) {
        if (mspti_record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
          msptiActivityKernel* kernelRecord = (msptiActivityKernel*)mspti_record;
          msptiActivity* pRecordCopy = (msptiActivity *)malloc(sizeof(msptiActivityKernel));
          memset(pRecordCopy, 0, sizeof(msptiActivityKernel));
          memcpy(pRecordCopy, kernelRecord, sizeof(msptiActivityKernel));
          g_profiling_map[kernelRecord->start] = pRecordCopy;

        } else {
          DLOGD("[mspti] [%s] ignored", GetActivityKindString(mspti_record->kind));
        }
      } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        DLOGW("[mspti] Consume data fail error is %s", GetResultCodeString(status));
        break;
      }
    } while (1);
  }
  free(buffer);
}

void SetUpMspti(msptiSubscriberHandle* subscriber) {
  DLOGD("[mspti] setup mspti");
  msptiSubscribe(subscriber, nullptr, nullptr);
  msptiActivityRegisterCallbacks(UserBufferRequest, UserBufferComplete);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
}

void TearDownMspti(msptiSubscriberHandle *subscriber) {
  DLOGD("[mspti] tear down mspti");
  msptiUnsubscribe(*subscriber);
  msptiActivityFlushAll(1);
}
int ProfilingBatchProcess(uint64_t workspace_size, std::vector<AutofuseTilingDataPerf>::iterator begin, std::vector<AutofuseTilingDataPerf>::iterator end) {
  uint64_t batch_size = end - begin;
  g_profiling_map.clear();
  msptiSubscriberHandle subscriber;
  SetUpMspti(&subscriber);

  static int64_t count = 0;
  count++;

  int64_t result = 0;
  for (auto it = begin; it != end; ++it) {
    it->best_perf = DBL_MAX;
    AutofuseTilingData &tiling_data = it->tiling_data;
    UpdateLaunchParam(tiling_data);
    for (uint64_t i = 0; i < loop; ++i) {
      result = WrapperOnlyLaunch(workspace_size, &tiling_data);
      if (result != 0) {
        DLOGE("ProfilingBatchProcess launch failed loop:%" PRIu64 "", i);
        TearDownMspti(&subscriber);
        return -1;
      }
    }
  }

  result = aclrtSynchronizeStream(g_stream);
  TearDownMspti(&subscriber);

  int flush_count = 0;
  while (g_profiling_map.size() < batch_size * loop && flush_count < max_flush_times) {
    flush_count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));
    msptiActivityFlushAll(1);
  }

  if (g_profiling_map.size() < batch_size * loop) {
    DLOGE("ProfilingBatchProcess g_profiling_map size %zu is less than batch_size * loop %" PRIu64 "", g_profiling_map.size(), batch_size * loop);
    for (auto &item : g_profiling_map) {
      free(item.second);
    }
    return -1;
  }

  auto it = g_profiling_map.begin();
  for (uint64_t i = 0; i < batch_size; ++i) {
    uint64_t total_duration = 0;
    std::vector<uint64_t> durations;
    for (uint64_t j = 0; j < loop; ++j) {
      msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(it->second);
      durations.push_back(kernel->end - kernel->start);
      std::advance(it, 1);
    }
    std::sort(durations.begin(), durations.end(), std::greater<uint64_t>());
    for (size_t k = 1; k < 6; ++k) {
      total_duration += durations[k];
    }
    double average_duration = static_cast<double>(total_duration) / 5;
    (begin + i)->best_perf = average_duration;
    if (best_perf > average_duration) {
      best_perf = average_duration;
    }
    DLOGD("average_duration:%f best_perf:%f count:%" PRId64 " batch_size:%" PRIu64 " flush_count:%d", average_duration, best_perf, count, batch_size, flush_count);
  }
  for (auto &item : g_profiling_map) {
    free(item.second);
  }
  return 0;
}

extern "C" long int PGOGetProfilingBatch(PgoTensorArgs *tensor_args, void* stream, uint64_t workspace_size, std::vector<AutofuseTilingDataPerf> *profiles) {
  int case_num = profiles->size();
  DLOGI("PGOGetProfilingBatch case_num:%d", case_num);
  if (workspace_size > 0) {
    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DLOGE("malloc workspace failed, size: %u, ERROR: %d", workspace_size, ret);
      return FAILED;
    }
  }
  int64_t result = 0;
  auto it = profiles->begin();
  while (it != profiles->end()) {
    auto end_it = (it + group_size >= profiles->end()) ? profiles->end() : it + group_size;
    size_t start_index = std::distance(profiles->begin(), it);
    for (int i = 0; i < 3; i++) {
      result = ProfilingBatchProcess(workspace_size, it, end_it);
      if (result != 0) {
        DLOGW("ProfilingBatchProcess failed at start_index:%zu retry time:%d", start_index, i);
      } else {
        break;
      }
    }
    it = end_it;
  }
  if (g_workspace != nullptr) {
    auto ret = aclrtFree(g_workspace);
    if (ret != ACL_SUCCESS) {
      DLOGE("free workspace failed, ERROR: %d", ret);
      return FAILED;
    }
  }
  return 0;
}

extern "C" long int PGOGetProfiling(PgoTensorArgs *tensor_args, void *stream, uint64_t workspace_size, AutofuseTilingData *tiling_data, double *outCostTime) {
  if (workspace_size > 0) {
    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DLOGE("malloc workspace failed, size: %u, ERROR: %d", workspace_size, ret);
      return FAILED;
    }
  }
  g_profiling_map.clear();
  msptiSubscriberHandle subscriber;
  SetUpMspti(&subscriber);

  int64_t result = -1;
  *outCostTime = DBL_MAX;
  static int64_t count = 0;
  count++;

  UpdateLaunchParam(*tiling_data);
  for (uint64_t j = 0; j < loop; ++j) {
    result = WrapperOnlyLaunch(workspace_size, tiling_data);
    if (result != 0) {
      DLOGE("launch failed loop:%" PRIu64 "", j);
      TearDownMspti(&subscriber);
      return -1;
    }
  }

  if (g_workspace != nullptr) {
    auto ret = aclrtFree(g_workspace);
    if (ret != ACL_SUCCESS) {
      DLOGE("free workspace failed, ERROR: %d", ret);
      TearDownMspti(&subscriber);
      return FAILED;
    }
  }
  result = aclrtSynchronizeStream(g_stream);
  if (result != 0) {
    DLOGE("sync stream failed");
    TearDownMspti(&subscriber);
    return -1;
  }
  TearDownMspti(&subscriber);

  int flush_count = 0;
  while (g_profiling_map.size() < loop && flush_count < max_flush_times) {
    flush_count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));
    msptiActivityFlushAll(1);
  }

  if (g_profiling_map.size() != loop) {
    DLOGE("map size %zu not equals to loop %" PRIu64 "", g_profiling_map.size(), loop);
    for (auto &item : g_profiling_map) {
      free(item.second);
    }
    return -1;
  }

  uint64_t total_duration = 0;
  std::vector<uint64_t> durations;
  for (const auto &pair : g_profiling_map) {
    msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(pair.second);
    durations.push_back(kernel->end - kernel->start);
    DLOGD("kernel duration:%" PRIu64 "", kernel->end - kernel->start);
  }
  std::sort(durations.begin(), durations.end(), std::greater<uint64_t>());
  for (size_t i = 1; i < 6; ++i) {
    total_duration += durations[i];
  }
  double average_duration = static_cast<double>(total_duration) / 5;
  *outCostTime = average_duration;

  if (best_perf > *outCostTime) {
    best_perf = *outCostTime;
  }
  DLOGD("average_duration:%f best_perf:%f count:%" PRId64 " flush_count:%d", *outCostTime, best_perf, count, flush_count);
  for (auto &item : g_profiling_map) {
    free(item.second);
  }
  return 0;
}

typedef int64_t (*PGOSearchType)(char *search_file, char *config_file, AutofuseTilingData *tiling_data, uint64_t *workspace_size, uint32_t *blockDim, void *resource_limit, PgoTensorArgs *tensor_args, void *stream, void *prof_callback, void *prof_batch_callback);
static PGOSearchType pgo_search_fn = reinterpret_cast<PGOSearchType>(GetFunc("PgoTilingSearch"));
int pgo() {
  AutofuseTilingData tiling_data = {0};
  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
  if (pgo_search_fn == nullptr) {
    DLOGE("pgo search func not found");
    return -1;
  }
  int64_t result = pgo_search_fn((char*)search_file, (char *)config_file, &tiling_data, &workspace_size, &block_dim, &g_res_limit,tensor_args, &g_stream, reinterpret_cast<void*>(PGOGetProfiling), reinterpret_cast<void*>(PGOGetProfilingBatch));
  if (result != 0) {
    DLOGE("pgo search failed. ERROR: %" PRId64 "", result);
    return -1;
  }
  return 0;
}

typedef int64_t (*AutofuseTilingWithConfigType)(const char *config_file, AutofuseTilingData *tiling, uint64_t *workspace_size, uint32_t *blockDim, ResLimit *res_limit);
static AutofuseTilingWithConfigType autofuse_tiling_with_config_fn = reinterpret_cast<AutofuseTilingWithConfigType>(GetFunc("AutofuseTilingWithConfig"));
int static_pgo(const char* config_file) {
  if (autofuse_tiling_with_config_fn == nullptr) {
    DLOGE("autofuse tiling with config func not found");
    return -1;
  }
  AutofuseTilingData tiling_data = {0};
  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;
  uint64_t workspace_size = 0;
  uint32_t block_dim = 0;
  int64_t result = autofuse_tiling_with_config_fn(config_file, &tiling_data, &workspace_size, &block_dim, &g_res_limit);
  if (result != 0) {
    DLOGE("autofuse tiling with config failed. ERROR: %" PRId64 "", result);
    return -1;
  }
  double out_cost = DBL_MAX;
  for (int i = 0; i < max_flush_times; i++) {
    result = PGOGetProfiling(tensor_args, g_stream, workspace_size, &tiling_data, &out_cost);
    if (result != 0 || IsEqual(out_cost, DBL_MAX)) {
      DLOGW("get profiling failed.");
    } else {
      break;
    }
  }
  AppendPgoSearchTilingData(tiling_data, out_cost);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    DLOGE("Usage: %s <type> <device_id> <aiv_num> <ub_size> <kernel_name>", argv[0]);
    return -1;
  }
  int32_t type = static_cast<int32_t>(atoi(argv[1]));
  int32_t device_id = static_cast<int32_t>(atoi(argv[2]));
  int32_t aiv_num = static_cast<int32_t>(atoi(argv[3]));
  int32_t ub_size = static_cast<int32_t>(atoi(argv[4]));
  g_kernel_name = argv[5];
  DLOGI("execute info : type: %d, device_id: %d, kernel_name: %s", type, device_id, g_kernel_name.c_str());
  DLOGI("execute limit: aiv_num is %d, ub_size is %d", aiv_num, ub_size);
  g_npu_lock_file = std::string(pgo_dir) + "/npu_lock_" + std::to_string(device_id) + ".lock";
  g_kernel_o_file = std::string(pgo_dir) + "/" + g_kernel_name + ".o";
  CardLock lock(g_npu_lock_file.c_str());
  g_res_limit.aiv_num = aiv_num;
  g_res_limit.ub_size = ub_size;
  auto ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl init failed, ERROR: %d", ret);
    return FAILED;
  }
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl set device failed, device id: %d, ERROR: %d", device_id, ret);
    aclFinalize();
    return FAILED;
  }
  ret = aclrtCreateStream(&g_stream);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl create stream failed, ERROR: %d", ret);
    aclrtResetDevice(device_id);
    aclFinalize();
    return FAILED;
  }

  g_pgo_tensor_args.inputs = g_pgo_inputs;
  g_pgo_tensor_args.input_num = 0U;
  g_pgo_tensor_args.outputs = g_pgo_outputs;
  g_pgo_tensor_args.output_num = 0U;
  ret = aclrtMalloc(&g_tiling_device_addr, sizeof(AutofuseTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl malloc tiling data failed, ERROR: %d", ret);
    return FAILED;
  }
  ret = LaunchParamsInit(&g_pgo_tensor_args);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  if (type == 0) {
    ret = pgo();
  } else if (type == 1) {
    g_is_static_kernel = true;
    ret = static_pgo(config_file);
  } else {
    DLOGE("Invalid type: %d", type);
    ret = -1;
  }
  LaunchParamsDeInit();

  if (g_tiling_device_addr != nullptr) {
    ret = aclrtFree(g_tiling_device_addr);
    if (ret != ACL_SUCCESS) {
      DLOGE("acl free tiling data failed, ERROR: %d", ret);
      return FAILED;
    }
    g_tiling_device_addr = nullptr;
  }
  ret = aclrtDestroyStream(g_stream);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl destroy stream failed, ERROR: %d", ret);
    return FAILED;
  }
  ret = aclrtResetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    DLOGE("acl reset device failed, device id: %d, ERROR: %d", device_id, ret);
    return FAILED;
  }
  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    DLOGE("acl finalize failed, ERROR: %d", ret);
    return FAILED;
  }
  DeInit();
  return ret;
}
)rawliteral";
}

static void AssertPgoCodegenContract(const std::string &pgo_codes) {
  EXPECT_FALSE(pgo_codes.empty());
  for (const auto &token :
       {"PgoTilingSearch", "PgoDsoCallGuard", "active_calls", "closing.store(true", "dso_cv.wait"}) {
    EXPECT_NE(pgo_codes.find(token), std::string::npos);
  }
  EXPECT_NE(pgo_codes.find("RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE"), std::string::npos);
}

static std::string GetCodegenTilingWithLambdaWithPGOExpectCode() {
  return R"rawliteral(#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cstdint>
#include "autofuse_tiling_data.h"
#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "platform/platform_infos_def.h"
#include "platform_ascendc.h"
#include "acl/acl.h"
#endif

#include <cfloat>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <array>

#ifndef AUTOFUSE_PGO_TENSOR_ARGS_DEFINED
#define AUTOFUSE_PGO_TENSOR_ARGS_DEFINED
struct PgoTensorArgs {
  void **inputs = nullptr;
  uint32_t input_num = 0;
  void **outputs = nullptr;
  uint32_t output_num = 0;
};
#endif
typedef long int (*ProfilingCallback)(PgoTensorArgs *tensor_args, void *stream, uint64_t workspaceSize, AutofuseTilingData *tiling_data, double *cost_time);
typedef long int (*ProfilingBatchCallback)(PgoTensorArgs *tensor_args, void *stream, uint64_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);
class PgoConfig {
public:
  static PgoConfig& Instance() {
    static PgoConfig instance;
    return instance;
  }
  void ResetRuntimeOverrides() {
    need_change_solver_run = false;
    pgo_threshold_index = 0;
    pgo_ub_threshold_list = {0.2, 0.1, 0, 0.05, 0.1};
    pgo_corenum_threshold_list = {0.4, 0.4, 1, 1, 0.8};
  }
  ProfilingCallback single_callback;
  ProfilingBatchCallback batch_callback;
  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num
  bool need_change_solver_run = false;
  size_t pgo_threshold_index = 0;
  constexpr static size_t pgo_threshold_list_size = 5;
  std::array<double, pgo_threshold_list_size> pgo_ub_threshold_list{0.2, 0.1, 0, 0.05, 0.1};
  std::array<double, pgo_threshold_list_size> pgo_corenum_threshold_list{0.4, 0.4, 1, 1, 0.8};
private:
  PgoConfig() = default;
  ~PgoConfig() = default;
  PgoConfig(const PgoConfig &) = delete;
  PgoConfig &operator=(const PgoConfig &) = delete;
};
class PgoConfigRuntimeGuard {
public:
  PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }
  ~PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }
};

#include <iostream>
#include <fstream>
#include <cinttypes>
#include <sys/syscall.h>
#include <unistd.h>
#include <securec.h>
#include "dlog_pub.h"
#define OP_LOGD(name, fmt, ...)
#define OP_LOGI(name, fmt, ...)
#define GE_MODULE_NAME static_cast<int32_t>(45)
inline uint64_t GetTid() {
     return static_cast<uint64_t>(syscall(__NR_gettid));
}
#define GELOGE(ERROR_CODE, fmt, ...)
#define OP_LOGE(name, fmt, ...)
#define OP_NAME "asc0000_autofused_abs"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define Pow(a, b) pow(a, b)
#define Rational(a, b) ((double)(a) / (double)(b))

namespace optiling {
extern "C" bool GetTiling(AutofuseTilingData& tiling_data, int32_t tilingCaseId=-1) {
  return true;
}
inline bool IsEqual(double a, double b) {
  return true;
}
struct SearchConfig;
bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData &tiling_data, int32_t tilingCaseId, AutofuseTilingData* output_tiling_data, PgoTensorArgs *tensor_args, void* stream, uint64_t workspaceSize, double& out_best_perf, std::unordered_map<int64_t, uint64_t> &workspace_map, std::vector<uint32_t*> block_dim_vec={}, const SearchConfig *search_cfg=nullptr) {
  return true;
}
bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData* tiling_data, uint32_t max_block_dim=48) {
  return true;
}
}

#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/tiling_context.h"
#endif
extern "C" size_t GetTilingDataSize()
{
  return sizeof(AutofuseTilingData);
}

uint64_t GetWorkspaceSize(const AutofuseTilingData &t) {
  using namespace optiling;
  uint64_t ws_size = 0;

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
constexpr ResLimit g_no_limit_res = {1, 2201, 0, 2201, {}};
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint64_t* workspaceSize, uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size)
{
  tiling->set_block_dim(aiv_num);
  tiling->set_ub_size(ub_size - 256);
  if (!optiling::GetTiling(*tiling, -1)) {
      return -1;
  }
  *blockDim = tiling->get_block_dim();
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}
bool PGOGetTilingKey(const char *config_file_path, AutofuseTilingData &tiling_data) {
  OP_LOGD(OP_NAME, "PGOGetTilingKey from file:%s.", config_file_path);
  static int best_config = 0;
  static AutofuseTilingData best_tiling;
  if (best_config == 0) {
    std::ifstream config_file(config_file_path);
    if (!config_file.is_open()) {
      OP_LOGD(OP_NAME, "failed to open or does not exist: %s.", config_file_path);
      return false;
    }
    OP_LOGD(OP_NAME, "[Start to use tiling result]: %s.", config_file_path);
    std::string line;
    // first line: 0:read every time; 1:read first time
    std::getline(config_file, line);
    std::istringstream iss0(line);
    int flag = -1;
    iss0 >> flag;
    OP_LOGD(OP_NAME, "best_config %d.", flag);
    // second line: tiling_data dumped as int32 decimals, space-separated
    std::getline(config_file, line);
    if (line.find('#') != std::string::npos) {
        line = line.substr(0, line.find('#'));
    }
    std::istringstream iss1(line);
    std::vector<int32_t> tiling_i32;
    tiling_i32.reserve((sizeof(tiling_data) + sizeof(int32_t) - 1) / sizeof(int32_t));
    int64_t tmp = 0;
    while (iss1 >> tmp) {
      tiling_i32.push_back(static_cast<int32_t>(tmp));
    }
    const size_t expect_num = (sizeof(tiling_data) + sizeof(int32_t) - 1) / sizeof(int32_t);
    tiling_i32.resize(expect_num, 0);
    memcpy_s(&tiling_data, sizeof(tiling_data), tiling_i32.data(), sizeof(tiling_data));
    config_file.close();
    if (flag == 1) {
      best_tiling = tiling_data;
      best_config = flag;
    }
  } else {
    tiling_data = best_tiling;
  }
  return true;
}

extern "C" int64_t AutofuseTilingWithConfig(const char *config_file, AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, int32_t tiling_case_id = -1)
{
 const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  tiling->set_block_dim(limit->aiv_num);
  tiling->set_ub_size(limit->ub_size);
  if (!PGOGetTilingKey(config_file, *tiling)) {
    if (!optiling::GetTiling(*tiling, tiling_case_id)) {
      return -1;
    }
  }
  *blockDim = tiling->get_block_dim();
  using namespace optiling;
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}
void PgoSaveTilingKey(const AutofuseTilingData &tiling_data, double best_perf, std::ofstream &out_file) {
  const size_t tiling_bytes = sizeof(tiling_data);
  const size_t tiling_bytes_align = (tiling_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  std::vector<int32_t> tiling_i32(tiling_bytes_align, 0);
  memcpy_s(tiling_i32.data(), tiling_i32.size() * sizeof(int32_t), &tiling_data, tiling_bytes);
  for (size_t idx = 0; idx < tiling_i32.size(); ++idx) {
    out_file << tiling_i32[idx] << " ";
  }
  out_file << "# " << best_perf << std::endl;
}
void SavePGOSearchTilingData(char *search_file, std::vector<AutofuseTilingDataPerf> &tiling_data_list, std::ios::openmode mode = std::ios::out) {
  OP_LOGI(OP_NAME, "SavePGOSearchTilingData to file:%s.", search_file);
  std::ofstream out_file(search_file, mode);
  if (!out_file.is_open()) {
    OP_LOGE(OP_NAME, "Failed to open file:%s.", search_file);
    return;
  }
  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {
    PgoSaveTilingKey(item->tiling_data, item->best_perf, out_file);
  }
  out_file.close();

  return;
}
void SavePGOConfigTilingData(char *file, std::vector<AutofuseTilingDataPerf> &tiling_data_list, double best_perf, std::ios::openmode mode = std::ios::out) {
  OP_LOGI(OP_NAME, "SavePGOConfigTilingData to file:%s.", file);
  std::ofstream out_file(file, mode);
  if (!out_file.is_open()) {
    OP_LOGE(OP_NAME, "Failed to open file:%s.", file);
    return;
  }
  if (PgoConfig::Instance().pgo_algorithm == 1) {
    for (auto item : tiling_data_list) {
      if (item.best_perf < best_perf) {
        best_perf = item.best_perf;
      }
    }
  }
  out_file << "1" << std::endl;
  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {
    if (optiling::IsEqual(item->best_perf, best_perf)) {
      PgoSaveTilingKey(item->tiling_data, item->best_perf, out_file);
      break;
    }
  }
  out_file.close();

  return;
}
extern "C" int64_t PgoTilingSearchByCoreNum(char *search_file, char *config_file, AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, PgoTensorArgs *tensor_args = nullptr, void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback prof_batch_callback=nullptr) {
  const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  double best_perf = DBL_MAX;
  uint32_t max_block_dim = limit->aiv_num;
  auto max_core_num = 1;
  tiling->set_block_dim(max_core_num);
  max_block_dim = max_core_num;
  using namespace optiling;
  std::vector<AutofuseTilingData> tiling_data_list;
  std::vector<AutofuseTilingDataPerf> tiling_data_perf_list;
  double axeorder_cost = DBL_MAX;
  AutofuseTiling(tiling, workspaceSize, blockDim, limit->aiv_num, limit->ub_size - 256);
  PgoConfig::Instance().single_callback(tensor_args, stream, *workspaceSize, tiling, &axeorder_cost);
  AutofuseTilingDataPerf tiling_data_axereorder_perf;
  tiling_data_axereorder_perf.tiling_data = *tiling;
  tiling_data_axereorder_perf.best_perf = axeorder_cost;
  tiling_data_perf_list.push_back(tiling_data_axereorder_perf);
  PgoConfig::Instance().need_change_solver_run = true;
  PgoConfig::Instance().pgo_threshold_index = 0;
  while (PgoConfig::Instance().pgo_threshold_index < PgoConfig::Instance().pgo_threshold_list_size) {
    if (!optiling::PGOByCoreNumSearchTilingKey(tiling_data_list, tiling, max_block_dim)) {
      return -1;
    }
    PgoConfig::Instance().pgo_threshold_index++;
  }
  double out_cost = DBL_MAX;
  *workspaceSize = 0;
  std::unordered_set<std::string> solver_filter;
  for (const auto &tiling_data_item : tiling_data_list) {
    const char *ptr = reinterpret_cast<const char*>(&tiling_data_item);
    std::string key(ptr, ptr + sizeof(AutofuseTilingData));
    if (!solver_filter.insert(key).second) {
      continue;
    }
    *workspaceSize = std::max(GetWorkspaceSize(tiling_data_item), *workspaceSize);
    AutofuseTilingDataPerf tiling_data_perf;
    tiling_data_perf.tiling_data = tiling_data_item;
    tiling_data_perf.best_perf = DBL_MAX;
    tiling_data_perf_list.push_back(tiling_data_perf);
  }
  *workspaceSize += 16 * 1024 * 1024;
  PgoConfig::Instance().batch_callback(tensor_args, stream, *workspaceSize, &tiling_data_perf_list);
  best_perf = DBL_MAX;
  SavePGOSearchTilingData(search_file, tiling_data_perf_list);
  SavePGOConfigTilingData(config_file, tiling_data_perf_list, best_perf);
  return 0;
}
extern "C" int64_t PgoTilingSearchPGO(char *search_file, char *config_file, AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, PgoTensorArgs *tensor_args = nullptr, void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback prof_batch_callback=nullptr) {
  const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  std::vector<AutofuseTilingDataPerf> tiling_data_list;
  double best_perf = DBL_MAX;
  uint32_t max_block_dim = limit->aiv_num;
  auto max_core_num = 1;
  tiling->set_block_dim(max_core_num);
  max_block_dim = max_core_num;
  AutofuseTiling(tiling, workspaceSize, blockDim, limit->aiv_num, limit->ub_size - 256);
  PgoConfig::Instance().single_callback(tensor_args, stream, *workspaceSize, tiling, &best_perf);
  if (optiling::IsEqual(best_perf, DBL_MAX)) {
    OP_LOGE(OP_NAME, "axesreorder solution get perf failed %lf", best_perf);
    return -1;
  }
  AutofuseTilingDataPerf tiling_perf;
  tiling_perf.tiling_data = *tiling;
  tiling_perf.best_perf = best_perf;
  tiling_data_list.push_back(tiling_perf);
  OP_LOGD(OP_NAME, "axesreorder solution base perf is %lf", best_perf);
  tiling->set_block_dim(max_block_dim);
  if (!optiling::PGOSearchTilingKey(tiling_data_list, *tiling, -1, tiling, tensor_args, stream, *workspaceSize, best_perf)) {
    return -1;
  }
  if (optiling::IsEqual(best_perf, DBL_MAX)) {
    OP_LOGE(OP_NAME, "pgo solution get perf failed %lf", best_perf);
    return -1;
  }
  SavePGOSearchTilingData(search_file, tiling_data_list);
  SavePGOConfigTilingData(config_file, tiling_data_list, best_perf);
  OP_LOGD(OP_NAME, "pgo solution best perf is %lf", best_perf);

  return 0;
}
extern "C" int64_t PgoTilingSearch(char *search_file, char *config_file, AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, PgoTensorArgs *tensor_args = nullptr, void *stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback prof_batch_callback=nullptr) {
  const char* var = std::getenv("AUTOFUSE_DFX_FLAGS");
  if ((var != nullptr) && (std::string(var).find("autofuse_pgo_algo=pruning") != std::string::npos)) {
    PgoConfig::Instance().pgo_algorithm = 0;
  } else {
    PgoConfig::Instance().pgo_algorithm = 1;
  }
  PgoConfig::Instance().single_callback = prof_callback;
  PgoConfig::Instance().batch_callback = prof_batch_callback;
  if (PgoConfig::Instance().pgo_algorithm == 0) {
    PgoTilingSearchPGO(search_file, config_file,  tiling, workspaceSize, blockDim, res_limit, tensor_args, stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);
  } else if (PgoConfig::Instance().pgo_algorithm == 1) {
    PgoTilingSearchByCoreNum(search_file, config_file,  tiling, workspaceSize, blockDim, res_limit, tensor_args, stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);
  }
  return 0;
}

#ifndef __CCE_KT_TEST__
extern "C" bool AutofuseIsStaticShape() {
  return true;
}
extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{

  return -1;
}
extern "C" uint64_t GetTilingKeyCount()
{
  return 0;
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
}
bool version_is_ASCEND950 = false;
struct AfTilingParseData{
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
 aiv_num = std::min(platform_core_num, static_cast<uint32_t>(1));
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
 if (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
 version_is_ASCEND950 = true;
 }
 ub_size -= (ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910 && ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910B && ub_size % 1024 == 0) ? 256 : 0;
 (*tiling_parse_data)->ub_size = ub_size;
 return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)
{
  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);
  auto input_data_num =  extend_context->GetInputValue<size_t>(0U);
  auto parse = extend_context->GetInputValue<AfTilingParseData*>(input_data_num + 1);
  uint64_t workspace_size;
  uint32_t block_dim;
  static const char* config_file = "/test_graph_config.txt";
  auto tiling_data =  context->GetTilingData<AutofuseTilingData>();
  ResLimit limit;
  limit.aiv_num = parse->aiv_num;
  limit.ub_size = (uint32_t)parse->ub_size;
  auto ret = AutofuseTilingWithConfig(config_file, tiling_data, &workspace_size, &block_dim, &limit);
  context->SetBlockDim(block_dim);
  *context->GetWorkspaceSizes(1) = workspace_size;

  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
  return ret;
}

extern "C" ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context)
{
  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);
  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);
  if (symbol_src_vec == nullptr) {
    return ge::GRAPH_FAILED;
  }

  symbol_src_vec->SetSize(0);
  return ge::GRAPH_SUCCESS;
}
extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;

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

std::string tiling_data_const_gen_result;
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
  (void)AutofuseTilingWithConfig(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit);
  std::string GenTilingDataValue_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.block_dim);
  std::string GenTilingDataValue_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.corenum);
  std::string GenTilingDataValue_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.ub_size);
  std::string GenTilingDataValue_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.hbm_size);
  std::string GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("graph0_tiling_key", TilingDataValue.graph0_tiling_key);

  tiling_data_const_gen_result = R"(#ifndef __Autofuse_Tiling_Data_H__
#define __Autofuse_Tiling_Data_H__
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

BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
  GenTilingDataValue_block_dim_field_DeclareFunc_def
  GenTilingDataValue_corenum_field_DeclareFunc_def
  GenTilingDataValue_ub_size_field_DeclareFunc_def
  GenTilingDataValue_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def
END_TILING_DATA_DEF_T;

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

  return tiling_data_const_gen_result.c_str();
}


#ifndef __CCE_KT_TEST__
extern "C" int64_t GetTilingKeyForStatic()
{
  return FindBestTilingKey(TilingDataValue);
}
std::string kernel_type;
extern "C" const char* GetTilingKeyKernelTypeForStatic()
{
  const std::map<int64_t, std::string> kernel_type_map = {
  };

  auto tiling_key = FindBestTilingKey(TilingDataValue);
  auto it = kernel_type_map.find(tiling_key);
  if (it != kernel_type_map.end()) {
    kernel_type = it->second;
  }
  return kernel_type.c_str();
}
#endif
)rawliteral";
}

TEST_F(E2E_LoadAbsStore, ConstructGraphWithAscir) {
  af::AscGraph test_graph("test_load_abs_store");
  LoadAbsStore_BeforeAutofuse(test_graph);
  GTEST_SKIP() << "Compare expect graph ir info here";
}

TEST_F(E2E_LoadAbsStore, GetApiInfo) {
  af::AscGraph expect_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);

  af::AscGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  af::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  af::AscGraph test_optimize_graph("test_optimize_graph");
  test_optimize_graph.CopyFrom(test_graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(test_optimize_graph);

  EXPECT_EQ(utils::DebugHintGraphStr(test_graph), utils::DebugHintGraphStr(expect_graph));
}

TEST_F(E2E_LoadAbsStore, Codegen_TilingData) {
  af::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  FusedScheduledResult fused_schedule_result;
  auto tiling_data_code = codegen.GenerateTilingData(fused_schedule_result);
  std::cout << tiling_data_code << std::endl;
  const std::string test_res = R"rawliteral(#ifndef __Autofuse_Tiling_Data_H__
#define __Autofuse_Tiling_Data_H__
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

BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);

END_TILING_DATA_DEF_T;

struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)rawliteral";
  EXPECT_EQ(tiling_data_code, test_res);
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling_With_Lambda) {
  af::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);
  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto shape_info = MakeDefaultShapeInfo();
  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  auto tiling_codes = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0");
  for (const auto &[key, value] : tiling_codes) {
    std::cout << key << std::endl;
    std::cout << value << std::endl;
  }
  std::string tiling_code;
  att::test::CombineTilings(tiling_codes, tiling_code);
  EXPECT_NE(tiling_code.find("extern \"C\" int64_t AutofuseTiling"), std::string::npos);
  EXPECT_NE(tiling_code.find("extern \"C\" int64_t AutofuseTilingWithConfig"), std::string::npos);
  EXPECT_NE(tiling_code.find("GenConstTilingData"), std::string::npos);
  EXPECT_NE(tiling_code.find("ResLimit limit"), std::string::npos);
  EXPECT_EQ(tiling_code.find("AscirCompileAndLaunch"), std::string::npos);
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling_With_Set_Vector_Core_Num) {
  af::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::string s0_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(0);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s1_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s2_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(2);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";

  std::map<std::string, std::string> shape_info = {{"s0", s0_source}, {"s1", s1_source}, {"s2", s2_source}};
  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  auto tiling_codes = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "20");
}

TEST_F(E2E_LoadAbsStore, Codegen_PGO_Code) {
  af::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);
  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);
  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  codegen::Codegen codegen(codegen::CodegenOptions{.tiling_lib_path = "asdf", .tiling_lib_codegen_symbol = "as"});
  std::string pgo_codes = codegen.GeneratorPgo(fused_schedule_result, "");
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=false", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  // PGO loader code intentionally evolves with its DSO lifetime safeguards. Keep this
  // test focused on the generated PGO contract instead of byte-for-byte source text.
  AssertPgoCodegenContract(pgo_codes);
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling_With_LambdaWithPGO) {
  af::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);
  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  auto shape_info = MakeDefaultShapeInfo();
  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_pgo_algo=pgo_algo_invalid;--autofuse_pgo_step_max=32", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  codegen::Codegen codegen(codegen::CodegenOptions{.tiling_lib_path = "asdf", .tiling_lib_codegen_symbol = "as"});
  auto tiling_codes = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "1");
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo, "true");
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().autofuse_pgo_algo_select, "core_select");
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().autofuse_pgo_algo_step_max, 32);

  for (const auto &[key, value] : tiling_codes) {
    std::cout << key << std::endl;
    std::cout << value << std::endl;
  }
  std::string tiling_code;
  att::test::CombineTilings(tiling_codes, tiling_code);
  EXPECT_NE(tiling_code.find("PgoConfig"), std::string::npos);
  EXPECT_NE(tiling_code.find("ProfilingCallback"), std::string::npos);
  EXPECT_NE(tiling_code.find("extern \"C\" int64_t AutofuseTilingWithConfig"), std::string::npos);
  EXPECT_NE(tiling_code.find("GenConstTilingData"), std::string::npos);
  EXPECT_EQ(tiling_code.find("AscirCompileAndLaunch"), std::string::npos);

  ScheduledResult scheduled_result;
  ScheduleGroup schedule_group;
  fused_schedule_result.node_idx_to_scheduled_results[0].push_back(scheduled_result);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group);
  tiling_codes = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0");
  tiling_code;
  att::test::CombineTilings(tiling_codes, tiling_code);
  EXPECT_NE(tiling_code.find("  std::unordered_map<int64_t, uint64_t> workspace_map;"), std::string::npos);
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=false", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
}
