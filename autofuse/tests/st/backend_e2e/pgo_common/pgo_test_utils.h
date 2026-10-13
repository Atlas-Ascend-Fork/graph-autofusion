/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_ST_BACKEND_E2E_PGO_COMMON_PGO_TEST_UTILS_H_
#define AUTOFUSE_TESTS_ST_BACKEND_E2E_PGO_COMMON_PGO_TEST_UTILS_H_

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

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

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};

typedef long int (*ProfilingCallback)(PgoTensorArgs *tensor_args, void *stream, uint64_t workspaceSize,
                                      AutofuseTilingData *tiling_data, double *cost_time);
typedef long int (*ProfilingBatchCallback)(PgoTensorArgs *tensor_args, void *stream, uint64_t workspaceSize,
                                           std::vector<AutofuseTilingDataPerf> *profiles);

extern "C" int64_t PgoTilingSearch(char *search_file, char *config_file, AutofuseTilingData *tiling,
                                   uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr,
                                   PgoTensorArgs *tensor_args = nullptr, void *stream = nullptr,
                                   ProfilingCallback prof_callback = nullptr,
                                   ProfilingBatchCallback prof_batch_callback = nullptr);

struct PgoTensorArgPack {
  void *inputs[2];
  void *outputs[1];
  PgoTensorArgs tensor_args;

  PgoTensorArgPack(void *input0, void *input1, void *output)
      : inputs{input0, input1}, outputs{output}, tensor_args{inputs, 2U, outputs, 1U} {}
};

inline int RunPgoTilingSearch(const std::filesystem::path &case_file, AutofuseTilingData *tiling,
                              uint64_t *workspace_size, uint32_t *block_dim, ResLimit *res_limit,
                              PgoTensorArgs *tensor_args, ProfilingCallback prof_callback,
                              ProfilingBatchCallback prof_batch_callback) {
  const std::string search_file = (case_file.parent_path() / "search.txt").string();
  const std::string config_file = (case_file.parent_path() / "config.txt").string();
  return PgoTilingSearch(const_cast<char *>(search_file.c_str()), const_cast<char *>(config_file.c_str()), tiling,
                         workspace_size, block_dim, res_limit, tensor_args, nullptr, prof_callback,
                         prof_batch_callback);
}

inline int RunPgoTilingSearchWithPruning(const std::filesystem::path &case_file, AutofuseTilingData *tiling,
                                         uint64_t *workspace_size, uint32_t *block_dim, ResLimit *res_limit,
                                         PgoTensorArgs *tensor_args, ProfilingCallback prof_callback,
                                         ProfilingBatchCallback prof_batch_callback) {
  const char *saved_flags = std::getenv("AUTOFUSE_DFX_FLAGS");
  const std::string saved_flags_str = saved_flags == nullptr ? "" : saved_flags;

  setenv("AUTOFUSE_DFX_FLAGS", "autofuse_pgo_algo=pruning", 1);
  int result = RunPgoTilingSearch(case_file, tiling, workspace_size, block_dim, res_limit, tensor_args, prof_callback,
                                  prof_batch_callback);

  if (saved_flags == nullptr) {
    unsetenv("AUTOFUSE_DFX_FLAGS");
  } else {
    setenv("AUTOFUSE_DFX_FLAGS", saved_flags_str.c_str(), 1);
  }
  return result;
}

#endif  // AUTOFUSE_TESTS_ST_BACKEND_E2E_PGO_COMMON_PGO_TEST_UTILS_H_
