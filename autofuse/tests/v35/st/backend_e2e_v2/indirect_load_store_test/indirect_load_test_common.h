/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Shared declarations and helpers for the IndirectLoad backend E2E test kernel.
// Kept in a separate header so the case bodies in
// test_e2e_indirect_load_store_kernel.cpp stay focused on data setup and oracles.

#ifndef AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_STORE_TEST_INDIRECT_LOAD_TEST_COMMON_H_
#define AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_STORE_TEST_INDIRECT_LOAD_TEST_COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"

#if !defined(IL_CASE_STORE) && !defined(IL_CASE_MIXED)
struct ResLimit;
extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint64_t *, uint32_t *, uint32_t, uint32_t);
extern "C" int64_t AutofuseTilingWithConfig(const char *, AutofuseTilingData *, uint64_t *, uint32_t *, ResLimit *,
                                            int32_t);
#endif

#if defined(IL_USER_FANOUT)
extern "C" __global__ __aicore__ void user_fanout(GM_ADDR indices, GM_ADDR embedding, GM_ADDR weight, GM_ADDR output0,
                                                  GM_ADDR output1, GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_USER_FANOUT_SIDE_INPUT)
extern "C" __global__ __aicore__ void user_fanout_side_input(GM_ADDR indices, GM_ADDR embedding, GM_ADDR side_input,
                                                             GM_ADDR output0, GM_ADDR output1, GM_ADDR workspace,
                                                             GM_ADDR gm_tiling_data);
#elif defined(IL_USER_SIDE_INPUT_FANOUT)
extern "C" __global__ __aicore__ void user_side_input_fanout(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                             GM_ADDR input3, GM_ADDR input4, GM_ADDR input5,
                                                             GM_ADDR output0, GM_ADDR output1, GM_ADDR workspace,
                                                             GM_ADDR gm_tiling_data);
#elif defined(IL_USER_EMBEDDING_EXP_ABS_ADD)
extern "C" __global__ __aicore__ void user_embedding_exp_abs_add(GM_ADDR indices, GM_ADDR embedding, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_USER_EMBEDDING_SUM)
extern "C" __global__ __aicore__ void user_embedding_sum(GM_ADDR table, GM_ADDR indices, GM_ADDR output,
                                                         GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_USER_LAYERNORM)
extern "C" __global__ __aicore__ void user_layernorm(GM_ADDR indices, GM_ADDR embedding, GM_ADDR weight,
                                                     GM_ADDR raw_output, GM_ADDR square_output, GM_ADDR workspace,
                                                     GM_ADDR gm_tiling_data);
#elif defined(IL_USER_SOFTMAX)
extern "C" __global__ __aicore__ void user_softmax(GM_ADDR indices, GM_ADDR embedding, GM_ADDR bias, GM_ADDR bmm,
                                                   GM_ADDR output, GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_DUAL_IL_GATHER)
extern "C" __global__ __aicore__ void user_add_gather(GM_ADDR input0, GM_ADDR input1, GM_ADDR indices, GM_ADDR output,
                                                      GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_USER_GATHER_SUM_TRANSPOSE)
extern "C" __global__ __aicore__ void autofused_gather_sum_transpose_9d393a67d2eb1a45f6dd4ea95c860eac(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_USER_ABS_EMBEDDING_SUM)
extern "C" __global__ __aicore__ void autofused_abs_embedding_sum_1ee65c13970c2252fddcc2b670e5212d(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_USER_INT64_GATHER_DENSE) || defined(IL_USER_INT64_GATHER_DENSE_SIMD) || \
    defined(IL_USER_INT64_GATHER_DENSE_SIMT)
extern "C" __global__ __aicore__ void user_int64_gather_dense(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                              GM_ADDR workspace, GM_ADDR gm_tiling_data);
#elif defined(IL_USER_DTYPE_INT64)
extern "C" __global__ __aicore__ void user_dtype_int64_embedding(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR gm_tiling_data);
#define USER_DTYPE_KERNEL user_dtype_int64_embedding
#elif defined(IL_USER_DTYPE_UINT64)
extern "C" __global__ __aicore__ void user_dtype_uint64_embedding(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                  GM_ADDR workspace, GM_ADDR gm_tiling_data);
#define USER_DTYPE_KERNEL user_dtype_uint64_embedding
#elif defined(IL_USER_DTYPE_INT8)
extern "C" __global__ __aicore__ void user_dtype_int8_embedding(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                GM_ADDR workspace, GM_ADDR gm_tiling_data);
#define USER_DTYPE_KERNEL user_dtype_int8_embedding
#elif defined(IL_USER_DTYPE_UINT8)
extern "C" __global__ __aicore__ void user_dtype_uint8_embedding(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR gm_tiling_data);
#define USER_DTYPE_KERNEL user_dtype_uint8_embedding
#elif defined(IL_USER_DTYPE_BOOL)
extern "C" __global__ __aicore__ void user_dtype_bool_embedding(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                                GM_ADDR workspace, GM_ADDR gm_tiling_data);
#define USER_DTYPE_KERNEL user_dtype_bool_embedding
#endif

namespace indirect_load_test {
inline void GmFree(void *ptr) {
  AscendC::GmFree(ptr);
}

// RAII GM buffer; released automatically even when a test assertion aborts early.
template <typename T>
std::unique_ptr<T, decltype(&GmFree)> AllocGmBuffer(int64_t count) {
  return {static_cast<T *>(AscendC::GmAlloc(static_cast<size_t>(count) * sizeof(T))), GmFree};
}

template <typename DataType, typename IndexType>
struct KernelData {
  KernelData(int64_t input_count, int64_t index_count, int64_t output_count)
      : input(reinterpret_cast<DataType *>(AscendC::GmAlloc(input_count * sizeof(DataType))), GmFree),
        index(reinterpret_cast<IndexType *>(AscendC::GmAlloc(index_count * sizeof(IndexType))), GmFree),
        output(reinterpret_cast<DataType *>(AscendC::GmAlloc(output_count * sizeof(DataType))), GmFree),
        expected(static_cast<size_t>(output_count)) {}

  [[nodiscard]] bool IsValid() const {
    return input != nullptr && index != nullptr && output != nullptr;
  }

  std::unique_ptr<DataType, decltype(&GmFree)> input;
  std::unique_ptr<IndexType, decltype(&GmFree)> index;
  std::unique_ptr<DataType, decltype(&GmFree)> output;
  std::vector<DataType> expected;
};

#if !defined(IL_CASE_STORE) && !defined(IL_CASE_MIXED)
struct KernelTiling {
  explicit KernelTiling(uint32_t core_num = 48U) : workspace(nullptr, GmFree) {
    EXPECT_EQ(AutofuseTiling(&data, &workspace_size, &block_dim, core_num, 192U * 1024U), 0);
#ifdef IL_FORCE_TILING_CASE
    EXPECT_EQ(AutofuseTilingWithConfig(nullptr, &data, &workspace_size, &block_dim, nullptr, IL_FORCE_TILING_CASE), 0);
    EXPECT_EQ(data.graph0_result0_g0_tiling_data.tiling_key, static_cast<uint32_t>(IL_FORCE_TILING_CASE));
#endif
    EXPECT_GT(data.block_dim, 0U);
    if (workspace_size != 0U) {
      workspace.reset(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspace_size)));
    }
  }

  [[nodiscard]] bool IsValid() const {
    return workspace_size == 0U || workspace != nullptr;
  }

  AutofuseTilingData data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  std::unique_ptr<uint8_t, decltype(&GmFree)> workspace;
};

// Shared tiling helpers for the user graphs whose shapes are symbolic size vars.
constexpr uint32_t kUserAivNum = 48U;
constexpr uint32_t kUserUbSize = 192U * 1024U;

using FillTilingFn = void (*)(AutofuseTilingData &, uint32_t, uint32_t, uint32_t, uint32_t);

// Fills the symbolic size vars, runs runtime tiling and exposes the tiling data
// plus the workspace required to launch the generated kernel.
struct UserGraphTiling {
  void Tile(FillTilingFn fill, uint32_t ks0, uint32_t ks1, uint32_t ks2, uint32_t ks3) {
    fill(data, ks0, ks1, ks2, ks3);
    ASSERT_EQ(AutofuseTiling(&data, &workspace_size, &block_dim, kUserAivNum, kUserUbSize), 0);
    if (workspace_size != 0U) {
      workspace.reset(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspace_size)));
    }
    ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  }

  AutofuseTilingData data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = kUserAivNum;
  std::unique_ptr<uint8_t, decltype(&GmFree)> workspace{nullptr, GmFree};
};

// Reports whether the given symbolic shapes tile successfully (dynamic-shape regression).
inline bool TilingSucceeds(FillTilingFn fill, uint32_t ks0, uint32_t ks1, uint32_t ks2, uint32_t ks3) {
  AutofuseTilingData tiling_data{};
  fill(tiling_data, ks0, ks1, ks2, ks3);
  uint64_t workspace_size = 0U;
  uint32_t block_dim = kUserAivNum;
  return AutofuseTiling(&tiling_data, &workspace_size, &block_dim, kUserAivNum, kUserUbSize) == 0;
}
#endif

// Runs one backend kernel on the CPU simulator.  Buffer arguments are forwarded
// ahead of the workspace and tiling-data pointers, matching the generated
// GM_ADDR signature.
template <typename KernelT, typename... BuffersT>
void RunKernel(KernelT kernel, uint32_t block_dim, uint8_t *workspace, uint8_t *tiling_data, BuffersT... buffers) {
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, block_dim, reinterpret_cast<uint8_t *>(buffers)..., workspace, tiling_data);
}
}  // namespace indirect_load_test

#endif  // AUTOFUSE_TESTS_V35_ST_BACKEND_E2E_V2_INDIRECT_LOAD_STORE_TEST_INDIRECT_LOAD_TEST_COMMON_H_
