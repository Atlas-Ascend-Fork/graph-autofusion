/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "indirect_load_test_common.h"

#if defined(IL_CASE_STORE) || defined(IL_CASE_MIXED)
#include <array>
#include <random>

#if defined(IL_MIXED_ELEMENTWISE) && IL_EXPECT_SIMT
extern "C" __global__ __aicore__ void indirect_load_mixed_elementwise_test(GM_ADDR x, GM_ADDR index0, GM_ADDR index1,
                                                                           GM_ADDR addend, GM_ADDR sign, GM_ADDR scale,
                                                                           GM_ADDR output, GM_ADDR workspace,
                                                                           GM_ADDR tiling);
#elif defined(IL_MIXED_ELEMENTWISE)
extern "C" __global__ __aicore__ void indirect_load_mixed_elementwise_test(GM_ADDR x0, GM_ADDR x1, GM_ADDR index0,
                                                                           GM_ADDR index1, GM_ADDR addend, GM_ADDR sign,
                                                                           GM_ADDR scale, GM_ADDR output,
                                                                           GM_ADDR workspace, GM_ADDR tiling);
#elif defined(IL_POST_REDUCE_ADD)
extern "C" __global__ __aicore__ void indirect_load_store_test(GM_ADDR x, GM_ADDR index, GM_ADDR addend, GM_ADDR y,
                                                               GM_ADDR workspace, GM_ADDR tiling);
#else
extern "C" __global__ __aicore__ void indirect_load_store_test(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                               GM_ADDR tiling);
#endif
#ifdef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint64_t *, uint32_t *, uint32_t, uint32_t);
#endif
#if IL_RANK == 2
#ifndef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, AutofuseTilingData *, uint64_t *,
                                  uint32_t *, uint32_t, uint32_t);
#endif
constexpr std::array<int32_t, 2> kInputShape = {IL_X_S0, IL_X_S1};
constexpr std::array<int32_t, 2> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1};
#elif IL_RANK == 3
#ifndef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                                  AutofuseTilingData *, uint64_t *, uint32_t *, uint32_t, uint32_t);
#endif
constexpr std::array<int32_t, 3> kInputShape = {IL_X_S0, IL_X_S1, IL_X_S2};
constexpr std::array<int32_t, 3> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2};
#elif IL_RANK == 4
#ifndef IL_STATIC_SHAPE
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
                                  uint32_t s6, uint32_t s7, AutofuseTilingData *, uint64_t *, uint32_t *, uint32_t,
                                  uint32_t);
#endif
constexpr std::array<int32_t, 4> kInputShape = {IL_X_S0, IL_X_S1, IL_X_S2, IL_X_S3};
constexpr std::array<int32_t, 4> kIndexShape = {IL_INDEX_S0, IL_INDEX_S1, IL_INDEX_S2, IL_INDEX_S3};
#endif

namespace {
constexpr int32_t kAxis = IL_AXIS < 0 ? IL_AXIS + IL_RANK : IL_AXIS;
#if defined(IL_DATA_BF16)
using DataType = bfloat16_t;
#elif defined(IL_DATA_INT16)
using DataType = int16_t;
#elif defined(IL_DATA_UINT32)
using DataType = uint32_t;
#elif defined(IL_DATA_FLOAT)
using DataType = float;
#else
using DataType = half;
#endif
#ifdef IL_OUTPUT_FLOAT
using OutputType = float;
#else
using OutputType = DataType;
#endif
#if IL_BOTH_TRANSPOSE
using IndexType = int32_t;
#elif defined(IL_INDEX_INT64)
using IndexType = int64_t;
#else
using IndexType = int32_t;
#endif

float BesselK0Reference(float x) {
  const float y = x * x / 4.0F;
  float series_term = 1.0F;
  float series_sum = 1.0F;
  for (int32_t order = 1; order <= 20; ++order) {
    const float scale = x * 0.5F / static_cast<float>(order);
    series_term *= scale * scale;
    series_sum += series_term;
  }
  const float poly =
      -0.57721566F +
      y * (0.42278420F +
           y * (0.23069756F + y * (0.03488590F + y * (0.00262698F + y * (0.00010750F + y * 0.00000740F)))));
  return -std::log(x * 0.5F) * series_sum + poly;
}

void RunTiling(AutofuseTilingData &tiling, uint64_t &workspace_size, uint32_t &block_dim) {
#ifdef IL_FORCE_TILING_CASE
  ASSERT_EQ(AutofuseTilingWithConfig(nullptr, &tiling, &workspace_size, &block_dim, nullptr, IL_FORCE_TILING_CASE), 0);
  ASSERT_EQ(tiling.graph0_tiling_key, 0U);
  ASSERT_EQ(tiling.graph0_result0_g0_tiling_data.tiling_key, static_cast<uint32_t>(IL_FORCE_TILING_CASE));
  return;
#endif
#ifdef IL_STATIC_SHAPE
  AutofuseTiling(&tiling, &workspace_size, &block_dim, 48, 192 * 1024);
#elif IL_RANK == 2
  AutofuseTiling(kInputShape[0], kInputShape[1], kIndexShape[0], kIndexShape[1], &tiling, &workspace_size, &block_dim,
                 48, 192 * 1024);
#elif IL_RANK == 3
  AutofuseTiling(kInputShape[0], kInputShape[1], kInputShape[2], kIndexShape[0], kIndexShape[1], kIndexShape[2],
                 &tiling, &workspace_size, &block_dim, 48, 192 * 1024);
#elif IL_RANK == 4
  AutofuseTiling(kInputShape[0], kInputShape[1], kInputShape[2], kInputShape[3], kIndexShape[0], kIndexShape[1],
                 kIndexShape[2], kIndexShape[3], &tiling, &workspace_size, &block_dim, 48, 192 * 1024);
#endif
}

template <size_t N>
constexpr int32_t ElementCount(const std::array<int32_t, N> &shape) {
  int32_t count = 1;
  for (int32_t dim : shape) {
    count *= dim;
  }
  return count;
}

constexpr int32_t ResultCount() {
#ifdef IL_POST_REDUCE
#ifdef IL_REDUCE_LAST_AXIS
  return kIndexShape[0] * kIndexShape[1] * kIndexShape[2];
#elif defined(IL_REDUCE_WITH_A)
  return kIndexShape[0] * kIndexShape[1] * kIndexShape[3];
#else
  return kIndexShape[0] * kIndexShape[1];
#endif
#else
  return ElementCount(kIndexShape);
#endif
}

int32_t InputStorageCount() {
#ifdef IL_INPUT_OUTER_STRIDE
  int32_t inner_count = 1;
  for (size_t dim = 1UL; dim < kInputShape.size(); ++dim) {
    inner_count *= kInputShape[dim];
  }
  return (kInputShape[0] - 1) * IL_INPUT_OUTER_STRIDE + inner_count;
#else
  return ElementCount(kInputShape);
#endif
}

void InitializeData(DataType *x, IndexType *index, DataType *addend, OutputType *expected, int32_t input_count,
                    int32_t output_count) {
#ifdef IL_RANDOM_INPUT_INDEX
  std::mt19937 generator(20260805U);
  std::uniform_real_distribution<float> input_distribution(1.0F, 2.0F);
  std::uniform_int_distribution<int64_t> index_distribution(1L, 9L);
#endif
  std::array<int32_t, IL_RANK> input_strides{};
  input_strides.back() = 1;
  for (size_t i = input_strides.size() - 1UL; i > 0UL; --i) {
    input_strides[i - 1UL] = kInputShape[i] * input_strides[i];
  }
#ifdef IL_INPUT_OUTER_STRIDE
  input_strides[0] = IL_INPUT_OUTER_STRIDE;
#endif
  for (int32_t i = 0; i < input_count; ++i) {
#if defined(IL_DATA_UINT32)
    x[i] = static_cast<DataType>(i % 5);
#elif defined(IL_POST_REDUCE_SIMD)
    x[i] = static_cast<DataType>(static_cast<float>((i % 8) + 1) * 0.25F);
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 1
    x[i] = static_cast<DataType>(static_cast<float>((i % 8) + 1) * 0.25F);
#elif defined(IL_RANDOM_INPUT_INDEX)
    x[i] = static_cast<DataType>(input_distribution(generator));
#else
    x[i] = static_cast<DataType>(static_cast<float>((i % 29) - 14) * 0.25F);
#endif
  }
  for (int32_t i = 0; i < output_count; ++i) {
#ifdef IL_POST_REDUCE_ADD
    addend[i] = static_cast<DataType>(static_cast<float>((i % 7) - 3) * 0.125F);
#else
    (void)addend;
#endif
    const int32_t index_value =
#ifdef IL_RANDOM_INPUT_INDEX
        static_cast<int32_t>(index_distribution(generator));
#else
        (i * 3 + 1) % kInputShape[kAxis];
#endif
#ifdef IL_RANDOM_INPUT_INDEX
    index[i] = static_cast<IndexType>(index_value);
#else
    index[i] = static_cast<IndexType>(i % 2 == 0 ? index_value : -index_value);
#endif
    int32_t remaining = i;
    int32_t input_offset = 0;
    for (size_t dim = kIndexShape.size(); dim-- > 0UL;) {
      const int32_t coord = remaining % kIndexShape[dim];
      remaining /= kIndexShape[dim];
      input_offset += (static_cast<int32_t>(dim) == kAxis ? index_value : coord) * input_strides[dim];
    }
    float value = static_cast<float>(x[input_offset]);
#ifdef IL_POST_REDUCE
#if IL_INPUT_PRE_TYPE == 3
    value = std::exp2(std::max(value, 0.0F));
#endif
#ifdef IL_POST_REDUCE_SIMD
    value = BesselK0Reference(value);
#endif
#ifdef IL_POST_REDUCE_ABS
    value = std::abs(value);
#endif
#ifdef IL_POST_REDUCE_EXP2
    value = std::exp2(value);
#endif
#ifdef IL_POST_REDUCE_ADD
    value += static_cast<float>(addend[i]);
#endif
#ifdef IL_REDUCE_LAST_AXIS
    const int32_t result_index = i / kIndexShape[3];
#elif defined(IL_REDUCE_WITH_A)
    const int32_t result_index = i / (kIndexShape[2] * kIndexShape[3]) * kIndexShape[3] + i % kIndexShape[3];
#else
    const int32_t result_index = i / (kIndexShape[2] * kIndexShape[3]);
#endif
    expected[result_index] = static_cast<OutputType>(static_cast<float>(expected[result_index]) + value);
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 1
    expected[i] = static_cast<DataType>(BesselK0Reference(value));
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 2
    expected[i] = static_cast<DataType>(value);
#elif defined(IL_OUTPUT_POST_TYPE) && IL_OUTPUT_POST_TYPE == 3
    expected[i] = static_cast<DataType>(std::exp2(value));
#elif IL_INPUT_PRE_TYPE == 5
    expected[i] = static_cast<DataType>(IL_USE_EXP2 ? std::exp2(value) : std::exp(value));
#else
#if IL_INPUT_PRE_TYPE == 1 || IL_INPUT_PRE_TYPE == 4
    value = std::max(value, 0.0F);
#elif IL_INPUT_PRE_TYPE == 3
    value = std::exp2(std::max(value, 0.0F));
#endif
#if IL_INPUT_PRE_TYPE == 2
    expected[i] = static_cast<DataType>(-std::exp2(value));
#elif IL_INPUT_PRE_TYPE == 3
    expected[i] = static_cast<DataType>(-value);
#elif IL_USE_EXP2
    expected[i] = static_cast<DataType>(-std::exp2(value));
#else
    expected[i] = static_cast<DataType>(-std::exp(value));
#endif
#endif
  }
}

#ifdef IL_MIXED_ELEMENTWISE
template <typename T>
std::unique_ptr<T, void (*)(void *)> Allocate(size_t count) {
  return {reinterpret_cast<T *>(AscendC::GmAlloc(count * sizeof(T))), AscendC::GmFree};
}

std::array<int32_t, IL_RANK> InputStrides() {
  std::array<int32_t, IL_RANK> strides{};
  strides.back() = 1;
  for (size_t i = strides.size() - 1UL; i > 0UL; --i) {
    strides[i - 1UL] = kInputShape[i] * strides[i];
  }
  return strides;
}

int32_t GatherInputOffset(int32_t output_index, int32_t gather_index,
                          const std::array<int32_t, IL_RANK> &input_strides) {
  int32_t remaining = output_index;
  int32_t input_offset = 0;
  for (size_t dim = kIndexShape.size(); dim-- > 0UL;) {
    const int32_t coord = remaining % kIndexShape[dim];
    remaining /= kIndexShape[dim];
    input_offset += (static_cast<int32_t>(dim) == kAxis ? gather_index : coord) * input_strides[dim];
  }
  return input_offset;
}

bool IsGatherShapeValid() {
  if (kInputShape == kIndexShape) {
    return false;
  }
  for (size_t dim = 0UL; dim < kInputShape.size(); ++dim) {
    if (static_cast<int32_t>(dim) != kAxis && kInputShape[dim] < kIndexShape[dim]) {
      return false;
    }
  }
  return true;
}

void RunMixedElementwiseCase() {
  ASSERT_TRUE(IsGatherShapeValid());
  const int32_t input_count = ElementCount(kInputShape);
  const int32_t output_count = ElementCount(kIndexShape);
  const auto input_strides = InputStrides();
  auto x0 = Allocate<DataType>(input_count);
#if !IL_EXPECT_SIMT
  auto x1 = Allocate<DataType>(input_count);
#endif
  auto index0 = Allocate<IndexType>(output_count);
  auto index1 = Allocate<IndexType>(output_count);
  auto addend = Allocate<DataType>(output_count);
  auto sign = Allocate<DataType>(output_count);
  auto scale = Allocate<DataType>(output_count);
  auto output = Allocate<DataType>(output_count);
  ASSERT_NE(x0, nullptr);
#if !IL_EXPECT_SIMT
  ASSERT_NE(x1, nullptr);
#endif
  ASSERT_NE(index0, nullptr);
  ASSERT_NE(index1, nullptr);
  ASSERT_NE(addend, nullptr);
  ASSERT_NE(sign, nullptr);
  ASSERT_NE(scale, nullptr);
  ASSERT_NE(output, nullptr);
  for (int32_t i = 0; i < input_count; ++i) {
    x0.get()[i] = static_cast<DataType>(static_cast<float>((i % 13) + 1) * 0.25F);
#if !IL_EXPECT_SIMT
    x1.get()[i] = static_cast<DataType>(i % 3 == 0 ? -1.0F : 1.0F);
#endif
  }
  std::vector<DataType> expected(static_cast<size_t>(output_count));
  for (int32_t i = 0; i < output_count; ++i) {
    const int32_t gather_index = (i * 3 + 1) % kInputShape[kAxis];
    index0.get()[i] = static_cast<IndexType>(gather_index / 2);
    index1.get()[i] = static_cast<IndexType>(gather_index - static_cast<int32_t>(index0.get()[i]));
    addend.get()[i] = static_cast<DataType>(static_cast<float>((i % 5) - 2) * 0.125F);
    sign.get()[i] = static_cast<DataType>(i % 3 == 0 ? -1.0F : 1.0F);
    scale.get()[i] = static_cast<DataType>(static_cast<float>((i % 4) + 1) * 0.25F);
#ifdef IL_SIMT_ELEMENTWISE_COVERAGE
    const int32_t effective_index =
        std::max(static_cast<int32_t>(index0.get()[i]), static_cast<int32_t>(index1.get()[i]));
#else
    const int32_t effective_index = gather_index;
#endif
    const int32_t input_offset = GatherInputOffset(i, effective_index, input_strides);
#if IL_EXPECT_SIMT
    const float gathered = static_cast<float>(x0.get()[input_offset]);
#else
    const float gathered =
        std::copysign(static_cast<float>(x0.get()[input_offset]), static_cast<float>(x1.get()[input_offset]));
#endif
#ifdef IL_SIMT_ELEMENTWISE_COVERAGE
    const float post =
        std::copysign((gathered - static_cast<float>(addend.get()[i])) / static_cast<float>(scale.get()[i]),
                      static_cast<float>(sign.get()[i]));
    expected[static_cast<size_t>(i)] = static_cast<DataType>(post);
#else
    const float post = std::copysign(gathered + static_cast<float>(addend.get()[i]), static_cast<float>(sign.get()[i]));
    expected[static_cast<size_t>(i)] = static_cast<DataType>(post * static_cast<float>(scale.get()[i]));
#endif
    output.get()[i] = static_cast<DataType>(0.0F);
  }
  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  RunTiling(tiling_data, workspace_size, block_dim);
#if IL_EXPECT_SIMT
  ASSERT_EQ(tiling_data.graph0_tiling_key, IL_TILING_KEY);
#endif
  ASSERT_GT(tiling_data.block_dim, 0U);
#if IL_EXPECT_SIMT
  indirect_load_test::RunKernel(indirect_load_mixed_elementwise_test, tiling_data.block_dim, nullptr,
                                reinterpret_cast<uint8_t *>(&tiling_data), x0.get(), index0.get(), index1.get(),
                                addend.get(), sign.get(), scale.get(), output.get());
#else
  indirect_load_test::RunKernel(indirect_load_mixed_elementwise_test, tiling_data.block_dim, nullptr,
                                reinterpret_cast<uint8_t *>(&tiling_data), x0.get(), x1.get(), index0.get(),
                                index1.get(), addend.get(), sign.get(), scale.get(), output.get());
#endif
  for (int32_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
  }
}
#endif
}  // namespace

TEST(E2EIndirectLoadStore, GeneratedKernelMatchesReference) {
#ifdef IL_MIXED_ELEMENTWISE
  RunMixedElementwiseCase();
#else
  const int32_t input_count = InputStorageCount();
  const int32_t output_count = ElementCount(kIndexShape);
  const int32_t result_count = ResultCount();
  const auto gm_free = [](void *ptr) { AscendC::GmFree(ptr); };
  std::unique_ptr<DataType, decltype(gm_free)> x(
      reinterpret_cast<DataType *>(AscendC::GmAlloc(input_count * sizeof(DataType))), gm_free);
  std::unique_ptr<IndexType, decltype(gm_free)> index(
      reinterpret_cast<IndexType *>(AscendC::GmAlloc(output_count * sizeof(IndexType))), gm_free);
#ifdef IL_POST_REDUCE_ADD
  std::unique_ptr<DataType, decltype(gm_free)> addend(
      reinterpret_cast<DataType *>(AscendC::GmAlloc(output_count * sizeof(DataType))), gm_free);
#else
  DataType *addend = nullptr;
#endif
  std::unique_ptr<OutputType, decltype(gm_free)> output(
      reinterpret_cast<OutputType *>(AscendC::GmAlloc(result_count * sizeof(OutputType))), gm_free);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(output, nullptr);
  std::vector<OutputType> expected(static_cast<size_t>(result_count));
#ifdef IL_POST_REDUCE_ADD
  ASSERT_NE(addend, nullptr);
  InitializeData(x.get(), index.get(), addend.get(), expected.data(), input_count, output_count);
#else
  InitializeData(x.get(), index.get(), addend, expected.data(), input_count, output_count);
#endif
  std::fill_n(output.get(), result_count, static_cast<OutputType>(0.0F));

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  RunTiling(tiling_data, workspace_size, block_dim);
#ifdef IL_POST_REDUCE
#ifdef IL_REDUCE_BEFORE_AXIS
  ASSERT_EQ(tiling_data.tiling_key, IL_TILING_KEY);
#elif defined(IL_EXPECT_ONLY_SIMT) || defined(IL_EXPECT_SIMT_ONLY)
  static_assert(IL_RANK == 4 && kAxis == 1);
  static_assert(IL_INPUT_PRE_TYPE == 3);
  static_assert(IL_REDUCE_WITH_A);
  ASSERT_EQ(tiling_data.tiling_key, IL_TILING_KEY);
#elif defined(IL_POST_REDUCE_SIMD)
  ASSERT_EQ(tiling_data.graph0_tiling_key, 0U);
  ASSERT_EQ(tiling_data.graph0_result0_g0_tiling_data.tiling_key, IL_TILING_KEY);
#else
#ifdef IL_EXPECT_SIMD_SELECTED
  ASSERT_EQ(tiling_data.graph0_tiling_key, 0U);
  ASSERT_EQ(tiling_data.graph0_result0_g0_tiling_data.tiling_key, IL_TILING_KEY);
#elif defined(IL_EXPECT_SIMT_SELECTED)
  ASSERT_EQ(tiling_data.graph0_tiling_key, 2U);
  ASSERT_EQ(tiling_data.graph0_result2_g0_tiling_data.tiling_key, IL_TILING_KEY);
#else
  ASSERT_TRUE(tiling_data.graph0_tiling_key == 0U || tiling_data.graph0_tiling_key == 2U);
  if (tiling_data.graph0_tiling_key == 0U) {
    ASSERT_EQ(tiling_data.graph0_result0_g0_tiling_data.tiling_key, IL_TILING_KEY);
  } else {
    const auto &simt_tiling_data = tiling_data.graph0_result2_g0_tiling_data;
    ASSERT_EQ(simt_tiling_data.tiling_key, IL_TILING_KEY);
    ASSERT_EQ(simt_tiling_data.block_dim, 48U);
#ifdef IL_REDUCE_LAST_AXIS
    ASSERT_EQ(simt_tiling_data.indirect_load_outerTb_size, 6U);
#else
    ASSERT_EQ(simt_tiling_data.indirect_load_outerTb_size, 2U);
#endif
  }
#endif
#endif
#elif !defined(IL_EXPECT_SIMT) || IL_EXPECT_SIMT
  ASSERT_EQ(tiling_data.graph0_tiling_key, IL_TILING_KEY);
#endif
  ASSERT_GT(tiling_data.block_dim, 0U);
  std::unique_ptr<uint8_t, decltype(gm_free)> workspace(nullptr, gm_free);
  if (workspace_size > 0U) {
    workspace.reset(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspace_size)));
    ASSERT_NE(workspace, nullptr);
    std::fill_n(workspace.get(), workspace_size, uint8_t{0});
  }
#ifdef IL_EXPECT_SIMT_MULTI_ROUND
  ASSERT_GT((static_cast<uint32_t>(output_count) + tiling_data.block_dim - 1U) / tiling_data.block_dim, 1024U);
#endif

#ifdef IL_POST_REDUCE_ADD
  indirect_load_test::RunKernel(indirect_load_store_test, tiling_data.block_dim, nullptr,
                                reinterpret_cast<uint8_t *>(&tiling_data), x.get(), index.get(), addend.get(),
                                output.get());
#else
  indirect_load_test::RunKernel(indirect_load_store_test, tiling_data.block_dim, workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling_data), x.get(), index.get(), output.get());
#endif
  for (int32_t i = 0; i < result_count; ++i) {
#if defined(IL_DATA_BF16) || defined(IL_DATA_INT16) || defined(IL_DATA_UINT32)
    EXPECT_EQ(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]))
        << "offset=" << i;
#else
    EXPECT_NEAR(static_cast<float>(output.get()[i]), static_cast<float>(expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
#endif
  }
#endif
}

#endif

#if defined(IL_CASE_BROADCAST)
#include <array>

#ifndef IL_COMPLEX_BROADCAST
#define IL_COMPLEX_BROADCAST 0
#endif
#ifndef IL_COMPLEX_SIMT
#define IL_COMPLEX_SIMT 0
#endif
#ifndef IL_COMPLEX_INPUT_BROADCAST
#define IL_COMPLEX_INPUT_BROADCAST 0
#endif
#ifndef IL_COMPLEX_INDEX_BROADCAST
#define IL_COMPLEX_INDEX_BROADCAST 0
#endif
#ifndef IL_BINARY_ELEMENT_KIND
#define IL_BINARY_ELEMENT_KIND 0
#endif
#ifndef IL_RETAIN_BROADCAST
#define IL_RETAIN_BROADCAST 0
#endif
#ifndef IL_BROADCAST_POST_REDUCE
#define IL_BROADCAST_POST_REDUCE 0
#endif
#ifndef IL_INDEX_MIXED_VIEW
#define IL_INDEX_MIXED_VIEW 0
#endif
#ifndef IL_INPUT_ABS_BEFORE_BROADCAST
#define IL_INPUT_ABS_BEFORE_BROADCAST 0
#endif
#ifndef IL_BOTH_TRANSPOSE
#define IL_BOTH_TRANSPOSE 0
#endif
#ifndef IL_INPUT_TRANSPOSE
#define IL_INPUT_TRANSPOSE 0
#endif
#ifndef IL_OUTPUT_S0
#define IL_OUTPUT_S0 4
#endif
#ifndef IL_OUTPUT_S1
#define IL_OUTPUT_S1 5
#endif
#ifndef IL_OUTPUT_S2
#define IL_OUTPUT_S2 4
#endif
#ifndef IL_OUTPUT_S3
#define IL_OUTPUT_S3 16
#endif

#if IL_AIC_REPRO
extern "C" __global__ __aicore__ void indirect_load_aic_repro(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                                              GM_ADDR workspace, GM_ADDR tiling);
#elif IL_OUTPUT_BROADCAST_ADD
extern "C" __global__ __aicore__ void indirect_load_broadcast_test(GM_ADDR x, GM_ADDR index, GM_ADDR bias, GM_ADDR y,
                                                                   GM_ADDR workspace, GM_ADDR tiling);
#else
extern "C" __global__ __aicore__ void indirect_load_broadcast_test(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                                                   GM_ADDR workspace, GM_ADDR tiling);
#endif

namespace {
#if IL_AIC_REPRO
using DataType = float;
using IndexType = int64_t;
constexpr int32_t kInputRows = 8192;
constexpr int32_t kRows = 512;
constexpr int32_t kColumns = 512;

void InitializeAicReproData(DataType *input, IndexType *index, DataType *expected) {
  for (int32_t row = 0; row < kInputRows; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      input[static_cast<int64_t>(row) * kColumns + column] =
          static_cast<DataType>((row % 97) * 0.25F + (column % 31) * 0.03125F);
    }
  }
  for (int32_t row = 0; row < kRows; ++row) {
    index[row] = static_cast<IndexType>((static_cast<int64_t>(row) * 97 + 13) % kInputRows);
    for (int32_t column = 0; column < kColumns; ++column) {
      const int64_t output_offset = static_cast<int64_t>(row) * kColumns + column;
      expected[output_offset] = input[index[row] * kColumns + column];
    }
  }
}
#else
using DataType = half;
#if IL_BOTH_TRANSPOSE
using IndexType = int32_t;
#else
using IndexType = int64_t;
#endif
constexpr std::array<int32_t, 4> kOutputShape = {IL_OUTPUT_S0, IL_OUTPUT_S1, IL_OUTPUT_S2, IL_OUTPUT_S3};
constexpr bool kInputBroadcast = IL_INPUT_BROADCAST;
constexpr bool kIndexBroadcast = IL_INDEX_BROADCAST;
constexpr bool kComplexBroadcast = IL_COMPLEX_BROADCAST;
constexpr bool kComplexSimt = IL_COMPLEX_SIMT;
constexpr bool kComplexInputBroadcast = IL_COMPLEX_INPUT_BROADCAST;
constexpr bool kComplexIndexBroadcast = IL_COMPLEX_INDEX_BROADCAST;
constexpr int32_t kBinaryElementKind = IL_BINARY_ELEMENT_KIND;
constexpr bool kRetainBroadcast = IL_RETAIN_BROADCAST;
constexpr bool kBroadcastPostReduce = IL_BROADCAST_POST_REDUCE;
constexpr bool kIndexMixedView = IL_INDEX_MIXED_VIEW;
constexpr bool kInputAbsBeforeBroadcast = IL_INPUT_ABS_BEFORE_BROADCAST;
constexpr bool kBothTranspose = IL_BOTH_TRANSPOSE;
constexpr bool kInputTranspose = IL_INPUT_TRANSPOSE;
constexpr uint32_t kBroadcastAxesMask = IL_BROADCAST_AXES_MASK;

constexpr std::array<int32_t, 4> MakeBroadcastSourceShape() {
  auto shape = kOutputShape;
  for (size_t dim = 0UL; dim < shape.size(); ++dim) {
    if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
      shape[dim] = 1;
    }
  }
  return shape;
}

constexpr std::array<int32_t, 4> kBroadcastSourceShape = MakeBroadcastSourceShape();
constexpr std::array<int32_t, 4> MakeTransposeSourceShape(const std::array<int32_t, 4> &shape) {
  return {shape[1], shape[0], shape[2], shape[3]};
}
constexpr std::array<int32_t, 4> kTransposeInputShape = MakeTransposeSourceShape(kOutputShape);
constexpr std::array<int32_t, 4> MakeGatherTransposeSourceShape(std::array<int32_t, 4> shape) {
  return {shape[2], shape[1], shape[0], shape[3]};
}
constexpr std::array<int32_t, 4> kInputShape =
    kBothTranspose    ? kTransposeInputShape
    : kInputTranspose ? MakeGatherTransposeSourceShape(kInputBroadcast ? kBroadcastSourceShape : kOutputShape)
                      : (kInputBroadcast && !kComplexSimt ? kBroadcastSourceShape : kOutputShape);
constexpr std::array<int32_t, 4> kIndexShape =
    kBothTranspose ? kTransposeInputShape : (kIndexBroadcast ? kBroadcastSourceShape : kOutputShape);
constexpr int32_t kInputElementCount = IL_HAS_INPUT_ELEMENT;
constexpr int32_t kIndexElementCount = IL_HAS_INDEX_ELEMENT;
constexpr bool kHasOutputRelu = IL_HAS_OUTPUT_RELU;

template <size_t N>
int32_t ElementCount(const std::array<int32_t, N> &shape) {
  int32_t count = 1;
  for (const int32_t dim : shape) {
    count *= dim;
  }
  return count;
}

int32_t ResultCount() {
  if constexpr (kBroadcastPostReduce) {
    return kOutputShape[0] * kOutputShape[1];
  }
  return ElementCount(kOutputShape);
}

int32_t DenseOffset(const std::array<int32_t, 4> &coordinate, const std::array<int32_t, 4> &shape) {
  int32_t offset = 0;
  for (size_t dim = 0UL; dim < coordinate.size(); ++dim) {
    offset = offset * shape[dim] + coordinate[dim];
  }
  return offset;
}

void InitializeData(DataType *x, IndexType *index, DataType *expected) {
  if constexpr (kBothTranspose) {
    for (int32_t a = 0; a < kInputShape[0]; ++a) {
      for (int32_t b = 0; b < kInputShape[1]; ++b) {
        for (int32_t c = 0; c < kInputShape[2]; ++c) {
          for (int32_t d = 0; d < kInputShape[3]; ++d) {
            const std::array<int32_t, 4> coordinate = {a, b, c, d};
            const int32_t value =
                kOutputShape[0] == kOutputShape[1] ? (a + b) * 7 + c * 3 + d : a * 11 + b * 3 + c * 5 + d;
            x[DenseOffset(coordinate, kInputShape)] = static_cast<DataType>(static_cast<float>(value) * 0.125F);
          }
        }
      }
    }
    for (int32_t a = 0; a < kIndexShape[0]; ++a) {
      for (int32_t b = 0; b < kIndexShape[1]; ++b) {
        for (int32_t c = 0; c < kIndexShape[2]; ++c) {
          for (int32_t d = 0; d < kIndexShape[3]; ++d) {
            const std::array<int32_t, 4> coordinate = {a, b, c, d};
            const int32_t value = kOutputShape[0] == kOutputShape[1] ? a + b + c + d : a * 5 + b * 2 + c + d;
            index[DenseOffset(coordinate, kIndexShape)] = static_cast<IndexType>(value % kInputShape[2]);
          }
        }
      }
    }
  } else {
    for (int32_t i = 0; i < ElementCount(kInputShape); ++i) {
      x[i] = static_cast<DataType>(static_cast<float>((i % 29) - 14) * 0.25F);
    }
    for (int32_t i = 0; i < ElementCount(kIndexShape); ++i) {
      const int32_t gathered_axis = (i * 3 + 1) % kInputShape[2];
      index[i] = static_cast<IndexType>(kIndexElementCount == 0 || i % 2 == 0 ? gathered_axis : -gathered_axis);
    }
  }
  for (int32_t i = 0; i < ElementCount(kOutputShape); ++i) {
    int32_t coordinate = i;
    const int32_t d = coordinate % kOutputShape[3];
    coordinate /= kOutputShape[3];
    const int32_t c = coordinate % kOutputShape[2];
    coordinate /= kOutputShape[2];
    const int32_t b = coordinate % kOutputShape[1];
    coordinate /= kOutputShape[1];
    const int32_t a = coordinate;
    std::array<int32_t, 4> index_coordinate =
        kBothTranspose ? std::array<int32_t, 4>{b, a, c, d} : std::array<int32_t, 4>{a, b, c, d};
    if (kIndexBroadcast) {
      for (size_t dim = 0UL; dim < index_coordinate.size(); ++dim) {
        if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
          index_coordinate[dim] = 0;
        }
      }
    }
    const int32_t index_offset = DenseOffset(index_coordinate, kIndexShape);
    int64_t gathered_index = static_cast<int64_t>(index[index_offset]);
    if constexpr (kComplexIndexBroadcast) {
      if constexpr (kBinaryElementKind == 1) {
        gathered_index = 0;
      } else if constexpr (kBinaryElementKind == 3) {
        gathered_index = std::max(gathered_index, int64_t{0});
      }
    }
    for (int32_t element = 0; element < kIndexElementCount; ++element) {
      gathered_index = std::abs(gathered_index);
    }
    const int32_t gathered_axis = static_cast<int32_t>(gathered_index);
    std::array<int32_t, 4> input_coordinate = kBothTranspose ? std::array<int32_t, 4>{b, a, gathered_axis, d}
                                                             : std::array<int32_t, 4>{a, b, gathered_axis, d};
    if (kInputBroadcast && !kComplexSimt) {
      for (size_t dim = 0UL; dim < input_coordinate.size(); ++dim) {
        if ((kBroadcastAxesMask & (1U << dim)) != 0U) {
          input_coordinate[dim] = 0;
        }
      }
    }
    if constexpr (kInputTranspose) {
      std::swap(input_coordinate[0], input_coordinate[2]);
    }
    const int32_t input_offset = DenseOffset(input_coordinate, kInputShape);
    float value = static_cast<float>(x[input_offset]);
    if constexpr (kInputAbsBeforeBroadcast) {
      value = std::abs(value);
    }
    if constexpr (kComplexInputBroadcast) {
      if constexpr (kBinaryElementKind == 1) {
        value = 0.0F;
      } else if constexpr (kBinaryElementKind == 3) {
        value = std::max(value, 0.0F);
      }
    }
    if (kRetainBroadcast) {
      value += 1.5F;
    } else if (kComplexBroadcast && !kComplexSimt) {
      value = value * 2.0F + 1.5F;
    } else if (kComplexSimt) {
      value += 1.5F;
    }
    for (int32_t element = 0; element < kInputElementCount; ++element) {
      value = std::abs(value);
    }
    if (kHasOutputRelu) {
      value = std::max(value, 0.0F);
    }
    if constexpr (kBroadcastPostReduce) {
      const int32_t result_offset = a * kOutputShape[1] + b;
      expected[result_offset] = static_cast<DataType>(static_cast<float>(expected[result_offset]) + value);
    } else {
      expected[i] = static_cast<DataType>(value);
    }
  }
}
#endif
}  // namespace

TEST(E2EIndirectLoadBroadcast, GeneratedKernelMatchesReference) {
#if IL_AIC_REPRO
  constexpr int64_t input_count = static_cast<int64_t>(kInputRows) * kColumns;
  constexpr int64_t index_count = kRows;
  constexpr int64_t output_count = static_cast<int64_t>(kRows) * kColumns;
  indirect_load_test::KernelData<DataType, IndexType> buffers(input_count, index_count, output_count);
  ASSERT_TRUE(buffers.IsValid());
  InitializeAicReproData(buffers.input.get(), buffers.index.get(), buffers.expected.data());
  std::fill_n(buffers.output.get(), output_count, 0.0F);
  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());

  indirect_load_test::RunKernel(indirect_load_aic_repro, tiling.data.block_dim, tiling.workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling.data), buffers.input.get(), buffers.index.get(),
                                buffers.output.get());
  for (int64_t i = 0; i < output_count; ++i) {
    EXPECT_FLOAT_EQ(buffers.output.get()[i], buffers.expected[static_cast<size_t>(i)]) << "offset=" << i;
  }
#else
  const int32_t input_count = ElementCount(kInputShape);
  const int32_t index_count = ElementCount(kIndexShape);
  const int32_t output_count = ResultCount();
  indirect_load_test::KernelData<DataType, IndexType> buffers(input_count, index_count, output_count);
  ASSERT_TRUE(buffers.IsValid());
  InitializeData(buffers.input.get(), buffers.index.get(), buffers.expected.data());
  std::fill_n(buffers.output.get(), output_count, static_cast<DataType>(0.0F));
  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());

#if IL_OUTPUT_BROADCAST_ADD
  auto bias = indirect_load_test::AllocGmBuffer<DataType>(kOutputShape[3]);
  ASSERT_NE(bias, nullptr);
  for (int32_t d = 0; d < kOutputShape[3]; ++d) {
    bias.get()[d] = static_cast<DataType>(static_cast<float>((d * 7) % 19 - 9) * 0.125F);
  }
  for (int32_t i = 0; i < output_count; ++i) {
    buffers.expected[i] = static_cast<DataType>(static_cast<float>(buffers.expected[i]) +
                                                static_cast<float>(bias.get()[i % kOutputShape[3]]));
  }
  indirect_load_test::RunKernel(indirect_load_broadcast_test, tiling.data.block_dim, tiling.workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling.data), buffers.input.get(), buffers.index.get(),
                                bias.get(), buffers.output.get());
#else
  indirect_load_test::RunKernel(indirect_load_broadcast_test, tiling.data.block_dim, tiling.workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling.data), buffers.input.get(), buffers.index.get(),
                                buffers.output.get());
#endif
  for (int32_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(static_cast<float>(buffers.output.get()[i]),
                static_cast<float>(buffers.expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
  }
#endif
}

#endif

#if defined(IL_USER_FANOUT)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 2;
  constexpr int32_t kDim = 16;
  constexpr int32_t kTableRows = 32;
  auto indices = indirect_load_test::AllocGmBuffer<int64_t>(kRows);
  auto embedding = indirect_load_test::AllocGmBuffer<bfloat16_t>(kTableRows * kDim);
  auto weight = indirect_load_test::AllocGmBuffer<bfloat16_t>(kRows * kDim);
  auto output0 = indirect_load_test::AllocGmBuffer<bfloat16_t>(kRows * kDim);
#if defined(IL_USER_FANOUT_REDUCE)
  auto output1 = indirect_load_test::AllocGmBuffer<bfloat16_t>(kRows);
#else
  auto output1 = indirect_load_test::AllocGmBuffer<bfloat16_t>(kRows * kDim);
#endif
  ASSERT_TRUE(indices && embedding && weight && output0 && output1);
  for (int32_t row = 0; row < kRows; ++row) {
    indices.get()[row] = row + 1;
  }
  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      embedding.get()[row * kDim + col] = static_cast<bfloat16_t>((row + 1) * 0.01F + col * 0.001F);
    }
  }
  std::fill_n(weight.get(), kRows * kDim, static_cast<bfloat16_t>(0.0F));
  std::fill_n(output0.get(), kRows * kDim, static_cast<bfloat16_t>(0.0F));
#if defined(IL_USER_FANOUT_REDUCE)
  std::fill_n(output1.get(), kRows, static_cast<bfloat16_t>(0.0F));
#else
  std::fill_n(output1.get(), kRows * kDim, static_cast<bfloat16_t>(0.0F));
#endif

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  indirect_load_test::RunKernel(user_fanout, tiling.block_dim, reinterpret_cast<uint8_t *>(tiling.workspace.get()),
                                reinterpret_cast<uint8_t *>(&tiling.data), indices.get(), embedding.get(), weight.get(),
                                output0.get(), output1.get());

  for (int32_t row = 0; row < kRows; ++row) {
#if defined(IL_USER_FANOUT_REDUCE)
    float expected_reduce = 0.0F;
#endif
    for (int32_t col = 0; col < kDim; ++col) {
      const float value = static_cast<float>(embedding.get()[indices.get()[row] * kDim + col]);
#if defined(IL_USER_FANOUT_POST)
      const float source = std::fabs(value);
      EXPECT_NEAR(static_cast<float>(output0.get()[row * kDim + col]), std::exp(source), 0.125F)
          << "output0 row=" << row << ", col=" << col;
#if defined(IL_USER_FANOUT_REDUCE)
      expected_reduce += source * source;
#else
      EXPECT_NEAR(static_cast<float>(output1.get()[row * kDim + col]), std::fabs(source), 0.125F)
          << "output1 row=" << row << ", col=" << col;
#endif
#else
      const float source = value;
      EXPECT_NEAR(static_cast<float>(output0.get()[row * kDim + col]), std::exp(source), 0.125F)
          << "output0 row=" << row << ", col=" << col;
#if defined(IL_USER_FANOUT_REDUCE)
      expected_reduce += source * source;
#else
      EXPECT_NEAR(static_cast<float>(output1.get()[row * kDim + col]), std::fabs(source), 0.125F)
          << "output1 row=" << row << ", col=" << col;
#endif
#endif
    }
#if defined(IL_USER_FANOUT_REDUCE)
    EXPECT_NEAR(static_cast<float>(output1.get()[row]), expected_reduce, 0.5F) << "output1 row=" << row;
#endif
  }
}

#elif defined(IL_USER_FANOUT_SIDE_INPUT)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 2;
  constexpr int32_t kDim = 16;
  constexpr int32_t kTableRows = 32;
  auto indices = indirect_load_test::AllocGmBuffer<int64_t>(kRows);
  auto embedding = indirect_load_test::AllocGmBuffer<bfloat16_t>(kTableRows * kDim);
  auto side_input = indirect_load_test::AllocGmBuffer<bfloat16_t>(kDim);
  auto output0 = indirect_load_test::AllocGmBuffer<bfloat16_t>(kRows * kDim);
  auto output1 = indirect_load_test::AllocGmBuffer<bfloat16_t>(kRows * kDim);
  ASSERT_TRUE(indices && embedding && side_input && output0 && output1);
  for (int32_t row = 0; row < kRows; ++row) {
    indices.get()[row] = row + 1;
  }
  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      embedding.get()[row * kDim + col] = static_cast<bfloat16_t>((row + 1) * 0.01F + col * 0.001F);
    }
  }
  for (int32_t col = 0; col < kDim; ++col) {
    side_input.get()[col] = static_cast<bfloat16_t>((col - 3) * 0.02F);
  }
  std::fill_n(output0.get(), kRows * kDim, static_cast<bfloat16_t>(0.0F));
  std::fill_n(output1.get(), kRows * kDim, static_cast<bfloat16_t>(0.0F));

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  indirect_load_test::RunKernel(user_fanout_side_input, tiling.block_dim,
                                reinterpret_cast<uint8_t *>(tiling.workspace.get()),
                                reinterpret_cast<uint8_t *>(&tiling.data), indices.get(), embedding.get(),
                                side_input.get(), output0.get(), output1.get());

  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      const float value = static_cast<float>(embedding.get()[indices.get()[row] * kDim + col]);
      const float bias = static_cast<float>(side_input.get()[col]);
      EXPECT_NEAR(static_cast<float>(output0.get()[row * kDim + col]), value, 0.125F)
          << "output0 row=" << row << ", col=" << col;
      EXPECT_NEAR(static_cast<float>(output1.get()[row * kDim + col]), value + bias, 0.125F)
          << "output1 row=" << row << ", col=" << col;
    }
  }
}

#elif defined(IL_USER_SIDE_INPUT_FANOUT)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kElements = 64;
  constexpr int32_t kTableRows = 32;
  auto input0 = indirect_load_test::AllocGmBuffer<float>(kTableRows * kElements);
  auto input1 = indirect_load_test::AllocGmBuffer<int64_t>(1);
  auto input2 = indirect_load_test::AllocGmBuffer<int64_t>(1);
  auto input3 = indirect_load_test::AllocGmBuffer<int64_t>(2);
  auto input4 = indirect_load_test::AllocGmBuffer<int64_t>(1);
  auto input5 = indirect_load_test::AllocGmBuffer<float>(kElements);
  auto output0 = indirect_load_test::AllocGmBuffer<float>(kElements);
  auto output1 = indirect_load_test::AllocGmBuffer<float>(kElements);
  ASSERT_TRUE(input0 && input1 && input2 && input3 && input4 && input5 && output0 && output1);

  input1.get()[0] = 0;
  input2.get()[0] = 16;
  input3.get()[0] = 0;
  input3.get()[1] = 1;
  input4.get()[0] = 0;
  for (int32_t row = 0; row < kTableRows; ++row)
    for (int32_t col = 0; col < kElements; ++col)
      input0.get()[row * kElements + col] = static_cast<float>(row * 0.25F + col * 0.01F);
  for (int32_t col = 0; col < kElements; ++col) input5.get()[col] = static_cast<float>(10.0F + col * 0.1F);
  std::fill_n(output0.get(), kElements, 0.0F);
  std::fill_n(output1.get(), kElements, 0.0F);

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  indirect_load_test::RunKernel(user_side_input_fanout, tiling.block_dim,
                                reinterpret_cast<uint8_t *>(tiling.workspace.get()),
                                reinterpret_cast<uint8_t *>(&tiling.data), input0.get(), input1.get(), input2.get(),
                                input3.get(), input4.get(), input5.get(), output0.get(), output1.get());

  for (int32_t col = 0; col < kElements; ++col) {
    const float expected0 = input0.get()[16];
    EXPECT_NEAR(output0.get()[col], expected0, 1.0e-5F) << "output0 col=" << col;
    EXPECT_NEAR(output1.get()[col], input5.get()[col] + expected0, 1.0e-5F) << "output1 col=" << col;
  }
}

#elif defined(IL_USER_EMBEDDING_EXP_ABS_ADD)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 8;
  constexpr int32_t kDim = 16;
  constexpr int32_t kTableRows = 100;
  constexpr int32_t kTableElements = kTableRows * kDim;
  auto *indices = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kRows));
  auto *embedding = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kTableElements));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kRows * kDim));
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(embedding, nullptr);
  ASSERT_NE(output, nullptr);

  for (int32_t row = 0; row < kRows; ++row) {
    indices[row] = static_cast<int64_t>(row + 1);
  }
  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      embedding[row * kDim + col] = static_cast<float>(row - 40) * 0.01F + static_cast<float>(col) * 0.001F;
    }
  }
  std::fill_n(output, kRows * kDim, 0.0F);

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);

  indirect_load_test::RunKernel(user_embedding_exp_abs_add, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), indices, embedding, output);

  for (int32_t row = 0; row < kRows; ++row) {
    const int64_t index = indices[row];
    for (int32_t col = 0; col < kDim; ++col) {
      const float value = embedding[index * kDim + col];
      const float expected = std::exp(value) + std::fabs(value);
      EXPECT_NEAR(output[row * kDim + col], expected, 1.0e-4F) << "row=" << row << ", col=" << col;
    }
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(indices);
  AscendC::GmFree(embedding);
  AscendC::GmFree(output);
}

#elif defined(IL_USER_EMBEDDING_SUM)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 4;
#if defined(IL_USER_EMBEDDING_SUM_RANK2)
  constexpr int32_t kLookups = 1;
  constexpr int32_t kDim = 16;
  constexpr int32_t kTableRows = 100;
  constexpr int32_t kIndexStride = 1;
#else
  constexpr int32_t kLookups = 4;
  constexpr int32_t kDim = 16;
  constexpr int32_t kTableRows = 100;
  constexpr int32_t kIndexStride = 8;
#endif
  const int32_t kIndexStorage = (kRows - 1) * kIndexStride + kLookups;
  auto *indices = static_cast<int32_t *>(AscendC::GmAlloc(sizeof(int32_t) * kIndexStorage));
  auto *table = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kTableRows * kDim));
  const int32_t output_elements =
#if defined(IL_USER_EMBEDDING_SUM_RANK2)
      kRows;
#else
      kRows * kDim;
#endif
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * output_elements));
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(table, nullptr);
  ASSERT_NE(output, nullptr);
  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      table[row * kDim + col] = static_cast<float>(row * kDim + col) * 0.01F;
    }
  }
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t lookup = 0; lookup < kLookups; ++lookup) {
      indices[row * kIndexStride + lookup] = (row * kLookups + lookup + 1) % kTableRows;
    }
  }
  std::fill_n(output, output_elements, 0.0F);

  AutofuseTilingData tiling_data{};
#if !defined(IL_USER_EMBEDDING_SUM_RANK2)
#if defined(IL_USER_EMBEDDING_SUM_SIMD)
  tiling_data.graph0_result0_g0_tiling_data.set_ks0(kRows);
  tiling_data.graph0_result0_g0_tiling_data.set_s44(kIndexStride);
#else
  tiling_data.set_ks0(kRows);
  tiling_data.set_s44(kIndexStride);
#endif
#endif
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(user_embedding_sum, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), table, indices, output);
#if defined(IL_USER_EMBEDDING_SUM_RANK2)
  for (int32_t row = 0; row < kRows; ++row) {
    float expected = 0.0F;
    const int64_t index = indices[row * kIndexStride];
    for (int32_t col = 0; col < kDim; ++col) {
      expected += table[index * kDim + col];
    }
    EXPECT_NEAR(output[row], expected, 0.0625F) << "row=" << row;
  }
#else
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      float expected = 0.0F;
      for (int32_t lookup = 0; lookup < kLookups; ++lookup) {
        expected += table[indices[row * kIndexStride + lookup] * kDim + col];
      }
      EXPECT_NEAR(output[row * kDim + col], expected, 0.0625F) << "row=" << row << ", col=" << col;
    }
  }
#endif
  if (workspace != nullptr) AscendC::GmFree(workspace);
  AscendC::GmFree(indices);
  AscendC::GmFree(table);
  AscendC::GmFree(output);
}

#elif defined(IL_USER_GATHER_SUM_TRANSPOSE)
namespace {
void FillGatherSumTiling(AutofuseTilingData &tiling_data, uint32_t ks0, uint32_t ks1, uint32_t ks2, uint32_t ks3) {
  tiling_data.set_block_dim(indirect_load_test::kUserAivNum);
  tiling_data.set_ub_size(indirect_load_test::kUserUbSize);
  tiling_data.graph0_result0_g0_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result0_g0_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result0_g0_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result0_g0_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result1_g0_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result1_g0_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result1_g0_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result1_g0_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result1_g1_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result1_g1_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result1_g1_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result1_g2_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result1_g2_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result1_g2_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result1_g2_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result1_g3_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result1_g3_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result1_g3_tiling_data.set_ks2(ks2);
}
}  // namespace

TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kKs0 = 2;
  constexpr int32_t kKs1 = 4;
  constexpr int32_t kKs2 = 3;
  constexpr int32_t kKs3 = 6;
  constexpr int64_t kTableCount = static_cast<int64_t>(kKs0) * kKs3 * kKs1 * kKs2;
  constexpr int64_t kIndexCount = static_cast<int64_t>(kKs0) * kKs1 * kKs1 * kKs2;
  constexpr int64_t kOutputCount = static_cast<int64_t>(kKs0) * kKs1 * kKs2;

  auto table = indirect_load_test::AllocGmBuffer<float>(kTableCount);
  auto index = indirect_load_test::AllocGmBuffer<int64_t>(kIndexCount);
  auto output = indirect_load_test::AllocGmBuffer<float>(kOutputCount);
  ASSERT_TRUE(table && index && output);

  for (int64_t i = 0; i < kTableCount; ++i) {
    table.get()[i] = static_cast<float>(i) * 0.01F + 1.0F;
  }
  for (int32_t i0 = 0; i0 < kKs0; ++i0) {
    for (int32_t i1 = 0; i1 < kKs1; ++i1) {
      for (int32_t i2 = 0; i2 < kKs1; ++i2) {
        for (int32_t i3 = 0; i3 < kKs2; ++i3) {
          const int64_t offset = ((static_cast<int64_t>(i0) * kKs1 + i1) * kKs1 + i2) * kKs2 + i3;
          index.get()[offset] = (i0 + i1 + i2 + i3) % kKs3;
        }
      }
    }
  }
  std::fill_n(output.get(), kOutputCount, 0.0F);

  indirect_load_test::UserGraphTiling tiling;
  tiling.Tile(FillGatherSumTiling, kKs0, kKs1, kKs2, kKs3);
  indirect_load_test::RunKernel(autofused_gather_sum_transpose_9d393a67d2eb1a45f6dd4ea95c860eac, tiling.block_dim,
                                tiling.workspace.get(), reinterpret_cast<uint8_t *>(&tiling.data), table.get(),
                                index.get(), output.get());

  for (int32_t i0 = 0; i0 < kKs0; ++i0) {
    for (int32_t i1 = 0; i1 < kKs1; ++i1) {
      for (int32_t i3 = 0; i3 < kKs2; ++i3) {
        float expected = 0.0F;
        for (int32_t i2 = 0; i2 < kKs1; ++i2) {
          const int64_t index_offset = ((static_cast<int64_t>(i0) * kKs1 + i1) * kKs1 + i2) * kKs2 + i3;
          const int64_t selected = index.get()[index_offset];
          const int64_t table_offset =
              (static_cast<int64_t>(i0) * kKs3 + selected) * kKs1 * kKs2 + static_cast<int64_t>(i2) * kKs2 + i3;
          expected += table.get()[table_offset];
        }
        const int32_t output_offset = (i0 * kKs1 + i1) * kKs2 + i3;
        EXPECT_NEAR(output.get()[output_offset], expected, 1.0e-4F) << "i0=" << i0 << ", i1=" << i1 << ", i3=" << i3;
      }
    }
  }
  // Real ATen shapes (case 0193) must also tile successfully via the SIMT fallback.
  EXPECT_TRUE(indirect_load_test::TilingSucceeds(FillGatherSumTiling, 32U, 8U, 64U, 16U));
}

#elif defined(IL_USER_ABS_EMBEDDING_SUM)
namespace {
void FillAbsEmbTiling(AutofuseTilingData &tiling_data, uint32_t ks0, uint32_t ks1, uint32_t ks2, uint32_t ks3) {
  tiling_data.set_block_dim(indirect_load_test::kUserAivNum);
  tiling_data.set_ub_size(indirect_load_test::kUserUbSize);
  tiling_data.graph0_result0_g0_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result0_g0_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result0_g0_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result0_g0_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result1_g0_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result1_g0_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result1_g0_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result1_g0_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result2_g0_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result2_g0_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result2_g0_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result2_g0_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result3_g0_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result3_g0_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result3_g1_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result3_g1_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result3_g1_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result3_g2_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result3_g2_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result3_g2_tiling_data.set_ks2(ks2);
  tiling_data.graph0_result3_g2_tiling_data.set_ks3(ks3);
  tiling_data.graph0_result3_g3_tiling_data.set_ks0(ks0);
  tiling_data.graph0_result3_g3_tiling_data.set_ks1(ks1);
  tiling_data.graph0_result3_g3_tiling_data.set_ks2(ks2);
}
}  // namespace

TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 6;
  constexpr int32_t kDim = 8;
  constexpr int32_t kTableRows = 16;
  constexpr int64_t kTableCount = static_cast<int64_t>(kTableRows) * kDim;

  auto table = indirect_load_test::AllocGmBuffer<float>(kTableCount);
  auto index = indirect_load_test::AllocGmBuffer<int64_t>(kRows);
  auto output = indirect_load_test::AllocGmBuffer<float>(kRows);
  ASSERT_TRUE(table && index && output);

  for (int32_t r = 0; r < kTableRows; ++r) {
    for (int32_t c = 0; c < kDim; ++c) {
      table.get()[static_cast<int64_t>(r) * kDim + c] = static_cast<float>((r - 8) * 0.25F + c * 0.01F);
    }
  }
  for (int32_t r = 0; r < kRows; ++r) {
    index.get()[r] = (r * 5 + 1) % kTableRows;
  }
  std::fill_n(output.get(), kRows, 0.0F);

  constexpr int32_t kKs0 = 2;
  constexpr int32_t kKs1 = 3;
  indirect_load_test::UserGraphTiling tiling;
  tiling.Tile(FillAbsEmbTiling, kKs0, kKs1, kDim, kTableRows);
  indirect_load_test::RunKernel(autofused_abs_embedding_sum_1ee65c13970c2252fddcc2b670e5212d, tiling.block_dim,
                                tiling.workspace.get(), reinterpret_cast<uint8_t *>(&tiling.data), table.get(),
                                index.get(), output.get());

  for (int32_t r = 0; r < kRows; ++r) {
    float expected = 0.0F;
    for (int32_t c = 0; c < kDim; ++c) {
      expected += std::fabs(table.get()[static_cast<int64_t>(index.get()[r]) * kDim + c]);
    }
    EXPECT_NEAR(output.get()[r], expected, 1.0e-3F) << "row=" << r;
  }
  // Real ATen shapes (case 0087) must also tile successfully via a feasible fallback.
  EXPECT_TRUE(indirect_load_test::TilingSucceeds(FillAbsEmbTiling, 400U, 230U, 160U, 500U));
}

#elif defined(IL_USER_INT64_GATHER_DENSE) || defined(IL_USER_INT64_GATHER_DENSE_SIMD) || \
    defined(IL_USER_INT64_GATHER_DENSE_SIMT)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kD0 = 4;
  constexpr int32_t kTableAxis1 = 16;
  constexpr int32_t kIndexAxis1 = 8;
  constexpr int32_t kD2 = 6;
  constexpr int64_t kTableCount = static_cast<int64_t>(kD0) * kTableAxis1 * kD2;
  constexpr int64_t kIndexCount = static_cast<int64_t>(kD0) * kIndexAxis1 * kD2;
  auto table = indirect_load_test::AllocGmBuffer<int64_t>(kTableCount);
  auto index = indirect_load_test::AllocGmBuffer<int64_t>(kIndexCount);
  auto output = indirect_load_test::AllocGmBuffer<int64_t>(kIndexCount);
  ASSERT_TRUE(table && index && output);

  for (int64_t i = 0; i < kTableCount; ++i) {
    table.get()[i] = static_cast<int64_t>(i) * 7 - 3;
  }
  for (int32_t i0 = 0; i0 < kD0; ++i0) {
    for (int32_t i1 = 0; i1 < kIndexAxis1; ++i1) {
      for (int32_t i2 = 0; i2 < kD2; ++i2) {
        const int64_t offset = (static_cast<int64_t>(i0) * kIndexAxis1 + i1) * kD2 + i2;
        index.get()[offset] = (i0 * 5 + i1 * 3 + i2) % kTableAxis1;
      }
    }
  }
  std::fill_n(output.get(), kIndexCount, static_cast<int64_t>(0));

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  indirect_load_test::RunKernel(user_int64_gather_dense, tiling.block_dim, tiling.workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling.data), table.get(), index.get(), output.get());

  for (int32_t i0 = 0; i0 < kD0; ++i0) {
    for (int32_t i1 = 0; i1 < kIndexAxis1; ++i1) {
      for (int32_t i2 = 0; i2 < kD2; ++i2) {
        const int64_t idx_offset = (static_cast<int64_t>(i0) * kIndexAxis1 + i1) * kD2 + i2;
        const int64_t selected = index.get()[idx_offset];
        const int64_t table_offset = (static_cast<int64_t>(i0) * kTableAxis1 + selected) * kD2 + i2;
        EXPECT_EQ(output.get()[idx_offset], table.get()[table_offset]) << "i0=" << i0 << ", i1=" << i1 << ", i2=" << i2;
      }
    }
  }
}

#elif defined(IL_USER_DTYPE_INT64) || defined(IL_USER_DTYPE_UINT64) || defined(IL_USER_DTYPE_INT8) || \
    defined(IL_USER_DTYPE_UINT8) || defined(IL_USER_DTYPE_BOOL)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
#if defined(IL_USER_DTYPE_INT64)
  using ValueT = int64_t;
  constexpr bool kIsBool = false;
#elif defined(IL_USER_DTYPE_UINT64)
  using ValueT = uint64_t;
  constexpr bool kIsBool = false;
#elif defined(IL_USER_DTYPE_INT8)
  using ValueT = int8_t;
  constexpr bool kIsBool = false;
#elif defined(IL_USER_DTYPE_UINT8)
  using ValueT = uint8_t;
  constexpr bool kIsBool = false;
#else
  using ValueT = bool;
  constexpr bool kIsBool = true;
#endif
  constexpr int32_t kRows = 8;
  constexpr int32_t kDim = 16;
  constexpr int32_t kTableRows = 32;
  auto table = indirect_load_test::AllocGmBuffer<ValueT>(kTableRows * kDim);
  auto index = indirect_load_test::AllocGmBuffer<int64_t>(kRows);
  auto output = indirect_load_test::AllocGmBuffer<ValueT>(kRows * kDim);
  ASSERT_TRUE(table && index && output);

  for (int32_t r = 0; r < kTableRows; ++r) {
    for (int32_t c = 0; c < kDim; ++c) {
      table.get()[static_cast<int64_t>(r) * kDim + c] =
          kIsBool ? static_cast<ValueT>(((r + c) % 2) != 0) : static_cast<ValueT>((r + c * 3) % 101);
    }
  }
  for (int32_t r = 0; r < kRows; ++r) {
    index.get()[r] = (r * 3 + 1) % kTableRows;
  }
  std::fill_n(output.get(), kRows * kDim, static_cast<ValueT>(0));

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  indirect_load_test::RunKernel(USER_DTYPE_KERNEL, tiling.block_dim, tiling.workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling.data), table.get(), index.get(), output.get());

  for (int32_t r = 0; r < kRows; ++r) {
    for (int32_t c = 0; c < kDim; ++c) {
      const ValueT expected = table.get()[static_cast<int64_t>(index.get()[r]) * kDim + c];
      EXPECT_EQ(output.get()[static_cast<int64_t>(r) * kDim + c], expected) << "row=" << r << ", col=" << c;
    }
  }
}

#elif defined(IL_USER_LAYERNORM)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 8;
  constexpr int32_t kDim = 256;
  constexpr int32_t kTableRows = 1024;
  auto *indices = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kRows));
  auto *embedding = static_cast<bfloat16_t *>(AscendC::GmAlloc(sizeof(bfloat16_t) * kTableRows * kDim));
  auto *weight = static_cast<bfloat16_t *>(AscendC::GmAlloc(sizeof(bfloat16_t) * kRows * kDim));
  auto *raw_output = static_cast<bfloat16_t *>(AscendC::GmAlloc(sizeof(bfloat16_t) * kRows * kDim));
  auto *square_output = static_cast<bfloat16_t *>(AscendC::GmAlloc(sizeof(bfloat16_t) * kRows));
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(embedding, nullptr);
  ASSERT_NE(weight, nullptr);
  ASSERT_NE(raw_output, nullptr);
  ASSERT_NE(square_output, nullptr);
  for (int32_t row = 0; row < kRows; ++row) indices[row] = row % kTableRows;
  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      embedding[row * kDim + col] = static_cast<bfloat16_t>((row + 1) * 0.01F + col * 0.001F);
    }
  }
  std::fill_n(weight, kRows * kDim, static_cast<bfloat16_t>(0.0F));
  std::fill_n(raw_output, kRows * kDim, static_cast<bfloat16_t>(0.0F));
  std::fill_n(square_output, kRows, static_cast<bfloat16_t>(0.0F));
  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(user_layernorm, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), indices, embedding, weight, raw_output,
                                square_output);
  for (int32_t row = 0; row < kRows; ++row) {
    float expected_square = 0.0F;
    for (int32_t col = 0; col < kDim; ++col) {
      const float expected_raw = static_cast<float>(embedding[indices[row] * kDim + col]);
      expected_square += expected_raw * expected_raw;
      EXPECT_NEAR(static_cast<float>(raw_output[row * kDim + col]), expected_raw, 0.0625F)
          << "raw row=" << row << ", col=" << col;
    }
    const float expected_square_bf16 = static_cast<float>(static_cast<bfloat16_t>(expected_square));
    EXPECT_NEAR(static_cast<float>(square_output[row]), expected_square_bf16, 0.5F) << "square row=" << row;
  }
  if (workspace != nullptr) AscendC::GmFree(workspace);
  AscendC::GmFree(indices);
  AscendC::GmFree(embedding);
  AscendC::GmFree(weight);
  AscendC::GmFree(raw_output);
  AscendC::GmFree(square_output);
}

#elif defined(IL_USER_SOFTMAX)
// gather(+bias+bmm) 后接尾轴 softmax 的融合 kernel：与 CPU 参考实现逐元素对比。
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 16;
  constexpr int32_t kDim = 64;
  constexpr int32_t kTableRows = 64;
  auto *indices = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kRows));
  auto *embedding = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kTableRows * kDim));
  auto *bias = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kRows * kDim));
  auto *bmm = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kRows * kDim));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kRows * kDim));
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(embedding, nullptr);
  ASSERT_NE(bias, nullptr);
  ASSERT_NE(bmm, nullptr);
  ASSERT_NE(output, nullptr);
  for (int32_t row = 0; row < kRows; ++row) indices[row] = (row * 7L) % kTableRows;
  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      embedding[row * kDim + col] = (row + 1) * 0.01F + col * 0.001F;
    }
  }
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t col = 0; col < kDim; ++col) {
      bias[row * kDim + col] = row * 0.002F + col * 0.0001F;
      bmm[row * kDim + col] = (row + col) * 0.003F - 0.5F;
    }
  }
  std::fill_n(output, kRows * kDim, 0.0F);
  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(user_softmax, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), indices, embedding, bias, bmm, output);
  for (int32_t row = 0; row < kRows; ++row) {
    float expected_max = -std::numeric_limits<float>::infinity();
    float expected_sum = 0.0F;
    for (int32_t col = 0; col < kDim; ++col) {
      const float value = embedding[indices[row] * kDim + col] + bias[row * kDim + col] + bmm[row * kDim + col];
      expected_max = std::max(expected_max, value);
    }
    for (int32_t col = 0; col < kDim; ++col) {
      const float value = embedding[indices[row] * kDim + col] + bias[row * kDim + col] + bmm[row * kDim + col];
      expected_sum += std::exp(value - expected_max);
    }
    for (int32_t col = 0; col < kDim; ++col) {
      const float value = embedding[indices[row] * kDim + col] + bias[row * kDim + col] + bmm[row * kDim + col];
      const float expected = std::exp(value - expected_max) / expected_sum;
      EXPECT_NEAR(output[row * kDim + col], expected, 1e-4F) << "row=" << row << ", col=" << col;
    }
  }
  if (workspace != nullptr) AscendC::GmFree(workspace);
  AscendC::GmFree(indices);
  AscendC::GmFree(embedding);
  AscendC::GmFree(bias);
  AscendC::GmFree(bmm);
  AscendC::GmFree(output);
}

#elif defined(IL_DUAL_IL_GATHER)
TEST(UserGraphConstruction, GeneratedKernelMatchesReference) {
  constexpr int32_t kRows = 1024 * 1025;
  constexpr int32_t kInputWidth = 10;
  constexpr int32_t kOutputWidth = 5;
  auto *input0 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kRows * kInputWidth));
  auto *input1 = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kRows * kInputWidth));
  auto *indices = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kRows * kOutputWidth));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kRows * kOutputWidth));
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(output, nullptr);
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t col = 0; col < kInputWidth; ++col) {
      input0[row * kInputWidth + col] = row * 0.001F + col;
      input1[row * kInputWidth + col] = row * 0.002F - col;
    }
    for (int32_t col = 0; col < kOutputWidth; ++col) indices[row * kOutputWidth + col] = col;
  }
  std::fill_n(output, kRows * kOutputWidth, 0.0F);
  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(user_add_gather, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), input0, input1, indices, output);
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t col = 0; col < kOutputWidth; ++col) {
      EXPECT_FLOAT_EQ(output[row * kOutputWidth + col],
                      input0[row * kInputWidth + col] + input1[row * kInputWidth + col])
          << "row=" << row << ", col=" << col;
    }
  }
  if (workspace != nullptr) AscendC::GmFree(workspace);
  AscendC::GmFree(input0);
  AscendC::GmFree(input1);
  AscendC::GmFree(indices);
  AscendC::GmFree(output);
}

#elif defined(IL_CASE_BROADCAST_WHERE)

extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint64_t *, uint32_t *, uint32_t, uint32_t);

#ifndef IL_ADD_IL_REDUCE
#if defined(IL_USER_POSITION_BIAS)
extern "C" __global__ __aicore__ void user_position_bias(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR output,
                                                         GM_ADDR workspace, GM_ADDR tiling);
#elif defined(IL_USER_POSITION_BIAS_EXP_SUM)
extern "C" __global__ __aicore__ void user_position_bias_exp_sum(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                                 GM_ADDR input3, GM_ADDR output, GM_ADDR workspace,
                                                                 GM_ADDR tiling);
#elif defined(IL_USER_MASKED_EMBEDDING_MINIMAL) || defined(IL_USER_MASKED_EMBEDDING_SUM_FULL)
extern "C" __global__ __aicore__ void user_masked_embedding_sum(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                                GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
#elif defined(IL_GRAPH_HINT_EMBEDDING_SLICE)
extern "C" __global__ __aicore__ void autofused_to_copy_embedding_slice_161da55df6362966862381c5d053b7e7(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
#elif defined(IL_GRAPH_HINT_SIMD_REPRO)
extern "C" __global__ __aicore__ void indirect_load_graph_hint_simd_repro(GM_ADDR input0, GM_ADDR input1,
                                                                          GM_ADDR output, GM_ADDR workspace,
                                                                          GM_ADDR tiling);
#elif defined(IL_GRAPH_HINT_REDUCE)
extern "C" __global__ __aicore__ void indirect_load_graph_hint_reduce_simt_test(GM_ADDR input0, GM_ADDR input1,
                                                                                GM_ADDR input2, GM_ADDR output,
                                                                                GM_ADDR workspace, GM_ADDR tiling);
#elif defined(IL_EMBEDDING_REDUCE)
extern "C" __global__ __aicore__ void indirect_load_embedding_reduce_simt_test(GM_ADDR input0, GM_ADDR input1,
                                                                               GM_ADDR input2, GM_ADDR output,
                                                                               GM_ADDR workspace, GM_ADDR tiling);
#else
extern "C" __global__ __aicore__ void indirect_load_broadcast_index_where_simt_test(GM_ADDR input0, GM_ADDR input1,
                                                                                    GM_ADDR input2, GM_ADDR output,
                                                                                    GM_ADDR workspace, GM_ADDR tiling);
#endif
#else
extern "C" __global__ __aicore__ void indirect_load_add_il_reduce_test(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2,
                                                                       GM_ADDR output, GM_ADDR workspace,
                                                                       GM_ADDR tiling);
#endif

namespace {
#ifndef IL_ADD_IL_REDUCE
#if defined(IL_USER_POSITION_BIAS)
constexpr int32_t kUserPositionBiasRows = 8;
constexpr int32_t kUserPositionBiasDim = 256;
constexpr int32_t kUserPositionBiasLookups = 256;
constexpr int32_t kUserPositionBiasTableRows = 32;

TEST(E2EUserPositionBias, GeneratedKernelMatchesReference) {
  const int64_t output_count =
      static_cast<int64_t>(kUserPositionBiasRows) * kUserPositionBiasDim * kUserPositionBiasLookups;
  auto table = indirect_load_test::AllocGmBuffer<float>(static_cast<int64_t>(kUserPositionBiasTableRows) *
                                                        kUserPositionBiasRows);
  auto side = indirect_load_test::AllocGmBuffer<float>(kUserPositionBiasLookups);
  auto input = indirect_load_test::AllocGmBuffer<float>(output_count);
  auto output = indirect_load_test::AllocGmBuffer<float>(output_count);
  ASSERT_TRUE(table && side && input && output);
  for (int32_t b = 0; b < kUserPositionBiasTableRows; ++b) {
    for (int32_t r = 0; r < kUserPositionBiasRows; ++r)
      table.get()[b * kUserPositionBiasRows + r] = b * 0.1F + r * 0.01F;
  }
  for (int32_t p = 0; p < kUserPositionBiasLookups; ++p) side.get()[p] = p * 0.001F;
  for (int64_t i = 0; i < output_count; ++i) input.get()[i] = static_cast<float>(i % 97) * 0.01F;
  std::fill_n(output.get(), output_count, 0.0F);
  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(user_position_bias, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), table.get(), side.get(), input.get(),
                                output.get());
  for (int32_t r = 0; r < kUserPositionBiasRows; ++r) {
    for (int32_t p1 = 0; p1 < kUserPositionBiasDim; ++p1) {
      for (int32_t p2 = 0; p2 < kUserPositionBiasLookups; ++p2) {
        const int64_t rel = static_cast<int64_t>(p2) - p1;
        const int64_t ab = std::abs(rel);
        int64_t where =
            ab < 8 ? ab : std::min<int64_t>(static_cast<int64_t>(std::log(ab / 8.0) / 2.772588722239781 * 8) + 8, 15);
        const int64_t bucket = (rel > 0 ? 16 : 0) + where;
        const int64_t off = (static_cast<int64_t>(r) * kUserPositionBiasDim + p1) * kUserPositionBiasLookups + p2;
        const float expected = input.get()[off] + table.get()[bucket * kUserPositionBiasRows + r] + side.get()[p2];
        EXPECT_NEAR(output.get()[off], expected, 1e-4F) << "r=" << r << ", p1=" << p1 << ", p2=" << p2;
      }
    }
  }
  if (workspace != nullptr) AscendC::GmFree(workspace);
}
#elif defined(IL_USER_POSITION_BIAS_EXP_SUM)
constexpr int32_t kUserPositionBiasExpSumRows = 8;
constexpr int32_t kUserPositionBiasExpSumDim = 256;
constexpr int32_t kUserPositionBiasExpSumLookups = 256;
constexpr int32_t kUserPositionBiasExpSumTableRows = 32;

TEST(E2EUserPositionBiasExpSum, GeneratedKernelMatchesReference) {
  const int64_t score_count =
      static_cast<int64_t>(kUserPositionBiasExpSumRows) * kUserPositionBiasExpSumDim * kUserPositionBiasExpSumLookups;
  const int64_t table_count = static_cast<int64_t>(kUserPositionBiasExpSumTableRows) * kUserPositionBiasExpSumRows;
  const int64_t bias_count = static_cast<int64_t>(kUserPositionBiasExpSumDim) * kUserPositionBiasExpSumLookups;
  const int64_t row_bias_count = static_cast<int64_t>(kUserPositionBiasExpSumRows) * kUserPositionBiasExpSumDim;
  auto score = indirect_load_test::AllocGmBuffer<float>(score_count);
  auto table = indirect_load_test::AllocGmBuffer<float>(table_count);
  auto bias = indirect_load_test::AllocGmBuffer<float>(bias_count);
  auto row_bias = indirect_load_test::AllocGmBuffer<float>(row_bias_count);
  auto output = indirect_load_test::AllocGmBuffer<float>(row_bias_count);
  ASSERT_TRUE(score && table && bias && row_bias && output);

  for (int64_t i = 0; i < score_count; ++i) {
    score.get()[i] = static_cast<float>(i % 97) * 0.02F - 0.9F;
  }
  for (int32_t bucket = 0; bucket < kUserPositionBiasExpSumTableRows; ++bucket) {
    for (int32_t r = 0; r < kUserPositionBiasExpSumRows; ++r) {
      table.get()[static_cast<int64_t>(bucket) * kUserPositionBiasExpSumRows + r] =
          static_cast<float>(bucket % 8) * 0.125F + static_cast<float>(r) * 0.03F;
    }
  }
  for (int64_t i = 0; i < bias_count; ++i) {
    bias.get()[i] = static_cast<float>(i % 53) * 0.04F - 1.0F;
  }
  for (int64_t i = 0; i < row_bias_count; ++i) {
    row_bias.get()[i] = static_cast<float>(i % 29) * 0.06F - 0.8F;
  }
  std::fill_n(output.get(), row_bias_count, 0.0F);

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);

  indirect_load_test::RunKernel(user_position_bias_exp_sum, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), score.get(), table.get(), bias.get(),
                                row_bias.get(), output.get());

  std::vector<int64_t> buckets(static_cast<size_t>(kUserPositionBiasExpSumLookups), 0);
  for (int32_t distance = 0; distance < kUserPositionBiasExpSumLookups; ++distance) {
    buckets[static_cast<size_t>(distance)] =
        distance < 16
            ? distance
            : std::min<int64_t>(static_cast<int64_t>(std::log(distance / 16.0) * 0.48089834696298783 * 16.0) + 16, 31);
  }
  for (int32_t r = 0; r < kUserPositionBiasExpSumRows; ++r) {
    for (int32_t p1 = 0; p1 < kUserPositionBiasExpSumDim; ++p1) {
      double expected = 0.0;
      for (int32_t p2 = 0; p2 < kUserPositionBiasExpSumLookups; ++p2) {
        const int64_t distance = p1 - p2 > 0 ? p1 - p2 : 0;
        const int64_t score_offset =
            (static_cast<int64_t>(r) * kUserPositionBiasExpSumDim + p1) * kUserPositionBiasExpSumLookups + p2;
        const double value =
            static_cast<double>(score.get()[score_offset]) +
            static_cast<double>(table.get()[buckets[static_cast<size_t>(distance)] * kUserPositionBiasExpSumRows + r]) +
            static_cast<double>(bias.get()[static_cast<int64_t>(p1) * kUserPositionBiasExpSumLookups + p2]) -
            static_cast<double>(row_bias.get()[static_cast<int64_t>(r) * kUserPositionBiasExpSumDim + p1]);
        expected += std::exp(value);
      }
      const int64_t output_offset = static_cast<int64_t>(r) * kUserPositionBiasExpSumDim + p1;
      EXPECT_NEAR(static_cast<double>(output.get()[output_offset]), expected, std::max(0.5, expected * 2e-3))
          << "r=" << r << ", p1=" << p1;
    }
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
}
#elif defined(IL_USER_MASKED_EMBEDDING_MINIMAL) || defined(IL_USER_MASKED_EMBEDDING_SUM_FULL)
#if defined(IL_USER_MASKED_EMBEDDING_SUM_FULL)
constexpr int32_t kUserMaskedRows = 128;
constexpr int32_t kUserMaskedDim = 128;
constexpr int32_t kUserMaskedLookups = 38;
constexpr int32_t kUserMaskedTableRows = 512;
#else
constexpr int32_t kUserMaskedRows = 2;
constexpr int32_t kUserMaskedDim = 2;
constexpr int32_t kUserMaskedLookups = 1;
constexpr int32_t kUserMaskedTableRows = 2;
#endif

TEST(E2EUserMaskedEmbeddingSum, GeneratedKernelMatchesReference) {
#if defined(IL_USER_MASKED_EMBEDDING_AUTO_SELECT)
  // AUTO 变体只做 codegen 断言（候选选择/policy 由 codegen_v2 target 的
  // GeneratesUserMaskedEmbeddingSumKernel 覆盖）：FULL 规模（128*128*38=622592 输出）
  // 的 rank3 Embedding policy 在 ICPU 仿真退化为元素级 async_invoke 线程调度，
  // 串行模式耗时超出 CI 时间预算（表现为卡住），真机 SIMT 执行无此问题。
  GTEST_SKIP() << "AUTO variant is codegen-only; ICPU sim of rank-3 embedding policy "
                  "exceeds CI time budget";
#else
  constexpr int64_t mask_count = static_cast<int64_t>(kUserMaskedRows) * kUserMaskedLookups;
  constexpr int64_t index_count = mask_count;
  constexpr int64_t embedding_count = static_cast<int64_t>(kUserMaskedTableRows) * kUserMaskedDim;
  constexpr int64_t output_count = static_cast<int64_t>(kUserMaskedRows) * kUserMaskedDim;
#if defined(IL_USER_MASKED_EMBEDDING_SUM_FULL)
  constexpr int64_t guard_count = 0;
#else
  // Minimal repro keeps guard elements after the physical [rows, lookups] view so the bad
  // output-index load reaches the oracle as a deterministic mask mismatch instead of a host SIGSEGV.
  constexpr int64_t guard_count = output_count * kUserMaskedLookups - mask_count;
#endif
  const int64_t storage_count = mask_count + guard_count;
  auto mask = indirect_load_test::AllocGmBuffer<int64_t>(storage_count);
  auto embedding = indirect_load_test::AllocGmBuffer<float>(embedding_count);
  auto index = indirect_load_test::AllocGmBuffer<int64_t>(index_count + guard_count);
  auto output = indirect_load_test::AllocGmBuffer<float>(output_count);
  ASSERT_TRUE(mask && embedding && index && output);

  std::fill_n(mask.get(), storage_count, static_cast<int64_t>(-1));
  std::fill_n(index.get(), index_count + guard_count, static_cast<int64_t>(0));
  for (int32_t row = 0; row < kUserMaskedTableRows; ++row) {
    for (int32_t column = 0; column < kUserMaskedDim; ++column) {
      embedding.get()[static_cast<int64_t>(row) * kUserMaskedDim + column] =
          static_cast<float>((row % 97) * 0.01F + (column % 31) * 0.001F);
    }
  }
  for (int32_t row = 0; row < kUserMaskedRows; ++row) {
    for (int32_t lookup = 0; lookup < kUserMaskedLookups; ++lookup) {
      const int64_t offset = static_cast<int64_t>(row) * kUserMaskedLookups + lookup;
      mask.get()[offset] = ((row * 7 + lookup * 11) % 5 == 0) ? -1 : (row + lookup) % 13;
      const int64_t positive = (static_cast<int64_t>(row) * 911 + lookup * 37) % kUserMaskedTableRows;
      index.get()[offset] = positive;
    }
  }
  std::fill_n(output.get(), output_count, 0.0F);

  indirect_load_test::KernelTiling tiling(1U);
  ASSERT_TRUE(tiling.IsValid());

  indirect_load_test::RunKernel(
      user_masked_embedding_sum, tiling.block_dim, reinterpret_cast<uint8_t *>(tiling.workspace.get()),
      reinterpret_cast<uint8_t *>(&tiling.data), mask.get(), embedding.get(), index.get(), output.get());

  for (int32_t row = 0; row < kUserMaskedRows; ++row) {
    for (int32_t column = 0; column < kUserMaskedDim; ++column) {
      float expected = 0.0F;
      for (int32_t lookup = 0; lookup < kUserMaskedLookups; ++lookup) {
        const int64_t offset = static_cast<int64_t>(row) * kUserMaskedLookups + lookup;
        if (mask.get()[offset] < 0) {
          continue;
        }
        const int64_t selected = index.get()[offset];
        expected += embedding.get()[selected * kUserMaskedDim + column];
      }
      EXPECT_NEAR(output.get()[static_cast<int64_t>(row) * kUserMaskedDim + column], expected, 1e-4F)
          << "row=" << row << ", column=" << column;
    }
  }
#endif
}
#elif defined(IL_GRAPH_HINT_EMBEDDING_SLICE)
constexpr int32_t kGraphHintEmbeddingSliceRows = 128;
constexpr int32_t kGraphHintEmbeddingSliceColumns = 512;
constexpr int32_t kGraphHintEmbeddingSliceTableRows = 512;
constexpr int32_t kGraphHintEmbeddingSliceIndexStride = 39;
constexpr int32_t kGraphHintEmbeddingSliceIndexOffset = 13;

TEST(E2EIndirectLoadGraphHintEmbeddingSlice, GeneratedKernelMatchesReference) {
  constexpr int64_t table_count =
      static_cast<int64_t>(kGraphHintEmbeddingSliceTableRows) * kGraphHintEmbeddingSliceColumns;
  constexpr int64_t index_count =
      static_cast<int64_t>(kGraphHintEmbeddingSliceIndexOffset) +
      static_cast<int64_t>(kGraphHintEmbeddingSliceRows - 1) * kGraphHintEmbeddingSliceIndexStride + 1;
  constexpr int64_t output_count = static_cast<int64_t>(kGraphHintEmbeddingSliceRows) * kGraphHintEmbeddingSliceColumns;
  auto table = indirect_load_test::AllocGmBuffer<float>(table_count);
  auto index = indirect_load_test::AllocGmBuffer<float>(index_count);
  auto output = indirect_load_test::AllocGmBuffer<float>(output_count);
  ASSERT_TRUE(table && index && output);

  for (int32_t row = 0; row < kGraphHintEmbeddingSliceTableRows; ++row) {
    for (int32_t column = 0; column < kGraphHintEmbeddingSliceColumns; ++column) {
      table.get()[static_cast<int64_t>(row) * kGraphHintEmbeddingSliceColumns + column] =
          static_cast<float>((row % 97) * 0.25F + (column % 31) * 0.001F);
    }
  }
  std::fill_n(index.get(), index_count, 0.0F);
  for (int32_t row = 0; row < kGraphHintEmbeddingSliceRows; ++row) {
    const int32_t positive_index = (row * 997 + 17) % kGraphHintEmbeddingSliceTableRows;
    const int32_t index_value = positive_index;
    index.get()[static_cast<int64_t>(kGraphHintEmbeddingSliceIndexOffset) +
                static_cast<int64_t>(row) * kGraphHintEmbeddingSliceIndexStride] = static_cast<float>(index_value);
  }
  std::fill_n(output.get(), output_count, 0.0F);

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  indirect_load_test::RunKernel(autofused_to_copy_embedding_slice_161da55df6362966862381c5d053b7e7, tiling.block_dim,
                                reinterpret_cast<uint8_t *>(tiling.workspace.get()),
                                reinterpret_cast<uint8_t *>(&tiling.data), table.get(), index.get(), output.get());

  for (int32_t row = 0; row < kGraphHintEmbeddingSliceRows; ++row) {
    const int32_t raw_index =
        static_cast<int32_t>(index.get()[static_cast<int64_t>(kGraphHintEmbeddingSliceIndexOffset) +
                                         static_cast<int64_t>(row) * kGraphHintEmbeddingSliceIndexStride]);
    const int32_t selected_index = raw_index;
    for (int32_t column = 0; column < kGraphHintEmbeddingSliceColumns; ++column) {
      const int64_t offset = static_cast<int64_t>(row) * kGraphHintEmbeddingSliceColumns + column;
      const int64_t table_offset = static_cast<int64_t>(selected_index) * kGraphHintEmbeddingSliceColumns + column;
      EXPECT_FLOAT_EQ(output.get()[offset], table.get()[table_offset]) << "row=" << row << ", column=" << column;
    }
  }
}
#elif defined(IL_GRAPH_HINT_SIMD_REPRO)
constexpr int32_t kGraphHintSimdRows = 30;
constexpr int32_t kGraphHintSimdIndexColumns = 3;
constexpr int32_t kGraphHintSimdInner = 23;
constexpr int32_t kGraphHintSimdInputRows = 6;

TEST(E2EIndirectLoadGraphHintSimdRepro, GeneratedKernelMatchesReference) {
  constexpr int64_t input_count =
      static_cast<int64_t>(kGraphHintSimdRows) * kGraphHintSimdInputRows * kGraphHintSimdInner;
  constexpr int64_t index_count = kGraphHintSimdIndexColumns;
  constexpr int64_t output_count = static_cast<int64_t>(kGraphHintSimdRows) * kGraphHintSimdIndexColumns;
  auto *input = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * input_count));
  auto *index = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * index_count));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * output_count));
  ASSERT_NE(input, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(output, nullptr);

  for (int32_t row = 0; row < kGraphHintSimdRows; ++row) {
    for (int32_t column = 0; column < kGraphHintSimdInputRows; ++column) {
      for (int32_t inner = 0; inner < kGraphHintSimdInner; ++inner) {
        input[(static_cast<int64_t>(row) * kGraphHintSimdInputRows + column) * kGraphHintSimdInner + inner] =
            static_cast<float>((row * kGraphHintSimdInputRows + column) * kGraphHintSimdInner + inner);
      }
    }
  }
  std::fill_n(index, index_count, static_cast<int64_t>(5));
  std::fill_n(output, output_count, 0.0F);

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);

  indirect_load_test::RunKernel(indirect_load_graph_hint_simd_repro, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), input, index, output);

  for (int32_t row = 0; row < kGraphHintSimdRows; ++row) {
    for (int32_t column = 0; column < kGraphHintSimdIndexColumns; ++column) {
      float expected = 0.0F;
      for (int32_t inner = 0; inner < kGraphHintSimdInner; ++inner) {
        expected +=
            input[(static_cast<int64_t>(row) * kGraphHintSimdInputRows + index[column]) * kGraphHintSimdInner + inner];
      }
      EXPECT_FLOAT_EQ(output[row * kGraphHintSimdIndexColumns + column], expected)
          << "row=" << row << ", column=" << column;
    }
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(input);
  AscendC::GmFree(index);
  AscendC::GmFree(output);
}
#elif defined(IL_GRAPH_HINT_REDUCE)
constexpr int32_t kGraphHintRows = 8;
constexpr int32_t kGraphHintColumns = 50;
constexpr int32_t kGraphHintTableRows = 512;
constexpr int32_t kGraphHintTableStride = 8;

TEST(E2EIndirectLoadGraphHintReduce, GeneratedKernelMatchesReference) {
  auto *index0 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kGraphHintColumns));
  auto *table = static_cast<float *>(
      AscendC::GmAlloc(sizeof(float) * (static_cast<int64_t>(kGraphHintTableRows - 1) * kGraphHintTableStride + 1)));
  auto *index2 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kGraphHintColumns));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kGraphHintRows));
  ASSERT_NE(index0, nullptr);
  ASSERT_NE(table, nullptr);
  ASSERT_NE(index2, nullptr);
  ASSERT_NE(output, nullptr);

  for (int32_t row = 0; row < kGraphHintTableRows; ++row) {
    table[static_cast<int64_t>(row) * kGraphHintTableStride] = static_cast<float>((row % 97) * 0.25F + 1.0F);
  }
  for (int32_t column = 0; column < kGraphHintColumns; ++column) {
    index0[column] = (column % 3 == 0) ? -1 : static_cast<int64_t>((column * 10000 + 7) % kGraphHintTableRows);
    index2[column] = static_cast<int64_t>((column * 20000 + 11) % kGraphHintTableRows);
  }

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 48;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);

  indirect_load_test::RunKernel(indirect_load_graph_hint_reduce_simt_test, block_dim,
                                reinterpret_cast<uint8_t *>(workspace), reinterpret_cast<uint8_t *>(&tiling_data),
                                index0, table, index2, output);

  for (int32_t row = 0; row < kGraphHintRows; ++row) {
    float expected = 0.0F;
    for (int32_t column = 0; column < kGraphHintColumns; ++column) {
      const int64_t selected = index0[column] == -1 ? index2[column] : index0[column];
      expected += table[selected * kGraphHintTableStride];
    }
    EXPECT_FLOAT_EQ(output[row], expected) << "row=" << row;
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(index0);
  AscendC::GmFree(table);
  AscendC::GmFree(index2);
  AscendC::GmFree(output);
}
#elif defined(IL_EMBEDDING_REDUCE)
constexpr int32_t kEmbRows = 2;
constexpr int32_t kEmbColumns = 2;
constexpr int32_t kEmbReduceSize = 2;
constexpr int32_t kEmbTableRows = 4;

TEST(E2EIndirectLoadEmbeddingReduce, GeneratedKernelMatchesReference) {
  const int64_t index_count = static_cast<int64_t>(kEmbRows) * kEmbReduceSize;
  const int64_t table_count = static_cast<int64_t>(kEmbTableRows) * kEmbColumns;
  const int64_t output_count = static_cast<int64_t>(kEmbRows) * kEmbColumns;
  auto *index0 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * index_count));
  auto *table = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * table_count));
  auto *unused_input = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * index_count));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * output_count));
  ASSERT_NE(index0, nullptr);
  ASSERT_NE(table, nullptr);
  ASSERT_NE(unused_input, nullptr);
  ASSERT_NE(output, nullptr);

  for (int32_t row = 0; row < kEmbTableRows; ++row) {
    for (int32_t col = 0; col < kEmbColumns; ++col) {
      table[static_cast<int64_t>(row) * kEmbColumns + col] = static_cast<float>(row * kEmbColumns + col);
    }
  }
  for (int32_t p0 = 0; p0 < kEmbRows; ++p0) {
    for (int32_t p2 = 0; p2 < kEmbReduceSize; ++p2) {
      const int32_t pos = p0 * kEmbReduceSize + p2;
      index0[pos] = static_cast<int64_t>((p0 * kEmbReduceSize + p2) % kEmbTableRows);
      unused_input[pos] = 0;
    }
  }

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 48;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(indirect_load_embedding_reduce_simt_test, block_dim,
                                reinterpret_cast<uint8_t *>(workspace), reinterpret_cast<uint8_t *>(&tiling_data),
                                index0, table, unused_input, output);

  for (int32_t p0 = 0; p0 < kEmbRows; ++p0) {
    for (int32_t p1 = 0; p1 < kEmbColumns; ++p1) {
      float expected = 0.0F;
      for (int32_t p2 = 0; p2 < kEmbReduceSize; ++p2) {
        const int64_t idx = index0[p0 * kEmbReduceSize + p2];
        expected += table[idx * kEmbColumns + p1] * (idx >= 0 ? 1.0F : 0.0F);
      }
      const int64_t offset = static_cast<int64_t>(p0) * kEmbColumns + p1;
      EXPECT_NEAR(output[offset], expected, 0.0625F) << "p0=" << p0 << ", p1=" << p1;
    }
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(index0);
  AscendC::GmFree(table);
  AscendC::GmFree(unused_input);
  AscendC::GmFree(output);
}
#else
#ifndef IL_INDEX_MIXED_VIEW
#define IL_INDEX_MIXED_VIEW 0
#endif
constexpr int32_t kRows = 6400;
constexpr int32_t kColumns = 32;
constexpr int32_t kTableRows = 315511;
constexpr bool kIndexMixedView = IL_INDEX_MIXED_VIEW;

TEST(E2EIndirectLoadBroadcastWhere, GeneratedKernelMatchesReference) {
  const int64_t table_count = static_cast<int64_t>(kTableRows) * kColumns;
  const int64_t output_count = static_cast<int64_t>(kRows) * kColumns;
  auto *index0 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kRows));
  auto *table = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * table_count));
  const int64_t index2_count = kIndexMixedView ? output_count : kRows;
  auto *index2 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * index2_count));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * output_count));
  ASSERT_NE(index0, nullptr);
  ASSERT_NE(table, nullptr);
  ASSERT_NE(index2, nullptr);
  ASSERT_NE(output, nullptr);

  for (int32_t row = 0; row < kTableRows; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      table[static_cast<int64_t>(row) * kColumns + column] =
          static_cast<float>((row % 97) * 0.25F + (column % 31) * 0.03125F);
    }
  }
  for (int32_t row = 0; row < kRows; ++row) {
    index0[row] = row % 2 == 0 ? -1 : static_cast<int64_t>((row * 17 + 3) % kTableRows);
    if constexpr (kIndexMixedView) {
      for (int32_t column = 0; column < kColumns; ++column) {
        index2[static_cast<int64_t>(row) * kColumns + column] =
            static_cast<int64_t>((row * 29 + column * 7 + 11) % kTableRows);
      }
    } else {
      index2[row] = static_cast<int64_t>((row * 29 + 7) % kTableRows);
    }
  }

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 48;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(indirect_load_broadcast_index_where_simt_test, block_dim,
                                reinterpret_cast<uint8_t *>(workspace), reinterpret_cast<uint8_t *>(&tiling_data),
                                index0, table, index2, output);

  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      const int64_t offset = static_cast<int64_t>(row) * kColumns + column;
      const int64_t selected = index0[row] == -1 ? index2[kIndexMixedView ? offset : row] : index0[row];
      const float expected = table[selected * kColumns + column];
      EXPECT_FLOAT_EQ(output[offset], expected) << "row=" << row << ", column=" << column;
    }
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(index0);
  AscendC::GmFree(table);
  AscendC::GmFree(index2);
  AscendC::GmFree(output);
}
#endif  // IL_EMBEDDING_REDUCE
#else
constexpr int32_t kAddIlReduceRows = 4;
constexpr int32_t kAddIlReduceColumns = 16;
constexpr int32_t kAddIlReduceTableRows = 8;

TEST(E2EIndirectLoadAddIlReduce, GeneratedKernelMatchesReference) {
  const int64_t table_count = static_cast<int64_t>(kAddIlReduceTableRows) * kAddIlReduceColumns;
  auto *index0 = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kAddIlReduceRows));
  auto *table = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * table_count));
  auto *offset = static_cast<int64_t *>(AscendC::GmAlloc(sizeof(int64_t) * kAddIlReduceRows));
  auto *output = static_cast<float *>(AscendC::GmAlloc(sizeof(float) * kAddIlReduceRows));
  ASSERT_NE(index0, nullptr);
  ASSERT_NE(table, nullptr);
  ASSERT_NE(offset, nullptr);
  ASSERT_NE(output, nullptr);

  for (int32_t row = 0; row < kAddIlReduceTableRows; ++row) {
    for (int32_t column = 0; column < kAddIlReduceColumns; ++column) {
      table[row * kAddIlReduceColumns + column] = static_cast<float>(row * kAddIlReduceColumns + column);
    }
  }
  for (int32_t row = 0; row < kAddIlReduceRows; ++row) {
    index0[row] = static_cast<int64_t>((row * 3) % kAddIlReduceTableRows);
    offset[row] = row % 2 == 0 ? 1 : -1;
  }

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0;
  uint32_t block_dim = 48;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
  void *workspace = workspace_size == 0U ? nullptr : AscendC::GmAlloc(workspace_size);
  ASSERT_TRUE(workspace_size == 0U || workspace != nullptr);
  indirect_load_test::RunKernel(indirect_load_add_il_reduce_test, block_dim, reinterpret_cast<uint8_t *>(workspace),
                                reinterpret_cast<uint8_t *>(&tiling_data), index0, table, offset, output);

  for (int32_t row = 0; row < kAddIlReduceRows; ++row) {
    // 与图语义一致：FLOAT 域做 Add 后 Cast 回 INT64 作为 gather 行号。
    const int64_t selected = static_cast<int64_t>(static_cast<float>(index0[row]) + static_cast<float>(offset[row]));
    float expected = 0.0F;
    for (int32_t column = 0; column < kAddIlReduceColumns; ++column) {
      expected += table[selected * kAddIlReduceColumns + column];
    }
    EXPECT_NEAR(output[row], expected, 0.0625F) << "row=" << row;
  }

  if (workspace != nullptr) {
    AscendC::GmFree(workspace);
  }
  AscendC::GmFree(index0);
  AscendC::GmFree(table);
  AscendC::GmFree(offset);
  AscendC::GmFree(output);
}
#endif  // IL_ADD_IL_REDUCE
}  // namespace

#endif

#if defined(IL_CASE_STRIDE_ZERO)
#include <array>

extern "C" __global__ __aicore__ void indirect_load_stride_zero_test(GM_ADDR x, GM_ADDR index, GM_ADDR y,
                                                                     GM_ADDR workspace, GM_ADDR tiling);

namespace {
using DataType = half;
using IndexType = int64_t;
constexpr std::array<int64_t, 4> kShape = {4, 5, 4, 16};
constexpr uint32_t kInputZeroStrideMask = IL_INPUT_ZERO_STRIDE_MASK;
constexpr uint32_t kIndexZeroStrideMask = IL_INDEX_ZERO_STRIDE_MASK;
constexpr bool kHasInputElement = IL_HAS_INPUT_ELEMENT > 0;
constexpr bool kHasIndexElement = IL_HAS_INDEX_ELEMENT > 0;

std::array<int64_t, 4> MakeStrides(uint32_t zero_stride_mask) {
  std::array<int64_t, 4> strides{};
  int64_t stride = 1;
  for (size_t index = kShape.size(); index > 0UL; --index) {
    const size_t dim = index - 1UL;
    if ((zero_stride_mask & (1U << dim)) == 0U) {
      strides[dim] = stride;
      stride *= kShape[dim];
    }
  }
  return strides;
}

int64_t StorageSpan(const std::array<int64_t, 4> &strides) {
  int64_t span = 1;
  for (size_t dim = 0; dim < kShape.size(); ++dim) {
    span += (kShape[dim] - 1) * strides[dim];
  }
  return span;
}

int64_t Offset(const std::array<int64_t, 4> &coordinate, const std::array<int64_t, 4> &strides) {
  int64_t offset = 0;
  for (size_t dim = 0; dim < coordinate.size(); ++dim) {
    offset += coordinate[dim] * strides[dim];
  }
  return offset;
}

void InitializeData(DataType *x, IndexType *index, DataType *expected, const std::array<int64_t, 4> &input_strides,
                    const std::array<int64_t, 4> &index_strides) {
  const int64_t input_count = StorageSpan(input_strides);
  const int64_t index_count = StorageSpan(index_strides);
  for (int64_t i = 0; i < input_count; ++i) {
    x[i] = static_cast<DataType>(static_cast<float>((i % 37) - 18) * 0.25F);
  }
  for (int64_t i = 0; i < index_count; ++i) {
    const int64_t gathered_axis = (i * 3 + 1) % kShape[2];
    index[i] = kHasIndexElement && (i % 2 == 1) ? -gathered_axis : gathered_axis;
  }

  const int64_t output_count = kShape[0] * kShape[1] * kShape[2] * kShape[3];
  for (int64_t output_offset = 0; output_offset < output_count; ++output_offset) {
    int64_t linear = output_offset;
    const int64_t d = linear % kShape[3];
    linear /= kShape[3];
    const int64_t c = linear % kShape[2];
    linear /= kShape[2];
    const int64_t b = linear % kShape[1];
    const int64_t a = linear / kShape[1];
    std::array<int64_t, 4> coordinate = {a, b, c, d};
    int64_t gathered_axis = index[Offset(coordinate, index_strides)];
    if (kHasIndexElement) {
      gathered_axis = std::abs(gathered_axis);
    }
    coordinate[2] = gathered_axis;
    float value = static_cast<float>(x[Offset(coordinate, input_strides)]);
    if (kHasInputElement) {
      value = std::abs(value);
    }
    expected[output_offset] = static_cast<DataType>(std::max(value, 0.0F));
  }
}
}  // namespace

TEST(E2EIndirectLoadStrideZero, GeneratedKernelMatchesPhysicalStrideReference) {
  const auto dense_strides = MakeStrides(0U);
  const auto input_strides = MakeStrides(kInputZeroStrideMask);
  const auto index_strides = MakeStrides(kIndexZeroStrideMask);
  const int64_t input_count = StorageSpan(input_strides);
  const int64_t index_count = StorageSpan(index_strides);
  const int64_t output_count = StorageSpan(dense_strides);
  indirect_load_test::KernelData<DataType, IndexType> buffers(input_count, index_count, output_count);
  ASSERT_TRUE(buffers.IsValid());
  InitializeData(buffers.input.get(), buffers.index.get(), buffers.expected.data(), input_strides, index_strides);
  std::fill_n(buffers.output.get(), output_count, static_cast<DataType>(0.0F));
  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());

  indirect_load_test::RunKernel(indirect_load_stride_zero_test, tiling.data.block_dim, tiling.workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling.data), buffers.input.get(), buffers.index.get(),
                                buffers.output.get());
  for (int64_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(static_cast<float>(buffers.output.get()[i]),
                static_cast<float>(buffers.expected[static_cast<size_t>(i)]), 0.0625F)
        << "offset=" << i;
  }
}

#endif

#if defined(IL_CASE_TORCH_STRIDED)

extern "C" __global__ __aicore__ void indirect_load_torch_gather_strided_test(GM_ADDR data, GM_ADDR index,
                                                                              GM_ADDR output, GM_ADDR workspace,
                                                                              GM_ADDR tiling);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *, uint64_t *, uint32_t *, uint32_t, uint32_t);

namespace {
constexpr int32_t kInputStride0 = IL_INPUT_STRIDE0;
constexpr int32_t kInputStride1 = IL_INPUT_STRIDE1;
constexpr int32_t kInputStride2 = IL_INPUT_STRIDE2;
constexpr int32_t kIndexStride0 = IL_INDEX_STRIDE0;
constexpr int32_t kIndexStride1 = IL_INDEX_STRIDE1;
constexpr int32_t kIndexStride2 = IL_INDEX_STRIDE2;
#ifdef IL_INDEX_SELECT_CASE
constexpr int32_t kEffectiveInputStride0 = 138;
constexpr int32_t kEffectiveInputStride1 = 23;
constexpr int32_t kEffectiveInputStride2 = 1;
#else
constexpr int32_t kEffectiveInputStride0 = kInputStride0;
constexpr int32_t kEffectiveInputStride1 = kInputStride1;
constexpr int32_t kEffectiveInputStride2 = kInputStride2;
#endif
#ifdef IL_INDEX_SELECT_CASE
constexpr int32_t kDim0 = 30;
constexpr int32_t kInputDim1 = 6;
constexpr int32_t kOutputDim1 = 3;
constexpr int32_t kDim2 = 23;
#else
constexpr int32_t kDim0 = 8;
constexpr int32_t kInputDim1 = 32;
constexpr int32_t kOutputDim1 = 16;
constexpr int32_t kDim2 = 5;
#endif
#ifdef IL_INDEX_SELECT_CASE
constexpr int32_t kInputStorageSize = kDim0 * kInputDim1 * kDim2;
constexpr int32_t kIndexStorageSize = kOutputDim1;
constexpr int32_t kOutputSize = kDim0 * kOutputDim1 * kDim2;
#else
constexpr int32_t kInputStorageSize = (8 - 1) * kInputStride0 + (32 - 1) * kInputStride1 + (5 - 1) * kInputStride2 + 1;
constexpr int32_t kIndexStorageSize = (8 - 1) * kIndexStride0 + (16 - 1) * kIndexStride1 + (5 - 1) * kIndexStride2 + 1;
constexpr int32_t kOutputSize = 8 * 16 * 5;
#endif

void Initialize(float *data, int64_t *index, std::vector<float> &expected) {
  for (int32_t i = 0; i < kInputStorageSize; ++i) {
    data[i] = 1.0F + static_cast<float>((i * 37) % 997) / 997.0F;
  }
  std::fill_n(index, kIndexStorageSize, int64_t{0});
  for (int32_t a0 = 0; a0 < kDim0; ++a0) {
    for (int32_t a1 = 0; a1 < kOutputDim1; ++a1) {
      for (int32_t a2 = 0; a2 < kDim2; ++a2) {
        const int32_t output_offset = kOutputDim1 * kDim2 * a0 + kDim2 * a1 + a2;
#ifdef IL_INDEX_SELECT_CASE
        const int32_t index_offset = a1;
        const int64_t index_value = (a1 * 2 + 1) % kInputDim1;
#else
        const int32_t index_offset = kIndexStride0 * a0 + kIndexStride1 * a1 + kIndexStride2 * a2;
        const int64_t index_value = (a0 * 17 + a1 * 7 + a2 * 3) % 32;
#endif
        index[index_offset] = index_value;
        expected[static_cast<size_t>(output_offset)] =
            data[kEffectiveInputStride0 * a0 + kEffectiveInputStride1 * index_value + kEffectiveInputStride2 * a2];
      }
    }
  }
}
}  // namespace

TEST(E2EIndirectLoadTorchGatherStrided, GeneratedKernelMatchesReference) {
  const auto gm_free = [](void *ptr) { AscendC::GmFree(ptr); };
  std::unique_ptr<float, decltype(gm_free)> data(
      reinterpret_cast<float *>(AscendC::GmAlloc(kInputStorageSize * sizeof(float))), gm_free);
  std::unique_ptr<int64_t, decltype(gm_free)> index(
      reinterpret_cast<int64_t *>(AscendC::GmAlloc(kIndexStorageSize * sizeof(int64_t))), gm_free);
  std::unique_ptr<float, decltype(gm_free)> output(
      reinterpret_cast<float *>(AscendC::GmAlloc(kOutputSize * sizeof(float))), gm_free);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(output, nullptr);

  std::vector<float> expected(kOutputSize);
  Initialize(data.get(), index.get(), expected);
  std::fill_n(output.get(), kOutputSize, 0.0F);

  AutofuseTilingData tiling_data{};
  uint64_t workspace_size = 0U;
  uint32_t block_dim = 48U;
  ASSERT_EQ(AutofuseTiling(&tiling_data, &workspace_size, &block_dim, 48U, 192U * 1024U), 0);
#if IL_EXPECT_SIMT
  ASSERT_EQ(tiling_data.get_tiling_key(), static_cast<uint32_t>(IL_EXPECT_TILING_KEY));
#else
  ASSERT_EQ(tiling_data.graph0_tiling_key, static_cast<uint32_t>(IL_EXPECT_TILING_KEY));
#endif
  ASSERT_GT(tiling_data.block_dim, 0U);
  std::unique_ptr<uint8_t, decltype(gm_free)> workspace(nullptr, gm_free);
  if (workspace_size > 0U) {
    workspace.reset(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspace_size)));
    ASSERT_NE(workspace, nullptr);
    std::fill_n(workspace.get(), workspace_size, uint8_t{0});
  }

  indirect_load_test::RunKernel(indirect_load_torch_gather_strided_test, tiling_data.block_dim, workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling_data), data.get(), index.get(), output.get());
  int32_t mismatch_count = 0;
  int32_t first_mismatch = -1;
  for (int32_t i = 0; i < kOutputSize; ++i) {
    if (output.get()[i] != expected[static_cast<size_t>(i)]) {
      if (first_mismatch < 0) {
        first_mismatch = i;
      }
      ++mismatch_count;
    }
  }
  if (mismatch_count > 0) {
    ADD_FAILURE() << "mismatch count=" << mismatch_count << ", first mismatch at offset=" << first_mismatch
                  << ", actual=" << output.get()[first_mismatch]
                  << ", expected=" << expected[static_cast<size_t>(first_mismatch)];
  }
}

#endif

#if defined(IL_CASE_EMBEDDING)
#ifndef IL_EMBEDDING_SIZE
#define IL_EMBEDDING_SIZE 32
#endif

extern "C" __global__ __aicore__ void indirect_load_embedding_test(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                                                   GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int32_t kInputRows = 64;
constexpr int32_t kEmbeddingSize = IL_EMBEDDING_SIZE;
constexpr int32_t kIndexRows = 32;

void InitializeData(float *input, int32_t *index, float *expected) {
  for (int32_t row = 0; row < kInputRows; ++row) {
    for (int32_t col = 0; col < kEmbeddingSize; ++col) {
      input[row * kEmbeddingSize + col] = static_cast<float>(row * kEmbeddingSize + col);
    }
  }
  for (int32_t row = 0; row < kIndexRows; ++row) {
    const int32_t index_value = static_cast<int32_t>((row * 7 + 3) % kInputRows);
    index[row] = index_value;
#if defined(IL_EMBEDDING_DIRECT)
    for (int32_t col = 0; col < kEmbeddingSize; ++col) {
      expected[row * kEmbeddingSize + col] = input[index_value * kEmbeddingSize + col];
    }
#else
    float sum = 0.0F;
    for (int32_t col = 0; col < kEmbeddingSize; ++col) {
      sum += 2.0F * (input[index[row] * kEmbeddingSize + col] + 0.1F);
    }
    expected[row] = sum;
#endif
  }
}
}  // namespace

TEST(E2EIndirectLoadEmbedding, GeneratedKernelMatchesReference) {
  constexpr int64_t input_count = static_cast<int64_t>(kInputRows) * kEmbeddingSize;
  constexpr int64_t index_count = kIndexRows;
#if defined(IL_EMBEDDING_DIRECT)
  constexpr int64_t output_count = static_cast<int64_t>(kIndexRows) * kEmbeddingSize;
#else
  constexpr int64_t output_count = kIndexRows;
#endif
  indirect_load_test::KernelData<float, int32_t> buffers(input_count, index_count, output_count);
  ASSERT_TRUE(buffers.IsValid());
  InitializeData(buffers.input.get(), buffers.index.get(), buffers.expected.data());
  std::fill_n(buffers.output.get(), output_count, 0.0F);

  indirect_load_test::KernelTiling tiling;
  ASSERT_TRUE(tiling.IsValid());
  indirect_load_test::RunKernel(indirect_load_embedding_test, tiling.data.block_dim, tiling.workspace.get(),
                                reinterpret_cast<uint8_t *>(&tiling.data), buffers.input.get(), buffers.index.get(),
                                buffers.output.get());

  // 设备侧 ReduceSum 走向量/树形归约，与 CPU 串行累加顺序不同，浮点结果存在数 ULP 差异，故用绝对容差比较。
  for (int64_t i = 0; i < output_count; ++i) {
    EXPECT_NEAR(buffers.output.get()[i], buffers.expected[static_cast<size_t>(i)], 0.0625F) << "offset=" << i;
  }
}

#endif
