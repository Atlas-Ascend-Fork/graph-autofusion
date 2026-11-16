/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_ARGMAX_WITH_VALUE_H__
#define __ASCENDC_API_ARGMAX_WITH_VALUE_H__

namespace AscendC {

/**
 * @brief UpdateMaxIndexAndValue - 更新最大值及其索引（tensor版本）
 * @note 使用CompareExtend和Where算子来更新保存的最大值和索引
 * @param max_index_temp 当次计算的索引tensor（局部索引，会被修改为加上offset后的值）
 * @param max_value_temp 当次计算的值tensor
 * @param max_index_saved 之前保存的索引tensor（会被更新）
 * @param max_value_saved 之前保存的值tensor（会被更新）
 * @param offset 索引偏移量（标量），用于将局部索引转换为全局索引
 * @param tmp_buf 临时buffer
 * @param cal_cnt 计算的数据量
 *
 * @note max_index_temp 会被原地修改为 max_index_temp + offset，调用后不应再使用原始值
 * @note 仅支持 cal_cnt > 1 的 tensor 情况
 */
template <typename T>
inline __aicore__ void UpdateMaxIndexAndValue(const LocalTensor<int64_t> &max_index_temp,
                                              const LocalTensor<T> &max_value_temp,
                                              const LocalTensor<int64_t> &max_index_saved,
                                              const LocalTensor<T> &max_value_saved, const int64_t offset,
                                              LocalTensor<uint8_t> &tmp_buf, const uint32_t cal_cnt) {
  // cal_cnt 是元素个数，已经32B对齐

  // 计算 mask 需要的空间（按位，转换为字节数，并对齐到32字节）
  uint32_t mask_size = ((cal_cnt + 7) / 8 + 31) / 32 * 32;

  // 计算 int32 临时索引需要的空间（按字节）
  uint32_t index32_size = cal_cnt * sizeof(int32_t);

  // 1. 分配 int32 临时索引空间（用于计算 offset，因为 Adds 不支持 int64）
  LocalTensor<uint8_t> index32_tmp_uint8 = tmp_buf[0];
  index32_tmp_uint8.SetSize(index32_size);
  LocalTensor<int32_t> index32_tmp = index32_tmp_uint8.ReinterpretCast<int32_t>();

  // 2. 分配 mask 空间（必须独立，不能被复用）
  LocalTensor<uint8_t> mask = tmp_buf[index32_size];
  mask.SetSize(mask_size);

  // 3. 分配 CompareExtend 和 Where 共用的临时空间
  LocalTensor<uint8_t> common_tmp_buf = tmp_buf[index32_size + mask_size];

  // 4. 将 max_index_temp 从 int64_t cast 为 int32_t（临时存储）
  // 注意：假设索引值在 int32 范围内
  Cast(index32_tmp, max_index_temp, RoundMode::CAST_NONE, cal_cnt);

  // 5. 对 int32_t 临时索引加上 offset（标量加到 tensor）
  Adds(index32_tmp, index32_tmp, static_cast<int32_t>(offset), cal_cnt);

  // 6. 将计算结果 cast 回 int64_t，并原地修改 max_index_temp
  Cast(max_index_temp, index32_tmp, RoundMode::CAST_NONE, cal_cnt);

  // 7. 使用 CompareExtend 比较 (max_value_temp > max_value_saved)，生成 mask
  CompareExtend(mask, max_value_temp, max_value_saved, CMPMODE::GT, cal_cnt, common_tmp_buf);

  // 8. 使用 Where 更新 max_value_saved（复用 common_tmp_buf）
  Where(max_value_saved, mask, max_value_temp, max_value_saved, cal_cnt, common_tmp_buf);

  // 9. 使用 Where 更新 max_index_saved（复用 common_tmp_buf）
  Where(max_index_saved, mask, max_index_temp, max_index_saved, cal_cnt, common_tmp_buf);
}

/**
 * @brief 在64个元素中找到最大值及其索引（标量归约）
 *
 * @details
 * 算法逻辑：
 * - 遍历64个元素，逐个比较
 * - 如果 current > max_value，则更新 max_value 和 max_index
 * - 特殊处理NaN：使用 !(current <= max_value) && (max_value == max_value)
 *   - 如果 max_value 是 NaN，则 max_value == max_value 为 false，不更新
 *   - 否则，如果 current 是 NaN，则 !(current <= max_value) 为 true，会更新
 *   - 这样可以正确处理包含NaN情况
 *
 * 数学公式：
 *   max_index = argmax_i src[i]  (i ∈ [0, 63])
 *   max_value = max_i src[i]     (i ∈ [0, 63])
 *
 * @param dst_index 输出：最大值的索引（作为float返回）
 * @param dst_value 输出：最大值
 * @param src 输入：64个元素的tensor
 *
 * @note 正确处理NaN：如果所有元素都是NaN，返回第一个NaN
 */
template <typename T>
inline __aicore__ void ScalarReduceMax64(float &dst_index, T &dst_value, const LocalTensor<T> &src) {
  T max_value = src(0);
  int32_t max_index = 0;

  for (uint32_t i = 1; i < 64; ++i) {
    T current = src(i);
    max_index = (!(current <= max_value) && (max_value == max_value)) ? i : max_index;
    max_value = (!(current <= max_value) && (max_value == max_value)) ? current : max_value;
  }

  dst_index = static_cast<float>(max_index);
  dst_value = max_value;
}

/**
 * @brief 在64个元素中根据mask找到最小值及其索引（标量归约）
 *
 * @details
 * 算法逻辑：
 * - mask用于标记哪些元素是有效的（mask的第i位为1表示src[i]有效）
 * - 只考虑mask中为1的位对应的元素
 * - 如果min_index == -1，说明还没有找到有效元素，直接使用当前元素
 * - 否则，如果 current < min_value，则更新
 *
 * 数学公式：
 *   有效索引集合 S = {i | ((mask >> i) & 1) == 1, i ∈ [0, 63]}
 *   min_index = argmin_i src[i]  (i ∈ S)
 *   min_value = min_i src[i]     (i ∈ S)
 *
 * @param dst_index 输出：最小值的索引（作为float返回）
 * @param dst_value 输出：最小值
 * @param src 输入：64个元素的tensor
 * @param mask 输入：64位掩码，标记哪些元素有效
 *
 * @note src不会包含NaN值，故不需要特殊处理
 */
template <typename T>
inline __aicore__ void ScalarReduceMin64(float &dst_index, T &dst_value, const LocalTensor<T> &src,
                                         const uint64_t mask) {
  T min_value;
  int32_t min_index = -1;

  for (uint32_t i = 0; i < 64; ++i) {
    T current = src(i);
    min_value = (((mask >> i) & 1) && (min_index == -1)) ? current : min_value;
    min_index = (((mask >> i) & 1) && (min_index == -1 || current < min_value)) ? i : min_index;
    min_value = (((mask >> i) & 1) && (current < min_value)) ? current : min_value;
  }

  dst_index = static_cast<float>(min_index);
  dst_value = min_value;
}

/**
 * @brief Argmax with value - AR模式，last维度 <= 64
 *
 * @details
 * 算法逻辑：
 * - 对每个first维度的元素，在last维度中找最大值及其索引
 * - 对于float类型，需要特殊处理±0的情况（+0和-0在比较时相等）
 *
 * 步骤：
 * 1. 处理±0：将所有0值统一为+0（通过比较 src != 0，然后选择）
 * 2. 初始化tail_tmp为最小值（-INFINITY或INT32_MIN）
 * 3. 复制当前first维度的数据到tail_tmp
 * 4. 使用ScalarReduceMax64在64个元素中找最大值及其索引
 * 5. 将结果存入dst_index和dst_value
 *
 * 数学公式：
 *   对于每个 i ∈ [0, repeat_times-1]:
 *     dst_index[i] = argmax_j src[i*last + j]  (j ∈ [0, last-1])
 *     dst_value[i] = max_j src[i*last + j]     (j ∈ [0, last-1])
 *
 * @param dst_index 输出：每个first维度的最大值索引
 * @param dst_value 输出：每个first维度的最大值
 * @param src 输入：2D tensor，形状为 [repeat_times, last]
 * @param shared_tmp_buffer 临时buffer
 * @param repeat_times first维度的元素个数
 * @param last last维度的元素个数（<= 64）
 *
 * @note 对于float类型，正确处理±0和NaN
 */
template <typename T, bool isReuseSource = false>
inline __aicore__ void ArgmaxLEOneRepeatWithValue(const LocalTensor<int64_t> &dst_index,
                                                  const LocalTensor<T> &dst_value, const LocalTensor<T> &src,
                                                  const LocalTensor<uint8_t> &shared_tmp_buffer,
                                                  const uint32_t repeat_times, const uint32_t last) {
  LocalTensor<T> tail_tmp = shared_tmp_buffer.ReinterpretCast<T>();
  LocalTensor<uint8_t> tmp_mask = shared_tmp_buffer[256];

  uint32_t copy_num = last;

  for (uint32_t i = 0; i < repeat_times; ++i) {
    // tail_tmp used for ±0 first, then process tail
    if constexpr (std::is_same_v<T, float>) {
      AscendC::PipeBarrier<PIPE_V>();
      Duplicate(tail_tmp, 0.0f, 64);
      AscendC::PipeBarrier<PIPE_V>();
      Compare(tmp_mask, src, tail_tmp, AscendC::CMPMODE::EQ, 64);
      AscendC::PipeBarrier<PIPE_V>();
      Select(src, tmp_mask, tail_tmp, src, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (std::is_same_v<T, float>) {
      Duplicate(tail_tmp, -INFINITY, 64);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      Duplicate(tail_tmp, INT32_MIN, 64);
    }
    AscendC::PipeBarrier<PIPE_V>();
    DataCopy(tail_tmp, src[i * last], copy_num);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float max_index;
    T max_value;
    ScalarReduceMax64<T>(max_index, max_value, tail_tmp);
    int32_t index = static_cast<int32_t>(max_index);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    dst_index(i) = static_cast<int64_t>(index);
    dst_value(i) = static_cast<T>(max_value);
  }
}

/**
 * @brief Argmax with value - AR模式，last维度 > 64
 *
 * @details
 * 算法逻辑：
 * - 通过滑动窗口的方式，在每64元素的块间逐块比较并跟踪最大值和索引
 * - 使用mask机制记录最大值出现的位置，避免丢失索引信息
 *
 * 步骤：
 * 1. 处理±0：将所有0值统一为+0（通过比较 src != 0，然后选择）
 * 2. 初始化max_before为当前行的第一个64元素块
 * 3. 遍历后续块（repeat_times_one_row - 2个完整块）：
 *    a. max_after = Max(max_before, next_block)
 *    b. 生成mask：mask1 = (max_before == max_before), mask2 = (max_after == max_before)
 *    c. 计算有效mask：mask2 = mask2 | (~mask1)，表示max_after中是从max_before继承的位置（或者NaN的位置）
 *    d. 更新索引：tmp_dst_tensor = mask2 ? tmp_dst_tensor : current_block_index
 *    e. max_before = max_after
 * 4. 处理尾部（用最小值补齐不足64个元素的元素块）
 * 5. 在tmp_dst_tensor中找索引最大值中block_index最小值对应的索引，得到相对索引 relative_index
 * 6. 计算最终索引：final_index = block_index + relative_index
 *
 * 数学公式：
 *   对于每个 i ∈ [0, first-1]:
 *     dst_index[i] = argmax_j src[i*last + j]  (j ∈ [0, last-1])
 *     dst_value[i] = max_j src[i*last + j]   (j ∈ [0, last-1])
 *
 * 内存布局说明：
 *   - 最小处理单位：64个元素
 *   - float/int32_t：64×4=256B（精确匹配）
 *   - uint8_t：64×1=64B（分配256B保持一致性）
 *   - 满足32字节对齐要求，DMA传输高效
 *
 * @param dst_index 输出：每个first维度的最大值全局索引
 * @param dst_value 输出：每个first维度的最大值
 * @param src 输入：2D tensor，形状为 [first, last]
 * @param shared_tmp_buffer 临时buffer
 * @param first first维度的元素个数
 * @param last last维度的元素个数（> 64）
 * @param repeat_times_one_row last维度按64分块的块数
 *
 * @note 使用mask机制确保在有多个最大值时，选择第一个出现的
 */
template <typename T, class pattern, bool isReuseSource = false>
__aicore__ void ArgmaxGTOneRepeatWithValue(const LocalTensor<int64_t> &dst_index, const LocalTensor<T> &dst_value,
                                           const LocalTensor<T> &src, const LocalTensor<uint8_t> &shared_tmp_buffer,
                                           const uint32_t first, const uint32_t last,
                                           const uint32_t repeat_times_one_row) {
  LocalTensor<float> tmp_dst_tensor = shared_tmp_buffer.ReinterpretCast<float>();
  LocalTensor<float> dup_index_tensor = shared_tmp_buffer[256].ReinterpretCast<float>();
  LocalTensor<uint8_t> tmp_mask = shared_tmp_buffer[512].ReinterpretCast<uint8_t>();
  LocalTensor<uint8_t> tmp_mask1 = shared_tmp_buffer[768].ReinterpretCast<uint8_t>();
  LocalTensor<uint8_t> tmp_mask2 = shared_tmp_buffer[1024].ReinterpretCast<uint8_t>();
  LocalTensor<T> max_before = shared_tmp_buffer[1280].ReinterpretCast<T>();
  LocalTensor<T> max_after = shared_tmp_buffer[1536].ReinterpretCast<T>();
  LocalTensor<T> tail_tmp = shared_tmp_buffer[1792].ReinterpretCast<T>();

  uint32_t tail_num = last % 64;
  tail_num = (tail_num == 0) ? 64 : tail_num;
  uint32_t copy_num = tail_num;
  uint32_t tail_ctrl = 1;  // force tail process

  for (uint32_t i = 0; i < first; ++i) {
    for (uint32_t j = 0; j < repeat_times_one_row; ++j) {
      // tail_tmp used for ±0 first, then process tail
      if constexpr (std::is_same_v<T, float>) {
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(tail_tmp, 0.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Compare(tmp_mask, src[i * last + j * 64], tail_tmp, AscendC::CMPMODE::EQ, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Select(src[i * last + j * 64], tmp_mask, tail_tmp, src[i * last + j * 64],
               AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
  }
  for (uint32_t i = 0; i < first; ++i) {
    Duplicate(tmp_dst_tensor, 0.0f, tmp_dst_tensor.GetSize());
    AscendC::PipeBarrier<PIPE_V>();
    DataCopy(max_before, src[i * last], 64);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < repeat_times_one_row - 2; ++j) {
      AscendC::PipeBarrier<PIPE_V>();
      Duplicate(dup_index_tensor, (j + 1) * 64.0f, 64);
      AscendC::PipeBarrier<PIPE_V>();
      Max(max_after, max_before, src[i * last + (j + 1) * 64], 64);
      AscendC::PipeBarrier<PIPE_V>();
      Compare(tmp_mask1, max_before, max_before, AscendC::CMPMODE::EQ, 64);
      AscendC::PipeBarrier<PIPE_V>();
      Compare(tmp_mask2, max_after, max_before, AscendC::CMPMODE::EQ, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      for (uint32_t k = 0; k < 8; ++k) {
        uint8_t val1 = tmp_mask1.GetValue(k);
        uint8_t val2 = tmp_mask2.GetValue(k);
        tmp_mask2.SetValue(k, val2 | (~val1));
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      Select(tmp_dst_tensor, tmp_mask2, tmp_dst_tensor, dup_index_tensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
             64);
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(max_before, max_after, 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::PipeBarrier<PIPE_V>();
    // tail process
    for (uint32_t j = 0; j < tail_ctrl; ++j) {
      if constexpr (std::is_same_v<T, float>) {
        Duplicate(tail_tmp, -INFINITY, 64);
      } else if constexpr (std::is_same_v<T, int32_t>) {
        Duplicate(tail_tmp, INT32_MIN, 64);
      }
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(tail_tmp, src[(i + 1) * last - copy_num], copy_num);
      AscendC::PipeBarrier<PIPE_V>();
      Duplicate(dup_index_tensor, (repeat_times_one_row - 1) * 64.0f, 64);
      AscendC::PipeBarrier<PIPE_V>();
      Max(max_after, max_before, tail_tmp, 64);
      AscendC::PipeBarrier<PIPE_V>();
      Compare(tmp_mask1, max_before, max_before, AscendC::CMPMODE::EQ, 64);
      AscendC::PipeBarrier<PIPE_V>();
      Compare(tmp_mask2, max_after, max_before, AscendC::CMPMODE::EQ, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      for (uint32_t k = 0; k < 8; ++k) {
        uint8_t val1 = tmp_mask1.GetValue(k);
        uint8_t val2 = tmp_mask2.GetValue(k);
        tmp_mask2.SetValue(k, val2 | (~val1));
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      Select(tmp_dst_tensor, tmp_mask2, tmp_dst_tensor, dup_index_tensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
             64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float max_index;
    T max_value;
    ScalarReduceMax64<T>(max_index, max_value, max_after);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    Compare(tmp_mask1, max_after, max_after, AscendC::CMPMODE::EQ, 64);
    AscendC::PipeBarrier<PIPE_V>();
    Compares(tmp_mask2, max_after, max_value, AscendC::CMPMODE::EQ, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (uint32_t k = 0; k < 8; ++k) {
      uint8_t val1 = tmp_mask1.GetValue(k);
      uint8_t val2 = tmp_mask2.GetValue(k);
      tmp_mask2.SetValue(k, val2 | (~val1));
    }
    uint64_t min_mask[1] = {(static_cast<uint64_t>(tmp_mask2(0)) << 0) | (static_cast<uint64_t>(tmp_mask2(1)) << 8) |
                            (static_cast<uint64_t>(tmp_mask2(2)) << 16) | (static_cast<uint64_t>(tmp_mask2(3)) << 24) |
                            (static_cast<uint64_t>(tmp_mask2(4)) << 32) | (static_cast<uint64_t>(tmp_mask2(5)) << 40) |
                            (static_cast<uint64_t>(tmp_mask2(6)) << 48) | (static_cast<uint64_t>(tmp_mask2(7)) << 56)};
    float tmp_index, tmp_value;
    ScalarReduceMin64<float>(tmp_index, tmp_value, tmp_dst_tensor, min_mask[0]);
    int32_t relative_index = static_cast<int32_t>(tmp_index);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    float final_index = tmp_dst_tensor(relative_index) + relative_index * 1.0f;
    dst_index(i) = static_cast<int64_t>(final_index);
    dst_value(i) = static_cast<T>(max_value);
  }
}

/**
 * @brief Argmax with value - RA模式（Reduce Axis）
 *
 * @details
 * 算法逻辑：
 * - 在last维度上计算argmax，结果形状为 [first, last]
 * - 对每个last维度的位置，在first维度上找最大值及其索引
 *
 * 步骤：
 * 1. 处理±0：将所有0值统一为+0（通过比较 src != 0，然后选择）
 * 2. 特殊处理：如果first == 1，直接复制数据
 * 3. 遍历每个64长度的元素块（除了尾部）：
 *    a. 初始化max_before为当前块
 *    b. 遍历first维度：
 *       i. max_after = Max(max_before, next_element)
 *       ii. 生成mask并更新索引（与ArgmaxGTOneRepeatWithValue类似）
 *       iii. 将索引结果存入dst_index
 *    c. 将max_after存入dst_value
 * 4. 处理尾部（不足64的元素）
 *
 * 数学公式：
 *   对于每个 j ∈ [0, last-1]:
 *     dst_index[j] = argmax_i src[i*last + j]  (i ∈ [0, first-1])
 *     dst_value[j] = max_i src[i*last + j]     (i ∈ [0, first-1])
 *
 * 内存布局说明：
 *   - 最小处理单位：64个元素
 *   - float/int32_t：64×4=256B（精确匹配）
 *   - uint8_t：64×1=64B（分配256B保持一致性）
 *   - 满足32字节对齐要求，DMA传输高效
 *
 * @param dst_index 输出：2D tensor，形状为 [first, last]，存储每个位置的argmax索引
 * @param dst_value 输出：2D tensor，形状为 [first, last]，存储每个位置的最大值
 * @param src 输入：2D tensor，形状为 [first, last]
 * @param shared_tmp_buffer 临时buffer
 * @param first first维度的元素个数
 * @param last last维度的元素个数
 * @param repeat_times_one_row last维度按64分块的块数
 */
template <typename T, class pattern, bool isReuseSource = false>
__aicore__ void ArgmaxRAWithValue(const LocalTensor<int64_t> &dst_index, const LocalTensor<T> &dst_value,
                                  const LocalTensor<T> &src, const LocalTensor<uint8_t> &shared_tmp_buffer,
                                  const uint32_t first, const uint32_t last, const uint32_t repeat_times_one_row) {
  LocalTensor<float> tmp_dst_tensor = shared_tmp_buffer.ReinterpretCast<float>();
  LocalTensor<float> dup_index_tensor = shared_tmp_buffer[256].ReinterpretCast<float>();
  LocalTensor<uint8_t> tmp_mask = shared_tmp_buffer[512].ReinterpretCast<uint8_t>();
  LocalTensor<uint8_t> tmp_mask1 = shared_tmp_buffer[768].ReinterpretCast<uint8_t>();
  LocalTensor<uint8_t> tmp_mask2 = shared_tmp_buffer[1024].ReinterpretCast<uint8_t>();
  LocalTensor<T> max_before = shared_tmp_buffer[1280].ReinterpretCast<T>();
  LocalTensor<T> max_after = shared_tmp_buffer[1536].ReinterpretCast<T>();
  LocalTensor<T> tail_tmp = shared_tmp_buffer[1792].ReinterpretCast<T>();

  Duplicate(tmp_dst_tensor, 0.0f, tmp_dst_tensor.GetSize());

  uint32_t tail_num = last % 64;
  tail_num = (tail_num == 0) ? 64 : tail_num;
  uint32_t copy_num = tail_num;
  uint32_t tail_ctrl = 1;  // force tail process
  uint32_t tail_offset = last - tail_num;

  for (uint32_t i = 0; i < first; ++i) {
    for (uint32_t j = 0; j < repeat_times_one_row; ++j) {
      // tail_tmp used for ±0 first, then process tail
      if constexpr (std::is_same_v<T, float>) {
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(tail_tmp, 0.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Compare(tmp_mask, src[i * last + j * 64], tail_tmp, AscendC::CMPMODE::EQ, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Select(src[i * last + j * 64], tmp_mask, tail_tmp, src[i * last + j * 64],
               AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
  }

  if (first == 1) {
    Duplicate(tmp_dst_tensor, 0.0f, dst_index.GetSize());
    AscendC::PipeBarrier<PIPE_V>();
    Cast<int64_t, float>(dst_index, tmp_dst_tensor, AscendC::RoundMode::CAST_RINT, dst_index.GetSize());
    DataCopy(dst_value, src, last);
  } else {
    for (uint32_t i = 0; i < repeat_times_one_row - 1; ++i) {
      Duplicate(tmp_dst_tensor, 0.0f, 64);
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(max_before, src[i * 64], 64);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t j = 0; j < first - 1; ++j) {
        Duplicate(dup_index_tensor, (j + 1) * 1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Max(max_after, max_before, src[i * 64 + (j + 1) * last], 64);
        AscendC::PipeBarrier<PIPE_V>();
        Compare(tmp_mask1, max_before, max_before, AscendC::CMPMODE::EQ, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Compare(tmp_mask2, max_after, max_before, AscendC::CMPMODE::EQ, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (uint32_t k = 0; k < 8; ++k) {
          uint8_t val1 = tmp_mask1.GetValue(k);
          uint8_t val2 = tmp_mask2.GetValue(k);
          tmp_mask2.SetValue(k, val2 | (~val1));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Select(tmp_dst_tensor, tmp_mask2, tmp_dst_tensor, dup_index_tensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
               64);
        AscendC::PipeBarrier<PIPE_V>();
        DataCopy(max_before, max_after, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Cast<int64_t, float>(dst_index[i * 64], tmp_dst_tensor, AscendC::RoundMode::CAST_RINT, 64);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(dst_value[i * 64], max_after, 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t j = 0; j < tail_ctrl; ++j) {
      Duplicate(tmp_dst_tensor, 0.0f, 64);
      if constexpr (std::is_same_v<T, float>) {
        Duplicate(tail_tmp, -INFINITY, 64);
      } else if constexpr (std::is_same_v<T, int32_t>) {
        Duplicate(tail_tmp, INT32_MIN, 64);
      }
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(tail_tmp, src[tail_offset], tail_num);
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(max_before, tail_tmp, 64);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t k = 0; k < first - 1; ++k) {
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(dup_index_tensor, (k + 1) * 1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, float>) {
          Duplicate(tail_tmp, -INFINITY, 64);
        } else if constexpr (std::is_same_v<T, int32_t>) {
          Duplicate(tail_tmp, INT32_MIN, 64);
        }
        AscendC::PipeBarrier<PIPE_V>();
        DataCopy(tail_tmp, src[(k + 1) * last + tail_offset], tail_num);
        AscendC::PipeBarrier<PIPE_V>();
        Max(max_after, max_before, tail_tmp, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Compare(tmp_mask1, max_before, max_before, AscendC::CMPMODE::EQ, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Compare(tmp_mask2, max_after, max_before, AscendC::CMPMODE::EQ, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (uint32_t k = 0; k < 8; ++k) {
          uint8_t val1 = tmp_mask1.GetValue(k);
          uint8_t val2 = tmp_mask2.GetValue(k);
          tmp_mask2.SetValue(k, val2 | (~val1));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        Select(tmp_dst_tensor, tmp_mask2, tmp_dst_tensor, dup_index_tensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
               64);
        AscendC::PipeBarrier<PIPE_V>();
        DataCopy(max_before, max_after, 64);
        AscendC::PipeBarrier<PIPE_V>();
        Cast<int64_t, float>(dst_index[tail_offset], tmp_dst_tensor, AscendC::RoundMode::CAST_RINT, tail_num);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(dst_value[tail_offset], max_after, tail_num);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }
}

/**
 * @brief Argmax with value 扩展函数 - 支持AR和RA两种模式
 *
 * @details
 * 算法逻辑：
 * - 根据reduce模式（AR或RA）和last维度大小，选择合适的实现
 * - AR模式：在last维度上reduce，输出形状为 [first]
 * - RA模式：在first维度上reduce，输出形状为 [last]
 *
 * 模式选择：
 * 1. AR模式：
 *    a. 如果last <= 64：调用ArgmaxLEOneRepeatWithValue
 *    b. 如果last > 64：调用ArgmaxGTOneRepeatWithValue
 * 2. RA模式：
 *    a. 调用ArgmaxRAWithValue
 *
 * @param dst_index 输出：argmax的索引
 * @param dst_value 输出：最大值
 * @param src 输入：2D tensor，形状为 [first, last]
 * @param shared_tmp_buffer_buffer 临时buffer
 * @param src_shape 输入tensor的形状，src_shape[0]=first, src_shape[1]=last
 * @param src_inner_pad 是否内部padding（未使用）
 *
 * @tparam T 索引类型（必须是int64_t）
 * @tparam U 值类型（float或int32_t）
 * @tparam pattern reduce模式（Pattern::Reduce::AR或Pattern::Reduce::RA）
 *
 * @note 仅支持AICORE平台
 * @note src仅支持float和int32_t数据类型
 * @note 索引类型必须是int64_t
 */
template <typename T, typename U, class pattern>
__aicore__ inline void ArgMaxWithValueExtend(const LocalTensor<T> &dst_index, const LocalTensor<U> &dst_value,
                                             const LocalTensor<U> &src, const LocalTensor<uint8_t> &shared_tmp_buffer,
                                             const uint32_t src_shape[], bool src_inner_pad) {
  if ASCEND_IS_AIC {
    return;
  }
  static_assert(std::is_same_v<U, float> || std::is_same_v<U, int32_t>,
                "ArgmaxWithValue src only support float and int32_t");
  static_assert(std::is_same_v<T, int64_t>, "ArgmaxWithValue dst_index only support int64_t dst type");
  static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(),
                "failed to check Reduce pattern, only support AR/RA pattern!");
  uint32_t first = src_shape[0];
  uint32_t last = src_shape[1];
  if constexpr (SupportType<pattern, Pattern::Reduce::AR>()) {
    if (last <= 64) {
      uint32_t repeat_times = first;
      ArgmaxLEOneRepeatWithValue<U, false>(dst_index, dst_value, src, shared_tmp_buffer, repeat_times, last);
    } else {
      uint32_t repeat_times_one_row = CeilDivision(last, 64);
      ArgmaxGTOneRepeatWithValue<U, Pattern::Reduce::AR, false>(dst_index, dst_value, src, shared_tmp_buffer, first,
                                                                last, repeat_times_one_row);
    }
  } else if constexpr (SupportType<pattern, Pattern::Reduce::RA>()) {
    uint32_t repeat_times_one_row = CeilDivision(last, 64);
    ArgmaxRAWithValue<U, Pattern::Reduce::RA, false>(dst_index, dst_value, src, shared_tmp_buffer, first, last,
                                                     repeat_times_one_row);
  }
}

}  // namespace AscendC

#endif  // __ASCENDC_API_ARGMAX_WITH_VALUE_H__
