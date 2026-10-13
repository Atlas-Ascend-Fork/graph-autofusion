/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <cstring>
#include <cinttypes>
#include <cstdint>
#include "graph/types_af.h"
#include "experiment/mmpa/mmpa_api.h"
#include "acl/acl.h"

template <typename T>
struct MakeUniq {
  using unique_object = std::unique_ptr<T>;
};

template <typename T>
struct MakeUniq<T[]> {
  using unique_array = std::unique_ptr<T[]>;
};

template <typename T, size_t B>
struct MakeUniq<T[B]> {
  struct invalid_type {};
};

template <typename T, typename... Args>
static inline typename MakeUniq<T>::unique_object MakeUnique(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  return std::unique_ptr<T>(new (std::nothrow) T_nc(std::forward<Args>(args)...));
}

template <typename T>
static inline typename MakeUniq<T>::unique_array MakeUnique(const size_t num) {
  return std::unique_ptr<T>(new (std::nothrow) typename std::remove_extent<T>::type[num]());
}

template <typename T, typename... Args>
static inline typename MakeUniq<T>::invalid_type MakeUnique(Args &&...) = delete;

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

struct AutofuseKernelConfig {
  int32_t input_num;
  int32_t output_num;
  std::vector<std::vector<int64_t>> input_shape;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<ge::DataType> input_data_type;
  std::vector<ge::DataType> output_data_type;
};

class AutofuseKernelInfo {
 public:
  AutofuseKernelInfo() {}
  ~AutofuseKernelInfo() {
    Release();
  }
  void Release() {
    (void)mmDlclose(handles_);
    if (workspace_ != nullptr) {
      aclrtFree(workspace_);
      workspace_ = nullptr;
    }
    for (int i = 0; i < kernel_config_.input_num; ++i) {
      if (input_addr_[i] != nullptr) {
        aclrtFree(input_addr_[i]);
        input_addr_[i] = nullptr;
      }
    }
    for (int i = 0; i < kernel_config_.output_num; ++i) {
      if (output_addr_[i] != nullptr) {
        aclrtFree(output_addr_[i]);
        output_addr_[i] = nullptr;
      }
    }
  }
  int32_t Init(void *stream, const std::string &config_path);
  int32_t LoadSoHandles();
  int32_t ParseTaskRunParam();
  int32_t DoTiling(std::unique_ptr<uint8_t[]> &tiling_data_holder, uint64_t &workspace_size);
  int32_t Distribute(uint64_t *tiling_data);
  size_t &GetTilingSize() {
    return tiling_size_;
  }
  int32_t MallocWorkSpace(uint64_t &size);
  int32_t SetGraphNameSnake(const std::string &name_snake) {
    graph_name_snake_ = name_snake;
  }
  AutofuseKernelConfig &GetKernelConfig() {
    return kernel_config_;
  }

 private:
  int32_t InitDeviceData();
  int32_t InitKernelConfig(const std::string &config_file_path);
  int32_t SetKernelConfig(std::map<std::string, std::string> &src_kernel_config);

 private:
  uint32_t block_dim_{0U};
  size_t tiling_size_{0UL};
  std::string graph_name_snake_ = "fused_graph_0";
  void *handles_{nullptr};
  void *stream_{nullptr};
  void *workspace_{nullptr};
  AutofuseKernelConfig kernel_config_;
  std::unique_ptr<void *[]> input_addr_;
  std::unique_ptr<void *[]> output_addr_;
};
