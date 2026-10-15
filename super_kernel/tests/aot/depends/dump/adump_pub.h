/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file adump_pub.h
 * @brief Stub header for dump/adump_pub.h - provides types for exception dump info
 */

#ifndef ADUMP_PUB_H
#define ADUMP_PUB_H

#include <cstdint>
#include <vector>

namespace Adx {

enum class TensorType : int32_t { INPUT, OUTPUT, WORKSPACE };

enum class AddressType : int32_t { TRADITIONAL, NOTILING, RAW };

enum TensorPlacement : int32_t {
  kOnDeviceHbm,  ///< Tensor位于Device上的HBM内存
  kOnHost,       ///< Tensor位于Host
  kFollowing,    ///< Tensor位于Host，且数据紧跟在结构体后面
  kOnDeviceP2p,  ///< Tensor位于Device上的P2p内存
  kTensorPlacementEnd
};

struct TensorInfo {
  TensorType type;    // tensor类型
  size_t tensorSize;  // tensor内存大小
  int32_t format;
  int32_t dataType;
  int64_t *tensorAddr;   // tensor数据地址
  AddressType addrType;  // 地址的类型
  int32_t placement;
  uint32_t argsOffSet;               // tensor数据地址在args里的偏移
  std::vector<int64_t> shape;        // shape
  std::vector<int64_t> originShape;  // originShape
};

constexpr uint32_t MAX_KERNELNAME_LEN = 1024U;
constexpr uint32_t EXCEPTION_DUMP_MAX_TENSOR_NUM = 128U;

struct ExceptionDumpInfo {
  uint32_t coreId;    // 异常核心ID
  uint32_t coreType;  // 核心类型(AIC=0/AIV=1)
  uint32_t argSize;
  void *argAddr;
  void *bin;
  char kernelName[MAX_KERNELNAME_LEN];
  char kernelDisplayName[MAX_KERNELNAME_LEN];
  uint32_t extraTensorNum;
  Adx::TensorInfo extraTensor[EXCEPTION_DUMP_MAX_TENSOR_NUM];
};
enum class ExceptionDumpMode : uint32_t { DUMP_MODE_NONE = 0, DUMP_MODE_OVERWRITE = 1, DUMP_MODE_ADDITIONAL = 2 };

using ExceptionDumpCallback = uint32_t (*)(void *exceptionInfo, ExceptionDumpInfo *exceptionDumpInfo,
                                           uint32_t exceptionDumpSize, uint32_t *exceptionDumpRealSize,
                                           ExceptionDumpMode *mode);

__attribute__((weak)) int32_t AdumpRegExceptionDumpCallback(ExceptionDumpCallback callback);

}  // namespace Adx

#endif  // ADUMP_PUB_H
