/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_TYPES_H__
#define __SK_TYPES_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include <limits>
#include <vector>
#include <memory>
#include <new>

#include "securec.h"
#include "acl/acl.h"
#include "sk_common.h"
#include "sk_log.h"
#include "aprof_pub.h"

constexpr uint32_t K_TYPE_AICORE = 1;
constexpr uint32_t K_TYPE_AIC = 2;
constexpr uint32_t K_TYPE_AIV = 3;
constexpr uint32_t K_TYPE_MIX_AIC_MAIN = 4;
constexpr uint32_t K_TYPE_MIX_AIV_MAIN = 5;
constexpr uint32_t K_TYPE_AIC_ROLLBACK = 6;
constexpr uint32_t K_TYPE_AIV_ROLLBACK = 7;

// Super Kernel Configuration Constants
constexpr uint32_t TASK_QUE_EXPAND_FACTOR = 2;  // TaskQue扩容因子

constexpr uint32_t MAX_SCOPE_NUM = 1024;
constexpr uint32_t MAX_FUNC_NAME_LEN = 4096;
constexpr size_t SK_TOTAL_UB_SIZE = 256U * 1024U;
constexpr uint16_t INVALID_SCOPE_ID = std::numeric_limits<uint16_t>::max();
constexpr uint32_t INVALID_STREAM_ID = std::numeric_limits<uint32_t>::max();
constexpr uint64_t INVALID_TASK_ID = std::numeric_limits<uint64_t>::max();
constexpr char UNKNOWN_MODEL_ID[] = "model_unknown";

class TaskQuePtr {
  std::unique_ptr<uint8_t[]> data_;
  TaskQue *taskQue_;
  size_t capacity_;

 public:
  bool Init(size_t cap) {
    data_.reset();
    capacity_ = 0;
    taskQue_ = nullptr;
    if (cap > (std::numeric_limits<size_t>::max() - sizeof(TaskQue)) / sizeof(TaskInfo)) {
      SK_LOGE("TaskQuePtr size overflow: requested cap=%zu, max size=%zu, sizeof(TaskInfo)=%zu", cap,
              std::numeric_limits<size_t>::max(), sizeof(TaskInfo));
      return false;
    }
    size_t size = sizeof(TaskQue) + sizeof(TaskInfo) * cap;
    std::unique_ptr<uint8_t[]> newData = std::make_unique<uint8_t[]>(size);
    TaskQue *newTaskQue = reinterpret_cast<TaskQue *>(newData.get());
    newTaskQue->cap = static_cast<uint32_t>(cap);
    newTaskQue->taskCnt = 0;
    data_ = std::move(newData);
    taskQue_ = newTaskQue;
    capacity_ = cap;
    return true;
  }

  bool Expand() {
    if (taskQue_ == nullptr) {
      return false;
    }

    size_t extendSize = static_cast<size_t>(taskQue_->cap) * static_cast<size_t>(TASK_QUE_EXPAND_FACTOR);
    if (extendSize <= taskQue_->cap || extendSize > std::numeric_limits<uint32_t>::max()) {
      SK_LOGE("TaskQuePtr expand cap overflow: current cap=%u, expansion factor=%u, max uint32=%u", taskQue_->cap,
              TASK_QUE_EXPAND_FACTOR, std::numeric_limits<uint32_t>::max());
      return false;
    }
    if (extendSize > (std::numeric_limits<size_t>::max() - sizeof(TaskQue)) / sizeof(TaskInfo)) {
      SK_LOGE("TaskQuePtr expand size overflow: extendSize=%zu, max size=%zu, sizeof(TaskInfo)=%zu", extendSize,
              std::numeric_limits<size_t>::max(), sizeof(TaskInfo));
      return false;
    }
    size_t newSize = sizeof(TaskQue) + sizeof(TaskInfo) * extendSize;

    std::unique_ptr<uint8_t[]> newData = std::make_unique<uint8_t[]>(newSize);

    TaskQue *newTaskQue = reinterpret_cast<TaskQue *>(newData.get());

    size_t oldSize = sizeof(TaskQue) + sizeof(TaskInfo) * taskQue_->cap;
    errno_t err = memcpy_s(newData.get(), newSize, data_.get(), oldSize);
    if (err != 0) {
      SK_LOGE("TaskQuePtr expand memcpy failed: errno=%d, srcSize=%zu, dstSize=%zu", err, oldSize, newSize);
      return false;
    }

    newTaskQue->cap = static_cast<uint32_t>(extendSize);
    data_ = std::move(newData);
    taskQue_ = newTaskQue;
    capacity_ = extendSize;
    return true;
  }

  TaskQue *get() const {
    return taskQue_;
  }

  TaskQuePtr() noexcept : data_(nullptr), taskQue_(nullptr), capacity_(0) {}

  TaskQuePtr(TaskQuePtr &&other) noexcept
      : data_(std::move(other.data_)), taskQue_(other.taskQue_), capacity_(other.capacity_) {
    other.taskQue_ = nullptr;
    other.capacity_ = 0;
  }

  TaskQuePtr &operator=(TaskQuePtr &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      taskQue_ = other.taskQue_;
      capacity_ = other.capacity_;
      other.taskQue_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }

  TaskQuePtr(const TaskQuePtr &) = delete;
  TaskQuePtr &operator=(const TaskQuePtr &) = delete;
};

class DeviceArgsPtr {
  std::unique_ptr<uint8_t[]> data_;
  SkDeviceEntryArgs *args_;

 public:
  bool Init(size_t size) {
    data_.reset();
    args_ = nullptr;
    if (size == 0) {
      return false;
    }
    std::unique_ptr<uint8_t[]> newData = std::make_unique<uint8_t[]>(size);
    args_ = reinterpret_cast<SkDeviceEntryArgs *>(newData.get());
    data_ = std::move(newData);
    return true;
  }

  SkDeviceEntryArgs *Get() const {
    return args_;
  }

  DeviceArgsPtr() noexcept : data_(nullptr), args_(nullptr) {}

  DeviceArgsPtr(DeviceArgsPtr &&other) noexcept : data_(std::move(other.data_)), args_(other.args_) {
    other.args_ = nullptr;
  }

  DeviceArgsPtr &operator=(DeviceArgsPtr &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      args_ = other.args_;
      other.args_ = nullptr;
    }
    return *this;
  }

  DeviceArgsPtr(const DeviceArgsPtr &) = delete;
  DeviceArgsPtr &operator=(const DeviceArgsPtr &) = delete;
};

struct SkTask {
  uint32_t numBlocks;
  TaskQuePtr taskQue;
  uint32_t funcCnt;
  SkKernelType nodeType = SkKernelType::DEFAULT;

  bool Init(size_t cap) {
    return taskQue.Init(cap);
  }

  TaskQue *GetTaskQue() {
    TaskQue *queue = taskQue.get();
    if (queue == nullptr) {
      return nullptr;
    }

    if (queue->taskCnt >= queue->cap) {
      if (!taskQue.Expand()) {
        return nullptr;
      }
      queue = taskQue.get();
    }

    return (queue != nullptr && queue->taskCnt < queue->cap) ? queue : nullptr;
  }

  const TaskQue *GetTaskQue() const {
    return taskQue.get();
  }

  size_t GetTaskQueSize() const {
    const TaskQue *queue = taskQue.get();
    if (queue == nullptr) {
      return 0;
    }
    return sizeof(TaskQue) + queue->taskCnt * sizeof(TaskInfo);
  }

  SkTask() : numBlocks(0), funcCnt(0) {};
  SkTask(const SkTask &) = delete;
  SkTask &operator=(const SkTask &) = delete;
  SkTask(SkTask &&) = default;
  SkTask &operator=(SkTask &&) = default;
};

struct SkHostEntryInfo {
  uint32_t numBlocks;
  uint32_t nodeCnt;
  SkKernelType entryType;
  aclrtFuncHandle skEntryFunc;

  SkHostEntryInfo() : numBlocks(0), nodeCnt(0), entryType(SkKernelType::DEFAULT), skEntryFunc(nullptr) {}

  SkHostEntryInfo(const SkHostEntryInfo &) = delete;
  SkHostEntryInfo &operator=(const SkHostEntryInfo &) = delete;

  SkHostEntryInfo(SkHostEntryInfo &&other) noexcept
      : numBlocks(other.numBlocks), nodeCnt(other.nodeCnt), entryType(other.entryType), skEntryFunc(other.skEntryFunc) {
    other.skEntryFunc = nullptr;
    other.numBlocks = 0;
    other.nodeCnt = 0;
    other.entryType = SkKernelType::DEFAULT;
  }

  SkHostEntryInfo &operator=(SkHostEntryInfo &&other) noexcept {
    if (this != &other) {
      numBlocks = other.numBlocks;
      nodeCnt = other.nodeCnt;
      entryType = other.entryType;
      skEntryFunc = other.skEntryFunc;
      other.skEntryFunc = nullptr;
      other.numBlocks = 0;
      other.nodeCnt = 0;
      other.entryType = SkKernelType::DEFAULT;
    }
    return *this;
  }

  ~SkHostEntryInfo() = default;
};

struct CacheopInfoBasic {
  uint32_t taskType;     // 算子的任务类型
  uint32_t numBlocks;    // blockdim
  uint64_t nodeId;       // 算子名的hashid
  uint64_t opType;       // 算子类型的hashid
  uint64_t attrId{0};    // 本次attr拼接放这里
  uint64_t reserve2{0};  // 不做处理
  uint32_t opFlag;       // 记录op属性标记的bitmap，bit0代表是否使能了HF32
  uint32_t tensorNum;    // tensor个数
  MsrofTensorData tensorData[0];
};

struct SkLaunchInfo {
  SkHostEntryInfo entryInfo;
  DeviceArgsPtr devArgs;
  void *
      cacheInfo;  // sk融合算子的shape信息，由sk_optimizer.cpp在构建launchInfo时填充，实际类型是CacheopInfoBasic，包含一个可变长度的tensorData数组
  size_t cacheopInfoSize;
  size_t skMaxDcacheSize = 0;          // 所有 SIMT 子任务共同允许的最大 DCache 大小
  bool useSimtEntry = false;           // 是否选择 SIMT entry 并配置对应的动态 UBUF
  bool isScheModeOn = false;           // 子算子或框架全核同步需要 batch 调度
  void *eventGmAddr;                   // 事件记录 GM 地址
  uint64_t modelIdIndex{0};            // modelId index registered on host
  uint32_t skId;                       // SK 标识
  std::string skFuncName = "Unknown";  // SK function name for profiling and debugging
};

inline const char *TaskTypeToString(aclmdlRITaskType type) {
  switch (type) {
    case ACL_MODEL_RI_TASK_KERNEL:
      return "KERNEL";
    case ACL_MODEL_RI_TASK_VALUE_WRITE:
      return "VALUE_WRITE";
    case ACL_MODEL_RI_TASK_VALUE_WAIT:
      return "VALUE_WAIT";
    case ACL_MODEL_RI_TASK_EVENT_RECORD:
      return "NOTIFY";
    case ACL_MODEL_RI_TASK_EVENT_WAIT:
      return "WAIT";
    case ACL_MODEL_RI_TASK_EVENT_RESET:
      return "RESET";
    case ACL_MODEL_RI_TASK_DEFAULT:
      return "DEFAULT";
    default:
      return "UNKNOWN";
  }
}

#endif  // __SK_TYPES_H__
