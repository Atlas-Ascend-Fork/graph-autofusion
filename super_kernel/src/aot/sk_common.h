/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_common.h
 * \brief
 */

#ifndef SK_COMMON_H
#define SK_COMMON_H
#include <array>
#include <string>
#include <cstddef>
#include <cstdint>
#include <acl/acl.h>

// Forward declaration for aclmdlRI
typedef void *aclmdlRI;

enum class SkKernelArch : uint32_t {
  UNKNOWN = 0,
  DAV_2201,
  DAV_3510,
};

inline const char *to_string(SkKernelArch arch) {
  switch (arch) {
    case SkKernelArch::DAV_2201:
      return "DAV_2201";
    case SkKernelArch::DAV_3510:
      return "DAV_3510";
    default:
      return "UNKNOWN";
  }
}

constexpr std::array<SkKernelArch, 2> SK_SUPPORTED_KERNEL_ARCHS = {
    SkKernelArch::DAV_2201,
    SkKernelArch::DAV_3510,
};

constexpr uint32_t SK_EVENT_DAV_2201_CORE_NUM = 75;
constexpr uint32_t SK_EVENT_DAV_3510_CORE_NUM = 108;
constexpr uint32_t SK_DAV_2201_TICK_US_MULTIPLIER = 50;
constexpr uint32_t SK_DAV_3510_TICK_US_MULTIPLIER = 1000;

constexpr uint32_t MAX_SCOPE_NAME_LEN = 256;
constexpr char DEFAULT_SK_SCOPE_NAME[] = "default_sk_scope_name";

struct ScopeKernelArgs {
  char name[MAX_SCOPE_NAME_LEN] = "";
};

struct SkRuntimeConfig {
  SkKernelArch kernelArch = SkKernelArch::DAV_2201;
  uint32_t eventCoreNum = SK_EVENT_DAV_2201_CORE_NUM;
  uint32_t tickUsMultiplier = SK_DAV_2201_TICK_US_MULTIPLIER;
};

void InitSkRuntimeConfig();
const SkRuntimeConfig &GetSkRuntimeConfig();
SkKernelArch GetCurrentSkKernelArch();
const char *GetSkKernelArchSymbolSuffix(SkKernelArch arch);

enum class SkNodeType : uint32_t {
  NODE_KERNEL = 0,
  NODE_NOTIFY = 1,
  NODE_WAIT = 2,
  NODE_RESET = 3,
  NODE_MEMORY_WRITE = 4,
  NODE_MEMORY_WAIT = 5,
  NODE_DEFAULT = 6,
};

inline const char *to_string(SkNodeType type) {
  switch (type) {
    case SkNodeType::NODE_KERNEL:
      return "KERNEL";
    case SkNodeType::NODE_NOTIFY:
      return "NOTIFY";
    case SkNodeType::NODE_WAIT:
      return "WAIT";
    case SkNodeType::NODE_RESET:
      return "RESET";
    case SkNodeType::NODE_MEMORY_WRITE:
      return "MEMORY_WRITE";
    case SkNodeType::NODE_MEMORY_WAIT:
      return "MEMORY_WAIT";
    case SkNodeType::NODE_DEFAULT:
      return "DEFAULT";
    default:
      return "UNKNOWN";
  }
}

enum class SkKernelType : uint8_t {
  AIC_ONLY = 1,
  AIV_ONLY = 2,
  MIX_AIV_1_0 = 3,
  MIX_AIC_1_0 = 4,
  MIX_AIC_1_1 = 5,
  MIX_AIC_1_2 = 6,
  DEFAULT = 0xFF,
};

constexpr size_t SK_KERNEL_TYPE_COUNT = 6;

inline const char *to_string(SkKernelType type) {
  switch (type) {
    case SkKernelType::AIC_ONLY:
      return "AIC_ONLY";
    case SkKernelType::AIV_ONLY:
      return "AIV_ONLY";
    case SkKernelType::MIX_AIV_1_0:
      return "MIX_AIV_1_0";
    case SkKernelType::MIX_AIC_1_0:
      return "MIX_AIC_1_0";
    case SkKernelType::MIX_AIC_1_1:
      return "MIX_1_1";
    case SkKernelType::MIX_AIC_1_2:
      return "MIX_1_2";
    case SkKernelType::DEFAULT:
      return "DEFAULT";
    default:
      return "UNKNOWN";
  }
}

enum class SkTaskType : uint8_t {
  TYPE_FUNC,
  TYPE_SYNC,
  TYPE_PRELOAD,
  TYPE_EVENT_NOTIFY,
  TYPE_EVENT_WAIT,
  TYPE_EVENT_RESET,
  TYPE_MAX,
};

enum class SkOpTraceType : uint8_t {
  ORIGIN = 0,
  SK_ENTRY_LAUNCHED,
  OP_LAUNCHED,
  OP_FINISHED,
  SK_ENTRY_FINISHED,
};

inline const char *to_string(SkTaskType type) {
  switch (type) {
    case SkTaskType::TYPE_FUNC:
      return "FUNC";
    case SkTaskType::TYPE_SYNC:
      return "SYNC";
    case SkTaskType::TYPE_PRELOAD:
      return "PRELOAD";
    case SkTaskType::TYPE_EVENT_NOTIFY:
      return "EVENT_NOTIFY";
    case SkTaskType::TYPE_EVENT_WAIT:
      return "EVENT_WAIT";
    case SkTaskType::TYPE_EVENT_RESET:
      return "EVENT_RESET";
    default:
      return "UNKNOWN";
  }
}

enum class SkCoreSyncType : uint8_t {
  ALL_SYNC = 0,
  CROSS_SYNC_AIC_TO_AIC,
  CROSS_SYNC_AIV_TO_AIV,
  INTER_SYNC_SET_AIC_TO_AIV,
  INTER_SYNC_SET_AIV_TO_AIC,
  INTER_SYNC_WAIT_AIC_TO_AIV,
  INTER_SYNC_WAIT_AIV_TO_AIC,
  SYNC_NONE,
};

inline const char *to_string(SkCoreSyncType type) {
  switch (type) {
    case SkCoreSyncType::ALL_SYNC:
      return "ALL_SYNC";
    case SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC:
      return "AIC_TO_AIC";
    case SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV:
      return "AIV_TO_AIV";
    case SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV:
      return "SET_AIC_TO_AIV";
    case SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC:
      return "SET_AIV_TO_AIC";
    case SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV:
      return "WAIT_AIC_TO_AIV";
    case SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC:
      return "WAIT_AIV_TO_AIC";
    case SkCoreSyncType::SYNC_NONE:
      return "SYNC_NONE";
    default:
      return "UNKNOWN";
  }
}

// Memory wait rule for VALUE_WAIT tasks.
// The numeric values must stay aligned with runtime's aclrtValueWait flag definition.
enum class SkMemoryWaitFlag : uint32_t {
  GEQ = 0x0,  // Wait until *addr >= value
  EQ = 0x1,   // Wait until *addr == value
  AND = 0x2,  // Wait until (*addr & value) != 0
  NOR = 0x3,  // Wait until ~(*addr | value) != 0
};

inline const char *to_string(SkMemoryWaitFlag flag) {
  switch (flag) {
    case SkMemoryWaitFlag::GEQ:
      return "GEQ";
    case SkMemoryWaitFlag::EQ:
      return "EQ";
    case SkMemoryWaitFlag::AND:
      return "AND";
    case SkMemoryWaitFlag::NOR:
      return "NOR";
    default:
      return "UNKNOWN";
  }
}

constexpr uint64_t SK_DEFAULT_NOTIFY_VALUE = 1;
constexpr uint64_t SK_DEFAULT_WAIT_VALUE = 1;
constexpr uint64_t SK_DEFAULT_RESET_VALUE = 0;
constexpr uint32_t SK_DEFAULT_WRITE_FLAG = 0;

struct TaskInfo {
  uint32_t index;
  SkTaskType type;
  SkKernelType relatedType = SkKernelType::DEFAULT;
  uint8_t numBlocks;
  uint8_t entryCnt;
  uint64_t entry[4];
  // 根据bit位确定option选项
  // 1： dcci_disable_on_kernel
  // 2： debug_sync_all
  // 4： dcci_before_kernel_start
  // 8： dcci_after_kernel_end
  // 16： debug_cross_core_sync_check
  // 32： enable_dcci_after_func - 直接指示kernel是否需要在func执行后插入dcci
  //      该bit由host端根据disableDcci和afterKernelEnd综合计算得出
  //      kernel侧只需检查此bit即可，无需组合判断
  uint64_t debugOptions;
  uint64_t extraInfo;
  uint64_t args;
  uint32_t argsSize;
  uint8_t isSimtKernel;
  uint8_t reservedList[3];
};

inline void SetEventTaskArgs(TaskInfo &taskInfo, uint64_t addr, uint64_t value, uint32_t flag) {
  taskInfo.args = addr;
  taskInfo.entry[0] = value;
  taskInfo.extraInfo = static_cast<uint64_t>(flag);
}

inline uint64_t GetEventTaskAddr(const TaskInfo &taskInfo) {
  return taskInfo.args;
}

enum class SkEarlyStartMask : uint32_t {
  NONE = 0U,
  AIC_TO_AIC_SET = 1U << 0,
  AIC_TO_AIC_WAIT = 1U << 1,
  AIC_TO_AIV_SET = 1U << 2,
  AIV_TO_AIC_WAIT = 1U << 3,
  AIV_TO_AIV_SET = 1U << 4,
  AIV_TO_AIV_WAIT = 1U << 5,
  AIV_TO_AIC_SET = 1U << 6,
  AIC_TO_AIV_WAIT = 1U << 7,
  SPLIT_CORE_CTRL = 1U << 15,
};

inline const char *to_string(SkEarlyStartMask mask) {
  switch (mask) {
    case SkEarlyStartMask::NONE:
      return "NONE";
    case SkEarlyStartMask::AIC_TO_AIC_SET:
      return "AIC_TO_AIC_SET";
    case SkEarlyStartMask::AIC_TO_AIC_WAIT:
      return "AIC_TO_AIC_WAIT";
    case SkEarlyStartMask::AIC_TO_AIV_SET:
      return "AIC_TO_AIV_SET";
    case SkEarlyStartMask::AIV_TO_AIC_WAIT:
      return "AIV_TO_AIC_WAIT";
    case SkEarlyStartMask::AIV_TO_AIV_SET:
      return "AIV_TO_AIV_SET";
    case SkEarlyStartMask::AIV_TO_AIV_WAIT:
      return "AIV_TO_AIV_WAIT";
    case SkEarlyStartMask::AIV_TO_AIC_SET:
      return "AIV_TO_AIC_SET";
    case SkEarlyStartMask::AIC_TO_AIV_WAIT:
      return "AIC_TO_AIV_WAIT";
    case SkEarlyStartMask::SPLIT_CORE_CTRL:
      return "SPLIT_CORE_CTRL";
    default:
      return "UNKNOWN";
  }
}

inline uint64_t GetEventTaskValue(const TaskInfo &taskInfo) {
  return taskInfo.entry[0];
}

inline uint32_t GetEventTaskFlag(const TaskInfo &taskInfo) {
  return static_cast<uint32_t>(taskInfo.extraInfo);
}

struct TaskQue {
  uint32_t taskCnt;
  uint32_t cap;
  TaskInfo taskInfos[0];
};

struct SkHeaderInfo {
  uint32_t aicQueSize;
  uint32_t aivQueSize;
  uint32_t aicQueOffset;
  uint32_t aivQueOffset;
  uint32_t counterOffset;
  uint32_t dfxOffset;
  uint32_t eventConfigOffset;  // 算子打印事件配置偏移量
  uint32_t nodeCnt;
  uint64_t modelIdIndexAndSkScopeId;
  uint64_t totalSize;
};

struct SkCounterInfo {
  uint32_t index;
  uint8_t opState;  // operator trace state, see SkOpTraceType
  // dcci single cacheline size is 64bytes
  uint8_t reserve[59];
};

struct SkDfxInfo {
  uint64_t binHdl;
  uint64_t funcHdlOri;
  uint32_t aicSize;
  uint32_t aivSize;
  uint64_t entryAic[4];
  uint64_t entryAiv[4];
  uint32_t numBlocks;         // 算子所需的 block 数量
  uint32_t cubeNum;           // 算子所需的 cube core 数量
  uint32_t vecNum;            // 算子所需的 vec core 数量
  uint32_t padding;           // 保留对齐
  uint64_t aicFuncOffset[4];  // AIC function offset within bin for each split
  uint64_t aivFuncOffset[4];  // AIV function offset within bin for each split
};

struct SkDeviceEntryArgs {
  SkHeaderInfo skHeader;
  uint8_t data[0];
};

// ==================== 事件记录相关结构体 ====================
// Kernel 侧的时间记录结构体
struct SkKernelEventRecord {
  uint64_t modelIdIndex;  // modelId index registered on host
  uint32_t skId;          // SK 标识
  uint32_t nodeId;        // 算子节点 ID
  uint8_t blockIdx;       // block 索引
  uint8_t blockNum;
  uint64_t startTime;  // 开始时间戳
  uint64_t endTime;    // 结束时间戳
};

// 每个 core 的缓冲区头部（Kernel 侧）
struct SkKernelEventCoreBuf {
  uint32_t offset;    // 当前写入偏移
  uint32_t reserved;  // 保留字段
};

// 事件记录配置信息（放在 SkHeaderInfo 的 dfxOffset 位置）
struct SkEventConfig {
  uint64_t eventGmAddr;   // 事件记录 GM 基地址
  uint64_t modelIdIndex;  // modelId index registered on host
  uint32_t skId;          // SK 标识
  uint8_t enabled;        // 是否启用
  uint32_t coreSize;      // 每个core的缓冲区大小（字节）
};

bool GetFuncSymbolInfo(aclrtBinHandle binHdl, const char *binAddr, size_t binSize, uint64_t funcAddr,
                       std::string &symbolName, uint64_t &funcSize, std::string &symbolBind);

/*!
 * \brief Get current device SoC name from ACL runtime
 * \return SoC name string, or empty string when runtime returns nullptr
 */
std::string GetSocName();

enum class ScheModeState : uint8_t {
  SCHE_MODE_OFF = 0,
  SCHE_MODE_ON = 1,
  NONE = 0xff,
};

// ==================== Device Core Number Utilities ====================

aclError GetDeviceCoreNums(int64_t &cubeNum, int64_t &vecNum);

#endif
