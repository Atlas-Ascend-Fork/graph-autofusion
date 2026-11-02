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
 * \file superkernel_opt_options.h
 * \brief
 */

#ifndef ACL_SUPERKERNEL_H
#define ACL_SUPERKERNEL_H

#include <cstdint>
#include <cstddef>
#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

enum class aclskOptionType : uint32_t {
  PRELOAD_CODE = 0,
  SPLIT_MODE = 1,
  STREAM_FUSION = 2,
  DCCI_DISABLE_ON_KERNEL = 3,
  DEBUG_SYNC_ALL = 4,
  KERNEL_MAP = 5,
  AUTO_OP_PARALLEL = 7,
  DCCI_BEFORE_KERNEL_START = 8,
  DEBUG_OP_EXEC_TRACE = 9,
  DEBUG_CROSS_CORE_SYNC_CHECK = 10,
  OPT_EXTEND_OPTION = 11,
  DEBUG_EXTEND_OPTION = 12,
  DCCI_AFTER_KERNEL_END = 13,
  AGGRESSIVE_OPT_STRATEGIES = 14,
  UBUF_LOCK_IGNORE_KERNEL = 15,
  EARLY_START = 16,                // early start global option
  DEBUG_PER_OP_MAX_CORE_NUM = 17,  // debug: per op max core num limit
  SK_OPTION_MAX
};

enum class SkInnerOptionType : uint32_t { MIX_KERNEL_SPLIT = 0, SIMT_OP_SUPPORT = 1, SK_INNER_OPTION_MAX };

enum aclskPreloadMode : uint32_t {
  ACLSK_PRELOAD_MAX = 0U,                // AIC/AIV use the maximum supported preload count
  ACLSK_PRELOAD_PER_FUNC = 1U,           // use function size, capped by the AIC/AIV limits
  ACLSK_PRELOAD_DISABLED = 2U,           // disable instruction-cache preload
  ACLSK_PRELOAD_MIN = 3U,                // AIC/AIV preload one 2-KB block
  ACLSK_PRELOAD_PER_FUNC_HALF_AIV = 4U,  // AIC uses function size; AIV uses half, rounded up
  ACLSK_PRELOAD_PER_FUNC_MIN_AIV = 5U,   // AIC uses function size; AIV uses at most one block
};

typedef struct aclskPreloadOption {
  uint32_t preloadMode;  // aclskPreloadMode
} aclskPreloadOption;

typedef struct aclskSplitModeOption {
  uint32_t splitCnt;
} aclskSplitModeOption;

typedef struct aclskStreamFusionOption {
  uint32_t streamFusion;
} aclskStreamFusionOption;

typedef struct aclskDcciOption {
  char **kernelNames;
  size_t kernelCnt;
} aclskDcciOption;

typedef struct aclskDebugSyncAllOption {
  uint32_t debugSyncAll;
} aclskDebugSyncAllOption;

typedef struct aclskKernelMap {
  char *globalName;
  char *sknlNames[4];
} aclskKernelMap;

typedef struct aclskKernelMapOption {
  aclskKernelMap *kernelMaps;
  size_t numKernels;
} aclskKernelMapOption;

typedef struct aclskAutoOpParallelOption {
  uint32_t enableAutoOpParallel;
} aclskAutoOpParallelOption;

typedef struct aclskDebugOpExecTraceOption {
  uint32_t enableOpExecTrace;
} aclskDebugOpExecTraceOption;

typedef struct aclskDebugCrossCoreSyncCheckOption {
  uint32_t enableCrossCoreSyncCheck;
} aclskDebugCrossCoreSyncCheckOption;

typedef struct aclskExtendOption {
  char *value;
} aclskExtendOption;

/**
 * aggressiveOpts carries aggressive fusion strategy switches:
 *  - eventBreakerBypass: reserved for event breaker policy
 *  - taskBreakerBypass: enable default-node bypass when set to 1
 *
 * aggressiveOpts.valueBreakerBypass is a bitmask-style value-memory wait policy:
 *  - ACLSK_VALUE_BREAKER_BYPASS_NONE (0b00): reject notify/wait pairing relation, keep wait unfusible
 *  - ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT (0b01): for paired notify+wait, existing rule must pass or SK exits
 *  - ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT (0b10): for waits without notify after rule check, allow wait fusion
 */
enum aclskValueBreakerBypassFlag : uint32_t {
  ACLSK_VALUE_BREAKER_BYPASS_NONE = 0b00U,
  ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT = 0b01U,
  ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT = 0b10U
};

typedef struct aclskAggressiveOptStrategies {
  uint32_t eventBreakerBypass;
  uint32_t valueBreakerBypass;
  uint32_t taskBreakerBypass;
} aclskAggressiveOptStrategies;

/**
 * Configure kernels that should ignore MIX kernel split by name pattern.
 */
typedef struct aclskUbufLockIgnoreKernelOption {
  uint32_t ubufLockIgnoreKernelCnt;
  char **ubufLockIgnoreKernel;
} aclskUbufLockIgnoreKernelOption;

enum aclskEarlyStartValue : uint32_t {
  ACLSK_EARLY_START_DISABLED = 0U,  // 默认值，表示不启用early start
  ACLSK_EARLY_START_ENABLED = 1U,   // 启用early start
};
typedef struct aclskEarlyStartOption {
  uint32_t enableEarlyStart;  // 0[default]: disable early start, 1: enable early start
} aclskEarlyStartOption;

typedef struct aclskDebugPerOpMaxCoreNumOption {
  uint32_t enableDebugPerOpMaxCoreNum;  // 0[default]: disable, 1: enable
} aclskDebugPerOpMaxCoreNumOption;

struct aclskOption {
  aclskOptionType optionType;
  union {
    aclskPreloadOption preload;
    aclskSplitModeOption splitMode;
    aclskStreamFusionOption streamFusion;
    aclskDcciOption disableKernelDcci;
    aclskDebugSyncAllOption debugSync;
    aclskKernelMapOption kernelMap;
    aclskAutoOpParallelOption autoOpParallel;
    aclskDcciOption dcciBeforeKernelStart;
    aclskDebugOpExecTraceOption debugOpExecTrace;
    aclskDebugCrossCoreSyncCheckOption debugCrossCoreSyncCheck;
    aclskExtendOption optExtend;
    aclskExtendOption debugExtend;
    aclskDcciOption dcciAfterKernelEnd;
    aclskAggressiveOptStrategies aggressiveOpts;
    aclskUbufLockIgnoreKernelOption ubufLockIgnoreKernel;
    aclskEarlyStartOption earlyStart;
    aclskDebugPerOpMaxCoreNumOption debugPerOpMaxCoreNum;
  };
};

typedef struct aclskOptions {
  aclskOption *options;
  size_t numOptions;
} aclskOptions;

typedef enum {
  ACLSK_SCOPE_VERIFY_NODE_WAIT = 0,
  ACLSK_SCOPE_VERIFY_NODE_NOTIFY = 1,
  ACLSK_SCOPE_VERIFY_NODE_COMPUTE = 2,
} aclskScopeVerifyNodeType;

typedef enum {
  ACLSK_SCOPE_VERIFY_KERNEL_NO_AICORE = 0,
  ACLSK_SCOPE_VERIFY_KERNEL_CUBE = 1,    // AI CUBE CORE
  ACLSK_SCOPE_VERIFY_KERNEL_VECTOR = 2,  // AI VECTOR CORE
  ACLSK_SCOPE_VERIFY_KERNEL_MIX = 3,
} aclskScopeVerifyKernelType;

typedef struct aclskScopeVerifyNodeInfo {
  int64_t taskId;
  int64_t streamId;
  int64_t eventId;
  int32_t scopeId;
  aclskScopeVerifyNodeType taskType;
  aclskScopeVerifyKernelType kernelType;
  uint32_t numBlocks;
  uint32_t taskRatio[2];
  int32_t scheMode;
  uint32_t flag;  // bit 0：是否动态，0表示静态，1表示动态；其余bit预留
  int32_t coreLimit[2];
  int32_t extendType;  // 当前必须是0
  void *extendInfo;    // 当前必须是nullptr
} aclskScopeVerifyNodeInfo;

typedef struct aclskScopeVerifyGraphInfo {
  aclskScopeVerifyNodeInfo *nodes;
  size_t nodeCount;
  int32_t extendType;  // 当前必须是0
  void *extendInfo;    // 当前必须是nullptr
} aclskScopeVerifyGraphInfo;

typedef enum {
  ACLSK_SCOPE_VERIFY_SPLIT_BEFORE_NODE = 0,  // scope 在 current node 前断开（全核同步）
  ACLSK_SCOPE_VERIFY_SPLIT_EXCLUDE_NODE = 2,  // scope 在 current node 前断开，且current node不参与scope融合（死锁）
} aclskScopeVerifySplitType;

typedef enum {
  ACLSK_SCOPE_VERIFY_DEADLOCK_DETECTED = 0,
  ACLSK_SCOPE_VERIFY_SYNCALL_OP_DROP = 1,
} aclskScopeVerifySplitReason;

typedef struct aclskScopeVerifySplitResult {
  aclskScopeVerifyNodeInfo *splitNode;
  aclskScopeVerifySplitType splitType;
  aclskScopeVerifySplitReason splitReason;
  int32_t extendType;  // 当前必须是0
  void *extendInfo;    // 当前必须是nullptr
} aclskScopeVerifySplitResult;

/**
 * @ingroup AscendCL
 * @brief Optimize model with super kernel
 *
 * @param model [IN]    Model handle to be optimized
 * @param options [IN]  Pointer to optimization options
 *
 * @retval ACL_ERROR_SUCCESS Optimization succeeded
 * @retval ACL_ERROR_INVALID_PARAM Invalid parameters
 * @retval ACL_ERROR_FAILURE Optimization failed
 *
 * @see aclskOptions
 */
ACL_FUNC_VISIBILITY aclError aclskOptimize(aclmdlRI model, aclskOptions *options);
ACL_FUNC_VISIBILITY aclError aclskScopeBegin(const char *scopeName, aclrtStream stream);
ACL_FUNC_VISIBILITY aclError aclskScopeEnd(const char *scopeName, aclrtStream stream);
/**
 * @ingroup AscendCL
 * @brief
 *
 * @param verifyGraph             传入的图
 * @param maxSplitResultCount     传入的splitResults数组的最大长度
 * @param splitResults            导致scope断开的node列表
 * @param realSplitResultCount    导致scope断开的node数量
 * @retval ACL_SUCCESS 接口调用成功
 * @retval ACL_ERROR_INVALID_PARAM 参数校验失败
 * @retval ACL_ERROR_FAILURE 接口调用失败
 */
ACL_FUNC_VISIBILITY aclError aclskScopeVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                                              aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount);

#ifdef __cplusplus
}
#endif
#endif
