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
 * \file sk_dfx_exception_handler.h
 * \brief
 */

#ifndef __SK_DFX_EXCEPTION_HANDLER_H__
#define __SK_DFX_EXCEPTION_HANDLER_H__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>
#include <set>
#include <string.h>
#include "acl/acl.h"
#include "sk_common.h"
#include "runtime/rt_external_kernel.h"
#include "dump/adump_pub.h"

namespace sk {
namespace logger {
class LogContextGuard;
}
}  // namespace sk

struct ExceptionRegInfo {
  uint32_t coreNum;
  rtExceptionErrRegInfo_t *errRegInfo;
};

struct KernelFuncName {
  std::string name;
};

class SuperKernelExceptionHandler {
 public:
  SuperKernelExceptionHandler();
  ~SuperKernelExceptionHandler() = default;

  void HandleException(aclrtExceptionInfo *exceptionInfo);

  aclError FillExceptionDumpInfo(Adx::ExceptionDumpInfo &dumpInfo, aclrtExceptionInfo *exceptionInfo);

  int32_t GetErrorNodeIdx() const {
    return errorNodeIdx_;
  }

  bool GetSubKernelTaskArgs(uint32_t nodeIdx, uint64_t &argsAddr, uint32_t &argsSize);

  uint32_t ProcessExceptionDump(aclrtExceptionInfo *exceptionInfo, Adx::ExceptionDumpInfo *exceptionDumpInfo,
                                uint32_t exceptionDumpSize, uint32_t *exceptionDumpRealSize,
                                Adx::ExceptionDumpMode *mode);

 private:
  bool IsSuperKernelException(aclrtExceptionInfo *exceptionInfo);
  bool ExtractSkEntryArgs(aclrtExceptionInfo *exceptionInfo, std::unique_ptr<sk::logger::LogContextGuard> &logContext);
  bool ExtractSkDeviceEntryArgsPtr(aclrtExceptionInfo *exceptionInfo);
  bool CopySkDeviceEntryArgsToHost(std::unique_ptr<sk::logger::LogContextGuard> &logContext);
  bool ExtractSkHeaderInfo();
  bool ValidateSkHeaderOffsets();
  bool ExtractTaskQueue();
  void ExtractAndPrintSkInfo();
  void PrintSkHeaderInfo() const;
  void PrintTaskQueue() const;
  void PrintCounterInfo() const;
  void PrintDfxInfo() const;
  bool ParseAndPrintSubKernelSymbols(aclrtExceptionInfo *exceptionInfo);
  KernelFuncName GetOrLoadKernelSymbols(uint32_t opId);
  struct ErrorNodeMatchInfo {
    uint32_t nodeIdx = 0;
    int entryIdx = -1;
    uint64_t entryAddr = 0;
    uint64_t endAddr = 0;
    uint32_t funcSize = 0;
    const uint64_t *entries = nullptr;
    const SkDfxInfo *dfxNode = nullptr;
  };
  bool FindErrorNodeByPC(rtCoreType_t coreType, uint64_t currentPC, ErrorNodeMatchInfo &matchInfo) const;
  void IdentifyErrorNodeByPC(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC);
  void PrintMatchedNodeBasicInfo(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC,
                                 uint32_t nodeIdx, int entryIdx, uint64_t entryAddr, uint64_t endAddr,
                                 uint32_t funcSize, const SkDfxInfo &dfxNode) const;
  void PrintFuncSymbolInfo(uint32_t coreId, rtCoreType_t coreType, uint32_t nodeIdx, int entryIdx,
                           const uint64_t *entries, const SkDfxInfo &dfxNode);
  void PrintNodeDevArgs(uint32_t coreId, rtCoreType_t coreType, uint32_t nodeIdx) const;
  void PrintNoMatchInfo(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC) const;
  void PrintCoreSymbols(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC);
  void PrintSymbolByCoreId(uint32_t coreId, rtCoreType_t coreType, uint64_t startPC, uint64_t currentPC,
                           const KernelFuncName &kernelFuncName);
  void PrintAllCoreSymbols();
  void FreeResources();

  bool StartsWith(const char *source, const char *prefix);
  bool GetExceptionRegInfo(const aclrtExceptionInfo &exception, ExceptionRegInfo &exceptionRegInfo);
  void ParseAndPrintCondInfo(uint32_t coreId, rtCoreType_t coreType, uint64_t condValue);
  void PrintCondSubKernelInfo(uint32_t coreId, uint64_t condValue);
  static uint64_t GetCondRegValue(const rtExceptionErrRegInfo_t &coreErrRegInfo);

  aclError PopulateDumpInfoFields(Adx::ExceptionDumpInfo &dumpInfo, int32_t errorNodeIdx,
                                  aclrtExceptionInfo *exceptionInfo, uint32_t coreId, rtCoreType_t coreType);

  aclError PrepareExceptionDump(aclrtExceptionInfo *exceptionInfo, ExceptionRegInfo &exceptionRegInfo,
                                std::unique_ptr<sk::logger::LogContextGuard> &logContext);

  aclError PopulateSkEntryFields(Adx::ExceptionDumpInfo &dumpInfo, aclrtExceptionInfo *exceptionInfo);
  void PopulateTensorFields(Adx::ExceptionDumpInfo &dumpInfo, uint32_t coreId, rtCoreType_t coreType);
  bool PopulateSubKernelFields(Adx::ExceptionDumpInfo &dumpInfo, int32_t errorNodeIdx,
                               aclrtExceptionInfo *exceptionInfo);
  uint32_t FillUniqueExceptionDumpInfos(aclrtExceptionInfo *exceptionInfo, const ExceptionRegInfo &exceptionRegInfo,
                                        Adx::ExceptionDumpInfo *exceptionDumpInfo, uint32_t exceptionDumpSize);

  uint32_t aicoreNums;
  std::map<uint32_t, KernelFuncName> opSymbolCache;  // Operator function name cache: opId -> KernelFuncName

  void *skDeviceEntryArgsDev;  // GM address directly obtained from aclrtGetArgsFromExceptionInfo
  uint32_t skDeviceEntryArgsPtrLen;

  SkDeviceEntryArgs *skDeviceEntryArgsHost;  // Host side, complete SkDeviceEntryArgs data copied at once

  SkHeaderInfo *skHeaderInfoHost;  // Points to skHeaderInfo in skDeviceEntryArgsHost

  void *aicTaskQueDevPtr;  // Reserved for recording device address (for debugging)
  void *aivTaskQueDevPtr;  // Reserved for recording device address (for debugging)
  uint32_t aicTaskCnt;
  uint32_t aivTaskCnt;
  std::set<uint32_t> funcNodeIndices_;  // Indices of nodes whose task type is TYPE_FUNC
  bool hasOpTrace_;                     // Whether sk_entry name contains "op_trace"
  int32_t errorNodeIdx_ = -1;
  aclError CheckError(aclError ret, const char *errorMsg);
};

void SuperKernelExceptionCallBackFunc(aclrtExceptionInfo *exceptionInfo);
uint32_t ExceptionDumpInfoCallBack(void *exceptionInfo, Adx::ExceptionDumpInfo *exceptionDumpInfo,
                                   uint32_t exceptionDumpSize, uint32_t *exceptionDumpRealSize,
                                   Adx::ExceptionDumpMode *mode);

#endif  // __SK_DFX_EXCEPTION_HANDLER_H__
