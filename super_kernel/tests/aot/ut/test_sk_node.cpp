/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include <memory>
#include <cstring>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <sys/stat.h>
#include <unistd.h>
#include <bitset>
#include <limits>

#define private public
#define protected public
#include "sk_graph.h"
#include "sk_node.h"
#include "sk_common.h"
#include "sk_options_manager.h"
#include "ut_common_stubs.h"
#include "runtime/kernel.h"
#include "securec.h"
#include "sk_lock_detector.h"

// Forward declaration for InitSuperKernelBindMap (defined in sk_dump_json.cpp)
SkBindMap InitSuperKernelBindMap(aclrtBinHandle binHdl);

namespace {

struct TestRITask {
  uint32_t taskId;
  aclmdlRITaskType type;
  aclmdlRITaskParams params;
};

std::unique_ptr<aclmdlRITask> MakeTaskHandle(TestRITask &task) {
  return std::make_unique<aclmdlRITask>(reinterpret_cast<aclmdlRITask>(&task));
}

void SetSimtOpSupportCapability(SuperKernelOptionsManager &options, uint32_t value) {
  options.innerOptionMap[SkInnerOptionType::SIMT_OP_SUPPORT]->SetValue(value);
}

}  // namespace

class SkNodeTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    sk::logger::FileLogger::Instance().SetEnabled(false);
  }

  void TearDown() override {
    GlobalMockObject::verify();
    sk::logger::FileLogger::Instance().SetEnabled(false);
    SkUtResetTestControls();
  }
};

struct JudgeTaskKernelInfo {
  bool isBegin = true;
  bool isEnd = false;
  bool isPlaceholder = false;
  bool isFuseEnable = true;
  std::unique_ptr<char[]> scopeName;
};

extern bool GetScopeKernelInfo(aclmdlRIKernelTaskParams params, JudgeTaskKernelInfo *info);
extern bool DumpSingleKernelBinary(const KernelInfos &kernelInfo, const std::string &kernelBinsDir);
aclError FakeAclrtGetFunctionNameFailure(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name);

TEST_F(SkNodeTest, GetScopeKernelInfo_GetFunctionName_Failed) {
  aclmdlRIKernelTaskParams params{};
  params.funcHandle = nullptr;
  JudgeTaskKernelInfo info;
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameFailure));
  bool ret = GetScopeKernelInfo(params, &info);
  EXPECT_EQ(ret, false);
}

int Fake_aclrtGetFunctionNameBeginDav2201(void *funcHandle, size_t size, char *name) {
  const char *src = "sk_scope_kernel_begin_dav_2201";
  snprintf_s(name, size, size, "%s", src);
  return 0;
}

int Fake_aclrtGetFunctionNameEndDav3510(void *funcHandle, size_t size, char *name) {
  const char *src = "sk_scope_kernel_end_dav_3510";
  snprintf_s(name, size, size, "%s", src);
  return 0;
}

int Fake_aclrtGetFunctionNamePlaceholderDav2201(void *funcHandle, size_t size, char *name) {
  const char *src = "sk_placeholder_kernel_dav_2201";
  snprintf_s(name, size, size, "%s", src);
  return 0;
}

int Fake_aclrtGetFunctionNameBeginWithoutSuffix(void *funcHandle, size_t size, char *name) {
  const char *src = "sk_scope_kernel_begin";
  snprintf_s(name, size, size, "%s", src);
  return 0;
}

int Fake_aclrtMemcpy(void *dst, size_t dstSize, const void *src, size_t count, aclrtMemcpyKind kind) {
  ScopeKernelArgs fakeArgs;
  snprintf_s(fakeArgs.name, sizeof(fakeArgs.name), sizeof(fakeArgs.name), "%s", DEFAULT_SK_SCOPE_NAME);
  fakeArgs.name[MAX_SCOPE_NAME_LEN - 1] = '\0';
  memcpy_s(dst, sizeof(ScopeKernelArgs), &fakeArgs, sizeof(ScopeKernelArgs));
  return 0;
}

int Fake_aclrtMemcpyFailed(void *dst, size_t dstSize, const void *src, size_t count, aclrtMemcpyKind kind) {
  return ACL_ERROR_FAILURE;
}

TEST_F(SkNodeTest, GetScopeKernelInfo_CopyScopeArgsFailed) {
  aclmdlRIKernelTaskParams params{};
  params.funcHandle = nullptr;
  JudgeTaskKernelInfo info;
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameBeginDav2201));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpyFailed));
  EXPECT_FALSE(GetScopeKernelInfo(params, &info));
}

TEST_F(SkNodeTest, GetScopeKernelInfo_Normal_ScopeName) {
  aclmdlRIKernelTaskParams params{};
  params.funcHandle = nullptr;
  JudgeTaskKernelInfo info;
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameBeginDav2201));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy));
  bool ret = GetScopeKernelInfo(params, &info);
  EXPECT_EQ(ret, true);
  EXPECT_TRUE(info.isBegin);
  EXPECT_FALSE(info.isEnd);
  EXPECT_FALSE(info.isPlaceholder);
}

TEST_F(SkNodeTest, GetScopeKernelInfo_Dav3510EndScopeName) {
  aclmdlRIKernelTaskParams params{};
  params.funcHandle = nullptr;
  JudgeTaskKernelInfo info;
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameEndDav3510));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy));
  bool ret = GetScopeKernelInfo(params, &info);
  EXPECT_EQ(ret, true);
  EXPECT_FALSE(info.isBegin);
  EXPECT_TRUE(info.isEnd);
  EXPECT_FALSE(info.isPlaceholder);
}

TEST_F(SkNodeTest, GetScopeKernelInfo_PlaceholderScopeName) {
  aclmdlRIKernelTaskParams params{};
  params.funcHandle = nullptr;
  JudgeTaskKernelInfo info;
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNamePlaceholderDav2201));
  MOCKER(aclrtMemcpy).stubs().will(invoke(Fake_aclrtMemcpy));
  bool ret = GetScopeKernelInfo(params, &info);
  EXPECT_EQ(ret, true);
  EXPECT_FALSE(info.isBegin);
  EXPECT_FALSE(info.isEnd);
  EXPECT_TRUE(info.isPlaceholder);
}

TEST_F(SkNodeTest, GetScopeKernelInfo_WithoutArchSuffix_ReturnsRegularKernelInfo) {
  aclmdlRIKernelTaskParams params{};
  params.funcHandle = nullptr;
  JudgeTaskKernelInfo info;
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(Fake_aclrtGetFunctionNameBeginWithoutSuffix));
  bool ret = GetScopeKernelInfo(params, &info);
  EXPECT_TRUE(ret);
  EXPECT_FALSE(info.isBegin);
  EXPECT_FALSE(info.isEnd);
  EXPECT_FALSE(info.isPlaceholder);
}

namespace {

struct UtSkNodeRITaskInternal {
  uint32_t taskId;
  aclmdlRITaskType type;
  aclmdlRITaskParams params;
};

std::unique_ptr<aclmdlRITask> MakeOriginTask(UtSkNodeRITaskInternal &task) {
  return std::make_unique<aclmdlRITask>(reinterpret_cast<aclmdlRITask>(&task));
}

aclError FakeAclrtGetFunctionNameRegular(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  const char *src = "regular_kernel";
  return snprintf_s(name, maxLen, maxLen, "%s", src) < 0 ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

static const std::string &MaxLengthKernelName() {
  static const std::string name = [] {
    const std::string prefix = "regular_kernel_";
    return prefix + std::string(MAX_FUNC_NAME_LEN - 1U - prefix.size(), 'x');
  }();
  return name;
}

aclError FakeAclrtGetFunctionName_MaxLengthKernel(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  const std::string &functionName = MaxLengthKernelName();
  if (name == nullptr || maxLen <= functionName.size()) {
    return ACL_ERROR_INVALID_PARAM;
  }
  errno_t ret = memcpy_s(name, maxLen, functionName.c_str(), functionName.size() + 1U);
  return ret == EOK ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

aclError FakeAclrtGetFunctionNameIgnoredMix(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  const char *src = "IgnoredMix";
  return snprintf_s(name, maxLen, maxLen, "%s", src) < 0 ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

aclError FakeAclrtGetFunctionAttributeMix11(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType,
                                            int64_t *attrValue) {
  (void)funcHandle;
  if (attrValue == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_TYPE) {
    *attrValue = ACL_KERNEL_TYPE_MIX;
    return ACL_SUCCESS;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_RATIO) {
    *attrValue = 1;
    return ACL_SUCCESS;
  }
  *attrValue = 0;
  return ACL_SUCCESS;
}

aclError FakeAclrtFunctionGetBinaryNonNull(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle) {
  (void)funcHandle;
  if (binHandle == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *binHandle = reinterpret_cast<aclrtBinHandle>(0x5000);
  return ACL_SUCCESS;
}

aclError FakeAclrtFunctionGetBinaryForSkNodeCap(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle) {
  if (binHandle == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (funcHandle == reinterpret_cast<aclrtFuncHandle>(0x6101)) {
    *binHandle = reinterpret_cast<aclrtBinHandle>(0x5101);
  } else {
    *binHandle = reinterpret_cast<aclrtBinHandle>(0x5102);
  }
  return ACL_SUCCESS;
}

aclError FakeAclrtFunctionGetBinaryForBindmapReason(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle) {
  if (binHandle == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (funcHandle == reinterpret_cast<aclrtFuncHandle>(0x6103)) {
    *binHandle = reinterpret_cast<aclrtBinHandle>(0x5103);
  } else if (funcHandle == reinterpret_cast<aclrtFuncHandle>(0x6104)) {
    *binHandle = reinterpret_cast<aclrtBinHandle>(0x5104);
  } else if (funcHandle == reinterpret_cast<aclrtFuncHandle>(0x6105)) {
    *binHandle = reinterpret_cast<aclrtBinHandle>(0x5105);
  } else {
    *binHandle = reinterpret_cast<aclrtBinHandle>(funcHandle);
  }
  return ACL_SUCCESS;
}

int FakeRtGetBinBufferForNodeDump(void *binHdl, int addrType, void **buffer, uint32_t *size) {
  (void)binHdl;
  (void)addrType;
  static const char kBinData[] = "node-binary";
  if (buffer == nullptr || size == nullptr) {
    return -1;
  }
  *buffer = const_cast<char *>(kBinData);
  *size = sizeof(kBinData);
  return 0;
}

int FakeRtGetBinBufferFailureForNodeDump(void *binHdl, int addrType, void **buffer, uint32_t *size) {
  (void)binHdl;
  (void)addrType;
  if (buffer != nullptr) {
    *buffer = nullptr;
  }
  if (size != nullptr) {
    *size = 0;
  }
  return -1;
}

aclError FakeAclrtBinaryGetDevAddressForNodeDump(aclrtBinHandle binHdl, void **devAddr, size_t *devSize) {
  (void)binHdl;
  if (devAddr == nullptr || devSize == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *devAddr = reinterpret_cast<void *>(0xABC000);
  *devSize = 0x1000;
  return ACL_SUCCESS;
}

aclError FakeAclrtBinaryGetDevAddressFailureForNodeDump(aclrtBinHandle binHdl, void **devAddr, size_t *devSize) {
  (void)binHdl;
  (void)devAddr;
  (void)devSize;
  return ACL_ERROR_FAILURE;
}

struct __attribute__((packed)) UtSkNodeSknlValuePayload {
  uint32_t res;
  SknlMapInfo info;
};

constexpr uint64_t UT_SK_NODE_BIN_DEV_ADDR = 0xABC000;
constexpr uint64_t UT_SK_NODE_AIC_GLOBAL_OFFSET = 0x100;
constexpr uint64_t UT_SK_NODE_AIV_GLOBAL_OFFSET = 0x200;

int FakeRtBinaryGetMetaNumTwoEntriesForSkNode(void *binHdl, int typeEnum, size_t *metaNum) {
  (void)binHdl;
  (void)typeEnum;
  if (metaNum == nullptr) {
    return -1;
  }
  *metaNum = 2;
  return 0;
}

void FillSkNodeBindPayloads(UtSkNodeSknlValuePayload (&payloads)[2], uint64_t aicCap, uint64_t aivCap) {
  payloads[0].info.cap = aicCap;
  payloads[0].info.globalFunc = reinterpret_cast<void *>(UT_SK_NODE_AIC_GLOBAL_OFFSET);
  payloads[0].info.sknlFunc[0] = reinterpret_cast<void *>(0x1000);
  payloads[0].info.sknlFunc[1] = reinterpret_cast<void *>(0x1100);
  payloads[0].info.sknlFunc[2] = reinterpret_cast<void *>(0x1200);
  payloads[0].info.sknlFunc[3] = reinterpret_cast<void *>(0x1300);

  payloads[1].info.cap = aivCap;
  payloads[1].info.globalFunc = reinterpret_cast<void *>(UT_SK_NODE_AIV_GLOBAL_OFFSET);
  payloads[1].info.sknlFunc[0] = reinterpret_cast<void *>(0x2000);
  payloads[1].info.sknlFunc[1] = reinterpret_cast<void *>(0x2100);
  payloads[1].info.sknlFunc[2] = reinterpret_cast<void *>(0x2200);
  payloads[1].info.sknlFunc[3] = reinterpret_cast<void *>(0x2300);
}

void CopySkNodeBindPayloads(void **dataList, size_t *sizeList, const UtSkNodeSknlValuePayload (&payloads)[2],
                            size_t metaNum) {
  for (size_t i = 0; i < metaNum && i < 2; ++i) {
    *static_cast<UtSkNodeSknlValuePayload *>(dataList[i]) = payloads[i];
    sizeList[i] = sizeof(UtSkNodeSknlValuePayload);
  }
}

int FakeRtBinaryGetMetaInfoSameCapForSkNode(void *binHdl, int typeEnum, size_t metaNum, void **dataList,
                                            size_t *sizeList) {
  (void)binHdl;
  (void)typeEnum;
  UtSkNodeSknlValuePayload payloads[2]{};
  FillSkNodeBindPayloads(payloads, 4, 4);
  CopySkNodeBindPayloads(dataList, sizeList, payloads, metaNum);
  return 0;
}

int FakeRtBinaryGetMetaInfoDifferentCapForSkNode(void *binHdl, int typeEnum, size_t metaNum, void **dataList,
                                                 size_t *sizeList) {
  (void)binHdl;
  (void)typeEnum;
  UtSkNodeSknlValuePayload payloads[2]{};
  FillSkNodeBindPayloads(payloads, 4, 8);
  CopySkNodeBindPayloads(dataList, sizeList, payloads, metaNum);
  return 0;
}

int FakeRtBinaryGetMetaInfoConflictForSkNode(void *binHdl, int typeEnum, size_t metaNum, void **dataList,
                                             size_t *sizeList) {
  (void)binHdl;
  (void)typeEnum;
  UtSkNodeSknlValuePayload payloads[2]{};
  FillSkNodeBindPayloads(payloads, 4, 4);
  payloads[1].info.globalFunc = reinterpret_cast<void *>(UT_SK_NODE_AIC_GLOBAL_OFFSET);
  payloads[1].info.sknlFunc[0] = reinterpret_cast<void *>(0x3000);
  CopySkNodeBindPayloads(dataList, sizeList, payloads, metaNum);
  return 0;
}

int FakeRtGetBinBufferEmptyForSkNode(void *binHdl, int addrType, void **buffer, uint32_t *size) {
  (void)binHdl;
  (void)addrType;
  if (buffer != nullptr) {
    *buffer = nullptr;
  }
  if (size != nullptr) {
    *size = 0;
  }
  return 0;
}

aclError FakeAclrtBinaryGetDevAddressForSkNode(aclrtBinHandle binHdl, void **devAddr, size_t *devSize) {
  (void)binHdl;
  if (devAddr == nullptr || devSize == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *devAddr = reinterpret_cast<void *>(UT_SK_NODE_BIN_DEV_ADDR);
  *devSize = 0x10000;
  return ACL_SUCCESS;
}

aclError FakeAclrtGetFunctionAddrForSkNode(aclrtFuncHandle funcHdl, void **addrAicore, void **addrAiv) {
  (void)funcHdl;
  if (addrAicore == nullptr || addrAiv == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *addrAicore = reinterpret_cast<void *>(UT_SK_NODE_BIN_DEV_ADDR + UT_SK_NODE_AIC_GLOBAL_OFFSET);
  *addrAiv = reinterpret_cast<void *>(UT_SK_NODE_BIN_DEV_ADDR + UT_SK_NODE_AIV_GLOBAL_OFFSET);
  return ACL_SUCCESS;
}

aclError FakeAclrtGetFunctionAddrNotInBindMapForSkNode(aclrtFuncHandle funcHdl, void **addrAicore, void **addrAiv) {
  (void)funcHdl;
  if (addrAicore == nullptr || addrAiv == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *addrAicore = reinterpret_cast<void *>(UT_SK_NODE_BIN_DEV_ADDR + 0x300);
  *addrAiv = reinterpret_cast<void *>(UT_SK_NODE_BIN_DEV_ADDR + 0x400);
  return ACL_SUCCESS;
}

std::string MakeTmpDir(const std::string &suffix) {
  std::string dir = "/tmp/sk_node_ut_" + std::to_string(getpid()) + "_" + suffix;
  mkdir(dir.c_str(), 0755);
  return dir;
}

}  // namespace

TEST_F(SkNodeTest, FusionFailReasonInfo_BindmapDetailsAndStrings) {
  FusionFailReasonInfo noneInfo(BindmapFailReason::NONE);
  EXPECT_EQ(noneInfo.GetBindmapFailReason(), BindmapFailReason::NONE);
  EXPECT_EQ(noneInfo.primary, FusionFailReason::OP_UNSUPPORT);
  EXPECT_EQ(FusionFailReasonToStr(noneInfo), "OP_UNSUPPORT");
  EXPECT_EQ(FusionFailReasonDetailToStr(noneInfo), "Failed to resolve SuperKernel bind map for the operator");
  EXPECT_STREQ(to_string(FusionFailReason::OP_UNSUPPORT), "OP_UNSUPPORT");

  FusionFailReasonInfo info(BindmapFailReason::FUNCHDL_NULL);
  EXPECT_EQ(info.GetBindmapFailReason(), BindmapFailReason::FUNCHDL_NULL);
  EXPECT_EQ(FusionFailReasonToStr(info), "OP_UNSUPPORT [FUNCHDL_NULL]");
  EXPECT_EQ(FusionFailReasonDetailToStr(info),
            "Failed to resolve SuperKernel bind map for the operator, funcHdl is null");

  info.SetBindmapFailReason(BindmapFailReason::FUNC_NOT_FOUND);
  EXPECT_EQ(info.GetBindmapFailReason(), BindmapFailReason::FUNC_NOT_FOUND);
  EXPECT_EQ(FusionFailReasonToStr(info), "OP_UNSUPPORT [FUNC_NOT_FOUND]");
  EXPECT_EQ(FusionFailReasonDetailToStr(info),
            "Failed to resolve SuperKernel bind map for the operator, "
            "function not found in bind map");

  EXPECT_STREQ(to_string(BindmapFailReason::BINDMAP_INIT_EMPTY), "BINDMAP_INIT_EMPTY");
  EXPECT_STREQ(to_string(BindmapFailReason::BINHDL_NULL), "BINHDL_NULL");
  EXPECT_STREQ(to_string(static_cast<BindmapFailReason>(255)), "UNKNOWN_BINDMAP_REASON");
  EXPECT_STREQ(FusionFailReasonInfo(BindmapFailReason::BINHDL_NULL).GetBindmapDetail(), "binHdl is null");
  EXPECT_STREQ(FusionFailReasonInfo(static_cast<BindmapFailReason>(255)).GetBindmapDetail(), "");

  struct BindmapReasonCase {
    BindmapFailReason reason;
    const char *name;
    const char *detail;
  };
  const std::vector<BindmapReasonCase> bindmapCases = {
      {BindmapFailReason::NONE, "NONE", ""},
      {BindmapFailReason::BINDMAP_INIT_EMPTY, "BINDMAP_INIT_EMPTY", "bindmap init empty"},
      {BindmapFailReason::BINHDL_NULL, "BINHDL_NULL", "binHdl is null"},
      {BindmapFailReason::FUNCHDL_NULL, "FUNCHDL_NULL", "funcHdl is null"},
      {BindmapFailReason::FUNC_NOT_FOUND, "FUNC_NOT_FOUND", "function not found in bind map"},
      {BindmapFailReason::BIN_DEV_ADDR_GET_FAILED, "BIN_DEV_ADDR_GET_FAILED", "failed to get binary device address"},
      {BindmapFailReason::FUNC_ADDR_GET_FAILED, "FUNC_ADDR_GET_FAILED", "failed to get function address"},
      {BindmapFailReason::BINDMAP_ENTRY_CONFLICT, "BINDMAP_ENTRY_CONFLICT", "bind map entry conflict"},
      {BindmapFailReason::BINDMAP_CAP_INCONSISTENT, "BINDMAP_CAP_INCONSISTENT", "bind map cap inconsistent"},
      {BindmapFailReason::BIN_HOST_ADDR_GET_FAILED, "BIN_HOST_ADDR_GET_FAILED", "failed to get binary host address"},
  };
  for (const auto &item : bindmapCases) {
    EXPECT_STREQ(to_string(item.reason), item.name);
    EXPECT_STREQ(FusionFailReasonInfo(item.reason).GetBindmapDetail(), item.detail);
  }
}

TEST_F(SkNodeTest, FusionFailReasonInfo_ScopeAndDeadlockDetails) {
  FusionFailReasonInfo scopeInfo(ScopeProcessStatus::RESOURCE_INSUFFICIENT);
  EXPECT_EQ(scopeInfo.primary, FusionFailReason::SCOPE_FUSE_PART);
  EXPECT_EQ(scopeInfo.GetScopeProcessStatus(), ScopeProcessStatus::RESOURCE_INSUFFICIENT);
  EXPECT_STREQ(scopeInfo.GetScopeDetail(), "Insufficient stream task slots or event memory resources");
  EXPECT_EQ(FusionFailReasonToStr(scopeInfo), "SCOPE_FUSE_PART [RESOURCE_INSUFFICIENT]");
  EXPECT_EQ(FusionFailReasonDetailToStr(scopeInfo),
            "scope fuse failed, "
            "Insufficient stream task slots or event memory resources");

  FusionFailReasonInfo deadlockInfo(static_cast<DeadlockFailReason>(1));
  EXPECT_EQ(deadlockInfo.primary, FusionFailReason::EXIST_DEADLOCK);
  EXPECT_EQ(deadlockInfo.GetDeadlockFailReason(), static_cast<DeadlockFailReason>(1));
  EXPECT_EQ(FusionFailReasonToStr(deadlockInfo), "EXIST_DEADLOCK [KERNEL_INSUFFICIENT_CORES]");
  EXPECT_EQ(FusionFailReasonDetailToStr(deadlockInfo),
            "deadlock exists, "
            "The wait node depends on a kernel node that requires more cores than available");
  EXPECT_STREQ(to_string(DeadlockFailReason::NOTIFY_NOT_IN_GRAPH), "NOTIFY_NOT_IN_GRAPH");
  EXPECT_STREQ(to_string(static_cast<DeadlockFailReason>(255)), "UNKNOWN_DEADLOCK_REASON");
  EXPECT_STREQ(FusionFailReasonInfo(DeadlockFailReason::FIRST_WAIT).GetDeadlockDetail(),
               "The wait node is first node in scope");
  EXPECT_STREQ(FusionFailReasonInfo(static_cast<DeadlockFailReason>(255)).GetDeadlockDetail(), "");

  struct DeadlockReasonCase {
    DeadlockFailReason reason;
    const char *name;
    const char *detail;
  };
  const std::vector<DeadlockReasonCase> deadlockCases = {
      {DeadlockFailReason::NOT_FIND_DEADLOCK, "NOT_FIND_DEADLOCK", ""},
      {DeadlockFailReason::KERNEL_INSUFFICIENT_CORES, "KERNEL_INSUFFICIENT_CORES",
       "The wait node depends on a kernel node that requires more cores than available"},
      {DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES, "NOTIFY_INSUFFICIENT_CORES",
       "The wait node depends on a notify node which has fused in other SK that it requires more cores"},
      {DeadlockFailReason::NOTIFY_NOT_IN_GRAPH, "NOTIFY_NOT_IN_GRAPH",
       "The wait node depends on a notify node that it is not in graph"},
      {DeadlockFailReason::NOTIFY_AFTER_SK_RANGE, "NOTIFY_AFTER_SK_RANGE",
       "The wait node depends on a notify node that it is after SK range"},
      {DeadlockFailReason::NOTIFY_INVALID, "NOTIFY_INVALID",
       "In check deadlock of this wait node, get invalid notify node"},
      {DeadlockFailReason::FIRST_WAIT, "FIRST_WAIT", "The wait node is first node in scope"},
      {DeadlockFailReason::NO_SUPPORT_NODE, "NO_SUPPORT_NODE", "The wait node waiting for unsupport node type"},
  };
  for (const auto &item : deadlockCases) {
    EXPECT_STREQ(to_string(item.reason), item.name);
    EXPECT_STREQ(FusionFailReasonInfo(item.reason).GetDeadlockDetail(), item.detail);
  }

  scopeInfo.SetScopeProcessStatus(ScopeProcessStatus::INIT);
  deadlockInfo.SetDeadlockFailReason(static_cast<DeadlockFailReason>(0));
  EXPECT_EQ(scopeInfo.GetScopeProcessStatus(), ScopeProcessStatus::INIT);
  EXPECT_EQ(deadlockInfo.GetDeadlockFailReason(), static_cast<DeadlockFailReason>(0));
}

TEST_F(SkNodeTest, FusionFailReasonStrings_CoverAllEnumNamesAndDetails) {
  struct FusionReasonCase {
    FusionFailReason reason;
    const char *name;
    const char *detail;
  };
  const std::vector<FusionReasonCase> fusionReasonCases = {
      {FusionFailReason::CAN_FUSE, "CAN_FUSE", "node can fuse"},
      {FusionFailReason::OP_UNSUPPORT, "OP_UNSUPPORT", "Failed to resolve SuperKernel bind map for the operator"},
      {FusionFailReason::DYNAMIC_TASK_UNSUPPORT, "DYNAMIC_TASK_UNSUPPORT",
       "The operator will refresh task information at runtime"},
      {FusionFailReason::NOT_IN_SCOPE, "NOT_IN_SCOPE", "The user actively marked that this operator is not fused"},
      {FusionFailReason::IN_UNFUSIBLE_SCOPE, "IN_UNFUSIBLE_SCOPE",
       "This operator is not within the fusion range marked by the user"},
      {FusionFailReason::EXCEED_CORE_MAX, "EXCEED_CORE_MAX",
       "exceeds the maximum number of kernels that the device can provide"},
      {FusionFailReason::RESET_TYPE_NODE, "RESET_TYPE_NODE", "reset type node in end"},
      {FusionFailReason::ISOLATED_EVENT, "ISOLATED_EVENT", "There is no kernel node on the stream"},
      {FusionFailReason::EXIST_DEADLOCK, "EXIST_DEADLOCK", "deadlock exists"},
      {FusionFailReason::SCOPE_FUSE_PART, "SCOPE_FUSE_PART", "scope fuse failed"},
      {FusionFailReason::EXTERNAL_DEPEND, "EXTERNAL_DEPEND", "event node has external dependency"},
      {FusionFailReason::UNSUPPORT_EVENT_TYPE, "UNSUPPORT_EVENT_TYPE", "unsupported event type"},
      {FusionFailReason::MEMORY_WAIT_NODE_ONLY, "MEMORY_WAIT_NODE_ONLY", "No memory write exists"},
      {FusionFailReason::MEMORY_WRITE_NODE_ONLY, "MEMORY_WRITE_NODE_ONLY", "only exists memory write nodes"},
      {FusionFailReason::DEFAULT_NODE, "DEFAULT_NODE", "default node uses aicpu resources"},
      {FusionFailReason::KERNEL_ATTR_GET_FAILED, "KERNEL_ATTR_GET_FAILED",
       "Failed to get kernel attribute for SuperKernel fusion"},
      {FusionFailReason::EXCEED_SCOPE_MAX, "EXCEED_SCOPE_MAX", "Exceeded maximum scope number limit"},
  };
  for (const auto &item : fusionReasonCases) {
    EXPECT_STREQ(to_string(item.reason), item.name);
    FusionFailReasonInfo info(item.reason);
    EXPECT_EQ(FusionFailReasonToStr(info), item.name);
    EXPECT_NE(FusionFailReasonDetailToStr(info).find(item.detail), std::string::npos);
  }

  EXPECT_STREQ(to_string(static_cast<FusionFailReason>(255)), "UNKNOWN_FUSION_FAIL_REASON");
  EXPECT_EQ(FusionFailReasonToStr(FusionFailReasonInfo(static_cast<FusionFailReason>(255))),
            "UNKNOWN_FUSION_FAIL_REASON");
  EXPECT_EQ(FusionFailReasonDetailToStr(FusionFailReasonInfo(static_cast<FusionFailReason>(255))), "UNKNOWN_REASON");
}

TEST_F(SkNodeTest, AlignUpAndClamp_CoversAlignmentAndCoreClamp) {
  EXPECT_EQ(AlignUpAndClamp(1, 0), 1);
  EXPECT_EQ(AlignUpAndClamp(0x801, 0), 2);
  EXPECT_EQ(AlignUpAndClamp(0x800 * 20, 0), 16);
  EXPECT_EQ(AlignUpAndClamp(0x800 * 20, 1), 8);
  EXPECT_EQ(AlignUpAndClamp(0x800 * 20, 2), 20);
}

TEST_F(SkNodeTest, InValidateNode_SetsInvalidatedFlag) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 7;
  task.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  task.params.type = ACL_MODEL_RI_TASK_VALUE_WRITE;

  SuperKernelMemoryNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, 0, INVALID_TASK_ID);
  node.SetNodeId(7);

  EXPECT_FALSE(node.IsInvalidated());
  EXPECT_EQ(node.InValidateNode(), ACL_SUCCESS);
  EXPECT_TRUE(node.IsInvalidated());
}

TEST_F(SkNodeTest, KernelUpdate_CustomParamsStoredSeparatelyForDump) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 8;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3008);
  task.params.kernelTaskParams.numBlocks = 1;

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  node.SetNodeId(8);
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMix11));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));
  ASSERT_TRUE(node.InitNode());

  int value = 0;
  aclmdlRITaskParams custom{};
  custom.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  custom.valueWriteTaskParams.devAddr = &value;
  custom.valueWriteTaskParams.value = 0x1234;

  UpdateContext ctx{};
  ctx.customParams = &custom;
  EXPECT_TRUE(node.Update(ctx));
  EXPECT_TRUE(node.IsUpdated());
  EXPECT_EQ(node.GetTaskParams().type, ACL_MODEL_RI_TASK_KERNEL);
  EXPECT_EQ(node.GetUpdateParams().type, ACL_MODEL_RI_TASK_VALUE_WRITE);
  EXPECT_EQ(node.GetUpdateParams().valueWriteTaskParams.devAddr, &value);
  EXPECT_EQ(node.GetUpdateParams().valueWriteTaskParams.value, 0x1234U);
}

TEST_F(SkNodeTest, KernelInitNode_AcceptsMaximumFunctionName) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 9;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3009);
  task.params.kernelTaskParams.numBlocks = 1;

  ASSERT_EQ(MaxLengthKernelName().size(), MAX_FUNC_NAME_LEN - 1U);
  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionName_MaxLengthKernel));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMix11));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  ASSERT_TRUE(node.InitNode());
  EXPECT_EQ(node.nodeInfos.kernelInfos.funcName, MaxLengthKernelName());
}

TEST_F(SkNodeTest, KernelUpdate_LaunchInfoBuildsBatchAndDynUbufCfg) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 18;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3018);
  task.params.kernelTaskParams.numBlocks = 1;
  task.params.reserved0[0] = 0x12;
  task.params.reserved1[0] = 0x34;
  task.params.kernelTaskParams.rsv[0] = 0x56;

  aclrtLaunchKernelAttr originAttr{};
  originAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  originAttr.value.schemMode = 1;
  aclrtLaunchKernelCfg originCfg{};
  originCfg.attrs = &originAttr;
  originCfg.numAttrs = 1;
  task.params.kernelTaskParams.cfg = &originCfg;

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMix11));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));
  ASSERT_TRUE(node.InitNode());

  SkLaunchInfo launchInfo{};
  launchInfo.entryInfo.skEntryFunc = reinterpret_cast<aclrtFuncHandle>(0x4018);
  launchInfo.entryInfo.numBlocks = 2;
  ASSERT_TRUE(launchInfo.devArgs.Init(sizeof(SkDeviceEntryArgs)));
  launchInfo.devArgs.Get()->skHeader.totalSize = sizeof(SkDeviceEntryArgs);
  launchInfo.useSimtEntry = true;
  launchInfo.isScheModeOn = true;
  launchInfo.skMaxDcacheSize = 32768;
  SetFunctionAllocUbufSize(4096);

  UpdateContext ctx{};
  ctx.launchInfo = &launchInfo;
  EXPECT_TRUE(node.Update(ctx));

  const auto &params = node.GetUpdateParams();
  ASSERT_NE(params.kernelTaskParams.cfg, nullptr);
  ASSERT_EQ(params.kernelTaskParams.cfg->numAttrs, 2U);
  ASSERT_NE(params.kernelTaskParams.cfg->attrs, nullptr);
  EXPECT_EQ(params.kernelTaskParams.cfg->attrs[0].id, ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE);
  EXPECT_EQ(params.kernelTaskParams.cfg->attrs[0].value.schemMode, 1U);
  EXPECT_EQ(params.kernelTaskParams.cfg->attrs[1].id, ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE);
  EXPECT_EQ(params.kernelTaskParams.cfg->attrs[1].value.dynUBufSize, SK_TOTAL_UB_SIZE - 32768U - 4096U);
  EXPECT_EQ(params.reserved0[0], 0U);
  EXPECT_EQ(params.reserved1[0], 0);
  EXPECT_EQ(params.kernelTaskParams.rsv[0], 0U);
  EXPECT_EQ(node.GetTaskParams().kernelTaskParams.cfg, &originCfg);
  EXPECT_EQ(node.GetTaskParams().reserved0[0], 0x12U);
  EXPECT_EQ(node.GetTaskParams().reserved1[0], 0x34U);
  EXPECT_EQ(node.GetTaskParams().kernelTaskParams.rsv[0], 0x56U);
  EXPECT_EQ(task.params.reserved0[0], 0U);
  EXPECT_EQ(task.params.reserved1[0], 0U);
  EXPECT_EQ(task.params.kernelTaskParams.rsv[0], 0U);
  ASSERT_NE(task.params.kernelTaskParams.cfg, nullptr);
  EXPECT_EQ(task.params.kernelTaskParams.cfg->numAttrs, 2U);
}

TEST_F(SkNodeTest, KernelUpdate_LaunchInfoDoesNotInheritOriginParams) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 19;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.reserved0[0] = 0x12;
  task.params.reserved1[0] = 0x34;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3019);
  task.params.kernelTaskParams.numBlocks = 1;
  task.params.kernelTaskParams.rsv[0] = 0x56;
  aclrtLaunchKernelAttr originAttr{};
  originAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  originAttr.value.schemMode = 0;
  aclrtLaunchKernelCfg originCfg{};
  originCfg.attrs = &originAttr;
  originCfg.numAttrs = 1;
  task.params.kernelTaskParams.cfg = &originCfg;

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMix11));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));
  ASSERT_TRUE(node.InitNode());

  SkLaunchInfo launchInfo{};
  launchInfo.entryInfo.skEntryFunc = reinterpret_cast<aclrtFuncHandle>(0x4019);
  launchInfo.entryInfo.numBlocks = 2;
  ASSERT_TRUE(launchInfo.devArgs.Init(sizeof(SkDeviceEntryArgs)));
  launchInfo.devArgs.Get()->skHeader.totalSize = sizeof(SkDeviceEntryArgs);

  UpdateContext ctx{};
  ctx.launchInfo = &launchInfo;
  EXPECT_TRUE(node.Update(ctx));

  const auto &params = node.GetUpdateParams();
  EXPECT_EQ(params.type, ACL_MODEL_RI_TASK_KERNEL);
  EXPECT_EQ(params.taskGrp, nullptr);
  EXPECT_EQ(params.opInfoPtr, nullptr);
  EXPECT_EQ(params.opInfoSize, 0U);
  EXPECT_EQ(params.kernelTaskParams.funcHandle, launchInfo.entryInfo.skEntryFunc);
  EXPECT_EQ(params.kernelTaskParams.args, launchInfo.devArgs.Get());
  EXPECT_EQ(params.kernelTaskParams.argsSize, sizeof(SkDeviceEntryArgs));
  EXPECT_EQ(params.kernelTaskParams.isHostArgs, 1U);
  EXPECT_EQ(params.kernelTaskParams.numBlocks, launchInfo.entryInfo.numBlocks);
  EXPECT_EQ(params.kernelTaskParams.cfg, nullptr);
  EXPECT_EQ(params.reserved0[0], 0U);
  EXPECT_EQ(params.reserved1[0], 0);
  EXPECT_EQ(params.kernelTaskParams.rsv[0], 0U);
  EXPECT_EQ(node.GetTaskParams().kernelTaskParams.cfg, &originCfg);
  EXPECT_EQ(node.GetTaskParams().reserved0[0], 0x12U);
  EXPECT_EQ(node.GetTaskParams().reserved1[0], 0x34U);
  EXPECT_EQ(node.GetTaskParams().kernelTaskParams.rsv[0], 0x56U);
  EXPECT_EQ(task.params.reserved0[0], 0U);
  EXPECT_EQ(task.params.reserved1[0], 0U);
  EXPECT_EQ(task.params.kernelTaskParams.rsv[0], 0U);
  EXPECT_EQ(task.params.kernelTaskParams.cfg, params.kernelTaskParams.cfg);
  EXPECT_EQ(originAttr.value.schemMode, 0U);
}

TEST_F(SkNodeTest, MemoryUpdate_CustomParamsSyncTaskParamsForDump) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 9;
  task.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  task.params.type = ACL_MODEL_RI_TASK_VALUE_WAIT;

  SuperKernelMemoryNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, 0, INVALID_TASK_ID);
  node.SetNodeId(9);

  int value = 0;
  aclmdlRITaskParams custom{};
  custom.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  custom.valueWaitTaskParams.devAddr = &value;
  custom.valueWaitTaskParams.value = 0x5678;
  custom.valueWaitTaskParams.flag = 0x9;

  UpdateContext ctx{};
  ctx.customParams = &custom;
  EXPECT_TRUE(node.Update(ctx));
  EXPECT_TRUE(node.IsUpdated());
  EXPECT_EQ(node.GetTaskParams().type, ACL_MODEL_RI_TASK_VALUE_WAIT);
  EXPECT_EQ(node.GetTaskParams().valueWaitTaskParams.devAddr, &value);
  EXPECT_EQ(node.GetTaskParams().valueWaitTaskParams.value, 0x5678U);
  EXPECT_EQ(node.GetTaskParams().valueWaitTaskParams.flag, 0x9U);
}

TEST_F(SkNodeTest, KernelInitNode_BindmapEmptyReasonIsRecorded) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 10;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2000);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::OP_UNSUPPORT);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::BINDMAP_INIT_EMPTY);
}

TEST_F(SkNodeTest, KernelInitNode_MixSplitFlagHonorsUbufLockIgnoreKernel) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 15;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2015);
  task.params.kernelTaskParams.numBlocks = 4;

  char ignoredMix[] = "IgnoredMix";
  char *ignoredMixKernels[] = {ignoredMix};
  aclskOption option{};
  option.optionType = aclskOptionType::UBUF_LOCK_IGNORE_KERNEL;
  option.ubufLockIgnoreKernel.ubufLockIgnoreKernelCnt = 1;
  option.ubufLockIgnoreKernel.ubufLockIgnoreKernel = ignoredMixKernels;
  SuperKernelOptionsManager opts;
  opts.SetOptOptionValue(&option);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameIgnoredMix));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMix11));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::MIX_AIC_1_1);
  EXPECT_FALSE(node.nodeInfos.kernelInfos.needMixKernelSplit);
}

TEST_F(SkNodeTest, KernelInitNode_NullFuncHandleRecordsBindmapReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 11;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = nullptr;
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::OP_UNSUPPORT);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::FUNCHDL_NULL);
}

TEST_F(SkNodeTest, SetupLaunchKernelCfg_IndependentAttrsDoNotRetainPreviousValues) {
  SuperKernelKernelNode node(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);
  SkLaunchInfo launchInfo{};
  launchInfo.entryInfo.skEntryFunc = reinterpret_cast<aclrtFuncHandle>(0x9000);
  launchInfo.skMaxDcacheSize = 32768;
  SetFunctionAllocUbufSize(4096);
  for (bool useSimtEntry : {true, false}) {
    for (bool scheModeOn : {true, false, true}) {
      launchInfo.useSimtEntry = useSimtEntry;
      launchInfo.isScheModeOn = scheModeOn;
      ASSERT_TRUE(node.SetupLaunchKernelCfg(launchInfo));
      ASSERT_EQ(node.launchKernelCfg_.numAttrs, static_cast<size_t>(useSimtEntry) + scheModeOn);
      if (scheModeOn) {
        EXPECT_EQ(node.launchKernelCfg_.attrs[0].id, ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE);
        EXPECT_EQ(node.launchKernelCfg_.attrs[0].value.schemMode, 1U);
      }
      if (useSimtEntry) {
        const auto &attr = node.launchKernelCfg_.attrs[scheModeOn ? 1 : 0];
        EXPECT_EQ(attr.id, ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE);
        EXPECT_EQ(attr.value.dynUBufSize, SK_TOTAL_UB_SIZE - 32768U - 4096U);
      }
    }
  }
}

TEST_F(SkNodeTest, KernelInitNode_RecordsConsistentCapInKernelInfos) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 13;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x6101);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForSkNodeCap));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmptyForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_TRUE(node.IsFusible());
  EXPECT_EQ(node.nodeInfos.kernelInfos.cap, 4U);
  EXPECT_EQ(node.nodeInfos.kernelInfos.bindmapFailReason, BindmapFailReason::NONE);
  EXPECT_EQ(node.nodeInfos.kernelInfos.resolvedNum, K_MAX_SPLIT_BIN_COUNT);
  EXPECT_EQ(node.nodeInfos.kernelInfos.resolvedFuncs[0].funcOffset[0], 0x1000U);
  EXPECT_EQ(node.nodeInfos.kernelInfos.resolvedFuncs[0].funcOffset[1], 0x2000U);
}

TEST_F(SkNodeTest, KernelInitNode_InconsistentCapRecordsBindmapReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 14;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x6102);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForSkNodeCap));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoDifferentCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.nodeInfos.kernelInfos.cap, 0U);
  EXPECT_EQ(node.nodeInfos.kernelInfos.bindmapFailReason, BindmapFailReason::BINDMAP_CAP_INCONSISTENT);
  EXPECT_EQ(node.nodeInfos.kernelInfos.resolvedNum, 0U);
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::OP_UNSUPPORT);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::BINDMAP_CAP_INCONSISTENT);
}

TEST_F(SkNodeTest, KernelInitNode_BindmapEntryConflictRecordsBindmapReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 16;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x6103);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoConflictForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.nodeInfos.kernelInfos.bindmapFailReason, BindmapFailReason::BINDMAP_ENTRY_CONFLICT);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::BINDMAP_ENTRY_CONFLICT);
}

TEST_F(SkNodeTest, KernelInitNode_FunctionNotInBindmapRecordsBindmapReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 17;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x6104);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrNotInBindMapForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.nodeInfos.kernelInfos.bindmapFailReason, BindmapFailReason::FUNC_NOT_FOUND);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::FUNC_NOT_FOUND);
}

TEST_F(SkNodeTest, KernelInitNode_BinHostAddrGetFailedRecordsBindmapReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 18;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x6105);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferFailureForNodeDump));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.nodeInfos.kernelInfos.bindmapFailReason, BindmapFailReason::BIN_HOST_ADDR_GET_FAILED);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::BIN_HOST_ADDR_GET_FAILED);
}

TEST_F(SkNodeTest, KernelInitNode_TaskGroupNotEmptyRecordsReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 12;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.taskGrp = reinterpret_cast<aclrtTaskGrp>(0x6000);
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2000);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::DYNAMIC_TASK_UNSUPPORT);
}

TEST_F(SkNodeTest, DumpKernelBinaries_DisabledLoggerSkipsWork) {
  SuperKernelGraph graph;
  EXPECT_TRUE(DumpKernelBinaries(graph, MakeTmpDir("disabled")));
}

TEST_F(SkNodeTest, DumpSingleKernelBinary_CoversSuccessAndFailurePaths) {
  KernelInfos kernelInfo;
  kernelInfo.binHdl = reinterpret_cast<aclrtBinHandle>(0x7000);
  kernelInfo.funcName = "kernel/name:with*chars";

  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferFailureForNodeDump));
  EXPECT_FALSE(DumpSingleKernelBinary(kernelInfo, MakeTmpDir("bin_fail")));
  GlobalMockObject::verify();

  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferForNodeDump));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressFailureForNodeDump));
  EXPECT_FALSE(DumpSingleKernelBinary(kernelInfo, MakeTmpDir("dev_fail")));
  GlobalMockObject::verify();

  std::string outDir = MakeTmpDir("bin_success");
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferForNodeDump));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForNodeDump));
  EXPECT_TRUE(DumpSingleKernelBinary(kernelInfo, outDir));
}

TEST_F(SkNodeTest, DumpKernelBinaries_EnabledDumpsKernelAndEntryBinary) {
  SuperKernelGraph graph;
  auto node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  node->SetNodeId(13);
  node->SetNodeType(SkNodeType::NODE_KERNEL);
  node->nodeInfos.kernelInfos.binHdl = reinterpret_cast<aclrtBinHandle>(0x8000);
  node->nodeInfos.kernelInfos.funcName = "regular_kernel";
  graph.graphMap[13] = std::move(node);

  sk::logger::FileLogger::Instance().SetEnabled(true);
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferForNodeDump));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForNodeDump));

  EXPECT_TRUE(DumpKernelBinaries(graph, MakeTmpDir("dump_graph")));
}

TEST_F(SkNodeTest, MemoryNodeInitNode_NullOriginTask_ReturnsFalse) {
  SuperKernelMemoryNode node(nullptr, ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

  EXPECT_FALSE(node.InitNode());
}

TEST_F(SkNodeTest, MemoryNodeInitNode_EventRecordInternalFusible) {
  TestRITask task{};
  task.taskId = 101;
  task.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
  task.params.eventRecordTaskParams.event = reinterpret_cast<aclrtEvent>(0x1000);
  task.params.eventRecordTaskParams.eventFlag = 0;
  task.params.eventRecordTaskParams.recordFlag = ACL_EVENT_RECORD_DEFAULT;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);

  ASSERT_TRUE(node.InitNode());
  EXPECT_EQ(node.GetNodeId(), 101U);
  EXPECT_EQ(node.GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_TRUE(node.IsFusible());
  EXPECT_EQ(node.nodeInfos.syncInfos.eventId, 0x1000U);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventFlag, 0U);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventTaskFlag, ACL_EVENT_RECORD_DEFAULT);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryValue, SK_DEFAULT_NOTIFY_VALUE);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryWaitFlag, SK_DEFAULT_WRITE_FLAG);
}

TEST_F(SkNodeTest, MemoryNodeInitNode_EventRecordExternalTaskFlagNotFusible) {
  TestRITask task{};
  task.taskId = 105;
  task.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
  task.params.eventRecordTaskParams.event = reinterpret_cast<aclrtEvent>(0x5000);
  task.params.eventRecordTaskParams.eventFlag = 0;
  task.params.eventRecordTaskParams.recordFlag = ACL_EVENT_RECORD_EXTERNAL;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);

  ASSERT_TRUE(node.InitNode());
  EXPECT_EQ(node.GetNodeType(), SkNodeType::NODE_NOTIFY);
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::EXTERNAL_DEPEND);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventId, 0x5000U);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventFlag, 0U);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventTaskFlag, ACL_EVENT_RECORD_EXTERNAL);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryValue, SK_DEFAULT_NOTIFY_VALUE);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryWaitFlag, SK_DEFAULT_WRITE_FLAG);
}

TEST_F(SkNodeTest, MemoryNodeInitNode_EventWaitExternalNotFusible) {
  TestRITask task{};
  task.taskId = 102;
  task.type = ACL_MODEL_RI_TASK_EVENT_WAIT;
  task.params.eventWaitTaskParams.event = reinterpret_cast<aclrtEvent>(0x2000);
  task.params.eventWaitTaskParams.eventFlag = ACL_EVENT_EXTERNAL;
  task.params.eventWaitTaskParams.waitFlag = ACL_EVENT_WAIT_EXTERNAL;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_EVENT_WAIT, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);

  ASSERT_TRUE(node.InitNode());
  EXPECT_EQ(node.GetNodeType(), SkNodeType::NODE_WAIT);
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.nodeInfos.syncInfos.eventId, 0x2000U);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventFlag, ACL_EVENT_EXTERNAL);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventTaskFlag, ACL_EVENT_WAIT_EXTERNAL);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryValue, SK_DEFAULT_WAIT_VALUE);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryWaitFlag, static_cast<uint32_t>(SkMemoryWaitFlag::EQ));
}

TEST_F(SkNodeTest, MemoryNodeInitNode_EventResetInternalNotFusible) {
  TestRITask task{};
  task.taskId = 103;
  task.type = ACL_MODEL_RI_TASK_EVENT_RESET;
  task.params.eventResetTaskParams.event = reinterpret_cast<aclrtEvent>(0x3000);
  task.params.eventResetTaskParams.eventFlag = 0;
  task.params.eventResetTaskParams.resetFlag = 1;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_EVENT_RESET, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);

  ASSERT_TRUE(node.InitNode());
  EXPECT_EQ(node.GetNodeType(), SkNodeType::NODE_RESET);
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::RESET_TYPE_NODE);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventId, 0x3000U);
  EXPECT_EQ(node.nodeInfos.syncInfos.eventTaskFlag, 1U);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryValue, SK_DEFAULT_RESET_VALUE);
  EXPECT_EQ(node.nodeInfos.syncInfos.memoryWaitFlag, SK_DEFAULT_WRITE_FLAG);
}

TEST_F(SkNodeTest, MemoryNodeInitNode_UnsupportedEventType_ReturnsFalse) {
  TestRITask task{};
  task.taskId = 104;
  task.type = ACL_MODEL_RI_TASK_KERNEL;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, INVALID_STREAM_ID, INVALID_TASK_ID);

  EXPECT_FALSE(node.InitNode());
}

TEST_F(SkNodeTest, MemoryNodeInitNode_ValueWriteAndWait) {
  TestRITask writeTask{};
  writeTask.taskId = 105;
  writeTask.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  writeTask.params.valueWriteTaskParams.devAddr = reinterpret_cast<void *>(0x4000);
  writeTask.params.valueWriteTaskParams.value = 0x55;
  SuperKernelMemoryNode writeNode(MakeTaskHandle(writeTask), ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, INVALID_STREAM_ID,
                                  INVALID_TASK_ID);

  ASSERT_TRUE(writeNode.InitNode());
  EXPECT_EQ(writeNode.GetNodeType(), SkNodeType::NODE_MEMORY_WRITE);
  EXPECT_FALSE(writeNode.IsFusible());
  EXPECT_EQ(writeNode.nodeInfos.syncInfos.eventId, 0x4000U);
  EXPECT_EQ(writeNode.nodeInfos.syncInfos.addrValue, reinterpret_cast<void *>(0x4000));
  EXPECT_EQ(writeNode.nodeInfos.syncInfos.memoryValue, 0x55U);

  TestRITask waitTask{};
  waitTask.taskId = 106;
  waitTask.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  waitTask.params.valueWaitTaskParams.devAddr = reinterpret_cast<void *>(0x5000);
  waitTask.params.valueWaitTaskParams.value = 0x66;
  waitTask.params.valueWaitTaskParams.flag = static_cast<uint32_t>(SkMemoryWaitFlag::AND);
  SuperKernelMemoryNode waitNode(MakeTaskHandle(waitTask), ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, INVALID_STREAM_ID,
                                 INVALID_TASK_ID);

  ASSERT_TRUE(waitNode.InitNode());
  EXPECT_EQ(waitNode.GetNodeType(), SkNodeType::NODE_MEMORY_WAIT);
  EXPECT_FALSE(waitNode.IsFusible());
  EXPECT_EQ(waitNode.nodeInfos.syncInfos.eventId, 0x5000U);
  EXPECT_EQ(waitNode.nodeInfos.syncInfos.addrValue, reinterpret_cast<void *>(0x5000));
  EXPECT_EQ(waitNode.nodeInfos.syncInfos.memoryValue, 0x66U);
  EXPECT_EQ(waitNode.nodeInfos.syncInfos.memoryWaitFlag, static_cast<uint32_t>(SkMemoryWaitFlag::AND));
}

TEST_F(SkNodeTest, MemoryNodeInitNode_ValueWriteNullAddr_ReturnsFalse) {
  TestRITask task{};
  task.taskId = 107;
  task.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  task.params.valueWriteTaskParams.devAddr = nullptr;
  task.params.valueWriteTaskParams.value = 0x55;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);

  EXPECT_FALSE(node.InitNode());
}

TEST_F(SkNodeTest, MemoryNodeInitNode_ValueWaitNullAddr_ReturnsFalse) {
  TestRITask task{};
  task.taskId = 108;
  task.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  task.params.valueWaitTaskParams.devAddr = nullptr;
  task.params.valueWaitTaskParams.value = 0x66;
  task.params.valueWaitTaskParams.flag = static_cast<uint32_t>(SkMemoryWaitFlag::EQ);

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_VALUE_WAIT, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);

  EXPECT_FALSE(node.InitNode());
}

// ==================== GetKernelTypeString Tests ====================

TEST_F(SkNodeTest, GetKernelTypeString_CoversAllKernelTypes) {
  uint32_t taskRatio[2] = {1, 0};

  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_CUBE, taskRatio), "AIC_ONLY");
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_VECTOR, taskRatio), "AIV_ONLY");

  taskRatio[1] = 0;
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "AIC_ONLY");

  taskRatio[1] = 1;
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "MIX_AIC_1_1");

  taskRatio[1] = 2;
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "MIX_AIC_1_2");

  // taskRatio[1] > 2 falls back to DEFAULT
  taskRatio[1] = 3;
  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_MIX, taskRatio), "DEFAULT");

  EXPECT_STREQ(GetKernelTypeString(ACL_KERNEL_TYPE_AICPU, taskRatio), "DEFAULT");
  EXPECT_STREQ(GetKernelTypeString(static_cast<uint32_t>(99), taskRatio), "DEFAULT");
}

// ==================== SuperKernelMemoryNode Corresponsive Tests ====================

TEST_F(SkNodeTest, MemoryNode_CorresponsiveNodeIdsManagement) {
  TestRITask task{};
  task.taskId = 200;
  task.type = ACL_MODEL_RI_TASK_EVENT_RECORD;
  task.params.eventRecordTaskParams.event = reinterpret_cast<aclrtEvent>(0x2000);
  task.params.eventRecordTaskParams.eventFlag = 0;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_EVENT_RECORD, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  auto waitIds = node.GetCorrespondingWaitNodeIds();
  EXPECT_TRUE(waitIds.empty());

  std::vector<uint64_t> newWaitIds = {301, 302, 303};
  node.SetCorrespondingWaitNodeIds(newWaitIds);
  waitIds = node.GetCorrespondingWaitNodeIds();
  EXPECT_EQ(waitIds.size(), 3);
  EXPECT_EQ(waitIds[0], 301);
  EXPECT_EQ(waitIds[1], 302);
  EXPECT_EQ(waitIds[2], 303);

  node.SetCorrespondingNotifyNodeId(100);
  EXPECT_EQ(node.GetCorrespondingNotifyNodeId(), 100U);
}

// ==================== ScopeBitFlags Tests ====================

TEST_F(SkNodeTest, Node_ScopeBitFlagsManagement) {
  TestRITask task{};
  task.taskId = 300;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  std::bitset<MAX_SCOPE_NUM> flags = node.GetScopeBitFlags();
  EXPECT_TRUE(flags.none());

  flags.set(0);
  flags.set(3);
  flags.set(5);
  node.SetScopeBitFlags(flags);
  flags = node.GetScopeBitFlags();
  EXPECT_TRUE(flags.test(0));
  EXPECT_TRUE(flags.test(3));
  EXPECT_TRUE(flags.test(5));

  node.ClearScopeBitFlags();
  flags = node.GetScopeBitFlags();
  EXPECT_TRUE(flags.none());

  node.SetIsScopeNode(true);
  EXPECT_TRUE(node.IsScopeNode());
  // isScopeBegin and isScopeEnd are set based on scopeKernelInfo during InitNode
  // Setting isScopeNode alone doesn't make it a scope begin/end
  EXPECT_FALSE(node.IsScopeBegin());
  EXPECT_FALSE(node.IsScopeEnd());
  EXPECT_FALSE(node.IsScopePlaceholder());
}

// ==================== Node Update After Update Tests ====================

TEST_F(SkNodeTest, KernelNode_UpdateAfterUpdate) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 400;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  // First update should succeed
  aclmdlRITaskParams custom1{};
  custom1.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  custom1.valueWriteTaskParams.devAddr = reinterpret_cast<void *>(0x1000);
  custom1.valueWriteTaskParams.value = 0x1111;

  UpdateContext ctx1{};
  ctx1.customParams = &custom1;
  EXPECT_TRUE(node.Update(ctx1));
  EXPECT_TRUE(node.IsUpdated());

  // Second update should fail (already updated)
  aclmdlRITaskParams custom2{};
  custom2.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  custom2.valueWaitTaskParams.devAddr = reinterpret_cast<void *>(0x2000);
  custom2.valueWaitTaskParams.value = 0x2222;

  UpdateContext ctx2{};
  ctx2.customParams = &custom2;
  EXPECT_FALSE(node.Update(ctx2));
}

TEST_F(SkNodeTest, MemoryNode_UpdateAfterUpdate) {
  TestRITask task{};
  task.taskId = 401;
  task.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  task.params.valueWriteTaskParams.devAddr = reinterpret_cast<void *>(0x1000);
  task.params.valueWriteTaskParams.value = 0x1111;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_VALUE_WRITE, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  aclmdlRITaskParams custom1{};
  custom1.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  custom1.valueWriteTaskParams.devAddr = reinterpret_cast<void *>(0x2000);
  custom1.valueWriteTaskParams.value = 0x2222;

  UpdateContext ctx1{};
  ctx1.customParams = &custom1;
  EXPECT_TRUE(node.Update(ctx1));
  EXPECT_TRUE(node.IsUpdated());

  aclmdlRITaskParams custom2{};
  custom2.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  custom2.valueWaitTaskParams.devAddr = reinterpret_cast<void *>(0x3000);
  custom2.valueWaitTaskParams.value = 0x3333;

  UpdateContext ctx2{};
  ctx2.customParams = &custom2;
  EXPECT_FALSE(node.Update(ctx2));
}

// ==================== Node FusionFailReason Tests ====================

TEST_F(SkNodeTest, Node_FusionFailReasonManagement) {
  TestRITask task{};
  task.taskId = 500;
  task.type = ACL_MODEL_RI_TASK_EVENT_RESET;
  task.params.eventResetTaskParams.event = reinterpret_cast<aclrtEvent>(0x5000);
  task.params.eventResetTaskParams.eventFlag = 0;

  SuperKernelMemoryNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_EVENT_RESET, 0, 0, INVALID_STREAM_ID,
                             INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::RESET_TYPE_NODE);

  node.SetFusionFailReason(ScopeProcessStatus::RESOURCE_INSUFFICIENT);
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::SCOPE_FUSE_PART);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetScopeProcessStatus(), ScopeProcessStatus::RESOURCE_INSUFFICIENT);

  node.SetFusionFailReason(DeadlockFailReason::NOTIFY_NOT_IN_GRAPH);
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::EXIST_DEADLOCK);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetDeadlockFailReason(), DeadlockFailReason::NOTIFY_NOT_IN_GRAPH);

  node.SetFusionFailReason(BindmapFailReason::FUNC_NOT_FOUND);
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::OP_UNSUPPORT);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::FUNC_NOT_FOUND);
}

// ==================== Node Stream and Index Tests ====================

TEST_F(SkNodeTest, Node_StreamAndIndexManagement) {
  TestRITask task{};
  task.taskId = 600;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.numBlocks = 8;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  // Constructor: (nodeIdxInStream, streamIdxInGraph, streamId, preNodeId)
  SuperKernelKernelNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_KERNEL, 3, 2, 5, INVALID_TASK_ID);

  EXPECT_EQ(node.GetStreamId(), 5U);
  EXPECT_EQ(node.GetNodeIdxInStream(), 3U);
  EXPECT_EQ(node.GetStreamIdxInGraph(), 2U);

  EXPECT_EQ(node.GetPreNodeId(), INVALID_TASK_ID);
  node.SetPreNodeId(100);
  EXPECT_EQ(node.GetPreNodeId(), 100U);

  EXPECT_EQ(node.GetNextNodeId(), INVALID_TASK_ID);
  node.SetNextNodeId(200);
  EXPECT_EQ(node.GetNextNodeId(), 200U);
}

// ==================== KernelNode ScheMode Tests ====================

TEST_F(SkNodeTest, KernelNode_GetScheMode) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 700;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  bool scheMode = node.GetScheMode();
  EXPECT_FALSE(scheMode);
}

// ==================== KernelNode NumBlocks VecNum CubeNum Tests ====================

TEST_F(SkNodeTest, KernelNode_NumBlocksVecNumCubeNum) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 800;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.numBlocks = 16;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  node.nodeInfos.kernelInfos.numBlocks = 16;
  node.nodeInfos.kernelInfos.vecNum = 8;
  node.nodeInfos.kernelInfos.cubeNum = 4;

  EXPECT_EQ(node.GetNumBlocks(), 16U);
  EXPECT_EQ(node.GetVecNum(), 8U);
  EXPECT_EQ(node.GetCubeNum(), 4U);
}

// ==================== SuperKernelDefaultNode Tests ====================

TEST_F(SkNodeTest, DefaultNode_BasicOperations) {
  TestRITask task{};
  task.taskId = 900;
  task.type = ACL_MODEL_RI_TASK_DEFAULT;

  SuperKernelDefaultNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_DEFAULT, 0, 0, INVALID_STREAM_ID,
                              INVALID_TASK_ID);

  // InitNode returns true but logs that node cannot be fused
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  // InValidateNode fails for default node
  EXPECT_EQ(node.InValidateNode(), ACL_ERROR_FAILURE);

  aclmdlRITaskParams custom{};
  UpdateContext ctx{};
  ctx.customParams = &custom;
  // Update returns true (base class Update succeeds)
  EXPECT_TRUE(node.Update(ctx));
}

// ==================== Node Visit Status Tests ====================

TEST_F(SkNodeTest, Node_VisitStatusManagement) {
  TestRITask task{};
  task.taskId = 1000;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeTaskHandle(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());

  EXPECT_FALSE(node.IsVisited());
  node.SetVisited(true);
  EXPECT_TRUE(node.IsVisited());
  node.SetVisited(false);
  EXPECT_FALSE(node.IsVisited());
}

// ==================== KernelInfosToJson Tests ====================

TEST_F(SkNodeTest, KernelInfosToJson_CoversBasicFields) {
  KernelInfos info;
  info.kernelType = SkKernelType::AIC_ONLY;
  info.numBlocks = 32;
  info.funcName = "test_kernel";
  info.vecNum = 8;
  info.cubeNum = 16;
  info.cap = 0x1234567890abcdef;

  Json json = KernelInfosToJson(info);
  EXPECT_EQ(json["kernelType"], "AIC_ONLY");
  EXPECT_EQ(json["numBlocks"], 32);
  EXPECT_EQ(json["funcName"], "test_kernel");
  EXPECT_EQ(json["cap"], "0x1234567890abcdef");
  EXPECT_EQ(json["isScheModeOn"], false);
  info.isScheModeOn = true;
  EXPECT_EQ(KernelInfosToJson(info)["isScheModeOn"], true);
  // vecNum and cubeNum are not included in KernelInfosToJson output
}

// ==================== SyncInfosToJson Tests ====================

TEST_F(SkNodeTest, SyncInfosToJson_CoversEventRecordAndWait) {
  SyncInfos syncInfo;
  syncInfo.eventId = 0x1234;
  syncInfo.memoryValue = 0x5678;

  Json recordJson = SyncInfosToJson(syncInfo, SkNodeType::NODE_MEMORY_WRITE);
  EXPECT_EQ(recordJson["eventId"], "0x1234");

  Json waitJson = SyncInfosToJson(syncInfo, SkNodeType::NODE_MEMORY_WAIT);
  EXPECT_EQ(waitJson["eventId"], "0x1234");
}

// ==================== InitSuperKernelBindMap Tests ====================

TEST_F(SkNodeTest, InitSuperKernelBindMap_NullBinHdlReturnsEmpty) {
  SkBindMap bindMap = InitSuperKernelBindMap(nullptr);
  EXPECT_TRUE(bindMap.empty());
}

TEST_F(SkNodeTest, InitSuperKernelBindMap_MetaNumZeroReturnsEmpty) {
  MOCKER(rtBinaryGetMetaNum).stubs().will(returnValue(0));
  aclrtBinHandle binHdl = reinterpret_cast<aclrtBinHandle>(0x9000);
  SkBindMap bindMap = InitSuperKernelBindMap(binHdl);
  EXPECT_TRUE(bindMap.empty());
}

// ==================== NodeInfosToJson Tests ====================

TEST_F(SkNodeTest, NodeInfosToJson_KernelNodeInfos) {
  NodeInfos nodeInfos;
  nodeInfos.kernelInfos.kernelType = SkKernelType::AIC_ONLY;
  nodeInfos.kernelInfos.kernelTypeInt = 1;
  nodeInfos.kernelInfos.numBlocks = 32;
  nodeInfos.kernelInfos.funcName = "test_kernel";
  nodeInfos.kernelInfos.vecNum = 8;
  nodeInfos.kernelInfos.cubeNum = 16;
  nodeInfos.kernelInfos.resolvedNum = 2;
  nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[0] = 0x1000;
  nodeInfos.kernelInfos.resolvedFuncs[0].funcAddr[1] = 0x2000;

  Json nodeInfosJson = NodeInfosToJson(nodeInfos, SkNodeType::NODE_KERNEL);

  EXPECT_TRUE(nodeInfosJson.contains("kernelInfos"));
  EXPECT_TRUE(nodeInfosJson["kernelInfos"].contains("kernelType"));
  EXPECT_EQ(nodeInfosJson["kernelInfos"]["kernelType"], "AIC_ONLY");
  EXPECT_EQ(nodeInfosJson["kernelInfos"]["numBlocks"], 32);
  EXPECT_EQ(nodeInfosJson["kernelInfos"]["funcName"], "test_kernel");
}

TEST_F(SkNodeTest, NodeInfosToJson_MemoryNodeInfos) {
  NodeInfos nodeInfos;
  nodeInfos.kernelInfos.kernelType = SkKernelType::DEFAULT;
  nodeInfos.syncInfos.eventId = 0x1234;
  nodeInfos.syncInfos.addrValue = reinterpret_cast<void *>(0x5678);
  nodeInfos.syncInfos.memoryValue = 0x9abc;
  nodeInfos.syncInfos.memoryWaitFlag = 7;

  Json nodeInfosJson = NodeInfosToJson(nodeInfos, SkNodeType::NODE_MEMORY_WRITE);

  EXPECT_TRUE(nodeInfosJson.contains("kernelInfos"));
  EXPECT_TRUE(nodeInfosJson.contains("syncInfos"));
  EXPECT_TRUE(nodeInfosJson["syncInfos"].contains("eventId"));
}

// ==================== KernelInfosToJson Extended Tests ====================

TEST_F(SkNodeTest, KernelInfosToJson_WithResolvedFuncs) {
  KernelInfos info;
  info.kernelType = SkKernelType::MIX_AIC_1_1;
  info.kernelTypeInt = 3;
  info.numBlocks = 64;
  info.funcName = "mixed_kernel";
  info.cap = 0xabcd;
  info.resolvedNum = 3;
  info.resolvedFuncs[0].funcAddr[0] = 0x1000;
  info.resolvedFuncs[0].funcAddr[1] = 0x2000;
  info.resolvedFuncs[0].prefetchCnt[0] = 10;
  info.resolvedFuncs[0].prefetchCnt[1] = 20;
  info.resolvedFuncs[0].funcOffset[0] = 0x50;
  info.resolvedFuncs[0].funcOffset[1] = 0x60;
  info.resolvedFuncs[0].symbolBind[0] = "GLOBAL";
  info.resolvedFuncs[0].symbolBind[1] = "WEAK";
  info.resolvedFuncs[1].funcAddr[0] = 0x3000;
  info.resolvedFuncs[1].funcAddr[1] = 0x4000;
  info.resolvedFuncs[2].funcAddr[0] = 0x5000;
  info.resolvedFuncs[2].funcAddr[1] = 0x6000;

  Json json = KernelInfosToJson(info);

  EXPECT_EQ(json["kernelType"], "MIX_1_1");
  EXPECT_EQ(json["kernelTypeInt"], 3);
  EXPECT_EQ(json["numBlocks"], 64);
  EXPECT_EQ(json["cap"], "0xabcd");
  EXPECT_EQ(json["resolvedNum"], 3);
  EXPECT_TRUE(json.contains("resolvedFuncs"));
  EXPECT_EQ(json["resolvedFuncs"].size(), 3);

  EXPECT_EQ(json["resolvedFuncs"][0]["funcAddr"][0], "0x1000");
  EXPECT_EQ(json["resolvedFuncs"][0]["funcAddr"][1], "0x2000");
  EXPECT_EQ(json["resolvedFuncs"][0]["prefetchCnt"][0], 10);
  EXPECT_EQ(json["resolvedFuncs"][0]["prefetchCnt"][1], 20);
  EXPECT_EQ(json["resolvedFuncs"][0]["funcOffset"][0], "0x50");
  EXPECT_EQ(json["resolvedFuncs"][0]["funcOffset"][1], "0x60");
  EXPECT_EQ(json["resolvedFuncs"][0]["symbolBind"][0], "GLOBAL");
  EXPECT_EQ(json["resolvedFuncs"][0]["symbolBind"][1], "WEAK");
}

TEST_F(SkNodeTest, KernelInfosToJson_WithTaskRatio) {
  KernelInfos info;
  info.kernelType = SkKernelType::MIX_AIC_1_2;
  info.kernelTypeInt = 4;
  info.taskRatio[0] = 5;
  info.taskRatio[1] = 10;

  Json json = KernelInfosToJson(info);

  EXPECT_EQ(json["kernelType"], "MIX_1_2");
  EXPECT_TRUE(json.contains("taskRatio"));
  EXPECT_EQ(json["taskRatio"].size(), 2);
  EXPECT_EQ(json["taskRatio"][0], 5);
  EXPECT_EQ(json["taskRatio"][1], 10);
}

TEST_F(SkNodeTest, KernelInfosToJson_WithSimtOpFlag) {
  KernelInfos info;
  info.funcName = "simt_kernel";
  info.isSimtOp = true;

  Json json = KernelInfosToJson(info);

  EXPECT_TRUE(json.contains("isSimtOp"));
  EXPECT_EQ(json["isSimtOp"], true);

  info.isSimtOp = false;
  json = KernelInfosToJson(info);
  EXPECT_TRUE(json.contains("isSimtOp"));
  EXPECT_EQ(json["isSimtOp"], false);
}

// ==================== SyncInfosToJson Extended Tests ====================

TEST_F(SkNodeTest, SyncInfosToJson_WithCorrespondingNodes) {
  SyncInfos syncInfo;
  syncInfo.eventId = 0x1234;
  syncInfo.addrValue = reinterpret_cast<void *>(0x5678);
  syncInfo.memoryValue = 0x9abc;
  syncInfo.memoryWaitFlag = 7;
  syncInfo.correspondingWaitNodeIds = {101, 102, 103};
  syncInfo.correspondingResetNodeIds = {201, 202};
  syncInfo.eventFlag = 0x55;
  syncInfo.eventTaskFlag = 1;

  Json notifyJson = SyncInfosToJson(syncInfo, SkNodeType::NODE_NOTIFY);

  EXPECT_EQ(notifyJson["eventId"], "0x1234");
  EXPECT_TRUE(notifyJson.contains("correspondingWaitNodeIds"));
  EXPECT_EQ(notifyJson["correspondingWaitNodeIds"].size(), 3);
  EXPECT_TRUE(notifyJson.contains("correspondingResetNodeIds"));
  EXPECT_EQ(notifyJson["correspondingResetNodeIds"].size(), 2);
  EXPECT_EQ(notifyJson["eventFlag"], "0x55");
  EXPECT_EQ(notifyJson["eventTaskFlag"], 1);
}

TEST_F(SkNodeTest, SyncInfosToJson_WaitNodeWithCorrespondingNotify) {
  SyncInfos syncInfo;
  syncInfo.eventId = 0x1234;
  syncInfo.correspondingNotifyNodeId = 100;
  syncInfo.addrValue = reinterpret_cast<void *>(0x5678);
  syncInfo.memoryValue = 0x9abc;
  syncInfo.memoryWaitFlag = 3;

  Json waitJson = SyncInfosToJson(syncInfo, SkNodeType::NODE_WAIT);

  EXPECT_EQ(waitJson["eventId"], "0x1234");
  EXPECT_TRUE(waitJson.contains("correspondingNotifyNodeId"));
  EXPECT_EQ(waitJson["correspondingNotifyNodeId"], 100);
  EXPECT_EQ(waitJson["memoryWaitFlag"], 3);
}

TEST_F(SkNodeTest, SyncInfosToJson_MemoryWaitNodeWithCorrespondingNotify) {
  SyncInfos syncInfo;
  syncInfo.eventId = 0x2222;
  syncInfo.correspondingNotifyNodeId = 200;
  syncInfo.addrValue = reinterpret_cast<void *>(0x3333);
  syncInfo.memoryValue = 0x4444;
  syncInfo.memoryWaitFlag = 5;

  Json memWaitJson = SyncInfosToJson(syncInfo, SkNodeType::NODE_MEMORY_WAIT);

  EXPECT_EQ(memWaitJson["eventId"], "0x2222");
  EXPECT_TRUE(memWaitJson.contains("correspondingNotifyNodeId"));
  EXPECT_EQ(memWaitJson["correspondingNotifyNodeId"], 200);
}

TEST_F(SkNodeTest, SyncInfosToJson_DefaultValuesFiltered) {
  SyncInfos syncInfo;
  syncInfo.eventId = 0x1234;
  syncInfo.addrValue = reinterpret_cast<void *>(0x5678);
  syncInfo.memoryValue = std::numeric_limits<uint64_t>::max();
  syncInfo.memoryWaitFlag = std::numeric_limits<uint32_t>::max();
  syncInfo.eventFlag = std::numeric_limits<uint32_t>::max();
  syncInfo.eventTaskFlag = std::numeric_limits<uint32_t>::max();
  syncInfo.correspondingWaitNodeIds = {};
  syncInfo.correspondingResetNodeIds = {};

  Json json = SyncInfosToJson(syncInfo, SkNodeType::NODE_NOTIFY);

  EXPECT_TRUE(json.contains("eventId"));
  EXPECT_TRUE(json.contains("addrValue"));
  EXPECT_FALSE(json.contains("memoryValue"));
  EXPECT_FALSE(json.contains("memoryWaitFlag"));
  EXPECT_FALSE(json.contains("eventFlag"));
  EXPECT_FALSE(json.contains("eventTaskFlag"));
  EXPECT_FALSE(json.contains("correspondingWaitNodeIds"));
  EXPECT_FALSE(json.contains("correspondingResetNodeIds"));
}

// ==================== KernelCapBits 结构体测试 ====================

TEST_F(SkNodeTest, KernelCapBits_DefaultValues) {
  KernelCapBits bits;
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_FALSE(bits.disableDcci);
  EXPECT_FALSE(bits.disableScheMode);
  EXPECT_FALSE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, KernelCapBits_SetDisableDcci) {
  KernelCapBits bits;
  bits.disableDcci = true;
  EXPECT_TRUE(bits.disableDcci);
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_FALSE(bits.disableScheMode);
  EXPECT_FALSE(bits.blockDimScaleUp);
}

// ==================== ParseKernelCapBits 函数测试 ====================

TEST_F(SkNodeTest, ParseKernelCapBits_AllBitsZero) {
  KernelCapBits bits = ParseKernelCapBits(0x0);
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_FALSE(bits.disableDcci);
  EXPECT_FALSE(bits.disableScheMode);
  EXPECT_FALSE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, ParseKernelCapBits_Bit0Set) {
  KernelCapBits bits = ParseKernelCapBits(0x1);
  EXPECT_TRUE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_FALSE(bits.disableDcci);
  EXPECT_FALSE(bits.disableScheMode);
  EXPECT_FALSE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, ParseKernelCapBits_Bit1Set) {
  KernelCapBits bits = ParseKernelCapBits(0x2);
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_TRUE(bits.earlyStartSetFlag);
  EXPECT_FALSE(bits.disableDcci);
  EXPECT_FALSE(bits.disableScheMode);
  EXPECT_FALSE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, ParseKernelCapBits_Bit2Set_DisableDcci) {
  KernelCapBits bits = ParseKernelCapBits(0x4);
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_TRUE(bits.disableDcci);
  EXPECT_FALSE(bits.disableScheMode);
  EXPECT_FALSE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, ParseKernelCapBits_Bit3Set_DisableScheMode) {
  KernelCapBits bits = ParseKernelCapBits(0x8);
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_FALSE(bits.disableDcci);
  EXPECT_TRUE(bits.disableScheMode);
  EXPECT_FALSE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, ParseKernelCapBits_Bit4Set_BlockDimScaleUp) {
  KernelCapBits bits = ParseKernelCapBits(0x10);
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_FALSE(bits.disableDcci);
  EXPECT_FALSE(bits.disableScheMode);
  EXPECT_TRUE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, ParseKernelCapBits_MultipleBitsSet) {
  KernelCapBits bits = ParseKernelCapBits(0x1F);
  EXPECT_TRUE(bits.earlyStartWaitFlag);
  EXPECT_TRUE(bits.earlyStartSetFlag);
  EXPECT_TRUE(bits.disableDcci);
  EXPECT_TRUE(bits.disableScheMode);
  EXPECT_TRUE(bits.blockDimScaleUp);
}

TEST_F(SkNodeTest, ParseKernelCapBits_OnlyDisableDcciAndDisableScheMode) {
  KernelCapBits bits = ParseKernelCapBits(0xC);
  EXPECT_FALSE(bits.earlyStartWaitFlag);
  EXPECT_FALSE(bits.earlyStartSetFlag);
  EXPECT_TRUE(bits.disableDcci);
  EXPECT_TRUE(bits.disableScheMode);
}

TEST_F(SkNodeTest, ParseKernelCapBits_LargeValue) {
  KernelCapBits bits = ParseKernelCapBits(0xFFFFFFFFFFFFFFFFULL);
  EXPECT_TRUE(bits.earlyStartWaitFlag);
  EXPECT_TRUE(bits.earlyStartSetFlag);
  EXPECT_TRUE(bits.disableDcci);
  EXPECT_TRUE(bits.disableScheMode);
  EXPECT_TRUE(bits.blockDimScaleUp);
}

class SkNodeScheModeTest : public SkNodeTest, public testing::WithParamInterface<std::tuple<uint64_t, bool>> {
 protected:
  static int GetMetaInfo(void *, int, size_t metaNum, void **dataList, size_t *sizeList) {
    UtSkNodeSknlValuePayload payloads[2]{};
    const uint64_t cap = std::get<0>(GetParam());
    FillSkNodeBindPayloads(payloads, cap, cap);
    CopySkNodeBindPayloads(dataList, sizeList, payloads, metaNum);
    return 0;
  }
};

TEST_P(SkNodeScheModeTest, KernelInitNode_OnlyDisableScheModeOverridesOriginalMode) {
  const uint64_t cap = std::get<0>(GetParam());
  const bool scheModeOn = std::get<1>(GetParam());
  UtSkNodeRITaskInternal task{};
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  // Bind maps are cached by binary handle; isolate each parameter combination.
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x9000 + cap * 2 + scheModeOn);
  task.params.kernelTaskParams.numBlocks = 4;
  aclrtLaunchKernelAttr scheModeAttr{};
  scheModeAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  scheModeAttr.value.schemMode = scheModeOn;
  aclrtLaunchKernelCfg launchCfg{&scheModeAttr, 1};
  task.params.kernelTaskParams.cfg = &launchCfg;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(GetMetaInfo));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmptyForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());
  ASSERT_TRUE(node.IsFusible());
  EXPECT_EQ(node.GetNodeInfos().kernelInfos.cap, cap);
  const bool expectedScheMode = scheModeOn && (cap == 0 || cap == 0x10);
  EXPECT_EQ(node.IsScheModeOn(), expectedScheMode);
  EXPECT_EQ(node.RequiresExactCoreMatch(), scheModeOn && cap == 0);
  EXPECT_EQ(scheModeAttr.value.schemMode, scheModeOn);
}

INSTANTIATE_TEST_SUITE_P(KernelCaps, SkNodeScheModeTest,
                         testing::Combine(testing::Values(0ULL, 0x8ULL, 0x10ULL, 0x18ULL), testing::Bool()));

TEST_F(SkNodeTest, KernelInfos_IsSimtOpFlag) {
  KernelInfos infos;
  infos.isSimtOp = true;
  EXPECT_TRUE(infos.isSimtOp);

  infos.isSimtOp = false;
  EXPECT_FALSE(infos.isSimtOp);
}

TEST_F(SkNodeTest, KernelInfos_FormatWithSimtFlag) {
  KernelInfos infos;
  infos.funcName = "test_kernel";
  infos.isSimtOp = true;
  std::string formatted = infos.Format();
  EXPECT_TRUE(formatted.find("isSimtOp:1") != std::string::npos);

  infos.isSimtOp = false;
  formatted = infos.Format();
  EXPECT_TRUE(formatted.find("isSimtOp") == std::string::npos);
}

TEST_F(SkNodeTest, SimtAivType_SimdOnly) {
  SetSimtAivType(1);
  uint32_t aivType = 0;
  int ret = rtFunctionGetMetaInfo(nullptr, RT_FUNCTION_TYPE_AIV_TYPE_FLAG, &aivType, sizeof(aivType));
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(aivType, 1);
}

TEST_F(SkNodeTest, SimtAivType_SimtVfOnly) {
  SetSimtAivType(3);
  uint32_t aivType = 0;
  int ret = rtFunctionGetMetaInfo(nullptr, RT_FUNCTION_TYPE_AIV_TYPE_FLAG, &aivType, sizeof(aivType));
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(aivType, 3);
}

TEST_F(SkNodeTest, SimtAivType_SimdSimtMix) {
  SetSimtAivType(4);
  uint32_t aivType = 0;
  int ret = rtFunctionGetMetaInfo(nullptr, RT_FUNCTION_TYPE_AIV_TYPE_FLAG, &aivType, sizeof(aivType));
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(aivType, 4);
}

TEST_F(SkNodeTest, BindmapFailReasonStrings_NewReasons) {
  EXPECT_STREQ(to_string(BindmapFailReason::BIN_DEV_ADDR_GET_FAILED), "BIN_DEV_ADDR_GET_FAILED");
  EXPECT_STREQ(to_string(BindmapFailReason::FUNC_ADDR_GET_FAILED), "FUNC_ADDR_GET_FAILED");
  EXPECT_STREQ(to_string(BindmapFailReason::BINDMAP_ENTRY_CONFLICT), "BINDMAP_ENTRY_CONFLICT");
  EXPECT_STREQ(to_string(BindmapFailReason::BINDMAP_CAP_INCONSISTENT), "BINDMAP_CAP_INCONSISTENT");
  EXPECT_STREQ(to_string(BindmapFailReason::BIN_HOST_ADDR_GET_FAILED), "BIN_HOST_ADDR_GET_FAILED");
}

TEST_F(SkNodeTest, FusionFailReasonInfo_BindmapNewReasons) {
  FusionFailReasonInfo info1(BindmapFailReason::BIN_DEV_ADDR_GET_FAILED);
  EXPECT_EQ(info1.GetBindmapFailReason(), BindmapFailReason::BIN_DEV_ADDR_GET_FAILED);
  std::string str1 = FusionFailReasonToStr(info1);
  EXPECT_EQ(str1, "OP_UNSUPPORT [BIN_DEV_ADDR_GET_FAILED]");

  FusionFailReasonInfo info2(BindmapFailReason::FUNC_ADDR_GET_FAILED);
  EXPECT_EQ(info2.GetBindmapFailReason(), BindmapFailReason::FUNC_ADDR_GET_FAILED);
  std::string str2 = FusionFailReasonToStr(info2);
  EXPECT_EQ(str2, "OP_UNSUPPORT [FUNC_ADDR_GET_FAILED]");

  info1.SetBindmapFailReason(BindmapFailReason::FUNC_ADDR_GET_FAILED);
  EXPECT_EQ(info1.GetBindmapFailReason(), BindmapFailReason::FUNC_ADDR_GET_FAILED);
}

TEST_F(SkNodeTest, FusionFailReasonToStr_NewKernelAttrFailed) {
  const char *str = to_string(FusionFailReason::KERNEL_ATTR_GET_FAILED);
  EXPECT_STREQ(str, "KERNEL_ATTR_GET_FAILED");
  FusionFailReasonInfo info(FusionFailReason::KERNEL_ATTR_GET_FAILED);
  EXPECT_EQ(FusionFailReasonToStr(info), "KERNEL_ATTR_GET_FAILED");
  EXPECT_NE(FusionFailReasonDetailToStr(info).find("Failed to get kernel attribute"), std::string::npos);
}

TEST_F(SkNodeTest, FusionFailReasonToStr_ExceedScopeMax) {
  const char *str = to_string(FusionFailReason::EXCEED_SCOPE_MAX);
  EXPECT_STREQ(str, "EXCEED_SCOPE_MAX");
  FusionFailReasonInfo info(FusionFailReason::EXCEED_SCOPE_MAX);
  EXPECT_EQ(FusionFailReasonToStr(info), "EXCEED_SCOPE_MAX");
  EXPECT_NE(FusionFailReasonDetailToStr(info).find("Exceeded maximum scope number limit"), std::string::npos);
}

aclError FakeAclrtBinaryGetDevAddressFailure(aclrtBinHandle binHdl, void **devAddr, size_t *devSize) {
  (void)binHdl;
  (void)devAddr;
  (void)devSize;
  return ACL_ERROR_FAILURE;
}

aclError FakeAclrtFunctionGetBinaryForBindmapBinDevAddrFail(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle) {
  (void)funcHandle;
  if (binHandle == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *binHandle = reinterpret_cast<aclrtBinHandle>(0x5100);
  return ACL_SUCCESS;
}

aclError FakeAclrtFunctionGetBinaryForBindmapFuncAddrFail(aclrtFuncHandle funcHandle, aclrtBinHandle *binHandle) {
  (void)funcHandle;
  if (binHandle == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *binHandle = reinterpret_cast<aclrtBinHandle>(0x5101);
  return ACL_SUCCESS;
}

TEST_F(SkNodeTest, KernelInitNode_BinaryDevAddrGetFailed_RecordsBindmapReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 1100;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3100);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapBinDevAddrFail));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressFailure));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::OP_UNSUPPORT);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::BIN_DEV_ADDR_GET_FAILED);
}

aclError FakeAclrtGetFunctionAddrFailure(aclrtFuncHandle funcHdl, void **addrAicore, void **addrAiv) {
  (void)funcHdl;
  (void)addrAicore;
  (void)addrAiv;
  return ACL_ERROR_FAILURE;
}

TEST_F(SkNodeTest, KernelInitNode_FunctionAddrGetFailed_RecordsBindmapReason) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 1101;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3101);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapFuncAddrFail));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrFailure));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::OP_UNSUPPORT);
  EXPECT_EQ(node.GetFusionFailReasonInfo().GetBindmapFailReason(), BindmapFailReason::FUNC_ADDR_GET_FAILED);
}

aclError FakeAclrtGetFunctionAttributeKernelTypeFailure(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType,
                                                        int64_t *attrValue) {
  (void)funcHandle;
  (void)attrType;
  (void)attrValue;
  return ACL_ERROR_FAILURE;
}

TEST_F(SkNodeTest, KernelInitNode_KernelTypeGetFailed_RecordsKernelAttrFailed) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 1102;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3002);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeKernelTypeFailure));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_FALSE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::KERNEL_ATTR_GET_FAILED);
}

aclError FakeAclrtGetFunctionNameFailure(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
  (void)funcHandle;
  (void)maxLen;
  (void)name;
  return ACL_ERROR_FAILURE;
}

TEST_F(SkNodeTest, KernelInitNode_FunctionNameGetFailed_ReturnsFalse) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 1103;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x3003);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameFailure));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_FALSE(node.InitNode());
  EXPECT_FALSE(node.IsFusible());
}

// ==================== IdentifyAndHandleSimtKernel Tests ====================

aclError FakeAclrtGetFunctionAttributeAicOnly(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType,
                                              int64_t *attrValue) {
  (void)funcHandle;
  if (attrValue == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_TYPE) {
    *attrValue = ACL_KERNEL_TYPE_CUBE;
    return ACL_SUCCESS;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_RATIO) {
    *attrValue = 0;
    return ACL_SUCCESS;
  }
  *attrValue = 0;
  return ACL_SUCCESS;
}

aclError FakeAclrtGetFunctionAttributeAivOnly(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType,
                                              int64_t *attrValue) {
  (void)funcHandle;
  if (attrValue == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_TYPE) {
    *attrValue = ACL_KERNEL_TYPE_VECTOR;
    return ACL_SUCCESS;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_RATIO) {
    *attrValue = 0;
    return ACL_SUCCESS;
  }
  *attrValue = 0;
  return ACL_SUCCESS;
}

aclError FakeAclrtGetFunctionAttributeMixAic10(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType,
                                               int64_t *attrValue) {
  (void)funcHandle;
  if (attrValue == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_TYPE) {
    *attrValue = ACL_KERNEL_TYPE_MIX;
    return ACL_SUCCESS;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_RATIO) {
    *attrValue = 0;
    return ACL_SUCCESS;
  }
  *attrValue = 0;
  return ACL_SUCCESS;
}

aclError FakeAclrtGetFunctionAttributeMix12(aclrtFuncHandle funcHandle, aclrtFuncAttribute attrType,
                                            int64_t *attrValue) {
  (void)funcHandle;
  if (attrValue == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_TYPE) {
    *attrValue = ACL_KERNEL_TYPE_MIX;
    return ACL_SUCCESS;
  }
  if (attrType == ACL_FUNC_ATTR_KERNEL_RATIO) {
    *attrValue = 2;
    return ACL_SUCCESS;
  }
  *attrValue = 0;
  return ACL_SUCCESS;
}

TEST_F(SkNodeTest, KernelInitNode_OddPureVecScheMode_RemainsFusibleForScopePlanning) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2013;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2013);
  task.params.kernelTaskParams.numBlocks = 45;

  aclrtLaunchKernelAttr scheModeAttr{};
  scheModeAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  scheModeAttr.value.schemMode = 1;
  aclrtLaunchKernelCfg launchCfg{&scheModeAttr, 1};
  task.params.kernelTaskParams.cfg = &launchCfg;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());
  EXPECT_TRUE(node.IsScheModeOn());
  EXPECT_EQ(node.GetCubeNum(), 0U);
  EXPECT_EQ(node.GetVecNum(), 45U);
  EXPECT_TRUE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SkNodeTest, KernelInitNode_EvenPureVecScheMode_RemainsFusible) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2014;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2014);
  task.params.kernelTaskParams.numBlocks = 44;

  aclrtLaunchKernelAttr scheModeAttr{};
  scheModeAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  scheModeAttr.value.schemMode = 1;
  aclrtLaunchKernelCfg launchCfg{&scheModeAttr, 1};
  task.params.kernelTaskParams.cfg = &launchCfg;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  ASSERT_TRUE(node.InitNode());
  EXPECT_TRUE(node.IsScheModeOn());
  EXPECT_EQ(node.GetCubeNum(), 0U);
  EXPECT_EQ(node.GetVecNum(), 44U);
  EXPECT_TRUE(node.IsFusible());
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_NullOptions_SkipCheck) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2001;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2001);
  task.params.kernelTaskParams.numBlocks = 4;

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(nullptr));
  EXPECT_FALSE(node.nodeInfos.kernelInfos.isSimtOp);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_SimtOpSupportDisabled_SkipAnalysis) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2002;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2002);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 0);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_FALSE(node.nodeInfos.kernelInfos.isSimtOp);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_AicOnly_SkipCheck) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2003;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2003);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAicOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::AIC_ONLY);
  EXPECT_FALSE(node.nodeInfos.kernelInfos.isSimtOp);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_MixAic10_SkipCheck) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2004;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2004);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMixAic10));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::AIC_ONLY);
  EXPECT_FALSE(node.nodeInfos.kernelInfos.isSimtOp);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_AivOnly_NotSimtType) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2005;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2005);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  SetSimtAivType(0);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::AIV_ONLY);
  EXPECT_FALSE(node.nodeInfos.kernelInfos.isSimtOp);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_AivOnly_SimtType3) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2006;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2006);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  SetSimtAivType(3);
  SkUtSetAclrtFunctionAvailDynUbufSize(4096);
  SetFunctionAllocUbufSize(1024);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmptyForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::AIV_ONLY);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.isSimtOp);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasDynUbufSize);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasAllocUbufSize);
  EXPECT_EQ(node.nodeInfos.kernelInfos.dynUbufSize, 4096U);
  EXPECT_EQ(node.nodeInfos.kernelInfos.allocUbufSize, 1024U);
  EXPECT_TRUE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_AivOnly_SimtType4) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2007;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2007);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  SetSimtAivType(4);
  SkUtSetAclrtFunctionAvailDynUbufSize(8192);
  SetFunctionAllocUbufSize(2048);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmptyForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::AIV_ONLY);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.isSimtOp);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasDynUbufSize);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasAllocUbufSize);
  EXPECT_EQ(node.nodeInfos.kernelInfos.dynUbufSize, 8192U);
  EXPECT_EQ(node.nodeInfos.kernelInfos.allocUbufSize, 2048U);
  EXPECT_TRUE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_Mix11_SimtType) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2008;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2008);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  SetSimtAivType(3);
  SkUtSetAclrtFunctionAvailDynUbufSize(12288);
  SetFunctionAllocUbufSize(4096);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMix11));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmptyForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::MIX_AIC_1_1);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.isSimtOp);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasDynUbufSize);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasAllocUbufSize);
  EXPECT_EQ(node.nodeInfos.kernelInfos.dynUbufSize, 12288U);
  EXPECT_EQ(node.nodeInfos.kernelInfos.allocUbufSize, 4096U);
  EXPECT_TRUE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_Mix12_SimtType) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2009;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2009);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  SetSimtAivType(4);
  SkUtSetAclrtFunctionAvailDynUbufSize(16384);
  SetFunctionAllocUbufSize(8192);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeMix12));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryForBindmapReason));
  MOCKER(rtBinaryGetMetaNum).stubs().will(invoke(FakeRtBinaryGetMetaNumTwoEntriesForSkNode));
  MOCKER(rtBinaryGetMetaInfo).stubs().will(invoke(FakeRtBinaryGetMetaInfoSameCapForSkNode));
  MOCKER(aclrtBinaryGetDevAddress).stubs().will(invoke(FakeAclrtBinaryGetDevAddressForSkNode));
  MOCKER(aclrtGetFunctionAddr).stubs().will(invoke(FakeAclrtGetFunctionAddrForSkNode));
  MOCKER(rtGetBinBuffer).stubs().will(invoke(FakeRtGetBinBufferEmptyForSkNode));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_EQ(node.nodeInfos.kernelInfos.kernelType, SkKernelType::MIX_AIC_1_2);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.isSimtOp);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasDynUbufSize);
  EXPECT_TRUE(node.nodeInfos.kernelInfos.hasAllocUbufSize);
  EXPECT_EQ(node.nodeInfos.kernelInfos.dynUbufSize, 16384U);
  EXPECT_EQ(node.nodeInfos.kernelInfos.allocUbufSize, 8192U);
  EXPECT_TRUE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::CAN_FUSE);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_DynUbufQueryFailed) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2010;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2010);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  SetSimtAivType(3);
  SkUtSetAclrtFunctionGetAvailDynUbufPerBlockRet(ACL_ERROR_FAILURE);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_TRUE(node.nodeInfos.kernelInfos.isSimtOp);
  EXPECT_FALSE(node.nodeInfos.kernelInfos.hasDynUbufSize);
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::KERNEL_ATTR_GET_FAILED);
}

TEST_F(SkNodeTest, IdentifyAndHandleSimtKernel_AllocUbufQueryFailed) {
  UtSkNodeRITaskInternal task{};
  task.taskId = 2011;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = reinterpret_cast<aclrtFuncHandle>(0x2011);
  task.params.kernelTaskParams.numBlocks = 4;

  SuperKernelOptionsManager opts;
  opts.RegisterDefaultOptions();
  SetSimtOpSupportCapability(opts, 1);
  SetSimtAivType(3);
  SkUtSetAclrtFunctionAvailDynUbufSize(4096);
  SetRtFunctionGetMetaInfoRet(-1);

  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionNameRegular));
  MOCKER(aclrtGetFunctionAttribute).stubs().will(invoke(FakeAclrtGetFunctionAttributeAivOnly));
  MOCKER(aclrtFunctionGetBinary).stubs().will(invoke(FakeAclrtFunctionGetBinaryNonNull));

  SuperKernelKernelNode node(MakeOriginTask(task), ACL_MODEL_RI_TASK_KERNEL, 0, 0, 0, INVALID_TASK_ID);
  EXPECT_TRUE(node.InitNode(&opts));
  EXPECT_TRUE(node.nodeInfos.kernelInfos.isSimtOp);
  EXPECT_FALSE(node.nodeInfos.kernelInfos.hasAllocUbufSize);
  EXPECT_FALSE(node.IsFusible());
  EXPECT_EQ(node.GetFusionFailReason(), FusionFailReason::KERNEL_ATTR_GET_FAILED);
}
