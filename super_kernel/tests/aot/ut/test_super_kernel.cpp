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
#include <mockcpp/mockcpp.hpp>
#include <cstdlib>
#include <string>

#include "super_kernel.h"
#include "sk_common.h"
#include "securec.h"
#include "ut_common_stubs.h"

namespace {

const aclrtFuncHandle kScopeBeginFunc = reinterpret_cast<aclrtFuncHandle>(0x1001);
const aclrtFuncHandle kScopeEndFunc = reinterpret_cast<aclrtFuncHandle>(0x1002);

struct TestRITask {
  uint32_t taskId = 0;
  aclmdlRITaskType type = ACL_MODEL_RI_TASK_DEFAULT;
  aclmdlRITaskParams params{};
};

TestRITask g_scopeTasks[2];

void InitScopeTask(TestRITask &task, uint32_t taskId, aclrtFuncHandle funcHandle) {
  task.taskId = taskId;
  task.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params = {};
  task.params.type = ACL_MODEL_RI_TASK_KERNEL;
  task.params.kernelTaskParams.funcHandle = funcHandle;
  task.params.kernelTaskParams.args = reinterpret_cast<void *>(0x2000 + taskId);
  task.params.kernelTaskParams.argsSize = sizeof(ScopeKernelArgs);
  task.params.kernelTaskParams.numBlocks = 1;
}

aclError FakeAclmdlRIGetTasksByStream(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks) {
  (void)stream;
  if (numTasks == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *numTasks = 2;
  if (tasks == nullptr) {
    return ACL_SUCCESS;
  }
  tasks[0] = reinterpret_cast<aclmdlRITask>(&g_scopeTasks[0]);
  tasks[1] = reinterpret_cast<aclmdlRITask>(&g_scopeTasks[1]);
  return ACL_SUCCESS;
}

aclError FakeAclmdlRIGetSingleTaskByStream(aclrtStream stream, aclmdlRITask *tasks, uint32_t *numTasks) {
  (void)stream;
  if (numTasks == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *numTasks = 1;
  if (tasks != nullptr) {
    tasks[0] = reinterpret_cast<aclmdlRITask>(&g_scopeTasks[0]);
  }
  return ACL_SUCCESS;
}

aclError FakeAclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char *name) {
  if (name == nullptr || maxLen == 0) {
    return ACL_ERROR_INVALID_PARAM;
  }
  const char *funcName = "ut_regular_kernel";
  if (funcHandle == kScopeBeginFunc) {
    funcName = "sk_scope_kernel_begin_dav_2201";
  } else if (funcHandle == kScopeEndFunc) {
    funcName = "sk_scope_kernel_end_dav_2201";
  }
  return snprintf_s(name, maxLen, maxLen - 1, "%s", funcName) < 0 ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

aclError FakeAclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  (void)src;
  (void)count;
  (void)kind;
  if (dst == nullptr || destMax < sizeof(ScopeKernelArgs)) {
    return ACL_ERROR_INVALID_PARAM;
  }
  auto *args = reinterpret_cast<ScopeKernelArgs *>(dst);
  return snprintf_s(args->name, sizeof(args->name), sizeof(args->name) - 1, "%s", "ut_scope") < 0 ? ACL_ERROR_FAILURE
                                                                                                  : ACL_SUCCESS;
}

class SuperKernelApiTest : public testing::Test {
 protected:
  void SetUp() override {
    SkUtResetTestControls();
    InitScopeTask(g_scopeTasks[0], 1, kScopeBeginFunc);
    InitScopeTask(g_scopeTasks[1], 2, kScopeEndFunc);
  }

  void TearDown() override {
    (void)SkUtInvokeModelDestroyCallback(model_);
    unsetenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META");
    SkUtResetTestControls();
    GlobalMockObject::verify();
  }

  aclmdlRI model_ = reinterpret_cast<aclmdlRI>(0xA0A0);
};

}  // namespace

TEST_F(SuperKernelApiTest, Optimize_InvalidModelOrGetModelIdFailure_ReturnsBeforeResourceRegistration) {
  EXPECT_EQ(aclskOptimize(nullptr, nullptr), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(SkUtGetDestroyRegisterCallbackCallCount(), 0U);

  SkUtSetAclmdlRIGetIdRet(ACL_ERROR_FAILURE);
  EXPECT_EQ(aclskOptimize(model_, nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(SkUtGetDestroyRegisterCallbackCallCount(), 0U);
}

TEST_F(SuperKernelApiTest, Optimize_GraphUpdateFailure_ReturnsError) {
  SkUtSetModelStreamNum(1);
  SkUtSetAclmdlRIUpdateRet(ACL_ERROR_FAILURE);
  MOCKER(aclmdlRIGetTasksByStream).stubs().will(invoke(FakeAclmdlRIGetTasksByStream));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionName));
  MOCKER(aclrtMemcpy).stubs().will(invoke(FakeAclrtMemcpy));

  EXPECT_EQ(aclskOptimize(model_, nullptr), ACL_ERROR_FAILURE);
}

TEST_F(SuperKernelApiTest, Optimize_UnpairedScopeBegin_ReturnsError) {
  SkUtSetModelStreamNum(1);
  MOCKER(aclmdlRIGetTasksByStream).stubs().will(invoke(FakeAclmdlRIGetSingleTaskByStream));
  MOCKER(aclrtGetFunctionName).stubs().will(invoke(FakeAclrtGetFunctionName));
  MOCKER(aclrtMemcpy).stubs().will(invoke(FakeAclrtMemcpy));

  EXPECT_EQ(aclskOptimize(model_, nullptr), ACL_ERROR_FAILURE);
  EXPECT_EQ(SkUtGetDebugJsonPrintCallCount(), 0U);
}

TEST_F(SuperKernelApiTest, Optimize_SuccessDumpsAfterUpdateRtsJson) {
  setenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META", "1", 1);

  ASSERT_EQ(aclskOptimize(model_, nullptr), ACL_SUCCESS);

  ASSERT_EQ(SkUtGetDebugJsonPrintCallCount(), 2U);
  const char *afterPath = SkUtGetDebugJsonPrintPath(1);
  ASSERT_NE(afterPath, nullptr);
  EXPECT_NE(std::string(afterPath).find("sk_mdl_updated.json"), std::string::npos);
  EXPECT_NE(std::string(afterPath).find("/model_41120_"), std::string::npos);
  EXPECT_EQ(std::string(afterPath).find("/model_model_"), std::string::npos);
}
