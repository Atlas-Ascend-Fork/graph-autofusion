/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <limits>

#include "exception_fixture.h"
#include "st_fixture.h"

namespace {
void ExpectExceptionLog(const std::string &text) {
  EXPECT_NE(ut_log::LogBuffer::Instance().GetContent().find(text), std::string::npos) << text;
}
rtExceptionErrRegInfo_t Core(uint32_t id, rtCoreType_t type, uint64_t pc = 0, uint64_t cond = 0) {
  rtExceptionErrRegInfo_t result{};
  result.coreId = id;
  result.coreType = type;
  result.startPC = pc;
  result.currentPC = pc;
  // Runtime register transport slots documented by the exception decoder:
  // DAV_2201 uses 20/21; DAV_3510 uses 32/33. COND is emitted by set_cond
  // as scope bits | (task index << 8) | state (0..4). No SK memory is decoded.
  for (size_t slot : {20U, 32U}) {
    result.errReg[slot] = static_cast<uint32_t>(cond);
    result.errReg[slot + 1] = static_cast<uint32_t>(cond >> 32);
  }
  return result;
}
}  // namespace

TEST_F(AotSystemTest, ExceptionCallbackReportsRealFusedQueuesAndMatchedSubkernels) {
  for (const auto type : {ACL_KERNEL_TYPE_CUBE, ACL_KERNEL_TYPE_VECTOR}) {
    SCOPED_TRACE(type);
    sk::test::Model model;
    const auto stream = model.AddStream();
    sk::test::KernelSpec spec;
    spec.type = type;
    const auto first = model.AddKernel(stream, "exception_first_" + std::to_string(type), spec);
    model.AddKernel(stream, "exception_second_" + std::to_string(type), spec);
    aclmdlRITaskParams original{};
    ASSERT_EQ(aclmdlRITaskGetParams(first, &original), ACL_SUCCESS);
    const auto pc = sk::test::RegisterExceptionBinary(first);
    ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
    sk::test::Exception exception(first);
    const auto coreType = type == ACL_KERNEL_TYPE_CUBE ? RT_CORE_TYPE_AIC : RT_CORE_TYPE_AIV;
    ASSERT_NE(pc, 0U);
    exception.registers = {Core(0, coreType, pc), Core(1, coreType, std::numeric_limits<uint64_t>::max())};
    const auto allocated = sk::test::OutstandingAllocations();
    ut_log::LogBuffer::Instance().Clear();
    exception.Raise();
    ExpectExceptionLog("=== SkHeaderInfo ===");
    ExpectExceptionLog("Origin function name: exception_first_");
    ExpectExceptionLog("Found in node[0]");
    ExpectExceptionLog("No sub kernel matched");
    ExpectExceptionLog("devArgs:");
    EXPECT_EQ(sk::test::OutstandingAllocations(), allocated);

    Adx::ExceptionDumpInfo dumps[2]{};
    uint32_t count = 0;
    Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;
    ASSERT_EQ(exception.Dump(dumps, 2, &count, &mode), ACL_SUCCESS);
    ASSERT_EQ(count, 2U);
    EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_OVERWRITE);
    EXPECT_STREQ(dumps[0].kernelName, ("exception_first_" + std::to_string(type)).c_str());
    EXPECT_EQ(dumps[0].argSize, original.kernelTaskParams.argsSize);
    EXPECT_EQ(dumps[0].argAddr, original.kernelTaskParams.args);
    EXPECT_EQ(dumps[0].coreType, coreType);
    EXPECT_EQ(dumps[0].extraTensorNum, 1U);
    EXPECT_EQ(dumps[0].extraTensor[0].tensorSize, exception.argsSize);
    EXPECT_EQ(dumps[0].extraTensor[0].tensorAddr, exception.args);
    EXPECT_EQ(dumps[0].extraTensor[0].shape, std::vector<int64_t>{exception.argsSize});
    EXPECT_EQ(dumps[1].argAddr, nullptr);
    EXPECT_EQ(dumps[1].argSize, 0U);
    EXPECT_EQ(sk::test::OutstandingAllocations(), allocated);
  }
}

TEST_F(AotSystemTest, ExceptionCallbackDecodesCondStatesFromExternalRegisters) {
  sk::test::Model model;
  const auto stream = model.AddStream();
  const auto task = model.AddKernel(stream, "cond_first");
  model.AddKernel(stream, "cond_second");
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  sk::test::Exception exception(task);
  const char *messages[] = {"No SK entry executed yet",     "SK entry launched, no sub-kernel executed yet",
                            "Currently running sub-kernel", "finished, about to run next",
                            "SK entry execution completed", "Unknown opState=5"};
  for (uint32_t state = 0; state < 6; ++state) {
    SCOPED_TRACE(state);
    exception.registers = {Core(0, RT_CORE_TYPE_AIV, 0, (1ULL << 16) | state)};
    ut_log::LogBuffer::Instance().Clear();
    exception.Raise();
    ExpectExceptionLog(messages[state]);
    if (state >= 1 && state <= 3) {
      ExpectExceptionLog("function name: cond_first");
    }
  }
}

TEST_F(AotSystemTest, ExceptionTraceReportsUnexecutedCoresAndRejectsOutOfRangeCore) {
  sk::test::Model model;
  const auto stream = model.AddStream();
  const auto task = model.AddKernel(stream, "trace_exception");
  aclskOption option{};
  option.optionType = aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM;
  option.debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum = 1;
  aclskOptions options{&option, 1};
  ASSERT_EQ(aclskOptimize(model.Handle(), &options), ACL_SUCCESS);
  sk::test::Exception exception(task);
  exception.registers = {Core(0, RT_CORE_TYPE_AIC), Core(10000, RT_CORE_TYPE_AIV)};
  ut_log::LogBuffer::Instance().Clear();
  const auto allocated = sk::test::OutstandingAllocations();
  exception.Raise();
  ExpectExceptionLog("Sub-kernel running info on all");
  ExpectExceptionLog("No SK entry executed yet");
  ExpectExceptionLog("coreId=10000 exceeds");
  EXPECT_EQ(sk::test::OutstandingAllocations(), allocated);
}

TEST_F(AotSystemTest, ExceptionCallbackRejectsMissingAndFailedRuntimeInformation) {
  sk::test::Model model;
  const auto stream = model.AddStream();
  const auto task = model.AddKernel(stream, "exception_errors");
  aclmdlRITaskParams original{};
  ASSERT_EQ(aclmdlRITaskGetParams(task, &original), ACL_SUCCESS);
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  sk::test::Exception exception(task);
  const auto allocated = sk::test::OutstandingAllocations();
  exception.Raise(true);
  ExpectExceptionLog("Exception info is null");
  exception.functionResult = ACL_ERROR_FAILURE;
  exception.Raise();
  ExpectExceptionLog("Failed to get func handle");
  exception.functionResult = ACL_SUCCESS;
  const auto optimizedFunction = exception.function;
  exception.function = original.kernelTaskParams.funcHandle;
  exception.Raise();
  ExpectExceptionLog("does not start with 'sk_entry'");
  exception.function = optimizedFunction;
  exception.argsResult = ACL_ERROR_FAILURE;
  exception.Raise();
  ExpectExceptionLog("aclrtGetArgsFromExceptionInfo failed");
  exception.argsResult = ACL_SUCCESS;
  const auto size = exception.argsSize;
  exception.argsSize = 0;
  exception.Raise();
  ExpectExceptionLog("no args, callback return");
  exception.argsSize = size - 1;
  exception.Raise();
  ExpectExceptionLog("exceeds skDeviceEntryArgsPtrLen");
  exception.argsSize = size;
  const auto validArgs = exception.args;
  exception.args = nullptr;
  exception.Raise();
  exception.args = validArgs;
  exception.registersResult = -1;
  exception.Raise();
  ExpectExceptionLog("Call rtGetExceptionRegInfo for error register information failed");
  EXPECT_EQ(sk::test::OutstandingAllocations(), allocated);
}

TEST_F(AotSystemTest, ExceptionDumpDeduplicatesBinaryAndHandlesExternalErrors) {
  sk::test::Model model;
  const auto stream = model.AddStream();
  const auto task = model.AddKernel(stream, "exception_dump");
  aclmdlRITaskParams original{};
  ASSERT_EQ(aclmdlRITaskGetParams(task, &original), ACL_SUCCESS);
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  sk::test::Exception exception(task);
  exception.registers = {Core(0, RT_CORE_TYPE_AIV), Core(1, RT_CORE_TYPE_AIV)};
  Adx::ExceptionDumpInfo output[2]{};
  uint32_t count = 0;
  Adx::ExceptionDumpMode mode = Adx::ExceptionDumpMode::DUMP_MODE_NONE;
  ASSERT_EQ(exception.Dump(output, 2, &count, &mode), ACL_SUCCESS);
  EXPECT_EQ(count, 1U);
  EXPECT_STREQ(output[0].kernelName, "sk_entry_aiv");
  ExpectExceptionLog("Skip duplicate ExceptionDumpInfo");
  // Use fresh vector-containing output objects on each call: the external API
  // fills output storage, rather than promising to recycle a prior dump.
  Adx::ExceptionDumpInfo empty[1]{};
  EXPECT_EQ(exception.Dump(empty, 0, &count, &mode), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(exception.Dump(nullptr, 1, &count, &mode), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(exception.Dump(empty, 1, nullptr, &mode), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(exception.Dump(empty, 1, &count, &mode, true), ACL_ERROR_INVALID_PARAM);
  exception.registersResult = -1;
  EXPECT_EQ(exception.Dump(empty, 1, &count, &mode), ACL_ERROR_FAILURE);
  exception.registersResult = 0;
  exception.registers.clear();
  ASSERT_EQ(exception.Dump(empty, 1, &count, &mode), ACL_SUCCESS);
  EXPECT_EQ(count, 0U);
  ExpectExceptionLog("array is empty, skip print");
  exception.function = original.kernelTaskParams.funcHandle;
  ASSERT_EQ(exception.Dump(empty, 1, &count, &mode), ACL_SUCCESS);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_NONE);
  EXPECT_EQ(count, 0U);
  exception.info.expandInfo.type = static_cast<rtExceptionExpandType_t>(99);
  ASSERT_EQ(exception.Dump(empty, 1, &count, &mode), ACL_SUCCESS);
  EXPECT_EQ(mode, Adx::ExceptionDumpMode::DUMP_MODE_NONE);
}

TEST_F(AotSystemTest, ExceptionCopyAndHostAllocationFailuresReleasePartialBuffers) {
  sk::test::Model model;
  auto stream = model.AddStream();
  auto task = model.AddKernel(stream, "exception_memory_failure");
  ASSERT_EQ(aclskOptimize(model.Handle(), nullptr), ACL_SUCCESS);
  sk::test::Exception exception(task);
  const auto allocated = sk::test::OutstandingAllocations();
  for (bool copyFailure : {false, true}) {
    for (int call : {1, 2}) {
      SCOPED_TRACE(copyFailure);
      SCOPED_TRACE(call);
      SkUtSetAclrtMemcpyFailOnCall(copyFailure ? call : 0);
      SkUtSetAclrtMallocHostFailOnCall(copyFailure ? 0 : call);
      ut_log::LogBuffer::Instance().Clear();
      exception.Raise();
      ExpectExceptionLog(copyFailure ? "aclrtMemcpy" : "aclrtMallocHost");
      EXPECT_EQ(sk::test::OutstandingAllocations(), allocated);
    }
  }
  SkUtSetAclrtMemcpyFailOnCall(0);
  SkUtSetAclrtMallocHostFailOnCall(0);
}
