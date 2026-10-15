/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "super_kernel.h"
#include "model_fixture.h"
#include "profiling_fixture.h"
#include "ut_common_stubs.h"
#include "dlog_pub.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
void Require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
}  // namespace
int main(int argc, char **argv) {
  try {
    Require(argc == 4, "expected work directory, profiling value and scenario");
    std::filesystem::current_path(argv[1]);
    const std::string value = argv[2];
    const std::string scenario = argv[3];
    unsetenv("ASCEND_OP_COMPILE_SAVE_KERNEL_META");
    Require(setenv("ASCEND_PROF_SK_ON", value.c_str(), 1) == 0, "set profiling environment");
    std::filesystem::create_directory("export");
    sk::test::SetProfilingPath(scenario == "empty_path" ? "" : (std::filesystem::current_path() / "export").string());
    if (scenario == "directory_failure") {
      std::ofstream("sk_meta") << "not a directory";
    }
    if (scenario == "memset_failure") {
      SkUtSetAclrtMemsetRet(ACL_ERROR_FAILURE);
    }
    const bool enabled = value == "1";
    for (size_t i = 0; i < 2; ++i) {
      sk::test::Model model;
      auto stream = model.AddStream();
      model.AddKernel(stream, "profile_first");
      model.AddKernel(stream, "profile_second");
      Require(aclskOptimize(model.Handle(), nullptr) == ACL_SUCCESS, "optimize profiling model");
      Require(model.SuccessfulUpdates() == 1, "profiling must preserve model update");
      bool hasEntry = false;
      for (const auto &task : model.Tasks(stream)) {
        if (!task.disabled) {
          Require(!task.args.empty(), "entry arguments");
          hasEntry = true;
        }
      }
      Require(hasEntry, "entry is missing");
    }
    Require(SkUtGetModelDestroyCallbackCount() == 0, "model callbacks leaked");
    Require(sk::test::HasProfilingCallback() == enabled, "unexpected profiling registration");
    const auto logs = ut_log::LogBuffer::Instance().GetContent();
    if (scenario == "memset_failure") {
      Require(logs.find("Failed to memset GM") != std::string::npos, "GM failure not reached");
    } else if (scenario == "directory_failure") {
      Require(logs.find("Failed to create output directory") != std::string::npos, "directory failure not reached");
    } else if (enabled) {
      Require(logs.find("Created context for device") != std::string::npos, "recorder context missing");
    } else {
      Require(sk::test::OutstandingAllocations() == 0, "disabled profiling leaked memory");
    }
    if (enabled && scenario != "memset_failure" && scenario != "directory_failure") {
      Require(sk::test::NotifyProfiling(), "start callback missing");
      if (scenario != "missing_stop") {
        Require(sk::test::NotifyProfiling(), "stop callback missing");
      }
    }
    // Normal return exercises the real recorder destructor and final export.
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    ut_log::LogBuffer::Instance().Flush();
    return 1;
  }
}
