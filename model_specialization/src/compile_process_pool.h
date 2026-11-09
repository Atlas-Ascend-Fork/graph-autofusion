/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MODEL_SPEC_COMPILE_PROCESS_POOL_H
#define MODEL_SPEC_COMPILE_PROCESS_POOL_H

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "compile_protocol.h"

namespace model_spec::compile {

enum class CompileOutcome { Success, CompileFailed, ExecutionFailed, TimedOut, ArtifactFailed, Skipped };

struct CompileJobResult {
  uint64_t jobId = NO_JOB_ID;
  CompileOutcome outcome = CompileOutcome::Skipped;
  std::optional<int32_t> aclrtcCode;
  std::string detail;
  uint64_t artifactSize = 0;
  std::string artifactSha256;
};

enum class PoolDisposition { Finished, RetrySerialBeforeDispatch };

struct PoolRunResult {
  PoolDisposition disposition = PoolDisposition::Finished;
  std::vector<CompileJobResult> jobs;
  bool preserveOutputDirectory = false;
};

struct PoolConfig {
  uint64_t workerCount = 1;
  std::string workerPath;
  std::string outputDirectory;
  std::chrono::milliseconds startupTimeout{10000};
  std::chrono::milliseconds taskTimeout{600000};
  std::chrono::milliseconds shutdownTimeout{5000};
  std::chrono::milliseconds termGrace{1000};
  std::chrono::milliseconds groupCheckTimeout{1000};
  std::chrono::milliseconds reapInterval{20};
};

struct CleanupReport {
  bool preserveOutputDirectory = false;
  std::vector<int64_t> unconfirmedWorkers;
};

class CompileProcessPool final {
 public:
  explicit CompileProcessPool(PoolConfig config);
  ~CompileProcessPool();
  CompileProcessPool(const CompileProcessPool &) = delete;
  CompileProcessPool &operator=(const CompileProcessPool &) = delete;

  PoolRunResult Run(const std::vector<CompileJob> &jobs);
  CleanupReport CloseAndReap();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace model_spec::compile
#endif
