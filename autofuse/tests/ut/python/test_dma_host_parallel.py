#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise completion, error propagation and worker shutdown without a device."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def host_jobs_binary(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    work = tmp_path_factory.mktemp("dma-host-jobs")
    source = work / "main.cpp"
    source.write_text(r"""#include "autofuse/tools/modeling/v35/dma/src/host_parallel.h"
#include <atomic>
#include <cstdlib>
int main(int argc, char **argv) {
  if (argc != 2) { return 1; }
  HostJobs jobs(static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)));
  std::array<std::atomic<unsigned>, 257> seen{};
  for (unsigned round = 0; round < 2; ++round) {
    jobs.Run(seen.size(), [&](uint64_t i) { ++seen[i]; });
  }
  for (const auto &count : seen) {
    if (count != 2) { return 2; }
  }
  jobs.Run(0, [](uint64_t) { std::abort(); });
  bool caught = false;
  try {
    jobs.Run(16, [](uint64_t i) {
      if (i == 2 || i == 5) { throw std::runtime_error(std::to_string(i)); }
    });
  } catch (const std::runtime_error &error) {
    caught = std::string(error.what()) == "2";
  }
  if (!caught) { return 3; }
  std::atomic<unsigned> completed{0};
  jobs.Run(17, [&](uint64_t) { ++completed; });
  return completed == 17 ? 0 : 4;
}
""")
    binary = work / "host_jobs"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return binary


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_host_jobs_completion_and_shutdown(host_jobs_binary, threads):
    subprocess.run([str(host_jobs_binary), str(threads)], check=True, timeout=10)
