/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <future>
#include <gtest/gtest.h>

#include "../common/inductor_split_compile_common.h"
#include "../common/inductor_split_compile_config.h"
#include <string>
#include <sys/wait.h>
#include <vector>

namespace {

using autofuse::tests::FileExists;
using autofuse::tests::ReadFile;

void VerifySplitHostArtifacts(const std::string &host_dir) {
  EXPECT_FALSE(FileExists(host_dir + "/autofuse_tiling_func_common.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_state.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_log.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_pgo.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_solver.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_api.h"));
  ASSERT_TRUE(FileExists(host_dir + "/inductor_topn_concat_tiling_func.cpp"));
  const std::string merged = ReadFile(host_dir + "/inductor_topn_concat_tiling_func.cpp");
  EXPECT_NE(merged.find("extern \"C\" int64_t AutofuseTiling"), std::string::npos);
  EXPECT_NE(merged.find("extern \"C\" int64_t GenerateTopnSolutions"), std::string::npos);
}

constexpr const char *kGraphName = "inductor_topn_concat";

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code, const std::string &output_file,
                     const std::string &work_dir, const std::string &tiling_repr) {
  return autofuse::tests::RunKernelCompile(tiling_def, device_code, output_file, work_dir, {kGraphName, tiling_repr});
}

}  // namespace
class TestBackendInductorTopnConcatSplitCompile : public testing::Test {};

void PrepareInputs(std::string &tiling_def, std::string &host_code, std::string &device_code) {
  tiling_def = ReadFile(TILING_DEF_FILE);
  host_code = ReadFile(HOST_CODE_FILE);
  device_code = ReadFile(DEVICE_CODE_FILE);
  ASSERT_FALSE(tiling_def.empty()) << "tiling_def empty";
  ASSERT_FALSE(host_code.empty()) << "host_code empty";
  ASSERT_FALSE(device_code.empty()) << "device_code empty";
}

void CompileAndVerifyKernels(const std::string &tiling_def, const std::string &device_code,
                             const std::string &tiling_repr) {
  const std::string static_dir = OUTPUT_DIR "/device_static";
  const std::string kernel_static = OUTPUT_DIR "/inductor_topn_concat_static.so";
  const std::string dynamic_dir = OUTPUT_DIR "/device_dynamic";
  const std::string kernel_dynamic = OUTPUT_DIR "/inductor_topn_concat_dynamic.so";
  autofuse::tests::ParallelCompileAndVerifySo(tiling_def, device_code, tiling_repr, static_dir, kernel_static,
                                              dynamic_dir, kernel_dynamic, RunKernelCompile);

  const std::string static_src = ReadFile(static_dir + "/device/inductor_topn_concat_op_kernel.cpp");
  EXPECT_NE(static_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "static kernel should have constexpr tiling";
  EXPECT_EQ(static_src.find("const AutofuseTilingData t;"), std::string::npos)
      << "static kernel should not have non-const tiling";

  const std::string dynamic_src = ReadFile(dynamic_dir + "/device/inductor_topn_concat_op_kernel.cpp");
  EXPECT_NE(dynamic_src.find("AutofuseTilingData t)"), std::string::npos)
      << "dynamic kernel should have tiling parameter";
  EXPECT_EQ(dynamic_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "dynamic kernel should not have constexpr tiling";
}

TEST_F(TestBackendInductorTopnConcatSplitCompile, TopnSearchProducesCandidatesBeforeDefaultFallback) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  // Reject fallback-only results so a successful public TopN call proves that ATT search produced candidates.
  const std::string fallback = "response.candidate_solutions.push_back(default_solution);";
  const auto fallback_pos = host_code.find(fallback);
  ASSERT_NE(fallback_pos, std::string::npos);
  host_code.insert(fallback_pos, "if (response.candidate_solutions.empty()) { return -1; }\n    ");

  const std::string host_bin = OUTPUT_DIR "/inductor_topn_concat_search_host.so";
  ASSERT_EQ(autofuse::tests::RunHostCompile(tiling_def, host_code, host_bin, "inductor_topn_concat", "-Werror"), 0);
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, OUTPUT_DIR "/search_tiling_repr.txt"), 0);
}

TEST_F(TestBackendInductorTopnConcatSplitCompile, SplitCompileChainWorks) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  const std::string host_bin = OUTPUT_DIR "/inductor_topn_concat_host.so";
  ASSERT_EQ(autofuse::tests::RunHostCompile(tiling_def, host_code, host_bin, "inductor_topn_concat", "-Werror"), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;
  ASSERT_TRUE(autofuse::tests::HasCxx11AbiSymbols(host_bin)) << "host so should use ABI=1: " << host_bin;
  VerifySplitHostArtifacts(OUTPUT_DIR "/host_out/host");
  const std::string tiling_repr_file = OUTPUT_DIR "/tiling_repr.txt";
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, tiling_repr_file), 0);
  std::string tiling_repr = ReadFile(tiling_repr_file);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}
