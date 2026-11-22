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
#include <sstream>
#include <string>
#include <sys/wait.h>

namespace {

using autofuse::tests::FileExists;
using autofuse::tests::ReadFile;
using autofuse::tests::RunCommand;
using autofuse::tests::WriteFile;

constexpr const char *kZeroUbThresholdInputConfigsJson =
    R"([{"ub_threshold":0.0,"corenum_threshold":1.0,"enable_multicore_ub_tradeoff":false}])";

bool HasDynamicSymbol(const std::string &path, const std::string &symbol) {
  return RunCommand("nm -D " + path + " 2>/dev/null | grep -q ' " + symbol + "$'") == 0;
}

void VerifySplitHostArtifacts(const std::string &host_dir) {
  // host 编译为单个 cpp 源文件（cpp 段合并），header 段拆出独立 .h 供 include 引用。
  // 新格式（含 TilingStateHeader）下 codegen 不再单独输出 TilingHead/common.h，
  // 公共结构体定义已内联进合并后的 cpp 翻译单元，故 common.h 与 base/entry/tail 一致均不存在。
  EXPECT_FALSE(FileExists(host_dir + "/autofuse_tiling_func_common.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_state.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_log.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_pgo.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_solver.h"));
  ASSERT_TRUE(FileExists(host_dir + "/autofuse_tiling_func_api.h"));
  EXPECT_FALSE(FileExists(host_dir + "/autofuse_tiling_func_base.h"));
  EXPECT_FALSE(FileExists(host_dir + "/autofuse_tiling_func_entry.h"));
  EXPECT_FALSE(FileExists(host_dir + "/autofuse_tiling_func_tail.h"));
  // cpp 段合并为单个源文件，不再拆分多个 segment cpp（HeaderSelfContainedCheck 生成的
  // *.self_contained.cpp 仅用于头文件自包含校验，不计入产物结构检查）。
  ASSERT_TRUE(FileExists(host_dir + "/inductor_topn_tiling_func.cpp"));
  EXPECT_FALSE(FileExists(host_dir + "/inductor_topn_tiling_func_schedule_group_tail.cpp"));
  EXPECT_FALSE(FileExists(host_dir + "/inductor_topn_tiling_func_solver_func.cpp"));
  const std::string merged = ReadFile(host_dir + "/inductor_topn_tiling_func.cpp");
  EXPECT_NE(merged.find("extern \"C\" int64_t AutofuseTiling"), std::string::npos);
  EXPECT_NE(merged.find("extern \"C\" int64_t GenerateTopnSolutions"), std::string::npos);
}

std::string HeaderSelfContainedCheck() {
  return "    header_names = [\n"
         "        'autofuse_tiling_func_state.h',\n"
         "        'autofuse_tiling_func_log.h',\n"
         "        'autofuse_tiling_func_pgo.h',\n"
         "        'autofuse_tiling_func_solver.h',\n"
         "        'autofuse_tiling_func_api.h']\n"
         "    host_dir = os.path.join(host_out, 'host')\n"
         "    compile_args = SimpleNamespace(soc_version='Ascend910B', compile_options='-Werror')\n"
         "    for header_name in header_names:\n"
         "        source_file = os.path.join(host_dir, header_name + '.self_contained.cpp')\n"
         "        with open(source_file, 'w') as source:\n"
         "            source.write('#include \\\"' + header_name + '\\\"\\n')\n"
         "        command = ascendc_compile.build_host_compile_cmd(\n"
         "            compile_args, host_out, source_file, source_file + '.o')\n"
         "        output_index = command.index('-o')\n"
         "        del command[output_index:output_index + 2]\n"
         "        command.remove('-c')\n"
         "        command.insert(-1, '-fsyntax-only')\n"
         "        ascendc_compile.run_compile_command(command, 'HeaderSelfContained')\n";
}

int RunHostCompile(const std::string &tiling_def, const std::string &host_code, const std::string &output_file) {
  WriteFile(OUTPUT_DIR "/host_tiling_def.h", tiling_def);
  WriteFile(OUTPUT_DIR "/host_impl.cpp", host_code);

  std::string script_path = std::string(OUTPUT_DIR) + "/run_host_compile.py";
  WriteFile(script_path, autofuse::tests::PythonPreamble() +
                             "try:\n"
                             "    from autofuse.compile_adapter import host_compile\n"
                             "    from autofuse import ascendc_compile\n"
                             "    from types import SimpleNamespace\n"
                             "    import os, shutil\n"
                             "    host_out = '" +
                             std::string(OUTPUT_DIR) +
                             "/host_out'\n"
                             "    if os.path.exists(host_out):\n"
                             "        shutil.rmtree(host_out)\n"
                             "    os.makedirs(host_out, exist_ok=True)\n"
                             "    td = open('" +
                             std::string(OUTPUT_DIR) +
                             "/host_tiling_def.h').read()\n"
                             "    hc = open('" +
                             std::string(OUTPUT_DIR) +
                             "/host_impl.cpp').read()\n"
                             "    host_compile(td, hc, [\n"
                             "        '--graph_name=inductor_topn',\n"
                             "        '--output_file=" +
                             output_file +
                             "',\n"
                             "        '--output_path=" +
                             std::string(OUTPUT_DIR) +
                             "/host_out',\n"
                             "        '--soc_version=Ascend910B',\n"
                             "        '--compile_options=-Werror'])\n" +
                             HeaderSelfContainedCheck() +
                             "except Exception:\n"
                             "    traceback.print_exc()\n"
                             "    sys.exit(1)\n");

  std::string cmd = "ASCEND_HOME_PATH=" + std::string(ASCEND_HOME_PATH) + " python3 " + script_path + " 2>&1";
  int ret = RunCommand(cmd);
  if (ret != 0) printf("host_compile failed, ret=%d\n", ret);
  return ret;
}

constexpr const char *kGraphName = "inductor_topn";

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code, const std::string &output_file,
                     const std::string &work_dir, const std::string &tiling_repr) {
  return autofuse::tests::RunKernelCompile(tiling_def, device_code, output_file, work_dir, {kGraphName, tiling_repr});
}

}  // namespace
class TestBackendInductorTopnSplitCompile : public testing::Test {};

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
  const std::string kernel_static = OUTPUT_DIR "/inductor_topn_static.so";
  const std::string dynamic_dir = OUTPUT_DIR "/device_dynamic";
  const std::string kernel_dynamic = OUTPUT_DIR "/inductor_topn_dynamic.so";
  autofuse::tests::ParallelCompileAndVerifySo(tiling_def, device_code, tiling_repr, static_dir, kernel_static,
                                              dynamic_dir, kernel_dynamic, RunKernelCompile);

  const std::string static_src = ReadFile(static_dir + "/device/inductor_topn_op_kernel.cpp");
  EXPECT_NE(static_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "static kernel should have constexpr tiling";
  EXPECT_EQ(static_src.find("const AutofuseTilingData t;"), std::string::npos)
      << "static kernel should not have non-const tiling";

  const std::string dynamic_src = ReadFile(dynamic_dir + "/device/inductor_topn_op_kernel.cpp");
  EXPECT_NE(dynamic_src.find("AutofuseTilingData t)"), std::string::npos)
      << "dynamic kernel should have tiling parameter";
  EXPECT_EQ(dynamic_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "dynamic kernel should not have constexpr tiling";
}

TEST_F(TestBackendInductorTopnSplitCompile, SplitCompileChainWorks) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  const std::string host_bin = OUTPUT_DIR "/inductor_topn_host.so";
  ASSERT_EQ(RunHostCompile(tiling_def, host_code, host_bin), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;
  ASSERT_TRUE(autofuse::tests::HasCxx11AbiSymbols(host_bin)) << "host so should use ABI=1: " << host_bin;
  ASSERT_TRUE(HasDynamicSymbol(host_bin, "AutofuseTiling")) << "AutofuseTiling missing in host so";
  ASSERT_TRUE(HasDynamicSymbol(host_bin, "GenerateTopnSolutions")) << "GenerateTopnSolutions missing in host so";
  ASSERT_TRUE(HasDynamicSymbol(host_bin, "GetTilingDataRepr")) << "GetTilingDataRepr missing in host so";
  VerifySplitHostArtifacts(OUTPUT_DIR "/host_out/host");
  const std::string tiling_repr_file = OUTPUT_DIR "/tiling_repr.txt";
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, tiling_repr_file), 0);
  const autofuse::tests::HostHelperOptions zero_ub_options = {kZeroUbThresholdInputConfigsJson, 1, true};
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, OUTPUT_DIR "/zero_ub_threshold_tiling_repr.txt", zero_ub_options),
            0);
  std::string tiling_repr = ReadFile(tiling_repr_file);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}

TEST_F(TestBackendInductorTopnSplitCompile, EmptyTensorFallbackHostCompilesWithWerror) {
  const std::string tiling_def = ReadFile(EMPTY_TILING_DEF_FILE);
  const std::string host_code = ReadFile(EMPTY_HOST_CODE_FILE);
  ASSERT_FALSE(tiling_def.empty());
  ASSERT_FALSE(host_code.empty());
  EXPECT_EQ(host_code.find("TemplateSemanticInfo"), std::string::npos);
  EXPECT_EQ(host_code.find("GetTemplateSemanticInfos"), std::string::npos);
  EXPECT_EQ(host_code.find("GetTilingCore(default_tiling, -1)"), std::string::npos);
  EXPECT_EQ(host_code.find("EmitFinalTilingByCase"), std::string::npos);
  EXPECT_EQ(host_code.find("ShouldEmitFinalTiling()"), std::string::npos);
  const std::string host_bin = OUTPUT_DIR "/inductor_topn_empty_host.so";
  ASSERT_EQ(RunHostCompile(tiling_def, host_code, host_bin), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "empty-tensor host so not found: " << host_bin;
}
