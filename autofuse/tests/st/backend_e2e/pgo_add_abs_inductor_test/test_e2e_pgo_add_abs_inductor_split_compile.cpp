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
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>

#include "../common/inductor_split_compile_common.h"
#include <sstream>
#include <string>
#include <sys/wait.h>

#include "../common/inductor_split_compile_config.h"

#ifndef PGO_TILING_DEF_FILE
#define PGO_TILING_DEF_FILE ""
#endif
#ifndef PGO_HOST_CODE_FILE
#define PGO_HOST_CODE_FILE ""
#endif
#ifndef PGO_DEVICE_CODE_FILE
#define PGO_DEVICE_CODE_FILE ""
#endif
#ifndef PGO_FAKE_CALLBACK_SRC
#define PGO_FAKE_CALLBACK_SRC ""
#endif
namespace {

using autofuse::tests::FileExists;
using autofuse::tests::HasCxx11AbiSymbols;
using autofuse::tests::PythonPreamble;
using autofuse::tests::ReadFile;
using autofuse::tests::RunCommand;
using autofuse::tests::WriteFile;

constexpr const char *kGraphName = "pgo_add_abs_inductor";

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code, const std::string &output_file,
                     const std::string &work_dir, const std::string &tiling_repr) {
  return autofuse::tests::RunKernelCompile(tiling_def, device_code, output_file, work_dir, {kGraphName, tiling_repr});
}

int RunPgoArtifactCompile(const std::string &tiling_def, const std::string &host_code, const std::string &device_code) {
  const std::string work_dir = OUTPUT_DIR "/pgo_compile";
  std::filesystem::remove_all(work_dir);
  RunCommand("mkdir -p " + work_dir);
  WriteFile(work_dir + "/tiling_def.h", tiling_def);
  WriteFile(work_dir + "/host_impl.cpp", host_code);
  WriteFile(work_dir + "/device_impl.cpp", device_code);
  const std::string script_path = work_dir + "/run_pgo_compile.py";
  WriteFile(script_path, PythonPreamble() +
                             "from autofuse import compile_adapter as ca\n"
                             "argv = ['--graph_name=pgo_add_abs_inductor',\n"
                             "        '--output_file=" +
                             work_dir +
                             "/tiling.so',\n"
                             "        '--output_path=" +
                             work_dir +
                             "',\n"
                             "        '--soc_version=Ascend910B']\n"
                             "td = open('" +
                             work_dir +
                             "/tiling_def.h').read()\n"
                             "hc = open('" +
                             work_dir +
                             "/host_impl.cpp').read()\n"
                             "ca.host_compile(td, hc, argv)\n"
                             "import glob, os, shutil\n"
                             "bundles = glob.glob('" +
                             work_dir +
                             "/tiling.so.pgo.*')\n"
                             "assert len(bundles) == 1\n"
                             "shutil.copy2(os.path.join(bundles[0], 'tiling.so.pgo_runner'), '" +
                             work_dir +
                             "/pgo_runner')\n"
                             "shutil.copy2(os.path.join(bundles[0], 'tiling.so.pgo_kernel.aicore_binary_elf_v1'), '" +
                             work_dir + "/pgo_kernel.aicore_binary_elf_v1')\n");
  return RunCommand("ASCEND_HOME_PATH=" + std::string(ASCEND_HOME_PATH) + " python3 " + script_path + " 2>&1");
}

std::string MeasuredTopnParserScript(const std::string &work_dir) {
  return PythonPreamble() +
         "from autofuse import compile_adapter as ca\n"
         "from autofuse import ascendc_compile as ac\n"
         "import subprocess\n"
         "argv = ['--graph_name=pgo_add_abs_inductor', '--output_file=" +
         work_dir + "/topn.json', '--output_path=" + work_dir +
         "', '--soc_version=Ascend910B']\n"
         "args, _, _ = ca.prepare_compile_context(argv, 'host', None)\n"
         "args.trace_stage = 'pgo_fake_callback_st'\n"
         "td = open('" +
         work_dir +
         "/tiling_def.h').read()\n"
         "hc = open('" +
         work_dir +
         "/host_impl.cpp').read()\n"
         "dc = open('" +
         work_dir +
         "/device_impl.cpp').read()\n"
         "ca.generate_file(args.temp_dir + '/host', 'autofuse_tiling_data.h', td)\n"
         "normal_files, _, _ = ca.write_inductor_pgo_sources(\n"
         "    args.temp_dir + '/host', args.temp_dir + '/device', args.graph_name, hc)\n"
         "parser_sources = [p for p in normal_files if 'ParseInductorPgoResult' in open(p).read()]\n"
         "assert len(parser_sources) == 1\n"
         "with open(parser_sources[0], 'a') as f:\n"
         "    f.write(r'''\n"
         "extern \"C\" int TestParseInductorPgoResult(const char *path, int64_t topn, bool expected) {\n"
         "  std::vector<AutofuseTilingData> tilings(1);\n"
         "  std::vector<int64_t> workspaces = {123};\n"
         "  std::vector<int64_t> block_dims = {456};\n"
         "  const bool parsed = ParseInductorPgoResult(path, topn, tilings, workspaces, block_dims);\n"
         "  if (parsed != expected) { return 1; }\n"
         "  if (!parsed && (tilings.size() != 1 || workspaces != std::vector<int64_t>{123} ||\n"
         "                  block_dims != std::vector<int64_t>{456})) { return 2; }\n"
         "  return parsed && (tilings.empty() || tilings.size() != workspaces.size() ||\n"
         "                    tilings.size() != block_dims.size()) ? 3 : 0;\n"
         "}\n''')\n";
}

std::string MeasuredTopnLinkScript(const std::string &work_dir) {
  return "args.pgo_generation = 'fakecallback'\n"
         "original_compile_options = args.compile_options\n"
         "args.compile_options += ' -Werror -DAUTOFUSE_PGO_RUNNER_TIMEOUT_SECONDS=1'\n"
         "args.host_files = normal_files\n"
         "objects = ac.compile_host_objs(args, args.temp_dir)\n"
         "args.compile_options = original_compile_options\n"
         "tiling = '" +
         work_dir +
         "/fake_tiling.so'\n"
         "ac.link_shared(tiling, objects, ac.HOST_LINK_LIBRARIES + ['ascendcl', 'runtime'])\n"
         "callback_src = '" +
         work_dir +
         "/host/pgo_measured_topn_fake_callback_main.cpp'\n"
         "callback_obj = ac.compile_host_obj_file(args, args.temp_dir, callback_src)\n"
         "binary = '" +
         work_dir +
         "/pgo_measured_topn_fake_callback'\n"
         "ac.link_pgo_executable(binary, [callback_obj])\n"
         "kernel = '" +
         work_dir +
         "/fake_kernel.aicore_binary_elf_v1'\n"
         "open(kernel, 'wb').write(b'fake-kernel')\n"
         "ac.publish_pgo_bundle(ac.PgoBundle(tiling, binary, kernel, tiling, 'fakecallback'))\n"
         "raise SystemExit(subprocess.run([binary, tiling], check=False).returncode)\n";
}

std::string MeasuredTopnCompileScript(const std::string &work_dir) {
  return MeasuredTopnParserScript(work_dir) + MeasuredTopnLinkScript(work_dir);
}

int RunMeasuredTopnFakeCallback(const std::string &tiling_def, const std::string &host_code,
                                const std::string &device_code) {
  const std::string work_dir = OUTPUT_DIR "/pgo_fake_callback";
  std::filesystem::remove_all(work_dir);
  RunCommand("mkdir -p " + work_dir + "/host");
  WriteFile(work_dir + "/tiling_def.h", tiling_def);
  WriteFile(work_dir + "/host_impl.cpp", host_code);
  WriteFile(work_dir + "/device_impl.cpp", device_code);
  WriteFile(work_dir + "/host/pgo_measured_topn_fake_callback_main.cpp", ReadFile(PGO_FAKE_CALLBACK_SRC));
  const std::string script_path = work_dir + "/run_pgo_fake_callback.py";
  WriteFile(script_path, MeasuredTopnCompileScript(work_dir));
  return RunCommand("ASCEND_HOME_PATH=" + std::string(ASCEND_HOME_PATH) + " python3 " + script_path + " 2>&1");
}

}  // namespace
class TestBackendPgoAddAbsInductorSplitCompile : public testing::Test {};

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
  const std::string kernel_static = OUTPUT_DIR "/pgo_add_abs_inductor_static.so";
  const std::string dynamic_dir = OUTPUT_DIR "/device_dynamic";
  const std::string kernel_dynamic = OUTPUT_DIR "/pgo_add_abs_inductor_dynamic.so";
  autofuse::tests::ParallelCompileAndVerifySo(tiling_def, device_code, tiling_repr, static_dir, kernel_static,
                                              dynamic_dir, kernel_dynamic, RunKernelCompile);

  const std::string static_src = ReadFile(static_dir + "/device/pgo_add_abs_inductor_op_kernel.cpp");
  EXPECT_NE(static_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "static kernel should have constexpr tiling";
  EXPECT_EQ(static_src.find("const AutofuseTilingData t;"), std::string::npos)
      << "static kernel should not have non-const tiling";

  const std::string dynamic_src = ReadFile(dynamic_dir + "/device/pgo_add_abs_inductor_op_kernel.cpp");
  EXPECT_NE(dynamic_src.find("AutofuseTilingData t)"), std::string::npos)
      << "dynamic kernel should have tiling parameter";
  EXPECT_EQ(dynamic_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "dynamic kernel should not have constexpr tiling";
}

TEST_F(TestBackendPgoAddAbsInductorSplitCompile, SplitCompileChainWorks) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  const std::string host_bin = OUTPUT_DIR "/pgo_add_abs_inductor_host.so";
  ASSERT_EQ(autofuse::tests::RunHostCompile(tiling_def, host_code, host_bin, "pgo_add_abs_inductor", "-Werror"), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;
  ASSERT_TRUE(autofuse::tests::HasCxx11AbiSymbols(host_bin)) << "host so should use ABI=1: " << host_bin;
  const std::string tiling_repr_file = OUTPUT_DIR "/tiling_repr.txt";
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, tiling_repr_file), 0);
  std::string tiling_repr = ReadFile(tiling_repr_file);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}

TEST_F(TestBackendPgoAddAbsInductorSplitCompile, PgoRunnerAndDynamicKernelCompile) {
  if (!autofuse::tests::MsptiAvailable()) {
    GTEST_SKIP() << "MSPTI is unavailable, skip PGO runner compile test";
  }
  const std::string tiling_def = ReadFile(PGO_TILING_DEF_FILE);
  const std::string host_code = ReadFile(PGO_HOST_CODE_FILE);
  const std::string device_code = ReadFile(PGO_DEVICE_CODE_FILE);
  ASSERT_FALSE(tiling_def.empty());
  ASSERT_FALSE(host_code.empty());
  ASSERT_FALSE(device_code.empty());
  EXPECT_NE(device_code.find("void add_abs_test("), std::string::npos);
  EXPECT_NE(host_code.find("constexpr char kInductorPgoKernelName[] = \"add_abs_test\";"), std::string::npos);
  EXPECT_NE(host_code.find("aclrtBinaryGetFunction(g_pgo_bin_handle, kInductorPgoKernelName"), std::string::npos);
  EXPECT_NE(host_code.find("AutofuseTilingData tiling_data;"), std::string::npos);
  EXPECT_EQ(host_code.find("g_kernel_name + \"_\""), std::string::npos);
  EXPECT_EQ(host_code.find("g_launch_params.aiv_args.tiling_addr"), std::string::npos);
  EXPECT_EQ(host_code.find("g_tiling_device_addr"), std::string::npos);
  ASSERT_EQ(RunPgoArtifactCompile(tiling_def, host_code, device_code), 0);

  const std::string work_dir = OUTPUT_DIR "/pgo_compile";
  const std::string runner = work_dir + "/pgo_runner";
  const std::string tiling_so = work_dir + "/tiling.so";
  const std::string device_binary = work_dir + "/pgo_kernel.aicore_binary_elf_v1";
  ASSERT_TRUE(FileExists(runner));
  ASSERT_TRUE(FileExists(tiling_so));
  ASSERT_TRUE(FileExists(device_binary));
  EXPECT_NE(RunCommand("nm -D " + runner + " | grep -q 'GenerateTopnSolutions'"), 0);
  EXPECT_EQ(RunCommand("strings " + runner + " | grep -q 'GenerateMeasuredTopnSolutions'"), 0);
  EXPECT_EQ(RunCommand("readelf -h " + device_binary + " | grep -q 'EXEC (Executable file)'"), 0);
  EXPECT_NE(RunCommand("ldd " + runner + " | grep -qi mspti"), 0);
}

TEST_F(TestBackendPgoAddAbsInductorSplitCompile, MeasuredTopnUsesCompleteFakeSampling) {
  const std::string tiling_def = ReadFile(PGO_TILING_DEF_FILE);
  const std::string host_code = ReadFile(PGO_HOST_CODE_FILE);
  const std::string device_code = ReadFile(PGO_DEVICE_CODE_FILE);
  ASSERT_FALSE(tiling_def.empty());
  ASSERT_FALSE(host_code.empty());
  ASSERT_FALSE(device_code.empty());
  EXPECT_EQ(RunMeasuredTopnFakeCallback(tiling_def, host_code, device_code), 0);
}
