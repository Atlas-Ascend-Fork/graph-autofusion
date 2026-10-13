/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_runtime.h"
#include "kernel_module.h"
#include "tensor_buffer.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace device_validation {
namespace {
class FakeRuntime final : public RuntimeApi {
 public:
  Status Init() override {
    return init_status;
  }
  Status SetDevice(int32_t) override {
    return set_device_status;
  }
  Status CreateStream(void **stream) override {
    if (create_stream_status == Status::kOk) *stream = stream_value;
    return create_stream_status;
  }
  Status Synchronize(void *) override {
    calls.push_back("synchronize");
    return synchronize_status;
  }
  Status DestroyStream(void *) override {
    calls.push_back("destroy");
    ++destroy_stream_calls;
    return destroy_stream_status;
  }
  Status ResetDevice(int32_t) override {
    calls.push_back("reset");
    ++reset_device_calls;
    return reset_device_status;
  }
  Status Finalize() override {
    calls.push_back("finalize");
    ++finalize_calls;
    return finalize_status;
  }
  Status Malloc(void **ptr, size_t) override {
    if (malloc_status == Status::kOk) *ptr = allocation_value;
    return malloc_status;
  }
  Status Free(void *) override {
    calls.push_back("free");
    ++free_calls;
    return free_status;
  }
  Status CopyToDevice(void *, const void *, size_t size) override {
    last_copy_size = size;
    ++h2d_calls;
    return h2d_status;
  }
  Status CopyToHost(void *, const void *, size_t size) override {
    last_copy_size = size;
    ++d2h_calls;
    return d2h_status;
  }

  Status init_status = Status::kOk;
  Status set_device_status = Status::kOk;
  Status create_stream_status = Status::kOk;
  Status synchronize_status = Status::kOk;
  Status destroy_stream_status = Status::kOk;
  Status reset_device_status = Status::kOk;
  Status finalize_status = Status::kOk;
  Status malloc_status = Status::kOk;
  Status free_status = Status::kOk;
  Status h2d_status = Status::kOk;
  Status d2h_status = Status::kOk;
  void *stream_value = reinterpret_cast<void *>(0x1);
  void *allocation_value = reinterpret_cast<void *>(0x2);
  size_t last_copy_size = 0;
  int destroy_stream_calls = 0;
  int reset_device_calls = 0;
  int finalize_calls = 0;
  int free_calls = 0;
  int h2d_calls = 0;
  int d2h_calls = 0;
  std::vector<std::string> calls;
};

class FakeLoader final : public ModuleLoader {
 public:
  static int64_t LegacyLaunch1(uint32_t, void *, void *, void *, void *, void *);
  static int64_t LegacyLaunch2(uint32_t, void *, void *, void *, void *, void *, void *);
  static int64_t LegacyLaunch(uint32_t, void *, void *, void *, void *, void *, void *, void *);

  void *Load(const std::filesystem::path &) override {
    return load_result;
  }
  void *Symbol(void *, const char *name) override {
    requested_symbols.emplace_back(name);
    if (missing_symbol == name) return nullptr;
    if (std::string(name) == "GetTilingDataSize") return reinterpret_cast<void *>(&GetSize);
    if (std::string(name) == "AutofuseTiling") return reinterpret_cast<void *>(&Tiling);
    if (std::string(name) == "AutofuseLaunchV2") {
      return has_v2 ? reinterpret_cast<void *>(&Launch) : nullptr;
    }
    if (std::string(name) == "AutofuseLaunch") {
      if (!has_legacy) return nullptr;
      if (legacy_arity == 1) return reinterpret_cast<void *>(&LegacyLaunch1);
      if (legacy_arity == 2) return reinterpret_cast<void *>(&LegacyLaunch2);
      return reinterpret_cast<void *>(&LegacyLaunch);
    }
    return nullptr;
  }
  Status Unload(void *) override {
    ++unload_calls;
    return unload_status;
  }

  static size_t GetSize() {
    return 24;
  }
  static ge::graphStatus Tiling(void *data, uint64_t *workspace, uint32_t *block, void *extra) {
    tiling_data = data;
    tiling_workspace = workspace;
    tiling_block = block;
    tiling_extra = extra;
    *workspace = 0x120000000;
    *block = 8;
    return tiling_status;
  }
  static uint32_t Launch(uint32_t block, void *stream, void **inputs, int32_t input_count, void **outputs,
                         int32_t output_count, void *workspace, void *tiling) {
    launch_block = block;
    launch_stream = stream;
    launch_inputs = inputs;
    launch_input_count = input_count;
    launch_outputs = outputs;
    launch_output_count = output_count;
    launch_workspace = workspace;
    launch_tiling = tiling;
    return launch_status;
  }
  void *load_result = reinterpret_cast<void *>(0x4);
  std::vector<std::string> requested_symbols;
  std::string missing_symbol;
  bool has_v2 = true;
  bool has_legacy = false;
  int legacy_arity = 3;
  Status unload_status = Status::kOk;
  int unload_calls = 0;
  inline static ge::graphStatus tiling_status = ge::GRAPH_SUCCESS;
  inline static void *tiling_data = nullptr;
  inline static uint64_t *tiling_workspace = nullptr;
  inline static uint32_t *tiling_block = nullptr;
  inline static void *tiling_extra = nullptr;
  inline static uint32_t launch_status = 0;
  inline static uint32_t launch_block = 0;
  inline static void *launch_stream = nullptr;
  inline static void **launch_inputs = nullptr;
  inline static int32_t launch_input_count = 0;
  inline static void **launch_outputs = nullptr;
  inline static int32_t launch_output_count = 0;
  inline static void *launch_workspace = nullptr;
  inline static void *launch_tiling = nullptr;
  inline static int64_t legacy_status = 0;
  inline static uint32_t legacy_block = 0;
  inline static void *legacy_stream = nullptr;
  inline static void *legacy_inputs[3] = {};
  inline static void *legacy_output = nullptr;
  inline static void *legacy_workspace = nullptr;
  inline static void *legacy_tiling = nullptr;
};

int64_t FakeLoader::LegacyLaunch(uint32_t block, void *stream, void *input0, void *input1, void *input2, void *output0,
                                 void *workspace, void *tiling) {
  legacy_block = block;
  legacy_stream = stream;
  legacy_inputs[0] = input0;
  legacy_inputs[1] = input1;
  legacy_inputs[2] = input2;
  legacy_output = output0;
  legacy_workspace = workspace;
  legacy_tiling = tiling;
  return legacy_status;
}

int64_t FakeLoader::LegacyLaunch1(uint32_t block, void *stream, void *input0, void *output0, void *workspace,
                                  void *tiling) {
  legacy_block = block;
  legacy_stream = stream;
  legacy_inputs[0] = input0;
  legacy_output = output0;
  legacy_workspace = workspace;
  legacy_tiling = tiling;
  return legacy_status;
}

int64_t FakeLoader::LegacyLaunch2(uint32_t block, void *stream, void *input0, void *input1, void *output0,
                                  void *workspace, void *tiling) {
  legacy_block = block;
  legacy_stream = stream;
  legacy_inputs[0] = input0;
  legacy_inputs[1] = input1;
  legacy_output = output0;
  legacy_workspace = workspace;
  legacy_tiling = tiling;
  return legacy_status;
}

TEST(AclRuntimeTest, CleansUpPartialInitializationInReverseOrder) {
  FakeRuntime api;
  api.create_stream_status = Status::kRuntimeError;
  AclRuntime runtime(&api);
  EXPECT_EQ(runtime.Initialize(2), Status::kRuntimeError);
  EXPECT_EQ(api.reset_device_calls, 1);
  EXPECT_EQ(api.finalize_calls, 1);
  EXPECT_EQ(api.destroy_stream_calls, 0);
}

TEST(AclRuntimeTest, PropagatesSynchronizeAndCleanupFailures) {
  FakeRuntime api;
  AclRuntime runtime(&api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  api.synchronize_status = Status::kRuntimeError;
  EXPECT_EQ(runtime.Synchronize(), Status::kRuntimeError);
  api.synchronize_status = Status::kOk;
  api.destroy_stream_status = Status::kRuntimeError;
  api.reset_device_status = Status::kRuntimeError;
  api.finalize_status = Status::kRuntimeError;
  EXPECT_EQ(runtime.Finalize(), Status::kRuntimeError);
  EXPECT_EQ(api.destroy_stream_calls, 1);
  EXPECT_EQ(api.reset_device_calls, 0);
  EXPECT_EQ(api.finalize_calls, 0);
  EXPECT_EQ(runtime.Finalize(), Status::kRuntimeError);
  EXPECT_EQ(api.destroy_stream_calls, 2);
  EXPECT_EQ(api.reset_device_calls, 0);
  EXPECT_EQ(api.finalize_calls, 0);
  api.destroy_stream_status = Status::kOk;
  api.reset_device_status = Status::kRuntimeError;
  EXPECT_EQ(runtime.Finalize(), Status::kRuntimeError);
  EXPECT_EQ(api.destroy_stream_calls, 3);
  api.reset_device_status = Status::kOk;
  api.finalize_status = Status::kRuntimeError;
  EXPECT_EQ(runtime.Finalize(), Status::kRuntimeError);
  EXPECT_EQ(api.reset_device_calls, 2);
  api.finalize_status = Status::kOk;
  EXPECT_EQ(runtime.Finalize(), Status::kOk);
  EXPECT_EQ(api.finalize_calls, 2);
}

TEST(TensorBufferTest, RejectsZeroOverflowUnsupportedAndRepeatedAllocation) {
  FakeRuntime api;
  AclRuntime runtime(&api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  TensorBuffer buffer;
  EXPECT_EQ(buffer.Allocate({{}, "float16"}, &runtime), Status::kInvalidArgument);
  EXPECT_EQ(buffer.Allocate({{std::numeric_limits<int64_t>::max(), 2}, "float16"}, &runtime), Status::kOverflow);
  EXPECT_EQ(buffer.Allocate({{1}, "complex256"}, &runtime), Status::kUnsupported);
  EXPECT_EQ(buffer.Allocate({{4}, "float16"}, &runtime), Status::kOk);
  EXPECT_EQ(buffer.Allocate({{4}, "float16"}, &runtime), Status::kInvalidArgument);
  EXPECT_EQ(runtime.Finalize(), Status::kInvalidArgument);
  EXPECT_EQ(buffer.Release(), Status::kOk);
  EXPECT_EQ(runtime.Finalize(), Status::kOk);
}

TEST(TensorBufferTest, PropagatesMallocCopyAndReleaseFailures) {
  FakeRuntime api;
  AclRuntime runtime(&api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  TensorBuffer buffer;
  api.malloc_status = Status::kRuntimeError;
  EXPECT_EQ(buffer.Allocate({{4}, "float16"}, &runtime), Status::kRuntimeError);
  api.malloc_status = Status::kOk;
  api.allocation_value = nullptr;
  EXPECT_EQ(buffer.Allocate({{4}, "float16"}, &runtime), Status::kRuntimeError);
  api.allocation_value = reinterpret_cast<void *>(0x2);
  EXPECT_EQ(buffer.Allocate({{4}, "float16"}, &runtime), Status::kOk);
  std::vector<char> data(8);
  api.h2d_status = Status::kRuntimeError;
  EXPECT_EQ(buffer.CopyToDevice(data.data(), data.size()), Status::kRuntimeError);
  api.h2d_status = Status::kOk;
  EXPECT_EQ(buffer.CopyToDevice(data.data(), data.size() - 1), Status::kInvalidArgument);
  api.d2h_status = Status::kRuntimeError;
  EXPECT_EQ(buffer.CopyToHost(data.data(), data.size()), Status::kRuntimeError);
  api.d2h_status = Status::kOk;
  EXPECT_EQ(buffer.CopyToHost(data.data(), data.size()), Status::kOk);
  api.free_status = Status::kRuntimeError;
  EXPECT_EQ(buffer.Release(), Status::kRuntimeError);
  api.free_status = Status::kOk;
  EXPECT_EQ(buffer.Release(), Status::kOk);
}

TEST(TensorBufferTest, RejectsReleaseAfterRuntimeDestruction) {
  FakeRuntime api;
  auto runtime = std::make_unique<AclRuntime>(&api);
  ASSERT_EQ(runtime->Initialize(0), Status::kOk);
  TensorBuffer buffer;
  ASSERT_EQ(buffer.Allocate({{4}, "float16"}, runtime.get()), Status::kOk);
  runtime.reset();
  EXPECT_EQ(buffer.Release(), Status::kOk);
  EXPECT_EQ(api.free_calls, 1);
  EXPECT_EQ(api.destroy_stream_calls, 1);
  EXPECT_EQ(api.reset_device_calls, 1);
  EXPECT_EQ(api.finalize_calls, 1);
  auto runtime2 = std::make_unique<AclRuntime>(&api);
  ASSERT_EQ(runtime2->Initialize(0), Status::kOk);
  auto buffer2 = std::make_unique<TensorBuffer>();
  ASSERT_EQ(buffer2->Allocate({{4}, "float16"}, runtime2.get()), Status::kOk);
  runtime2.reset();
  buffer2.reset();
  EXPECT_EQ(api.free_calls, 2);
}

TEST(TensorBufferTest, DeferredCleanupSynchronizesBeforeFree) {
  FakeRuntime api;
  auto runtime = std::make_unique<AclRuntime>(&api);
  ASSERT_EQ(runtime->Initialize(0), Status::kOk);
  auto buffer = std::make_unique<TensorBuffer>();
  ASSERT_EQ(buffer->Allocate({{4}, "float16"}, runtime.get()), Status::kOk);
  api.synchronize_status = Status::kRuntimeError;
  runtime.reset();
  api.free_status = Status::kRuntimeError;
  EXPECT_EQ(buffer->Release(), Status::kRuntimeError);
  EXPECT_EQ(api.free_calls, 0);
  api.synchronize_status = Status::kOk;
  api.free_status = Status::kOk;
  EXPECT_EQ(buffer->Release(), Status::kOk);
  EXPECT_EQ(api.free_calls, 1);
  ASSERT_GE(api.calls.size(), 3U);
  EXPECT_EQ(api.calls[0], "synchronize");
  EXPECT_EQ(api.calls[1], "synchronize");
  EXPECT_EQ(api.calls[2], "free");
  buffer.reset();
}

TEST(KernelModuleTest, ReportsEachMissingAbiSymbolInDeclarationOrder) {
  for (const std::string &missing : {"GetTilingDataSize", "AutofuseTiling"}) {
    FakeLoader loader;
    loader.missing_symbol = missing;
    KernelModule module(&loader);
    EXPECT_EQ(module.Load("module.so", {}), Status::kNotFound);
    EXPECT_EQ(module.missing_symbol(), missing);
    EXPECT_EQ(loader.unload_calls, 1);
    EXPECT_FALSE(module.IsLoaded());
  }
  FakeLoader loader;
  loader.has_v2 = false;
  KernelModule module(&loader);
  EXPECT_EQ(module.Load("module.so", {}), Status::kNotFound);
  EXPECT_EQ(module.missing_symbol(), "AutofuseLaunchV2");
}

TEST(KernelModuleTest, V2RequestRejectsLegacyOnlyModule) {
  FakeLoader loader;
  loader.has_v2 = false;
  loader.has_legacy = true;
  KernelModule module(&loader);
  EXPECT_EQ(module.Load("module.so", {}), Status::kNotFound);
  EXPECT_EQ(module.missing_symbol(), "AutofuseLaunchV2");
  ASSERT_EQ(loader.requested_symbols.size(), 3U);
  EXPECT_EQ(loader.requested_symbols.back(), "AutofuseLaunchV2");
}

TEST(KernelModuleTest, V2RequestDoesNotProbeLegacySymbol) {
  FakeLoader loader;
  loader.has_v2 = false;
  loader.has_legacy = true;
  KernelModule module(&loader);
  EXPECT_EQ(module.Load("module.so", {}), Status::kNotFound);
  EXPECT_EQ(module.missing_symbol(), "AutofuseLaunchV2");
  EXPECT_EQ(std::find(loader.requested_symbols.begin(), loader.requested_symbols.end(), "AutofuseLaunch"),
            loader.requested_symbols.end());
}

TEST(KernelModuleTest, LegacyRequestLoadsLegacySymbolOnly) {
  FakeLoader loader;
  loader.has_v2 = false;
  loader.has_legacy = true;
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("module.so", {"GetTilingDataSize", "AutofuseTiling", "AutofuseLaunch"}), Status::kOk);
  EXPECT_EQ(module.abi(), "AutofuseLaunch");
  EXPECT_EQ(loader.requested_symbols.back(), "AutofuseLaunch");
}

TEST(KernelModuleTest, LegacyRequestRejectsV2OnlyModule) {
  FakeLoader loader;
  KernelModule module(&loader);
  EXPECT_EQ(module.Load("module.so", {"GetTilingDataSize", "AutofuseTiling", "AutofuseLaunch"}), Status::kNotFound);
  EXPECT_EQ(module.missing_symbol(), "AutofuseLaunch");
}

TEST(KernelModuleTest, RejectsUnsupportedLegacyTensorCountsBeforeCalling) {
  FakeLoader loader;
  loader.has_v2 = false;
  loader.has_legacy = true;
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("module.so", {"GetTilingDataSize", "AutofuseTiling", "AutofuseLaunch", 3, 1}), Status::kOk);
  FakeRuntime runtime_api;
  AclRuntime runtime(&runtime_api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  void *inputs[] = {reinterpret_cast<void *>(0x6), reinterpret_cast<void *>(0x7)};
  void *outputs[] = {reinterpret_cast<void *>(0x9)};
  int tiling = 0;
  FakeLoader::legacy_status = 0;
  EXPECT_EQ(module.Launch(&runtime, 8, runtime.stream(), inputs, 2, outputs, 1, nullptr, &tiling),
            Status::kInvalidArgument);
}

TEST(KernelModuleTest, CallsTypedLegacyOneInputAbiInOrder) {
  FakeLoader loader;
  loader.has_v2 = false;
  loader.has_legacy = true;
  loader.legacy_arity = 1;
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("module.so", {"GetTilingDataSize", "AutofuseTiling", "AutofuseLaunch", 1, 1}), Status::kOk);
  FakeRuntime runtime_api;
  AclRuntime runtime(&runtime_api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  void *inputs[] = {reinterpret_cast<void *>(0x6)};
  void *outputs[] = {reinterpret_cast<void *>(0x9)};
  int tiling = 0;
  FakeLoader::legacy_status = 0;
  EXPECT_EQ(module.Launch(&runtime, 8, runtime.stream(), inputs, 1, outputs, 1, reinterpret_cast<void *>(0xa), &tiling),
            Status::kOk);
  EXPECT_EQ(FakeLoader::legacy_inputs[0], inputs[0]);
  EXPECT_EQ(FakeLoader::legacy_output, outputs[0]);
}

TEST(KernelModuleTest, CallsTypedLegacyTwoInputAbiInOrder) {
  FakeLoader loader;
  loader.has_v2 = false;
  loader.has_legacy = true;
  loader.legacy_arity = 2;
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("module.so", {"GetTilingDataSize", "AutofuseTiling", "AutofuseLaunch", 2, 1}), Status::kOk);
  FakeRuntime runtime_api;
  AclRuntime runtime(&runtime_api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  void *inputs[] = {reinterpret_cast<void *>(0x6), reinterpret_cast<void *>(0x7)};
  void *outputs[] = {reinterpret_cast<void *>(0x9)};
  int tiling = 0;
  FakeLoader::legacy_status = 0;
  EXPECT_EQ(module.Launch(&runtime, 8, runtime.stream(), inputs, 2, outputs, 1, reinterpret_cast<void *>(0xa), &tiling),
            Status::kOk);
  EXPECT_EQ(FakeLoader::legacy_inputs[0], inputs[0]);
  EXPECT_EQ(FakeLoader::legacy_inputs[1], inputs[1]);
  EXPECT_EQ(FakeLoader::legacy_output, outputs[0]);
}

TEST(KernelModuleTest, PrefersV2WhenBothLaunchSymbolsExist) {
  FakeLoader loader;
  loader.has_v2 = true;
  loader.has_legacy = true;
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("module.so", {}), Status::kOk);
  FakeRuntime runtime_api;
  AclRuntime runtime(&runtime_api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  void *inputs[] = {reinterpret_cast<void *>(0x6)};
  void *outputs[] = {reinterpret_cast<void *>(0x9)};
  int tiling = 0;
  EXPECT_EQ(module.Launch(&runtime, 8, runtime.stream(), inputs, 1, outputs, 1, nullptr, &tiling), Status::kOk);
  EXPECT_EQ(FakeLoader::launch_input_count, 1);
}

TEST(KernelModuleTest, RetainsHandleAfterUnloadFailureAndRetries) {
  FakeLoader loader;
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("module.so", {}), Status::kOk);
  loader.unload_status = Status::kRuntimeError;
  EXPECT_EQ(module.Unload(), Status::kRuntimeError);
  EXPECT_TRUE(module.IsLoaded());
  size_t size = 0;
  EXPECT_EQ(module.GetTilingDataSize(&size), Status::kOk);
  loader.unload_status = Status::kOk;
  EXPECT_EQ(module.Unload(), Status::kOk);
  EXPECT_FALSE(module.IsLoaded());
  EXPECT_EQ(module.GetTilingDataSize(&size), Status::kInvalidArgument);
}

TEST(KernelModuleTest, ReportsMissingSharedObjectWithoutUnloading) {
  FakeLoader loader;
  loader.load_result = nullptr;
  KernelModule module(&loader);
  EXPECT_EQ(module.Load("missing.so", {}), Status::kNotFound);
  EXPECT_EQ(loader.unload_calls, 0);
  EXPECT_FALSE(module.IsLoaded());
}

void ExpectTilingSetupAndLaunchCapture() {
  FakeLoader loader;
  KernelModule module(&loader);
  ASSERT_EQ(module.Load("module.so", {}), Status::kOk);
  size_t size = 0;
  EXPECT_EQ(module.GetTilingDataSize(&size), Status::kOk);
  EXPECT_EQ(size, 24U);
  uint64_t workspace = 0;
  uint32_t block = 0;
  int tiling_data = 0;
  void *extra = reinterpret_cast<void *>(0x5);
  EXPECT_EQ(module.RunTiling(&tiling_data, &workspace, &block, extra), Status::kOk);
  EXPECT_EQ(FakeLoader::tiling_data, &tiling_data);
  EXPECT_EQ(FakeLoader::tiling_workspace, &workspace);
  EXPECT_EQ(FakeLoader::tiling_block, &block);
  EXPECT_EQ(FakeLoader::tiling_extra, extra);
  EXPECT_EQ(workspace, 0x120000000U);
  EXPECT_EQ(block, 8U);
  void *inputs[] = {reinterpret_cast<void *>(0x6)};
  void *outputs[] = {reinterpret_cast<void *>(0x7)};
  FakeRuntime runtime_api;
  AclRuntime runtime(&runtime_api);
  ASSERT_EQ(runtime.Initialize(0), Status::kOk);
  EXPECT_EQ(
      module.Launch(&runtime, 8, runtime.stream(), inputs, 1, outputs, 1, reinterpret_cast<void *>(0x9), &tiling_data),
      Status::kOk);
  EXPECT_EQ(FakeLoader::launch_block, 8U);
  EXPECT_EQ(FakeLoader::launch_stream, runtime.stream());
  EXPECT_EQ(FakeLoader::launch_inputs, inputs);
  EXPECT_EQ(FakeLoader::launch_input_count, 1);
  EXPECT_EQ(FakeLoader::launch_outputs, outputs);
  EXPECT_EQ(FakeLoader::launch_output_count, 1);
}

struct KernelGuardTestContext {
  FakeLoader loader;
  KernelModule module{&loader};
  FakeRuntime runtime_api;
  AclRuntime runtime{&runtime_api};
  AclRuntime uninitialized{&runtime_api};
  uint64_t workspace = 0;
  uint32_t block = 0;
  int tiling_data = 0;
  void *extra = reinterpret_cast<void *>(0x5);
  void *inputs[1] = {reinterpret_cast<void *>(0x6)};
  void *outputs[1] = {reinterpret_cast<void *>(0x7)};
  void *stream = nullptr;
};

void ExpectInvalidLaunchGuards(KernelGuardTestContext *ctx) {
  EXPECT_EQ(ctx->module.Launch(&ctx->uninitialized, 8, ctx->runtime.stream(), ctx->inputs, 1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
  FakeLoader::tiling_status = ge::GRAPH_FAILED;
  EXPECT_EQ(ctx->module.RunTiling(&ctx->tiling_data, &ctx->workspace, &ctx->block, ctx->extra), Status::kRuntimeError);
  FakeLoader::launch_status = 1;
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, ctx->runtime.stream(), ctx->inputs, 1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kRuntimeError);
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, reinterpret_cast<void *>(0x8), ctx->inputs, 1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, ctx->runtime.stream(), nullptr, -1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, ctx->runtime.stream(), nullptr, 1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, ctx->runtime.stream(), ctx->inputs, 1, nullptr, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, ctx->runtime.stream(), ctx->inputs, (1 << 20) + 1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
}

void ExpectUnloadedGuards(KernelGuardTestContext *ctx) {
  EXPECT_EQ(ctx->runtime.Finalize(), Status::kOk);
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, ctx->stream, ctx->inputs, 1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
  EXPECT_EQ(ctx->module.Unload(), Status::kOk);
  size_t size = 0;
  EXPECT_EQ(ctx->module.GetTilingDataSize(&size), Status::kInvalidArgument);
  EXPECT_EQ(ctx->module.RunTiling(&ctx->tiling_data, &ctx->workspace, &ctx->block, ctx->extra),
            Status::kInvalidArgument);
  EXPECT_EQ(ctx->module.Launch(&ctx->runtime, 8, reinterpret_cast<void *>(0x8), ctx->inputs, 1, ctx->outputs, 1,
                               reinterpret_cast<void *>(0x9), &ctx->tiling_data),
            Status::kInvalidArgument);
}

void ExpectUnloadedAndInvalidLaunchGuards() {
  KernelGuardTestContext ctx;
  ASSERT_EQ(ctx.module.Load("module.so", {}), Status::kOk);
  ASSERT_EQ(ctx.runtime.Initialize(0), Status::kOk);
  ctx.stream = ctx.runtime.stream();
  ExpectInvalidLaunchGuards(&ctx);
  ExpectUnloadedGuards(&ctx);
}

TEST(KernelModuleTest, CallsExactAbiAndProtectsUnloadedFunctions) {
  ExpectTilingSetupAndLaunchCapture();
  ExpectUnloadedAndInvalidLaunchGuards();
}
}  // namespace
}  // namespace device_validation
