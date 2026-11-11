/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/Emit/AutofuseLaunchEmitter.h"

#include "ascir/Target/Asc/CodeEmitter.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {
// `layout.arguments` is already in canonical Input(asc), Output(asc), Workspace order
// (runGenAggregateKernelEntry reorders via getKernelArgumentPermutation), so a single
// pass suffices; this just validates the workspace count while collecting names.
FailureOr<SmallVector<std::string>> buildArgumentNames(const AggregateArgumentLayout &layout) {
  SmallVector<std::string> names;
  size_t workspaceCount = 0;
  for (const KernelArgumentIdentity &argument : layout.arguments) {
    switch (argument.kind) {
      // 按值标量与 GM 张量共用一个序号空间，名字也共用一套：legacy 的
      // `GetScheduledResultInputOutput` 对两者都打 `input<i>`。
      case KernelArgumentKind::ScalarInput:
      case KernelArgumentKind::Input:
        names.push_back("input" + std::to_string(argument.index));
        break;
      case KernelArgumentKind::Output:
        names.push_back("output" + std::to_string(argument.index));
        break;
      case KernelArgumentKind::Workspace:
        names.push_back("workspace");
        ++workspaceCount;
        break;
      case KernelArgumentKind::Tiling:
        // host launch 包装器自行构造 TilingData，不从 layout.arguments 接收。
        // 若此处遇到 tiling 参数，说明上游错误地将其纳入了 launch 实参。
        return failure();
    }
  }
  if (workspaceCount != 1) {
    return failure();
  }
  return names;
}

FailureOr<std::string> emitCType(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  {
    CodeEmitter emitter(os);
    if (failed(emitter.emitType(UnknownLoc::get(type.getContext()), type))) {
      return failure();
    }
  }
  os.flush();
  return text;
}
}  // namespace

FailureOr<std::string> mlir::ascend::kernel::emitAutofuseLaunch(const AggregateArgumentLayout &layout,
                                                                ArrayRef<Type> argumentTypes, StringRef graphName,
                                                                StringRef tilingTypeName) {
  if (layout.tiling_arg_index != layout.arguments.size()) {
    return failure();
  }
  auto names = buildArgumentNames(layout);
  if (failed(names)) {
    return failure();
  }
  if (argumentTypes.size() != names->size()) {
    return failure();
  }
  SmallVector<std::string> cTypes;
  for (Type type : argumentTypes) {
    auto cType = emitCType(type);
    if (failed(cType)) {
      return failure();
    }
    cTypes.push_back(std::move(*cType));
  }

  std::string text;
  llvm::raw_string_ostream os(text);
  // host 签名和 kernel 调用均需区分 GM 指针与按值标量，与 legacy
  // GetScheduledResultInputOutput 一致：指针声明为 void*，调用时转换为 cType；
  // 标量直接声明为 cType，调用时不转换。
  // verifyCanonicalKernelArguments 保证 scalar_input 角色与非 shaped 类型对应，
  // 因此这里可按参数类型判断。
  SmallVector<bool> byValue;
  byValue.reserve(argumentTypes.size());
  for (Type type : argumentTypes) {
    byValue.push_back(!isa<ShapedType>(type));
  }

  os << "extern \"C\" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, ";
  for (auto [name, cType, scalar] : llvm::zip(*names, cTypes, byValue)) {
    os << (scalar ? cType : std::string("void*")) << " " << name << ", ";
  }
  os << tilingTypeName << "* tiling_data)" << "\n";
  os << "{" << "\n";
  os << "  " << graphName << "<<<blockDim, nullptr, stream>>>(";
  for (auto [name, cType, scalar] : llvm::zip(*names, cTypes, byValue)) {
    if (scalar) {
      os << name << ", ";
      continue;
    }
    os << "(" << cType << ")" << name << ", ";
  }
  os << "*tiling_data);" << "\n";
  os << "  return 0;" << "\n";
  os << "}" << "\n";
  return text;
}
