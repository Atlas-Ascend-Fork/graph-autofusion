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
        // host 侧的 launch 包装器自己构造 TilingData，不从实参表里接——它不在
        // `layout.arguments` 里，走到这里说明上游把 ABI 参数也收进来了。
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
  // 按值标量形参与 GM 指针形参在 host 侧的写法不同，两处都要分流（legacy 的
  // `GetScheduledResultInputOutput` 是同一套写法）：
  //
  //   | | 签名里 | 调用 kernel 时 |
  //   | GM 张量 | `void* input<i>` | `(<cType>)input<i>` |
  //   | 按值标量 | `<cType> input<i>` | `input<i>`（**不 cast**） |
  //
  // 「是不是按值标量」按参数类型判：ABI 层已经把它与 `scalar_input` 角色绑成一条双向
  // 不变量（`verifyCanonicalKernelArguments`），到这里两者等价。
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
