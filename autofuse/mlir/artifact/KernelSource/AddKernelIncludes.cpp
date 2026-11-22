/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/AddKernelIncludes.h"
#include "artifact/KernelSource/Emit/ApiExtensionOrder.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"

#include "dialect/AFIR/IR/PlanMetadata.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <set>
#include <string>

using namespace mlir;
using namespace mlir::ascend::kernel;
using namespace mlir::afir;

namespace {
constexpr llvm::StringLiteral kKernel = "kernel_operator.h";
constexpr llvm::StringLiteral kTiling = "autofuse_tiling_data.h";
constexpr llvm::StringLiteral kCstdint = "cstdint";
constexpr llvm::StringLiteral kListTensor = "kernel_operator_list_tensor_intf.h";
constexpr llvm::StringLiteral kMatmul = "lib/matmul_intf.h";
// Autofuse 扩展头沿用 legacy kernel 前导，可直接使用 LocalTensor、BinaryRepeatParams
// 和 ONE_REPEAT_BYTE_SIZE 等 AscendC 名称，因此内联前需要 using namespace AscendC。
// legacy 的 IncludeAndDefines 无条件写入此声明，这里只在使用扩展头时写入。
// abs.h / neg.h 自带完整限定名，不能覆盖缺少声明的情况；其他扩展头会在 device 编译时报错。
constexpr llvm::StringLiteral kAscendCUsingDirective = "using namespace AscendC;";

struct IncludeSpec {
  llvm::StringRef path;
  bool isStandardInclude;
};

struct Requirements {
  bool hasListTensor = false;
  bool hasMatmul = false;
  // Autofuse API extension headers declared by operations in the module, together
  // with the order constraints those declarations carry. Header names are globally
  // unique across the mem-based and reg-based families, so this stays
  // platform-agnostic: whichever implementation was selected has already decided
  // which name lands here.
  ApiExtensionOrder extensionHeaders;
  // 前导头（今天只有 `utils.h` 一族）要排在**所有**扩展头前面——这比扩展头之间那种两两
  // 先后更强，所以它单独一张表，而不是当作每个 op 的头列表的第一项：混进去就只能保证它排在
  // 同一个 op 的头前面，排不到别的 op 的头前面（`utils.h` 字典序在 `reciprocal.h` 之后）。
  // 哪个扩展头需要哪个前导由 op 自己声明，这一层照样只收名字。前导彼此之间同样可以有先后
  // （`bessel_j0` 要 `bessel_j_utils` 排在 `trigonometric_function_utils` 之前），所以这张表
  // 与扩展头那张走同一套排序。
  ApiExtensionOrder preludeHeaders;
  // CANN 官方头（`ApiExtensionHeader` 第三段列表）：真头文件，由 device 编译器的 `-I`
  // 解析，不内联文本。它们是自足翻译单元，没有扩展头那种先后约束，也不参与家族判别
  // （CANN 头不是哪张供给表的标记），所以只收名字、字典序发。
  std::set<llvm::StringRef> officialHeaders;
};

bool ValidateAggregateModule(mlir::ModuleOp module) {
  auto schema = module->getAttrOfType<mlir::TypeAttr>(mlir::ascend::kernel::kAscendCTilingSchemaAttrName);
  return mlir::afir::isEntryModule(module) && schema && mlir::ascend::kernel::getTilingSchema(schema.getValue());
}

void CollectRequirements(mlir::ModuleOp module, Requirements &requirements) {
  module.walk([&](mlir::Operation *operation) {
    requirements.hasListTensor |=
        mlir::isa<mlir::ascendc::ListTensorDescOp, mlir::ascendc::ListTensorDescV2Op, mlir::ascendc::TensorDescOp>(
            operation);
    requirements.hasMatmul |= mlir::isa<mlir::ascendc::RegistMatmulObjOp>(operation);
    if (auto extension = mlir::dyn_cast<mlir::ascendc::ApiExtensionOp>(operation)) {
      requirements.extensionHeaders.declare(operation, extension.getExtensionHeaders());
      requirements.preludeHeaders.declare(operation, extension.getExtensionPreludeHeaders());
      const llvm::ArrayRef<llvm::StringRef> officials = extension.getExtensionOfficialIncludes();
      requirements.officialHeaders.insert(officials.begin(), officials.end());
    }
  });
}

// Rewrite fixed includes and extension headers currently declared by the module.
// Leave undeclared extension includes untouched; this pass does not own them.
bool IsManaged(mlir::emitc::IncludeOp include, const Requirements &requirements) {
  llvm::StringRef path = include.getInclude();
  return path == kKernel || path == kTiling || path == kCstdint || path == kListTensor || path == kMatmul ||
         requirements.preludeHeaders.contains(path) || requirements.extensionHeaders.contains(path) ||
         requirements.officialHeaders.count(path) != 0;
}

void AddKernelIncludes(mlir::ModuleOp module, const Requirements &requirements,
                       llvm::ArrayRef<llvm::StringRef> preludeOrder, llvm::ArrayRef<llvm::StringRef> extensionOrder) {
  constexpr std::array<IncludeSpec, 5> include_specs = {
      {{kKernel, false}, {kTiling, false}, {kCstdint, true}, {kListTensor, false}, {kMatmul, false}}};
  llvm::SmallVector<mlir::Operation *> existing_managed;
  module.walk([&](mlir::emitc::IncludeOp include) {
    if (!IsManaged(include, requirements)) {
      return;
    }
    existing_managed.push_back(include);
  });
  module.walk([&](mlir::emitc::VerbatimOp verbatim) {
    if (verbatim.getValue() == kAscendCUsingDirective) {
      existing_managed.push_back(verbatim);
    }
  });
  for (mlir::Operation *existing : existing_managed) {
    existing->erase();
  }

  mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  for (const IncludeSpec &include_spec : include_specs) {
    bool needed = include_spec.path == kKernel || include_spec.path == kTiling || include_spec.path == kCstdint;
    needed |= include_spec.path == kListTensor && requirements.hasListTensor;
    needed |= include_spec.path == kMatmul && requirements.hasMatmul;
    if (needed) {
      builder.create<mlir::emitc::IncludeOp>(include_spec.path, include_spec.isStandardInclude);
    }
  }
  // CANN 官方头：固定块之后、扩展头之前——与 legacy 的 include 顺序一致。它们是
  // device 编译器 `-I` 解析的真头文件（不内联文本），也不需要那条 using-directive。
  for (llvm::StringRef official : requirements.officialHeaders) {
    builder.create<mlir::emitc::IncludeOp>(official, /*isStandardInclude=*/false);
  }
  // After the fixed block, so an extension header can rely on kernel_operator.h
  // having been included, matching the order legacy codegen emits them in.
  //
  // 前导只在真的有扩展头时发，没有扩展头的 kernel 一个字节都不变。顺序照 legacy：
  // using-directive 在最前，因为 `utils.h` 自己也用了不带限定的 `LocalTensor`。各扩展共用一条 directive。
  if (!requirements.extensionHeaders.empty()) {
    builder.create<mlir::emitc::VerbatimOp>(kAscendCUsingDirective);
  }
  for (llvm::StringRef prelude : preludeOrder) {
    builder.create<mlir::emitc::IncludeOp>(prelude, /*isStandardInclude=*/false);
  }
  for (llvm::StringRef header : extensionOrder) {
    builder.create<mlir::emitc::IncludeOp>(header, /*isStandardInclude=*/false);
  }
}
}  // namespace

mlir::LogicalResult mlir::ascend::kernel::runAddKernelIncludes(mlir::ModuleOp entryModule) {
  if (!ValidateAggregateModule(entryModule)) {
    return entryModule.emitError("runAddKernelIncludes requires a valid aggregate module");
  }

  Requirements requirements;
  CollectRequirements(entryModule, requirements);
  // 两张表各自排序。排不出来（成环）时诊断已经报在声明它的那个 op 上，这里只把失败传出去。
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> preludeOrder = requirements.preludeHeaders.resolve();
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef>> extensionOrder = requirements.extensionHeaders.resolve();
  if (failed(preludeOrder) || failed(extensionOrder)) {
    return mlir::failure();
  }
  AddKernelIncludes(entryModule, requirements, *preludeOrder, *extensionOrder);
  return mlir::success();
}
