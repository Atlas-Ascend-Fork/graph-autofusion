/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AscendCAttributes.h"

#include "target/Ascend/AscendTargetProfile.h"

#include <cassert>
#include <optional>

#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"

namespace mlir::ascend::kernel {

FailureOr<ascend::MemorySpace> toTargetMemorySpace(afir::Space space) {
  switch (space) {
    case afir::Space::GM:
      return ascend::MemorySpace::GM;
    case afir::Space::UB:
      return ascend::MemorySpace::UB;
    case afir::Space::L1:
      return ascend::MemorySpace::L1;
    case afir::Space::L2:
      return ascend::MemorySpace::L2;
    case afir::Space::L0A:
      return ascend::MemorySpace::L0A;
    case afir::Space::L0B:
      return ascend::MemorySpace::L0B;
    case afir::Space::L0C:
      return ascend::MemorySpace::L0C;
  }
  return failure();
}

FailureOr<ascend::MemorySpace> getValueMemorySpace(Value value) {
  // 先读类型、再回落 descriptor，与 `CanonicalBindings.cpp` 的 `resolveValueSpace` 同一条
  // 判据。两份此前不一样：这里只读 descriptor，于是同一个值在两处可能得到不同的空间——
  // 类型上写着 `#afir.space<ub>`、descriptor 却没标或标了别的时，规划侧按类型归组、取用侧
  // 按 descriptor 发形态，两边对不上而没人报错。
  //
  // `memref` 的空间由类型自带，是最权威的一手事实；AscendC 的 local/global tensor 类型
  // 不带空间属性，`getMemorySpace` 返回空，这时才问 descriptor。
  if (!value) {
    return failure();
  }
  if (std::optional<afir::Space> space = afir::getMemorySpace(value.getType())) {
    return toTargetMemorySpace(*space);
  }
  afir::BufferDescAttr descriptor = afir::getBufferDescAttr(value);
  if (!descriptor || !descriptor.getSpace()) {
    return failure();
  }
  return toTargetMemorySpace(descriptor.getSpace().getValue());
}

}  // namespace mlir::ascend::kernel

namespace mlir::ascend::kernel {
namespace {

FailureOr<ascend::AICoreUnit> parseAscendCExecutionUnit(llvm::StringRef value) {
  if (value == "AiCore.Vector") {
    return ascend::AICoreUnit::Vector;
  }
  return failure();
}

FailureOr<llvm::StringRef> stringifyAscendCExecutionUnit(ascend::AICoreUnit unit) {
  switch (unit) {
    case ascend::AICoreUnit::Vector:
      return llvm::StringRef("AiCore.Vector");
    case ascend::AICoreUnit::Cube:
      return failure();
  }
  return failure();
}

FailureOr<ascend::ExecutionModel> parseExecutionModel(llvm::StringRef value) {
  if (value == "SIMD") {
    return ascend::ExecutionModel::SIMD;
  }
  if (value == "SIMT") {
    return ascend::ExecutionModel::SIMT;
  }
  return failure();
}

llvm::StringRef stringifyExecutionModel(ascend::ExecutionModel model) {
  return model == ascend::ExecutionModel::SIMD ? "SIMD" : "SIMT";
}

FailureOr<ascend::InstructionForm> parseInstructionForm(llvm::StringRef value) {
  if (value == "MemoryBased") {
    return ascend::InstructionForm::MemoryBased;
  }
  if (value == "RegBased") {
    return ascend::InstructionForm::RegBased;
  }
  return failure();
}

llvm::StringRef stringifyInstructionForm(ascend::InstructionForm form) {
  return form == ascend::InstructionForm::MemoryBased ? "MemoryBased" : "RegBased";
}

}  // namespace

FailureOr<ascend::ExecutionSpec> getExecutionSpec(Operation *op) {
  if (!op) {
    return failure();
  }
  auto unit = op->getAttrOfType<StringAttr>(kAscendCExecutionUnitAttrName);
  auto executionModel = op->getAttrOfType<StringAttr>(kAscendCExecutionModelAttrName);
  Attribute instructionFormAttr = op->getAttr(kAscendCInstructionFormAttrName);
  if (!unit || !executionModel || (instructionFormAttr && !isa<StringAttr>(instructionFormAttr))) {
    return failure();
  }
  FailureOr<ascend::AICoreUnit> parsedUnit = parseAscendCExecutionUnit(unit.getValue());
  FailureOr<ascend::ExecutionModel> parsedModel = parseExecutionModel(executionModel.getValue());
  if (failed(parsedUnit) || failed(parsedModel)) {
    return failure();
  }
  std::optional<ascend::InstructionForm> parsedForm;
  if (auto instructionForm = dyn_cast_if_present<StringAttr>(instructionFormAttr)) {
    FailureOr<ascend::InstructionForm> form = parseInstructionForm(instructionForm.getValue());
    if (failed(form)) {
      return failure();
    }
    parsedForm = *form;
  }
  ascend::ExecutionSpec spec{ascend::ProcessorType::AICore, *parsedUnit, *parsedModel, parsedForm};
  if (failed(ascend::validateExecutionSpec(spec))) {
    return failure();
  }
  return spec;
}

LogicalResult setExecutionSpec(Operation *op, const ascend::ExecutionSpec &spec) {
  if (!op || failed(ascend::validateExecutionSpec(spec)) || spec.processor != ascend::ProcessorType::AICore ||
      !spec.unit || !spec.executionModel) {
    return failure();
  }
  FailureOr<llvm::StringRef> unit = stringifyAscendCExecutionUnit(*spec.unit);
  if (failed(unit)) {
    return failure();
  }
  MLIRContext *context = op->getContext();
  StringAttr unitAttr = StringAttr::get(context, *unit);
  StringAttr executionModelAttr = StringAttr::get(context, stringifyExecutionModel(*spec.executionModel));
  std::optional<StringAttr> instructionFormAttr;
  if (spec.instructionForm) {
    instructionFormAttr = StringAttr::get(context, stringifyInstructionForm(*spec.instructionForm));
  }

  op->setAttr(kAscendCExecutionUnitAttrName, unitAttr);
  op->setAttr(kAscendCExecutionModelAttrName, executionModelAttr);
  if (instructionFormAttr) {
    op->setAttr(kAscendCInstructionFormAttrName, *instructionFormAttr);
  } else {
    op->removeAttr(kAscendCInstructionFormAttrName);
  }
  return success();
}

FailureOr<bool> hasAscendCAiCoreAttr(Operation *op) {
  if (!op) {
    return failure();
  }
  Attribute attr = op->getAttr(ascendc::attr::aicore);
  if (!attr) {
    return false;
  }
  if (!isa<UnitAttr>(attr)) {
    return failure();
  }
  return true;
}

void setAscendCAiCore(Operation *op) {
  assert(op && "expected a valid operation");
  op->setAttr(ascendc::attr::aicore, UnitAttr::get(op->getContext()));
}

FailureOr<IntegerAttr> getAscendCNumInputs(Operation *op) {
  if (!op) {
    return failure();
  }
  Attribute attr = op->getAttr(kAscendCNumInputsAttrName);
  if (!attr) {
    return IntegerAttr{};
  }
  auto inputCount = dyn_cast<IntegerAttr>(attr);
  if (!inputCount) {
    return failure();
  }
  return inputCount;
}

void setAscendCNumInputs(Operation *op, unsigned inputCount) {
  assert(op && "expected a valid operation");
  Builder builder(op->getContext());
  op->setAttr(kAscendCNumInputsAttrName, builder.getI32IntegerAttr(inputCount));
}

FailureOr<bool> hasAscendCGlobalAttr(Operation *op) {
  if (!op) {
    return failure();
  }
  Attribute attr = op->getAttr(ascendc::attr::global);
  if (!attr) {
    return false;
  }
  if (!isa<UnitAttr>(attr)) {
    return failure();
  }
  return true;
}

void setAscendCGlobal(Operation *op, bool enabled) {
  assert(op && "expected a valid operation");
  if (enabled) {
    op->setAttr(ascendc::attr::global, UnitAttr::get(op->getContext()));
    return;
  }
  op->removeAttr(ascendc::attr::global);
}

}  // namespace mlir::ascend::kernel
