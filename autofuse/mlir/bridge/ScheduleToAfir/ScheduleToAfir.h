/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ScheduleToAfir.h - ScheduledResult → AFIR (in-process) -*- C++ -*-===//
//
// Autofuse MLIR in-process direct bridge.
//
// Converts one real ascir::ScheduledResult (a plan) into one AFIR MLIR
// module, with zero serialization. The enclosing FusedScheduledResult
// supplies the shared graph name, origin shape variables and ShapeEnv guards
// only.
//
// Hooked at codegen.cpp's Codegen::GenerateForInductor, gated by
// ENABLE_AUTOFUSE_MLIR. Default off (no legacy behavior change);
// dumping is driven by the shared AUTOFUSE_DFX_FLAGS configuration.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOFUSE_MLIR_BRIDGE_SCHEDULETOAFIR_H
#define AUTOFUSE_MLIR_BRIDGE_SCHEDULETOAFIR_H

#include <cstddef>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "schedule_result.h"  // ascir::FusedScheduledResult

namespace mlir {
namespace afir {
namespace bridge {

// The initial Schedule buffer-form path is limited to operations whose
// destination-style lowering is complete.
bool isAFIRBufferFormOperationSupported(llvm::StringRef scheduler_name);

namespace detail {

// Convert scheduler enum values to the canonical spellings consumed by
// AfirBuilder. Unsupported values return std::nullopt instead of being silently
// coerced to a valid AFIR type or axis kind.
std::optional<std::string> DTypeStr(ge::DataType dtype);

// 检查 dtype 是否属于 bridge 当前支持的 AFIR 类型域，与 DTypeStr 的名称转换分开。
// i8 / i64 / ui32 / ui64 等未建模类型应提前返回不支持，不能等到 ODS 校验才编译失败。
// 建模类型由 support/AfirElementTypes.h 的 AFIR_ELEMENT_TYPE_TABLE 统一定义，
// 暂缓接入的类型及原因列于实现中的 AFIR_ELEMENT_TYPE_HELD_BACK，避免头文件说明与
// 实际接入状态分别维护。注册位、ODS、字节宽度及比对断言均从公共表生成。
bool IsAfirModeledDType(ge::DataType dtype);

// 暂缓名单上的一位：建模得出来，但 bridge 先不收，以及为什么。
//
// 唯一的读点是 `ScheduleToAfirTest` 那条断言——它查名单上每一行都还是表上的一位，否则
// 那一行是句死话，而死话会一直替它挡着。
struct HeldBackDType {
  uint32_t bit;
  llvm::StringRef reason;
};
llvm::ArrayRef<HeldBackDType> HeldBackDTypes();
std::optional<std::string> AxisTypeStr(af::Axis::Type type);
std::optional<std::string> ExecConditionStr(af::ExecuteCondition condition);
std::optional<std::string> MakeSizeBytesExpr(llvm::ArrayRef<std::string> repeats, int64_t element_bytes);

}  // namespace detail

// Build one AFIR module directly from one real ScheduledResult (a plan).
// Returns null on error (diagnostics emitted to the context).
mlir::OwningOpRef<mlir::ModuleOp> PlanToAfir(mlir::MLIRContext &context,
                                             const ascir::FusedScheduledResult &fused_result,
                                             const ascir::ScheduledResult &scheduled_result, size_t subgraph_id,
                                             size_t plan_id,
                                             llvm::ArrayRef<std::vector<size_t>> selected_templates = {});

// If the shared DFX dump configuration is enabled, serialize an already-built AFIR module
// (as returned by PlanToAfir) to a file under that directory. The file name is
// derived from the module's own symbol name, which PlanToAfir already set from
// the subgraph/plan identity. Call once per successfully-built module; avoids
// re-running PlanToAfir just to dump. No-op when dump_root is empty. Never
// throws into the codegen path.
void MaybeDumpAfirModule(const std::string &stage, mlir::ModuleOp module, const std::string &dump_root);

}  // namespace bridge
}  // namespace afir
}  // namespace mlir

#endif  // AUTOFUSE_MLIR_BRIDGE_SCHEDULETOAFIR_H
