/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRDialect.cpp - AFIR dialect implementation -------------*- C++ -*-===//
//
// AFIR dialect registration and custom attribute parsing/printing.
//
//===----------------------------------------------------------------------===//

#include "dialect/AFIR/IR/AFIR.h"
#include "common/tiling_schema/canonical_resource_names.h"
#include "dialect/AFIR/IR/AFIRElementwiseSupport.h"
#include "dialect/AFIR/IR/AFIRVerifierUtils.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace {

bool hasAxisDefinition(mlir::Operation *owner, int64_t axisId) {
  mlir::func::FuncOp function = mlir::dyn_cast<mlir::func::FuncOp>(owner);
  if (!function) {
    function = owner->getParentOfType<mlir::func::FuncOp>();
  }
  if (!function) {
    return false;
  }
  auto graph = function->getAttrOfType<mlir::afir::AscGraphAttrGroupsAttr>(mlir::afir::kAscGraphAttrName);
  if (!graph) {
    return false;
  }
  return llvm::any_of(graph.getAxes(), [axisId](mlir::afir::AxisAttr axis) { return axis.getId() == axisId; });
}

// `subject` 已带好引号与定位后缀（`'afir.workspace_plan' element 0` /
// `'afir.buffer_desc' at argument 0`），因为同一份 descriptor 现在挂在三种载体上。
mlir::LogicalResult verifyLifeTimeAxis(mlir::Operation *op, const llvm::Twine &subject,
                                       mlir::afir::BufferDescAttr buffer) {
  int64_t lifeTimeAxisId = buffer.getLifeTimeAxisId();
  if (lifeTimeAxisId < 0 || hasAxisDefinition(op, lifeTimeAxisId)) {
    return mlir::success();
  }
  return op->emitError() << mlir::afir::kInvalidLifetimeDiagnostic << ": " << subject
                         << " references unknown life_time_axis_id " << lifeTimeAxisId;
}

}  // namespace

namespace {

bool isKnownArgOrResultAttribute(mlir::StringRef name) {
  static constexpr mlir::StringLiteral known[] = {mlir::afir::kSymbolicShapeAttrName, mlir::afir::kKernelArgAttrName,
                                                  mlir::afir::kBufferDescAttrName};
  return llvm::is_contained(known, name);
}

mlir::LogicalResult verifyFunctionArgOrResultAttribute(mlir::Operation *op, unsigned index, mlir::StringRef kind,
                                                       mlir::NamedAttribute attribute) {
  mlir::StringRef name = attribute.getName().getValue();
  if (!name.starts_with(mlir::afir::kAFIRAttrPrefix)) {
    return mlir::success();
  }
  if (!isKnownArgOrResultAttribute(name)) {
    return op->emitError() << mlir::afir::kUnsupportedOperationDiagnostic << ": unknown AFIR function " << kind
                           << " attribute '" << attribute.getName() << "' at index " << index;
  }
  // 跨参数的约束（序号在角色内不重复、workspace / tiling 至多各一个）在这里查，而不是
  // 只留给某个下游 pass：A3 的立论是「旧机制两个都挂是合法 IR，只能在消费端事后发现」
  // ——如果新机制的跨参数约束同样要等到 memory-allocate 跑过才生效，那就是同一个形态
  // 换了个名字。这条路径在**解析期**就跑，两个 `role = workspace` 的参数进不到任何
  // 消费端。
  //
  // 只回头看 `argIndex` 之前的参数：本函数会被每个参数各调一次，往回查就够覆盖所有
  // 组合，也避免同一对冲突报两遍。
  if (name == mlir::afir::kKernelArgAttrName) {
    if (kind != "argument") {
      return mlir::success();  // 结果位上的角色声明在 tensor 形态下是合法的中间形态。
    }
    auto declared = mlir::dyn_cast<mlir::afir::KernelArgAttr>(attribute.getValue());
    if (!declared) {
      return op->emitError() << mlir::afir::kDescriptorTypeMismatchDiagnostic << ": '" << name << "' at " << kind << " "
                             << index << " must be a #" << mlir::afir::AFIRDialect::getDialectNamespace() << "."
                             << mlir::afir::KernelArgAttr::getMnemonic();
    }
    auto function = mlir::dyn_cast<mlir::FunctionOpInterface>(op);
    for (unsigned earlier = 0; function && earlier < index && earlier < function.getNumArguments(); ++earlier) {
      mlir::afir::KernelArgAttr other = mlir::afir::getKernelArgAttr(function, earlier);
      if (other == declared) {
        return op->emitError() << mlir::afir::kInvalidDescriptorDiagnostic << ": duplicate kernel_arg role '"
                               << mlir::afir::stringifyKernelArgRole(declared.getRole()) << "' at argument " << index
                               << "; argument " << earlier << " already declares it";
      }
    }
    return mlir::success();
  }
  // kernel 参数与函数结果是值的定义点之一，规划结果就挂在这里；`verifyOperationAttribute`
  // 看不到 arg / result attr，所以同样的强类型与生命周期检查必须在这条路径上再做一次。
  if (name != mlir::afir::kBufferDescAttrName) {
    return mlir::success();
  }
  auto descriptor = mlir::dyn_cast<mlir::afir::BufferDescAttr>(attribute.getValue());
  if (!descriptor) {
    return op->emitError() << mlir::afir::kDescriptorTypeMismatchDiagnostic << ": '" << name << "' at " << kind << " "
                           << index << " must be a #" << mlir::afir::AFIRDialect::getDialectNamespace() << "."
                           << mlir::afir::BufferDescAttr::getMnemonic();
  }
  return verifyLifeTimeAxis(op, "'" + name + "' at " + kind + " " + llvm::Twine(index), descriptor);
}

// 只登记 `afir.` 前缀的函数属性。候选身份（`afplan.variant` / `afplan.score_func`）
// 迁出 `afir` 之后不再经过这张表，由 `AFPlanDialect::verifyOperationAttribute` 自己把关。
bool isKnownFunctionAttribute(mlir::StringRef name) {
  static constexpr mlir::StringLiteral known[] = {
      mlir::afir::kArgDescsAttrName,          mlir::afir::kResultDescsAttrName,     mlir::afir::kWorkspacePlanAttrName,
      mlir::afir::kAscGraphAttrName,          mlir::afir::kFunctionRoleAttrName,    mlir::afir::kDimSymbolsAttrName,
      mlir::afir::kOriginalGraphNameAttrName, mlir::afir::kDoubleBufferAttrName,    mlir::afir::kQueueIdsAttrName,
      mlir::afir::kBufferIdsAttrName,         mlir::afir::kApiTilingFieldsAttrName, mlir::afir::kWorkspaceSizeAttrName,
      mlir::afir::kAxisBindingsAttrName};
  return llvm::is_contained(known, name);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Buffer descriptor lookup: one copy, at the value's definition
//===----------------------------------------------------------------------===//

namespace {

// 函数结果位上的 descriptor。tensor 形态里 kernel 输出没有别的定义点——产出它的 Op 只
// 描述自己那个节点的落点（调度器给的暂存缓冲），而这个值真正所在的存储是 ABI 说的
// 那块。`hasBufferDescriptors` 早就把 result attr 当载体，查询侧漏掉它就会对同一个值
// 给出两种答案。
mlir::afir::BufferDescAttr getFunctionResultBufferDescAttr(mlir::Value value) {
  mlir::Region *region = value.getParentRegion();
  auto function = region ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(region->getParentOp()) : nullptr;
  if (!function || function.getBody().empty() || !function.front().mightHaveTerminator()) {
    return {};
  }
  auto terminator = mlir::dyn_cast<mlir::func::ReturnOp>(function.front().getTerminator());
  if (!terminator) {
    return {};
  }
  for (auto [index, returned] : llvm::enumerate(terminator->getOperands())) {
    if (returned == value && index < function.getNumResults()) {
      return function.getResultAttrOfType<mlir::afir::BufferDescAttr>(index, mlir::afir::kBufferDescAttrName);
    }
  }
  return {};
}

}  // namespace

mlir::OperandRange mlir::afir::OperandPartition::inputs(mlir::Operation *op) const {
  return op->getOperands().take_front(inputCount);
}

mlir::OperandRange mlir::afir::OperandPartition::destinations(mlir::Operation *op) const {
  return op->getOperands().drop_front(inputCount).take_front(destinationCount);
}

mlir::FailureOr<mlir::afir::OperandPartition> mlir::afir::getOperandPartition(mlir::Operation *op) {
  OperandPartition partition;
  if (auto destinationStyle = mlir::dyn_cast<AFIRDestinationStyleInterface>(op)) {
    partition.destinationCount = destinationStyle.getNumDestinations();
    partition.tmpBufferCount = destinationStyle.getNumTmpBuffers();
  }
  if (partition.destinationCount + partition.tmpBufferCount > op->getNumOperands()) {
    return mlir::failure();
  }
  partition.inputCount = op->getNumOperands() - partition.destinationCount - partition.tmpBufferCount;
  return partition;
}

mlir::Value mlir::afir::getTiedDestination(mlir::Value value) {
  auto destinationStyle = mlir::dyn_cast_or_null<AFIRDestinationStyleInterface>(value.getDefiningOp());
  if (!destinationStyle) {
    return {};
  }
  mlir::Operation *op = destinationStyle.getOperation();
  FailureOr<OperandPartition> partition = getOperandPartition(op);
  if (failed(partition) || partition->destinationCount == 0) {
    return {};
  }
  OperandRange destinations = partition->destinations(op);
  for (unsigned index = 0; index < partition->destinationCount; ++index) {
    FailureOr<unsigned> tied = destinationStyle.getTiedResultIndex(index);
    if (succeeded(tied) && *tied < op->getNumResults() && op->getResult(*tied) == value) {
      return destinations[index];
    }
  }
  return {};
}

mlir::afir::BufferDescAttr mlir::afir::getBufferDescAttr(mlir::Value value) {
  if (!value) {
    return {};
  }
  if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(blockArgument.getOwner()->getParentOp());
    if (!function || blockArgument.getOwner() != &function.front()) {
      return {};
    }
    return function.getArgAttrOfType<BufferDescAttr>(blockArgument.getArgNumber(), kBufferDescAttrName);
  }

  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return {};
  }
  // buffer 形态：结果与它绑定的 destination 是同一块存储，descriptor 只在
  // destination 的定义点存一份。迁移前这里是两份必须相等的拷贝。
  if (Value tied = getTiedDestination(value)) {
    return getBufferDescAttr(tied);
  }
  if (auto descriptor = definingOp->getAttrOfType<BufferDescAttr>(kBufferDescAttrName)) {
    return descriptor;
  }
  return getFunctionResultBufferDescAttr(value);
}

mlir::FailureOr<mlir::afir::BufferDescAttr> mlir::afir::requireBufferDescAttr(mlir::Value value,
                                                                              mlir::Operation *reporter) {
  BufferDescAttr descriptor = getBufferDescAttr(value);
  if (descriptor) {
    return descriptor;
  }
  if (!reporter && !value) {
    return mlir::failure();
  }
  mlir::InFlightDiagnostic diagnostic = reporter ? reporter->emitError() : mlir::emitError(value.getLoc());
  diagnostic << kInvalidDescriptorDiagnostic << ": value has no '" << kBufferDescAttrName << "' at its definition";
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// Kernel argument classification
//===----------------------------------------------------------------------===//

mlir::afir::KernelArgAttr mlir::afir::getKernelArgAttr(mlir::FunctionOpInterface function, unsigned index) {
  if (index >= function.getNumArguments()) {
    return {};
  }
  return function.getArgAttrOfType<KernelArgAttr>(index, kKernelArgAttrName);
}

void mlir::afir::setKernelArgAttr(mlir::FunctionOpInterface function, unsigned index, KernelArgRole role,
                                  int64_t argIndex) {
  function.setArgAttr(index, kKernelArgAttrName, KernelArgAttr::get(function.getContext(), role, argIndex));
}

std::optional<unsigned> mlir::afir::findKernelArgument(mlir::FunctionOpInterface function, KernelArgRole role) {
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    KernelArgAttr argument = getKernelArgAttr(function, index);
    if (argument && argument.getRole() == role) {
      return index;
    }
  }
  return std::nullopt;
}

bool mlir::afir::hasKernelArgRoles(mlir::FunctionOpInterface function) {
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (getKernelArgAttr(function, index)) {
      return true;
    }
  }
  return false;
}

mlir::LogicalResult mlir::afir::verifyKernelArgumentRoles(mlir::FunctionOpInterface function) {
  llvm::SmallDenseSet<std::pair<uint8_t, int64_t>> seen;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    KernelArgAttr argument = getKernelArgAttr(function, index);
    if (!argument) {
      continue;
    }
    // 不带序号的角色用同一个哨兵（-1）进集合——于是「第二个 workspace」、「两个都不带号的
    // tiling」与「两个 input 抢同一个序号」走的是同一条判重路径。
    if (!seen.insert({static_cast<uint8_t>(argument.getRole()), argument.getIndex()}).second) {
      return function->emitError() << kInvalidDescriptorDiagnostic << ": duplicate kernel_arg role '"
                                   << stringifyKernelArgRole(argument.getRole()) << "' at argument " << index;
    }
  }
  return mlir::success();
}

bool mlir::afir::hasBufferDescriptors(mlir::Operation *function) {
  auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(function);
  if (!func) {
    return false;
  }
  for (unsigned index = 0; index < func.getNumArguments(); ++index) {
    if (func.getArgAttr(index, kBufferDescAttrName)) {
      return true;
    }
  }
  for (unsigned index = 0; index < func.getNumResults(); ++index) {
    if (func.getResultAttr(index, kBufferDescAttrName)) {
      return true;
    }
  }
  mlir::WalkResult walk = func.walk([&](mlir::Operation *operation) {
    return operation->hasAttr(kBufferDescAttrName) ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return walk.wasInterrupted();
}

void mlir::afir::setBufferDescAttr(mlir::Operation *definingOp, mlir::afir::BufferDescAttr descriptor) {
  if (definingOp && descriptor) {
    definingOp->setAttr(kBufferDescAttrName, descriptor);
  }
}

//===----------------------------------------------------------------------===//
// #afir.axis 的可缺省尾部：align / from / split_pair_other_id
//===----------------------------------------------------------------------===//
//
// 三项各自可缺省，且在文本里都以同一个 `,` 开头。ODS 的可选组按组内**首个字面量**
// 判断该组是否出现，三组的首字面量都是逗号，于是解析器一见到逗号就锁定 `align`，
// 后面写 `from` 会报 "expected 'align'"。打印方向没有这个问题，所以症状是**打印得出、
// 读不回**——一个带 `from`、不带 `align` 的轴打印出来再解析就失败。
//
// 这里逐项按关键字识别，产出的文本与原 ODS 形态逐字一致（包括嵌套属性打印成
// `<"1">` 这种去前缀形式），因此没有任何既有 IR 需要跟着改。

namespace mlir::afir {

static ::mlir::ParseResult parseAxisTail(::mlir::AsmParser &parser, ExprAttr &align,
                                         ::llvm::SmallVectorImpl<int64_t> &from, int64_t &splitPairOtherId) {
  align = ExprAttr();
  splitPairOtherId = -1;

  bool seenAlign = false;
  bool seenFrom = false;
  bool seenSplitPair = false;
  while (::mlir::succeeded(parser.parseOptionalComma())) {
    if (!seenAlign && !seenFrom && !seenSplitPair && ::mlir::succeeded(parser.parseOptionalKeyword("align"))) {
      ::mlir::FailureOr<ExprAttr> parsed =
          (::mlir::failed(parser.parseEqual()) ? ::mlir::FailureOr<ExprAttr>(::mlir::failure())
                                               : ::mlir::FieldParser<ExprAttr>::parse(parser));
      if (::mlir::failed(parsed)) {
        return parser.emitError(parser.getCurrentLocation(), "expected an #afir.expr alignment expression");
      }
      align = *parsed;
      seenAlign = true;
      continue;
    }
    if (!seenFrom && !seenSplitPair && ::mlir::succeeded(parser.parseOptionalKeyword("from"))) {
      if (::mlir::failed(parser.parseEqual()) ||
          ::mlir::failed(parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::Square,
                                                        [&]() { return parser.parseInteger(from.emplace_back()); }))) {
        return ::mlir::failure();
      }
      seenFrom = true;
      continue;
    }
    if (!seenSplitPair && ::mlir::succeeded(parser.parseOptionalKeyword("split_pair_other_id"))) {
      if (::mlir::failed(parser.parseEqual()) || ::mlir::failed(parser.parseInteger(splitPairOtherId))) {
        return ::mlir::failure();
      }
      seenSplitPair = true;
      continue;
    }
    // 逗号后面不是这三个关键字中还没出现过的那个：要么拼错，要么顺序颠倒，要么重复。
    // 这三种都不能当成「尾部结束」放过去——那样后面的 `>` 会先失败，诊断指向的位置
    // 与真正的错误差一整个属性。
    return parser.emitError(parser.getCurrentLocation(),
                            "expected 'align', 'from' or 'split_pair_other_id', each at most once and in that order");
  }
  return ::mlir::success();
}

static void printAxisTail(::mlir::AsmPrinter &printer, ExprAttr align, ::llvm::ArrayRef<int64_t> from,
                          int64_t splitPairOtherId) {
  if (align) {
    printer << ", align = ";
    printer.printStrippedAttrOrType(align);
  }
  if (!from.empty()) {
    printer << ", from = [";
    ::llvm::interleaveComma(from, printer);
    printer << "]";
  }
  if (splitPairOtherId != -1) {
    printer << ", split_pair_other_id = " << splitPairOtherId;
  }
}

}  // namespace mlir::afir

//===----------------------------------------------------------------------===//
// AFIR dialect initialization
//===----------------------------------------------------------------------===//

// Include generated dialect implementation
#include "AFIR/IR/AFIRDialect.cpp.inc"

// Include generated enum definitions
#include "AFIR/IR/AFIREnums.cpp.inc"

// Include generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "AFIR/IR/AFIRAttrs.cpp.inc"

// Include generated type definitions
#define GET_TYPEDEF_CLASSES
#include "AFIR/IR/AFIRTypes.cpp.inc"

// Include generated operation definitions
#define GET_OP_CLASSES
#include "AFIR/IR/AFIROps.cpp.inc"

void mlir::afir::AFIRDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "AFIR/IR/AFIRTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "AFIR/IR/AFIRAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "AFIR/IR/AFIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//
//
// AFIR 的 payload 类型全部复用标准方言（约束 6）：张量是 `tensor`，片上与片外存储是
// `memref`。自有类型只用来表达标准方言表达不了的昇腾语义——目前只有 `!afir.token`
// （跨 pipe 的顺序边，ADR-19）。

mlir::Type mlir::afir::AFIRDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  mlir::Type type;
  mlir::OptionalParseResult parseResult = generatedTypeParser(parser, &mnemonic, type);
  if (parseResult.has_value() && mlir::succeeded(parseResult.value())) {
    return type;
  }
  return mlir::Type();
}

void mlir::afir::AFIRDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}

// 同一条 pipe 内程序序即执行序，不需要事件。写成同 pipe 的 token 说明作者对这条边的
// 意图与它实际能保证的事情不符——放过去它会变成一条无害的空边，而作者以为自己挡住了
// 某个依赖，失败方式是静默的数据竞争。
mlir::LogicalResult mlir::afir::TokenType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Pipe from,
                                                  Pipe to) {
  if (from == to) {
    return emitError() << kInvalidDescriptorDiagnostic << ": token pipes must differ, got '" << stringifyPipe(from)
                       << "' twice; ordering within one pipe is program order";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

mlir::Attribute mlir::afir::AFIRDialect::parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const {
  llvm::StringRef attrType;
  mlir::Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrType, type, attr);
  if (parseResult.has_value() && mlir::succeeded(parseResult.value())) {
    return attr;
  }
  return mlir::Attribute();
}

void mlir::afir::AFIRDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
  (void)generatedAttributePrinter(attr, printer);
}

namespace {

// 属性挂在哪种容器上：名字对但挂错 Op，消费端根本不会去那里找。
mlir::LogicalResult verifyAttributeContainer(mlir::Operation *op, mlir::NamedAttribute attribute) {
  mlir::StringRef name = attribute.getName().getValue();
  if (mlir::isa<mlir::func::FuncOp>(op) && name.starts_with(mlir::afir::kAFIRAttrPrefix) &&
      !isKnownFunctionAttribute(name)) {
    op->emitError() << mlir::afir::kUnsupportedOperationDiagnostic << ": unknown AFIR function attribute '"
                    << attribute.getName() << "'";
    return mlir::failure();
  }
  if (name == mlir::afir::kReduceScratchScopeAttr) {
    auto axis = mlir::dyn_cast<mlir::IntegerAttr>(attribute.getValue());
    if (!axis || !axis.getType().isInteger(64) || axis.getInt() < -1 || op->getNumResults() != 1) {
      return op->emitError() << mlir::afir::kInvalidLifetimeDiagnostic
                             << ": scratch_scope requires one result and an axis ID >= -1";
    }
  }
  if (name == mlir::afir::kReduceScratchRolesAttr &&
      !mlir::isa<mlir::afir::SumOp, mlir::afir::MaxOp, mlir::afir::MinOp, mlir::afir::MeanOp, mlir::afir::ProdOp,
                 mlir::afir::AnyOp, mlir::afir::AllOp>(op)) {
    return op->emitError() << mlir::afir::kInvalidDescriptorDiagnostic
                           << ": afir.scratch_roles is only valid on reduce operations";
  }
  if (name == mlir::afir::kWorkspacePlanAttrName && !mlir::isa<mlir::func::FuncOp>(op)) {
    op->emitError() << mlir::afir::kInvalidDescriptorDiagnostic << ": '" << mlir::afir::kWorkspacePlanAttrName
                    << "' is only valid on func.func";
    return mlir::failure();
  }
  // 名字对、类型不对等于退回迁移前的无类型编码。两者都在这里判，而不是留到 artifact 层
  // 生成 host tiling 时才报「缺资源上限」。
  if (name == mlir::afir::kTargetAttrName) {
    if (!mlir::isa<mlir::ModuleOp>(op)) {
      op->emitError() << mlir::afir::kInvalidDescriptorDiagnostic << ": '" << mlir::afir::kTargetAttrName
                      << "' is only valid on builtin.module";
      return mlir::failure();
    }
    if (!mlir::isa<mlir::afir::TargetAttr>(attribute.getValue())) {
      op->emitError() << mlir::afir::kDescriptorTypeMismatchDiagnostic << ": '" << mlir::afir::kTargetAttrName
                      << "' must be a #" << mlir::afir::AFIRDialect::getDialectNamespace() << "."
                      << mlir::afir::TargetAttr::getMnemonic();
      return mlir::failure();
    }
  }
  return mlir::success();
}

// 该描述符数组应有多少个元素。返回 nullopt 表示这个名字在这个 Op 上没有确定的期望数，
// 不做数量校验。
std::optional<unsigned> getExpectedDescriptorCount(mlir::Operation *op, llvm::StringRef name) {
  if (name == mlir::afir::kOperandDescsAttrName || name == mlir::afir::kDestinationDescsAttrName) {
    mlir::FailureOr<mlir::afir::OperandPartition> partition = mlir::afir::getOperandPartition(op);
    if (mlir::failed(partition)) {
      return std::nullopt;
    }
    return name == mlir::afir::kOperandDescsAttrName ? partition->inputCount : partition->destinationCount;
  }
  if (name == mlir::afir::kResultDescsAttrName) {
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
      return function.getNumResults();
    }
    return op->getNumResults();
  }
  if (name == mlir::afir::kArgDescsAttrName) {
    auto function = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (!function) {
      return std::nullopt;
    }
    mlir::FailureOr<unsigned> inputCount =
        mlir::afir::getCanonicalDataArgumentCount(function, function.getNumArguments());
    if (mlir::failed(inputCount)) {
      return std::nullopt;
    }
    return *inputCount;
  }
  return std::nullopt;
}

mlir::LogicalResult verifyWorkspacePlanEntries(mlir::Operation *op, llvm::StringRef name, mlir::ArrayAttr array) {
  for (auto [index, element] : llvm::enumerate(array)) {
    auto buffer = mlir::cast<mlir::afir::BufferDescAttr>(element);
    // workspace 是 host 分配、按参数传进来的 GM 暂存，不是片上资源：ATT 侧建 workspace
    // 张量时就把 position 钉成 GM、并清掉 `buf.id` / `que.id`（`SetGlobalMemInfo`）。
    // 片上的临时量不在这里：它是一块带 `afir.buffer_desc` 的 `memref.alloc`。
    if (buffer.getSpace() && buffer.getSpace().getValue() != mlir::afir::Space::GM) {
      op->emitError() << mlir::afir::kInvalidDescriptorDiagnostic << ": '" << name << "' element " << index
                      << " requires gm storage, got " << mlir::afir::stringifySpace(buffer.getSpace().getValue());
      return mlir::failure();
    }
    if (buffer.getLifeTimeAxisId() == std::numeric_limits<int64_t>::min()) {
      op->emitError() << mlir::afir::kInvalidLifetimeDiagnostic << ": '" << name << "' element " << index
                      << " requires life_time_axis_id";
      return mlir::failure();
    }
    if (mlir::failed(verifyLifeTimeAxis(op, "'" + name + "' element " + llvm::Twine(index), buffer))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

// `attribute` 不叫 `name` 时直接放行——调用方按名字逐个试过来。
mlir::LogicalResult verifyDescriptorArray(mlir::Operation *op, mlir::NamedAttribute attribute, llvm::StringRef name,
                                          bool expectTensor) {
  if (attribute.getName() != name) {
    return mlir::success();
  }
  if (name == mlir::afir::kArgDescsAttrName && !mlir::isa<mlir::func::FuncOp>(op)) {
    op->emitError() << mlir::afir::kInvalidDescriptorDiagnostic << ": '" << name << "' is only valid on func.func";
    return mlir::failure();
  }
  auto array = mlir::dyn_cast<mlir::ArrayAttr>(attribute.getValue());
  if (!array) {
    op->emitError() << mlir::afir::kDescriptorTypeMismatchDiagnostic << ": '" << name << "' must be an ArrayAttr";
    return mlir::failure();
  }
  for (auto [index, element] : llvm::enumerate(array)) {
    bool matches = expectTensor ? static_cast<bool>(mlir::dyn_cast<mlir::afir::TensorDescAttr>(element))
                                : static_cast<bool>(mlir::dyn_cast<mlir::afir::BufferDescAttr>(element));
    if (!matches) {
      op->emitError() << mlir::afir::kDescriptorTypeMismatchDiagnostic << ": '" << name << "' element " << index
                      << " has an unexpected attribute type";
      return mlir::failure();
    }
  }
  std::optional<unsigned> expectedCount = getExpectedDescriptorCount(op, name);
  if (expectedCount && array.size() != *expectedCount) {
    op->emitError() << mlir::afir::kDescriptorCountMismatchDiagnostic << ": '" << name << "' expected "
                    << *expectedCount << " element but found " << array.size();
    return mlir::failure();
  }
  if (name == mlir::afir::kWorkspacePlanAttrName) {
    return verifyWorkspacePlanEntries(op, name, array);
  }
  return mlir::success();
}

mlir::LogicalResult verifyAllDescriptorArrays(mlir::Operation *op, mlir::NamedAttribute attribute) {
  static constexpr llvm::StringLiteral tensorDescriptorNames[] = {
      mlir::afir::kOperandDescsAttrName, mlir::afir::kResultDescsAttrName, mlir::afir::kArgDescsAttrName,
      mlir::afir::kDestinationDescsAttrName};
  for (llvm::StringRef name : tensorDescriptorNames) {
    if (mlir::failed(verifyDescriptorArray(op, attribute, name, true))) {
      return mlir::failure();
    }
  }
  return verifyDescriptorArray(op, attribute, mlir::afir::kWorkspacePlanAttrName, false);
}

mlir::LogicalResult verifyFunctionRole(mlir::Operation *op, mlir::Attribute value) {
  auto role = mlir::dyn_cast<mlir::afir::FunctionRoleAttr>(value);
  if (!role) {
    op->emitError("'") << mlir::afir::kFunctionRoleAttrName << "' must use #"
                       << mlir::afir::AFIRDialect::getDialectNamespace() << "."
                       << mlir::afir::FunctionRoleAttr::getMnemonic();
    return mlir::failure();
  }
  if (op->getName().getStringRef() != "func.func") {
    op->emitError("'") << mlir::afir::kFunctionRoleAttrName << "' is only valid on "
                       << mlir::func::FuncOp::getOperationName();
    return mlir::failure();
  }

  auto visibility = op->getAttrOfType<mlir::StringAttr>("sym_visibility");
  bool isPrivate = visibility && visibility.getValue() == "private";
  if (role.getValue() == mlir::afir::FunctionRole::KernelBody && !isPrivate) {
    op->emitError("AFIR kernel body functions must be private");
    return mlir::failure();
  }
  if (role.getValue() == mlir::afir::FunctionRole::KernelEntry && isPrivate) {
    op->emitError("AFIR kernel entry functions must not be private");
    return mlir::failure();
  }
  return mlir::success();
}

}  // namespace

mlir::SmallVector<mlir::afir::AxisBindingAttr> mlir::afir::getAxisBindings(Operation *op) {
  SmallVector<AxisBindingAttr> bindings;
  Attribute attr = op->getAttr(kAxisBindingAttrName);
  if (auto binding = dyn_cast_or_null<AxisBindingAttr>(attr)) {
    bindings.push_back(binding);
  } else if (auto array = dyn_cast_or_null<ArrayAttr>(attr)) {
    for (Attribute entry : array) {
      bindings.push_back(cast<AxisBindingAttr>(entry));
    }
  }
  return bindings;
}

bool mlir::afir::hasAxisBinding(Operation *op, int64_t axis, AxisValueRole role) {
  return llvm::any_of(getAxisBindings(op),
                      [&](AxisBindingAttr binding) { return binding.getAxis() == axis && binding.getRole() == role; });
}

void mlir::afir::addAxisBinding(Operation *op, AxisBindingAttr binding) {
  SmallVector<AxisBindingAttr> bindings = getAxisBindings(op);
  if (llvm::is_contained(bindings, binding)) {
    return;
  }
  bindings.push_back(binding);
  llvm::sort(bindings, [](AxisBindingAttr lhs, AxisBindingAttr rhs) {
    if (lhs.getAxis() != rhs.getAxis()) {
      return lhs.getAxis() < rhs.getAxis();
    }
    return lhs.getRole() < rhs.getRole();
  });
  if (bindings.size() == 1) {
    op->setAttr(kAxisBindingAttrName, binding);
  } else {
    SmallVector<Attribute> entries(bindings.begin(), bindings.end());
    op->setAttr(kAxisBindingAttrName, ArrayAttr::get(op->getContext(), entries));
  }
}

mlir::LogicalResult mlir::afir::AFIRDialect::verifyOperationAttribute(mlir::Operation *op,
                                                                      mlir::NamedAttribute attribute) {
  if (mlir::failed(verifyAttributeContainer(op, attribute)) || mlir::failed(verifyAllDescriptorArrays(op, attribute))) {
    return mlir::failure();
  }
  if (attribute.getName() == kAxisBindingAttrName && !isa<AxisBindingAttr>(attribute.getValue())) {
    auto bindings = dyn_cast<ArrayAttr>(attribute.getValue());
    if (!bindings || bindings.empty() ||
        !llvm::all_of(bindings, [](Attribute entry) { return isa<AxisBindingAttr>(entry); })) {
      return op->emitError("afir.axis_binding must be an axis binding or a non-empty array of axis bindings");
    }
  }
  // 值的规划结果：一个值一份，挂在定义点上，不再有数组与下标对齐（D3）。
  if (attribute.getName() == kBufferDescAttrName) {
    auto descriptor = mlir::dyn_cast<mlir::afir::BufferDescAttr>(attribute.getValue());
    if (!descriptor) {
      op->emitError() << kDescriptorTypeMismatchDiagnostic << ": '" << kBufferDescAttrName << "' must be a #"
                      << AFIRDialect::getDialectNamespace() << "." << BufferDescAttr::getMnemonic();
      return mlir::failure();
    }
    if (mlir::failed(verifyLifeTimeAxis(op, "'" + llvm::Twine(kBufferDescAttrName) + "'", descriptor))) {
      return mlir::failure();
    }
  }
  if (attribute.getName() != kFunctionRoleAttrName) {
    return mlir::success();
  }
  return verifyFunctionRole(op, attribute.getValue());
}

mlir::LogicalResult mlir::afir::AFIRDialect::verifyRegionArgAttribute(mlir::Operation *op, unsigned /*regionIndex*/,
                                                                      unsigned argIndex,
                                                                      mlir::NamedAttribute attribute) {
  return verifyFunctionArgOrResultAttribute(op, argIndex, "argument", attribute);
}

mlir::LogicalResult mlir::afir::AFIRDialect::verifyRegionResultAttribute(mlir::Operation *op, unsigned /*regionIndex*/,
                                                                         unsigned resultIndex,
                                                                         mlir::NamedAttribute attribute) {
  return verifyFunctionArgOrResultAttribute(op, resultIndex, "result", attribute);
}

mlir::LogicalResult mlir::afir::ExprAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                 mlir::StringAttr value) {
  auto expression = mlir::afir::Expression::Parse(value.getValue());
  if (!expression) {
    return emitError() << "invalid AFIR symbolic expression: " << expression.error().message;
  }
  return mlir::success();
}

// 归约域必须是**非空**的、**一一对应**的、**不重复**的轴集合。三条都是为了同一件事：
// `N = Π sizes` 这个乘积有意义。空集的积是 1（除以 1 等于不除），少一个 size 的积缺一个
// 因子，重复的 id 把同一根轴数两遍——三者都不会在任何下游报错，只会让 Mean 静默算错。
mlir::LogicalResult mlir::afir::ReductionDomainAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                            llvm::ArrayRef<int64_t> axes,
                                                            llvm::ArrayRef<mlir::afir::ExprAttr> sizes) {
  if (axes.empty()) {
    return emitError() << kInvalidDescriptorDiagnostic << ": reduction_domain must name at least one axis";
  }
  if (axes.size() != sizes.size()) {
    return emitError() << kInvalidDescriptorDiagnostic << ": reduction_domain has " << axes.size() << " axes but "
                       << sizes.size() << " sizes";
  }
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t axis : axes) {
    if (axis < 0) {
      return emitError() << kInvalidDescriptorDiagnostic << ": reduction_domain axis must be non-negative";
    }
    if (!seen.insert(axis).second) {
      return emitError() << kInvalidDescriptorDiagnostic << ": reduction_domain repeats axis " << axis;
    }
  }
  return mlir::success();
}

mlir::LogicalResult mlir::afir::TensorDescAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                       llvm::ArrayRef<int64_t> axis,
                                                       llvm::ArrayRef<mlir::afir::ExprAttr> repeats,
                                                       llvm::ArrayRef<mlir::afir::ExprAttr> strides,
                                                       llvm::ArrayRef<int64_t> vectorizedAxis,
                                                       llvm::ArrayRef<mlir::afir::ExprAttr> vectorizedStrides,
                                                       mlir::afir::RoleAttr role) {
  (void)role;
  auto checkNonNegative = [&](llvm::ArrayRef<int64_t> values, llvm::StringRef field) -> mlir::LogicalResult {
    for (int64_t value : values) {
      if (value < 0) {
        return emitError() << kInvalidDescriptorDiagnostic << ": " << field << " entries must be non-negative";
      }
    }
    return mlir::success();
  };

  if (mlir::failed(checkNonNegative(axis, "axis")) ||
      mlir::failed(checkNonNegative(vectorizedAxis, "vectorized_axis"))) {
    return mlir::failure();
  }
  if (!repeats.empty() && repeats.size() != strides.size()) {
    return emitError() << kDescriptorCountMismatchDiagnostic << ": repeats and strides must have the same length";
  }
  if (!vectorizedAxis.empty() && vectorizedAxis.size() != vectorizedStrides.size()) {
    return emitError() << kDescriptorCountMismatchDiagnostic
                       << ": vectorized_axis and vectorized_strides must have the same length";
  }
  if (!vectorizedStrides.empty() && vectorizedAxis.empty()) {
    return emitError() << kDescriptorCountMismatchDiagnostic << ": vectorized_strides requires vectorized_axis";
  }
  if (!axis.empty() && !repeats.empty() && axis.size() != repeats.size()) {
    return emitError() << kDescriptorCountMismatchDiagnostic << ": axis and repeats must have the same length";
  }
  return mlir::success();
}

// 轴引用必须能在 `#afir.asc_graph` 里查得到，因此 id 不能是负数；`AxisAttr` 的 id
// 默认值 -1 表示「未赋号」，那种轴不可能被引用。
mlir::LogicalResult mlir::afir::AxisBindingAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                        int64_t axis, mlir::afir::AxisValueRole role) {
  (void)role;
  if (axis < 0) {
    return emitError() << kInvalidDescriptorDiagnostic << ": axis_binding axis must be non-negative";
  }
  return mlir::success();
}

// input/output/scalar_input 必须带参数序号；input 与 scalar_input 共用编号空间。
// workspace_offset 必须带 buffer id；workspace 基址不带序号。
// tiling 可带 stage id，供 Plan 的多 stage 入口使用；未编号参数仍须满足角色唯一性。
mlir::LogicalResult mlir::afir::KernelArgAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                      mlir::afir::KernelArgRole role, int64_t index) {
  bool isDataArgument = role == KernelArgRole::Input || role == KernelArgRole::Output ||
                        role == KernelArgRole::ScalarInput || role == KernelArgRole::WorkspaceOffset;
  if (isDataArgument && index < 0) {
    return emitError() << kInvalidDescriptorDiagnostic << ": kernel_arg role '" << stringifyKernelArgRole(role)
                       << "' requires a non-negative index";
  }
  if (role == KernelArgRole::Workspace && index != -1) {
    return emitError() << kInvalidDescriptorDiagnostic << ": kernel_arg role '" << stringifyKernelArgRole(role)
                       << "' cannot carry an index";
  }
  if (role == KernelArgRole::Tiling && index < -1) {
    return emitError() << kInvalidDescriptorDiagnostic << ": kernel_arg role '" << stringifyKernelArgRole(role)
                       << "' index must be a stage id or absent";
  }
  return mlir::success();
}

// 资源上限的值域此前写在消费端（`artifact/TilingSource/TilingProgramSpec.cpp` 的
// `ReadPlanModuleMetadata`），两个字段各判一段；现在在属性构造时一次判掉，与
// `#afplan.candidate` 把「非负」收进 verifier 是同一笔。
//
// 上界是 uint32：这两个数最终落进生成的 host tiling C++ 里 `ResLimit` 的 `uint32_t`
// 字段。UB 的下界是 257 而不是 1，因为发射端写的是 `set_ub_size(limit->ub_size - 256U)`
// ——256 字节留给 tiling 自身，容量不超过它就没有可分配的空间。
//
// `soc` 非空同样在这里判：它今天只写不读（dump 的目标标识），正因为没有读点，空串
// 不会在任何下游被发现——只写字段的唯一门禁就是产出时的这一道。
mlir::LogicalResult mlir::afir::TargetAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                   mlir::StringAttr soc, int64_t aivNum, int64_t ubSize) {
  constexpr int64_t kUint32Max = std::numeric_limits<uint32_t>::max();
  constexpr int64_t kUbSizeReservedBytes = 256;
  if (!soc || soc.getValue().empty()) {
    return emitError() << kInvalidDescriptorDiagnostic << ": target soc must not be empty";
  }
  if (aivNum <= 0 || aivNum > kUint32Max) {
    return emitError() << kInvalidDescriptorDiagnostic << ": target aiv_num expected a value in [1, " << kUint32Max
                       << "], got " << aivNum;
  }
  if (ubSize <= kUbSizeReservedBytes || ubSize > kUint32Max) {
    return emitError() << kInvalidDescriptorDiagnostic << ": target ub_size expected a value in ["
                       << (kUbSizeReservedBytes + 1) << ", " << kUint32Max << "], got " << ubSize;
  }
  return mlir::success();
}

// `loop_axis` 与 `axis` 的一致性此前由消费端各查一遍（`LoopGenerate.cpp:225` 是唯一
// 检查点，其余读点直接信任），现在在属性构造时就钉死：最内层循环轴必须确实在该算子
// 被调度的轴集合里，否则这条调度信息自相矛盾。
mlir::LogicalResult mlir::afir::ScheduleAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                     llvm::ArrayRef<int64_t> axis, int64_t loopAxis,
                                                     mlir::afir::ExecCondition execCondition) {
  for (int64_t value : axis) {
    if (value < 0) {
      return emitError() << kInvalidDescriptorDiagnostic << ": sched axis entries must be non-negative";
    }
  }
  // 缓存周期是「在 `loop_axis` 上隔多久重算一次」，没有 `loop_axis` 就没有它谈论的对象：
  // 谓词要用那根循环的归纳变量建。这个组合让 `AFIRExecCondition` 只能报「找不到循环」，
  // 而那时已经离产生它的地方很远了。
  if (execCondition != ExecCondition::NoCache && loopAxis < 0) {
    return emitError() << kInvalidDescriptorDiagnostic << ": sched exec_condition "
                       << stringifyExecCondition(execCondition) << " requires a loop_axis";
  }
  if (loopAxis == -1) {
    return mlir::success();
  }
  if (loopAxis < -1) {
    return emitError() << kInvalidDescriptorDiagnostic << ": sched loop_axis must be -1 or non-negative";
  }
  if (!llvm::is_contained(axis, loopAxis)) {
    return emitError() << kInvalidDescriptorDiagnostic << ": sched loop_axis " << loopAxis << " is not present in axis";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// LayoutAttr: memref layout (ADR-16 正交分解)
//===----------------------------------------------------------------------===//

namespace {
// 布局只作用于最后两维；`row_major` + 不分块就是 identity，任何 rank 都成立。
bool isIdentityLayout(mlir::afir::LayoutOrder outer, mlir::afir::TileOrder tiling) {
  return outer == mlir::afir::LayoutOrder::RowMajor && tiling == mlir::afir::TileOrder::None;
}
}  // namespace

mlir::LogicalResult mlir::afir::LayoutAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                   LayoutOrder outer, TileOrder tiling, llvm::ArrayRef<int64_t> tile,
                                                   int64_t rank) {
  if (rank < 0) {
    return emitError() << "afir.layout rank must be non-negative";
  }
  // 只有分块布局才有基块；两者必须同时在或同时不在，否则「不分块却带基块尺寸」这种
  // 自相矛盾的属性也能构造出来。
  if (tiling == TileOrder::None && !tile.empty()) {
    return emitError() << "afir.layout tile is only valid when tiling is not 'none'";
  }
  if (tiling != TileOrder::None) {
    if (tile.size() != 2) {
      return emitError() << "afir.layout tile must give exactly two extents, one per tiled dimension";
    }
    for (int64_t extent : tile) {
      if (extent <= 0) {
        return emitError() << "afir.layout tile extents must be positive";
      }
    }
  }
  // identity 之外的任何布局都在谈「最后两维」，rank 不足 2 时那句话没有意义。
  if (!isIdentityLayout(outer, tiling) && rank < 2) {
    return emitError() << "afir.layout requires rank >= 2 unless it is the identity layout "
                          "(outer = row_major, tiling = none)";
  }
  return mlir::success();
}

// `MemRefLayoutAttrInterface` 契约：不接收 rank、不允许返回 NULL，且定义域维数必须等于
// memref 的 rank。因此 rank 由属性自己携带（见 ODS 说明）。
//
// 产出的是**多结果**映射（逻辑下标 -> 物理下标空间），不是线性 offset：NZ 的线性 offset
// 需要矩阵一整条边的长度，而本接口同样拿不到 shape。多结果形式只依赖基块尺寸，与 shape 无关。
mlir::AffineMap mlir::afir::LayoutAttr::getAffineMap() const {
  mlir::MLIRContext *context = getContext();
  auto rank = static_cast<unsigned>(getRank());
  if (isIdentityLayout(getOuter(), getTiling())) {
    return mlir::AffineMap::getMultiDimIdentityMap(rank, context);
  }

  llvm::SmallVector<mlir::AffineExpr> results;
  results.reserve(rank + 2);
  // 前导维原样透传：双缓冲的 stage 维不参与矩阵解释。
  for (unsigned dim = 0; dim + 2 < rank; ++dim) {
    results.push_back(mlir::getAffineDimExpr(dim, context));
  }
  mlir::AffineExpr row = mlir::getAffineDimExpr(rank - 2, context);
  mlir::AffineExpr col = mlir::getAffineDimExpr(rank - 1, context);

  if (getTiling() == TileOrder::None) {
    // 不分块：外层布局直接决定最后两维的顺序。走到这里必然是 col_major——row_major +
    // 不分块已在上面按 identity 返回——所以这里只有转置一种结果。
    results.push_back(col);
    results.push_back(row);
    return mlir::AffineMap::get(rank, 0, results, context);
  }

  const int64_t rowExtent = getTile()[0];
  const int64_t colExtent = getTile()[1];
  // 外层决定基块网格的遍历序，分块层决定基块内部的遍历序——正交的两维。
  if (getOuter() == LayoutOrder::RowMajor) {
    results.push_back(row.floorDiv(rowExtent));
    results.push_back(col.floorDiv(colExtent));
  } else {
    results.push_back(col.floorDiv(colExtent));
    results.push_back(row.floorDiv(rowExtent));
  }
  if (getTiling() == TileOrder::RowMajor) {
    results.push_back(row % rowExtent);
    results.push_back(col % colExtent);
  } else {
    results.push_back(col % colExtent);
    results.push_back(row % rowExtent);
  }
  return mlir::AffineMap::get(rank, 0, results, context);
}

mlir::LogicalResult mlir::afir::LayoutAttr::verifyLayout(
    llvm::ArrayRef<int64_t> shape, llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  if (static_cast<int64_t>(shape.size()) != getRank()) {
    return emitError() << "afir.layout rank " << getRank() << " does not match memref rank " << shape.size();
  }
  if (getTiling() == TileOrder::None) {
    return mlir::success();
  }
  // 基块必须整除对应维度：除不尽意味着最后一块越出逻辑张量，物理布局无法定义。
  // 这正是 NZ 输入需要先补齐到基块整数倍的原因。动态维在编译期无从判断，放行。
  for (unsigned offset = 0; offset < 2; ++offset) {
    int64_t dim = shape[shape.size() - 2 + offset];
    int64_t extent = getTile()[offset];
    if (!mlir::ShapedType::isDynamic(dim) && dim % extent != 0) {
      return emitError() << "afir.layout tile extent " << extent << " does not divide dimension "
                         << (shape.size() - 2 + offset) << " of size " << dim;
    }
  }
  return mlir::success();
}

mlir::LogicalResult mlir::afir::BufferDescAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                       mlir::afir::SpaceAttr space, int64_t bufferId, int64_t reuseId,
                                                       int64_t slotCount, int64_t queueDepth,
                                                       mlir::afir::ExprAttr sizeBytesExpr, bool sizeFromResource,
                                                       int64_t alignmentBytes, int64_t lifeTimeAxisId) {
  auto checkOptionalNonNegative = [&](int64_t value, llvm::StringRef field) -> mlir::LogicalResult {
    if (value < -1) {
      return emitError() << kInvalidBindingDiagnostic << ": " << field << " must be -1 or non-negative";
    }
    return mlir::success();
  };
  if (mlir::failed(checkOptionalNonNegative(bufferId, "buffer_id")) ||
      mlir::failed(checkOptionalNonNegative(reuseId, "reuse_id")) ||
      mlir::failed(checkOptionalNonNegative(slotCount, "slot_count")) ||
      mlir::failed(checkOptionalNonNegative(queueDepth, "queue_depth")) ||
      mlir::failed(checkOptionalNonNegative(alignmentBytes, "alignment_bytes")) ||
      (lifeTimeAxisId != std::numeric_limits<int64_t>::min() &&
       mlir::failed(checkOptionalNonNegative(lifeTimeAxisId, "life_time_axis_id")))) {
    return mlir::failure();
  }
  if (slotCount == 0) {
    return emitError() << kInvalidBindingDiagnostic << ": slot_count must be -1 or at least 1";
  }
  if (queueDepth == 0) {
    return emitError() << kInvalidBindingDiagnostic << ": queue_depth must be -1 or at least 1";
  }
  if (alignmentBytes == 0) {
    return emitError() << kInvalidBindingDiagnostic << ": alignment_bytes must be -1 or at least 1";
  }
  if (lifeTimeAxisId != std::numeric_limits<int64_t>::min() && lifeTimeAxisId < -1) {
    return emitError() << kInvalidLifetimeDiagnostic << ": life_time_axis_id must be -1 or a loop axis id";
  }
  // 字节大小的两个来源互斥。两个都写等于又一次把同一个事实编码两遍（D2），而且没有
  // 哪一处会去核对它们是否一致。
  if (sizeFromResource && sizeBytesExpr) {
    return emitError() << kInvalidBindingDiagnostic
                       << ": size_from_resource and size_bytes_expr are mutually exclusive";
  }
  // 引用的目标是 descriptor 自己的 `buffer_id`。没有 id 就没有可引用的资源，也就拼不出
  // `q<id>_size` / `b<id>_size` 这个字段名。
  if (sizeFromResource && bufferId < 0) {
    return emitError() << kInvalidBindingDiagnostic << ": size_from_resource requires buffer_id";
  }
  // 只有片上资源有 canonical 尺寸字段。GM 与 workspace 的大小是算出来的，不在 TilingData
  // 里按 id 存一份。
  if (sizeFromResource && space && space.getValue() != Space::UB) {
    return emitError() << kInvalidBindingDiagnostic << ": size_from_resource is only valid for UB resources, got "
                       << stringifySpace(space.getValue());
  }
  // 片上资源的尺寸字段名不再写进表达式。允许它意味着允许 `buffer_id = 2` 配
  // `size_bytes_expr = <"q7_size">` 这种自指错资源的 descriptor——没有任何一处会发现，
  // 症状要到物化时才以「TilingData is missing canonical size expression field」出现，
  // 而且那句话说的是另一件事。
  if (sizeBytesExpr && mlir::common::IsCanonicalResourceSizeName(sizeBytesExpr.getValue().getValue())) {
    return emitError() << kInvalidBindingDiagnostic << ": '" << sizeBytesExpr.getValue().getValue()
                       << "' names a canonical resource size field; use size_from_resource instead";
  }
  return mlir::success();
}
