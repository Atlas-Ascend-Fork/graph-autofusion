/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRKernelBodySignature.cpp - AFIR kernel body signature ------------===//

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"

#include <set>
#include <string>

#include "dialect/AFIR/Analysis/ResourceIds.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {

bool hasCanonicalBufferDescriptors(func::FuncOp func) {
  return afir::hasBufferDescriptors(func.getOperation());
}

FailureOr<Type> getCanonicalTilingDataType(func::FuncOp func) {
  auto baseType = getTilingSchema(getAscendCTilingDataType(func));
  if (!baseType) {
    return func.emitError("failed to create AscendC TilingData type");
  }
  std::set<int64_t> queueIds;
  std::set<int64_t> bufferIds;
  if (failed(afir::collectCanonicalResourceIds(func, queueIds, bufferIds))) {
    return failure();
  }
  MLIRContext *context = func.getContext();
  SmallVector<Attribute> fieldTypes;
  SmallVector<Attribute> fieldNames;
  llvm::StringSet<> seen;
  for (Attribute type : baseType.getTypesAttr()) {
    fieldTypes.push_back(type);
  }
  for (Attribute name : baseType.getNamesAttr()) {
    auto stringName = cast<StringAttr>(name);
    fieldNames.push_back(stringName);
    seen.insert(stringName.getValue());
  }
  // `afir.api_tiling_fields`（host tiling 的嵌套 struct 布局，bridge 按节点发）在相位 A
  // 就并进 canonical 实参：conversion（相位 C）按字段名在实参上建 `emitasc.member` 访问，
  // 字段没并进去的话 mem 档报「TilingData is missing the ... field」。成员一律 ui32、
  // 嵌套类型名取条目的 "type" 键，与 GenTilingData 的 stage schema 同一口径；字段序也
  // 对齐那份的注释（size_var → API tiling → q/b 尺寸）。
  if (auto apiFields = func->getAttrOfType<ArrayAttr>(kApiTilingFieldsAttrName)) {
    Type ui32 = IntegerType::get(context, 32, IntegerType::Unsigned);
    for (Attribute entry : apiFields) {
      auto item = dyn_cast<DictionaryAttr>(entry);
      auto typeName = item ? item.getAs<StringAttr>(kApiTilingTypeKey) : StringAttr();
      auto fieldName = item ? item.getAs<StringAttr>(kApiTilingFieldKey) : StringAttr();
      auto memberNames = item ? item.getAs<ArrayAttr>(kApiTilingFieldsKey) : ArrayAttr();
      if (!typeName || !fieldName || !memberNames) {
        return func.emitError("kernel function has a malformed ") << kApiTilingFieldsAttrName << " entry";
      }
      if (!seen.insert(fieldName.getValue()).second) {
        return func.emitError("API tiling field conflicts with an existing TilingData field: ") << fieldName.getValue();
      }
      SmallVector<Attribute> memberTypes(memberNames.size(), TypeAttr::get(ui32));
      auto apiType =
          mlir::ascend::kernel::TilingSchema::get(context, typeName, ArrayAttr::get(context, memberTypes), memberNames);
      if (!apiType) {
        return func.emitError("invalid API tiling field layout for ") << fieldName.getValue();
      }
      fieldTypes.push_back(TypeAttr::get(apiType.getType()));
      fieldNames.push_back(fieldName);
    }
  }
  // 字段名由 `makeCanonicalResourceSizeName` 派生，与 memory allocate 把
  // `size_from_resource` 解析回 `afir.tiling_param` 用的是同一处拼法——此前这里是手写的
  // 字符串拼接，与 bridge 那一份只靠约定对齐。
  auto appendField = [&](bool queue, int64_t id) -> LogicalResult {
    std::string name = afir::makeCanonicalResourceSizeName(queue, id);
    if (!seen.insert(name).second) {
      return func.emitError("canonical resource size field conflicts with a shape field: ") << name;
    }
    fieldTypes.push_back(TypeAttr::get(IntegerType::get(context, 64)));
    fieldNames.push_back(StringAttr::get(context, name));
    return success();
  };
  for (int64_t id : queueIds) {
    if (failed(appendField(/*queue=*/true, id))) {
      return failure();
    }
  }
  for (int64_t id : bufferIds) {
    if (failed(appendField(/*queue=*/false, id))) {
      return failure();
    }
  }
  auto schema = mlir::ascend::kernel::TilingSchema::get(
      context, baseType.getNameAttr(), ArrayAttr::get(context, fieldTypes), ArrayAttr::get(context, fieldNames));
  if (!schema) {
    return func.emitError("invalid canonical TilingData field layout");
  }
  return schema.getType();
}

struct AFIRKernelBodySignaturePass : public PassWrapper<AFIRKernelBodySignaturePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRKernelBodySignaturePass)

  StringRef getArgument() const final {
    return "afir-kernel-body-signature";
  }

  StringRef getDescription() const final {
    return "Materialize the AscendC ABI for tensor or canonical buffer AFIR functions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, func::FuncDialect, memref::MemRefDialect,
                    emitasc::EmitAscDialect>();
  }

  void runOnOperation() override {
    // Signature changes affect argument uses and function types together. Work
    // on a clone so a rejected function or final verifier failure cannot leave
    // any function in the input module with a partially materialized ABI.
    OwningOpRef<ModuleOp> prepared = cast<ModuleOp>(getOperation()->clone());
    if (failed(prepareModule(*prepared)) || failed(verify(*prepared))) {
      signalPassFailure();
      return;
    }
    getOperation().getBodyRegion().takeBody(prepared->getBodyRegion());
  }

 private:
  static LogicalResult prepareModule(ModuleOp module) {
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      afir::FunctionRoleAttr role = afir::getFunctionRoleAttr(func);
      if (!role) {
        continue;
      }
      FailureOr<bool> hasKernelABI = checkAscendCKernelABI(func, role.getValue());
      if (failed(hasKernelABI)) {
        return failure();
      }
      if (*hasKernelABI) {
        continue;
      }
      if (!containsAfir(func)) {
        return func.emitError("function with an AFIR role must contain AFIR operations");
      }
      // Kernel calls are built after this pass. Rewriting an already-used
      // symbol would also require rewriting every caller's arguments/results.
      if (!SymbolTable::symbolKnownUseEmpty(func, module)) {
        return func.emitError("kernel signature must be materialized before creating symbol uses");
      }
      if (failed(rewriteFunctionSignature(func, role.getValue()))) {
        return failure();
      }
    }
    return success();
  }

  static bool containsAfir(func::FuncOp func) {
    bool found = false;
    func.walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "afir") {
        found = true;
      }
    });
    return found;
  }

  static MemRefType convertTensorType(RankedTensorType tensor) {
    return getAscendCGlobalMemRefType(tensor);
  }

  static LogicalResult collectBoundaryTypes(func::FuncOp func, SmallVectorImpl<RankedTensorType> &inputTensorTypes,
                                            SmallVectorImpl<MemRefType> &inputMemrefTypes,
                                            SmallVectorImpl<RankedTensorType> &resultTensorTypes,
                                            SmallVectorImpl<MemRefType> &resultMemrefTypes) {
    for (auto [index, type] : llvm::enumerate(func.getArgumentTypes())) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor) {
        return func.emitError("expected ranked tensor input at index ") << index;
      }
      inputTensorTypes.push_back(tensor);
      inputMemrefTypes.push_back(convertTensorType(tensor));
    }
    for (auto [index, type] : llvm::enumerate(func.getResultTypes())) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor) {
        return func.emitError("expected ranked tensor result at index ") << index;
      }
      resultTensorTypes.push_back(tensor);
      resultMemrefTypes.push_back(convertTensorType(tensor));
    }
    return success();
  }

  static LogicalResult replaceInputBoundaries(Block &entry, ArrayRef<RankedTensorType> tensorTypes,
                                              ArrayRef<MemRefType> memrefTypes) {
    SmallVector<SmallVector<OpOperand *>> originalUses;
    originalUses.reserve(tensorTypes.size());
    for (unsigned index = 0; index < tensorTypes.size(); ++index) {
      BlockArgument argument = entry.getArgument(index);
      originalUses.emplace_back();
      for (OpOperand &use : argument.getUses()) {
        originalUses.back().push_back(&use);
      }
      argument.setType(memrefTypes[index]);
    }

    OpBuilder builder(&entry, entry.begin());
    for (unsigned index = 0; index < tensorTypes.size(); ++index) {
      BlockArgument argument = entry.getArgument(index);
      Value tensor = builder.create<bufferization::ToTensorOp>(argument.getLoc(), tensorTypes[index], argument);
      for (OpOperand *use : originalUses[index]) {
        use->set(tensor);
      }
    }
    return success();
  }

  // 收集本函数实际绑定的 GM workspace，并按 buffer id 升序排列。
  // 判据为 GM buffer_desc 的 buffer_id 出现在模块的 afir.workspace_tensor_ids 中。
  // 该列表是 fused 级信息，非空不代表每个 plan 都使用 workspace，例如 RCommon plan。
  // 只为实际绑定的 workspace 增加形参，避免改变无关 kernel 的签名（RA 设计 3.8）。
  static SmallVector<int64_t> collectBoundWorkspaceIds(func::FuncOp func) {
    auto module = func->getParentOfType<ModuleOp>();
    auto declared =
        module ? module->getAttrOfType<DenseI64ArrayAttr>(afir::kWorkspaceTensorIdsAttrName) : DenseI64ArrayAttr();
    SmallVector<int64_t> bound;
    if (!declared) {
      return bound;
    }
    func.walk([&](Operation *op) {
      auto descriptor = op->getAttrOfType<afir::BufferDescAttr>(afir::kBufferDescAttrName);
      if (!descriptor || descriptor.getSpace().getValue() != afir::Space::GM) {
        return;
      }
      const int64_t bufferId = descriptor.getBufferId();
      if (bufferId < 0 || !llvm::is_contained(declared.asArrayRef(), bufferId) || llvm::is_contained(bound, bufferId)) {
        return;
      }
      bound.push_back(bufferId);
    });
    llvm::sort(bound);
    return bound;
  }

  static LogicalResult appendKernelArguments(func::FuncOp func, ArrayRef<MemRefType> resultMemrefTypes,
                                             ArrayRef<DictionaryAttr> resultAttrs,
                                             SmallVectorImpl<BlockArgument> &outputArguments, Type tilingDataType) {
    MLIRContext *context = func.getContext();
    DictionaryAttr emptyAttrs = DictionaryAttr::get(context);
    for (auto [index, type] : llvm::enumerate(resultMemrefTypes)) {
      unsigned argumentIndex = func.getNumArguments();
      if (failed(func.insertArgument(argumentIndex, type, resultAttrs[index], func.getLoc()))) {
        return func.emitError("failed to append output argument at index ") << index;
      }
      outputArguments.push_back(func.getArgument(argumentIndex));
    }
    // 在参数定义点声明追加的 workspace 和 tiling 身份，消费方不按类型或倒数位置推断。
    // AscendC C++ ABI 仍要求固定顺序，但该顺序由校验保证（设计文档 3.6）。
    unsigned workspaceIndex = func.getNumArguments();
    if (failed(func.insertArgument(workspaceIndex, getAscendCWorkspaceType(context), emptyAttrs, func.getLoc()))) {
      return func.emitError("failed to append workspace argument");
    }
    afir::setKernelArgAttr(func, workspaceIndex, afir::KernelArgRole::Workspace);
    // 为实际绑定的每块 workspace 增加字节偏移参数，位于 workspace 基址与 tiling 之间，
    // 与 legacy 的 (..., GM_ADDR workspace, uint64_t workspace<ID>..., const T *t) 一致。
    // entry 从 tiling 读取偏移后传入，语义见 AFIREnums.td 的 WorkspaceOffset。
    for (int64_t bufferId : collectBoundWorkspaceIds(func)) {
      unsigned offsetIndex = func.getNumArguments();
      if (failed(func.insertArgument(offsetIndex, getAscendCWorkspaceOffsetType(context), emptyAttrs, func.getLoc()))) {
        return func.emitError("failed to append workspace offset argument for buffer ") << bufferId;
      }
      afir::setKernelArgAttr(func, offsetIndex, afir::KernelArgRole::WorkspaceOffset, bufferId);
    }
    unsigned tilingIndex = func.getNumArguments();
    if (failed(func.insertArgument(tilingIndex, tilingDataType, emptyAttrs, func.getLoc()))) {
      return func.emitError("failed to append tiling data argument");
    }
    afir::setKernelArgAttr(func, tilingIndex, afir::KernelArgRole::Tiling);
    return success();
  }

  static LogicalResult replaceOutputBoundary(func::FuncOp func, func::ReturnOp returnOp,
                                             ArrayRef<BlockArgument> outputArguments) {
    if (returnOp.getNumOperands() != outputArguments.size()) {
      return returnOp.emitError("return operand count does not match function result count");
    }

    OpBuilder builder(returnOp);
    for (auto [source, destination] : llvm::zip(returnOp.getOperands(), outputArguments)) {
      auto materialize =
          builder.create<bufferization::MaterializeInDestinationOp>(returnOp.getLoc(), source, destination);
      materialize.setWritable(true);
    }
    builder.create<func::ReturnOp>(returnOp.getLoc());
    returnOp.erase();

    BitVector erasedResults(func.getNumResults(), true);
    if (failed(func.eraseResults(erasedResults))) {
      return func.emitError("failed to erase function results");
    }
    return success();
  }

  static LogicalResult rewriteFunctionSignature(func::FuncOp func, afir::FunctionRole role) {
    if (!llvm::hasSingleElement(func.getBody())) {
      return func.emitError("function signature preparation expects a single-block function");
    }

    Block &entry = func.front();
    if (entry.empty()) {
      return func.emitError("function signature preparation expects a nonempty block");
    }
    auto returnOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
    if (!returnOp) {
      return func.emitError("expected func.return terminator");
    }

    // Validate the complete original boundary before changing argument types.
    if (returnOp.getNumOperands() != func.getNumResults()) {
      return returnOp.emitError("return operand count does not match function result count");
    }
    FailureOr<Type> tilingDataType = getCanonicalTilingDataType(func);
    if (failed(tilingDataType)) {
      return failure();
    }
    if (hasCanonicalBufferDescriptors(func)) {
      if (failed(rewriteCanonicalSignature(func, role, *tilingDataType))) {
        return failure();
      }
    } else if (failed(rewriteTensorSignature(func, returnOp, role, *tilingDataType))) {
      return failure();
    }
    return markNeedSyncAll(func, role);
  }

  // 按 legacy codegen_kernel.cpp 的规则设置跨 stage 全核同步：
  // need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1
  // workspace 使用情况来自 collectBoundWorkspaceIds，j 来自 #afplan.variant，
  // stage 总数来自 afplan.stages。元数据缺失时报错，不能跳过同步，否则下一阶段可能读取
  // 其他核尚未写完的 workspace；单核测试无法覆盖此问题。
  static LogicalResult markNeedSyncAll(func::FuncOp func, afir::FunctionRole role) {
    if (role != afir::FunctionRole::KernelBody || collectBoundWorkspaceIds(func).empty()) {
      return success();
    }
    afplan::VariantAttr variant = afplan::getVariant(func);
    auto module = func->getParentOfType<ModuleOp>();
    afplan::StagesAttr stages = module ? afplan::getStages(module) : afplan::StagesAttr();
    if (!variant || !stages) {
      return func.emitError("kernel body binds a workspace but declares no ")
             << (variant ? afplan::kStagesAttrName : afplan::kVariantAttrName)
             << ", so whether the next stage needs a cross-core sync cannot be decided";
    }
    const int64_t lastStage = static_cast<int64_t>(stages.getTemplateCounts().size()) - 1;
    if (variant.getStage() >= lastStage) {
      return success();
    }
    func->setAttr(mlir::ascend::kernel::kAscendCNeedSyncAllAttrName, UnitAttr::get(func.getContext()));
    return success();
  }

  static LogicalResult rewriteCanonicalSignature(func::FuncOp func, afir::FunctionRole role, Type tilingDataType) {
    if (!func.getResultTypes().empty()) {
      return func.emitError("canonical buffer-form kernel body must have no function results");
    }
    if (failed(verifyCanonicalKernelArguments(func, func.getNumArguments()))) {
      return failure();
    }
    const unsigned dataArgumentCount = func.getNumArguments();
    SmallVector<Type> argumentTypes;
    argumentTypes.reserve(dataArgumentCount);
    for (Type type : func.getArgumentTypes()) {
      // ScalarData 按值传递，不含需要改写的地址空间，保持原类型。
      // verifyCanonicalKernelArguments 已校验 scalar_input 与非 shaped 类型对应，因此可按类型分流。
      // 使用 dyn_cast 处理非 memref 参数，避免 cast 触发断言。
      auto memref = dyn_cast<MemRefType>(type);
      if (!memref) {
        argumentTypes.push_back(type);
        continue;
      }
      argumentTypes.push_back(MemRefType::get(memref.getShape(), memref.getElementType(), memref.getLayout(),
                                              getAscendCGlobalMemorySpaceAttr(func.getContext())));
    }
    for (auto [argument, type] : llvm::zip(func.getArguments(), argumentTypes)) {
      argument.setType(type);
    }
    func.setType(FunctionType::get(func.getContext(), argumentTypes, {}));
    SmallVector<BlockArgument> unusedOutputs;
    if (failed(appendKernelArguments(func, {}, {}, unusedOutputs, tilingDataType))) {
      return failure();
    }
    setAscendCKernelAttributes(func, role, dataArgumentCount);
    return success();
  }

  static LogicalResult rewriteTensorSignature(func::FuncOp func, func::ReturnOp returnOp, afir::FunctionRole role,
                                              Type tilingDataType) {
    Block &entry = func.front();
    SmallVector<RankedTensorType> inputTensorTypes;
    SmallVector<MemRefType> inputMemrefTypes;
    SmallVector<RankedTensorType> resultTensorTypes;
    SmallVector<MemRefType> resultMemrefTypes;
    if (failed(collectBoundaryTypes(func, inputTensorTypes, inputMemrefTypes, resultTensorTypes, resultMemrefTypes))) {
      return failure();
    }

    SmallVector<DictionaryAttr> resultAttrs;
    func.getAllResultAttrs(resultAttrs);

    // 输出改为函数参数时，将 afir.result_descs 同步移入 afir.arg_descs。
    // eraseResults 后继续保留 result_descs 会违反 getExpectedDescriptorCount 校验；
    // arg_descs 也必须统一覆盖全部数据参数，不能因原来是 tensor 或 buffer 形态而改变含义。
    // 这样 getCanonicalDataArgumentCount 才能按同一规则校验描述符与参数表。
    if (!resultMemrefTypes.empty()) {
      auto argDescs = func->getAttrOfType<ArrayAttr>(afir::kArgDescsAttrName);
      auto resultDescs = func->getAttrOfType<ArrayAttr>(afir::kResultDescsAttrName);
      if (argDescs && resultDescs) {
        SmallVector<Attribute> merged(argDescs.begin(), argDescs.end());
        merged.append(resultDescs.begin(), resultDescs.end());
        func->setAttr(afir::kArgDescsAttrName, ArrayAttr::get(func.getContext(), merged));
      }
      func->removeAttr(afir::kResultDescsAttrName);
    }

    SmallVector<Type> functionInputs(inputMemrefTypes.begin(), inputMemrefTypes.end());
    SmallVector<Type> functionResults(resultTensorTypes.begin(), resultTensorTypes.end());
    func.setType(FunctionType::get(func.getContext(), functionInputs, functionResults));
    if (failed(replaceInputBoundaries(entry, inputTensorTypes, inputMemrefTypes))) {
      return failure();
    }

    SmallVector<BlockArgument> outputArguments;
    if (failed(appendKernelArguments(func, resultMemrefTypes, resultAttrs, outputArguments, tilingDataType)) ||
        failed(replaceOutputBoundary(func, returnOp, outputArguments))) {
      return failure();
    }

    setAscendCKernelAttributes(func, role, inputTensorTypes.size());
    return success();
  }
};

}  // namespace

namespace mlir::ascend::kernel {

std::unique_ptr<Pass> createAFIRKernelBodySignaturePass() {
  return std::make_unique<AFIRKernelBodySignaturePass>();
}

void registerAFIRKernelBodySignaturePass() {
  PassRegistration<AFIRKernelBodySignaturePass>();
}

}  // namespace mlir::ascend::kernel
