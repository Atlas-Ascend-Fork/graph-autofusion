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

  // 本函数体真正绑定了哪几块 workspace，按 buffer id 升序。
  //
  // 判据是「GM 的 `buffer_desc` 且 `buffer_id` 在 module 的 `afir.workspace_tensor_ids` 里」。
  // 不能只看那张表非空就给每个 plan 都追形参：它是 **fused 级**的，对一个根本用不到
  // workspace 的 plan（今天全部 RCommon 语料）照样非空，见 RA 设计文档 3.8 第 2 条。
  // 按「谁真的绑定了」驱动，既有 kernel 的签名才逐字节不变。
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
    // ABI 追加出来的两个参数在参数自己的定义点上声明身份，而不是留给消费端按「倒数
    // 第二个且类型等于 workspace 类型」「倒数第一个且结构体名字是 TilingData」去反推
    // （设计文档 3.6）。位置约定本身仍在——AscendC 生成的 C++ 签名要求这个顺序——但
    // 它现在是被检查的约定，不是被依赖的推断依据。
    unsigned workspaceIndex = func.getNumArguments();
    if (failed(func.insertArgument(workspaceIndex, getAscendCWorkspaceType(context), emptyAttrs, func.getLoc()))) {
      return func.emitError("failed to append workspace argument");
    }
    afir::setKernelArgAttr(func, workspaceIndex, afir::KernelArgRole::Workspace);
    // 每块被本函数体绑定的 workspace 再追一个字节偏移形参，夹在 workspace 与 tiling 之间
    // ——与 legacy 的 `(..., GM_ADDR workspace, uint64_t workspace<ID>..., const T *t)`
    // 同形（`codegen_kernel.cpp:1935`）。偏移由 entry 从 tiling 上读出来再传进来，理由见
    // `AFIREnums.td` 的 `WorkspaceOffset`。
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

  // 这个 stage 跑完之后要不要全核同步，判据与 legacy 同源（`codegen_kernel.cpp:2748`）：
  //
  //     need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1
  //
  // `has_workspace_node` 即本函数体绑定了 workspace，与偏移形参共用
  // `collectBoundWorkspaceIds`；`j` 取自 `#afplan.variant`，总数取自 `afplan.stages`。
  // 这两个属性缺失时报错而不是跳过：漏掉同步，phase 2 会读到别的核没写完的 workspace，
  // 结果每次不同，而且单核上必然看不出来。
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
      // 按值标量形参（`ScalarData`）没有地址空间可换，原样穿过。`verifyCanonicalKernelArguments`
      // 已经把「角色是 scalar_input ⟺ 类型非 shaped」判成一条双向不变量，所以这里按类型
      // 分流与按角色分流等价；用 `dyn_cast` 而不是 `cast` 还顺带把原先那个 assert 失败
      // 换成了正常控制流。
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

    // 输出从函数结果变成函数参数，描述它们的 `afir.result_descs` 也要跟着搬进
    // `afir.arg_descs`。不搬有两个后果：
    //   1. 函数结果已被 `eraseResults` 清空，而 `afir.result_descs` 仍按结果个数校验
    //      （AFIRDialect.cpp 的 `getExpectedDescriptorCount`），于是物化后的 tensor
    //      形态直接通不过方言 verifier——这是一个此前没有用例走到的既有缺陷；
    //   2. `afir.arg_descs` 的含义会随来源形态而变（buffer 形态覆盖全部数据参数、
    //      tensor 形态只覆盖输入），下游就无法用一条规则算出它该有多长，
    //      「描述符与参数表对不上」也就无从检查（见 getCanonicalDataArgumentCount）。
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
