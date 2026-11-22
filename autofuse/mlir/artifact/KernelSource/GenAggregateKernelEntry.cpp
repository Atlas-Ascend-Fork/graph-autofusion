/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/GenAggregateKernelEntry.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "dialect/AFIR/IR/PlanMetadata.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSet.h"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {
Value createMemberOp(OpBuilder &builder, Location loc, Value value, StringRef name) {
  return createTilingSchemaMember(builder, loc, value, name);
}

struct PlanDeclaration {
  func::FuncOp function;
  // 本 Plan 各 stage 在聚合 TilingData 里的成员名，按 stage 次序。`_main` 一 stage 收一个
  // tiling 形参，两张表一一对应。
  SmallVector<StringAttr> tilingFields;
  size_t tilingArgument;
  SmallVector<unsigned> argumentMapping;
  // 本 Plan 用到的 workspace 张量号。聚合入口要把它们的偏移从顶层拷进每个 stage 的结构体，
  // 理由见 `copyWorkspaceOffsets`。
  SmallVector<int64_t> workspaceIds;
};

std::string makeWorkspaceOffsetFieldName(int64_t tensorId) {
  return "workspace" + std::to_string(tensorId);
}

struct AggregateContract {
  mlir::ascend::kernel::TilingSchema schema;
  ArrayAttr planEntries;
};

struct DeclarationCollection {
  SmallVector<PlanDeclaration> declarations;
  FunctionType mainType;
  SmallVector<KernelArgumentIdentity> arguments;
  SmallVector<Type> argumentTypes;
};

// _main 每个 stage 对应一个 tiling 参数，按 stage 顺序连续排列在参数表末尾。
// 参数的 #afir.kernel_arg 声明 tiling 角色和 stage index，不按类型或位置推断身份。
// 返回首个 tiling 参数下标，stage i 对应返回值加 i。
FailureOr<size_t> findTilingArguments(func::FuncOp function, size_t stageCount) {
  SmallVector<size_t> tilingArguments;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    afir::KernelArgAttr argument = mlir::afir::getKernelArgAttr(function, index);
    if (argument && argument.getRole() == mlir::afir::KernelArgRole::Tiling) {
      tilingArguments.push_back(index);
    }
  }
  if (tilingArguments.size() != stageCount) {
    return function.emitError("GenAggregateKernelEntry: plan main declares ")
           << tilingArguments.size() << " tiling arguments but the Plan has " << stageCount << " stages";
  }
  for (auto [stage, index] : llvm::enumerate(tilingArguments)) {
    if (index != tilingArguments.front() + stage ||
        mlir::afir::getKernelArgAttr(function, index).getIndex() != static_cast<int64_t>(stage)) {
      return function.emitError("GenAggregateKernelEntry: plan main tiling argument ")
             << index << " must declare stage " << stage;
    }
  }
  return tilingArguments.front();
}

// GenTilingData.cpp 的 collectStageFields 为各 stage 添加 workspace<ID> 字段，Plan 主函数
// 从该字段读取偏移。但 ATT 仅写顶层 fusion 作用域中的 workspace 字段
// （见 tiling_field_schema.cpp）。必须将顶层偏移复制到 stage，否则 stage 保持初始值 0，
// 多块 workspace 将使用同一基址并相互覆盖。
void copyWorkspaceOffsets(OpBuilder &builder, Location loc, Value aggregateTiling, Value stageTiling,
                          ArrayRef<int64_t> workspaceIds) {
  for (int64_t tensorId : workspaceIds) {
    const std::string field = makeWorkspaceOffsetFieldName(tensorId);
    builder.create<emitasc::SetMemberOp>(loc, stageTiling, field, createMemberOp(builder, loc, aggregateTiling, field));
  }
}

SmallVector<Value> buildCallArguments(OpBuilder &builder, Location loc, Block *entry,
                                      const PlanDeclaration &declaration, Value aggregateTiling) {
  func::FuncOp planMain = declaration.function;
  FunctionType planType = planMain.getFunctionType();
  SmallVector<Value> arguments;
  size_t stage = 0;
  for (auto [position, index] : llvm::enumerate(declaration.argumentMapping)) {
    if (index == std::numeric_limits<unsigned>::max()) {
      Value stageTiling = createMemberOp(builder, loc, aggregateTiling, declaration.tilingFields[stage].getValue());
      copyWorkspaceOffsets(builder, loc, aggregateTiling, stageTiling, declaration.workspaceIds);
      arguments.push_back(stageTiling);
      ++stage;
      continue;
    }
    Value argument = entry->getArgument(index);
    // 保留 Plan 声明供函数体克隆；独立提供的 ranked 声明在调用点适配，不产生数据搬运。
    Type expected = planType.getInput(position);
    if (argument.getType() != expected) {
      argument = builder.create<memref::CastOp>(loc, expected, argument);
    }
    arguments.push_back(argument);
  }
  return arguments;
}

FailureOr<AggregateContract> collectAggregateContract(ArrayRef<ModuleOp> planModules, ModuleOp entryModule) {
  if (!mlir::afir::isEntryModule(entryModule)) {
    return failure();
  }

  auto schemaAttr = entryModule->getAttrOfType<TypeAttr>(mlir::ascend::kernel::kAscendCTilingSchemaAttrName);
  auto schema = schemaAttr ? getTilingSchema(schemaAttr.getValue()) : mlir::ascend::kernel::TilingSchema();
  auto planEntries = entryModule->getAttrOfType<ArrayAttr>(mlir::ascend::kernel::kAscendCPlanEntriesAttrName);
  if (!schema || !planEntries || planEntries.size() != planModules.size()) {
    return failure();
  }
  return AggregateContract{schema, planEntries};
}

// 候选身份是一个 uniqued 的属性，匹配即一次相等比较；不必把它拆成两个整数再逐字段比。
ModuleOp findPlanModule(ArrayRef<ModuleOp> planModules, afplan::CandidateAttr wanted) {
  if (!wanted) {
    return {};
  }
  for (ModuleOp module : planModules) {
    if (afplan::getCandidate(module) == wanted) {
      return module;
    }
  }
  return {};
}

// 按身份合并 Plan 参数，GM 参数始终使用 unranked ABI，使签名不依赖候选集。
// 调用点适配 ranked Plan 声明；元素类型或地址空间冲突仍报错。
FailureOr<unsigned> appendAggregateArgument(DeclarationCollection &collection, func::FuncOp planMain, unsigned index,
                                            const KernelArgumentIdentity &argument) {
  Type planType = planMain.getFunctionType().getInput(index);
  // 仅统一 GM 输入、输出；workspace 字节基址和标量保持原类型。
  const bool isGmTensor = argument.kind == KernelArgumentKind::Input || argument.kind == KernelArgumentKind::Output;
  Type aggregateType = isGmTensor ? getAscendCAbiArgumentType(planType) : planType;
  auto existing = llvm::find(collection.arguments, argument);
  if (existing == collection.arguments.end()) {
    unsigned aggregateIndex = collection.arguments.size();
    collection.arguments.push_back(argument);
    collection.argumentTypes.push_back(aggregateType);
    return aggregateIndex;
  }
  unsigned aggregateIndex = static_cast<unsigned>(existing - collection.arguments.begin());
  if (collection.argumentTypes[aggregateIndex] != aggregateType) {
    return planMain.emitError("GenAggregateKernelEntry: argument ")
           << getKernelArgumentKey(argument)
           << " has incompatible ABI types across plans: " << collection.argumentTypes[aggregateIndex] << " and "
           << aggregateType;
  }
  return aggregateIndex;
}

// 聚合入口要读的那一侧：各 Plan 模块，以及入口模块上的聚合契约。
struct AggregateInputs {
  ArrayRef<ModuleOp> planModules;
  const AggregateContract &contract;
};

FailureOr<SmallVector<int64_t>> collectPlanWorkspaceIds(ModuleOp planModule, func::FuncOp planMain,
                                                        AggregateInputs inputs, ArrayRef<StringAttr> tilingFields) {
  // copyWorkspaceOffsets 依赖顶层和各 stage 的 workspace 字段，两侧都必须校验。
  // 字段缺失时报错，不能跳过复制并让 stage 使用零偏移。
  SmallVector<int64_t> workspaceIds;
  if (auto ids = planModule->getAttrOfType<DenseI64ArrayAttr>(mlir::afir::kWorkspaceTensorIdsAttrName)) {
    for (int64_t tensorId : ids.asArrayRef()) {
      if (tensorId < 0) {
        continue;
      }
      const std::string field = makeWorkspaceOffsetFieldName(tensorId);
      if (!inputs.contract.schema.getField(field)) {
        return planMain.emitError("GenAggregateKernelEntry: aggregate tiling data has no field ") << field;
      }
      for (StringAttr stageField : tilingFields) {
        if (!getTilingSchemaField(inputs.contract.schema.getField(stageField.getValue()), field)) {
          return planMain.emitError("GenAggregateKernelEntry: stage tiling data ")
                 << stageField.getValue() << " has no field " << field;
        }
      }
      workspaceIds.push_back(tensorId);
    }
  }

  return workspaceIds;
}

FailureOr<SmallVector<unsigned>> collectPlanArgumentMapping(func::FuncOp planMain, unsigned tilingArgument,
                                                            size_t tilingFieldCount,
                                                            DeclarationCollection &collection) {
  llvm::StringSet<> identities;
  SmallVector<unsigned> argumentMapping;
  for (unsigned index = 0; index < planMain.getNumArguments(); ++index) {
    if (index >= tilingArgument && index < tilingArgument + tilingFieldCount) {
      argumentMapping.push_back(std::numeric_limits<unsigned>::max());
      continue;
    }
    auto key = getKernelArgumentIdentity(planMain, index, "GenAggregateKernelEntry");
    if (failed(key)) {
      return failure();
    }
    auto keyString = getKernelArgumentKey(*key);
    if (!identities.insert(keyString).second) {
      return planMain.emitError("GenAggregateKernelEntry: duplicate argument identity ") << keyString;
    }
    auto aggregateIndex = appendAggregateArgument(collection, planMain, index, *key);
    if (failed(aggregateIndex)) {
      return failure();
    }
    argumentMapping.push_back(*aggregateIndex);
  }

  return argumentMapping;
}

FailureOr<PlanDeclaration> createPlanDeclaration(OpBuilder &builder, Location loc, AggregateInputs inputs,
                                                 DeclarationCollection &collection, DictionaryAttr entry) {
  afplan::CandidateAttr candidate = afplan::getCandidate(entry);
  auto tilingFieldsAttr = entry ? entry.getAs<ArrayAttr>(kAscendCPlanEntryTilingFieldKey) : ArrayAttr();
  ModuleOp planModule = findPlanModule(inputs.planModules, candidate);
  auto symbol = planModule ? planModule.getSymName() : std::optional<StringRef>();
  func::FuncOp planMain =
      planModule && symbol ? planModule.lookupSymbol<func::FuncOp>(symbol->str() + "_main") : nullptr;
  if (!entry || !candidate || !tilingFieldsAttr || tilingFieldsAttr.empty() || !symbol || symbol->empty() ||
      !planMain) {
    return failure();
  }
  // 检查每个 stage 的成员名都存在于聚合 schema 中，避免将空类型传入生成阶段。
  SmallVector<StringAttr> tilingFields;
  for (Attribute attr : tilingFieldsAttr) {
    auto field = dyn_cast<StringAttr>(attr);
    if (!field || !inputs.contract.schema.getField(field.getValue())) {
      return planMain.emitError("GenAggregateKernelEntry: aggregate tiling data has no stage field");
    }
    tilingFields.push_back(field);
  }

  auto workspaceIds = collectPlanWorkspaceIds(planModule, planMain, inputs, tilingFields);
  if (failed(workspaceIds)) {
    return failure();
  }

  auto tilingArgument = findTilingArguments(planMain, tilingFields.size());
  if (failed(tilingArgument)) {
    return failure();
  }
  FunctionType functionType = planMain.getFunctionType();
  auto argumentMapping = collectPlanArgumentMapping(planMain, *tilingArgument, tilingFields.size(), collection);
  if (failed(argumentMapping)) {
    return failure();
  }

  auto declaration = builder.create<func::FuncOp>(loc, symbol->str() + "_main", functionType);
  declaration.setPrivate();
  return PlanDeclaration{declaration, std::move(tilingFields), *tilingArgument, std::move(*argumentMapping),
                         std::move(*workspaceIds)};
}

// Reorders aggregate-main arguments and updates each plan declaration's call
// mapping so it still refers to the same argument after reordering.
void reorderAggregateArguments(DeclarationCollection &collection) {
  auto permutation = getKernelArgumentPermutation(collection.arguments);
  SmallVector<unsigned> inverse(permutation.size());
  SmallVector<KernelArgumentIdentity> arguments;
  SmallVector<Type> argumentTypes;
  arguments.reserve(permutation.size());
  argumentTypes.reserve(permutation.size());
  for (auto [newIndex, oldIndex] : llvm::enumerate(permutation)) {
    inverse[oldIndex] = newIndex;
    arguments.push_back(collection.arguments[oldIndex]);
    argumentTypes.push_back(collection.argumentTypes[oldIndex]);
  }
  collection.arguments = std::move(arguments);
  collection.argumentTypes = std::move(argumentTypes);

  // Declarations were mapped while arguments had their discovery order; update them to the canonical ABI order.
  for (auto &declaration : collection.declarations) {
    for (unsigned &index : declaration.argumentMapping) {
      if (index != std::numeric_limits<unsigned>::max()) {
        index = inverse[index];
      }
    }
  }
}

FailureOr<DeclarationCollection> collectPlanDeclarations(OpBuilder &builder, Location loc, AggregateInputs inputs) {
  DeclarationCollection collection;
  for (Attribute attr : inputs.contract.planEntries) {
    auto entry = dyn_cast<DictionaryAttr>(attr);
    auto declaration = createPlanDeclaration(builder, loc, inputs, collection, entry);
    if (failed(declaration)) {
      return failure();
    }
    collection.declarations.push_back(std::move(*declaration));
  }
  reorderAggregateArguments(collection);
  SmallVector<Type> mainInputs;
  for (Type argumentType : collection.argumentTypes) {
    mainInputs.push_back(argumentType);
  }
  mainInputs.push_back(inputs.contract.schema);
  collection.mainType = FunctionType::get(inputs.contract.schema.getContext(), mainInputs, {});
  return collection;
}

FailureOr<SmallVector<SmallVector<unsigned>>> groupDeclarationsBySubgraph(ArrayAttr planEntries) {
  SmallVector<SmallVector<unsigned>> declarationsBySubgraph;
  for (auto [declarationIndex, attr] : llvm::enumerate(planEntries)) {
    auto entry = dyn_cast<DictionaryAttr>(attr);
    afplan::CandidateAttr candidate = afplan::getCandidate(entry);
    if (!candidate) {
      return failure();
    }
    auto subgraph = static_cast<size_t>(candidate.getSubgraph());
    if (declarationsBySubgraph.size() <= subgraph) {
      declarationsBySubgraph.resize(subgraph + 1);
    }
    declarationsBySubgraph[subgraph].push_back(declarationIndex);
  }
  return declarationsBySubgraph;
}

func::FuncOp createAggregateMain(OpBuilder &builder, ModuleOp entryModule, FunctionType mainType,
                                 ArrayRef<KernelArgumentIdentity> arguments) {
  auto main = builder.create<func::FuncOp>(entryModule.getLoc(), "main", mainType);
  main->setAttr(mlir::ascend::kernel::kAscendCFusedMainAttrName, builder.getUnitAttr());
  // 与 `createPlanMain` 同理：每个参数都声明身份，包括 workspace 与末尾的聚合 tiling。
  for (auto [index, argument] : llvm::enumerate(arguments)) {
    switch (argument.kind) {
      case KernelArgumentKind::Input:
        mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Input, argument.index);
        break;
      // 标量必须保留 ScalarInput 角色；改为 Input 会声明成 GM 张量，
      // 不满足 verifyCanonicalKernelArguments 的角色与类型校验。
      case KernelArgumentKind::ScalarInput:
        mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::ScalarInput, argument.index);
        break;
      case KernelArgumentKind::Output:
        mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Output, argument.index);
        break;
      case KernelArgumentKind::Workspace:
        mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Workspace);
        break;
      case KernelArgumentKind::Tiling:
        mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Tiling);
        break;
    }
  }
  mlir::afir::setKernelArgAttr(main, main.getNumArguments() - 1, mlir::afir::KernelArgRole::Tiling);
  main.addEntryBlock();
  return main;
}

// 正在填充的聚合 main：函数本身、入口块，以及末尾那个聚合 tiling 参数。
struct AggregateMain {
  func::FuncOp function;
  Block *entry;
  Value tiling;

  Location getLoc() const {
    return function->getLoc();
  }
};

void buildSinglePlanCall(OpBuilder &builder, const AggregateMain &main, const PlanDeclaration &declaration) {
  builder.create<func::CallOp>(main.getLoc(), declaration.function,
                               buildCallArguments(builder, main.getLoc(), main.entry, declaration, main.tiling));
}

LogicalResult buildGraphDispatch(OpBuilder &builder, const AggregateMain &main, size_t subgraphId,
                                 ArrayRef<unsigned> declarationIndices, ArrayRef<PlanDeclaration> declarations) {
  Location loc = main.getLoc();
  Value selector = createMemberOp(builder, loc, main.tiling, "graph" + std::to_string(subgraphId) + "_tiling_key");
  if (!selector || !isa<IntegerType>(selector.getType())) {
    return failure();
  }
  selector = builder.create<emitc::CastOp>(loc, builder.getIndexType(), selector);

  SmallVector<int64_t> cases;
  for (size_t index = 0; index < declarationIndices.size(); ++index) {
    cases.push_back(static_cast<int64_t>(index));
  }
  auto dispatch = builder.create<scf::IndexSwitchOp>(loc, TypeRange{}, selector, cases, cases.size());
  dispatch.getDefaultRegion().emplaceBlock();
  // host tiling 写入的 graph<N>_tiling_key 必须对应已下降的候选编号。
  // 不匹配表示 host 与 kernel 的候选集不一致，例如 host 选中了未编入 kernel 的候选。
  // 默认分支触发 trap；仅 yield 会静默跳过计算，并让 host 读取未写入的输出。
  OpBuilder defaultBuilder = OpBuilder::atBlockEnd(&dispatch.getDefaultRegion().front());
  defaultBuilder.create<ascendc::TrapOp>(loc);
  defaultBuilder.create<scf::YieldOp>(loc);
  for (auto [caseIndex, declarationIndex] : llvm::enumerate(declarationIndices)) {
    Region &region = dispatch.getCaseRegions()[caseIndex];
    region.emplaceBlock();
    OpBuilder caseBuilder = OpBuilder::atBlockEnd(&region.front());
    buildSinglePlanCall(caseBuilder, main, declarations[declarationIndex]);
    caseBuilder.create<scf::YieldOp>(loc);
  }
  return success();
}
}  // namespace

LogicalResult mlir::ascend::kernel::validateAggregateContract(ArrayRef<ModuleOp> planModules, ModuleOp entryModule) {
  return collectAggregateContract(planModules, entryModule);
}

FailureOr<mlir::ascend::kernel::AggregateArgumentLayout> mlir::ascend::kernel::runGenAggregateKernelEntry(
    ArrayRef<ModuleOp> planModules, ModuleOp entryModule) {
  auto contract = collectAggregateContract(planModules, entryModule);
  if (failed(contract)) {
    return failure();
  }

  // 聚合调用点可能创建 `memref.cast`，克隆的 Plan 体也包含 template 调用处的视图转换。
  // 本函数也可被直接调用，不能只依赖 AFIRAggregatePass 的方言注册。
  entryModule.getContext()->getOrLoadDialect<memref::MemRefDialect>();
  // 缺省分支要建 `ascendc.trap`，同理需要方言在场。
  entryModule.getContext()->getOrLoadDialect<ascendc::AscendCDialect>();
  OpBuilder builder(entryModule.getBodyRegion());
  builder.setInsertionPointToEnd(entryModule.getBody());
  auto declarationCollection =
      collectPlanDeclarations(builder, entryModule.getLoc(), AggregateInputs{planModules, *contract});
  if (failed(declarationCollection)) {
    return failure();
  }
  auto declarationsBySubgraph = groupDeclarationsBySubgraph(contract->planEntries);
  if (failed(declarationsBySubgraph)) {
    return failure();
  }

  auto function =
      createAggregateMain(builder, entryModule, declarationCollection->mainType, declarationCollection->arguments);
  Block *entry = &function.front();
  builder.setInsertionPointToEnd(entry);
  const AggregateMain main{function, entry, entry->getArgument(entry->getNumArguments() - 1)};
  if (declarationCollection->declarations.size() == 1) {
    buildSinglePlanCall(builder, main, declarationCollection->declarations.front());
    builder.create<func::ReturnOp>(main.getLoc());
    return AggregateArgumentLayout{declarationCollection->arguments, static_cast<size_t>(entry->getNumArguments() - 1)};
  }
  for (auto [subgraphId, declarationIndices] : llvm::enumerate(*declarationsBySubgraph)) {
    if (failed(
            buildGraphDispatch(builder, main, subgraphId, declarationIndices, declarationCollection->declarations))) {
      return failure();
    }
  }
  builder.create<func::ReturnOp>(main.getLoc());
  return AggregateArgumentLayout{declarationCollection->arguments, static_cast<size_t>(entry->getNumArguments() - 1)};
}
