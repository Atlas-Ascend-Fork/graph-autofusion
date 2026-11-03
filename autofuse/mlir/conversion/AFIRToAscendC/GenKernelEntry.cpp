/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "dialect/AFIR/IR/PlanMetadata.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <limits>
#include <cassert>

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {

struct MainSignature {
  std::string entryName;
  FunctionType entryType;
  // 本 Plan 的 stage schema 载体：字段按 stage 次序，字段名就是这些 stage 结构体在最外层
  // `AutofuseTilingData` 里的成员名。载体本身不是一个会被发射的结构体，见 GenTilingData。
  mlir::ascend::kernel::TilingSchema tilingType;
  // **第一个** tiling 形参的下标。ATT 的最外层布局里没有 plan 这一层，各 stage 结构体
  // 平铺在根上，所以 `_main` 也一 stage 收一个 tiling 形参，按 stage 次序紧挨在参数表尾部：
  // stage `i` 就是 `tilingArgument + i`。
  size_t tilingArgument;
  SmallVector<Type> argumentTypes;
  SmallVector<KernelArgumentIdentity> arguments;
};

// 一个 template 形参的实参从哪来。此前这张表是 `SmallVector<unsigned>`，用
// `UINT_MAX` 当哨兵表示「传 stage tiling」。workspace 偏移要来自 stage tiling 的某个**成员**
// 而不是它本身，再加一个哨兵就等于把两处约定写进同一个整数里，所以写成三态。
struct TemplateArgumentSource {
  enum class Kind { MainArgument, StageTiling, StageTilingWorkspaceOffset };
  Kind kind = Kind::MainArgument;
  // MainArgument：主参下标。StageTilingWorkspaceOffset：workspace 的 buffer id。
  unsigned index = 0;
};

struct PlanInfo {
  ModuleOp module;
  int64_t id;
  SmallVector<int64_t> templateCountsPerStage;
  mlir::ascend::kernel::TilingSchema tilingType;
  SmallVector<SmallVector<func::FuncOp>> templates;
  SmallVector<SmallVector<int64_t>> tilingKeys;
  SmallVector<SmallVector<SmallVector<TemplateArgumentSource>>> argumentMappings;
  std::string mainName;
  FunctionType mainType;
};

FailureOr<MainSignature> deriveMainSignature(ModuleOp module) {
  auto symbol = module.getSymName();
  if (!symbol || symbol->empty()) {
    return module.emitError("GenKernelEntry: missing Plan module symbol");
  }
  auto tilingAttr = module->getAttrOfType<TypeAttr>(mlir::ascend::kernel::kAscendCTilingSchemaAttrName);
  if (!tilingAttr) {
    return module.emitError("GenKernelEntry: missing Plan ") << mlir::ascend::kernel::kAscendCTilingSchemaAttrName;
  }
  auto tilingType = getTilingSchema(tilingAttr.getValue());
  if (!tilingType) {
    return module.emitError("GenKernelEntry: invalid Plan tiling schema");
  }
  MainSignature signature{symbol->str() + "_main", {}, tilingType, 0};
  return signature;
}

FailureOr<func::FuncOp> collectTemplate(ModuleOp module, size_t stage, size_t templateId) {
  // 属性是 uniqued 的，坐标匹配即一次属性相等比较，不必逐字段取值再比。
  auto wanted =
      afplan::VariantAttr::get(module.getContext(), static_cast<int64_t>(stage), static_cast<int64_t>(templateId));
  func::FuncOp match;
  for (func::FuncOp candidate : module.getOps<func::FuncOp>()) {
    if (afplan::getVariant(candidate) != wanted) {
      continue;
    }
    if (match) {
      return candidate.emitError("GenKernelEntry: duplicate template coordinates");
    }
    match = candidate;
  }
  if (!match || !match.isPrivate()) {
    return module.emitError("GenKernelEntry: missing template or template must be private");
  }
  return match;
}

FailureOr<int64_t> getTemplateTilingKey(func::FuncOp templateFunc) {
  auto graph = templateFunc->getAttrOfType<afir::AscGraphAttrGroupsAttr>(mlir::afir::kAscGraphAttrName);
  if (!graph || graph.getTilingKey() < 0) {
    return templateFunc.emitError("GenKernelEntry: template requires a non-negative ")
           << mlir::afir::kAscGraphAttrName << " tiling_key";
  }
  return graph.getTilingKey();
}

LogicalResult validateTemplateAttributes(func::FuncOp templateFunc, ArrayRef<int64_t> templateCountsPerStage) {
  // 非负性由 `#afplan.variant` 的 verifier 保证；这里只剩「坐标落在本 plan 的分派表内」
  // 这条跨属性约束。
  afplan::VariantAttr variant = afplan::getVariant(templateFunc);
  if (!variant || variant.getStage() >= static_cast<int64_t>(templateCountsPerStage.size()) ||
      variant.getTemplateId() >= templateCountsPerStage[variant.getStage()]) {
    return templateFunc.emitError("GenKernelEntry: invalid template coordinates");
  }
  if (Attribute syncAttr = templateFunc->getAttr(mlir::ascend::kernel::kAscendCNeedSyncAllAttrName);
      syncAttr && !isa<UnitAttr>(syncAttr)) {
    return templateFunc.emitError("GenKernelEntry: ")
           << mlir::ascend::kernel::kAscendCNeedSyncAllAttrName << " must be a unit attribute";
  }
  return success();
}

// 按身份合并 template 参数，GM 参数始终使用 unranked ABI，使签名不依赖候选视图。
// template 保留 ranked 签名以满足 afir.arg_descs 的秩约束，调用点恢复对应视图。
// 元素类型或地址空间冲突仍报错；mapping 记录合并后的参数下标。
LogicalResult appendMainArgument(MainSignature &signature, func::FuncOp templateFunc, unsigned index,
                                 const KernelArgumentIdentity &argument, SmallVector<TemplateArgumentSource> &mapping) {
  Type templateType = templateFunc.getFunctionType().getInput(index);
  const bool isGmTensor = argument.kind == KernelArgumentKind::Input || argument.kind == KernelArgumentKind::Output;
  Type mainType = isGmTensor ? getAscendCAbiArgumentType(templateType) : templateType;
  auto existing =
      llvm::find_if(signature.arguments, [&](const KernelArgumentIdentity &other) { return other == argument; });
  unsigned mainIndex = static_cast<unsigned>(existing - signature.arguments.begin());
  if (existing == signature.arguments.end()) {
    mainIndex = signature.arguments.size();
    signature.arguments.push_back(argument);
    signature.argumentTypes.push_back(mainType);
  } else if (signature.argumentTypes[mainIndex] != mainType) {
    return templateFunc.emitError("GenKernelEntry: incompatible ABI types for ")
           << getKernelArgumentKey(argument) << " across templates: " << signature.argumentTypes[mainIndex] << " and "
           << mainType;
  }
  mapping.push_back({TemplateArgumentSource::Kind::MainArgument, mainIndex});
  return success();
}

LogicalResult collectTemplateAbi(func::FuncOp templateFunc, Type stageType, MainSignature &signature,
                                 SmallVector<TemplateArgumentSource> &mapping) {
  FunctionType templateType = templateFunc.getFunctionType();
  SmallVector<size_t> tilingArguments;
  for (auto [index, type] : llvm::enumerate(templateType.getInputs())) {
    if (type == stageType) {
      tilingArguments.push_back(index);
    }
  }
  if (tilingArguments.size() != 1) {
    return templateFunc.emitError("GenKernelEntry: template must contain exactly one stage tiling argument");
  }
  size_t tilingArgument = tilingArguments.front();
  llvm::StringSet<> templateIdentities;
  mapping.reserve(tilingArgument + 1);
  for (unsigned index = 0; index < templateFunc.getNumArguments(); ++index) {
    if (index == tilingArgument) {
      mapping.push_back({TemplateArgumentSource::Kind::StageTiling, 0});
      continue;
    }
    auto key = getKernelArgumentIdentity(templateFunc, index);
    if (failed(key)) {
      return failure();
    }
    // workspace 偏移不进 main 的形参表：它是从 stage tiling 的 `workspace<ID>` 成员上读出来
    // 的一个值，main 自己并不收它。所以这一格记成「去 stage tiling 上取这个 buffer id」。
    if (key->kind == KernelArgumentKind::WorkspaceOffset) {
      mapping.push_back({TemplateArgumentSource::Kind::StageTilingWorkspaceOffset, static_cast<unsigned>(key->index)});
      continue;
    }
    auto keyString = getKernelArgumentKey(*key);
    if (!templateIdentities.insert(keyString).second) {
      return templateFunc.emitError("GenKernelEntry: duplicate argument identity ") << keyString;
    }
    if (failed(appendMainArgument(signature, templateFunc, index, *key, mapping))) {
      return failure();
    }
  }
  return success();
}

// Reorders the generated plan-main arguments and updates every template call
// mapping from the old main-argument indices to the new canonical indices.
void reorderMainArguments(MainSignature &signature, PlanInfo &planInfo) {
  auto permutation = getKernelArgumentPermutation(signature.arguments);
  SmallVector<unsigned> inverse(permutation.size());
  SmallVector<KernelArgumentIdentity> arguments;
  SmallVector<Type> argumentTypes;
  arguments.reserve(permutation.size());
  argumentTypes.reserve(permutation.size());
  for (auto [newIndex, oldIndex] : llvm::enumerate(permutation)) {
    inverse[oldIndex] = newIndex;
    arguments.push_back(signature.arguments[oldIndex]);
    argumentTypes.push_back(signature.argumentTypes[oldIndex]);
  }
  signature.arguments = std::move(arguments);
  signature.argumentTypes = std::move(argumentTypes);

  // Mappings were collected with the old argument order; convert each old index to its new position.
  // 只有「取主参」那一态是下标，另外两态的 `index` 不是主参下标（一个没有意义、一个是
  // buffer id），跟着置换会把它们改坏。
  for (auto &stageMappings : planInfo.argumentMappings) {
    for (auto &mapping : stageMappings) {
      for (TemplateArgumentSource &source : mapping) {
        if (source.kind == TemplateArgumentSource::Kind::MainArgument) {
          source.index = inverse[source.index];
        }
      }
    }
  }
}

// 定位 Plan 里的一个 stage：所在 module、stage 序号、该 stage 的 template 个数。
struct StageRef {
  ModuleOp module;
  size_t stage;
  int64_t templateCount;
};

FailureOr<SmallVector<func::FuncOp>> collectStageTemplates(
    StageRef stageRef, mlir::ascend::kernel::TilingSchema stageType, MainSignature &signature,
    SmallVector<SmallVector<TemplateArgumentSource>> &argumentMappings) {
  SmallVector<func::FuncOp> templates;
  for (int64_t templateId = 0; templateId < stageRef.templateCount; ++templateId) {
    auto function = collectTemplate(stageRef.module, stageRef.stage, templateId);
    SmallVector<TemplateArgumentSource> mapping;
    if (failed(function) || failed(collectTemplateAbi(*function, stageType, signature, mapping))) {
      return failure();
    }
    templates.push_back(*function);
    argumentMappings.push_back(std::move(mapping));
  }
  return templates;
}

FailureOr<SmallVector<int64_t>> collectStageTilingKeys(ArrayRef<func::FuncOp> templates) {
  SmallVector<int64_t> keys;
  for (func::FuncOp templateFunc : templates) {
    FailureOr<int64_t> tilingKey = getTemplateTilingKey(templateFunc);
    if (failed(tilingKey)) {
      return failure();
    }
    if (llvm::is_contained(keys, *tilingKey)) {
      // 兜底。正常路径上 `ScheduleToAfir` 已经带着成因拦下来了（图级 `SetTilingKey` + 一个 stage
      // 多模板）；能走到这里说明 AFIR 是别处造的，所以把成因一并说出来。
      return templateFunc.emitError("GenKernelEntry: duplicate tiling_key in stage ")
             << afplan::getVariant(templateFunc).getStage()
             << "; each template in a stage needs its own key because host tiling selects by it";
    }
    keys.push_back(*tilingKey);
  }
  return keys;
}

FailureOr<mlir::ascend::kernel::TilingSchema> getStageTilingType(mlir::ascend::kernel::TilingSchema tilingType,
                                                                 size_t stage) {
  // 载体的字段按 stage 次序铺（`buildStageSchemasAndAppendArgs` 就是这么攒的），所以这里
  // 按下标取。不按名字取是因为名字里带 (子图, plan) 坐标，在这儿重算一遍等于把同一条
  // 命名公式再抄一份。
  ArrayAttr stageTypes = tilingType.getTypesAttr();
  if (stage >= stageTypes.size()) {
    return failure();
  }
  auto stageType = getTilingSchema(cast<TypeAttr>(stageTypes[stage]).getValue());
  if (!stageType) {
    return failure();
  }
  if (!isa<IntegerType>(stageType.getField("tiling_key"))) {
    return failure();
  }
  return stageType;
}

LogicalResult validatePlanAttributes(ModuleOp module, afplan::CandidateAttr candidate, afplan::StagesAttr stages,
                                     std::optional<StringRef> symbol) {
  // 非负与非空都是 `#afplan.candidate` / `#afplan.stages` 的 verifier 已经保证的事；
  // 这里只判它们在不在。
  if (!candidate || !stages || !symbol) {
    return module.emitError("GenKernelEntry: invalid Plan attributes");
  }
  if (Attribute cubeTypeAttr = module->getAttr(mlir::afir::kCubeTypeAttrName)) {
    auto cubeType = dyn_cast<IntegerAttr>(cubeTypeAttr);
    if (!cubeType || !cubeType.getType().isSignlessInteger(64)) {
      return module.emitError("GenKernelEntry: ") << mlir::afir::kCubeTypeAttrName << " must be i64";
    }
    if (cubeType.getInt() != -1) {
      return module.emitError("GenKernelEntry: Cube/CV Plan is not supported, ")
             << mlir::afir::kCubeTypeAttrName << "=" << cubeType.getInt();
    }
  }
  return success();
}

FailureOr<bool> collectStageInfo(StageRef stageRef, mlir::ascend::kernel::TilingSchema tilingType,
                                 MainSignature &signature, PlanInfo &planInfo) {
  ModuleOp module = stageRef.module;
  size_t stage = stageRef.stage;
  if (stageRef.templateCount <= 0) {
    return module.emitError("GenKernelEntry: template count must be positive");
  }
  auto stageType = getStageTilingType(tilingType, stage);
  if (failed(stageType)) {
    return module.emitError("GenKernelEntry: invalid Plan stage tiling data type");
  }
  if (planInfo.argumentMappings.size() <= stage) {
    planInfo.argumentMappings.resize(stage + 1);
  }
  auto templates = collectStageTemplates(stageRef, *stageType, signature, planInfo.argumentMappings[stage]);
  if (failed(templates)) {
    return failure();
  }
  auto stageTilingKeys = collectStageTilingKeys(*templates);
  if (failed(stageTilingKeys)) {
    return failure();
  }
  planInfo.templates.push_back(std::move(*templates));
  planInfo.tilingKeys.push_back(std::move(*stageTilingKeys));
  return true;
}

FailureOr<PlanInfo> collectTemplates(ModuleOp module, MainSignature &signature) {
  afplan::CandidateAttr candidate = afplan::getCandidate(module);
  afplan::StagesAttr stages = afplan::getStages(module);
  auto symbol = module.getSymName();
  if (failed(validatePlanAttributes(module, candidate, stages, symbol))) {
    return failure();
  }
  auto tilingType = signature.tilingType;
  PlanInfo planInfo;
  planInfo.module = module;
  planInfo.id = candidate.getPlan();
  planInfo.templateCountsPerStage.assign(stages.getTemplateCounts().begin(), stages.getTemplateCounts().end());
  planInfo.tilingType = tilingType;
  planInfo.mainName = signature.entryName;
  for (func::FuncOp templateFunc : module.getOps<func::FuncOp>()) {
    if (afplan::getVariant(templateFunc) &&
        failed(validateTemplateAttributes(templateFunc, planInfo.templateCountsPerStage))) {
      return failure();
    }
  }
  for (auto [stage, count] : llvm::enumerate(planInfo.templateCountsPerStage)) {
    if (failed(collectStageInfo({module, stage, count}, tilingType, signature, planInfo))) {
      return failure();
    }
  }
  reorderMainArguments(signature, planInfo);
  signature.tilingArgument = signature.arguments.size();
  SmallVector<Type> mainInputs(signature.argumentTypes);
  for (Attribute stageType : signature.tilingType.getTypesAttr()) {
    mainInputs.push_back(cast<TypeAttr>(stageType).getValue());
  }
  signature.entryType = FunctionType::get(signature.tilingType.getContext(), mainInputs, {});
  planInfo.mainType = signature.entryType;
  return planInfo;
}

Value readMember(OpBuilder &builder, Location loc, Value value, StringRef field) {
  return createTilingSchemaMember(builder, loc, value, field);
}

Value castSelectorToIndex(OpBuilder &builder, Location loc, Value selector) {
  if (selector.getType().isIndex()) {
    return selector;
  }
  if (!cast<IntegerType>(selector.getType()).isSignless()) {
    return builder.create<emitc::CastOp>(loc, builder.getIndexType(), selector);
  }
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), selector);
}

// `mapping` 的三态各自去哪儿取实参。workspace 偏移取的是 stage tiling 的 `workspace<ID>`
// 成员：`GenTilingData` 把 workspace 字段复制进每个 Stage 结构体（`GenTilingData.cpp` 的
// `collectStageFields`），入口里所有 tiling 成员也都是从 stage 结构体上读的。成员不存在就
// 报错，不退化成 0——退化会让两块 workspace 叠在同一个地址上写，是静默的数据损坏。
FailureOr<SmallVector<Value>> buildTemplateArguments(OpBuilder &builder, Location loc, ValueRange arguments,
                                                     ArrayRef<TemplateArgumentSource> mapping, Value stageTiling,
                                                     func::FuncOp reportOn) {
  SmallVector<Value> result;
  result.reserve(mapping.size());
  for (auto [position, source] : llvm::enumerate(mapping)) {
    switch (source.kind) {
      case TemplateArgumentSource::Kind::MainArgument: {
        // 将 ABI 参数转回 template 的 ranked 视图，不产生数据搬运。
        Value argument = arguments[source.index];
        Type expected = reportOn.getFunctionType().getInput(position);
        if (argument.getType() != expected) {
          argument = builder.create<memref::CastOp>(loc, expected, argument);
        }
        result.push_back(argument);
        break;
      }
      case TemplateArgumentSource::Kind::StageTiling:
        result.push_back(stageTiling);
        break;
      case TemplateArgumentSource::Kind::StageTilingWorkspaceOffset: {
        const std::string field = "workspace" + std::to_string(source.index);
        auto schema = getTilingSchema(stageTiling.getType());
        if (!schema || !schema.getField(field)) {
          return reportOn.emitError("GenKernelEntry: tiling data has no '")
                 << field << "', so the offset of workspace " << source.index
                 << " would never be assigned; ATT did not declare this workspace";
        }
        result.push_back(readMember(builder, loc, stageTiling, field));
        break;
      }
    }
  }
  return result;
}

// 一张 index switch 分派表：选择子、各分支取值，以及分支对应的被调函数。
struct SwitchDispatch {
  Value selector;
  ArrayRef<int64_t> cases;
  ArrayRef<func::FuncOp> callees;
};

void buildSwitch(OpBuilder &builder, Location loc, SwitchDispatch dispatch,
                 function_ref<void(OpBuilder &, func::FuncOp, size_t)> buildCase) {
  Value selector = castSelectorToIndex(builder, loc, dispatch.selector);
  ArrayRef<int64_t> cases = dispatch.cases;
  ArrayRef<func::FuncOp> callees = dispatch.callees;
  assert(cases.size() == callees.size());
  auto switchOp = builder.create<scf::IndexSwitchOp>(loc, TypeRange{}, selector, cases, cases.size());
  switchOp.getDefaultRegion().emplaceBlock();
  // 与聚合入口的 plan 分派同理：stage 的 `tiling_key` 由 host 侧 tiling 写，落不进已下降的
  // 模板编号就说明两侧对模板集的认知不一致。空 `scf.yield` 会静默跳过这个 stage，后面的
  // stage 照常跑在没算过的数据上——错得比整条不执行还隐蔽。
  OpBuilder defaultBuilder = OpBuilder::atBlockEnd(&switchOp.getDefaultRegion().front());
  defaultBuilder.create<ascendc::TrapOp>(loc);
  defaultBuilder.create<scf::YieldOp>(loc);
  for (auto [index, callee] : llvm::enumerate(callees)) {
    Region &region = switchOp.getCaseRegions()[index];
    region.emplaceBlock();
    OpBuilder caseBuilder = OpBuilder::atBlockEnd(&region.front());
    buildCase(caseBuilder, callee, index);
    caseBuilder.create<scf::YieldOp>(loc);
  }
}

// 逐 stage 生成分派时不变的那部分：往哪儿建、建哪个 Plan、main 的参数与 tiling 参数位置。
struct StageEmitContext {
  OpBuilder &builder;
  Location loc;
  const PlanInfo &planInfo;
  ValueRange arguments;
  size_t tilingArgument;
  // `buildSwitch` 的分支回调不能返回 failure（它要被 `scf.index_switch` 的区域构造器
  // 吃掉），所以实参解析的失败经这里带出来。
  bool &failed;
};

void buildStage(const StageEmitContext &ctx, size_t stage) {
  const PlanInfo &planInfo = ctx.planInfo;
  if (stage == planInfo.templates.size()) {
    return;
  }
  Location loc = ctx.loc;
  Value stageTiling = ctx.arguments[ctx.tilingArgument + stage];
  Value selector = readMember(ctx.builder, loc, stageTiling, "tiling_key");
  buildSwitch(ctx.builder, loc, {selector, planInfo.tilingKeys[stage], planInfo.templates[stage]},
              [&](OpBuilder &caseBuilder, func::FuncOp templateFunc, size_t index) {
                ArrayRef<TemplateArgumentSource> mapping = planInfo.argumentMappings[stage][index];
                FailureOr<SmallVector<Value>> callArguments =
                    buildTemplateArguments(caseBuilder, loc, ctx.arguments, mapping, stageTiling, templateFunc);
                if (failed(callArguments)) {
                  ctx.failed = true;
                  return;
                }
                caseBuilder.create<func::CallOp>(loc, templateFunc, *callArguments);
                if (templateFunc->hasAttr(mlir::ascend::kernel::kAscendCNeedSyncAllAttrName)) {
                  caseBuilder.create<ascendc::SyncAllHardOp>(loc, caseBuilder.getUnitAttr());
                }
              });
  buildStage(ctx, stage + 1);
}

// `KernelArgumentIdentity` 到参数声明的翻译。两者同形，但前者是 pass 内部的中间结果，
// 后者是写进 IR 的公开声明——分开写，免得哪天中间结构改了顺手改掉 IR 形态。
void setPlanMainArgumentRole(func::FuncOp main, unsigned index, const KernelArgumentIdentity &argument) {
  switch (argument.kind) {
    case KernelArgumentKind::Input:
      mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Input, argument.index);
      return;
    case KernelArgumentKind::ScalarInput:
      mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::ScalarInput, argument.index);
      return;
    case KernelArgumentKind::Output:
      mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Output, argument.index);
      return;
    case KernelArgumentKind::Workspace:
      mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Workspace);
      return;
    case KernelArgumentKind::WorkspaceOffset:
      // main 不收它——`collectTemplateAbi` 已把这一格从形参表里摘掉，实参在调用点从根
      // tiling 上读。走到这里说明摘漏了。
      main.emitError("GenKernelEntry: workspace offset must not become a plan main argument");
      return;
    case KernelArgumentKind::Tiling:
      mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Tiling);
      return;
  }
}

LogicalResult createPlanMain(PlanInfo &planInfo, const MainSignature &signature) {
  OpBuilder builder(planInfo.module.getBodyRegion());
  builder.setInsertionPointToEnd(planInfo.module.getBody());
  auto main = builder.create<func::FuncOp>(planInfo.module.getLoc(), planInfo.mainName, planInfo.mainType);
  // 每个参数都声明身份，workspace 与 tiling 也不例外：下游的 GenAggregateKernelEntry
  // 逐参数问 `getKernelArgumentIdentity`，漏标的参数在那里就没有身份可言。
  for (auto [index, argument] : llvm::enumerate(signature.arguments)) {
    setPlanMainArgumentRole(main, index, argument);
  }
  // tiling 形参一 stage 一个，序号就是 stage：`GenAggregateKernelEntry` 靠它把每个形参和
  // 聚合 TilingData 上对应的那个 group 成员对上，而不是靠「第几个 tiling 形参」这种位置约定。
  for (size_t index = signature.tilingArgument; index < main.getNumArguments(); ++index) {
    mlir::afir::setKernelArgAttr(main, index, mlir::afir::KernelArgRole::Tiling,
                                 static_cast<int64_t>(index - signature.tilingArgument));
  }
  Block *entry = main.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  bool failed = false;
  StageEmitContext stageCtx{builder, main.getLoc(), planInfo, entry->getArguments(), signature.tilingArgument, failed};
  buildStage(stageCtx, 0);
  if (failed) {
    return mlir::failure();
  }
  builder.create<func::ReturnOp>(main.getLoc());
  return mlir::success();
}

struct GenKernelEntryPass : public PassWrapper<GenKernelEntryPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenKernelEntryPass)

  StringRef getArgument() const final {
    return "afir-gen-kernel-entry";
  }

  StringRef getDescription() const final {
    return "Generate Plan and fused kernel entries";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    // `memref`：`_main` 的 GM 形参按 ABI 形态声明，调用 template 的点上用 `memref.cast` 转回各
    // template 声明的视图（见 `appendMainArgument`）。
    registry.insert<afplan::AFPlanDialect, arith::ArithDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect,
                    emitc::EmitCDialect, func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    if (mlir::afir::isPlanModule(getOperation())) {
      auto signature = deriveMainSignature(getOperation());
      if (failed(signature)) {
        return signalPassFailure();
      }
      auto planInfo = collectTemplates(getOperation(), *signature);
      if (failed(planInfo)) {
        return signalPassFailure();
      }
      if (failed(createPlanMain(*planInfo, *signature))) {
        return signalPassFailure();
      }
      return;
    }
    getOperation()->emitError("GenKernelEntry: expected ") << mlir::afir::kModuleKindAttrName << "=plan";
    signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> mlir::ascend::kernel::createGenKernelEntryPass() {
  return std::make_unique<GenKernelEntryPass>();
}

void mlir::ascend::kernel::registerGenKernelEntryPass() {
  PassRegistration<GenKernelEntryPass>();
}
