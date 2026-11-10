/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/GenAggregateTilingData.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "dialect/AFIR/IR/PlanMetadata.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "common/tiling_schema/schedule_group_names.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

#include <string>
#include <tuple>

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {
struct AggregatePlanInfo {
  mlir::afplan::CandidateAttr candidate;
  ModuleOp module;
  // 本 Plan 的 stage schema 载体（字段按 stage 次序，字段名即它们在根上的成员名）。
  mlir::ascend::kernel::TilingSchema tilingType;
};

// ATT 的最外层 TilingData 里没有 plan 这一层：每个 schedule group 的结构体直接挂在根上，
// 叫 `graph<子图>_result<plan>_g<stage>_tiling_data`（`codegen_tiling_data.cpp` 的
// `ProcessMultiGroup`，期望布局由 `tests/ut/codegen/test_codegen_tiling_data.cpp:487` 钉住）。
// 所以这里把载体的字段原样摊开，而不是把载体本身当成一个字段挂上去。
size_t getStageCount(const AggregatePlanInfo &info) {
  return info.tilingType.getNamesAttr().size();
}

// 扁平形态：单子图、单 plan、单 stage。与 ATT `tiling_field_schema.cpp` 的 `is_flat`
// 同一个判据——它决定 ATT 用裸 `tiling_key` 还是 `graph<子图>_tiling_key`，也决定根上要不要
// 自带 `block_dim`/`corenum`/`ub_size`/`hbm_size`。判据缺了 stage 这一维，两 stage 单 plan
// 时 ATT 生成 `graph0_tiling_key` 而根上没有这个字段。
bool isFlatLayout(ArrayRef<AggregatePlanInfo> infos) {
  return infos.size() == 1U && !infos.front().module->hasAttr(mlir::afplan::kStaticSelectionAttrName) &&
         infos.front().candidate.getSubgraph() == 0 && getStageCount(infos.front()) == 1U;
}

FailureOr<SmallVector<AggregatePlanInfo>> collectAndValidateAggregatePlanInfos(ArrayRef<ModuleOp> planModules,
                                                                               MLIRContext *context) {
  SmallVector<AggregatePlanInfo> infos;
  for (ModuleOp module : planModules) {
    if (module.getContext() != context) {
      module.emitError("GenAggregateTilingData: Plan modules must use the same MLIRContext");
      return failure();
    }
    mlir::afplan::CandidateAttr candidate = mlir::afplan::getCandidate(module);
    auto schemaAttr = module->getAttrOfType<TypeAttr>(mlir::ascend::kernel::kAscendCTilingSchemaAttrName);
    auto schema = schemaAttr ? getTilingSchema(schemaAttr.getValue()) : mlir::ascend::kernel::TilingSchema();
    if (!mlir::afir::isPlanModule(module) || !candidate || !schema) {
      module.emitError("GenAggregateTilingData: invalid Plan contract");
      return failure();
    }
    if (auto cube = module->getAttrOfType<IntegerAttr>(mlir::afir::kCubeTypeAttrName); cube && cube.getInt() != -1) {
      module.emitError("GenAggregateTilingData: Cube/CV Plan is not supported, ")
          << mlir::afir::kCubeTypeAttrName << "=" << cube.getInt();
      return failure();
    }
    infos.push_back({candidate, module, schema});
  }
  return infos;
}

LogicalResult sortAndValidateSubgraphAndPlanIds(SmallVector<AggregatePlanInfo> &infos) {
  llvm::sort(infos, [](const AggregatePlanInfo &lhs, const AggregatePlanInfo &rhs) {
    return std::make_pair(lhs.candidate.getSubgraph(), lhs.candidate.getPlan()) <
           std::make_pair(rhs.candidate.getSubgraph(), rhs.candidate.getPlan());
  });
  int64_t currentSubgraph = -1;
  int64_t previousPlan = -1;
  for (AggregatePlanInfo &info : infos) {
    if (info.candidate.getSubgraph() != currentSubgraph) {
      if (info.candidate.getSubgraph() != currentSubgraph + 1) {
        info.module.emitError("GenAggregateTilingData: subgraph IDs must be contiguous");
        return failure();
      }
      currentSubgraph = info.candidate.getSubgraph();
      previousPlan = -1;
    }
    const auto planId = info.candidate.getPlan();
    if (planId <= previousPlan) {
      info.module.emitError("GenAggregateTilingData: duplicate or invalid plan ID");
      return failure();
    }
    // Frozen winners retain their original plan ID, even when it is not zero.
    if (!info.module->hasAttr(mlir::afplan::kStaticSelectionAttrName) && planId != previousPlan + 1) {
      info.module.emitError("GenAggregateTilingData: plan IDs must be contiguous");
      return failure();
    }
    previousPlan = planId;
  }
  return success();
}

OwningOpRef<ModuleOp> createEntryModule(MLIRContext *context) {
  OpBuilder builder(context);
  auto entry = ModuleOp::create(builder.getUnknownLoc(), "AutofuseEntry");
  entry->setAttr(mlir::afir::kModuleKindAttrName, builder.getStringAttr(mlir::afir::kEntryModuleKind));
  return entry;
}

// 非扁平形态下 ATT 的 host tiling 在根上直接读写这四个量，再拷进各 group 的结构体
// （`graph0_result0_g0_tiling_data.set_ub_size(tiling_data.get_ub_size())`），legacy 的根
// 结构体也自带它们。扁平形态下**不能**加：那时根上这几个名字是转发到唯一 stage 结构体的
// 访问器（`TilingDataHeaderEmitter` 的 `emitRootForwardingAccessors`），根一旦自带同名
// 实字段，ATT 写的是根、device 读的是 stage 的那一份，两边静默对不上。
void addCommonScalarFields(OpBuilder &builder, ArrayRef<AggregatePlanInfo> infos, SmallVector<Attribute> &types,
                           SmallVector<Attribute> &names) {
  if (isFlatLayout(infos)) {
    return;
  }
  auto ui32 = IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned);
  for (llvm::StringRef field : {"block_dim", "corenum", "ub_size", "hbm_size"}) {
    types.push_back(TypeAttr::get(ui32));
    names.push_back(builder.getStringAttr(field));
  }
}

void addGraphSelectorFields(OpBuilder &builder, ArrayRef<AggregatePlanInfo> infos, SmallVector<Attribute> &types,
                            SmallVector<Attribute> &names) {
  if (infos.empty() || isFlatLayout(infos)) {
    return;
  }
  int64_t graphCount = infos.back().candidate.getSubgraph() + 1;
  for (int64_t graphId = 0; graphId < graphCount; ++graphId) {
    types.push_back(TypeAttr::get(IntegerType::get(builder.getContext(), 32, IntegerType::Unsigned)));
    names.push_back(builder.getStringAttr("graph" + std::to_string(graphId) + "_tiling_key"));
  }
}

void addPlanTilingFields(ArrayRef<AggregatePlanInfo> infos, SmallVector<Attribute> &types,
                         SmallVector<Attribute> &names) {
  for (const AggregatePlanInfo &info : infos) {
    for (auto [name, stageType] : llvm::zip(info.tilingType.getNamesAttr(), info.tilingType.getTypesAttr())) {
      types.push_back(stageType);
      names.push_back(name);
    }
  }
}

void addWorkspaceFields(OpBuilder &builder, ArrayRef<AggregatePlanInfo> infos, SmallVector<Attribute> &types,
                        SmallVector<Attribute> &names) {
  SmallVector<int64_t> workspaceIds;
  for (const AggregatePlanInfo &info : infos) {
    auto ids = info.module->getAttrOfType<DenseI64ArrayAttr>(mlir::afir::kWorkspaceTensorIdsAttrName);
    if (!ids) {
      continue;
    }
    for (int64_t id : ids.asArrayRef()) {
      if (id >= 0) {
        workspaceIds.push_back(id);
      }
    }
  }
  llvm::sort(workspaceIds);
  workspaceIds.erase(std::unique(workspaceIds.begin(), workspaceIds.end()), workspaceIds.end());
  auto ui64 = IntegerType::get(builder.getContext(), 64, IntegerType::Unsigned);
  for (int64_t id : workspaceIds) {
    types.push_back(TypeAttr::get(ui64));
    names.push_back(builder.getStringAttr("workspace" + std::to_string(id)));
  }
}

mlir::ascend::kernel::TilingSchema buildAggregateTilingSchema(OpBuilder &builder, ArrayRef<AggregatePlanInfo> infos) {
  SmallVector<Attribute> types;
  SmallVector<Attribute> names;
  addCommonScalarFields(builder, infos, types, names);
  addGraphSelectorFields(builder, infos, types, names);
  addWorkspaceFields(builder, infos, types, names);
  addPlanTilingFields(infos, types, names);
  return mlir::ascend::kernel::TilingSchema::get(builder.getContext(), builder.getStringAttr("AutofuseTilingData"),
                                                 builder.getArrayAttr(types), builder.getArrayAttr(names));
}

void attachAggregateMetadata(ModuleOp entry, mlir::ascend::kernel::TilingSchema aggregate) {
  entry->setAttr(mlir::ascend::kernel::kAscendCTilingSchemaAttrName, TypeAttr::get(aggregate));
}

void attachPlanContracts(OpBuilder &builder, ModuleOp entry, ArrayRef<AggregatePlanInfo> infos) {
  SmallVector<Attribute> contracts;
  for (const AggregatePlanInfo &info : infos) {
    // 候选身份在这里与挂在 plan module 上的是同一个属性，而不是把两个整数拆键复述一遍
    // ——后者是「第三份编码」，读回时还得靠 `readPlanEntryMetadata` 补上非负校验。
    //
    // tiling 字段是**一 stage 一个**、按 stage 次序排的一张表，而不是单个 plan 级字段名：
    // 根上没有 plan 这一层，`_main` 也一 stage 收一个 tiling 形参。
    contracts.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr(mlir::afplan::kCandidateAttrName, info.candidate),
         builder.getNamedAttr(mlir::ascend::kernel::kAscendCPlanEntryTilingFieldKey,
                              ArrayAttr::get(builder.getContext(), info.tilingType.getNamesAttr().getValue()))}));
  }
  entry->setAttr(mlir::ascend::kernel::kAscendCPlanEntriesAttrName, builder.getArrayAttr(contracts));
}
}  // namespace

FailureOr<OwningOpRef<ModuleOp>> mlir::ascend::kernel::runGenAggregateTilingData(ArrayRef<ModuleOp> planModules) {
  if (planModules.empty()) {
    return failure();
  }
  MLIRContext *context = const_cast<ModuleOp &>(planModules.front()).getContext();
  auto infos = collectAndValidateAggregatePlanInfos(planModules, context);
  if (failed(infos) || failed(sortAndValidateSubgraphAndPlanIds(*infos))) {
    return failure();
  }

  OpBuilder builder(context);
  auto entry = createEntryModule(context);
  auto aggregate = buildAggregateTilingSchema(builder, *infos);
  if (!aggregate) {
    (*entry).emitError("GenAggregateTilingData: invalid aggregate tiling field layout");
    return failure();
  }
  attachAggregateMetadata(*entry, aggregate);
  attachPlanContracts(builder, *entry, *infos);
  return entry;
}
