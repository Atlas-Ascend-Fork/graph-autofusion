/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/TilingSource/TilingProgramSpec.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "common/tiling_schema/schedule_group_names.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::bridge {

namespace {

struct PlanModuleMetadata {
  int64_t subgraph_id;
  int64_t plan_id;
  mlir::ascend::kernel::TilingSchema schema;
  mlir::afir::TargetAttr target;
};

struct RelationTarget {
  int64_t dst_stage;
  int64_t src_stage;
  std::string dst_var;
};

FailureOr<PlanModuleMetadata> ReadPlanModuleMetadata(ModuleOp module) {
  // #afplan.candidate 的 verifier 已校验身份字段存在且非负。
  // 此处只检查属性是否存在，不重复逐字段校验。
  mlir::afplan::CandidateAttr candidate = mlir::afplan::getCandidate(module);
  if (!candidate) {
    module.emitError("TilingProgramSpec: missing or invalid ") << mlir::afplan::kCandidateAttrName;
    return failure();
  }
  auto schemaAttr = module->getAttrOfType<TypeAttr>(mlir::ascend::kernel::kAscendCTilingSchemaAttrName);
  auto schema =
      schemaAttr ? ascend::kernel::getTilingSchema(schemaAttr.getValue()) : mlir::ascend::kernel::TilingSchema();
  if (!schema) {
    module.emitError("TilingProgramSpec: missing or invalid ")
        << mlir::ascend::kernel::kAscendCTilingSchemaAttrName << " for subgraph " << candidate.getSubgraph() << " plan "
        << candidate.getPlan();
    return failure();
  }
  // #afir.target 的 verifier 已校验核数与 UB 容量的值域，此处只检查属性是否存在。
  mlir::afir::TargetAttr target = mlir::afir::getTargetAttr(module);
  if (!target) {
    module.emitError("TilingProgramSpec: missing or invalid ")
        << mlir::afir::kTargetAttrName << " for subgraph " << candidate.getSubgraph() << " plan "
        << candidate.getPlan();
    return failure();
  }
  return PlanModuleMetadata{candidate.getSubgraph(), candidate.getPlan(), schema, target};
}

// 聚合 host tiling 只生成一个 ResLimit，因此各候选必须使用相同目标和资源上限。
// 目标属性经过 uniquing，可直接比较属性，无需拆分字段逐项比较。
LogicalResult MergeTargetMetadata(ModuleOp module, const PlanModuleMetadata &metadata, TilingProgramSpec &spec) {
  if (!spec.target) {
    spec.target = metadata.target;
    return success();
  }
  if (spec.target != metadata.target) {
    module.emitError("TilingProgramSpec: mismatched ")
        << mlir::afir::kTargetAttrName << " for subgraph " << metadata.subgraph_id << " plan " << metadata.plan_id
        << "; expected " << spec.target << ", got " << metadata.target;
    return failure();
  }
  return success();
}

FailureOr<std::vector<std::string>> ReadShapeParams(ModuleOp module, const PlanModuleMetadata &metadata) {
  std::vector<std::string> shapeParams;
  if (auto shapeVars = module->getAttrOfType<ArrayAttr>(mlir::afir::kFrontendShapeVarsAttrName)) {
    for (Attribute attr : shapeVars) {
      auto name = dyn_cast<StringAttr>(attr);
      if (!name) {
        module.emitError("TilingProgramSpec: ")
            << mlir::afir::kFrontendShapeVarsAttrName << " must contain only strings for subgraph "
            << metadata.subgraph_id << " plan " << metadata.plan_id;
        return failure();
      }
      shapeParams.push_back(name.getValue().str());
    }
  }
  return shapeParams;
}

LogicalResult MergeShapeParams(ModuleOp module, const PlanModuleMetadata &metadata,
                               std::vector<std::string> moduleShapeParams, TilingProgramSpec &spec) {
  if (spec.plans.empty()) {
    spec.shape_params = std::move(moduleShapeParams);
  } else if (spec.shape_params != moduleShapeParams) {
    module.emitError("TilingProgramSpec: mismatched ") << mlir::afir::kFrontendShapeVarsAttrName << " for subgraph "
                                                       << metadata.subgraph_id << " plan " << metadata.plan_id;
    return failure();
  }
  return success();
}

FailureOr<std::vector<RelationTarget>> ReadRelationTargets(ModuleOp module, const PlanModuleMetadata &metadata) {
  std::vector<RelationTarget> relationTargets;
  if (auto relations = module->getAttrOfType<ArrayAttr>(afir::kVarRelationsAttrName)) {
    for (Attribute attr : relations) {
      auto relation = dyn_cast<DictionaryAttr>(attr);
      auto dstStage = relation ? relation.getAs<IntegerAttr>(afir::kVarRelationDstStageKey) : IntegerAttr();
      auto srcStage = relation ? relation.getAs<IntegerAttr>(afir::kVarRelationSrcStageKey) : IntegerAttr();
      auto dstVar = relation ? relation.getAs<StringAttr>(afir::kVarRelationDstVarKey) : StringAttr();
      auto srcExpr = relation ? relation.getAs<afir::ExprAttr>(afir::kVarRelationSrcExprKey) : afir::ExprAttr();
      if (!dstStage || dstStage.getInt() < 0 || !srcStage || srcStage.getInt() < 0 || !dstVar ||
          dstVar.getValue().empty() || !srcExpr) {
        module.emitError("TilingProgramSpec: malformed ") << mlir::afir::kVarRelationsAttrName << " entry for subgraph "
                                                          << metadata.subgraph_id << " plan " << metadata.plan_id;
        return failure();
      }
      relationTargets.push_back({dstStage.getInt(), srcStage.getInt(), dstVar.getValue().str()});
    }
  }
  return relationTargets;
}

FailureOr<std::vector<TilingStageSpec>> BuildStageSpecs(ModuleOp module, const PlanModuleMetadata &metadata) {
  std::vector<TilingStageSpec> stages;
  int64_t stageId = -1;
  for (auto [nameAttr, typeAttr] : llvm::zip(metadata.schema.getNamesAttr(), metadata.schema.getTypesAttr())) {
    ++stageId;
    auto stageName = dyn_cast<StringAttr>(nameAttr);
    auto stageTypeAttr = dyn_cast<TypeAttr>(typeAttr);
    auto stageType = stageTypeAttr ? ascend::kernel::getTilingSchema(stageTypeAttr.getValue())
                                   : mlir::ascend::kernel::TilingSchema();
    if (!stageName || !stageType) {
      module.emitError("TilingProgramSpec: ")
          << mlir::ascend::kernel::kAscendCTilingSchemaAttrName << " contains an invalid stage for subgraph "
          << metadata.subgraph_id << " plan " << metadata.plan_id;
      return failure();
    }
    // 载体字段按 stage 排列，stage id 可直接用作下标。
    // 按共享命名规则校验成员名，该名称也用于聚合 TilingData 的 tiling_field_name。
    const std::string expected =
        mlir::common::MakeScheduleGroupTilingFieldName(metadata.subgraph_id, metadata.plan_id, stageId);
    if (stageName.getValue() != expected) {
      module.emitError("TilingProgramSpec: stage field name '")
          << stageName.getValue() << "' does not match '" << expected << "' for subgraph " << metadata.subgraph_id
          << " plan " << metadata.plan_id;
      return failure();
    }
    TilingStageSpec stage;
    stage.stage_id = stageId;
    stage.tiling_field_name = expected;
    for (Attribute fieldAttr : stageType.getNamesAttr()) {
      auto field = dyn_cast<StringAttr>(fieldAttr);
      if (!field) {
        module.emitError("TilingProgramSpec: stage ") << stageId << " schema contains a non-string field name";
        return failure();
      }
      stage.fields.push_back(field.getValue().str());
    }
    stages.push_back(std::move(stage));
  }
  return stages;
}

LogicalResult ApplyRelationTargets(ModuleOp module, const PlanModuleMetadata &metadata,
                                   ArrayRef<RelationTarget> relationTargets, std::vector<TilingStageSpec> &stages) {
  for (const RelationTarget &relation : relationTargets) {
    auto dstStage =
        llvm::find_if(stages, [&](const TilingStageSpec &stage) { return stage.stage_id == relation.dst_stage; });
    auto srcStage =
        llvm::find_if(stages, [&](const TilingStageSpec &stage) { return stage.stage_id == relation.src_stage; });
    if (dstStage == stages.end() || srcStage == stages.end()) {
      module.emitError("TilingProgramSpec: ")
          << mlir::afir::kVarRelationsAttrName << " references an unknown stage for subgraph " << metadata.subgraph_id
          << " plan " << metadata.plan_id << ": src_stage=" << relation.src_stage
          << ", dst_stage=" << relation.dst_stage;
      return failure();
    }
    if (!llvm::is_contained(dstStage->fields, relation.dst_var)) {
      module.emitError("TilingProgramSpec: ")
          << mlir::afir::kVarRelationsAttrName << " destination '" << relation.dst_var << "' is not a field of stage "
          << relation.dst_stage << " for subgraph " << metadata.subgraph_id << " plan " << metadata.plan_id;
      return failure();
    }
    if (!llvm::is_contained(dstStage->derived_shape_vars, relation.dst_var)) {
      dstStage->derived_shape_vars.push_back(relation.dst_var);
    }
  }
  return success();
}

LogicalResult AppendTemplateSpecs(ModuleOp module, const PlanModuleMetadata &metadata,
                                  std::vector<TilingStageSpec> &stages) {
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    mlir::afplan::VariantAttr variant = mlir::afplan::getVariant(function);
    if (!variant) {
      continue;
    }
    auto workspace = function->getAttrOfType<afir::ExprAttr>(mlir::afir::kWorkspaceSizeAttrName);
    if (!workspace) {
      function.emitError("TilingProgramSpec: missing or invalid stage/template/workspace contract");
      return failure();
    }
    auto stage =
        llvm::find_if(stages, [&](const TilingStageSpec &item) { return item.stage_id == variant.getStage(); });
    if (stage == stages.end()) {
      function.emitError("TilingProgramSpec: template references unknown stage ") << variant.getStage();
      return failure();
    }
    afir::Result<afir::Expression> workspaceExpression = afir::Expression::Parse(workspace.getValue().getValue());
    auto workspaceSymbols = workspaceExpression
                                ? afir::detail::ExpressionAccess::GetFreeSymbols(workspaceExpression.value())
                                : afir::Result<std::vector<std::string>>(workspaceExpression.error());
    if (!workspaceSymbols) {
      function.emitError("TilingProgramSpec: invalid ")
          << mlir::afir::kWorkspaceSizeAttrName << ": " << workspaceSymbols.error().message;
      return failure();
    }
    for (const std::string &symbol : workspaceSymbols.value()) {
      if (!llvm::is_contained(stage->fields, symbol)) {
        function.emitError("TilingProgramSpec: ")
            << mlir::afir::kWorkspaceSizeAttrName << " references unknown stage field '" << symbol << "'";
        return failure();
      }
    }
    auto graph = function->getAttrOfType<afir::AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
    stage->templates.push_back(
        {variant.getTemplateId(), workspace, graph ? graph.getTilingKey() : variant.getTemplateId()});
  }
  for (TilingStageSpec &stage : stages) {
    llvm::sort(stage.templates, [](const TilingTemplateSpec &lhs, const TilingTemplateSpec &rhs) {
      return lhs.template_id < rhs.template_id;
    });
    if (stage.templates.empty()) {
      module.emitError("TilingProgramSpec: stage ")
          << stage.stage_id << " has no template for subgraph " << metadata.subgraph_id << " plan " << metadata.plan_id;
      return failure();
    }
  }
  return success();
}

FailureOr<TilingPlanSpec> BuildPlanSpec(ModuleOp module, const PlanModuleMetadata &metadata) {
  auto relationTargets = ReadRelationTargets(module, metadata);
  if (failed(relationTargets)) {
    return failure();
  }
  auto stages = BuildStageSpecs(module, metadata);
  if (failed(stages)) {
    return failure();
  }
  if (failed(ApplyRelationTargets(module, metadata, *relationTargets, *stages)) ||
      failed(AppendTemplateSpecs(module, metadata, *stages))) {
    return failure();
  }
  TilingPlanSpec plan{metadata.subgraph_id, metadata.plan_id, std::move(*stages), {}};
  if (auto ids = module->getAttrOfType<DenseI64ArrayAttr>(afir::kWorkspaceTensorIdsAttrName)) {
    plan.workspace_tensor_ids.assign(ids.asArrayRef().begin(), ids.asArrayRef().end());
  }
  return plan;
}

}  // namespace

FailureOr<TilingProgramSpec> buildAggregateTilingProgramSpec(ArrayRef<ModuleOp> planModules) {
  if (planModules.empty()) {
    return failure();
  }
  TilingProgramSpec spec;
  for (ModuleOp module : planModules) {
    auto metadata = ReadPlanModuleMetadata(module);
    if (failed(metadata) || failed(MergeTargetMetadata(module, *metadata, spec))) {
      return failure();
    }
    auto shapeParams = ReadShapeParams(module, *metadata);
    if (failed(shapeParams) || failed(MergeShapeParams(module, *metadata, std::move(*shapeParams), spec))) {
      return failure();
    }
    auto plan = BuildPlanSpec(module, *metadata);
    if (failed(plan)) {
      return failure();
    }
    spec.plans.push_back(std::move(*plan));
  }
  llvm::sort(spec.plans, [](const TilingPlanSpec &lhs, const TilingPlanSpec &rhs) {
    return std::tie(lhs.subgraph_id, lhs.plan_id) < std::tie(rhs.subgraph_id, rhs.plan_id);
  });
  return spec;
}

}  // namespace mlir::bridge
