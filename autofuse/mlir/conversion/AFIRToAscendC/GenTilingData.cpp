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
#include "common/tiling_schema/canonical_resource_names.h"
#include "common/tiling_schema/schedule_group_names.h"
#include "support/CxxIdentifier.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <algorithm>
#include <set>
#include <string>
#include <utility>

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {

bool isFieldIdentifier(StringRef name) {
  if (name.empty() || !std::isalpha(static_cast<unsigned char>(name.front()))) {
    return false;
  }
  return llvm::all_of(name, [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'; });
}

// Collects workspace tensor IDs from the module-level attribute and adds
// "workspace<ID>" fields for each non-negative ID.
void addWorkspaceFields(func::FuncOp function, ModuleOp module, const std::function<void(StringRef)> &addField) {
  auto workspaceIds = module->getAttrOfType<DenseI64ArrayAttr>(mlir::afir::kWorkspaceTensorIdsAttrName);
  if (!workspaceIds) {
    return;
  }
  for (int64_t id : workspaceIds.asArrayRef()) {
    if (id >= 0) {
      addField("workspace" + std::to_string(id));
    }
  }
}

// Collects size_var field names from all templates of a stage. Constant
// expressions are skipped to match legacy codegen behavior.
LogicalResult addSizeVarFields(ArrayRef<func::FuncOp> functions, func::FuncOp reportOn,
                               const std::function<void(StringRef)> &addField) {
  for (func::FuncOp templateFunc : functions) {
    auto graph = templateFunc->getAttrOfType<afir::AscGraphAttrGroupsAttr>(mlir::afir::kAscGraphAttrName);
    for (afir::ExprAttr sizeVar : graph.getSizeVar()) {
      StringRef name = sizeVar.getValue().getValue();
      if (name.empty()) {
        return reportOn.emitError("GenTilingData: invalid size_var field") << name;
      }
      int64_t constantValue = 0;
      if (!name.getAsInteger(10, constantValue)) {
        continue;
      }
      if (!isFieldIdentifier(name)) {
        return reportOn.emitError("GenTilingData: size_var is not a field identifier") << name;
      }
      addField(name);
    }
  }
  return success();
}

// Collects queue and buffer IDs recorded by the schedule bridge. tmp buffer 不单列一张
// 表：它的 id 与普通缓冲同一个 id 空间、同一套 `b{id}_size` 命名，bridge 收集时就并进
// `afir.buffer_ids` 了（`ToTemplateInfo`）。这里因此只有两张表，没有第三条分支。
void collectQueueBufferIds(ArrayRef<func::FuncOp> functions, std::set<int64_t> &queueIds,
                           std::set<int64_t> &bufferIds) {
  for (func::FuncOp templateFunc : functions) {
    for (StringRef attrName : {mlir::afir::kQueueIdsAttrName, mlir::afir::kBufferIdsAttrName}) {
      if (auto ids = templateFunc->getAttrOfType<DenseI64ArrayAttr>(attrName)) {
        for (int64_t id : ids.asArrayRef()) {
          if (id < 0) {
            continue;
          }
          (attrName == mlir::afir::kQueueIdsAttrName ? queueIds : bufferIds).insert(id);
        }
      }
    }
  }
}

// Builds an API tiling nested struct type from the template's api_tiling_fields
// attribute and appends it to the stage schema. Returns failure on malformed
// attribute data.
// 正在攒的 TilingData 字段表：类型与名字两列并行推进，外加建属性要用的 context 与
// 默认字段类型 ui32，以及按名字去重的集合。
struct SchemaFields {
  MLIRContext *context;
  Type ui32;
  SmallVector<Attribute> types;
  SmallVector<Attribute> names;
  llvm::StringSet<> seen;

  void addUInt32(StringRef name) {
    if (seen.insert(name).second) {
      types.push_back(TypeAttr::get(ui32));
      names.push_back(StringAttr::get(context, name));
    }
  }
};

LogicalResult addApiTilingFields(ArrayRef<func::FuncOp> functions, func::FuncOp reportOn, SchemaFields &fields) {
  MLIRContext *context = fields.context;
  for (func::FuncOp templateFunc : functions) {
    auto apiFields = templateFunc->getAttrOfType<ArrayAttr>(mlir::afir::kApiTilingFieldsAttrName);
    if (!apiFields) {
      continue;
    }
    for (Attribute field : apiFields) {
      auto item = dyn_cast<DictionaryAttr>(field);
      auto typeName = item ? item.getAs<StringAttr>(afir::kApiTilingTypeKey) : StringAttr();
      auto fieldName = item ? item.getAs<StringAttr>(afir::kApiTilingFieldKey) : StringAttr();
      auto names = item ? item.getAs<ArrayAttr>(afir::kApiTilingFieldsKey) : ArrayAttr();
      if (!typeName || !fieldName || !names || !isFieldIdentifier(fieldName.getValue())) {
        return reportOn.emitError("GenTilingData: invalid API tiling field");
      }
      SmallVector<Attribute> apiTypes;
      SmallVector<Attribute> apiNames;
      for (Attribute apiField : names) {
        auto apiName = dyn_cast<StringAttr>(apiField);
        if (!apiName || !isFieldIdentifier(apiName.getValue())) {
          return reportOn.emitError("GenTilingData: invalid API tiling member");
        }
        apiTypes.push_back(TypeAttr::get(fields.ui32));
        apiNames.push_back(apiName);
      }
      auto apiType = mlir::ascend::kernel::TilingSchema::get(context, typeName, ArrayAttr::get(context, apiTypes),
                                                             ArrayAttr::get(context, apiNames));
      if (!apiType) {
        return reportOn.emitError("GenTilingData: invalid API tiling field layout");
      }
      auto existing = llvm::find_if(fields.names, [&](Attribute name) { return name == fieldName; });
      if (existing != fields.names.end()) {
        auto existingType = cast<TypeAttr>(fields.types[existing - fields.names.begin()]).getValue();
        if (existingType != apiType) {
          return reportOn.emitError("GenTilingData: conflicting API tiling schema for field ") << fieldName.getValue();
        }
        continue;
      }
      fields.types.push_back(TypeAttr::get(apiType));
      fields.names.push_back(fieldName);
    }
  }
  return success();
}

// Field order: common, workspace, tiling_key, size_var, API tiling, q/b sizes.
LogicalResult collectStageFields(ArrayRef<func::FuncOp> functions, SchemaFields &fields) {
  func::FuncOp function = functions.front();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  auto addField = [&](StringRef name) { fields.addUInt32(name); };

  addField("block_dim");
  addField("corenum");
  addField("ub_size");
  addField("hbm_size");
  addWorkspaceFields(function, module, [&](StringRef name) {
    if (fields.seen.insert(name).second) {
      fields.types.push_back(TypeAttr::get(getAscendCWorkspaceOffsetType(fields.context)));
      fields.names.push_back(StringAttr::get(fields.context, name));
    }
  });
  addField("tiling_key");
  if (failed(addSizeVarFields(functions, function, addField))) {
    return failure();
  }

  std::set<int64_t> queueIds;
  std::set<int64_t> bufferIds;
  collectQueueBufferIds(functions, queueIds, bufferIds);

  if (failed(addApiTilingFields(functions, function, fields))) {
    return failure();
  }

  for (int64_t id : queueIds) {
    addField(mlir::common::MakeQueueSizeName(id));
  }
  for (int64_t id : bufferIds) {
    addField(mlir::common::MakeBufferSizeName(id));
  }
  return success();
}

// Builds the TilingData schema for one stage by merging all template facts.
FailureOr<mlir::ascend::kernel::TilingSchema> buildStageSchema(ArrayRef<func::FuncOp> functions,
                                                               afplan::CandidateAttr candidate, int64_t stage) {
  if (functions.empty()) {
    return failure();
  }
  for (func::FuncOp function : functions) {
    if (!function->getAttrOfType<afir::AscGraphAttrGroupsAttr>(mlir::afir::kAscGraphAttrName)) {
      return function.emitError("GenTilingData: template is missing ") << mlir::afir::kAscGraphAttrName << " facts";
    }
  }
  func::FuncOp function = functions.front();
  MLIRContext *context = function.getContext();
  SchemaFields fields{context, IntegerType::get(context, 32, IntegerType::Unsigned)};
  if (failed(collectStageFields(functions, fields))) {
    return failure();
  }

  std::string typeName =
      mlir::common::MakeScheduleGroupTilingTypeName(candidate.getSubgraph(), candidate.getPlan(), stage);
  auto schema = mlir::ascend::kernel::TilingSchema::get(context, StringAttr::get(context, typeName),
                                                        ArrayAttr::get(context, fields.types),
                                                        ArrayAttr::get(context, fields.names));
  if (!schema) {
    return function.emitError("GenTilingData: invalid stage tiling field layout");
  }
  return schema;
}

// Validates that the module is a non-Cube/CV Plan.
LogicalResult validatePlanModule(ModuleOp module) {
  auto symbol = module.getSymName();
  if (!symbol || symbol->empty()) {
    module.emitError("GenTilingData: missing Plan module symbol");
    return failure();
  }
  if (!mlir::afir::isPlanModule(module)) {
    module.emitError("GenTilingData: expected ") << mlir::afir::kModuleKindAttrName << "=plan";
    return failure();
  }
  if (auto cube = module->getAttrOfType<IntegerAttr>(mlir::afir::kCubeTypeAttrName); cube && cube.getInt() != -1) {
    module.emitError("GenTilingData: Cube/CV Plan is not supported, ")
        << mlir::afir::kCubeTypeAttrName << "=" << cube.getInt();
    return failure();
  }
  return success();
}

// Collects template functions from the module. A template is a function carrying
// `#afplan.variant`. 迁移前这里还要判「stage_id 与 template_id 都在且都非负」——现在
// 坐标是一个属性，半个坐标的状态无从表达，非负性也已由该属性的 verifier 钉死，因此
// 只剩「一个 template 都没有」这一种失败。
FailureOr<SmallVector<func::FuncOp>> collectTemplates(ModuleOp module) {
  SmallVector<func::FuncOp> templates;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (afplan::getVariant(function)) {
      templates.push_back(function);
    }
  }
  if (templates.empty()) {
    module.emitError("GenTilingData: Plan has no templates");
    return failure();
  }
  return templates;
}

LogicalResult replaceTemplateTilingArgument(func::FuncOp templateFunc, Type stageType, ModuleOp module) {
  // 「哪个参数是 tiling」问参数自己的声明，不按「最后一个参数的类型长得像」反推（A3）。
  // 按类型猜的旧写法依赖结构体字面叫 "TilingData"，那正是本 pass 马上要改掉的名字。
  unsigned tilingArgument =
      afir::findKernelArgument(templateFunc, afir::KernelArgRole::Tiling).value_or(templateFunc.getNumArguments());
  if (tilingArgument == templateFunc.getNumArguments()) {
    return templateFunc.insertArgument(templateFunc.getNumArguments(), stageType,
                                       DictionaryAttr::get(module.getContext()), templateFunc.getLoc());
  }
  SmallVector<Type> inputTypes(templateFunc.getFunctionType().getInputs());
  inputTypes[tilingArgument] = stageType;
  templateFunc.setType(FunctionType::get(module.getContext(), inputTypes, templateFunc.getFunctionType().getResults()));
  templateFunc.getArgument(tilingArgument).setType(stageType);
  return success();
}

// Groups templates by `#afplan.variant` stage and builds one schedule-group TilingData
// per stage. Replaces the generic signature TilingData argument with the stage type.
// Also collects the stage types and names for the outer Plan schema.
LogicalResult buildStageSchemasAndAppendArgs(ArrayRef<func::FuncOp> templates, ModuleOp module,
                                             afplan::CandidateAttr candidate, SmallVector<Attribute> &stageTypes,
                                             SmallVector<Attribute> &stageNames) {
  int64_t currentStage = -1;
  for (func::FuncOp templateFunc : templates) {
    int64_t stage = afplan::getVariant(templateFunc).getStage();
    // Only the Type is needed below; the stage schema was already validated
    // when it was built (or last read back) so it is not re-walked here.
    Type stageType;
    if (stage == currentStage) {
      stageType = cast<TypeAttr>(stageTypes.back()).getValue();
    } else {
      SmallVector<func::FuncOp> stageTemplates;
      for (func::FuncOp candidate : templates) {
        afplan::VariantAttr candidateVariant = afplan::getVariant(candidate);
        if (candidateVariant && candidateVariant.getStage() == stage) {
          stageTemplates.push_back(candidate);
        }
      }
      auto stageSchema = buildStageSchema(stageTemplates, candidate, stage);
      if (failed(stageSchema)) {
        return failure();
      }
      stageType = *stageSchema;
      stageTypes.push_back(TypeAttr::get(stageType));
      stageNames.push_back(StringAttr::get(
          module.getContext(),
          mlir::common::MakeScheduleGroupTilingFieldName(candidate.getSubgraph(), candidate.getPlan(), stage)));
      currentStage = stage;
    }
    if (failed(replaceTemplateTilingArgument(templateFunc, stageType, module))) {
      templateFunc.emitError("GenTilingData: failed to materialize stage tiling argument");
      return failure();
    }
  }
  return success();
}

struct GenTilingDataPass : PassWrapper<GenTilingDataPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenTilingDataPass)

  StringRef getArgument() const final {
    return "afir-gen-tiling-data";
  }
  StringRef getDescription() const final {
    return "Generate the TilingData schema for one Plan";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<afplan::AFPlanDialect, emitasc::EmitAscDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(validatePlanModule(module))) {
      return signalPassFailure();
    }

    auto templatesResult = collectTemplates(module);
    if (failed(templatesResult)) {
      return signalPassFailure();
    }
    SmallVector<func::FuncOp> templates = std::move(*templatesResult);

    llvm::sort(templates, [](func::FuncOp lhs, func::FuncOp rhs) {
      afplan::VariantAttr lhsVariant = afplan::getVariant(lhs);
      afplan::VariantAttr rhsVariant = afplan::getVariant(rhs);
      return std::make_pair(lhsVariant.getStage(), lhsVariant.getTemplateId()) <
             std::make_pair(rhsVariant.getStage(), rhsVariant.getTemplateId());
    });

    afplan::CandidateAttr candidate = afplan::getCandidate(module);
    if (!candidate) {
      module.emitError("GenTilingData: missing ") << afplan::kCandidateAttrName;
      return signalPassFailure();
    }

    SmallVector<Attribute> stageTypes;
    SmallVector<Attribute> stageNames;
    if (failed(buildStageSchemasAndAppendArgs(templates, module, candidate, stageTypes, stageNames))) {
      return signalPassFailure();
    }

    // 这张 plan 级 schema 是**载体**，不是一个会被发射出来的 C++ 结构体：它只把本 plan 的
    // 各 stage 结构体连同它们在最外层 TilingData 里的成员名带给下游
    // （`GenKernelEntry` 按它铺 `_main` 的 tiling 形参，`GenAggregateTilingData` 把它的
    // 字段原样摊进 `AutofuseTilingData`）。ATT 的最外层布局里没有 plan 这一层
    // （`codegen_tiling_data.cpp` 的 `ProcessMultiGroup` 逐 group 直接挂在根上），所以它
    // 不能进 `emitTilingDataHeader` 的递归——只要它不是根的某个字段就不会进。
    std::string typeName = mlir::afir::makeCxxIdentifier(*module.getSymName()) + "TilingData";
    auto schema = mlir::ascend::kernel::TilingSchema::get(
        module.getContext(), StringAttr::get(module.getContext(), typeName),
        ArrayAttr::get(module.getContext(), stageTypes), ArrayAttr::get(module.getContext(), stageNames));
    if (!schema) {
      module.emitError("GenTilingData: invalid plan tiling field layout");
      return signalPassFailure();
    }
    module->setAttr(mlir::ascend::kernel::kAscendCTilingSchemaAttrName, TypeAttr::get(schema));
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::ascend::kernel::createGenTilingDataPass() {
  return std::make_unique<GenTilingDataPass>();
}
void mlir::ascend::kernel::registerGenTilingDataPass() {
  PassRegistration<GenTilingDataPass>();
}
