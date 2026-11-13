/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "artifact/TilingSource/TilingFuncEmitter.h"
#include "common/tiling_schema/resource_budget.h"

#include <map>
#include <set>
#include <sstream>
#include <string>

#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Diagnostics.h"

// Legacy behavior baseline (this file is not a line-for-line copy):
//   autofuse/codegen/codegen_tiling.cpp
//     TilingFuncDefForInductor, GenTilingFuncForInductor, GenPlainInductorTilingTail,
//     GenInductorShapeDim, TilingSetShapeDim, GenGetWorkspaceSizeFunc, GenImplGraphWorkspaceSize
//   autofuse/codegen/codegen_tiling_pgo_search.cpp
//     GenGetResLimitStru
//   autofuse/codegen/codegen.cpp
//     split tiling source assembly helpers
// Output protocol compatibility reference:
//   autofuse/compiler/python/compile_adapter.py
//     AUTOFUSE_SPLIT_FILE_BEGIN/AUTOFUSE_SPLIT_FILE_END parsing
// Reference revision: a44d791da96154db596657cab065e3c35b7e2de4
// Reviewed at: 2026-09-04T07:58:26Z
// Scope: preserve the non-Cube Inductor host tiling ABI and behavior while rebuilding it from canonical MLIR metadata.

namespace mlir::bridge {
namespace {

constexpr char kTilingHead[] = "TilingHead";
constexpr char kTilingStateHeader[] = "TilingStateHeader";
constexpr char kTilingLogHeader[] = "TilingLogHeader";
constexpr char kTilingPgoHeader[] = "TilingPgoHeader";
constexpr char kTilingSolverHeader[] = "TilingSolverHeader";
constexpr char kTilingApiHeader[] = "TilingApiHeader";
constexpr char kTilingBaseHeader[] = "TilingBaseHeader";
constexpr char kTilingEntryHeader[] = "TilingEntryHeader";
constexpr char kTilingTailHeader[] = "TilingTailHeader";
constexpr char kCubeKernelTilingWrapper[] = "CubeKernelTilingWrapperHpp";

std::string StripTilingHeadGuard(const std::string &content) {
  std::stringstream input(content);
  std::stringstream output;
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("__AUTOFUSE_TILING_FUNC_COMMON_H__") == std::string::npos) {
      output << line << '\n';
    }
  }
  return output.str();
}

bool IsSplitHeader(const std::string &key) {
  return key == kTilingStateHeader || key == kTilingLogHeader || key == kTilingPgoHeader ||
         key == kTilingSolverHeader || key == kTilingApiHeader || key == kTilingBaseHeader ||
         key == kTilingEntryHeader || key == kTilingTailHeader;
}

std::string RemoveSplitIncludes(const std::string &content) {
  static const std::set<std::string> includes = {
      "#include \"autofuse_tiling_func_common.h\"", "#include \"autofuse_tiling_func_state.h\"",
      "#include \"autofuse_tiling_func_log.h\"",    "#include \"autofuse_tiling_func_pgo.h\"",
      "#include \"autofuse_tiling_func_base.h\"",   "#include \"autofuse_tiling_func_solver.h\"",
      "#include \"autofuse_tiling_func_api.h\"",    "#include \"autofuse_tiling_func_entry.h\"",
      "#include \"autofuse_tiling_func_tail.h\"",   "#include \"cube_kernel_tiling_wrapper.h\""};
  std::stringstream input(content);
  std::stringstream output;
  std::string line;
  bool leadingIncludes = true;
  while (std::getline(input, line)) {
    if (leadingIncludes && !line.empty() && line.rfind("#include ", 0) != 0) {
      leadingIncludes = false;
    }
    if (!leadingIncludes || includes.count(line) == 0) {
      output << line << '\n';
    }
  }
  return output.str();
}

void AppendSplitSource(const std::string &key, const std::string &content, std::string &result) {
  result += "// AUTOFUSE_SPLIT_FILE_BEGIN: " + key + "\n";
  result += content;
  if (!result.empty() && result.back() != '\n') {
    result.push_back('\n');
  }
  result += "// AUTOFUSE_SPLIT_FILE_END: " + key + "\n";
}

bool HasNewHeaders(const std::map<std::string, std::string> &sources) {
  for (const auto *key :
       {kTilingStateHeader, kTilingLogHeader, kTilingPgoHeader, kTilingSolverHeader, kTilingApiHeader}) {
    if (sources.find(key) != sources.end()) {
      return true;
    }
  }
  return false;
}

bool IsSkippedSource(const std::string &key) {
  return key == kTilingHead || IsSplitHeader(key) || key == kCubeKernelTilingWrapper ||
         key.find("TilingData") != std::string::npos;
}

std::string AssembleSources(const std::map<std::string, std::string> &sources) {
  auto head = sources.find(kTilingHead);
  if (head == sources.end()) {
    return {};
  }
  std::string result;
  const bool newFormat = HasNewHeaders(sources);
  if (!newFormat) {
    std::string header = StripTilingHeadGuard(head->second);
    auto wrapper = sources.find(kCubeKernelTilingWrapper);
    if (wrapper != sources.end()) {
      if (!header.empty() && header.back() != '\n') {
        header.push_back('\n');
      }
      header += wrapper->second;
    }
    AppendSplitSource(kTilingHead, header, result);
  } else {
    for (const auto *key : {kTilingStateHeader, kTilingLogHeader, kTilingPgoHeader, kTilingSolverHeader,
                            kTilingApiHeader, kCubeKernelTilingWrapper}) {
      auto source = sources.find(key);
      if (source != sources.end()) {
        AppendSplitSource(key, source->second, result);
      }
    }
  }
  for (const auto &[key, value] : sources) {
    if (IsSkippedSource(key)) {
      continue;
    }
    AppendSplitSource(key, newFormat ? value : RemoveSplitIncludes(value), result);
  }
  return result;
}

FailureOr<std::string> EmitWorkspaceExpression(const afir::detail::ExpressionNode &node, const TilingStageSpec &stage);

FailureOr<std::string> EmitWorkspaceArguments(const afir::detail::ExpressionNode &node, const TilingStageSpec &stage,
                                              StringRef separator) {
  std::string result;
  for (const afir::detail::ExpressionNode &argument : node.arguments) {
    auto emitted = EmitWorkspaceExpression(argument, stage);
    if (failed(emitted)) {
      return failure();
    }
    if (!result.empty()) {
      result += separator;
    }
    result += *emitted;
  }
  return result;
}

FailureOr<std::string> EmitWorkspaceCall(const afir::detail::ExpressionNode &node, const TilingStageSpec &stage,
                                         StringRef callee) {
  auto arguments = EmitWorkspaceArguments(node, stage, ", ");
  if (failed(arguments)) {
    return failure();
  }
  return (callee + "(" + *arguments + ")").str();
}

FailureOr<std::string> EmitFoldedWorkspaceCall(const afir::detail::ExpressionNode &node, const TilingStageSpec &stage,
                                               StringRef callee) {
  if (node.arguments.empty()) {
    return failure();
  }
  auto result = EmitWorkspaceExpression(node.arguments.front(), stage);
  if (failed(result)) {
    return failure();
  }
  for (const afir::detail::ExpressionNode &argument : llvm::drop_begin(node.arguments)) {
    auto emitted = EmitWorkspaceExpression(argument, stage);
    if (failed(emitted)) {
      return failure();
    }
    *result = (callee + "(" + *result + ", " + *emitted + ")").str();
  }
  return result;
}

// workspace 表达式中的函数模板和宏需要不同的限定方式：
// - Max / Min / Abs / Ceiling / Floor / Mod 是 ATT 定义在 optiling 中的函数模板，
//   调用时须加 optiling::。CalculateWorkspaceSize 和 GetWorkspaceSize 保持在全局作用域，
//   与 ATT 的全局前置声明一致，不能通过移动函数作用域省略限定。
// - Pow / Rational / Log / Expect* / Logic* 是宏，不能加命名空间限定，
//   否则 optiling:: 会被拼接到宏展开体之前。
// 仅含常量的表达式不经过算子生成，例如单核 RCommon 的 workspace 大小为 0，
// 因此测试还需覆盖非平凡 workspace 表达式。
constexpr llvm::StringLiteral kOptilingScope = "optiling::";

FailureOr<std::string> EmitWorkspaceFunction(const afir::detail::ExpressionNode &node, const TilingStageSpec &stage) {
  using afir::detail::NodeKind;
  switch (node.kind) {
    case NodeKind::Pow:
      return EmitWorkspaceCall(node, stage, "Pow");
    case NodeKind::Max:
      return EmitFoldedWorkspaceCall(node, stage, kOptilingScope.str() + "Max");
    case NodeKind::Min:
      return EmitFoldedWorkspaceCall(node, stage, kOptilingScope.str() + "Min");
    case NodeKind::Mod:
      return EmitWorkspaceCall(node, stage, kOptilingScope.str() + "Mod");
    case NodeKind::Log:
      return EmitWorkspaceCall(node, stage, "Log");
    case NodeKind::Ceiling:
      return EmitWorkspaceCall(node, stage, kOptilingScope.str() + "Ceiling");
    case NodeKind::Floor:
      return EmitWorkspaceCall(node, stage, kOptilingScope.str() + "Floor");
    case NodeKind::Abs:
      return EmitWorkspaceCall(node, stage, kOptilingScope.str() + "Abs");
    case NodeKind::Rational:
      return EmitWorkspaceCall(node, stage, "Rational");
    case NodeKind::Equal:
      return EmitWorkspaceCall(node, stage, "ExpectEq");
    case NodeKind::NotEqual:
      return EmitWorkspaceCall(node, stage, "ExpectNe");
    case NodeKind::LessEqual:
      return EmitWorkspaceCall(node, stage, "ExpectLe");
    case NodeKind::LessThan:
      return EmitWorkspaceCall(node, stage, "ExpectLt");
    case NodeKind::LogicAnd:
      return EmitFoldedWorkspaceCall(node, stage, "LogicAnd");
    case NodeKind::LogicOr:
      return EmitFoldedWorkspaceCall(node, stage, "LogicOr");
    default:
      return failure();
  }
}

FailureOr<std::string> EmitWorkspaceExpression(const afir::detail::ExpressionNode &node, const TilingStageSpec &stage) {
  using afir::detail::NodeKind;
  switch (node.kind) {
    case NodeKind::Integer:
    case NodeKind::Real:
    case NodeKind::Boolean:
      return node.value;
    case NodeKind::Symbol:
      if (!llvm::is_contained(stage.fields, node.value)) {
        return failure();
      }
      return "static_cast<uint64_t>(t." + stage.tiling_field_name + "." + node.value + ")";
    case NodeKind::Add: {
      auto arguments = EmitWorkspaceArguments(node, stage, " + ");
      if (failed(arguments)) {
        return failure();
      }
      return "(" + *arguments + ")";
    }
    case NodeKind::Mul: {
      auto arguments = EmitWorkspaceArguments(node, stage, " * ");
      if (failed(arguments)) {
        return failure();
      }
      return "(" + *arguments + ")";
    }
    default:
      return EmitWorkspaceFunction(node, stage);
  }
  return failure();
}

LogicalResult AppendStageWorkspace(const TilingStageSpec &stage, std::stringstream &out) {
  for (auto [index, tmpl] : llvm::enumerate(stage.templates)) {
    afir::Result<afir::Expression> expression = afir::Expression::Parse(tmpl.workspace_size.getValue().getValue());
    if (!expression) {
      return failure();
    }
    afir::Result<afir::detail::ExpressionNode> tree = afir::detail::ExpressionAccess::GetTree(expression.value());
    if (!tree) {
      return failure();
    }
    auto workspace = EmitWorkspaceExpression(tree.value(), stage);
    if (failed(workspace)) {
      return failure();
    }
    out << (index == 0 ? "    if (" : "    else if (") << "t." << stage.tiling_field_name
        << ".tiling_key == " << (tmpl.tiling_key >= 0 ? tmpl.tiling_key : tmpl.template_id) << ") {\n"
        << "      const long double stage_ws_size = static_cast<long double>(" << *workspace << ");\n"
        << "      if (!(stage_ws_size >= 0.0L) || stage_ws_size >= kWorkspaceSizeExclusiveLimit) {\n"
        << "        return false;\n"
        << "      }\n"
        << "      const uint64_t stage_ws_size_u64 = static_cast<uint64_t>(stage_ws_size);\n"
        << "      if (ws_size > kMaxWorkspaceSize - stage_ws_size_u64) { return false; }\n"
        << "      ws_size += stage_ws_size_u64;\n"
        << "    }\n";
  }
  return success();
}

FailureOr<std::string> EmitWorkspaceCalculation(const TilingProgramSpec &spec) {
  std::stringstream out;
  out << "bool CalculateWorkspaceSize(const AutofuseTilingData &t, uint64_t &workspace_size) {\n"
      << "  constexpr uint64_t kMaxWorkspaceSize = std::numeric_limits<uint64_t>::max();\n"
      << "  // Exclusive uint64_t bound; converting UINT64_MAX to floating point may round up.\n"
      << "  constexpr long double kWorkspaceSizeExclusiveLimit = 0x1p64L;\n"
      << "  constexpr uint64_t kWorkspaceAlignmentBytes = 512U;  // Legacy workspace alignment.\n"
      << "  constexpr uint64_t kWorkspaceAlignmentPadding = kWorkspaceAlignmentBytes - 1U;\n"
      << "  uint64_t ws_size = 0;\n";
  const bool hasGraphSelectors = spec.plans.size() != 1;
  int64_t currentSubgraph = -1;
  size_t planIndex = 0;
  for (const TilingPlanSpec &plan : spec.plans) {
    if (hasGraphSelectors) {
      if (plan.subgraph_id != currentSubgraph) {
        currentSubgraph = plan.subgraph_id;
        planIndex = 0;
      }
      out << (planIndex++ == 0 ? "  if (" : "  else if (") << "t.graph" << plan.subgraph_id
          << "_tiling_key == " << plan.plan_id << ") {\n";
    }
    for (const TilingStageSpec &stage : plan.stages) {
      if (failed(AppendStageWorkspace(stage, out))) {
        return failure();
      }
    }
    if (hasGraphSelectors) {
      out << "  }\n";
    }
  }
  out << "  if (ws_size > kMaxWorkspaceSize - kWorkspaceAlignmentPadding) { return false; }\n"
      << "  workspace_size = (ws_size + kWorkspaceAlignmentPadding) / kWorkspaceAlignmentBytes\n"
      << "                   * kWorkspaceAlignmentBytes;\n"
      << "  return true;\n}\n";
  return out.str();
}

std::string EmitResLimit(const TilingProgramSpec &spec) {
  std::stringstream out;
  out << "struct ResLimit {\n"
      << "  uint32_t valid_num = 0;\n  uint32_t aiv_num = 0;\n  uint32_t aic_num = 0;\n"
      << "  uint32_t ub_size = 0;\n  uint32_t resv[10];\n};\n"
      << "constexpr ResLimit g_no_limit_res = {1, " << spec.target.getAivNum() << ", 0, " << spec.target.getUbSize()
      << ", {}};\n";
  return out.str();
}

void AppendShapeAssignments(const TilingProgramSpec &spec, const std::string &var, std::stringstream &out) {
  for (const TilingPlanSpec &plan : spec.plans) {
    for (const TilingStageSpec &stage : plan.stages) {
      if (!llvm::is_contained(stage.derived_shape_vars, var) && llvm::is_contained(stage.fields, var)) {
        out << "  tiling->" << stage.tiling_field_name << ".set_" << var << "(" << var << ");\n";
      }
    }
  }
}

FailureOr<std::string> EmitAbi(const TilingProgramSpec &spec) {
  std::stringstream out;
  // Const-tiling source emission is not implemented by the MLIR path yet.
  // Export the legacy probe and opt out so compile_adapter keeps the normal
  // runtime-tiling kernel instead of requesting the unavailable static ABI.
  out << "extern \"C\" bool AutofuseIsStaticShape() {\n  return false;\n}\n"
      << "extern \"C\" size_t GetTilingDataSize() {\n  return sizeof(AutofuseTilingData);\n}\n";
  auto workspaceHelper = EmitWorkspaceCalculation(spec);
  if (failed(workspaceHelper)) {
    return failure();
  }
  out << *workspaceHelper << "bool GetWorkspaceSize(const AutofuseTilingData &t, uint64_t &workspace_size) {\n"
      << "  uint64_t checked_size = 0;\n"
      << "  if (!CalculateWorkspaceSize(t, checked_size)) { return false; }\n"
      << "  workspace_size = checked_size;\n"
      << "  return true;\n}\n";

  out << EmitResLimit(spec);
  out << "extern \"C\" int64_t AutofuseTiling(";
  bool first = true;
  for (const std::string &var : spec.shape_params) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << "uint32_t " << var;
  }
  if (!first) {
    out << ", ";
  }
  out << "AutofuseTilingData *tiling, uint64_t *workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr) {\n"
      << "  if (tiling == nullptr || workspaceSize == nullptr || blockDim == nullptr) { return -1; }\n"
      << "  const ResLimit *limit = (res_limit == nullptr || res_limit->aiv_num == 0) ? &g_no_limit_res : res_limit;\n"
      << "  if (limit->aiv_num == 0 || limit->aiv_num > g_no_limit_res.aiv_num || limit->ub_size <= "
      << common::kTilingReservedUbBytes << " ||\n"
      << "      limit->ub_size > g_no_limit_res.ub_size) { return -1; }\n";
  for (const std::string &var : spec.shape_params) {
    AppendShapeAssignments(spec, var, out);
  }
  out << "  tiling->set_block_dim(limit->aiv_num);\n  tiling->set_ub_size(limit->ub_size - "
      << common::kTilingReservedUbBytes << "U);\n"
      << "  if (!optiling::GetTiling(*tiling, -1, nullptr)) { return -1; }\n"
      << "  *blockDim = tiling->get_block_dim();\n"
      << "  if (!CalculateWorkspaceSize(*tiling, *workspaceSize)) { return -1; }\n"
      << "  return 0;\n}\n";
  return out.str();
}

void AppendTilingFunctionIncludes(const TilingCodeBundle &bundle, std::string &result) {
  result +=
      "// AUTOFUSE_SPLIT_FILE_BEGIN: MlirTilingFunc\n#include <cstddef>\n#include <limits>\n"
      "#include \"autofuse_tiling_data.h\"\n";
  if (HasNewHeaders(bundle.tiling_functions)) {
    result +=
        "#include \"autofuse_tiling_func_api.h\"\n"
        "#include \"autofuse_tiling_func_solver.h\"\n";
  } else {
    result += "#include \"autofuse_tiling_func_common.h\"\n";
  }
}

void AppendGenerateTopnSolutions(std::string &result) {
  result += R"(
#include <map>
#include <string>
#include <vector>
extern "C" int64_t GenerateTopnSolutions(
    const std::vector<std::map<std::string, std::string>> &input_configs, int64_t topn,
    std::vector<AutofuseTilingData> &tiling_datas, std::vector<int64_t> &workspaces,
    std::vector<int64_t> &block_dims, ResLimit *res_limit = nullptr) {
  tiling_datas.clear();
  workspaces.clear();
  block_dims.clear();
  if (topn <= 0) { return -1; }
  // This single-solution adapter supports the default solver configuration only.
  for (const auto &config : input_configs) {
    if (!config.empty()) { return -1; }
  }
  AutofuseTilingData tiling{};
  uint64_t workspace = 0;
  uint32_t block_dim = 0;
  if (AutofuseTiling(&tiling, &workspace, &block_dim, res_limit) != 0 ||
      workspace > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return -1;
  }
  tiling_datas.push_back(tiling);
  workspaces.push_back(static_cast<int64_t>(workspace));
  block_dims.push_back(block_dim);
  return 0;
}
)";
}

}  // namespace

FailureOr<std::string> emitInductorTilingFunc(const TilingProgramSpec &spec, const TilingCodeBundle &bundle) {
  std::string result = AssembleSources(bundle.tiling_functions);
  if (result.empty()) return failure();
  AppendTilingFunctionIncludes(bundle, result);
  auto abi = EmitAbi(spec);
  if (failed(abi)) {
    return failure();
  }
  result += *abi;
  if (spec.shape_params.empty()) {
    // Inductor accepts fewer solutions than requested. Keep ATT's normal
    // modeled winner and reuse its resource/workspace validation unchanged.
    AppendGenerateTopnSolutions(result);
  }
  result += "// AUTOFUSE_SPLIT_FILE_END: MlirTilingFunc\n";
  return result;
}

FailureOr<std::string> assembleTilingSources(const TilingCodeBundle &bundle) {
  std::string result = AssembleSources(bundle.tiling_functions);
  if (result.empty()) {
    return failure();
  }
  return result;
}

}  // namespace mlir::bridge
