/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/AfirBuilder/WorkspaceViews.h"
#include <map>
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::afir::bridge {
namespace {
// Scheduled repeats cover the full GM domain. Collapse only proven dense GM
// strides, not UB vectorized_strides. Cancel stage-local tile symbols before
// using the view in another stage.
FailureOr<std::string> denseExtent(const TensorDescDTO &view, const TemplateInfo &tmpl) {
  if (view.repeats.empty() || view.repeats.size() != view.strides.size()) {
    return failure();
  }
  std::string extent = "1";
  for (size_t i = view.repeats.size(); i-- > 0;) {
    if (ProveExpressionsEqual(view.repeats[i], "1")) {
      continue;
    }
    if (!ProveExpressionsEqual(view.strides[i], extent)) {
      return failure();
    }
    auto product = MultiplyExpressions(extent, view.repeats[i]);
    if (!product) {
      return failure();
    }
    extent = product.value();
  }
  llvm::StringSet<> originalSymbols;
  for (const auto &axis : tmpl.axes) {
    if (axis.type != "original") {
      continue;
    }
    auto size = Expression::Parse(axis.size);
    if (!size) {
      return failure();
    }
    auto symbols = detail::ExpressionAccess::GetFreeSymbols(size.value());
    if (!symbols) {
      return failure();
    }
    for (const auto &symbol : symbols.value()) {
      originalSymbols.insert(symbol);
    }
  }
  auto expression = Expression::Parse(extent);
  if (!expression) {
    return failure();
  }
  auto symbols = detail::ExpressionAccess::GetFreeSymbols(expression.value());
  if (!symbols) {
    return failure();
  }
  for (const auto &symbol : symbols.value()) {
    if (!originalSymbols.contains(symbol)) {
      return failure();
    }
  }
  return extent;
}

// Resolve only original shape symbols, in their owning stage. Relation edges
// must point to an earlier stage, so recursive substitution cannot form a cycle
// or accidentally equate a stage-local tiling variable with a global extent.
FailureOr<llvm::StringSet<>> collectOriginalStageSymbols(const PlanInfo &plan, int64_t stage) {
  llvm::StringSet<> originalSymbols;
  bool foundStage = false;
  for (const auto &tmpl : plan.templates) {
    if (tmpl.stage_id != stage) {
      continue;
    }
    foundStage = true;
    for (const auto &axis : tmpl.axes) {
      if (axis.type != "original") {
        continue;
      }
      auto size = Expression::Parse(axis.size);
      if (!size) {
        return failure();
      }
      auto symbols = detail::ExpressionAccess::GetFreeSymbols(size.value());
      if (!symbols) {
        return failure();
      }
      for (const auto &symbol : symbols.value()) {
        originalSymbols.insert(symbol);
      }
    }
  }
  if (!foundStage) {
    return failure();
  }
  return std::move(originalSymbols);
}

FailureOr<std::string> normalizeWorkspaceExtent(const PlanInfo &plan, int64_t stage, StringRef extent) {
  auto originalSymbols = collectOriginalStageSymbols(plan, stage);
  if (failed(originalSymbols)) {
    return failure();
  }
  auto expression = Expression::Parse(extent.str());
  if (!expression) {
    return failure();
  }
  auto symbols = detail::ExpressionAccess::GetFreeSymbols(expression.value());
  if (!symbols) {
    return failure();
  }
  std::map<std::string, std::string> replacements;
  for (const auto &symbol : symbols.value()) {
    if (!originalSymbols->contains(symbol)) {
      return failure();
    }
    const VarRelationInfo *binding = nullptr;
    for (const auto &relation : plan.var_relations) {
      if (relation.dst_stage == stage && relation.dst_var == symbol) {
        if (binding || relation.src_stage < 0 || relation.src_stage >= stage) {
          return failure();
        }
        binding = &relation;
      }
    }
    if (binding) {
      auto value = normalizeWorkspaceExtent(plan, binding->src_stage, binding->src_expr);
      if (failed(value)) {
        return failure();
      }
      replacements.emplace(symbol, *value);
    }
  }
  auto normalized = detail::ExpressionAccess::Substitute(expression.value(), replacements);
  if (!normalized) {
    return failure();
  }
  return normalized.value().Serialize();
}

const NodeInfo *writerFor(const TemplateInfo &tmpl, const NodeInfo &workspace) {
  if (workspace.input_names.size() != 1) {
    return nullptr;
  }
  const NodeInfo *writer = nullptr;
  for (const auto &node : tmpl.nodes) {
    if (node.type != "Store" || !llvm::is_contained(node.output_names, workspace.input_names.front())) {
      continue;
    }
    if (writer) {
      return nullptr;
    }
    writer = &node;
  }
  return writer;
}
}  // namespace

static LogicalResult mergeWorkspaceWriterExtent(const PlanInfo &plan, const TemplateInfo &reader,
                                                const TemplateInfo &candidate, const NodeInfo &sink,
                                                const NodeInfo &source, Location loc,
                                                std::optional<std::string> &extent) {
  const int64_t id = source.output.buffer.bufferId;
  auto reject = [&](StringRef reason) -> LogicalResult {
    return emitError(loc) << "InvalidWorkspace: workspace " << id << " in stage " << reader.stage_id
                          << " cannot recover its source view: " << reason;
  };
  const NodeInfo *writer = writerFor(candidate, sink);
  if (!writer || writer->output.buffer.space != Space::GM || sink.output.buffer.space != Space::GM) {
    return reject("writer is not a unique GM Store");
  }
  if (writer->output.tensor.dtype != source.output.tensor.dtype) {
    return reject("writer/reader dtype mismatch");
  }
  auto current = denseExtent(writer->output.tensor, candidate);
  if (failed(current)) {
    return reject("writer GM layout is not provably dense and stage independent");
  }
  current = normalizeWorkspaceExtent(plan, candidate.stage_id, *current);
  if (failed(current)) {
    return reject("writer extent has invalid cross-stage shape bindings");
  }
  if (extent && !ProveExpressionsEqual(*extent, *current)) {
    return reject("writer templates disagree");
  }
  extent = *current;
  return success();
}

static LogicalResult findWorkspaceWriterExtent(const PlanInfo &plan, const TemplateInfo &reader, const NodeInfo &source,
                                               Location loc, std::optional<std::string> &extent) {
  const int64_t id = source.output.buffer.bufferId;
  auto reject = [&](StringRef reason) -> LogicalResult {
    return emitError(loc) << "InvalidWorkspace: workspace " << id << " in stage " << reader.stage_id
                          << " cannot recover its source view: " << reason;
  };
  std::optional<int64_t> writerStage;
  for (const auto &candidate : plan.templates) {
    for (const auto &sink : candidate.nodes) {
      if (sink.type != "Workspace" || sink.output.buffer.bufferId != id || sink.input_names.empty()) {
        continue;
      }
      if (candidate.stage_id < 0 || candidate.stage_id >= reader.stage_id) {
        return reject("writer is not in an earlier stage");
      }
      if (writerStage && *writerStage != candidate.stage_id) {
        return reject("multiple writer stages");
      }
      writerStage = candidate.stage_id;
      if (failed(mergeWorkspaceWriterExtent(plan, reader, candidate, sink, source, loc, extent))) {
        return failure();
      }
    }
  }
  if (!extent) {
    return reject("no matching writer");
  }
  for (const auto &candidate : plan.templates) {
    if (candidate.stage_id != *writerStage) {
      continue;
    }
    const auto count = llvm::count_if(candidate.nodes, [&](const NodeInfo &node) {
      return node.type == "Workspace" && node.output.buffer.bufferId == id && !node.input_names.empty();
    });
    if (count != 1) {
      return reject("each writer template must write this workspace exactly once");
    }
  }
  return success();
}

static FailureOr<std::string> resolveWorkspaceReaderExtent(const PlanInfo &plan, const TemplateInfo &reader,
                                                           const NodeInfo &source, StringRef extent, Location loc) {
  const int64_t id = source.output.buffer.bufferId;
  auto reject = [&](StringRef reason) -> FailureOr<std::string> {
    emitError(loc) << "InvalidWorkspace: workspace " << id << " in stage " << reader.stage_id
                   << " cannot recover its source view: " << reason;
    return failure();
  };
  bool hasConsumer = false;
  std::string readerExtent;
  for (const auto &consumer : reader.nodes) {
    for (const auto &input : consumer.input_names) {
      if (!llvm::is_contained(source.output_names, input)) {
        continue;
      }
      hasConsumer = true;
      if (consumer.type != "Load") {
        return reject("source consumer is not a Load");
      }
      auto readExtent = denseExtent(consumer.output.tensor, reader);
      auto normalizedReadExtent = succeeded(readExtent) ? normalizeWorkspaceExtent(plan, reader.stage_id, *readExtent)
                                                        : FailureOr<std::string>(failure());
      if (failed(normalizedReadExtent) || !ProveExpressionsEqual(extent.str(), *normalizedReadExtent) ||
          consumer.output.tensor.dtype != source.output.tensor.dtype) {
        return reject("read/write GM views disagree");
      }
      readerExtent = *readExtent;
    }
  }
  if (!hasConsumer) {
    return reject("no Load consumer");
  }
  return readerExtent;
}

LogicalResult resolveWorkspaceSourceViews(PlanInfo &plan, Location loc) {
  for (auto &reader : plan.templates) {
    if (!reader.buffer_form) {
      continue;
    }
    for (auto &source : reader.nodes) {
      if (source.type != "Workspace" || !source.input_names.empty() || !source.output.tensor.shape.empty()) {
        continue;
      }
      const int64_t id = source.output.buffer.bufferId;
      auto reject = [&](StringRef reason) -> LogicalResult {
        return emitError(loc) << "InvalidWorkspace: workspace " << id << " in stage " << reader.stage_id
                              << " cannot recover its source view: " << reason;
      };
      if (id < 0 || reader.stage_id < 0 || source.output.buffer.space != Space::GM) {
        return reject("missing workspace identity or stage");
      }
      std::optional<std::string> extent;
      if (failed(findWorkspaceWriterExtent(plan, reader, source, loc, extent))) {
        return failure();
      }
      auto readExtent = resolveWorkspaceReaderExtent(plan, reader, source, *extent, loc);
      if (failed(readExtent)) {
        return failure();
      }
      const std::string &readerExtent = *readExtent;
      // The shared object is a flat GM storage view. Each Load/Store keeps its
      // own scheduled descriptor and offset calculation. No writer-local axis
      // or tiling variable is imported into the reader stage.
      TensorValueDTO resolved = source.output;
      resolved.tensor = TensorDescDTO{};
      resolved.tensor.dtype = source.output.tensor.dtype;
      int64_t staticExtent = ShapedType::kDynamic;
      int64_t parsedExtent;
      if (!StringRef(readerExtent).getAsInteger(10, parsedExtent)) {
        if (parsedExtent < 0) {
          return reject("negative extent");
        }
        staticExtent = parsedExtent;
      }
      resolved.tensor.shape = {staticExtent};
      resolved.tensor.shape_exprs = {readerExtent};
      Type elementType = parseDtype(loc.getContext(), resolved.tensor.dtype);
      if (!elementType) {
        return reject("unsupported element type");
      }
      auto bytes = MultiplyExpressions(readerExtent, std::to_string(elementType.getIntOrFloatBitWidth() / 8));
      if (!bytes) {
        return reject("cannot compute reader view byte size");
      }
      // This is the GM view size. The actual allocation remains in workspace_plan.
      resolved.buffer.sizeBytesExpr = bytes.value();
      source.output = resolved;
      source.outputs = {resolved};
      for (auto &consumer : reader.nodes) {
        for (size_t i = 0; i < consumer.input_names.size(); ++i) {
          if (llvm::is_contained(source.output_names, consumer.input_names[i])) {
            if (i >= consumer.inputs.size()) {
              return reject("missing consumer descriptor");
            }
            consumer.inputs[i] = resolved;
          }
        }
      }
    }
  }
  return success();
}
}  // namespace mlir::afir::bridge
