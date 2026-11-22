/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_AXES_SOLVE_TOPOLOGY_H_
#define ATT_PROGRAM_AXES_SOLVE_TOPOLOGY_H_
#include "axes_program_evaluation.h"

namespace att {
struct AxesVariableTopology {
  int64_t initial_value = -1;
  int64_t align = 1;
  int64_t prompt_align = 1;
  int64_t data_type_size = 4;
  bool mc_related = false;
  size_t order = 0;
  std::vector<uint32_t> constraints;
};

enum class AxesConstraintResource : uint32_t { kFixed, kRuntimeUb };

struct AxesConstraintTopology {
  uint32_t type;
  int64_t resource;
  AxesConstraintResource resource_source = AxesConstraintResource::kFixed;
};

struct AxesSolveTopology {
  std::vector<AxesVariableTopology> variables;
  std::vector<AxesConstraintTopology> constraints;
  std::vector<uint32_t> pure_mc;
  std::vector<uint32_t> local_buffer;
  std::vector<uint32_t> ordered_to_canonical;
};

inline bool VerifyAxesSolveTopology(const AxesSolveTopology &topology) {
  constexpr uint32_t kLocalBufferConstraint = 0;
  constexpr uint32_t kMaxConstraintType = 2;  // ConstraintType::MC_MIXED in the shared solver.
  const auto variables = topology.variables.size();
  const auto constraints = topology.constraints.size();
  if (variables > kTilingExpressionMaxItems || constraints > kTilingExpressionMaxItems ||
      topology.pure_mc.size() > variables || topology.local_buffer.size() > variables ||
      topology.ordered_to_canonical.size() != topology.local_buffer.size()) {
    return false;
  }
  for (const auto &constraint : topology.constraints) {
    if (constraint.type > kMaxConstraintType ||
        (constraint.resource_source != AxesConstraintResource::kFixed &&
         (constraint.resource_source != AxesConstraintResource::kRuntimeUb ||
          constraint.type != kLocalBufferConstraint || constraint.resource != 0))) {
      return false;
    }
  }
  for (const auto &variable : topology.variables) {
    if (variable.align <= 0 || variable.prompt_align <= 0 || variable.data_type_size <= 0 ||
        variable.order > UINT32_MAX || variable.constraints.size() > constraints) {
      return false;
    }
    for (const auto index : variable.constraints) {
      if (index >= constraints) {
        return false;
      }
    }
  }
  const auto mark = [](const std::vector<uint32_t> &indices, std::vector<bool> &seen) {
    for (const auto index : indices) {
      if (index >= seen.size() || seen[index]) {
        return false;
      }
      seen[index] = true;
    }
    return true;
  };
  std::vector<bool> grouped(variables);
  std::vector<bool> ordered(topology.local_buffer.size());
  return mark(topology.pure_mc, grouped) && mark(topology.local_buffer, grouped) &&
         mark(topology.ordered_to_canonical, ordered);
}

// Creates a fresh pointer graph for each solve from index-only topology.
template <class Input>
class AxesSolveTopologyStorage final {
  using Tiling = std::remove_pointer_t<std::remove_reference_t<decltype(std::declval<Input>().tiling_vars[0])>>;
  using Shape = std::remove_pointer_t<std::remove_reference_t<decltype(std::declval<Input>().input_vars[0])>>;
  using Constraint = std::remove_pointer_t<std::remove_reference_t<decltype(std::declval<Input>().all_cons[0])>>;

 public:
  static TilingExpressionStatus Build(const AxesSolveTopology &topology, const std::vector<int64_t> &shapes,
                                      const Input &options, std::unique_ptr<AxesSolveTopologyStorage> *output) {
    using Status = TilingExpressionStatus;
    if (output == nullptr) {
      return Status::kInvalidArgument;
    }
    if (shapes.size() > kTilingExpressionMaxItems || !VerifyAxesSolveTopology(topology)) {
      return Status::kInvalidProgram;
    }

    auto storage = std::make_unique<AxesSolveTopologyStorage>(AxesSolveTopologyStorage::ConstructionKey{});
    storage->input_ = options;
    storage->variables_.resize(topology.variables.size());
    storage->constraints_.resize(topology.constraints.size());
    storage->shapes_.resize(shapes.size());
    storage->related_.resize(topology.variables.size());
    storage->shape_ptrs_.reserve(shapes.size());
    storage->tiling_ptrs_.reserve(topology.variables.size());
    storage->constraint_ptrs_.reserve(topology.constraints.size());
    storage->ordered_local_.reserve(topology.ordered_to_canonical.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
      storage->shapes_[i].value = shapes[i];
      storage->shape_ptrs_.push_back(&storage->shapes_[i]);
    }
    for (size_t i = 0; i < topology.constraints.size(); ++i) {
      const auto &source = topology.constraints[i];
      auto &constraint = storage->constraints_[i];
      constraint.type = static_cast<decltype(constraint.type)>(source.type);
      constraint.rel_hw_spec =
          source.resource_source == AxesConstraintResource::kRuntimeUb ? options.ub_size : source.resource;
      storage->constraint_ptrs_.push_back(&constraint);
    }
    for (size_t i = 0; i < topology.variables.size(); ++i) {
      const auto &source = topology.variables[i];
      auto &variable = storage->variables_[i];
      variable.value = source.initial_value;
      variable.align = source.align;
      variable.prompt_align = source.prompt_align;
      variable.data_type_size = source.data_type_size;
      variable.mc_related = source.mc_related;
      variable.order = source.order;
      auto &related = storage->related_[i];
      related.reserve(source.constraints.size());
      for (const auto index : source.constraints) {
        related.push_back(&storage->constraints_[index]);
      }
      variable.rel_cons = related.data();
      variable.rel_cons_size = static_cast<uint32_t>(related.size());
      storage->tiling_ptrs_.push_back(&variable);
    }
    storage->BindGroup(topology.pure_mc, storage->mc_);
    storage->BindGroup(topology.local_buffer, storage->local_);
    for (const auto index : topology.ordered_to_canonical) {
      storage->ordered_local_.push_back(storage->local_[index]);
    }
    storage->ordered_to_canonical_ = topology.ordered_to_canonical;
    storage->PublishPointers();
    *output = std::move(storage);
    return Status::kOk;
  }
  Input &GetInput() {
    return input_;
  }

 private:
  struct ConstructionKey {
    explicit ConstructionKey() = default;
  };

 public:
  explicit AxesSolveTopologyStorage(ConstructionKey) {}

 private:
  void BindGroup(const std::vector<uint32_t> &indices, std::vector<Tiling *> &group) {
    group.reserve(indices.size());
    for (const auto index : indices) {
      group.push_back(&variables_[index]);
    }
  }
  void PublishPointers() {
    input_.input_vars_size = static_cast<uint32_t>(shape_ptrs_.size());
    input_.tiling_vars_size = static_cast<uint32_t>(tiling_ptrs_.size());
    input_.pure_mc_vars_size = static_cast<uint32_t>(mc_.size());
    input_.local_buffer_vars_size = static_cast<uint32_t>(local_.size());
    input_.all_cons_size = static_cast<uint32_t>(constraint_ptrs_.size());
    // The native core forms begin/end pointer pairs even for empty groups.
    if (mc_.empty()) {
      mc_.push_back(nullptr);
    }
    if (local_.empty()) {
      local_.push_back(nullptr);
    }
    input_.input_vars = shape_ptrs_.data();
    input_.tiling_vars = tiling_ptrs_.data();
    input_.pure_mc_vars = mc_.data();
    input_.local_buffer_vars = local_.data();
    input_.all_cons = constraint_ptrs_.data();
    input_.ordered_local_buffer_vars = ordered_local_.data();
    input_.ordered_to_canonical = ordered_to_canonical_.data();
  }
  Input input_;
  std::vector<Shape> shapes_;
  std::vector<Tiling> variables_;
  std::vector<Constraint> constraints_;
  std::vector<Shape *> shape_ptrs_;
  std::vector<Tiling *> tiling_ptrs_, mc_, local_, ordered_local_;
  std::vector<Constraint *> constraint_ptrs_;
  std::vector<std::vector<Constraint *>> related_;
  std::vector<uint32_t> ordered_to_canonical_;
};
}  // namespace att
#endif
