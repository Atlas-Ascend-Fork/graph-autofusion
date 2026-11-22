/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_AXES_CALLBACK_BINDINGS_H_
#define ATT_PROGRAM_AXES_CALLBACK_BINDINGS_H_
#include "axes_program_evaluation.h"

namespace att {
constexpr uint32_t kAxesTilingRelatedBit = uint32_t{1} << 31;
struct AxesConstraintBinding {
  std::shared_ptr<const AxesProgramEvaluation> callback;
  std::vector<uint32_t> tiling_indices;
  std::vector<uint32_t> shape_indices;
};
constexpr uint32_t kAxesShapeParentBit = uint32_t{1} << 31;
struct AxesUpperBoundBinding {
  std::shared_ptr<const AxesProgramEvaluation> callback;
  uint32_t target_index = 0;
  std::vector<uint32_t> parent_indices;
};
struct AxesCallbackBindings {
  std::vector<AxesConstraintBinding> constraints;
  std::vector<AxesUpperBoundBinding> upper_bounds;
};
TilingExpressionStatus VerifyAxesCallbackBindings(const AxesCallbackBindings &bindings, uint32_t tiling_count,
                                                  uint32_t shape_count);
// Owns one solve's pointer arrays and callback definitions. The bound input objects
// must remain alive and must not be used after this storage is destroyed.
template <class Input, class Context, class Binding, class Output>
class AxesSolveCallbackStorage final {
  using TilingPtr = std::remove_reference_t<decltype(std::declval<Input>().tiling_vars[0])>;
  using ShapePtr = std::remove_reference_t<decltype(std::declval<Input>().input_vars[0])>;
  struct ConstraintArrays {
    std::vector<TilingPtr> tiling;
    std::vector<ShapePtr> shapes;
  };

 public:
  AxesSolveCallbackStorage() = default;
  AxesSolveCallbackStorage(const AxesSolveCallbackStorage &) = delete;
  AxesSolveCallbackStorage &operator=(const AxesSolveCallbackStorage &) = delete;

  TilingExpressionStatus Bind(const AxesCallbackBindings &bindings, Input &input) {
    using Status = TilingExpressionStatus;
    if (bound_ || bindings.constraints.size() != input.all_cons_size ||
        (input.all_cons_size != 0 && input.all_cons == nullptr) ||
        (input.tiling_vars_size != 0 && input.tiling_vars == nullptr) ||
        (input.input_vars_size != 0 && input.input_vars == nullptr)) {
      return Status::kInvalidArgument;
    }
    const auto status = VerifyAxesCallbackBindings(bindings, input.tiling_vars_size, input.input_vars_size);
    if (status != Status::kOk) {
      return status;
    }
    std::vector<ConstraintArrays> constraints(bindings.constraints.size());
    std::vector<std::vector<ShapePtr>> parents(bindings.upper_bounds.size());
    std::vector<std::shared_ptr<const AxesProgramEvaluation>> definitions;
    definitions.reserve(bindings.constraints.size() + bindings.upper_bounds.size());
    for (size_t i = 0; i < bindings.constraints.size(); ++i) {
      if (input.all_cons[i] == nullptr ||
          !Resolve(bindings.constraints[i].tiling_indices, input.tiling_vars, constraints[i].tiling) ||
          !ResolveMixed(bindings.constraints[i].shape_indices, kAxesTilingRelatedBit, input.input_vars,
                        input.tiling_vars, constraints[i].shapes)) {
        return Status::kInvalidArgument;
      }
      definitions.push_back(bindings.constraints[i].callback);
    }
    for (size_t i = 0; i < bindings.upper_bounds.size(); ++i) {
      const auto &bound = bindings.upper_bounds[i];
      if (input.tiling_vars[bound.target_index] == nullptr) {
        return Status::kInvalidArgument;
      }
      if (!ResolveMixed(bound.parent_indices, kAxesShapeParentBit, input.tiling_vars, input.input_vars, parents[i])) {
        return Status::kInvalidArgument;
      }
      definitions.push_back(bound.callback);
    }
    constraints_ = std::move(constraints);
    parents_ = std::move(parents);
    definitions_ = std::move(definitions);
    for (size_t i = 0; i < constraints_.size(); ++i) {
      auto &constraint = *input.all_cons[i];
      auto &arrays = constraints_[i];
      constraint.rel_tiling_vars = arrays.tiling.data();
      constraint.rel_tiling_vars_size = static_cast<uint32_t>(arrays.tiling.size());
      constraint.rel_in_shapes = arrays.shapes.data();
      constraint.rel_in_shapes_size = static_cast<uint32_t>(arrays.shapes.size());
      constraint.eval_binding = {};
      constraint.eval_binding.expression_id = static_cast<uint32_t>(i);
      constraint.eval_binding.definition = definitions_[i].get();
      constraint.eval_entry = AxesProgramConstraintEntry<Context, Binding, Output>;
    }
    for (size_t i = 0; i < parents_.size(); ++i) {
      auto &variable = *input.tiling_vars[bindings.upper_bounds[i].target_index];
      const size_t id = constraints_.size() + i;
      variable.upper_bound_vars = parents_[i].data();
      variable.upper_bound_vars_size = static_cast<uint32_t>(parents_[i].size());
      variable.upper_bound_binding = {};
      variable.upper_bound_binding.expression_id = static_cast<uint32_t>(id);
      variable.upper_bound_binding.definition = definitions_[id].get();
      variable.upper_bound_entry = AxesProgramUpperBoundEntry<Context, Binding, Output>;
    }
    bound_ = true;
    return Status::kOk;
  }

 private:
  template <class Source, class Target>
  static bool Resolve(const std::vector<uint32_t> &indices, Source *source, std::vector<Target> &target) {
    target.reserve(indices.size());
    for (const auto index : indices) {
      if (source[index] == nullptr) {
        return false;
      }
      target.push_back(source[index]);
    }
    return true;
  }

  template <class Plain, class Tagged>
  static bool ResolveMixed(const std::vector<uint32_t> &indices, uint32_t flag, Plain plain, Tagged tagged,
                           std::vector<ShapePtr> &target) {
    target.reserve(indices.size());
    for (const auto encoded : indices) {
      const auto index = encoded & ~flag;
      ShapePtr item = (encoded & flag) != 0 ? tagged[index] : plain[index];
      if (item == nullptr) {
        return false;
      }
      target.push_back(item);
    }
    return true;
  }

  bool bound_ = false;
  std::vector<ConstraintArrays> constraints_;
  std::vector<std::vector<ShapePtr>> parents_;
  std::vector<std::shared_ptr<const AxesProgramEvaluation>> definitions_;
};
}  // namespace att
#endif
