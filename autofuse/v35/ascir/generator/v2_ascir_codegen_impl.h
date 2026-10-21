/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __V2_ASCIR_CODEGEN_IMPL__
#define __V2_ASCIR_CODEGEN_IMPL__

#include <algorithm>
#include <set>
#include "ascendc_ir.h"
#include "indirect_load_utils.h"
#include "reg_func/defalut_reg_func.h"
#include "reg_func/default_reg_func_v2.h"
#include "symbolizer/symbolic_utils.h"
#include "ascir_codegen_v2.h"
#include "schedule_result.h"
#include "generator/ascir_common.h"
#include "optimize/pre_process/pre_process_config.h"

namespace af {
namespace ascir {

namespace {
bool HasSimtArity(const AscNode &node, size_t input_count) {
  AscNode &mutable_node = const_cast<AscNode &>(node);
  return mutable_node.inputs.Size() == input_count && mutable_node.outputs().size() == 1UL;
}

ge::DataType GetSimtInputDtype(const AscNode &node) {
  return const_cast<AscNode &>(node).inputs[0].attr.dtype;
}

ge::DataType GetSimtOutputDtype(const AscNode &node) {
  return const_cast<AscNode &>(node).outputs[0].attr.dtype;
}

bool HasSameSimtUnaryDtype(const AscNode &node) {
  return HasSimtArity(node, 1UL) && GetSimtInputDtype(node) == GetSimtOutputDtype(node);
}

bool HasSameSimtBinaryDtype(const AscNode &node) {
  AscNode &mutable_node = const_cast<AscNode &>(node);
  if (!HasSimtArity(node, 2UL)) {
    return false;
  }
  const ge::DataType dtype = GetSimtOutputDtype(node);
  return mutable_node.inputs[0].attr.dtype == dtype && mutable_node.inputs[1].attr.dtype == dtype;
}

bool IsSimtFloatDtype(ge::DataType dtype) {
  return dtype == DT_FLOAT || dtype == DT_FLOAT16 || dtype == DT_BF16;
}

const char *GetSimtDtypeName(ge::DataType dtype) {
  static constexpr std::pair<ge::DataType, const char *> kDtypeNames[] = {
      {DT_FLOAT, "float"},     {DT_FLOAT16, "half"},  {DT_BF16, "bfloat16_t"}, {DT_INT8, "int8_t"},
      {DT_UINT8, "uint8_t"},   {DT_INT16, "int16_t"}, {DT_UINT16, "uint16_t"}, {DT_INT32, "int32_t"},
      {DT_UINT32, "uint32_t"}, {DT_INT64, "int64_t"}, {DT_UINT64, "uint64_t"}, {DT_BOOL, "bool"},
  };
  for (const auto &entry : kDtypeNames) {
    if (entry.first == dtype) {
      return entry.second;
    }
  }
  return nullptr;
}

bool HasSameSimtBinaryArithmeticDtype(const AscNode &node) {
  return HasSameSimtBinaryDtype(node) && GetSimtOutputDtype(node) != DT_BOOL &&
         GetSimtDtypeName(GetSimtOutputDtype(node)) != nullptr;
}

ge::graphStatus GenerateSimtBinaryExpr(const std::string &op, const std::vector<std::string> &inputs,
                                       std::string &expr) {
  expr = "(" + inputs[0] + " " + op + " " + inputs[1] + ")";
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenerateSimtFloatBinaryExpr(const std::string &op, const std::vector<std::string> &inputs,
                                            std::string &expr) {
  expr = "(static_cast<float>(" + inputs[0] + ") " + op + " static_cast<float>(" + inputs[1] + "))";
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenerateSimtFloatUnaryExpr(const std::string &api, const std::vector<std::string> &inputs,
                                           std::string &expr) {
  expr = "AscendC::Simt::" + api + "(static_cast<float>(" + inputs[0] + "))";
  return ge::GRAPH_SUCCESS;
}

bool IsSimtFloatToIntPair(const AscNode &node) {
  static constexpr std::pair<ge::DataType, ge::DataType> kPairs[] = {
      {DT_FLOAT, DT_INT64},   {DT_FLOAT, DT_INT32},  {DT_FLOAT, DT_INT16},   {DT_FLOAT16, DT_INT16},
      {DT_FLOAT16, DT_INT32}, {DT_FLOAT16, DT_INT8}, {DT_FLOAT16, DT_UINT8}, {DT_BF16, DT_INT32},
  };
  const std::pair<ge::DataType, ge::DataType> pair{GetSimtInputDtype(node), GetSimtOutputDtype(node)};
  return std::find(std::begin(kPairs), std::end(kPairs), pair) != std::end(kPairs);
}

ge::graphStatus GenerateSimtFloatToIntExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                           const std::string &round_mode, std::string &expr) {
  const ge::DataType output_dtype = GetSimtOutputDtype(node);
  const char *output_type = GetSimtDtypeName(output_dtype);
  GE_ASSERT_NOTNULL(output_type, "Node %s[%s] has unsupported SIMT output dtype %d", node.GetTypePtr(),
                    node.GetNamePtr(), static_cast<int32_t>(output_dtype));
  const char *intermediate_type = output_dtype == DT_INT64 ? "int64_t" : "int32_t";
  const std::string intermediate_expr = std::string("AscendC::Simt::Cast<") + intermediate_type +
                                        ", float, AscendC::RoundMode::" + round_mode + ">(" + "static_cast<float>(" +
                                        inputs[0] + "))";
  expr = output_dtype == DT_INT32 || output_dtype == DT_INT64
             ? intermediate_expr
             : std::string("static_cast<") + output_type + ">(" + intermediate_expr + ")";
  return ge::GRAPH_SUCCESS;
}

}  // namespace

class SimtFloatUnaryAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtUnaryDtype(node) && IsSimtFloatDtype(GetSimtInputDtype(node));
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatUnaryExpr(GetSimtScalarApiName(), inputs, expr);
  }

 protected:
  [[nodiscard]] virtual std::string GetSimtScalarApiName() const = 0;
};

/*********************************************************************************/
class VfAscIrCodegenImpl : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "VfCall";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"utils_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
};

/*********************************************************************************/
class DataAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Data";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
};

class ScalarAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Scalar";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
};

class IndexExprAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IndexExpr";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class ArangeAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    auto &mutable_node = const_cast<AscNode &>(node);
    if (mutable_node.inputs.Size() != 0UL || mutable_node.outputs().size() != 1UL) {
      return false;
    }
    auto &output = mutable_node.outputs[0].attr;
    if (output.dtype != DT_INT32 && output.dtype != DT_INT64) {
      return false;
    }
    return std::count_if(output.strides.begin(), output.strides.end(), [](const Expression &stride) {
             return SymbolicUtils::StaticCheckEq(stride, sym::kSymbolZero) != TriBool::kTrue;
           }) == 1UL;
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    GE_ASSERT_EQ(inputs.size(), 1UL);
    auto &mutable_node = const_cast<AscNode &>(node);
    GE_ASSERT_NOTNULL(mutable_node.attr.ir_attr, "Arange node has no IR attributes.");
    Expression base;
    Expression step;
    GE_ASSERT_GRAPH_SUCCESS(mutable_node.attr.ir_attr->GetAttrValue("base", base));
    GE_ASSERT_GRAPH_SUCCESS(mutable_node.attr.ir_attr->GetAttrValue("step", step));
    const auto &output = mutable_node.outputs[0].attr;
    const auto varying = std::find_if(output.strides.begin(), output.strides.end(), [](const Expression &stride) {
      return SymbolicUtils::StaticCheckEq(stride, sym::kSymbolZero) != TriBool::kTrue;
    });
    GE_ASSERT_TRUE(varying != output.strides.end(), "Arange has no varying axis.");
    const auto dtype = output.dtype == DT_INT64 ? "int64_t" : "int32_t";
    // Cast each operand before multiplication. The linear SIMT index is
    // unsigned; multiplying it by a negative Arange step before the final
    // cast would wrap to uint32 and produce a large positive coordinate.
    expr = "(static_cast<" + std::string(dtype) + ">(" + base.Str().get() + ") + static_cast<" + std::string(dtype) +
           ">(" + inputs[0] + ") * static_cast<" + std::string(dtype) + ">(" + step.Str().get() + "))";
    return ge::GRAPH_SUCCESS;
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    auto &mutable_node = const_cast<AscNode &>(node);
    if (mutable_node.outputs().size() != 1UL) {
      return false;
    }
    const auto &output = mutable_node.outputs[0].attr;
    if (output.dtype != DT_INT32 && output.dtype != DT_INT64) {
      return false;
    }
    if (output.vectorized_axis.empty() || output.vectorized_axis.size() != output.vectorized_strides.size()) {
      return false;
    }
    for (auto iter = output.vectorized_strides.rbegin(); iter != output.vectorized_strides.rend(); ++iter) {
      const auto stride = iter->Simplify();
      if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue) {
        continue;
      }
      return af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolOne) == af::TriBool::kTrue;
    }
    for (size_t i = 0UL; i < output.vectorized_axis.size(); ++i) {
      const auto axis_iter = std::find(output.axis.begin(), output.axis.end(), output.vectorized_axis[i]);
      if (axis_iter == output.axis.end()) {
        return false;
      }
      const auto axis_index = static_cast<size_t>(std::distance(output.axis.begin(), axis_iter));
      if (axis_index >= output.repeats.size() ||
          af::SymbolicUtils::StaticCheckEq(output.repeats[axis_index].Simplify(), af::sym::kSymbolOne) !=
              af::TriBool::kTrue) {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ArangeApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Arange";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroArangeApiCall";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Arange";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
};

class OutputAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Output";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
};

class WorkspaceAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Workspace";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class LoadAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "LoadRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DataCopyPadExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("datacopy") + std::string("_reg_base.h")};
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroLoadApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Load";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class NddmaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "NddmaApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DataCopyNddma";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"datacopy_nddma_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class BroadcastAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BroadcastRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BroadcastExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("broadcast") + std::string("_reg_base.h"), "duplicate.h"};
  }
  // 返回api call类的名称
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroScalarBroadcastApiCall";
  }

  // 返回api的名称
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Duplicate";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    AscNodeInputs node_inputs = node.inputs;
    auto vectorized_strides = node_inputs[0].attr.vectorized_strides;
    return std::all_of(vectorized_strides.begin(), vectorized_strides.end(), [](const Expression &i) {
      return SymbolicUtils::StaticCheckEq(i, sym::kSymbolZero) == TriBool::kTrue;
    });
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSimtArity(node, 1UL) && GetSimtInputDtype(node) == GetSimtOutputDtype(node);
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    GE_ASSERT_TRUE(inputs.size() == 1UL, "SIMT Broadcast expects one input, got %zu", inputs.size());
    expr = inputs[0];
    return ge::GRAPH_SUCCESS;
  }

  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/pad/broadcast.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_transpose_intf.h",
    };
  }
};

class NopAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Nop";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class CastAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CastV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "CastExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("cast") + std::string("_reg_base.h")};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroCastApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Cast";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const ge::AscNode &node) const override {
    std::map<ge::DataType, std::set<ge::DataType>> supported_map = {
        {DT_FLOAT, {DT_FLOAT16, DT_INT16, DT_INT32, DT_BF16}},
        {DT_FLOAT16, {DT_UINT8, DT_INT8, DT_FLOAT}},
        {DT_INT32, {DT_FLOAT, DT_INT16}},
        {DT_BF16, {DT_FLOAT}},
        {DT_UINT8, {DT_FLOAT16}},
        {DT_INT8, {DT_FLOAT16, DT_INT16}},
        {DT_INT16, {DT_FLOAT, DT_UINT8}},
    };
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    uint32_t input_dtype_size = GetSizeByDataType(node_inputs[0].attr.dtype);
    uint32_t output_dtype_size = GetSizeByDataType(node_outputs[0].attr.dtype);
    // Cast只能处理2倍及以内位宽变化的场景
    if ((input_dtype_size > output_dtype_size * 2U) || (output_dtype_size > input_dtype_size * 2U)) {
      return false;
    }
    // 8字节与4字节之间的转换在 MicroAPI 层要求源/目的使用 RegTraitNumTwo 双寄存器
    // (dav_m510 的 s642f32/f322s64 实现按 srcReg.reg[0]/reg[1] 取数)。VF codegen 只
    // 生成单寄存器张量, 该组合在设备上仅填充半数 lane(真机验证: 恰好 50% mismatch,
    // 交替 -0.0 模式), 因此拒绝 VF 融合, 保留在根图走 CastExtend(真机验证正确)。
    if ((input_dtype_size == 8U) != (output_dtype_size == 8U)) {
      return false;
    }

    auto iter = supported_map.find(node_inputs[0].attr.dtype);
    if (iter != supported_map.end()) {
      if (iter->second.find(node_outputs[0].attr.dtype) != iter->second.end()) {
        return true;
      }
    }
    return false;
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    if (!HasSimtArity(node, 1UL)) {
      return false;
    }
    static constexpr std::pair<ge::DataType, ge::DataType> kSupportedPairs[] = {
        {DT_FLOAT, DT_FLOAT},   {DT_FLOAT, DT_FLOAT16}, {DT_FLOAT, DT_INT64},   {DT_FLOAT, DT_INT32},
        {DT_FLOAT, DT_INT16},   {DT_FLOAT, DT_BF16},    {DT_FLOAT, DT_INT8},    {DT_FLOAT16, DT_FLOAT},
        {DT_FLOAT16, DT_INT32}, {DT_FLOAT16, DT_INT16}, {DT_FLOAT16, DT_INT8},  {DT_FLOAT16, DT_UINT8},
        {DT_UINT64, DT_INT64},  {DT_FLOAT16, DT_INT64}, {DT_UINT16, DT_INT16},  {DT_UINT8, DT_FLOAT16},
        {DT_UINT8, DT_FLOAT},   {DT_UINT8, DT_INT32},   {DT_UINT8, DT_INT16},   {DT_UINT8, DT_INT8},
        {DT_UINT8, DT_INT64},   {DT_UINT32, DT_INT32},  {DT_INT8, DT_FLOAT16},  {DT_INT8, DT_UINT8},
        {DT_INT8, DT_FLOAT},    {DT_INT8, DT_INT16},    {DT_INT16, DT_FLOAT16}, {DT_INT16, DT_FLOAT},
        {DT_INT16, DT_UINT16},  {DT_INT16, DT_INT8},    {DT_INT16, DT_UINT8},   {DT_INT32, DT_FLOAT},
        {DT_INT32, DT_INT64},   {DT_INT32, DT_INT16},   {DT_INT32, DT_FLOAT16}, {DT_INT32, DT_UINT32},
        {DT_INT64, DT_INT32},   {DT_INT64, DT_FLOAT},   {DT_INT64, DT_UINT8},   {DT_INT64, DT_UINT64},
        {DT_INT64, DT_FLOAT16}, {DT_INT64, DT_BF16},    {DT_BF16, DT_FLOAT},    {DT_BF16, DT_INT32},
        {DT_FLOAT, DT_BOOL},
    };
    const std::pair<ge::DataType, ge::DataType> pair{GetSimtInputDtype(node), GetSimtOutputDtype(node)};
    return std::find(std::begin(kSupportedPairs), std::end(kSupportedPairs), pair) != std::end(kSupportedPairs);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    auto outputs = const_cast<AscNode &>(node).outputs();
    GE_ASSERT_TRUE(outputs.size() == 1UL, "Cast node %s[%s] must have one output", node.GetTypePtr(),
                   node.GetNamePtr());
    const ge::DataType input_dtype = GetSimtInputDtype(node);
    const ge::DataType output_dtype_value = outputs[0]->attr.dtype;
    const char *output_dtype = GetSimtDtypeName(output_dtype_value);
    GE_ASSERT_NOTNULL(output_dtype, "Cast node %s[%s] has unsupported SIMT output dtype %d", node.GetTypePtr(),
                      node.GetNamePtr(), static_cast<int32_t>(output_dtype_value));
    const bool integer_output = output_dtype_value == DT_INT8 || output_dtype_value == DT_UINT8 ||
                                output_dtype_value == DT_INT16 || output_dtype_value == DT_UINT16 ||
                                output_dtype_value == DT_INT32 || output_dtype_value == DT_INT64;
    if (!IsSimtFloatDtype(input_dtype) && output_dtype_value == DT_BF16) {
      expr = "static_cast<bfloat16_t>(static_cast<float>(" + inputs[0] + "))";
    } else if (IsSimtFloatDtype(input_dtype) && integer_output) {
      const char *intermediate_type = output_dtype_value == DT_INT64 ? "int64_t" : "int32_t";
      const std::string intermediate_expr = std::string("AscendC::Simt::Cast<") + intermediate_type +
                                            ", float, AscendC::RoundMode::CAST_ZERO>(static_cast<float>(" + inputs[0] +
                                            "))";
      expr = output_dtype_value == DT_INT32 || output_dtype_value == DT_INT64
                 ? intermediate_expr
                 : std::string("static_cast<") + output_dtype + ">(" + intermediate_expr + ")";
    } else {
      expr = std::string("static_cast<") + output_dtype + ">(" + inputs[0] + ")";
    }
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AbsAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Abs";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Abs";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    if (!HasSameSimtUnaryDtype(node)) {
      return false;
    }
    const ge::DataType dtype = GetSimtOutputDtype(node);
    return dtype == DT_FLOAT || dtype == DT_FLOAT16 || dtype == DT_BF16 || dtype == DT_INT8 || dtype == DT_INT16 ||
           dtype == DT_INT32 || dtype == DT_INT64 || dtype == DT_UINT8;
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    const ge::DataType dtype = GetSimtOutputDtype(node);
    if (dtype == DT_FLOAT || dtype == DT_FLOAT16 || dtype == DT_INT32 || dtype == DT_INT64) {
      expr = "AscendC::Simt::Abs(" + inputs[0] + ")";
    } else if (dtype == DT_UINT8) {
      expr = inputs[0];
    } else if (dtype == DT_BF16) {
      expr = "AscendC::Simt::Abs(static_cast<float>(" + inputs[0] + "))";
    } else {
      expr = "(" + inputs[0] + " < 0) ? -(" + inputs[0] + ") : " + inputs[0];
    }
    return ge::GRAPH_SUCCESS;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &abs_node) const override {
    (void)abs_node;
    return true;
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}, {DT_UINT8, DT_INT16}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ExpAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Exp";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Exp";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &exp_node) const override {
    (void)exp_node;
    return true;
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtUnaryDtype(node) && IsSimtFloatDtype(GetSimtInputDtype(node));
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    auto &outputs = const_cast<AscNode &>(node).outputs;
    if (outputs[0].attr.dtype == DT_FLOAT) {
      expr = "AscendC::Simt::Exp(" + inputs[0] + ")";
    } else {
      expr = "AscendC::Simt::Exp(static_cast<float>(" + inputs[0] + "))";
    }
    return ge::GRAPH_SUCCESS;
  }
};

class Exp2AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcExp2TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Exp2";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"exp2_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/power.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtUnaryDtype(node) && IsSimtFloatDtype(GetSimtInputDtype(node));
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    auto &outputs = const_cast<AscNode &>(node).outputs;
    if (outputs[0].attr.dtype == DT_FLOAT) {
      expr = "AscendC::Simt::Exp2(" + inputs[0] + ")";
    } else {
      expr = "AscendC::Simt::Exp2(static_cast<float>(" + inputs[0] + "))";
    }
    return ge::GRAPH_SUCCESS;
  }
};

class FloorAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Floor";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Floor";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/floor.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ModifiedBesselI0AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcModifiedBesselI0TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ModifiedBesselI0Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "CylBesselI0";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i0_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ModifiedBesselI1AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcModifiedBesselI1TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ModifiedBesselI1Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "CylBesselI1";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i1_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ModifiedBesselK0AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcModifiedBesselK0TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ModifiedBesselK0Extend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i0_reg_base.h", "modified_bessel_k0_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ModifiedBesselK1AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcModifiedBesselK1TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ModifiedBesselK1Extend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i1_reg_base.h", "modified_bessel_k1_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class I0AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "I0Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "j0";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"i0_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class I0eAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "I0eExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "j0";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i0_reg_base.h", "i0e_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class I1eAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "I1eExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "j1";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i1_reg_base.h", "i1e_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class BesselJ0AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BesselJ0Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "J0";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h", "bessel_j0_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class BesselJ1AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BesselJ1Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "J1";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h", "bessel_j1_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class BesselY0AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BesselY0Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Y0";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h", "bessel_j0_reg_base.h",
            "bessel_y0_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class BesselY1AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BesselY1Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Y1";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h", "bessel_j1_reg_base.h",
            "bessel_y1_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ScaledModifiedBesselK0AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcScaledModifiedBesselK0TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ScaledModifiedBesselK0Extend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i0_reg_base.h", "modified_bessel_k0_reg_base.h",
            "scaled_modified_bessel_k0_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ScaledModifiedBesselK1AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcScaledModifiedBesselK1TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ScaledModifiedBesselK1Extend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"modified_bessel_utils_reg_base.h", "modified_bessel_i1_reg_base.h", "modified_bessel_k1_reg_base.h",
            "scaled_modified_bessel_k1_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SphericalBesselJ0AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SphericalBesselJ0Extend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bessel_j_utils_reg_base.h", "trigonometric_function_utils_reg_base.h", "spherical_bessel_j0_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class NdtrAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "NdtrExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Normcdf";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"ndtr_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class NdtriAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "NdtriExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"ndtri_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LogNdtrAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogNdtrExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Normcdf";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"log_ndtr_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class NextAfterAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "NextAfterExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"next_after_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class PolyGammaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcPolygammaTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "PolyGammaExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"trigonometric_function_utils_reg_base.h", "zeta_reg_base.h", "polygamma_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "adv_api/math/digamma.h",
        "adv_api/math/lgamma.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SignBitAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SignBitExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"signbit_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class FrexpAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryDoubleOutputApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "FrexpExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"utils_reg_base.h", "frexp_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    AscNodeOutputs node_outputs = node.outputs;
    GE_ASSERT_TRUE(node_outputs().size() == 2U, "Node %s[%s] output tensor size is not equal with 2", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithOutput(node, 0U),
                      "Node %s[%s] check mantissa shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithOutput(node, 1U),
                      "Node %s[%s] check exponent shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class IgammaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcIgammaTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IgammaExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"igammac_helper/series_reg_base.h",
            "igammac_helper/continued_fraction_reg_base.h",
            "igammac_helper/asymptotic_series_reg_base.h",
            "igammac_helper/series_complement_reg_base.h",
            "igammac_reg_base.h",
            "igamma_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/math/erfc.h",
        "adv_api/math/lgamma.h",
        "adv_api/math/power.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class IgammacAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcIgammacTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IgammacExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"igammac_helper/series_reg_base.h", "igammac_helper/continued_fraction_reg_base.h",
            "igammac_helper/asymptotic_series_reg_base.h", "igammac_helper/series_complement_reg_base.h",
            "igammac_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/math/erfc.h",
        "adv_api/math/lgamma.h",
        "adv_api/math/power.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ZetaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ZetaExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"zeta_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ShiftedChebyshevPolynomialTAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftedChebyshevPolynomialTExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"shifted_chebyshev_polynomial_utils_reg_base.h", "shifted_chebyshev_polynomial_t_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ShiftedChebyshevPolynomialUAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftedChebyshevPolynomialUExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"shifted_chebyshev_polynomial_utils_reg_base.h", "shifted_chebyshev_polynomial_u_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ShiftedChebyshevPolynomialVAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftedChebyshevPolynomialVExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"shifted_chebyshev_polynomial_utils_reg_base.h", "shifted_chebyshev_polynomial_v_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ShiftedChebyshevPolynomialWAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftedChebyshevPolynomialWExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"shifted_chebyshev_polynomial_utils_reg_base.h", "shifted_chebyshev_polynomial_w_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ChebyshevPolynomialTAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ChebyshevPolynomialTExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"chebyshev_polynomial_utils_reg_base.h", "chebyshev_polynomial_t_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ChebyshevPolynomialUAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ChebyshevPolynomialUExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"chebyshev_polynomial_utils_reg_base.h", "chebyshev_polynomial_u_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ChebyshevPolynomialVAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ChebyshevPolynomialVExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"chebyshev_polynomial_utils_reg_base.h", "chebyshev_polynomial_v_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ChebyshevPolynomialWAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryTemplateAttrApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ChebyshevPolynomialWExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"chebyshev_polynomial_utils_reg_base.h", "chebyshev_polynomial_w_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class HermitePolynomialHAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "HermitePolynomialHExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"hermite_polynomial_h_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class HermitePolynomialHeAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "HermitePolynomialHeExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"hermite_polynomial_he_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LaguerrePolynomialLAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LaguerrePolynomialLExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"laguerre_polynomial_l_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LegendrePolynomialPAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LegendrePolynomialPExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"legendre_polynomial_p_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AiryAiAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AiryAiExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"trigonometric_function_utils_reg_base.h", "airy_ai_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ErfinvAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Erfinv";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Erfinv";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"erfinv_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class FmaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "TernaryApiTmpV2Call";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Fma";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT}, {DT_INT8, DT_FLOAT16}, {DT_INT16, DT_FLOAT}, {DT_UINT8, DT_FLOAT16}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/fma.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class RemovePadAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "RemovePadApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "RemovePad";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"removepad.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class PadAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiTilingTypeName() const override {
    return "PadTiling";
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "PadApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Pad";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/pad/pad.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class RoundAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Round";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Round";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/round.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LnAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Ln";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Ln";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &ln_node) const override {
    (void)ln_node;
    return true;
  }

  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtUnaryDtype(node) && IsSimtFloatDtype(GetSimtInputDtype(node));
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    auto &outputs = const_cast<AscNode &>(node).outputs;
    if (outputs[0].attr.dtype == DT_FLOAT) {
      expr = "AscendC::Simt::Log(" + inputs[0] + ")";
    } else {
      expr = "AscendC::Simt::Log(static_cast<float>(" + inputs[0] + "))";
    }
    return ge::GRAPH_SUCCESS;
  }
};

class Expm1AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Expm1Extend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Expm1";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"expm1_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "simt_api/cpp/kernel_simt_intf.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class Log2AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcLog2TmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Log2";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Log2";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    // 与CalcLog2TmpSizeV2中的dtype_conversion_map表格同步维护
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
    ;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/log.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LShiftAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftLeft";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ModAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "FmodExtend";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT}, {DT_INT16, DT_FLOAT}, {DT_INT8, DT_FLOAT16}, {DT_UINT8, DT_FLOAT16}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/fmod.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"fmod_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SqrtAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Sqrt";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Sqrt";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &sqrt_node) const override {
    (void)sqrt_node;
    return true;
  }

  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &sqrt_node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(sqrt_node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtUnaryDtype(node) && IsSimtFloatDtype(GetSimtInputDtype(node));
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    auto &outputs = const_cast<AscNode &>(node).outputs;
    if (outputs[0].attr.dtype == DT_FLOAT) {
      expr = "AscendC::Simt::Sqrt(" + inputs[0] + ")";
    } else {
      expr = "AscendC::Simt::Sqrt(static_cast<float>(" + inputs[0] + "))";
    }
    return ge::GRAPH_SUCCESS;
  }
};

class RsqrtAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    if (IsRsqrtInBlacklist()) {
      return std::vector<std::unique_ptr<TmpBufDesc>>();
    }
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return IsRsqrtInBlacklist() ? "UnaryApiCall" : "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return IsRsqrtInBlacklist() ? "Rsqrt" : "RsqrtExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Rsqrt";
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroRsqrtApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Rsqrt";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &rsqrt_node) const override {
    (void)rsqrt_node;
    return true;
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &rsqrt_node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(rsqrt_node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    if (IsRsqrtInBlacklist()) {
      return {};
    }
    return {"rsqrt_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

 private:
  // Rsqrt 命中精度提升黑名单（配置 Rsqrt 或 all）时，codegen 回落内置 Rsqrt（Vector
  // adv_api）快速路径，跳过 regbase Extend 高精度实现；类型字符串与 REG_ASC_IR(Rsqrt) 注册名一致
  static bool IsRsqrtInBlacklist() {
    return af::pre_process::PreProcessConfig::Instance().IsInImprovePrecisionBlacklist("Rsqrt");
  }
};

class NegAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "NegExtend";
  }

  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"neg_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Neg";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    if (!HasSameSimtUnaryDtype(node)) {
      return false;
    }
    const ge::DataType dtype = GetSimtOutputDtype(node);
    return dtype == DT_BF16 || dtype == DT_INT8 || dtype == DT_INT16 || dtype == DT_INT32 || dtype == DT_INT64 ||
           dtype == DT_FLOAT16 || dtype == DT_FLOAT;
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    auto &outputs = const_cast<AscNode &>(node).outputs;
    if (outputs[0].attr.dtype == DT_FLOAT16 || outputs[0].attr.dtype == DT_BF16) {
      expr = "-static_cast<float>(" + inputs[0] + ")";
    } else {
      expr = "-(" + inputs[0] + ")";
    }
    return ge::GRAPH_SUCCESS;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &neg_node) const override {
    (void)neg_node;
    return true;
  }

  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    const std::map<ge::DataType, ge::DataType> neg_dtype_map = {{ge::DataType::DT_INT8, ge::DataType::DT_INT16},
                                                                {ge::DataType::DT_BF16, ge::DataType::DT_FLOAT}};
    return GetConversionFromDtypeMap(node, neg_dtype_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ReluAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Relu";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Relu";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    if (!HasSameSimtUnaryDtype(node)) {
      return false;
    }
    const ge::DataType dtype = GetSimtOutputDtype(node);
    return dtype == DT_INT32 || dtype == DT_FLOAT16 || dtype == DT_FLOAT || dtype == DT_UINT8 || dtype == DT_INT64;
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "AscendC::Simt::Max(" + inputs[0] + ", static_cast<decltype(" + inputs[0] + ")>(0))";
    return ge::GRAPH_SUCCESS;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &relu_node) const override {
    (void)relu_node;
    return true;
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_UINT8, DT_FLOAT16}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ReciprocalAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    if (IsReciprocalInBlacklist()) {
      return std::vector<std::unique_ptr<TmpBufDesc>>();
    }
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return IsReciprocalInBlacklist() ? "UnaryApiCall" : "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return IsReciprocalInBlacklist() ? "Reciprocal" : "ReciprocalExtend";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatDtype(GetSimtInputDtype(node)) && GetSimtInputDtype(node) == GetSimtOutputDtype(node);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "1.0F / static_cast<float>(" + inputs[0] + ")";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &reciprocal_node) const override {
    (void)reciprocal_node;
    return true;
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    if (IsReciprocalInBlacklist()) {
      return {};
    }
    return {"reciprocal_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    if (IsReciprocalInBlacklist()) {
      return {
          "basic_api/kernel_operator_vec_unary_intf.h",
      };
    }
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

 private:
  // Reciprocal 命中精度提升黑名单（配置 Reciprocal 或 all）时，codegen 回落内置 Reciprocal（Vector
  // adv_api）快速路径，跳过 regbase Extend 高精度实现；类型字符串与 REG_ASC_IR(Reciprocal) 注册名一致
  static bool IsReciprocalInBlacklist() {
    return af::pre_process::PreProcessConfig::Instance().IsInImprovePrecisionBlacklist("Reciprocal");
  }
};

class SignAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SignExtend";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    const ge::DataType dtype = GetSimtInputDtype(node);
    return dtype == GetSimtOutputDtype(node) &&
           (IsSimtFloatDtype(dtype) || dtype == DT_INT32 || dtype == DT_INT64 || dtype == DT_UINT8);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "(" + inputs[0] + " > 0) ? 1 : ((" + inputs[0] + " < 0) ? -1 : 0)";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"cast.h", "sign_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "adv_api/math/sign.h",
        "utils/std/type_traits.h",
    };
  }

  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_UINT8, DT_FLOAT16}, {DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }

  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class IsnanAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IsNan";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatDtype(GetSimtInputDtype(node)) &&
           (GetSimtOutputDtype(node) == DT_UINT8 || GetSimtOutputDtype(node) == DT_BOOL);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatUnaryExpr("IsNan", inputs, expr);
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/is_nan.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class IsFiniteAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IsFinite";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatDtype(GetSimtInputDtype(node)) &&
           (GetSimtOutputDtype(node) == DT_UINT8 || GetSimtOutputDtype(node) == DT_BOOL);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatUnaryExpr("IsFinite", inputs, expr);
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/is_finite.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class IsInfAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IsInf";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatDtype(GetSimtInputDtype(node)) &&
           (GetSimtOutputDtype(node) == DT_UINT8 || GetSimtOutputDtype(node) == DT_BOOL);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatUnaryExpr("IsInf", inputs, expr);
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/is_inf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LogicalNotAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogicalNotExtend";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    const ge::DataType input_dtype = GetSimtInputDtype(node);
    const ge::DataType output_dtype = GetSimtOutputDtype(node);
    return (input_dtype == DT_BOOL && output_dtype == DT_BOOL) ||
           (input_dtype != DT_BOOL && GetSimtDtypeName(input_dtype) != nullptr && output_dtype == DT_UINT8);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "!(" + inputs[0] + ")";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("logical_not") + std::string("_reg_base.h")};
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &not_node) const override {
    (void)not_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/logical_not.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MaxAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ReduceMax";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init_reg_base.h", "reduce_extend_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    const std::map<ge::DataType, ge::DataType> max_dtype_map = {{ge::DataType::DT_UINT8, ge::DataType::DT_INT16}};
    return GetConversionFromDtypeMap(node, max_dtype_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class SumAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ReduceSum";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init_reg_base.h", "reduce_extend_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    const std::map<ge::DataType, ge::DataType> sum_dtype_map = {{ge::DataType::DT_BF16, ge::DataType::DT_FLOAT},
                                                                {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT}};
    return GetConversionFromDtypeMap(node, sum_dtype_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class MinAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ReduceMin";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init_reg_base.h", "reduce_extend_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    const std::map<ge::DataType, ge::DataType> min_dtype_map = {{ge::DataType::DT_UINT8, ge::DataType::DT_INT16}};
    return GetConversionFromDtypeMap(node, min_dtype_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class MeanAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ReduceMean";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init_reg_base.h", "reduce_extend_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class ProdAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ReduceProd";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init_reg_base.h", "reduce_extend_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class AnyAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ReduceAny";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init_reg_base.h", "reduce_extend_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class AllAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "RegReduceApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ReduceAll";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init_reg_base.h", "reduce_extend_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

/*********************************************************************************/
class CompareAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return GetCompareSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CompareV2ApiCall";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"compare_reg_base.h"};
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroCompareApiCall";
  }
  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    if (!IsAllVecAxisContinuous(node)) {
      return false;
    }
    const auto first_input_type = node.GetInDataNodes().at(0)->GetType();
    if (first_input_type == "Scalar" || first_input_type == "IndexExpr") {
      return false;
    }
    // MicroAPI::Compare 不支持 bool，BOOL 输入不走 VF，回落到 CompareV2ApiCall（regbase 内部按 uint8 执行）
    AscNodeInputs node_inputs = node.inputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DT_BOOL) {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    AscNode &mutable_node = const_cast<AscNode &>(node);
    if (!HasSimtArity(node, 2UL) || mutable_node.inputs[0].attr.dtype != mutable_node.inputs[1].attr.dtype) {
      return false;
    }
    const ge::DataType input_dtype = mutable_node.inputs[0].attr.dtype;
    const ge::DataType output_dtype = GetSimtOutputDtype(node);
    return input_dtype != DT_BOOL && GetSimtDtypeName(input_dtype) != nullptr &&
           (output_dtype == DT_BOOL || output_dtype == DT_UINT8);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtBinaryExpr(GetSimtScalarOperator(), inputs, expr);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

 protected:
  [[nodiscard]] virtual std::string GetSimtScalarOperator() const = 0;
};

class GeAscIrCodegenImplV2 : public CompareAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiName() const override {
    return "GE";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "GE";
  }

 protected:
  [[nodiscard]] std::string GetSimtScalarOperator() const override {
    return ">=";
  }
};

class EqAscIrCodegenImplV2 : public CompareAscIrCodegenImplV2 {
 public:
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "EQ";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "EQ";
  }

 protected:
  [[nodiscard]] std::string GetSimtScalarOperator() const override {
    return "==";
  }
};

class NeAscIrCodegenImplV2 : public CompareAscIrCodegenImplV2 {
 public:
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const ge::AscNode &node) {
    std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> conversion_dtype;
    AscNodeInputs node_inputs = node.inputs;
    AscNodeOutputs node_outputs = node.outputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DataType::DT_UINT8) {
        conversion_dtype.first.emplace_back(ge::DataType::DT_INT16);
      } else {
        conversion_dtype.first.emplace_back(node_inputs[i].attr.dtype);
      }
    }
    for (size_t i = 0; i < node_outputs().size(); i++) {
      conversion_dtype.second.emplace_back(node_outputs[i].attr.dtype);
    }
    return conversion_dtype;
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "NE";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "NE";
  }

 protected:
  [[nodiscard]] std::string GetSimtScalarOperator() const override {
    return "!=";
  }
};

class GtAscIrCodegenImplV2 : public CompareAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiName() const override {
    return "GT";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "GT";
  }

 protected:
  [[nodiscard]] std::string GetSimtScalarOperator() const override {
    return ">";
  }
};

class LeAscIrCodegenImplV2 : public CompareAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiName() const override {
    return "LE";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "LE";
  }

 protected:
  [[nodiscard]] std::string GetSimtScalarOperator() const override {
    return "<=";
  }
};

class LtAscIrCodegenImplV2 : public CompareAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiName() const override {
    return "LT";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "LT";
  }

 protected:
  [[nodiscard]] std::string GetSimtScalarOperator() const override {
    return "<";
  }
};

class SigmoidAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Sigmoid";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatDtype(GetSimtInputDtype(node)) && GetSimtInputDtype(node) == GetSimtOutputDtype(node);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "1.0F / (1.0F + AscendC::Simt::Exp(-static_cast<float>(" + inputs[0] + ")))";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/activation/sigmoid.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class Ub2ubAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "Ub2ubApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DataCopy";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

/**************************************************************/
class DivAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DivExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"div_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroDivApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Div";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryDtype(node) &&
           (GetSimtOutputDtype(node) == DT_FLOAT16 || GetSimtOutputDtype(node) == DT_FLOAT);
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatBinaryExpr("/", inputs, expr);
  }

  [[nodiscard]] bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list;  // 支持任意输入是scalar
    return true;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &div_node) const override {
    (void)div_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SubAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SubExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"sub_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Sub";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryArithmeticDtype(node);
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtBinaryExpr("-", inputs, expr);
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &sub_node) const override {
    (void)sub_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AddAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Add";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_add.h"};
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroBinaryScalarApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Add";
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryArithmeticDtype(node);
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtBinaryExpr("+", inputs, expr);
  }

  [[nodiscard]] bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &add_node) const override {
    (void)add_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MulAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Mul";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_mul.h"};
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Mul";
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryArithmeticDtype(node) && GetSimtOutputDtype(node) != DT_UINT16;
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtBinaryExpr("*", inputs, expr);
  }

  [[nodiscard]] bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &mul_node) const override {
    (void)mul_node;
    return true;
  }

  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    const std::map<ge::DataType, ge::DataType> mul_dtype_map = {{ge::DataType::DT_INT8, ge::DataType::DT_INT16},
                                                                {ge::DataType::DT_UINT8, ge::DataType::DT_INT16}};
    return GetConversionFromDtypeMap(node, mul_dtype_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class TrueDivAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DivExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"div_reg_base.h"};
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroDivApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Div";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryDtype(node) && IsSimtFloatDtype(GetSimtOutputDtype(node));
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatBinaryExpr("/", inputs, expr);
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &true_div_node) const override {
    (void)true_div_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MinimumAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AscendC::Min";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_minimum.h"};
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroBinaryScalarApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Min";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }
  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryArithmeticDtype(node);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "(" + inputs[0] + " < " + inputs[1] + " ? " + inputs[0] + " : " + inputs[1] + ")";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &minimum_node) const override {
    (void)minimum_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MaximumAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AscendC::Max";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_maximum.h"};
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroBinaryScalarApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Max";
  }

  [[nodiscard]] bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryArithmeticDtype(node);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "(" + inputs[0] + " > " + inputs[1] + " ? " + inputs[0] + " : " + inputs[1] + ")";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &maximum_node) const override {
    (void)maximum_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/

class WhereAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "WhereRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "WhereExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"where_reg_base.h"};
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroWhereApiCall";
  }
  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Select";
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return !is_scalar_list[0];  // 除第1个外都支持Scalar
  }
  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    if (!IsAllVecAxisContinuous(node)) {
      return false;
    }
    return true;
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    AscNode &mutable_node = const_cast<AscNode &>(node);
    if (!HasSimtArity(node, 3UL)) {
      return false;
    }
    const ge::DataType mask_dtype = mutable_node.inputs[0].attr.dtype;
    const ge::DataType output_dtype = GetSimtOutputDtype(node);
    bool value_dtype_supported = output_dtype != DT_BOOL && GetSimtDtypeName(output_dtype) != nullptr;
    if (node.GetType() == "Select") {
      value_dtype_supported = output_dtype == DT_FLOAT16 || output_dtype == DT_FLOAT || output_dtype == DT_BF16 ||
                              output_dtype == DT_INT16 || output_dtype == DT_INT32 || output_dtype == DT_INT64;
    }
    return (mask_dtype == DT_BOOL || mask_dtype == DT_UINT8) && mutable_node.inputs[1].attr.dtype == output_dtype &&
           mutable_node.inputs[2].attr.dtype == output_dtype && value_dtype_supported;
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "((" + inputs[0] + " != 0) ? " + inputs[1] + " : " + inputs[2] + ")";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "adv_api/math/where.h",
        "utils/std/type_traits.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1, 2}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

// MaskedFill reuses Where codegen. IsScalarInputSupported checks original input order
// (before MaskedFillInputReorderPass): inputs[1]=mask cannot be scalar.
class MaskedFillAscIrCodegenImplV2 : public WhereAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcMaskedFillTmpSize(node);
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return !is_scalar_list[1];
  }
  [[nodiscard]] bool IsSimtScalarSupported([[maybe_unused]] const AscNode &node) const override {
    // MaskedFillInputReorderPass converts supported nodes to Select(mask, value, x).
    return false;
  }
};
/*********************************************************************************/
class LeakyReluAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "LeakyReluApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LeakyRelu";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroLeakyReluApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "LeakyRelu";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return true;
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &leaky_relu_node) const override {
    (void)leaky_relu_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtUnaryDtype(node) &&
           (GetSimtOutputDtype(node) == DT_FLOAT16 || GetSimtOutputDtype(node) == DT_FLOAT);
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    float negative_slope = 0.0f;
    GE_ASSERT_NOTNULL(node.attr.ir_attr, "LeakyRelu node %s has no ir attr", node.GetNamePtr());
    GE_ASSERT_GRAPH_SUCCESS(
        const_cast<AscIrAttrDefBase *>(node.attr.ir_attr.get())->GetAttrValue("negative_slope", negative_slope));
    const std::string &input = inputs[0];
    expr = "(" + input + " > static_cast<decltype(" + input + ")>(0)) ? " + input + " : static_cast<decltype(" + input +
           ")>(" + std::to_string(negative_slope) + ") * " + input;
    return ge::GRAPH_SUCCESS;
  }
};

/*********************************************************************************/
class ClipByValueAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "ClipByValueApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ClipByValue";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"clipbyvalue_reg_base.h"};
  }
  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list;  // 支持任意输入是scalar
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/clamp.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {0, 1, 2}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class StoreAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "StoreRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "DataCopyPadExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("datacopy") + std::string("_reg_base.h")};
  }
  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroStoreApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Store";
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
};
/*********************************************************************************/
class ConcatAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcConcatTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "ConcatRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Concat";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"concat_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"basic_api/reg_compute/kernel_reg_compute_intf.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class SplitAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSplitTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "SplitRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Split";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"split_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"basic_api/reg_compute/kernel_reg_compute_intf.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class GatherAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcGatherTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "GatherRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "GatherExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"gather_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_scalar_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "simt_api/cpp/kernel_simt_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
namespace {
bool CalcIndirectLoadSkTmpSize(const AscNode &node, const Expression &offset_size, Expression &tmp_size) {
  constexpr uint32_t kUbBlockAlignBytes = 32U;
  auto inputs = node.inputs;
  const auto &input = inputs[0].attr;
  int64_t axis = 0;
  (void)node.attr.ir_attr->GetAttrValue("axis", axis);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  const auto op_desc = node.GetOpDesc();
  if (op_desc != nullptr) {
    logical_view = op_desc->TryGetExtAttr(ascgen_utils::indirect_load::kTemplateLogicalViewAttr,
                                          ascgen_utils::indirect_load::TemplateLogicalView{});
  }
  const bool has_logical_view = !logical_view.input.sizes.empty() &&
                                logical_view.input.sizes.size() == logical_view.input.strides.size() &&
                                logical_view.input.sizes.size() == logical_view.index.sizes.size();
  GE_ASSERT_TRUE(has_logical_view, "IndirectLoad SK logical view is missing.");
  const auto &input_repeats = logical_view.input.sizes;
  const auto &input_strides = logical_view.input.strides;
  const auto &index_repeats = logical_view.index.sizes;
  if (input_repeats.empty() || input_repeats.size() != input_strides.size() ||
      input_repeats.size() != index_repeats.size()) {
    GELOGW("[IndirectLoad] Skip invalid SK tmp layout, input rank[%zu], stride rank[%zu], index rank[%zu].",
           input_repeats.size(), input_strides.size(), index_repeats.size());
    return false;
  }
  const int64_t rank = static_cast<int64_t>(index_repeats.size());
  if (axis < -rank || axis >= rank) {
    GELOGW("[IndirectLoad] Skip invalid SK tmp axis[%ld], rank[%ld].", axis, rank);
    return false;
  }
  const size_t axis_index = static_cast<size_t>(axis < 0L ? axis + rank : axis);
  Expression outer_count = Symbol(1);
  for (size_t dim = 0UL; dim < axis_index; ++dim) {
    outer_count = outer_count * index_repeats[dim];
  }
  Expression input_slice_count = Symbol(1);
  for (size_t dim = axis_index; dim < index_repeats.size(); ++dim) {
    const Expression &repeat = dim == axis_index ? input_repeats[dim] : index_repeats[dim];
    input_slice_count = input_slice_count + (repeat - Symbol(1)) * input_strides[dim];
  }
  const Expression input_size = outer_count * input_slice_count * Symbol(GetSizeByDataType(input.dtype));
  tmp_size = af::sym::Align(input_size, kUbBlockAlignBytes) + offset_size;
  return true;
}
}  // namespace

class IndirectLoadAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    const auto template_id = ::ascir::GetTemplateIdOrDefault(node);
    if (template_id != ::ascir::TemplateId::kIndirectLoadSimd && template_id != ::ascir::TemplateId::kIndirectLoadSK) {
      // SIMD uses an offsets buffer, SK uses an input cache and offsets buffer, while SIMT accesses GM directly.
      return {};
    }
    auto node_outputs = node.outputs;
    const auto &y = node_outputs[0].attr;
    Expression offset_size = Symbol(sizeof(uint32_t));
    for (const auto &repeat : y.repeats) {
      offset_size = offset_size * repeat;
    }
    Expression tmp_size = offset_size;
    if (template_id == ::ascir::TemplateId::kIndirectLoadSK &&
        !CalcIndirectLoadSkTmpSize(node, offset_size, tmp_size)) {
      return {};
    }
    TmpBufDesc desc = {af::sym::Align(tmp_size, 32), -1};
    std::vector<std::unique_ptr<TmpBufDesc>> tmp_buf_descs;
    tmp_buf_descs.emplace_back(std::make_unique<TmpBufDesc>(desc));
    return tmp_buf_descs;
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "IndirectLoadRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "IndirectLoad";
  }
  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    (void)node;
    return false;
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"datacopy_reg_base.h", "indirect_load_simd_policy_reg_base.h", "indirect_load_simd_reg_base.h",
            "indirect_load_sk_reg_base.h", "indirect_load_simt_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"basic_api/kernel_operator_vec_gather_intf.h", "basic_api/reg_compute/kernel_reg_compute_intf.h",
            "simt_api/cpp/kernel_simt_intf.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class TransposeAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcTransposeTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "TransposeRegApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "TransposeExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"transpose_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/

class ErfAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ErfExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Erf";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"erf_reg_base.h"};
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"adv_api/math/erf.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class CeilAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcCeilTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "CeilExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Ceil";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"ceil_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/ceil.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class CosAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcCosTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "CosExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Cos";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
        {DT_FLOAT16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"cos_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/cos.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AcosAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAcosTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AcosExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Acos";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
        {DT_FLOAT16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"acos_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/acos.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AcoshAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAcoshTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AcoshExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Acosh";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"acosh_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/acosh.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class CoshAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcCoshTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "CoshExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Cosh";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"cosh_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/cosh.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AsinAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAsinTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AsinExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Asin";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
        {DT_FLOAT16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"asin_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/asin.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AsinhAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAsinhTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AsinhExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Asinh";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"asinh_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/asinh.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AtanAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAtanTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AtanExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Atan";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"atan_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/atan.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AtanhAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAtanhTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AtanhExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Atanh";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
        {DT_FLOAT16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"atanh_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/atanh.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class DigammaAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcDigammaTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Digamma";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/digamma.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ErfcAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcErfcTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Erfc";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Erfc";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/erfc.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ErfcxAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcErfcTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ErfcxExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Erfcx";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "adv_api/math/erfc.h",
    };
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"erfcx_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class Atan2AscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAtanTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Atan2Extend";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/atan.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"atan2_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class CopySignAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "CopySignExtend";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    const ge::DataType dtype = GetSimtOutputDtype(node);
    return HasSameSimtBinaryDtype(node) && (dtype == DT_FLOAT16 || dtype == DT_FLOAT);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "AscendC::Simt::CopySign(static_cast<float>(" + inputs[0] + "), static_cast<float>(" + inputs[1] + "))";
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"copy_sign_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class Ceil2IntAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "CastV2ApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Ceil2IntExtend";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatDtype(GetSimtInputDtype(node)) && GetSimtOutputDtype(node) == DT_INT32;
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatToIntExpr(node, inputs, "CAST_CEIL", expr);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("cast") + std::string("_reg_base.h")};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class TanhAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "TanhExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Tanh";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"tanh_reg_base.h"};
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/tanh.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class GeluAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Gelu";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"adv_api/activation/gelu.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class LogicalOrAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogicalOrExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"logical_reg_base.h"};
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return HasSameSimtBinaryDtype(node) &&
           (GetSimtOutputDtype(node) == DT_BOOL || GetSimtOutputDtype(node) == DT_UINT8);
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "((" + inputs[0] + " != 0) || (" + inputs[1] + " != 0))";
    return ge::GRAPH_SUCCESS;
  }

  [[nodiscard]] bool IsInplaceSupported(const AscNode &or_node) const override {
    (void)or_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "adv_api/math/logical_and.h",
        "adv_api/math/logical_ands.h",
        "adv_api/math/logical_or.h",
        "adv_api/math/logical_ors.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LogicalAndAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogicalAndExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"logical_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  [[nodiscard]] bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &and_node) const override {
    (void)and_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/logical_and.h",
        "adv_api/math/logical_ands.h",
        "adv_api/math/logical_or.h",
        "adv_api/math/logical_ors.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class BitwiseAndAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseAndExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bitwise_and_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/bitwise_and.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class BitwiseNotAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseNotExtend";
  }

  [[nodiscard]] std::string GetMicroApiCallName() const override {
    return "MicroApiCall";
  }

  [[nodiscard]] std::string GetMicroApiName() const override {
    return "Not";
  }

  [[nodiscard]] bool IsVectorFunctionSupported(const AscNode &node) const override {
    // VF 融合路径走 MicroAPI::Not 的按位非指令，bool 输入时语义错误（~1 = 0xFE 非 0/1 值），禁止融合，
    // 回落到 UnaryApiCall（regbase BitwiseNotExtend 内部按 LogicalNot 执行）
    AscNodeInputs node_inputs = node.inputs;
    for (size_t i = 0; i < node_inputs().size(); i++) {
      if (node_inputs[i].attr.dtype == ge::DT_BOOL) {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bitwise_not_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/bitwise_not.h",
        "adv_api/math/logical_not.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }

  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    if (!HasSameSimtUnaryDtype(node)) {
      return false;
    }
    const ge::DataType dtype = GetSimtOutputDtype(node);
    return dtype == DT_INT8 || dtype == DT_INT16 || dtype == DT_INT32 || dtype == DT_INT64 || dtype == DT_UINT8 ||
           dtype == DT_UINT16 || dtype == DT_UINT32 || dtype == DT_UINT64;
  }

  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr([[maybe_unused]] const AscNode &node,
                                                       const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    expr = "~(" + inputs[0] + ")";
    return ge::GRAPH_SUCCESS;
  }
};

class BitwiseOrAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseOrExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bitwise_or_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/bitwise_or.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class BitwiseXorAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "BitwiseXorExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bitwise_xor_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/bitwise_xor.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class FloorDivAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "FloorDivExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"floor_div_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_INT8, DT_FLOAT}, {DT_INT16, DT_FLOAT}, {DT_UINT8, DT_FLOAT16}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class FloorToIntAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "FloorToIntApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "FloorToInt";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatDtype(GetSimtInputDtype(node)) && GetSimtOutputDtype(node) == DT_INT32;
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatToIntExpr(node, inputs, "CAST_FLOOR", expr);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class FmodAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "FmodExtend";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/fmod.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"fmod_reg_base.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class HypotAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Hypot";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/hypot.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LgammaAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcLgammaTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Lgamma";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Lgamma";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/lgamma.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class Log10AscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Log10";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Log10";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/log.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LogicalXorAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "LogicalXorExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"logical_xor_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/logical_xor.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class Log1pAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Log1p";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Log1p";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"log1p_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/

class PowAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcPowTmpSizeV2(node);
  }

  [[nodiscard]] std::string GetApiCallName() const override {
    return "PowApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Pow";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"pow_reg_base.h"};
  }

  [[nodiscard]] bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list;  // 支持任意输入是scalar
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "utils/std/type_traits.h",
        "adv_api/math/power.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AxpyAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAxpyTmpSize(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "AxpyApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "AxpyExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"axpy.h"};
  }
  [[nodiscard]] bool IsInplaceSupported(const AscNode &axpy_node) const override {
    (void)axpy_node;
    return true;
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MatMulAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "MatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles(bool is_dynamic) const override {
    if (is_dynamic) {
      return {"mat_mul_tiling_key_dynamic.h", "matmul_include_headers.h", "mat_mul_pingpong_basic_cmct_dynamic.h",
              "matmul_dynamic.h"};
    } else {
      return {"mat_mul_tiling_key.h", "matmul_include_headers.h", "mat_mul_pingpong_basic_cmct.h", "matmul.h"};
    }
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "utils/std/algorithm.h",
        "basic_api/kernel_operator_common_intf.h",
        "basic_api/kernel_operator_set_atomic_intf.h",
        "adv_api/matmul/matmul.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class BatchMatMulAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "BatchMatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles(bool is_dynamic) const override {
    if (is_dynamic) {
      return {"batch_mat_mul_v3_tiling_key_dynamic.h", "batch_matmul_include_headers.h",
              "mat_mul_pingpong_basic_cmct_dynamic.h", "batch_matmul_dynamic.h"};
    } else {
      return {"batch_mat_mul_v3_tiling_key.h", "batch_matmul_include_headers.h", "mat_mul_pingpong_basic_cmct.h",
              "batch_matmul.h"};
    }
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "utils/std/algorithm.h",
        "basic_api/kernel_operator_common_intf.h",
        "basic_api/kernel_operator_set_atomic_intf.h",
        "adv_api/matmul/matmul.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class Conv2DAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  std::string GetApiCallName() const override {
    return "Conv2DApiCall";
  }
  std::string GetApiName() const override {
    return "Conv2D";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"conv2d_include_headers.h", "conv2d_v2_tilingkey_cv.h", "conv_pingpong_basic_atcos.h", "conv2d.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"basic_api/kernel_operator_common_intf.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    std::string node_type = node.GetType();
    const auto &inputs = node.GetInDataNodes();
    GE_ASSERT_TRUE(!(inputs.size() > 0 && inputs.at(0)->GetType() == "Scalar"),
                   "Node %s[%s] does not support scalar input at index 0", node.GetTypePtr(), node.GetNamePtr());
    GE_ASSERT_TRUE(!(inputs.size() > 1 && inputs.at(1)->GetType() == "Scalar"),
                   "Node %s[%s] does not support scalar input at index 1", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SinAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SinExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Sin";
  }
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSinTmpSizeV2(node);
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
        {DT_FLOAT16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"sin_reg_base.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/sin.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class RShiftAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "ShiftRight";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class RoundToIntAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "RoundToIntApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "RoundToInt";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatToIntPair(node);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatToIntExpr(node, inputs, "CAST_EVEN", expr);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("cast") + std::string("_reg_base.h")};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class TruncToIntAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "TruncToIntApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "TruncToInt";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    return IsSimtFloatToIntPair(node);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    return GenerateSimtFloatToIntExpr(node, inputs, "CAST_ZERO", expr);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {std::string("cast") + std::string("_reg_base.h")};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class TruncAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcTruncTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Trunc";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Trunc";
  }
  // 如果需要插入cast节点，返回cast的目的类型
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/trunc.h",
    };
  }
};

class SinhAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSinhTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SinhExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Sinh";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"sinh_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/sinh.h",
    };
  }
};

class TanAscIrCodegenImplV2 : public SimtFloatUnaryAscIrCodegenImplV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcTanTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "TanExtend";
  }
  [[nodiscard]] std::string GetSimtScalarApiName() const override {
    return "Tan";
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
        {DT_FLOAT16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"tan_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/tan.h",
        "simt_api/cpp/kernel_simt_utils.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class XorAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Xor";
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/xor.h",
    };
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_UINT8, DT_INT16}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcXorTmpSizeV2(node);
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class TruncDivAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiCallV2";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "TruncDivExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"trunc_div_reg_base.h"};
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
        {DT_BF16, DT_FLOAT},
    };
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SquareAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "SquareApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Square";
  }
  [[nodiscard]] bool IsSimtScalarSupported(const AscNode &node) const override {
    const ge::DataType dtype = GetSimtInputDtype(node);
    return dtype == GetSimtOutputDtype(node) && (IsSimtFloatDtype(dtype) || dtype == DT_UINT8 || dtype == DT_INT16 ||
                                                 dtype == DT_UINT16 || dtype == DT_UINT32 || dtype == DT_UINT64);
  }
  [[nodiscard]] ge::graphStatus GenerateSimtScalarExpr(const AscNode &node, const std::vector<std::string> &inputs,
                                                       std::string &expr) const override {
    if (GetSimtInputDtype(node) == DT_FLOAT16 || GetSimtInputDtype(node) == DT_BF16) {
      expr = "static_cast<float>(" + inputs[0] + ") * static_cast<float>(" + inputs[0] + ")";
    } else if (GetSimtInputDtype(node) == DT_UINT16) {
      expr = "static_cast<uint32_t>(" + inputs[0] + ") * static_cast<uint32_t>(" + inputs[0] + ")";
    } else {
      expr = "(" + inputs[0] + ") * (" + inputs[0] + ")";
    }
    return ge::GRAPH_SUCCESS;
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_mul.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/reg_compute/kernel_reg_compute_intf.h",
    };
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_UINT8, DT_FLOAT16}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class RemainderAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcVoidTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "RemainderExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"remainder_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "simt_api/cpp/kernel_simt_intf.h",
        "simt_api/math_functions.h",
    };
  }
  [[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(
      const AscNode &node) override {
    std::map<ge::DataType, ge::DataType> dtype_conversion_map = {{DT_BF16, DT_FLOAT}};
    return GetConversionFromDtypeMap(node, dtype_conversion_map);
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

/*********************************************************************************/
class SoftmaxAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSoftmaxTmpSizeV2(node);
  }
  [[nodiscard]] std::string GetApiCallName() const override {
    return "SoftmaxApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "SoftmaxARFullLoadExtend";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"softmax_af_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_scalar_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

/*********************************************************************************/
class RandAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryOutputApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Rand";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"random_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"adv_api/math/philox.h"};
  }
};

/*********************************************************************************/
class RandnAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "UnaryOutputApiCall";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Rand";
  }
  [[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"random_reg_base.h"};
  }
  [[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"adv_api/math/philox.h"};
  }
};

/*********************************************************************************/
class UnsupportedAscIrCodegenImplV2 : public AscIrCodegenV2 {
 public:
  [[nodiscard]] std::string GetApiCallName() const override {
    return "";
  }
  [[nodiscard]] std::string GetApiName() const override {
    return "Unsupported";
  }
};
}  // namespace ascir
}  // namespace af

#endif  //__ASCIR_CODEGEN_IMPL__
