/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_broadcast_api_call.h"
#include "common_utils.h"
#include <limits>
#include <sstream>
#include "ascir_ops.h"
#include "common/ge_common/debug/log.h"
#include "api_call/utils/api_call_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/broadcast/broadcast_api_call.h"
#include "ascir_node_param/ascir_node_param.h"
#include "codegen/expression_convert_struct.h"
#include "reg_api_call_utils.h"

namespace codegen {
namespace semantic = ascgen_utils::template_semantic;
using namespace std;
using namespace ascgen_utils;

namespace {
constexpr const char *kAscirNodeParams = "AscirNodeParams";
constexpr const char *kCompactPddingMode = "AscendC::PaddingMode::Compact";
constexpr int64_t kDataBlockSize = 32;

std::string GetBroadcastSemanticFeature(const Tensor &input, const Tensor &output) {
  if (IsBroadcastConstantTensor(input)) {
    return semantic::kBroadcastScalar;
  }
  const size_t axis_count = std::min(input.vectorized_axis.size(), output.vectorized_axis.size());
  bool has_non_tail_broadcast = false;
  bool has_tail_broadcast = false;
  for (size_t pos = 0UL; pos < axis_count; ++pos) {
    const auto input_size = input.axis_size[input.vectorized_axis_pos[pos]];
    const auto output_size = output.axis_size[output.vectorized_axis_pos[pos]];
    if (af::SymbolicUtils::StaticCheckEq(input_size, output_size) == af::TriBool::kTrue) {
      continue;
    }
    if (pos + 1UL == axis_count) {
      has_tail_broadcast = true;
    } else {
      has_non_tail_broadcast = true;
    }
  }
  if (has_non_tail_broadcast) {
    return semantic::kBroadcastNonTail;
  }
  return has_tail_broadcast ? semantic::kBroadcastTail : semantic::kBroadcast;
}

af::Status ResetBroadcastNodeParams(const af::AscNodePtr &node, ascir_param::BroadcastNodeParams *&broadcast_params) {
  GE_ASSERT_NOTNULL(node);
  auto params = ascir_param::GetAscirNodeParams(node);
  if (params == nullptr) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    params = std::make_shared<ascir_param::AscirNodeParams>();
    GE_ASSERT_TRUE(op_desc->SetExtAttr(kAscirNodeParams, params), "Node:%s SetExtAttr failed", node->GetNamePtr());
  }
  auto *specific_params = std::get_if<ascir_param::BroadcastNodeParams>(&params->specific_params);
  if (specific_params == nullptr) {
    params->specific_params = ascir_param::BroadcastNodeParams{};
    specific_params = std::get_if<ascir_param::BroadcastNodeParams>(&params->specific_params);
  }
  GE_ASSERT_NOTNULL(specific_params, "Broadcast specific params is null, node[%s].", node->GetNamePtr());
  params->api_name = node->GetType();
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  *specific_params = ascir_param::BroadcastNodeParams{};
  broadcast_params = specific_params;
  return af::SUCCESS;
}

std::vector<ascir_param::ParamExprLeaf> GetBroadcastShape(const TPipe &tpipe, const Tensor &input, const Tensor &output,
                                                          bool is_src, bool is_compact_padding) {
  std::vector<ascir_param::ParamExprLeaf> shape;
  const auto axis_count = input.vectorized_axis.size();
  shape.reserve(axis_count);
  for (size_t pos = 0UL; pos < axis_count; ++pos) {
    if (is_src &&
        af::SymbolicUtils::StaticCheckEq(input.axis_size[input.vectorized_axis_pos[pos]],
                                         output.axis_size[output.vectorized_axis_pos[pos]]) != af::TriBool::kTrue) {
      shape.push_back({af::Symbol(1U), ascir_param::ParamExprRole::kSemantic});
      continue;
    }
    const auto axis_id = output.vectorized_axis[pos];
    auto role = tpipe.tiler.GetAxis(axis_id).type != ascir::Axis::Type::kAxisTypeTileInner ||
                        output.vectorized_axis[0] == axis_id
                    ? ascir_param::ParamExprRole::kActualSize
                    : ascir_param::ParamExprRole::kSize;
    if (pos != axis_count - 1UL) {
      shape.push_back({output.axis_size[output.vectorized_axis_pos[pos]], role});
      continue;
    }
    size_t pre_pos = std::numeric_limits<size_t>::max();
    for (size_t i = 0UL; i < pos; ++i) {
      if (output.vectorized_strides[i] != 0) {
        pre_pos = i;
      }
    }
    const auto expr = pre_pos == std::numeric_limits<size_t>::max() ? output.axis_size[output.vectorized_axis_pos[pos]]
                                                                    : output.vectorized_strides[pre_pos];
    if (pre_pos != std::numeric_limits<size_t>::max() && is_compact_padding) {
      role = ascir_param::ParamExprRole::kActualSize;
    }
    shape.push_back({expr, role});
  }
  return shape;
}

af::Status GetBroadcastScalarDuplicateCount(const TPipe &tpipe, const Tensor &output, af::Expression &count) {
  const auto parse_tiling_expr = [](std::string value) {
    size_t pos = 0UL;
    while ((pos = value.find("t->", pos)) != std::string::npos) {
      value.erase(pos, 3UL);
    }
    return af::Expression::Parse(value.c_str());
  };
  count = af::Expression::Parse("0");
  if (output.vectorized_strides.size() != output.vectorized_axis.size()) {
    count = af::Expression::Parse("1");
    for (const auto axis_pos : output.vectorized_axis_pos) {
      const auto axis_size_str = tpipe.tiler.Size(output.axis_size[axis_pos]);
      const auto axis_size = parse_tiling_expr(axis_size_str);
      GE_ASSERT_TRUE(axis_size.IsValid(), "Broadcast scalar output axis size is invalid: %s", axis_size_str.c_str());
      count = af::sym::Mul(count, axis_size);
    }
    return af::SUCCESS;
  }
  for (size_t i = 0UL; i < output.vectorized_axis.size(); ++i) {
    if (output.vectorized_strides[i] == 0) {
      continue;
    }
    const auto axis_pos = output.vectorized_axis_pos[i];
    const auto axis_size_str = tpipe.tiler.Size(output.axis_size[axis_pos]);
    const auto axis_size = parse_tiling_expr(axis_size_str);
    GE_ASSERT_TRUE(axis_size.IsValid(), "Broadcast scalar output axis size is invalid: %s", axis_size_str.c_str());
    auto term = af::sym::Sub(axis_size, af::Expression::Parse("1"));
    if (output.vectorized_strides[i] != 1) {
      const auto stride = parse_tiling_expr(tpipe.tiler.Size(output.vectorized_strides[i]));
      GE_ASSERT_TRUE(stride.IsValid(), "Broadcast scalar output stride is invalid.");
      term = af::sym::Mul(term, stride);
    }
    count = af::sym::Add(count, term);
  }
  count = af::sym::Add(count, af::Expression::Parse("1"));
  return af::SUCCESS;
}

// 尾轴广播时，广播源的内层可能被对齐成“每元素独占一个 data block”（见尾轴广播对齐策略），
// 源在 UB 上的 stride 不再是按 shape 连续反推的值。此时代码生成需以 srcInnerPad 实例化
// BroadcastExtend，使其按 ONE_BLK_SIZE / sizeof(T) 反推源 stride，避免源取址错位。
bool NeedTailBroadcastSrcInnerPad(const Tensor &x, const Tensor &y) {
  const size_t rank = x.vectorized_axis.size();
  if (rank < 2UL || x.vectorized_axis_pos.size() != rank || x.vectorized_strides.size() < rank ||
      y.vectorized_axis_pos.size() != y.vectorized_axis.size()) {
    return false;
  }
  // 尾轴必须是广播轴：源尾轴 stride 为 0、源内层为 1 且输出不为 1
  if (af::SymbolicUtils::StaticCheckEq(x.vectorized_strides.back(), af::sym::kSymbolZero) != af::TriBool::kTrue) {
    return false;
  }
  if (af::SymbolicUtils::StaticCheckEq(x.axis_size[x.vectorized_axis_pos.back()], af::sym::kSymbolOne) !=
          af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(y.axis_size[y.vectorized_axis_pos.back()], af::sym::kSymbolOne) ==
          af::TriBool::kTrue) {
    return false;
  }
  const int64_t dtype_size = static_cast<int64_t>(ge::GetSizeByDataType(x.dtype));
  if (dtype_size <= 0) {
    return false;
  }
  // 8 字节类型在 GetBroadcastTilingInfoExtendImpl 中会被 ×2 预处理改写尾轴布局，
  // srcInnerPad 语义不同（正确基准不是 ONE_BLK_SIZE / sizeof(T)）。当前仅支持
  // 1/2/4 字节，与 device 侧保持一致，避免生成 srcInnerPad=true 却被静默忽略。
  if (dtype_size >= static_cast<int64_t>(sizeof(uint64_t))) {
    return false;
  }
  const int64_t block_align = kDataBlockSize / dtype_size;
  // 源内层最近一个非零 stride 应为 32B 对齐块（每元素独占一个 data block）
  for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
    if (af::SymbolicUtils::StaticCheckEq(x.vectorized_strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;
    }
    return af::SymbolicUtils::StaticCheckEq(x.vectorized_strides[i], af::Symbol(block_align)) == af::TriBool::kTrue;
  }
  return false;
}

// 校验广播输入/输出的向量化轴数量：均在 [1, 9] 且两侧一致。
af::Status ValidateBroadcastVecAxisSizes(const Tensor &x, const Tensor &y) {
  constexpr size_t kMinVectorizedAxisSize = 1UL;
  constexpr size_t kMaxVectorizedAxisSize = 9UL;
  if (x.vectorized_axis.size() < kMinVectorizedAxisSize || x.vectorized_axis.size() > kMaxVectorizedAxisSize) {
    GELOGE(af::FAILED, "Codegen broadcast input vec axis size[%zu] is either 0 or greater than 9",
           x.vectorized_axis.size());
    return af::FAILED;
  }
  if (y.vectorized_axis.size() < kMinVectorizedAxisSize || y.vectorized_axis.size() > kMaxVectorizedAxisSize) {
    GELOGE(af::FAILED, "Codegen broadcast output vec axis size[%zu] is either 0 or greater than 9",
           y.vectorized_axis.size());
    return af::FAILED;
  }
  if (x.vectorized_axis.size() != y.vectorized_axis.size()) {
    GELOGE(af::FAILED, "Codegen broadcast input vec axis size[%zu] not equal output vec axis size[%zu]",
           x.vectorized_axis.size(), y.vectorized_axis.size());
    return af::FAILED;
  }
  return af::SUCCESS;
}

// 常量（标量）广播：生成 Duplicate 调用并登记 scalar 相关参数。
af::Status GenerateScalarBroadcast(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &x,
                                   const Tensor &y, const af::AscNodePtr &node, std::string &result) {
  int64_t id = -1;
  BroadcastScalar(tpipe, current_axis, x, y, id, result, false);
  ascir_param::BroadcastNodeParams *broadcast_params = nullptr;
  if (ResetBroadcastNodeParams(node, broadcast_params) == af::SUCCESS && broadcast_params != nullptr) {
    broadcast_params->valid = true;
    broadcast_params->is_scalar = true;
    broadcast_params->const_rank = -1;
    GE_ASSERT_SUCCESS(GetBroadcastScalarDuplicateCount(tpipe, y, broadcast_params->duplicate_count));
    broadcast_params->duplicate_count_role = ascir_param::ParamExprRole::kActualSize;
  }
  return af::SUCCESS;
}

// 发射 BroadcastExtend 调用：模板形参 + dst/src 偏移 + 形状数组名。
void GenBroadcastCall(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &x,
                      const Tensor &y, const std::string &api_name, std::stringstream &ss) {
  std::string dtype_name;
  Tensor::DtypeName(x.dtype, dtype_name);
  ss << api_name << "<" << dtype_name << "," << x.vectorized_axis.size();
  if (NeedTailBroadcastSrcInnerPad(x, y)) {
    ss << ",true";
  }
  ss << ">(";
  // 传入参数  const LocalTensor<T> &dst;
  ss << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], ";
  // 传入参数  const LocalTensor<T> &src;
  ss << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], ";
  ss << "dst_shape_" << x.id << "_brc_to_" << y.id << ", src_shape_" << x.id << "_brc_to_" << y.id << ");\n";
}

// 登记 ascir 参数缓存中的广播 dst/src shape。
void RegisterBroadcastNodeParams(const TPipe &tpipe, const af::AscNodePtr &node, const Tensor &x, const Tensor &y,
                                 bool is_compact_padding) {
  ascir_param::BroadcastNodeParams *broadcast_params = nullptr;
  if (ResetBroadcastNodeParams(node, broadcast_params) == af::SUCCESS && broadcast_params != nullptr) {
    broadcast_params->valid = true;
    broadcast_params->const_rank = static_cast<int32_t>(x.vectorized_axis.size());
    broadcast_params->dst_shape = GetBroadcastShape(tpipe, x, y, false, is_compact_padding);
    broadcast_params->src_shape = GetBroadcastShape(tpipe, x, y, true, is_compact_padding);
  }
}

}  // namespace

static void GenParams(const TPipe &tpipe, const Tensor &input, const Tensor &output, std::stringstream &ss, bool is_src,
                      bool is_compact_padding) {
  // 只保证在仅对张量尾轴做32B对齐的场景下有效，若对中间轴做了对齐，则还需要增加处理逻辑
  auto vectorized_axis_size = input.vectorized_axis.size();
  const char *shape_prefix = is_src ? "src_shape_" : "dst_shape_";
  ss << "const uint32_t " << shape_prefix << input.id << "_brc_to_" << output.id << "[" << vectorized_axis_size
     << "] = {";
  const char *sep = "";
  for (size_t pos = 0UL; pos < vectorized_axis_size; pos++) {
    ss << sep;
    sep = ", ";
    ss << "static_cast<uint32_t>(";
    // 处理输入广播轴
    if (is_src) {
      auto input_repeat = input.axis_size[input.vectorized_axis_pos[pos]];
      auto output_repeat = output.axis_size[output.vectorized_axis_pos[pos]];
      if (af::SymbolicUtils::StaticCheckEq(input_repeat, output_repeat) != af::TriBool::kTrue) {
        ss << "1)";
        continue;
      }
    }
    if (pos != vectorized_axis_size - 1UL) {
      GetOneAxisSize(tpipe, output, pos, ss);
      ss << ")";
      continue;
    }
    // 找到最近一个stride非0的轴
    size_t pre_pos = std::numeric_limits<size_t>::max();
    for (size_t i = 0UL; i < pos; ++i) {
      if (output.vectorized_strides[i] != 0) {
        pre_pos = i;
      }
    }
    if (pre_pos == std::numeric_limits<size_t>::max()) {
      GetOneAxisSize(tpipe, output, pos, ss);
      ss << ")";
      continue;
    }
    ascir::AxisId axis_id = output.vectorized_axis[pos];
    auto last_dim_size = output.vectorized_strides[pre_pos];
    if (tpipe.tiler.GetAxis(axis_id).type != ascir::Axis::Type::kAxisTypeTileInner ||
        output.vectorized_axis[0] == axis_id || is_compact_padding) {
      ss << tpipe.tiler.ActualSize(last_dim_size);
    } else {
      ss << tpipe.tiler.Size(last_dim_size);
    }
    ss << ")";
  }
  ss << "};\n";
}

Status BroadcastRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                     const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                     const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                     std::string &result) const {
  const auto &x = inputs[0].get();
  const auto &y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  GE_ASSERT_SUCCESS(SetSemanticFeature(GetBroadcastSemanticFeature(x, y)));
  if (IsBroadcastConstantTensor(x)) {
    return GenerateScalarBroadcast(tpipe, current_axis, x, y, this->node, result);
  }
  const af::Status size_status = ValidateBroadcastVecAxisSizes(x, y);
  if (size_status != af::SUCCESS) {
    return size_status;
  }

  std::stringstream ss;
  DataCopyParams data_copy_param;
  (void)CalculateDmaParams(tpipe, y, y, data_copy_param);
  const auto has_transpose = IsGraphHasTransposeNode(this->node);
  const bool is_compact_padding = GetPaddingMode(y, data_copy_param, has_transpose) == kCompactPddingMode;
  // 生成参数 const uint32_t *dst_shape;
  GenParams(tpipe, x, y, ss, false, is_compact_padding);
  // 生成参数 const uint32_t *src_shape;
  GenParams(tpipe, x, y, ss, true, is_compact_padding);
  GenBroadcastCall(tpipe, current_axis, x, y, this->api_name_, ss);

  result = ss.str();
  RegisterBroadcastNodeParams(tpipe, this->node, x, y, is_compact_padding);
  return af::SUCCESS;
}

static ApiCallRegister<BroadcastRegApiCall> register_broadcast_reg_api_call("BroadcastRegApiCall");
}  // namespace codegen
