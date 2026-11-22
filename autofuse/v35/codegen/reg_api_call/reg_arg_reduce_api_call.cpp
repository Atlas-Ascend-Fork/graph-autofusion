/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_arg_reduce_api_call.h"

#include "codegen/api_call/reduce/reduce_api_call_base.h"
#include <algorithm>
#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "arg_reduce_utils.h"
#include "common/ge_common/debug/log.h"
#include "common/checker.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"
#include "reg_api_call_utils.h"
#include "codegen_api_param/codegen_api_param.h"
#include "ascir_node_param/ascir_node_param.h"

namespace codegen {
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;
using namespace reduce_base;

namespace {
std::string ArgReduceApiName(const std::string &type, const char *suffix) {
  return std::string(IsArgMinType(type) ? "ArgMin" : "ArgMax") + suffix;
}

const char *ArgReduceDirection(const std::string &type) {
  return IsArgMinType(type) ? "false" : "true";
}

// 注入属性供ATT性能建模链路消费
Status FillArgReduceNodeParams(const af::AscNodePtr &node, const TPipe &tpipe, const Tensor &src, const Tensor &dst,
                               const Tensor &index_dst, bool phase2, bool with_value, bool tile_merge,
                               const Tensor *index_src = nullptr, bool accumulate = false,
                               const ge::Expression &physical_last = Zero, const ge::Expression &valid_a = Zero) {
  GE_ASSERT_NOTNULL(node);
  auto params = ascir_param::GetOrCreateAscirNodeParams(node);
  GE_ASSERT_NOTNULL(params);
  ascir_param::ArgReduceNodeParams arg_reduce;
  arg_reduce.is_ar = phase2 ? dst.vectorized_strides.back() == Zero : src.isAr;
  arg_reduce.is_phase2 = phase2;
  arg_reduce.with_value = with_value;
  arg_reduce.has_tile_merge = tile_merge;
  arg_reduce.has_index_cast = index_dst.dtype != ge::DT_INT32;
  arg_reduce.has_accumulate = accumulate;
  arg_reduce.valid_a = valid_a;
  std::vector<ge::Expression> repeats;
  for (const auto axis : src.vectorized_axis) {
    // Axis::size is the original tiling expression, unlike AxisSize()/actual_size.Str().
    repeats.emplace_back(tpipe.tiler.GetAxis(axis).size);
  }
  auto shape =
      BuildReduceMergedShape(repeats, src.vectorized_strides, dst.vectorized_strides, ge::GetSizeByDataType(src.dtype));
  GE_ASSERT_TRUE(shape.valid, "Invalid ArgReduce merged shape, node[%s].", node->GetNamePtr());
  if (physical_last != Zero) {
    shape.last = physical_last;
  }
  arg_reduce.src_shape = {shape.first, shape.last};
  if (phase2) {
    GE_ASSERT_NOTNULL(index_src);
    const auto index_shape = BuildReduceMergedShape(repeats, index_src->vectorized_strides, dst.vectorized_strides,
                                                    ge::GetSizeByDataType(index_src->dtype));
    GE_ASSERT_TRUE(index_shape.valid);
    arg_reduce.src_shape.emplace_back(index_shape.last);
    arg_reduce.valid_extent = repeats.back();
  } else if (!arg_reduce.is_ar) {
    arg_reduce.valid_extent = shape.first;
  } else {
    arg_reduce.valid_extent = One;
    GE_ASSERT_TRUE(src.vectorized_strides.size() == dst.vectorized_strides.size());
    for (size_t i = 0; i < repeats.size(); ++i) {
      if (dst.vectorized_strides[i] == Zero && src.vectorized_strides[i] != Zero) {
        arg_reduce.valid_extent = arg_reduce.valid_extent * repeats[i];
      }
    }
  }
  arg_reduce.valid = true;
  params->api_name = node->GetType();
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  params->specific_params = arg_reduce;
  return af::SUCCESS;
}

// ArgReduce 内部 index 统一为 int32；面向前端的输出若为 int64，在生成 kernel 的出口处
// 用 CastExtend 做一次 int32→int64 转换。
constexpr const char *kArgReduceIndexDtype = "int32_t";

bool NeedIndexCast(const Tensor &index) {
  return index.dtype != ge::DT_INT32;
}

// 生成 ArgReduce 内部 int32 index → 前端 index 输出的转换。
void EmitIndexCastToFrontend(const Tensor &y, const std::string &dst_expr, const std::string &src_i32,
                             std::vector<std::string> &post_process) {
  std::string dst_dtype_name;
  GE_CHK_STATUS(Tensor::DtypeName(y.dtype, dst_dtype_name), "Codegen get index dtype:%d failed",
                static_cast<int32_t>(y.dtype));
  post_process.emplace_back("{\n");
  post_process.emplace_back("  uint32_t arg_reduce_cast_dims[] = {" + y.actual_size.Str() + "};\n");
  post_process.emplace_back("  CastExtend<" + std::string(kArgReduceIndexDtype) + ", " + dst_dtype_name + ", 1>(" +
                            dst_expr + ", " + src_i32 +
                            ", arg_reduce_cast_dims, arg_reduce_cast_dims, arg_reduce_cast_dims);\n");
  post_process.emplace_back("}\n");
}

// index_src 非空时（Phase2 双输入：value(T) + index(int32)），额外按 index dtype
// 计算其末轴 32B 对齐行跨度 last_tail：value/index 每行元素数可能不同（如 int64 4 vs int32 8），
// Phase2 需分别寻址，故 tmp_reduce_shape = {first_actual, last, last_tail}。
void BuildArgReducePreProcess(const TPipe &tpipe, const Tensor &x, const Tensor &y,
                              std::vector<std::string> &pre_process, const Tensor *index_src = nullptr,
                              const ge::Expression &physical_last = Zero) {
  ReduceMergedSizeCodeGen(tpipe, pre_process, x, y);
  if (physical_last != Zero) {
    for (auto &line : pre_process) {
      if (line.find("uint32_t last = ") == 0U) {
        line = "uint32_t last = " + tpipe.tiler.Size(physical_last) + ";\n";
      } else if (line.find("uint32_t last_actual = ") == 0U) {
        line = "uint32_t last_actual = last;\n";
      }
    }
  }
  if (index_src == nullptr) {
    pre_process.emplace_back("uint32_t tmp_reduce_shape[] = {first_actual, last};\n");
    return;
  }
  // 复用 ReduceMergedSizeCodeGen 的尾轴对齐逻辑，用 index tensor 的 dtype 得到 last_tail。
  // is_tail=true 使声明名带 _tail 后缀，避免与上面的 first/last 冲突；其首行 "{" 需跳过。
  std::vector<std::string> index_lines;
  ReduceMergedSizeCodeGen(tpipe, index_lines, *index_src, y, true);
  for (size_t i = 1UL; i < index_lines.size(); ++i) {
    pre_process.emplace_back(index_lines[i]);
  }
  pre_process.emplace_back("uint32_t tmp_reduce_shape[] = {first_actual, last, last_tail};\n");
}

void DeclareIndexTmp(const TPipe &tpipe, const std::unordered_map<int64_t, int64_t> &tmp_buf_id,
                     const std::string &name, int64_t life_time, std::vector<std::string> &pre_process) {
  auto it = tmp_buf_id.find(life_time);
  if (it == tmp_buf_id.end()) {
    return;
  }
  pre_process.emplace_back("LocalTensor<int32_t> " + name + ";\n");
  pre_process.emplace_back(name + " = " + tpipe.tmp_buf.name + "_" + std::to_string(it->second) +
                           ".ReinterpretCast<int32_t>();\n");
}

void EmitArRoundMerge(const Tensor &y, const std::string &value_dtype_name, const std::string &loop_axis,
                      const std::string &api_name, std::vector<std::string> &post) {
  post.emplace_back("if (" + loop_axis + " == 0) {\n");
  post.emplace_back("ArgReduceCopy(tmp_arg_reduce_index_saved[0], tmp_arg_reduce_index[0], " + y.actual_size.Str() +
                    ");\n");
  post.emplace_back("ArgReduceCopy(tmp_arg_reduce_value_saved[0], tmp_arg_reduce_value[0], " + y.actual_size.Str() +
                    ");\n");
  post.emplace_back("} else {\n");
  post.emplace_back("ArgReduceMergeRound<" + value_dtype_name + ", " + ArgReduceDirection(api_name) +
                    ">(tmp_arg_reduce_value_saved[0], tmp_arg_reduce_index_saved[0], "
                    "tmp_arg_reduce_value[0], tmp_arg_reduce_index[0], " +
                    y.actual_size.Str() + ");\n");
  post.emplace_back("}\n");
}

Status EmitMultiRoundIndexAndAccumulate(const Tensor &index_out, bool need_cast, bool defer_index_output,
                                        const std::string &loop_axis, const std::string &loop_size,
                                        const std::string &accum_step, std::vector<std::string> &post) {
  if (defer_index_output) {
    post.emplace_back("if (" + loop_axis + " + 1 == " + loop_size + ") {\n");
  }
  if (need_cast) {
    EmitIndexCastToFrontend(index_out, index_out.Str() + "[0]", "tmp_arg_reduce_index_saved[0]", post);
  } else {
    post.emplace_back("ArgReduceCopy(" + index_out.Str() + "[0], tmp_arg_reduce_index_saved[0], " +
                      index_out.actual_size.Str() + ");\n");
  }
  if (defer_index_output) {
    post.emplace_back("}\n");
  }
  post.emplace_back("accumulated_offset += " + accum_step + ";\n");
  return af::SUCCESS;
}

std::string GetArgReduceValidRExpr(const TPipe &tpipe, const Tensor &src, const Tensor &dst) {
  if (!src.isAr) {
    return "first_actual";
  }
  std::string valid_r;
  const size_t axis_count = std::min(src.vectorized_strides.size(), dst.vectorized_strides.size());
  for (size_t i = 0; i < axis_count; ++i) {
    if (dst.vectorized_strides[i] == 0 && src.vectorized_strides[i] != 0) {
      const auto &axis = tpipe.tiler.GetAxis(src.vectorized_axis[i]);
      const auto actual_size = axis.size == One ? "1" : axis.actual_size.Str();
      valid_r = valid_r.empty() ? actual_size : "(" + valid_r + ") * (" + actual_size + ")";
    }
  }
  return valid_r.empty() ? "1" : valid_r;
}

ge::Expression PrepareArgReduceLayout(const Tiler &tiler, Tensor &src, Tensor &dst, std::string &pattern) {
  ge::Expression physical_last = Zero;
  size_t axes = src.vectorized_axis.size();
  while (axes > 0U &&
         IsReduceMergedZeroTail(src.vectorized_strides[axes - 1U] == Zero, dst.vectorized_strides[axes - 1U] == Zero)) {
    --axes;
  }
  if (axes < src.vectorized_axis.size()) {
    bool unit_tail = true;
    for (size_t i = axes; i < src.vectorized_axis.size(); ++i) {
      unit_tail = unit_tail && tiler.GetAxis(src.vectorized_axis[i]).size == One;
    }
    if (unit_tail) {
      if (axes == 0U || dst.vectorized_strides[axes - 1U] != Zero) {
        physical_last = axes == 0U ? One : src.vectorized_strides[axes - 1U];
        src.vectorized_strides.back() = One;
      } else if (src.vectorized_strides[axes - 1U] != One) {
        // Scalar RA: keep the physical A pitch and limit the API to one valid output column.
        physical_last = src.vectorized_strides[axes - 1U];
        src.vectorized_strides.back() = One;
        dst.vectorized_strides.back() = One;
      } else {
        src.vectorized_axis.resize(axes);
        src.vectorized_strides.resize(axes);
        dst.vectorized_axis.resize(axes);
        dst.vectorized_strides.resize(axes);
      }
    }
  }
  GetIsArAndPattern(src, dst, src.isAr, pattern);
  return physical_last;
}

Status GetArgReducePhase1TileExpr(const Tiler &tiler, const std::vector<ascir::AxisId> &current_axis, const Tensor &src,
                                  const Tensor &dst, std::string &index_base, std::string &loop_index) {
  index_base = "0";
  loop_index = "0";
  GE_ASSERT_TRUE(src.vectorized_axis.size() == src.vectorized_strides.size());
  GE_ASSERT_TRUE(src.vectorized_strides.size() == dst.vectorized_strides.size());
  bool has_reduction = false;
  for (size_t i = 0; i < src.vectorized_axis.size(); ++i) {
    // Zero strides on both sides describe a degenerate axis, including a removed R=1 loop.
    if (src.vectorized_strides[i] == Zero || dst.vectorized_strides[i] != Zero) {
      continue;
    }
    const auto &r_axis = tiler.GetAxis(src.vectorized_axis[i]);
    const bool is_tile_inner = r_axis.type == ascir::Axis::Type::kAxisTypeTileInner;
    GE_ASSERT_TRUE(!is_tile_inner || !r_axis.from.empty());
    const auto origin = is_tile_inner ? r_axis.from[0] : r_axis.id;
    const auto &origin_axis = tiler.GetAxis(origin);
    if (origin_axis.type == ascir::Axis::Type::kAxisTypeOriginal && origin_axis.size == One) {
      continue;
    }
    has_reduction = true;
    // Match the actual reduction lineage against loops enclosing this call, not the AR/RA label.
    for (auto it = current_axis.rbegin(); it != current_axis.rend(); ++it) {
      const auto &axis = tiler.GetAxis(*it);
      if (axis.type != ascir::Axis::Type::kAxisTypeBlockInner || !tiler.IsFrom(axis.id, origin)) {
        continue;
      }
      GE_ASSERT_TRUE(!axis.from.empty());
      index_base = tiler.Offset(current_axis, {axis.from[0]}, {r_axis.size});
      loop_index = axis.Str();
      return af::SUCCESS;
    }
  }
  GE_ASSERT_TRUE(!has_reduction, "ArgMaxMultiRPhase1 cannot locate an enclosing R axis block-inner axis.");
  return af::SUCCESS;
}
}  // namespace

Status RegArgReduceApiCall::BuildSingleReduceParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                                   const Tensor &y, bool need_cast, CodegenApiParam &api_param) const {
  const std::string y_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y);
  std::string index_dst = y.Str();
  if (need_cast) {
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index", 0L, api_param.api_pre_process);
    index_dst = "tmp_arg_reduce_index";
  }
  api_param.api_name = ArgReduceApiName(this->api_name_, "Extend");
  api_param.output_params.emplace_back(
      index_dst, true, need_cast ? CombinedExprFactory::Constant(0) : CombinedExprFactory::SymbolVar(y_offset));
  if (need_cast) {
    EmitIndexCastToFrontend(y, y.Str() + "[" + y_offset + "]", "tmp_arg_reduce_index[0]", api_param.api_post_process);
  }
  return af::SUCCESS;
}

Status RegArgReduceApiCall::BuildMultiReducePreProcess(const TPipe &tpipe, const std::string &value_dtype,
                                                       const std::string &loop_axis, CodegenApiParam &api_param) const {
  // State spans tiles, so the first tile resets offsets left by any previous kernel launch.
  api_param.api_pre_process.emplace_back("static int64_t accumulated_offset = 0;\n");
  api_param.api_pre_process.emplace_back("if (" + loop_axis + " == 0) {\n");
  api_param.api_pre_process.emplace_back("  accumulated_offset = 0;\n");
  api_param.api_pre_process.emplace_back("}\n");
  DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index", 0L, api_param.api_pre_process);
  DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index_saved", 3L, api_param.api_pre_process);
  auto value_tmp_it = this->tmp_buf_id.find(1L);
  auto saved_tmp_it = this->tmp_buf_id.find(2L);
  GE_ASSERT_TRUE(value_tmp_it != this->tmp_buf_id.end() && saved_tmp_it != this->tmp_buf_id.end());
  api_param.api_pre_process.emplace_back("LocalTensor<" + value_dtype + "> tmp_arg_reduce_value;\n");
  api_param.api_pre_process.emplace_back("tmp_arg_reduce_value = " + tpipe.tmp_buf.name + "_" +
                                         std::to_string(value_tmp_it->second) + ".ReinterpretCast<" + value_dtype +
                                         ">();\n");
  api_param.api_pre_process.emplace_back("LocalTensor<" + value_dtype + "> tmp_arg_reduce_value_saved;\n");
  api_param.api_pre_process.emplace_back("tmp_arg_reduce_value_saved = " + tpipe.tmp_buf.name + "_" +
                                         std::to_string(saved_tmp_it->second) + ".ReinterpretCast<" + value_dtype +
                                         ">();\n");
  return af::SUCCESS;
}

Status RegArgReduceApiCall::BuildMultiReduceParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                                  const Tensor &x, const Tensor &y, const std::string &value_dtype,
                                                  bool need_cast, CodegenApiParam &api_param) const {
  const auto &reduce_axis = tpipe.tiler.GetAxis(current_axis.back());
  const std::string loop_axis = reduce_axis.Str();
  GE_ASSERT_SUCCESS(BuildMultiReducePreProcess(tpipe, value_dtype, loop_axis, api_param));
  api_param.api_name = ArgReduceApiName(this->api_name_, x.isAr ? "WithValueExtend" : "WithValueAccumulateExtend");
  api_param.output_params.emplace_back(x.isAr ? "tmp_arg_reduce_index" : "tmp_arg_reduce_index_saved", true,
                                       CombinedExprFactory::Constant(0));
  api_param.output_params.emplace_back(x.isAr ? "tmp_arg_reduce_value" : "tmp_arg_reduce_value_saved", true,
                                       CombinedExprFactory::Constant(0));
  if (!x.isAr) {
    api_param.api_pre_process.emplace_back("bool arg_reduce_first_tile = (" + loop_axis + " == 0);\n");
    api_param.input_params.emplace_back("tmp_arg_reduce_index", true, CombinedExprFactory::Constant(0));
    api_param.input_params.emplace_back("tmp_arg_reduce_value", true, CombinedExprFactory::Constant(0));
  }
  if (x.isAr) {
    EmitArRoundMerge(y, value_dtype, loop_axis, this->api_name_, api_param.api_post_process);
  }
  const std::string accum_step =
      x.isAr ? tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size.Str() : "first_actual";
  ascir::AxisId consumer_axis = af::kIdNone;
  GE_CHK_STATUS_RET(
      EmitMultiRoundIndexAndAccumulate(y, need_cast, IsReadOutersideWrite(consumer_axis), loop_axis,
                                       reduce_axis.loop_size.Str(), accum_step, api_param.api_post_process),
      "Emit ArgReduce multi-round output failed.");
  return af::SUCCESS;
}

Status RegArgReduceApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                          const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                          const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();

  std::string reduce_pattern;
  const auto physical_last = PrepareArgReduceLayout(tpipe.tiler, x, y, reduce_pattern);
  has_valid_a_ = physical_last != Zero && !x.isAr;
  valid_r_expr_ = GetArgReduceValidRExpr(tpipe, x, y);

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, x, y, value_dtype_name),
                    "GetDtypeNameForReduce failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgReducePreProcess(tpipe, x, y, api_param->api_pre_process, nullptr, physical_last);

  const bool need_multi_reduce = IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back());
  const bool need_cast = NeedIndexCast(y);
  GE_ASSERT_SUCCESS(FillArgReduceNodeParams(this->node, tpipe, x, y, y, false, need_multi_reduce, need_multi_reduce,
                                            nullptr, need_multi_reduce && !x.isAr, physical_last,
                                            has_valid_a_ ? One : Zero));

  api_param->template_params = {value_dtype_name, reduce_pattern};
  api_param->input_params.emplace_back(
      x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");
  if (!need_multi_reduce) {
    GE_ASSERT_SUCCESS(BuildSingleReduceParam(tpipe, current_axis, y, need_cast, *api_param));
  } else {
    GE_ASSERT_SUCCESS(BuildMultiReduceParam(tpipe, current_axis, x, y, value_dtype_name, need_cast, *api_param));
  }

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  api_param->semantic_feature = std::string(x.isAr ? template_semantic::kAr : template_semantic::kRa) +
                                (need_multi_reduce ? "/" + std::string(template_semantic::kMultiRound) : "");
  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgReduceApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                              std::stringstream &ss) const {
  if (api_param.api_name == ArgReduceApiName(this->api_name_, "WithValueAccumulateExtend")) {
    ss << api_param.cal_count.ToStr(tiler) << ", accumulated_offset, " << valid_r_expr_ << ", arg_reduce_first_tile"
       << (has_valid_a_ ? ", 1" : "") << ");" << std::endl;
  } else if (api_param.api_name == ArgReduceApiName(this->api_name_, "WithValueExtend")) {
    // multi_reduce 场景：传递 accumulated_offset 作为 indexBase（index 全局化）
    ss << api_param.cal_count.ToStr(tiler) << ", false, accumulated_offset, " << valid_r_expr_
       << (has_valid_a_ ? ", 1" : "") << ");" << std::endl;
  } else {
    // validR 传入实际有效 R 长度，屏蔽 tail padding lane
    ss << api_param.cal_count.ToStr(tiler) << ", false, " << valid_r_expr_ << (has_valid_a_ ? ", 1" : "") << ");"
       << std::endl;
  }
  return af::SUCCESS;
}

Status RegArgReducePhase1ApiCall::BuildTileState(const TPipe &tpipe, const std::string &value_dtype,
                                                 const std::string &index_base, const std::string &loop_axis,
                                                 CodegenApiParam &api_param) const {
  api_param.api_pre_process.emplace_back("uint32_t arg_reduce_index_base = " + index_base + ";\n");
  api_param.api_pre_process.emplace_back("bool arg_reduce_first_tile = (" + loop_axis + " == 0);\n");
  DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index", 0L, api_param.api_pre_process);
  const auto value_tmp_it = this->tmp_buf_id.find(1L);
  GE_ASSERT_TRUE(value_tmp_it != this->tmp_buf_id.end(), "Arg-reduce Phase1 requires index/value temporary buffers.");
  api_param.api_pre_process.emplace_back("LocalTensor<" + value_dtype + "> tmp_arg_reduce_value;\n");
  api_param.api_pre_process.emplace_back("tmp_arg_reduce_value = " + tpipe.tmp_buf.name + "_" +
                                         std::to_string(value_tmp_it->second) + ".ReinterpretCast<" + value_dtype +
                                         ">();\n");
  return af::SUCCESS;
}

Status RegArgReducePhase1ApiCall::BuildApiParam(
    const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
    const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  GE_ASSERT_TRUE(outputs.size() >= 2UL, "ArgMaxMultiRPhase1 requires at least 2 outputs.");
  auto x = inputs[0].get();
  auto y_value = outputs[0].get();
  auto y_index = outputs[1].get();
  GE_ASSERT_TRUE(y_index.dtype == ge::DT_INT32, "ArgMaxMultiRPhase1 index output must be int32");

  std::string reduce_pattern;
  const auto physical_last = PrepareArgReduceLayout(tpipe.tiler, x, y_value, reduce_pattern);
  has_valid_a_ = physical_last != Zero && !x.isAr;
  valid_r_expr_ = GetArgReduceValidRExpr(tpipe, x, y_value);

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, x, y_index, value_dtype_name),
                    "GetDtypeNameForReduce failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgReducePreProcess(tpipe, x, y_value, api_param->api_pre_process, nullptr, physical_last);
  GE_ASSERT_SUCCESS(FillArgReduceNodeParams(this->node, tpipe, x, y_value, y_index, false, true, true, nullptr, true,
                                            physical_last, has_valid_a_ ? One : Zero));
  // Use the unaligned tile size, as in the input's global offset, rather than padded first/last.
  std::string index_base;
  std::string loop_axis;
  GE_ASSERT_SUCCESS(GetArgReducePhase1TileExpr(tpipe.tiler, current_axis, x, y_value, index_base, loop_axis));
  GE_ASSERT_SUCCESS(BuildTileState(tpipe, value_dtype_name, index_base, loop_axis, *api_param));

  // Preserve RA/MultiRound classification: only the accumulation implementation changes.
  api_param->api_name = ArgReduceApiName(this->api_name_, "WithValueAccumulateExtend");
  api_param->semantic_feature =
      std::string(x.isAr ? template_semantic::kAr : template_semantic::kRa) + "/" + template_semantic::kMultiRound;
  api_param->template_params.emplace_back(value_dtype_name);
  api_param->template_params.emplace_back(reduce_pattern);
  api_param->output_params.emplace_back(y_index.Str(), true, CombinedExprFactory::Constant(0));
  api_param->output_params.emplace_back(y_value.Str(), true, CombinedExprFactory::Constant(0));
  api_param->input_params.emplace_back(
      x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
  api_param->input_params.emplace_back("tmp_arg_reduce_index", true, CombinedExprFactory::Constant(0));
  api_param->input_params.emplace_back("tmp_arg_reduce_value", true, CombinedExprFactory::Constant(0));
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgReducePhase1ApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                                    std::stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ", arg_reduce_index_base, " << valid_r_expr_ << ", arg_reduce_first_tile"
     << (has_valid_a_ ? ", 1" : "") << ");" << std::endl;
  return af::SUCCESS;
}

Status RegArgReducePhase2ApiCall::BuildApiParam(
    const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
    const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  GE_ASSERT_TRUE(inputs.size() >= 2UL, "ArgMaxMultiRPhase2 requires at least 2 inputs.");
  auto value = inputs[0].get();
  auto index = inputs[1].get();
  auto y = outputs[0].get();
  GE_ASSERT_TRUE(index.dtype == ge::DT_INT32, "ArgMaxMultiRPhase2 index input must be int32");
  GE_ASSERT_TRUE(!value.vectorized_axis.empty(), "ArgMaxMultiRPhase2 requires non-empty vectorized axis.");
  valid_a_expr_ = tpipe.tiler.GetAxis(value.vectorized_axis.back()).actual_size.Str();

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, value, y, value_dtype_name),
                    "GetDtypeNameForReduce failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgReducePreProcess(tpipe, value, y, api_param->api_pre_process, &index);
  GE_ASSERT_SUCCESS(FillArgReduceNodeParams(this->node, tpipe, value, y, y, true, false, false, &index));

  const bool need_cast = NeedIndexCast(y);
  const std::string y_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y);
  std::string index_dst = y.Str();
  if (need_cast) {
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index", 0L, api_param->api_pre_process);
    index_dst = "tmp_arg_reduce_index";
  }

  // ArgMaxMultiRPhase2VectorExtend<value_dtype>(y, value, index, ...)：内部 index 固定 int32。
  api_param->api_name = ArgReduceApiName(this->api_name_, "MultiRPhase2VectorExtend");
  api_param->template_params.emplace_back(value_dtype_name);
  api_param->output_params.emplace_back(
      index_dst, true, need_cast ? CombinedExprFactory::Constant(0) : CombinedExprFactory::SymbolVar(y_offset));
  api_param->input_params.emplace_back(
      value.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, value)));
  api_param->input_params.emplace_back(
      index.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, index)));
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

  if (need_cast) {
    EmitIndexCastToFrontend(y, y.Str() + "[" + y_offset + "]", "tmp_arg_reduce_index[0]", api_param->api_post_process);
  }

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgReducePhase2ApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                                    std::stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ", " << valid_a_expr_ << ");" << std::endl;
  return af::SUCCESS;
}

Status RegArgReduceWithValueApiCall::BuildSingleReduceParam(const TPipe &tpipe,
                                                            const std::vector<ascir::AxisId> &current_axis,
                                                            const Tensor &y_value, const Tensor &y_index,
                                                            bool need_cast, CodegenApiParam &api_param) const {
  const std::string y_index_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y_index);
  std::string index_dst = y_index.Str();
  if (need_cast) {
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index", 0L, api_param.api_pre_process);
    index_dst = "tmp_arg_reduce_index";
  }
  api_param.api_name = ArgReduceApiName(this->api_name_, "WithValueExtend");
  api_param.output_params.emplace_back(
      index_dst, true, need_cast ? CombinedExprFactory::Constant(0) : CombinedExprFactory::SymbolVar(y_index_offset));
  api_param.output_params.emplace_back(
      y_value.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y_value)));
  if (need_cast) {
    EmitIndexCastToFrontend(y_index, y_index.Str() + "[" + y_index_offset + "]", "tmp_arg_reduce_index[0]",
                            api_param.api_post_process);
  }
  api_param.api_post_process.emplace_back("}\n");
  return af::SUCCESS;
}

Status RegArgReduceWithValueApiCall::BuildMultiReducePreProcess(const TPipe &tpipe, const std::string &value_dtype,
                                                                const std::string &loop_axis,
                                                                CodegenApiParam &api_param) const {
  api_param.api_pre_process.emplace_back("static int64_t accumulated_offset = 0;\n");
  api_param.api_pre_process.emplace_back("if (" + loop_axis + " == 0) {\n");
  api_param.api_pre_process.emplace_back("  accumulated_offset = 0;\n");
  api_param.api_pre_process.emplace_back("}\n");
  DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index", 0L, api_param.api_pre_process);
  DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index_saved", 2L, api_param.api_pre_process);
  auto value_tmp_it = this->tmp_buf_id.find(1L);
  GE_ASSERT_TRUE(value_tmp_it != this->tmp_buf_id.end(),
                 "ArgReduceWithValue multi_reduce requires index/value tmp buffers.");
  api_param.api_pre_process.emplace_back("LocalTensor<" + value_dtype + "> tmp_arg_reduce_value;\n");
  api_param.api_pre_process.emplace_back("tmp_arg_reduce_value = " + tpipe.tmp_buf.name + "_" +
                                         std::to_string(value_tmp_it->second) + ".ReinterpretCast<" + value_dtype +
                                         ">();\n");
  return af::SUCCESS;
}

Status RegArgReduceWithValueApiCall::BuildMultiReduceParam(const TPipe &tpipe,
                                                           const std::vector<ascir::AxisId> &current_axis,
                                                           const Tensor &x, const Tensor &y_value,
                                                           const Tensor &y_index, const std::string &value_dtype,
                                                           bool need_cast, CodegenApiParam &api_param) const {
  const auto &reduce_axis = tpipe.tiler.GetAxis(current_axis.back());
  const std::string loop_axis = reduce_axis.Str();
  GE_ASSERT_SUCCESS(BuildMultiReducePreProcess(tpipe, value_dtype, loop_axis, api_param));
  api_param.api_name = ArgReduceApiName(this->api_name_, "WithValueExtend");
  api_param.output_params.emplace_back("tmp_arg_reduce_index", true, CombinedExprFactory::Constant(0));
  api_param.output_params.emplace_back("tmp_arg_reduce_value", true, CombinedExprFactory::Constant(0));
  api_param.api_post_process.emplace_back("if (" + loop_axis + " == 0) {\n");
  api_param.api_post_process.emplace_back("ArgReduceCopy(tmp_arg_reduce_index_saved[0], tmp_arg_reduce_index[0], " +
                                          y_index.actual_size.Str() + ");\n");
  api_param.api_post_process.emplace_back("ArgReduceCopy(" + y_value.Str() + "[0], tmp_arg_reduce_value[0], " +
                                          y_value.actual_size.Str() + ");\n");
  api_param.api_post_process.emplace_back("} else {\n");
  api_param.api_post_process.emplace_back(
      "ArgReduceMergeRound<" + value_dtype + ", " + ArgReduceDirection(this->api_name_) + ">(" + y_value.Str() +
      "[0], tmp_arg_reduce_index_saved[0], " + "tmp_arg_reduce_value[0], tmp_arg_reduce_index[0], " +
      y_value.actual_size.Str() + ");\n");
  api_param.api_post_process.emplace_back("}\n");
  const std::string accum_step =
      x.isAr ? tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size.Str() : "first_actual";
  ascir::AxisId consumer_axis = af::kIdNone;
  GE_CHK_STATUS_RET(
      EmitMultiRoundIndexAndAccumulate(y_index, need_cast, IsReadOutersideWrite(consumer_axis), loop_axis,
                                       reduce_axis.loop_size.Str(), accum_step, api_param.api_post_process),
      "Emit ArgReduce multi-round output failed.");
  api_param.api_post_process.emplace_back("}\n");
  return af::SUCCESS;
}

Status RegArgReduceWithValueApiCall::BuildApiParam(
    const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
    const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  GE_ASSERT_TRUE(outputs.size() >= 2UL, "ArgReduceWithValue requires at least 2 outputs.");
  auto x = inputs[0].get();
  auto y_value = outputs[0].get();
  auto y_index = outputs[1].get();

  std::string reduce_pattern;
  const auto physical_last = PrepareArgReduceLayout(tpipe.tiler, x, y_value, reduce_pattern);
  has_valid_a_ = physical_last != Zero && !x.isAr;
  valid_r_expr_ = GetArgReduceValidRExpr(tpipe, x, y_value);

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, x, y_index, value_dtype_name),
                    "GetDtypeNameForReduce failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgReducePreProcess(tpipe, x, y_value, api_param->api_pre_process, nullptr, physical_last);

  multi_reduce_ = IsNeedMultiReduce(tpipe.tiler, x, y_value, current_axis.back());
  const bool need_cast = NeedIndexCast(y_index);
  GE_ASSERT_SUCCESS(FillArgReduceNodeParams(this->node, tpipe, x, y_value, y_index, false, true, multi_reduce_, nullptr,
                                            false, physical_last, has_valid_a_ ? One : Zero));

  api_param->template_params = {value_dtype_name, reduce_pattern};
  api_param->input_params.emplace_back(
      x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");
  if (!multi_reduce_) {
    GE_ASSERT_SUCCESS(BuildSingleReduceParam(tpipe, current_axis, y_value, y_index, need_cast, *api_param));
  } else {
    GE_ASSERT_SUCCESS(
        BuildMultiReduceParam(tpipe, current_axis, x, y_value, y_index, value_dtype_name, need_cast, *api_param));
  }

  api_param->semantic_feature = std::string(x.isAr ? template_semantic::kAr : template_semantic::kRa) +
                                (multi_reduce_ ? "/" + std::string(template_semantic::kMultiRound) : "");
  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgReduceWithValueApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                                       std::stringstream &ss) const {
  if (multi_reduce_) {
    ss << api_param.cal_count.ToStr(tiler) << ", false, accumulated_offset, " << valid_r_expr_
       << (has_valid_a_ ? ", 1" : "") << ");" << std::endl;
  } else {
    ss << api_param.cal_count.ToStr(tiler) << ", false, 0, " << valid_r_expr_ << (has_valid_a_ ? ", 1" : "") << ");"
       << std::endl;
  }
  return af::SUCCESS;
}

Status RegArgReducePhase2WithValueApiCall::BuildApiParam(
    const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
    const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  GE_ASSERT_TRUE(inputs.size() >= 2UL, "ArgMaxMultiRPhase2WithValue requires at least 2 inputs.");
  GE_ASSERT_TRUE(outputs.size() >= 2UL, "ArgMaxMultiRPhase2WithValue requires at least 2 outputs.");
  auto value = inputs[0].get();
  auto index = inputs[1].get();
  auto y_value = outputs[0].get();
  auto y_index = outputs[1].get();
  GE_ASSERT_TRUE(index.dtype == ge::DT_INT32, "ArgMaxMultiRPhase2WithValue index input must be int32");
  GE_ASSERT_TRUE(!value.vectorized_axis.empty(), "ArgMaxMultiRPhase2WithValue requires non-empty vectorized axis.");
  valid_a_expr_ = tpipe.tiler.GetAxis(value.vectorized_axis.back()).actual_size.Str();

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, value, y_index, value_dtype_name),
                    "GetDtypeNameForReduce failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgReducePreProcess(tpipe, value, y_value, api_param->api_pre_process, &index);
  GE_ASSERT_SUCCESS(FillArgReduceNodeParams(this->node, tpipe, value, y_value, y_index, true, true, false, &index));

  const bool need_cast = NeedIndexCast(y_index);
  const std::string y_index_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y_index);
  std::string index_dst = y_index.Str();
  if (need_cast) {
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_arg_reduce_index", 0L, api_param->api_pre_process);
    index_dst = "tmp_arg_reduce_index";
  }

  api_param->api_name = ArgReduceApiName(this->api_name_, "MultiRPhase2WithValueVectorExtend");
  api_param->template_params.emplace_back(value_dtype_name);
  api_param->output_params.emplace_back(
      y_value.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y_value)));
  api_param->output_params.emplace_back(
      index_dst, true, need_cast ? CombinedExprFactory::Constant(0) : CombinedExprFactory::SymbolVar(y_index_offset));
  api_param->input_params.emplace_back(
      value.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, value)));
  api_param->input_params.emplace_back(
      index.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, index)));
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

  if (need_cast) {
    EmitIndexCastToFrontend(y_index, y_index.Str() + "[" + y_index_offset + "]", "tmp_arg_reduce_index[0]",
                            api_param->api_post_process);
  }

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgReducePhase2WithValueApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                                             std::stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ", " << valid_a_expr_ << ");" << std::endl;
  return af::SUCCESS;
}

static ApiCallRegister<RegArgmaxApiCall> register_argmax_api_call("RegArgmaxApiCall");
static ApiCallRegister<RegArgMaxPhase1ApiCall> register_argmax_phase1_api_call("RegArgMaxPhase1ApiCall");
static ApiCallRegister<RegArgMaxPhase2ApiCall> register_argmax_phase2_api_call("RegArgMaxPhase2ApiCall");
static ApiCallRegister<RegArgmaxWithValueApiCall> register_argmax_with_value_api_call("RegArgmaxWithValueApiCall");
static ApiCallRegister<RegArgMaxPhase2WithValueApiCall> register_argmax_phase2_with_value_api_call(
    "RegArgMaxPhase2WithValueApiCall");

static ApiCallRegister<RegArgminApiCall> register_argmin_api_call("RegArgminApiCall");
static ApiCallRegister<RegArgMinPhase1ApiCall> register_argmin_phase1_api_call("RegArgMinPhase1ApiCall");
static ApiCallRegister<RegArgMinPhase2ApiCall> register_argmin_phase2_api_call("RegArgMinPhase2ApiCall");
static ApiCallRegister<RegArgminWithValueApiCall> register_argmin_with_value_api_call("RegArgminWithValueApiCall");
static ApiCallRegister<RegArgMinPhase2WithValueApiCall> register_argmin_phase2_with_value_api_call(
    "RegArgMinPhase2WithValueApiCall");

}  // namespace codegen
