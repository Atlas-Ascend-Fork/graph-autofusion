/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_argmax_api_call.h"

#include "codegen/api_call/reduce/reduce_api_call_base.h"
#include <algorithm>
#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "common/checker.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"
#include "reg_api_call_utils.h"
#include "codegen_api_param/codegen_api_param.h"

namespace codegen {
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;
using namespace reduce_base;

namespace {
// Argmax 内部 index 统一为 int32；面向前端的输出若为 int64，在生成 kernel 的出口处
// 用 CastExtend 做一次 int32→int64 转换。
constexpr const char *kArgmaxIndexDtype = "int32_t";

bool NeedIndexCast(const Tensor &index) {
  return index.dtype != ge::DT_INT32;
}

// 生成 argmax 内部 int32 index → 前端 index 输出的转换。
void EmitIndexCastToFrontend(const Tensor &y, const std::string &dst_expr, const std::string &src_i32,
                             std::vector<std::string> &post_process) {
  std::string dst_dtype_name;
  GE_CHK_STATUS(Tensor::DtypeName(y.dtype, dst_dtype_name), "Codegen get index dtype:%d failed",
                static_cast<int32_t>(y.dtype));
  post_process.emplace_back("{\n");
  post_process.emplace_back("  uint32_t argmax_cast_dims[] = {" + y.actual_size.Str() + "};\n");
  post_process.emplace_back("  CastExtend<" + std::string(kArgmaxIndexDtype) + ", " + dst_dtype_name + ", 1>(" +
                            dst_expr + ", " + src_i32 + ", argmax_cast_dims, argmax_cast_dims, argmax_cast_dims);\n");
  post_process.emplace_back("}\n");
}

// index_src 非空时（Phase2 双输入：value(T) + index(int32)），额外按 index dtype
// 计算其末轴 32B 对齐行跨度 last_tail：value/index 每行元素数可能不同（如 int64 4 vs int32 8），
// Phase2 需分别寻址，故 tmp_reduce_shape = {first_actual, last, last_tail}。
void BuildArgmaxCommonPreProcess(const TPipe &tpipe, const Tensor &x, const Tensor &y,
                                 std::vector<std::string> &pre_process, const Tensor *index_src = nullptr) {
  ReduceMergedSizeCodeGen(tpipe, pre_process, x, y);
  ReduceDimACodeGen(x, "Argmax", pre_process);
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

Status GetArgmaxValueDtypeName(const std::string &api_name, const Tensor &x, const Tensor &y,
                               std::string &value_dtype_name) {
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(api_name, x, y, value_dtype_name),
                    "Codegen get value dtype failed for api:%s", api_name.c_str());
  return af::SUCCESS;
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

std::string GetArgmaxValidRExpr(const TPipe &tpipe, const Tensor &src, const Tensor &dst) {
  if (!src.isAr) {
    return "first_actual";
  }
  std::string valid_r;
  const size_t axis_count = std::min(src.vectorized_strides.size(), dst.vectorized_strides.size());
  for (size_t i = 0; i < axis_count; ++i) {
    if (dst.vectorized_strides[i] == 0 && src.vectorized_strides[i] != 0) {
      const auto actual_size = tpipe.tiler.GetAxis(src.vectorized_axis[i]).actual_size.Str();
      valid_r = valid_r.empty() ? actual_size : "(" + valid_r + ") * (" + actual_size + ")";
    }
  }
  return valid_r.empty() ? "1" : valid_r;
}
}  // namespace

Status RegArgmaxApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();

  std::string reduce_pattern;
  GetIsArAndPattern(x, y, x.isAr, reduce_pattern);
  valid_r_expr_ = GetArgmaxValidRExpr(tpipe, x, y);

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetArgmaxValueDtypeName(this->api_name_, x, y, value_dtype_name),
                    "GetArgmaxValueDtypeName failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgmaxCommonPreProcess(tpipe, x, y, api_param->api_pre_process);

  const bool need_multi_reduce = IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back());
  const bool need_cast = NeedIndexCast(y);

  if (!need_multi_reduce) {
    // 单次归约（R 轴完整）：ArgMaxExtend 一次归约输出 int32 index。
    // 面向前端为 int64 时先写 tmp int32，再在出口 Cast 成 int64。
    const std::string y_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y);
    std::string index_dst = y.Str();
    if (need_cast) {
      DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index", 0L, api_param->api_pre_process);
      index_dst = "tmp_argmax_index";
    }
    api_param->api_name = "ArgMaxExtend";
    api_param->template_params.emplace_back(value_dtype_name);
    api_param->template_params.emplace_back(reduce_pattern);
    api_param->output_params.emplace_back(
        index_dst, true, need_cast ? CombinedExprFactory::Constant(0) : CombinedExprFactory::SymbolVar(y_offset));
    api_param->input_params.emplace_back(
        x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
    api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

    if (need_cast) {
      EmitIndexCastToFrontend(y, y.Str() + "[" + y_offset + "]", "tmp_argmax_index[0]", api_param->api_post_process);
    }
  } else {
    // multi_reduce（R 轴分块）：每个 tile 用 ArgMaxWithValueExtend(indexBase=accumulated_offset)
    // 输出全局 index + value（index 已经全局化），再与历史最佳做向量合并
    // （值大者胜、值等取最小全局 index），最后累加 R 轴偏移。
    //
    // accumulated_offset 需在同一核内跨 tile 累加（循环内声明），故用 static 保存状态；但 device 侧
    // static 局部变量跨 kernel launch 持久存在，若在首个 tile 不复位，会把上一次 launch 的残留偏移
    // 叠加到本次 index 上（脏偏移）。因此在 loop 轴为 0（首个 tile）时显式复位。
    const auto &reduce_axis = tpipe.tiler.GetAxis(current_axis.back());
    const std::string loop_axis = reduce_axis.Str();
    api_param->api_pre_process.emplace_back("static int64_t accumulated_offset = 0;\n");
    api_param->api_pre_process.emplace_back("if (" + loop_axis + " == 0) {\n");
    api_param->api_pre_process.emplace_back("  accumulated_offset = 0;\n");
    api_param->api_pre_process.emplace_back("}\n");
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index", 0L, api_param->api_pre_process);
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index_saved", 3L, api_param->api_pre_process);
    auto value_tmp_it = this->tmp_buf_id.find(1L);
    auto saved_tmp_it = this->tmp_buf_id.find(2L);
    GE_ASSERT_TRUE(value_tmp_it != this->tmp_buf_id.end() && saved_tmp_it != this->tmp_buf_id.end());
    api_param->api_pre_process.emplace_back("LocalTensor<" + value_dtype_name + "> tmp_argmax_value;\n");
    api_param->api_pre_process.emplace_back("tmp_argmax_value = " + tpipe.tmp_buf.name + "_" +
                                            std::to_string(value_tmp_it->second) + ".ReinterpretCast<" +
                                            value_dtype_name + ">();\n");
    api_param->api_pre_process.emplace_back("LocalTensor<" + value_dtype_name + "> tmp_argmax_value_saved;\n");
    api_param->api_pre_process.emplace_back("tmp_argmax_value_saved = " + tpipe.tmp_buf.name + "_" +
                                            std::to_string(saved_tmp_it->second) + ".ReinterpretCast<" +
                                            value_dtype_name + ">();\n");

    api_param->api_name = "ArgMaxWithValueExtend";
    api_param->template_params.emplace_back(value_dtype_name);
    api_param->template_params.emplace_back(reduce_pattern);
    api_param->output_params.emplace_back("tmp_argmax_index", true, CombinedExprFactory::Constant(0));
    api_param->output_params.emplace_back("tmp_argmax_value", true, CombinedExprFactory::Constant(0));
    api_param->input_params.emplace_back(
        x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
    api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

    // 后处理：第一次迭代直接复制到 int32 累加器，后续迭代做合并；随后累加 R 轴偏移。
    api_param->api_post_process.emplace_back("if (" + loop_axis + " == 0) {\n");
    api_param->api_post_process.emplace_back("ArgmaxCopy(tmp_argmax_index_saved[0], tmp_argmax_index[0], " +
                                             y.actual_size.Str() + ");\n");
    api_param->api_post_process.emplace_back("ArgmaxCopy(tmp_argmax_value_saved[0], tmp_argmax_value[0], " +
                                             y.actual_size.Str() + ");\n");
    api_param->api_post_process.emplace_back("} else {\n");
    api_param->api_post_process.emplace_back("ArgmaxMergeRound<" + value_dtype_name +
                                             ">(tmp_argmax_value_saved[0], tmp_argmax_index_saved[0], "
                                             "tmp_argmax_value[0], tmp_argmax_index[0], " +
                                             y.actual_size.Str() + ");\n");
    api_param->api_post_process.emplace_back("}\n");
    ascir::AxisId consumer_axis = af::kIdNone;
    const bool defer_index_output = IsReadOutersideWrite(consumer_axis);
    if (defer_index_output) {
      api_param->api_post_process.emplace_back("if (" + loop_axis + " + 1 == " + reduce_axis.loop_size.Str() + ") {\n");
    }
    if (need_cast) {
      EmitIndexCastToFrontend(y, y.Str() + "[0]", "tmp_argmax_index_saved[0]", api_param->api_post_process);
    } else {
      api_param->api_post_process.emplace_back("ArgmaxCopy(" + y.Str() + "[0], tmp_argmax_index_saved[0], " +
                                               y.actual_size.Str() + ");\n");
    }
    if (defer_index_output) {
      api_param->api_post_process.emplace_back("}\n");
    }
    if (x.isAr) {
      api_param->api_post_process.emplace_back(
          "accumulated_offset += " + tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size.Str() + ";\n");
    } else {
      api_param->api_post_process.emplace_back("accumulated_offset += first_actual;\n");
    }
  }

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgmaxApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                           std::stringstream &ss) const {
  if (api_param.api_name == "ArgMaxWithValueExtend") {
    // multi_reduce 场景：传递 accumulated_offset 作为 indexBase（index 全局化）
    ss << api_param.cal_count.ToStr(tiler) << ", false, accumulated_offset, " << valid_r_expr_ << ");" << std::endl;
  } else {
    // validR 传入实际有效 R 长度，屏蔽 tail padding lane
    ss << api_param.cal_count.ToStr(tiler) << ", false, " << valid_r_expr_ << ");" << std::endl;
  }
  return af::SUCCESS;
}

Status RegArgMaxPhase1ApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                             const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                             const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  GE_ASSERT_TRUE(outputs.size() >= 2UL, "ArgMaxMultiRPhase1 requires at least 2 outputs.");
  auto x = inputs[0].get();
  auto y_value = outputs[0].get();
  auto y_index = outputs[1].get();
  GE_ASSERT_TRUE(y_index.dtype == ge::DT_INT32, "ArgMaxMultiRPhase1 index output must be int32");

  std::string reduce_pattern;
  GetIsArAndPattern(x, y_value, x.isAr, reduce_pattern);
  valid_r_expr_ = GetArgmaxValidRExpr(tpipe, x, y_value);

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetArgmaxValueDtypeName(this->api_name_, x, y_index, value_dtype_name),
                    "GetArgmaxValueDtypeName failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgmaxCommonPreProcess(tpipe, x, y_value, api_param->api_pre_process);
  // R 轴分核 Phase1：每个 core 沿自己那一段 R_local 归约，index 需要加上该核在 R 轴上的全局
  // 分块偏移才能得到全局 index（供 Phase2 沿核维合并，值等取最小全局 index）。
  // 必须显式定位 R 轴自身的 tile-outer / block-inner：AR 时最内层循环是 R，RA 时最内层可能是 A，
  // 用 current_axis.back() 会在 RA 下把 A 的分块序号当成 R 的序号，导致 index 基址错误。
  GE_ASSERT_TRUE(!x.vectorized_axis.empty(), "ArgMaxMultiRPhase1 requires non-empty vectorized axis.");
  const ascir::AxisId r_axis_id = x.isAr ? x.vectorized_axis.back() : x.vectorized_axis.front();
  const auto &r_axis = tpipe.tiler.GetAxis(r_axis_id);
  // r_axis_id 在真实调度图里有三种形态：
  //   1) TileInner（最常见）：from[0] 是被切分的原始轴，配对的 TileOuter 在 split_pair_other_id 上；
  //   2) BlockInner：它本身就是要找的循环轴；
  //   3) 原始轴 / TileOuter：在 axis_map 中找 from[0] == r_axis_id 的 TileOuter/BlockOuter 子轴。
  // 直接对 TileInner 取 from[0] 会拿到原始轴，导致找不到 block-inner。
  ascir::AxisId r_tile_outer_id = r_axis_id;
  const codegen::Axis *r_block_inner = nullptr;
  if (r_axis.type == ascir::Axis::Type::kAxisTypeBlockInner) {
    r_block_inner = &r_axis;
    if (!r_axis.from.empty()) {
      r_tile_outer_id = r_axis.from[0];
    }
  } else {
    if (r_axis.type == ascir::Axis::Type::kAxisTypeTileInner && r_axis.split_pair_other_id != af::kIdNone) {
      r_tile_outer_id = r_axis.split_pair_other_id;
    } else if (!r_axis.from.empty()) {
      r_tile_outer_id = r_axis.from[0];
    } else {
      for (const auto &axis_pair : tpipe.tiler.axis_map) {
        const auto &axis = axis_pair.second;
        if ((axis.type == ascir::Axis::Type::kAxisTypeTileOuter ||
             axis.type == ascir::Axis::Type::kAxisTypeBlockOuter) &&
            !axis.from.empty() && axis.from[0] == r_axis_id) {
          r_tile_outer_id = axis_pair.first;
          break;
        }
      }
    }
    for (const auto &axis_pair : tpipe.tiler.axis_map) {
      const auto &axis = axis_pair.second;
      if (axis.type == ascir::Axis::Type::kAxisTypeBlockInner && !axis.from.empty() &&
          axis.from[0] == r_tile_outer_id) {
        r_block_inner = &axis;
        break;
      }
    }
    // 兼容：若 r_tile_outer_id 本身在 BindBlock 后被提升为 BlockOuter，则其 split_pair 是 BlockInner。
    if (r_block_inner == nullptr) {
      const auto &candidate = tpipe.tiler.GetAxis(r_tile_outer_id);
      if (candidate.type == ascir::Axis::Type::kAxisTypeBlockOuter && candidate.split_pair_other_id != af::kIdNone) {
        r_block_inner = &tpipe.tiler.GetAxis(candidate.split_pair_other_id);
      }
    }
  }
  GE_ASSERT_NOTNULL(r_block_inner, "ArgMaxMultiRPhase1 cannot locate the R axis block-inner axis.");
  const auto &r_tile_outer = tpipe.tiler.GetAxis(r_tile_outer_id);
  const std::string r_tile_index = r_tile_outer.Str();
  // index base 的步长必须与取数一致：ReduceMergedSizeCodeGen 生成的 first/last 是 32B 对齐后的
  // 长度，而 DataCopy/Gather 的全局偏移按未对齐的 tile size 递增。若用对齐长度计算 base，当
  // tile size 非 32B 对齐（或存在不能整除的 tail）时，每个 R 分块会累积漂移，index 错误。
  api_param->api_pre_process.emplace_back("uint32_t argmax_index_base = " + r_tile_index + " * " +
                                          tpipe.tiler.AxisSize(r_axis_id) + ";\n");

  // Phase1 每次调用对应核内一个 R tile；用 tmp 暂存当前 tile 的 value/index，
  // 第一个 tile 直接写输出（作为跨 tile 累加器），后续 tile 与输出做 ArgmaxMergeRound：
  // 值大者胜、值等取更小全局 index。index base 已按 tile 全局化，无需再累加。
  const std::string loop_axis = r_block_inner->Str();
  DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index", 0L, api_param->api_pre_process);
  auto value_tmp_it = this->tmp_buf_id.find(1L);
  GE_ASSERT_TRUE(value_tmp_it != this->tmp_buf_id.end(),
                 "ArgMaxMultiRPhase1 requires index/value tmp buffers for multi-tile merge.");
  api_param->api_pre_process.emplace_back("LocalTensor<" + value_dtype_name + "> tmp_argmax_value;\n");
  api_param->api_pre_process.emplace_back("tmp_argmax_value = " + tpipe.tmp_buf.name + "_" +
                                          std::to_string(value_tmp_it->second) + ".ReinterpretCast<" +
                                          value_dtype_name + ">();\n");

  // ArgMaxWithValueExtend<value_dtype, pattern>(indexDst, valueDst, src, ...)：索引输出在前
  api_param->api_name = "ArgMaxWithValueExtend";
  api_param->template_params.emplace_back(value_dtype_name);
  api_param->template_params.emplace_back(reduce_pattern);
  api_param->output_params.emplace_back("tmp_argmax_index", true, CombinedExprFactory::Constant(0));
  api_param->output_params.emplace_back("tmp_argmax_value", true, CombinedExprFactory::Constant(0));
  api_param->input_params.emplace_back(
      x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

  // 跨 tile 合并：首个 tile 初始化输出累加器（int32 index），后续 tile 与输出合并。
  api_param->api_post_process.emplace_back("if (" + loop_axis + " == 0) {\n");
  api_param->api_post_process.emplace_back("ArgmaxCopy(" + y_index.Str() + "[0], tmp_argmax_index[0], " +
                                           y_index.actual_size.Str() + ");\n");
  api_param->api_post_process.emplace_back("ArgmaxCopy(" + y_value.Str() + "[0], tmp_argmax_value[0], " +
                                           y_value.actual_size.Str() + ");\n");
  api_param->api_post_process.emplace_back("} else {\n");
  api_param->api_post_process.emplace_back("ArgmaxMergeRound<" + value_dtype_name + ">(" + y_value.Str() + "[0], " +
                                           y_index.Str() + "[0], tmp_argmax_value[0], tmp_argmax_index[0], " +
                                           y_value.actual_size.Str() + ");\n");
  api_param->api_post_process.emplace_back("}\n");

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgMaxPhase1ApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                                 std::stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ", false, argmax_index_base, " << valid_r_expr_ << ");" << std::endl;
  return af::SUCCESS;
}

Status RegArgMaxPhase2ApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
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
  GE_CHK_STATUS_RET(GetArgmaxValueDtypeName(this->api_name_, value, y, value_dtype_name),
                    "GetArgmaxValueDtypeName failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgmaxCommonPreProcess(tpipe, value, y, api_param->api_pre_process, &index);

  const bool need_cast = NeedIndexCast(y);
  const std::string y_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y);
  std::string index_dst = y.Str();
  if (need_cast) {
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index", 0L, api_param->api_pre_process);
    index_dst = "tmp_argmax_index";
  }

  // ArgMaxMultiRPhase2VectorExtend<value_dtype>(y, value, index, ...)：内部 index 固定 int32。
  api_param->api_name = "ArgMaxMultiRPhase2VectorExtend";
  api_param->template_params.emplace_back(value_dtype_name);
  api_param->output_params.emplace_back(
      index_dst, true, need_cast ? CombinedExprFactory::Constant(0) : CombinedExprFactory::SymbolVar(y_offset));
  api_param->input_params.emplace_back(
      value.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, value)));
  api_param->input_params.emplace_back(
      index.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, index)));
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

  if (need_cast) {
    EmitIndexCastToFrontend(y, y.Str() + "[" + y_offset + "]", "tmp_argmax_index[0]", api_param->api_post_process);
  }

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgMaxPhase2ApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                                 std::stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ", " << valid_a_expr_ << ");" << std::endl;
  return af::SUCCESS;
}

Status RegArgmaxWithValueApiCall::BuildApiParam(
    const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
    const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  GE_ASSERT_TRUE(outputs.size() >= 2UL, "ArgmaxWithValue requires at least 2 outputs.");
  auto x = inputs[0].get();
  auto y_value = outputs[0].get();
  auto y_index = outputs[1].get();

  std::string reduce_pattern;
  GetIsArAndPattern(x, y_value, x.isAr, reduce_pattern);
  valid_r_expr_ = GetArgmaxValidRExpr(tpipe, x, y_value);

  std::string value_dtype_name;
  GE_CHK_STATUS_RET(GetArgmaxValueDtypeName(this->api_name_, x, y_index, value_dtype_name),
                    "GetArgmaxValueDtypeName failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgmaxCommonPreProcess(tpipe, x, y_value, api_param->api_pre_process);

  multi_reduce_ = IsNeedMultiReduce(tpipe.tiler, x, y_value, current_axis.back());
  const bool need_cast = NeedIndexCast(y_index);

  if (!multi_reduce_) {
    // 单次归约：ArgMaxWithValueExtend 直接写 value；index 先写 int32 tmp（前端 int64 时再 Cast）。
    const std::string y_index_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y_index);
    std::string index_dst = y_index.Str();
    if (need_cast) {
      DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index", 0L, api_param->api_pre_process);
      index_dst = "tmp_argmax_index";
    }
    api_param->api_name = "ArgMaxWithValueExtend";
    api_param->template_params.emplace_back(value_dtype_name);
    api_param->template_params.emplace_back(reduce_pattern);
    api_param->output_params.emplace_back(
        index_dst, true, need_cast ? CombinedExprFactory::Constant(0) : CombinedExprFactory::SymbolVar(y_index_offset));
    api_param->output_params.emplace_back(
        y_value.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y_value)));
    api_param->input_params.emplace_back(
        x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
    api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

    if (need_cast) {
      EmitIndexCastToFrontend(y_index, y_index.Str() + "[" + y_index_offset + "]", "tmp_argmax_index[0]",
                              api_param->api_post_process);
    }

    // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
    api_param->api_post_process.emplace_back("}\n");
  } else {
    // R 轴分块：每个 tile 写入 int32 tmp，首 tile 初始化 int32 累加器，后续 tile 与累加器合并；
    // accumulated_offset 作为 indexBase 使 index 全局化。
    const auto &reduce_axis = tpipe.tiler.GetAxis(current_axis.back());
    const std::string loop_axis = reduce_axis.Str();
    api_param->api_pre_process.emplace_back("static int64_t accumulated_offset = 0;\n");
    api_param->api_pre_process.emplace_back("if (" + loop_axis + " == 0) {\n");
    api_param->api_pre_process.emplace_back("  accumulated_offset = 0;\n");
    api_param->api_pre_process.emplace_back("}\n");
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index", 0L, api_param->api_pre_process);
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index_saved", 2L, api_param->api_pre_process);
    auto value_tmp_it = this->tmp_buf_id.find(1L);
    GE_ASSERT_TRUE(value_tmp_it != this->tmp_buf_id.end(),
                   "ArgmaxWithValue multi_reduce requires index/value tmp buffers.");
    api_param->api_pre_process.emplace_back("LocalTensor<" + value_dtype_name + "> tmp_argmax_value;\n");
    api_param->api_pre_process.emplace_back("tmp_argmax_value = " + tpipe.tmp_buf.name + "_" +
                                            std::to_string(value_tmp_it->second) + ".ReinterpretCast<" +
                                            value_dtype_name + ">();\n");

    api_param->api_name = "ArgMaxWithValueExtend";
    api_param->template_params.emplace_back(value_dtype_name);
    api_param->template_params.emplace_back(reduce_pattern);
    api_param->output_params.emplace_back("tmp_argmax_index", true, CombinedExprFactory::Constant(0));
    api_param->output_params.emplace_back("tmp_argmax_value", true, CombinedExprFactory::Constant(0));
    api_param->input_params.emplace_back(
        x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
    api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

    api_param->api_post_process.emplace_back("if (" + loop_axis + " == 0) {\n");
    api_param->api_post_process.emplace_back("ArgmaxCopy(tmp_argmax_index_saved[0], tmp_argmax_index[0], " +
                                             y_index.actual_size.Str() + ");\n");
    api_param->api_post_process.emplace_back("ArgmaxCopy(" + y_value.Str() + "[0], tmp_argmax_value[0], " +
                                             y_value.actual_size.Str() + ");\n");
    api_param->api_post_process.emplace_back("} else {\n");
    api_param->api_post_process.emplace_back(
        "ArgmaxMergeRound<" + value_dtype_name + ">(" + y_value.Str() + "[0], tmp_argmax_index_saved[0], " +
        "tmp_argmax_value[0], tmp_argmax_index[0], " + y_value.actual_size.Str() + ");\n");
    api_param->api_post_process.emplace_back("}\n");
    ascir::AxisId consumer_axis = af::kIdNone;
    const bool defer_index_output = IsReadOutersideWrite(consumer_axis);
    if (defer_index_output) {
      api_param->api_post_process.emplace_back("if (" + loop_axis + " + 1 == " + reduce_axis.loop_size.Str() + ") {\n");
    }
    if (need_cast) {
      EmitIndexCastToFrontend(y_index, y_index.Str() + "[0]", "tmp_argmax_index_saved[0]", api_param->api_post_process);
    } else {
      api_param->api_post_process.emplace_back("ArgmaxCopy(" + y_index.Str() + "[0], tmp_argmax_index_saved[0], " +
                                               y_index.actual_size.Str() + ");\n");
    }
    if (defer_index_output) {
      api_param->api_post_process.emplace_back("}\n");
    }
    if (x.isAr) {
      api_param->api_post_process.emplace_back(
          "accumulated_offset += " + tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size.Str() + ";\n");
    } else {
      api_param->api_post_process.emplace_back("accumulated_offset += first_actual;\n");
    }
    // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
    api_param->api_post_process.emplace_back("}\n");
  }

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgmaxWithValueApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                                    std::stringstream &ss) const {
  if (multi_reduce_) {
    ss << api_param.cal_count.ToStr(tiler) << ", false, accumulated_offset, " << valid_r_expr_ << ");" << std::endl;
  } else {
    ss << api_param.cal_count.ToStr(tiler) << ", false, 0, " << valid_r_expr_ << ");" << std::endl;
  }
  return af::SUCCESS;
}

Status RegArgMaxPhase2WithValueApiCall::BuildApiParam(
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
  GE_CHK_STATUS_RET(GetArgmaxValueDtypeName(this->api_name_, value, y_index, value_dtype_name),
                    "GetArgmaxValueDtypeName failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  BuildArgmaxCommonPreProcess(tpipe, value, y_value, api_param->api_pre_process, &index);

  const bool need_cast = NeedIndexCast(y_index);
  const std::string y_index_offset = tpipe.tiler.TensorVectorizedOffset(current_axis, y_index);
  std::string index_dst = y_index.Str();
  if (need_cast) {
    DeclareIndexTmp(tpipe, this->tmp_buf_id, "tmp_argmax_index", 0L, api_param->api_pre_process);
    index_dst = "tmp_argmax_index";
  }

  api_param->api_name = "ArgMaxMultiRPhase2WithValueVectorExtend";
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
    EmitIndexCastToFrontend(y_index, y_index.Str() + "[" + y_index_offset + "]", "tmp_argmax_index[0]",
                            api_param->api_post_process);
  }

  // 结束代码块（与 ReduceMergedSizeCodeGen 生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegArgMaxPhase2WithValueApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
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

}  // namespace codegen
