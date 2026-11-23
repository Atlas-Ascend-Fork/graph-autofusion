/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_api_call_base.h"
#include "reduce_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"
#include "codegen/expression_convert_struct.h"
#include "codegen_api_param/codegen_api_param.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;
using namespace reduce_base;

#define ARGMAXMULTIRPHASE_OUTPUT_AND_INPUT_NUM (2)

int64_t ReduceApiCall::GetTmpBufIdByLifeTime(int64_t life_time, const std::string &api_name) const {
  auto it = this->tmp_buf_id.find(life_time);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "ReduceApiCall(%s) cannot find tmp buffer id for life_time=%ld.",
                 api_name.c_str(), life_time);
  return it->second;
}

Status ReduceApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                    const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  auto iter = reduce_type_map.find(this->api_name_);
  GE_CHK_BOOL_RET_STATUS(iter != reduce_type_map.end(), af::FAILED, "Codegen unsupported reduce api::%s",
                         this->api_name_.c_str());
  auto &[type_value, instr_type] = iter->second;

  auto x = inputs[0].get();
  auto y = outputs[0].get();

  // 获取tmp_buf复用TBuf的id
  int64_t id = GetTmpBufIdByLifeTime(-1L, this->api_name_);

  std::string reduce_pattern;
  GetIsArAndPattern(x, y, x.isAr, reduce_pattern);
  CheckReduceSpecificParamsForCodegen({this->node, this->api_name_, &tpipe, &x, &y, current_axis.back()});

  // 获取dtype_name：Argmax系列算子使用value类型（x.dtype），其他算子使用输出类型（y.dtype）
  std::string dtype_name;
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, x, y, dtype_name),
                    "Codegen get dtype name failed for api:%s", this->api_name_.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  ReduceMergedSizeCodeGen(tpipe, api_param->api_pre_process, x, y);

  ReduceDimACodeGen(x, this->api_name_, api_param->api_pre_process);

  ReduceInitCodeGen(x, y, type_value, api_param->api_pre_process, tpipe, dtype_name);

  // 生成accumulated_offset变量声明（Argmax和ArgMaxMultiRPhase1需要）
  GenAccumulatedOffsetDeclForArgmax(this->api_name_, x, y, tpipe, api_param->api_pre_process);

  api_param->api_pre_process.emplace_back("uint32_t tmp_reduce_shape[] = {first_actual, last};\n");

  // Mean算子实际调用ReduceSum（或ReduceSumInt32），其他算子映射到真实API名
  std::string new_api_name = this->api_name_ == "ReduceMean" ? "ReduceSum" : this->api_name_;
  // ReduceSum + int32_t 特殊处理：实际调用 ReduceSumInt32
  if (new_api_name == "ReduceSum" && dtype_name == "int32_t") {
    new_api_name = "ReduceSumInt32";
  }

  if (!IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back())) {
    if (this->api_name_ == "Argmax") {
      api_param->api_name = "ArgMaxExtend";
      api_param->template_params.emplace_back("int64_t");
      api_param->template_params.emplace_back(dtype_name);
      api_param->template_params.emplace_back(reduce_pattern);
      api_param->output_params.emplace_back(
          y.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y)));
      api_param->input_params.emplace_back(
          x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
      api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);
      api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");
    } else {
      // 普通Reduce 不需要multi_reduce（包括Mean在内）
      api_param->api_name = new_api_name;
      api_param->template_params.emplace_back(dtype_name);
      api_param->template_params.emplace_back(reduce_pattern);
      api_param->template_params.emplace_back("false");
      api_param->output_params.emplace_back(
          y.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y)));
      api_param->input_params.emplace_back(
          x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
      api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);
      api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

      // Mean算子在非multi_reduce时需要额外的后处理（除法）
      if (this->api_name_ == "ReduceMean") {
        ReduceMeanCodeGen(dtype_name, tpipe, x, y, api_param->api_post_process);
      }
    }
  } else {
    int64_t tmp_lifetime_0_id = GetTmpBufIdByLifeTime(0L, this->api_name_);

    if (this->api_name_ == "Argmax") {
      // Argmax 特殊处理：需要维护全局最大值和索引，以及累加的 offset
      // Argmax 有三个额外的tmp_buf：
      //   - desc2(life_time=0)：索引的临时存储
      //   - desc3(life_time=1)：当前迭代的value临时存储
      //   - desc4(life_time=2)：value的历史最大结果

      // 索引：生命周期0的 tmp_argmax_index (desc2)
      api_param->api_pre_process.emplace_back("LocalTensor<int64_t> tmp_argmax_index;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax_index = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_0_id) +
                                              ".template ReinterpretCast<int64_t>();\n");

      // 当前计算的value：生命周期1的 tmp_argmax_value (desc3)
      int64_t tmp_lifetime_1_id = GetTmpBufIdByLifeTime(1L, "Argmax");
      api_param->api_pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_argmax_value;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax_value = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_1_id) + ".template ReinterpretCast<" +
                                              dtype_name + ">();\n");

      // 历史最大value：生命周期2的 tmp_argmax_value_saved (desc4)
      int64_t tmp_lifetime_2_id = GetTmpBufIdByLifeTime(2L, "Argmax");
      api_param->api_pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_argmax_value_saved;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax_value_saved = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_2_id) + ".template ReinterpretCast<" +
                                              dtype_name + ">();\n");

      // ArgMaxWithValueExtend 核心 API 参数
      api_param->api_name = "ArgMaxWithValueExtend";
      api_param->template_params.emplace_back("int64_t");
      api_param->template_params.emplace_back(dtype_name);
      api_param->template_params.emplace_back(reduce_pattern);
      api_param->output_params.emplace_back("tmp_argmax_index", true, CombinedExprFactory::Constant(0));
      api_param->output_params.emplace_back("tmp_argmax_value", true, CombinedExprFactory::Constant(0));
      api_param->input_params.emplace_back(
          x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
      api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");
      api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);

      // 后处理段：Argmax multi_reduce
      api_param->api_post_process.emplace_back("AscendC::PipeBarrier<PIPE_V>();\n");
      // 如果是第一次迭代，直接赋值；否则使用 UpdateMaxIndexAndValue 更新全局最大值和索引
      api_param->api_post_process.emplace_back("uint32_t temp_size_index = " + KernelUtils::SizeAlign() + "(" +
                                               y.actual_size.Str() + ", 4);\n");
      api_param->api_post_process.emplace_back("uint32_t temp_size_value = " + KernelUtils::SizeAlign() + "(" +
                                               y.actual_size.Str() + ", 32/sizeof(" + dtype_name + "));\n");
      api_param->api_post_process.emplace_back("if (" + tpipe.tiler.GetAxis(current_axis.back()).Str() + " == 0) {\n");
      api_param->api_post_process.emplace_back("DataCopyExtend(" + y.Str() +
                                               "[0], tmp_argmax_index[0], temp_size_index);\n");
      api_param->api_post_process.emplace_back(
          "DataCopyExtend(tmp_argmax_value_saved[0], tmp_argmax_value[0], temp_size_value);\n");
      api_param->api_post_process.emplace_back("} else {\n");
      // 使用当前的 accumulated_offset（标量）来更新全局最大值和索引
      // tmp_argmax_value_current是当前计算的value，tmp_argmax_value_saved是历史最大
      api_param->api_post_process.emplace_back("UpdateMaxIndexAndValue<" + dtype_name +
                                               ">(tmp_argmax_index[0], tmp_argmax_value[0], " + y.Str() +
                                               "[0], tmp_argmax_value_saved[0], " + "accumulated_offset, " +
                                               tpipe.tmp_buf.name + "_" + std::to_string(id) + ", temp_size_value);\n");
      api_param->api_post_process.emplace_back("}\n");

      // 累加 offset：accumulated_offset += 本次处理的 R 轴 actual_size
      // 根据 AR/RA 模式决定累加哪个值
      if (x.isAr) {
        // AR 模式：累加 vectorized_axis 的 actual_size
        api_param->api_post_process.emplace_back(
            "accumulated_offset += " + tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size.Str() + ";\n");
      } else {
        // RA 模式：累加 first_actual
        api_param->api_post_process.emplace_back("accumulated_offset += first_actual;\n");
      }
    } else if (this->api_name_ == "ArgMaxMultiRPhase1") {
      // ArgMaxMultiRPhase1特殊处理：在IsNeedMultiReduce分支中也需要处理
      // ArgMaxMultiRPhase1 有两个额外的tmp_buf：
      //   - desc2(life_time=0)：索引的临时存储
      //   - desc3(life_time=1)：当前迭代的value临时存储
      // 注意：ArgMaxMultiRPhase1本身自带两个输出，所以不需要历史最大值的tmp_buf

      // 索引：生命周期0的 tmp_argmax1_index (desc2)
      api_param->api_pre_process.emplace_back("LocalTensor<int64_t> tmp_argmax1_index;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax1_index = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_0_id) +
                                              ".template ReinterpretCast<int64_t>();\n");

      // 当前value：生命周期1的 tmp_argmax1_value (desc3)
      int64_t tmp_lifetime_1_id = GetTmpBufIdByLifeTime(1L, "ArgMaxMultiRPhase1");
      api_param->api_pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_argmax1_value;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax1_value = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_1_id) + ".template ReinterpretCast<" +
                                              dtype_name + ">();\n");

      // ArgMaxWithValueExtend 核心 API 参数
      api_param->api_name = "ArgMaxWithValueExtend";
      api_param->template_params.emplace_back("int64_t");
      api_param->template_params.emplace_back(dtype_name);
      api_param->template_params.emplace_back(reduce_pattern);
      api_param->output_params.emplace_back("tmp_argmax1_index", true, CombinedExprFactory::Constant(0));
      api_param->output_params.emplace_back("tmp_argmax1_value", true, CombinedExprFactory::Constant(0));
      api_param->input_params.emplace_back(
          x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
      api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);
      api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

      // 后处理段：ArgMaxMultiRPhase1
      GE_ASSERT_TRUE(outputs.size() >= ARGMAXMULTIRPHASE_OUTPUT_AND_INPUT_NUM,
                     "ArgMaxMultiRPhase1 requires at least 2 outputs, actual size: %zu.", outputs.size());
      auto y_value = outputs[0].get();
      auto y_index = outputs[1].get();

      api_param->api_post_process.emplace_back("AscendC::PipeBarrier<PIPE_V>();\n");
      api_param->api_post_process.emplace_back("uint32_t temp_size_index = " + KernelUtils::SizeAlign() + "(" +
                                               y_index.actual_size.Str() + ", 4);\n");
      api_param->api_post_process.emplace_back("uint32_t temp_size_value = " + KernelUtils::SizeAlign() + "(" +
                                               y_value.actual_size.Str() + ", 32/sizeof(" + dtype_name + "));\n");
      api_param->api_post_process.emplace_back("if (" + tpipe.tiler.GetAxis(current_axis.back()).Str() + " == 0) {\n");
      // 第一次迭代：直接复制到输出
      api_param->api_post_process.emplace_back("DataCopyExtend(" + y_value.Str() +
                                               "[0], tmp_argmax1_value[0], temp_size_value);\n");
      api_param->api_post_process.emplace_back("DataCopyExtend(" + y_index.Str() +
                                               "[0], tmp_argmax1_index[0], temp_size_index);\n");
      api_param->api_post_process.emplace_back("} else {\n");
      // 后续迭代：使用UpdateMaxIndexAndValue更新全局最大值和索引
      // 注意：这里需要offset，offset = 当前核id * R轴每块大小 + 累加的offset
      // 暂时传入accumulated_offset，需要在循环外初始化
      api_param->api_post_process.emplace_back(
          "UpdateMaxIndexAndValue<" + dtype_name + ">(tmp_argmax1_index[0], tmp_argmax1_value[0], " + y_index.Str() +
          "[0], " + y_value.Str() + "[0], " + "accumulated_offset + block_dim * r_axis_block_size, " +
          tpipe.tmp_buf.name + "_" + std::to_string(id) + ", temp_size_value);\n");
      api_param->api_post_process.emplace_back("}\n");

      // 累加 offset：accumulated_offset += 本次处理的 R 轴 actual_size
      // 根据 AR/RA 模式决定累加哪个值
      if (x.isAr) {
        // AR 模式：累加 vectorized_axis 的 actual_size
        api_param->api_post_process.emplace_back(
            "accumulated_offset += " + tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size.Str() + ";\n");
      } else {
        // RA 模式：累加 first_actual
        api_param->api_post_process.emplace_back("accumulated_offset += first_actual;\n");
      }
    } else if (this->api_name_ == "ArgMaxMultiRPhase2") {
      // ArgMaxMultiRPhase2特殊处理：有两个输入和一个输出，需要处理多次迭代
      //   - inputs[0]: value (来自Phase1的value输出，是该块的最大值)
      //   - inputs[1]: index (来自Phase1的index输出，是该块最大值的位置)
      //   - outputs[0]: 最终的index输出
      // 注意：Phase2也是R轴分核，需要调用ArgmaxExtend和ReduceMax

      GE_ASSERT_TRUE(inputs.size() >= ARGMAXMULTIRPHASE_OUTPUT_AND_INPUT_NUM,
                     "ArgMaxMultiRPhase2 requires at least 2 inputs, actual size: %zu.", inputs.size());
      GE_ASSERT_TRUE(outputs.size() >= 1, "ArgMaxMultiRPhase2 requires at least 1 output, actual size: %zu.",
                     outputs.size());  // ArgMaxMultiRPhase2有1个输出
      auto x_value = inputs[0].get();
      auto x_index = inputs[1].get();

      // 索引：生命周期0的第一个 tmp_argmax2_index
      api_param->api_pre_process.emplace_back("LocalTensor<int64_t> tmp_argmax2_index;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax2_index = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_0_id) +
                                              ".template ReinterpretCast<int64_t>();\n");

      // 当前计算的value：生命周期1的 tmp_argmax2_value (desc3)
      int64_t tmp_lifetime_1_id = GetTmpBufIdByLifeTime(1L, "ArgMaxMultiRPhase2");
      api_param->api_pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_argmax2_value;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax2_value = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_1_id) + ".template ReinterpretCast<" +
                                              dtype_name + ">();\n");

      // 历史最大value：生命周期2的 tmp_argmax2_value_saved (desc4)
      int64_t tmp_lifetime_2_id = GetTmpBufIdByLifeTime(2L, "ArgMaxMultiRPhase2");
      api_param->api_pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_argmax2_value_saved;\n");
      api_param->api_pre_process.emplace_back("tmp_argmax2_value_saved = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_2_id) + ".template ReinterpretCast<" +
                                              dtype_name + ">();\n");

      // 调用 ArgMaxWithValueExtend 获取本次迭代的局部索引和最大值
      api_param->api_name = "ArgMaxWithValueExtend";
      api_param->template_params.emplace_back("int64_t");
      api_param->template_params.emplace_back(dtype_name);
      api_param->template_params.emplace_back(reduce_pattern);
      api_param->output_params.emplace_back("tmp_argmax2_index", true, CombinedExprFactory::Constant(0));
      api_param->output_params.emplace_back("tmp_argmax2_value", true, CombinedExprFactory::Constant(0));
      api_param->input_params.emplace_back(
          x_value.Str(), true,
          CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x_value)));
      api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);
      api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

      // 后处理段：ArgMaxMultiRPhase2
      api_param->api_post_process.emplace_back("AscendC::PipeBarrier<PIPE_V>();\n");
      // 如果是第一次迭代，直接赋值；否则使用UpdateMaxIndexAndValue更新全局最大值和索引
      api_param->api_post_process.emplace_back("uint32_t temp_size_index = " + KernelUtils::SizeAlign() + "(" +
                                               y.actual_size.Str() + ", 4);\n");
      api_param->api_post_process.emplace_back("uint32_t temp_size_value = " + KernelUtils::SizeAlign() + "(" +
                                               y.actual_size.Str() + ", 32/sizeof(" + dtype_name + "));\n");
      api_param->api_post_process.emplace_back("if (" + tpipe.tiler.GetAxis(current_axis.back()).Str() + " == 0) {\n");
      api_param->api_post_process.emplace_back("DataCopyExtend(" + y.Str() +
                                               "[0], tmp_argmax2_index[0], temp_size_index);\n");
      api_param->api_post_process.emplace_back(
          "DataCopyExtend(tmp_argmax2_value_saved[0], tmp_argmax2_value[0], temp_size_value);\n");
      api_param->api_post_process.emplace_back("} else {\n");
      // 使用UpdateMaxIndexAndValue更新，注意这里offset传入0（因为Phase1已经处理了offset）
      api_param->api_post_process.emplace_back("UpdateMaxIndexAndValue<" + dtype_name +
                                               ">(tmp_argmax2_index[0], tmp_argmax2_value[0], " + y.Str() +
                                               "[0], tmp_argmax2_value_saved[0], " + "0, " + tpipe.tmp_buf.name + "_" +
                                               std::to_string(id) + ", temp_size_value);\n");
      api_param->api_post_process.emplace_back("}\n");
    } else {
      // 普通Reduce multi_reduce（包括Mean在内）：声明 tmp_reduce 变量
      api_param->api_pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_reduce;\n");
      api_param->api_pre_process.emplace_back("tmp_reduce = " + tpipe.tmp_buf.name + "_" +
                                              std::to_string(tmp_lifetime_0_id) + ".template ReinterpretCast<" +
                                              dtype_name + ">();\n");

      // 核心 API 参数
      api_param->api_name = new_api_name;
      api_param->template_params.emplace_back(dtype_name);
      api_param->template_params.emplace_back(reduce_pattern);
      api_param->template_params.emplace_back("false");
      api_param->output_params.emplace_back("tmp_reduce", true, CombinedExprFactory::Constant(0));
      api_param->input_params.emplace_back(
          x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
      api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);
      api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

      // 后处理段：普通Reduce multi_reduce
      api_param->api_post_process.emplace_back("AscendC::PipeBarrier<PIPE_V>();\n");
      api_param->api_post_process.emplace_back("uint32_t temp_size = " + KernelUtils::SizeAlign() + "(" +
                                               y.actual_size.Str() + ", 32/sizeof(" + dtype_name + "));\n");
      api_param->api_post_process.emplace_back("if (" + tpipe.tiler.GetAxis(current_axis.back()).Str() + " == 0) {\n");
      api_param->api_post_process.emplace_back("DataCopyExtend(" + y.Str() + "[0], tmp_reduce[0], temp_size);\n");
      api_param->api_post_process.emplace_back("} else {\n");
      api_param->api_post_process.emplace_back("AscendC::" + instr_type + "(" + y.Str() + "[0], tmp_reduce[0], " +
                                               y.Str() + "[0], temp_size);\n");
      api_param->api_post_process.emplace_back("}\n");
    }
  }

  // 结束代码块（与ReduceMergedSizeCodeGen生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status ReduceApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                        std::stringstream &ss) const {
  // Argmax 系列（ArgMaxExtend / ArgMaxWithValueExtend）最后一个布尔参数为 false（src_inner_pad 未使用），
  // 普通 Reduce 为 true，与重构前 Generate 内直接拼接的语义保持一致
  const bool is_argmax = (api_param.api_name == "ArgMaxExtend") || (api_param.api_name == "ArgMaxWithValueExtend");
  ss << api_param.cal_count.ToStr(tiler) << ", " << (is_argmax ? "false" : "true") << ");" << std::endl;
  return af::SUCCESS;
}

static ApiCallRegister<ReduceApiCall> register_reduce_api_call("ReduceApiCall");

}  // namespace codegen
