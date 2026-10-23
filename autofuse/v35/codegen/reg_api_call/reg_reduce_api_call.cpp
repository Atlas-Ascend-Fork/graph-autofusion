/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reg_reduce_api_call.h"

#include "codegen/api_call/reduce/reduce_api_call_base.h"
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
#include "ascir/reg_func/reduce_reuse_utils.h"

namespace codegen {
namespace semantic = ascgen_utils::template_semantic;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;
using namespace reduce_base;

Status RegReduceApiCall::ParseAttr(const ascir::NodeView &node) {
  const auto &in_node = af::ascir::GetReduceInputNode(node);
  GE_CHECK_NOTNULL(in_node);
  if (af::ascir::HasSingleOutNode(in_node) && !af::ascir::HasUpstreamBroadcastOnReduceAxis(*node)) {
    is_reuse_source_ = "true";
  }
  return af::SUCCESS;
}

Status RegReduceApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end() || is_reuse_source_ == "true",
                 "RegReduceApiCall(id) cannot find tmp buffer id to use.");
  if (it != this->tmp_buf_id.end()) {
    id = it->second;
  }

  auto iter = reduce_type_map.find(this->api_name_);
  GE_CHK_BOOL_RET_STATUS(iter != reduce_type_map.end(), af::FAILED, "Codegen unsupported reg reduce api::%s",
                         this->api_name_.c_str());
  auto &[type_value, instr_type] = iter->second;

  auto x = inputs[0].get();
  auto y = outputs[0].get();

  std::string reduce_pattern;
  GetIsArAndPattern(y, x.isAr, reduce_pattern);
  CheckReduceSpecificParamsForCodegen(
      {this->node, this->api_name_, &tpipe, &x, &y, current_axis.back(), true, is_reuse_source_ == "true"});

  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen(reg reduce) get data type:%d failed",
                    static_cast<int32_t>(y.dtype));
  GELOGI("Tensor::DtypeName(y.dtype) == %s", dtype_name.c_str());

  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  ReduceMergedSizeCodeGen(tpipe, api_param->api_pre_process, x, y);

  ReduceDimACodeGen(x, this->api_name_, api_param->api_pre_process);

  ReduceInitCodeGen(x, y, type_value, api_param->api_pre_process, tpipe, dtype_name);
  api_param->api_pre_process.emplace_back("uint32_t tmp_reduce_shape[] = {first_actual, last};\n");
  std::string new_api_name = this->api_name_ == "ReduceMean" ? "ReduceSumExtend" : this->api_name_ + "Extend";
  api_param->api_name = new_api_name;
  api_param->template_params.emplace_back(dtype_name);
  api_param->template_params.emplace_back(reduce_pattern);
  api_param->template_params.emplace_back(is_reuse_source_);

  bool need_multi_reduce = IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back());

  if (need_multi_reduce) {
    life_time_axis_id = 0L;
    int64_t tmp_reduce_id = -1L;
    auto it_tmp = this->tmp_buf_id.find(life_time_axis_id);
    GE_ASSERT_TRUE(it_tmp != this->tmp_buf_id.end(),
                   "RegReduceApiCall(tmp_reduce_id) cannot find tmp buffer id to use.");
    tmp_reduce_id = it_tmp->second;
    api_param->api_pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_reduce;\n");
    api_param->api_pre_process.emplace_back("tmp_reduce = " + tpipe.tmp_buf.name + "_" + std::to_string(tmp_reduce_id) +
                                            ".template ReinterpretCast<" + dtype_name + ">();\n");
  }

  if (need_multi_reduce) {
    // need_multi_reduce 时输出到 tmp_reduce[0]
    api_param->output_params.emplace_back("tmp_reduce", true, CombinedExprFactory::Constant(0));
    api_param->input_params.emplace_back(
        x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
  } else {
    api_param->output_params.emplace_back(
        y.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y)));
    api_param->input_params.emplace_back(
        x.Str(), true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
  }

  // 设置tmp_buf_name（如果不是reuse_source）
  if (is_reuse_source_ != "true") {
    api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);
  }
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

  // 后处理代码
  if (!need_multi_reduce && this->api_name_ == "ReduceMean") {
    // Mean算子在非multi_reduce时需要额外的后处理（除法）
    ReduceMeanCodeGen(dtype_name, tpipe, x, y, api_param->api_post_process);
  }

  if (need_multi_reduce) {
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

  std::string semantic_feature = x.isAr ? semantic::kAr : semantic::kRa;
  semantic_feature +=
      is_reuse_source_ == "true" ? "/" + std::string(semantic::kReuseSrc) : "/" + std::string(semantic::kNoReuseSrc);
  if (need_multi_reduce) {
    semantic_feature += "/" + std::string(semantic::kMultiRound);
  }
  api_param->semantic_feature = semantic_feature;

  // 结束代码块（与ReduceMergedSizeCodeGen生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param), "CodegenApiParam Register failed for node %s",
                    this->node_name.c_str());
  return af::SUCCESS;
}

Status RegReduceApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                           std::stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ", true);" << std::endl;
  return af::SUCCESS;
}

static ApiCallRegister<RegReduceApiCall> register_reduce_api_call("RegReduceApiCall");

}  // namespace codegen
