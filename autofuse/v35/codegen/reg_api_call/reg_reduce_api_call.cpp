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

namespace {
bool IsLogicalReduceApi(const std::string &api_name) {
  return api_name == "ReduceAny" || api_name == "ReduceAll";
}

Status GetReduceExecDtypeName(const std::string &api_name, ge::DataType dtype, std::string &dtype_name) {
  if (dtype == ge::DT_BOOL) {
    GE_CHK_BOOL_RET_STATUS(IsLogicalReduceApi(api_name), af::FAILED,
                           "Codegen(reg reduce) unsupported bool dtype for api:%s", api_name.c_str());
    dtype_name = "uint8_t";
    return af::SUCCESS;
  }
  return Tensor::DtypeName(dtype, dtype_name);
}

void AppendBoolLogicalReducePreprocess(const Tensor &x, const Tensor &y, const std::string &x_name_after_reinterpret,
                                       const std::string &output_name, std::vector<std::string> &pre_process) {
  pre_process.emplace_back("LocalTensor<uint8_t> " + x_name_after_reinterpret + " = " + x.Str() +
                           ".template ReinterpretCast<uint8_t>();\n");
  pre_process.emplace_back("LocalTensor<uint8_t> " + output_name + " = " + y.Str() +
                           ".template ReinterpretCast<uint8_t>();\n");
}

void AppendMultiReduceTmp(const TPipe &tpipe, const std::string &dtype_name, int64_t tmp_reduce_id,
                          std::vector<std::string> &pre_process) {
  pre_process.emplace_back("LocalTensor<" + dtype_name + "> tmp_reduce;\n");
  pre_process.emplace_back("tmp_reduce = " + tpipe.tmp_buf.name + "_" + std::to_string(tmp_reduce_id) +
                           ".template ReinterpretCast<" + dtype_name + ">();\n");
}

void AppendReduceInputOutputParams(const TPipe &tpipe, const Tensor &x, const Tensor &y,
                                   const std::vector<ascir::AxisId> &current_axis, const std::string &input_name,
                                   const std::string &output_name, bool need_multi_reduce, CodegenApiParam &api_param) {
  if (need_multi_reduce) {
    api_param.output_params.emplace_back("tmp_reduce", true, CombinedExprFactory::Constant(0));
    api_param.input_params.emplace_back(
        input_name, true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
    return;
  }
  api_param.output_params.emplace_back(
      output_name, true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, y)));
  api_param.input_params.emplace_back(
      input_name, true, CombinedExprFactory::SymbolVar(tpipe.tiler.TensorVectorizedOffset(current_axis, x)));
}

void AppendReduceMeanPostProcess(const TPipe &tpipe, const Tensor &x, const Tensor &y, const std::string &dtype_name,
                                 std::vector<std::string> &post_process) {
  std::string mutable_dtype_name = dtype_name;
  ReduceMeanCodeGen(mutable_dtype_name, tpipe, x, y, post_process);
}

void AppendMultiReducePostProcess(const TPipe &tpipe, const Tensor &y, const std::vector<ascir::AxisId> &current_axis,
                                  const std::string &output_name, const std::string &dtype_name,
                                  const std::string &instr_type, std::vector<std::string> &post_process) {
  post_process.emplace_back("AscendC::PipeBarrier<PIPE_V>();\n");
  post_process.emplace_back("uint32_t temp_size = " + KernelUtils::SizeAlign() + "(" + y.actual_size.Str() +
                            ", 32/sizeof(" + dtype_name + "));\n");
  post_process.emplace_back("if (" + tpipe.tiler.GetAxis(current_axis.back()).Str() + " == 0) {\n");
  post_process.emplace_back("DataCopyExtend(" + output_name + "[0], tmp_reduce[0], temp_size);\n");
  post_process.emplace_back("} else {\n");
  post_process.emplace_back("AscendC::" + instr_type + "(" + output_name + "[0], tmp_reduce[0], " + output_name +
                            "[0], temp_size);\n");
  post_process.emplace_back("}\n");
}

std::string BuildReduceSemanticFeature(bool is_ar, bool is_reuse_source, bool need_multi_reduce) {
  std::string semantic_feature = is_ar ? semantic::kAr : semantic::kRa;
  semantic_feature +=
      is_reuse_source ? "/" + std::string(semantic::kReuseSrc) : "/" + std::string(semantic::kNoReuseSrc);
  if (need_multi_reduce) {
    semantic_feature += "/" + std::string(semantic::kMultiRound);
  }
  return semantic_feature;
}

Status GetReduceTmpBufferId(const std::unordered_map<int64_t, int64_t> &tmp_buf_id, int64_t life_time_axis_id,
                            const std::string &buffer_name, int64_t &buffer_id) {
  const auto iterator = tmp_buf_id.find(life_time_axis_id);
  GE_CHK_BOOL_RET_STATUS(iterator != tmp_buf_id.end(), af::FAILED,
                         "RegReduceApiCall(%s) cannot find tmp buffer id to use.", buffer_name.c_str());
  buffer_id = iterator->second;
  return af::SUCCESS;
}

Status GetReduceMainTmpBufferId(const std::unordered_map<int64_t, int64_t> &tmp_buf_id,
                                const std::string &is_reuse_source, int64_t &buffer_id) {
  const auto iterator = tmp_buf_id.find(-1L);
  if (iterator != tmp_buf_id.end()) {
    buffer_id = iterator->second;
    return af::SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS(is_reuse_source == "true", af::FAILED,
                         "RegReduceApiCall(id) cannot find tmp buffer id to use.");
  return af::SUCCESS;
}

Status PrepareReduceApiParam(const TPipe &tpipe, const Tensor &x, const Tensor &y, const std::string &api_name,
                             const std::string &dtype_name, const std::string &x_name_after_reinterpret,
                             int32_t type_value, const std::string &reduce_pattern, const std::string &is_reuse_source,
                             std::shared_ptr<CodegenApiParam> &api_param) {
  api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);

  ReduceMergedSizeCodeGen(tpipe, api_param->api_pre_process, x, y);
  if (y.dtype == ge::DT_BOOL && IsLogicalReduceApi(api_name)) {
    const std::string output_name = y.Str() + "_exec";
    AppendBoolLogicalReducePreprocess(x, y, x_name_after_reinterpret, output_name, api_param->api_pre_process);
  }
  ReduceDimACodeGen(x, api_name, api_param->api_pre_process);
  ReduceInitCodeGen(x, y, type_value, api_param->api_pre_process, tpipe, dtype_name, x_name_after_reinterpret);
  api_param->api_pre_process.emplace_back("uint32_t tmp_reduce_shape[] = {first_actual, last};\n");
  api_param->api_name = api_name == "ReduceMean" ? "ReduceSumExtend" : api_name + "Extend";
  api_param->template_params.emplace_back(dtype_name);
  api_param->template_params.emplace_back(reduce_pattern);
  api_param->template_params.emplace_back(is_reuse_source);
  return af::SUCCESS;
}

void AppendReducePostProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &x,
                             const Tensor &y, const std::string &api_name, const std::string &output_name,
                             const std::string &dtype_name, const std::string &instr_type, bool need_multi_reduce,
                             CodegenApiParam &api_param) {
  if (!need_multi_reduce && api_name == "ReduceMean") {
    // Mean算子在非multi_reduce时需要额外的后处理（除法）
    AppendReduceMeanPostProcess(tpipe, x, y, dtype_name, api_param.api_post_process);
  }
  if (need_multi_reduce) {
    AppendMultiReducePostProcess(tpipe, y, current_axis, output_name, dtype_name, instr_type,
                                 api_param.api_post_process);
  }
}

Status RegisterReduceApiParam(const af::AscNodePtr &node, const std::string &node_name,
                              std::shared_ptr<CodegenApiParam> &api_param) {
  GE_CHK_STATUS_RET(CodegenApiParam::Register(node, api_param), "CodegenApiParam Register failed for node %s",
                    node_name.c_str());
  return af::SUCCESS;
}
}  // namespace

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
  int64_t id = -1L;
  GE_CHK_STATUS_RET(GetReduceMainTmpBufferId(this->tmp_buf_id, is_reuse_source_, id));

  auto iter = reduce_type_map.find(this->api_name_);
  GE_CHK_BOOL_RET_STATUS(iter != reduce_type_map.end(), af::FAILED, "Codegen unsupported reg reduce api::%s",
                         this->api_name_.c_str());
  auto &[type_value, instr_type] = iter->second;

  auto x = inputs[0].get();
  auto y = outputs[0].get();

  std::string reduce_pattern;
  GetIsArAndPattern(x, y, x.isAr, reduce_pattern);
  CheckReduceSpecificParamsForCodegen(
      {this->node, this->api_name_, &tpipe, &x, &y, current_axis.back(), true, is_reuse_source_ == "true"});

  std::string dtype_name;
  GE_CHK_STATUS_RET(GetReduceExecDtypeName(this->api_name_, y.dtype, dtype_name),
                    "Codegen(reg reduce) get data type:%d failed", static_cast<int32_t>(y.dtype));
  GELOGI("Tensor::DtypeName(y.dtype) == %s", dtype_name.c_str());

  const bool is_bool_logical_reduce = y.dtype == ge::DT_BOOL && IsLogicalReduceApi(this->api_name_);
  const std::string input_name = is_bool_logical_reduce ? x.Str() + "_exec" : x.Str();
  const std::string output_name = is_bool_logical_reduce ? y.Str() + "_exec" : y.Str();

  std::shared_ptr<CodegenApiParam> api_param;
  GE_CHK_STATUS_RET(PrepareReduceApiParam(tpipe, x, y, this->api_name_, dtype_name, input_name, type_value,
                                          reduce_pattern, is_reuse_source_, api_param),
                    "Codegen(reg reduce) prepare api param failed for node %s", this->node_name.c_str());

  bool need_multi_reduce = IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back());

  if (need_multi_reduce) {
    int64_t life_time_axis_id = 0L;
    int64_t tmp_reduce_id = -1L;
    GE_CHK_STATUS_RET(GetReduceTmpBufferId(this->tmp_buf_id, life_time_axis_id, "tmp_reduce_id", tmp_reduce_id));
    AppendMultiReduceTmp(tpipe, dtype_name, tmp_reduce_id, api_param->api_pre_process);
  }

  // need_multi_reduce 时输出到 tmp_reduce[0]
  AppendReduceInputOutputParams(tpipe, x, y, current_axis, input_name, output_name, need_multi_reduce, *api_param);

  // 设置tmp_buf_name（如果不是reuse_source）
  if (is_reuse_source_ != "true") {
    api_param->tmp_buf_name = tpipe.tmp_buf.name + "_" + std::to_string(id);
  }
  api_param->cal_count = CombinedExprFactory::SymbolVar("tmp_reduce_shape");

  AppendReducePostProcess(tpipe, current_axis, x, y, this->api_name_, output_name, dtype_name, instr_type,
                          need_multi_reduce, *api_param);

  api_param->semantic_feature = BuildReduceSemanticFeature(x.isAr, is_reuse_source_ == "true", need_multi_reduce);

  // 结束代码块（与ReduceMergedSizeCodeGen生成的 { 对应）
  api_param->api_post_process.emplace_back("}\n");

  return RegisterReduceApiParam(this->node, this->node_name, api_param);
}

Status RegReduceApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                           std::stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ", true);" << std::endl;
  return af::SUCCESS;
}

static ApiCallRegister<RegReduceApiCall> register_reduce_api_call("RegReduceApiCall");

}  // namespace codegen
