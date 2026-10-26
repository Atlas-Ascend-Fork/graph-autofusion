/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "api_perf_utils.h"
#include "common_utils.h"
#include "att_utils.h"
#include "api_perf_register/perf_param.h"
#include "api_perf_register/api_perf_factory.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/soc_perf_config.h"
#include "utils/node_expr_id.h"

namespace att {
namespace {
float ParseOptionalFloatParam(const std::map<std::string, float> &param_map, const std::string &key,
                              float default_val) {
  const auto iter = param_map.find(key);
  if (iter != param_map.end()) {
    return iter->second;
  }
  return default_val;
}

Expr CalculateBlockCountByIndex(const std::vector<Expr> &dims, int32_t block_count_idx) {
  // block_count_idx = 0: 累积到倒数第1个（排除尾轴）
  // block_count_idx = 1: 累积到倒数第2个
  // block_count_idx = n: 累积到倒数第(n+1)个
  size_t end_offset = std::min(static_cast<size_t>(block_count_idx + 1), dims.size());
  return accumulate(dims.begin(), dims.begin() + end_offset, CreateExpr(1), [](Expr a, Expr b) { return a * b; });
}

// 限制stride 上限（静态 shape 时）
Expr LimitedStrideUpperBound(const Expr &stride, const Expr &upper_val) {
  const auto is_over_upper_bound = af::sym::Gt(stride, upper_val);
  bool is_over_upper_bound_val = false;
  if (is_over_upper_bound.IsConstExpr()) {
    is_over_upper_bound.GetConstValue(is_over_upper_bound_val);
    return is_over_upper_bound_val ? upper_val : stride;
  }
  return stride;
}

Expr GetDataTypeSizeExpr(const std::map<std::string, float> &param_map) {
  const auto iter = param_map.find("data_type_size");
  if (iter != param_map.end()) {
    return CreateExpr(static_cast<int32_t>(iter->second));
  }
  return CreateExpr(1);  // 默认为 1
}

void SetLoadVectorizedShape(const TensorShapeInfo &tensor_shape_info, NodeDetail &node_info) {
  node_info.vectorized_shape.clear();
  if (node_info.optype != kLoad || node_info.input_dims.empty() || tensor_shape_info.repeats.empty() ||
      tensor_shape_info.repeats.size() != tensor_shape_info.gm_strides.size() ||
      tensor_shape_info.repeats.size() != tensor_shape_info.strides.size()) {
    return;
  }

  if (!ascgen_utils::NeedAppendDataCopyTailAxis(tensor_shape_info.gm_strides.back(),
                                                tensor_shape_info.strides.back())) {
    return;
  }

  // codegen 在 DataCopyParams 的尾轴非 0/1 时补一个 repeat=1 的逻辑轴。
  // 使用合轴后的 input_dims 保留 case0 的 128*z0z1t 等 block_count，不能直接使用原始 repeats。
  node_info.vectorized_shape = node_info.input_dims;
  node_info.vectorized_shape.emplace_back(af::sym::kSymbolOne);
}

// 计算维度的惩罚项，非连续轴较多会对性能产生非常大的影响，需要额外的计算
Expr CalculateStridePenalty(int32_t block_count_idx, const Expr &stride, float penalty_coeff) {
  if (block_count_idx > 0 && penalty_coeff > 0.0f) {
    Expr block_count_idx_expr = CreateExpr(block_count_idx);
    Expr penalty_coeff_expr = CreateExpr(penalty_coeff);
    return af::sym::Mul(block_count_idx_expr, af::sym::Mul(stride, penalty_coeff_expr));
  }
  return CreateExpr(0);
}

std::string GetDefaultDataType(const std::string dtype) {
  static const std::map<std::string, std::string> kTypeToDType = {
      {"int8", "float16"},  {"uint8", "float16"},  {"uint16", "float16"}, {"int16", "float16"},   {"uint32", "float32"},
      {"int32", "float32"}, {"uint64", "float32"}, {"int64", "float32"},  {"float32", "float32"},
  };
  auto it = kTypeToDType.find(dtype);
  if (it != kTypeToDType.end()) {
    return it->second;
  }
  return "float16";
}

af::Status StringToJson(const std::string &json_str, Json &json) {
  std::stringstream ss;
  ss << json_str;
  try {
    ss >> json;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::PARAM_INVALID, "Failed to init json object, err = %s, json_str = %s", e.what(), json_str.c_str());
    return ge::PARAM_INVALID;
  }
  return af::SUCCESS;
}

af::Status LinearFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims, const Expr &stride,
                      Expr &res) {
  (void)stride;
  GE_ASSERT_TRUE(!dims.empty(), "Dims are empty.");
  GE_ASSERT_TRUE(param_map.find("k") != param_map.end(), "Param k not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("b") != param_map.end(), "Param b not found in param_map.");
  Expr k = CreateExpr(param_map.at("k"));
  Expr b = CreateExpr(param_map.at("b"));
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  res = k * dim_product + b;
  return af::SUCCESS;
}

af::Status LoadStoreStrideFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                               const Expr &stride, Expr &res) {
  GE_ASSERT_TRUE(!dims.empty(), "Dims are empty.");
  GE_ASSERT_TRUE(param_map.find("k") != param_map.end(), "Param k not found in param_map.");
  Expr k = CreateExpr(param_map.at("k"));
  Expr block_count = accumulate(dims.begin(), dims.end() - 1, CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  Expr op1 = af::sym::Mul(k, block_count);
  Expr op2 = af::sym::Mod(stride, CreateExpr(256));
  res = af::sym::Mul(op1, op2);
  GELOGD("stride[%s], k[%s], block_count[%s], perf_res[%s]", Str(stride).c_str(), Str(k).c_str(),
         Str(block_count).c_str(), Str(res).c_str());
  return af::SUCCESS;
}

// 检查节点是否有输出（空输出跳过重排）
bool ShouldSkipReorderForEmptyOutputs(const af::AscNodePtr &node) {
  if (node->outputs().empty()) {
    GELOGD("Node[%s,%s] has no outputs, skipping gm_strides reordering", node->GetNamePtr(), node->GetType().c_str());
    return true;
  }
  return false;
}

// 检查相关数据是否为空（空数据跳过重排）
bool ShouldSkipReorderForEmptyData(const af::AscNodePtr &node, const TensorShapeInfo &tensor,
                                   const std::vector<int64_t> &vectorized_axis, const std::vector<int64_t> &axis) {
  if (tensor.repeats.empty()) {
    GELOGD("Node[%s,%s] gm_strides is empty, skipping reordering", node->GetNamePtr(), node->GetType().c_str());
    return true;
  }
  if (vectorized_axis.empty() || axis.empty()) {
    GELOGD("Node[%s,%s] vectorized_axis or axis is empty, skipping gm_strides reordering", node->GetNamePtr(),
           node->GetType().c_str());
    return true;
  }
  return false;
}

// 检查是否需要重排（轴顺序是否不同）
bool NeedsReordering(const std::vector<int64_t> &vectorized_axis, const std::vector<int64_t> &axis) {
  size_t check_size = std::min({vectorized_axis.size(), axis.size()});
  for (size_t i = 0; i < check_size; ++i) {
    if (vectorized_axis[i] != axis[i]) {
      return true;
    }
  }
  return false;
}

// 找出 axis 里哪些轴未在 vectorized_axis 里出现，并记录对应的下标
std::vector<size_t> FindIndicesToRemove(const std::vector<int64_t> &vectorized_axis, const std::vector<int64_t> &axis) {
  std::vector<size_t> indices_to_remove;
  for (size_t i = 0; i < axis.size(); ++i) {
    int64_t axis_id = axis[i];
    bool found = false;
    for (size_t j = 0; j < vectorized_axis.size(); ++j) {
      if (vectorized_axis[j] == axis_id) {
        found = true;
        break;
      }
    }
    if (!found) {
      indices_to_remove.push_back(i);
    }
  }
  return indices_to_remove;
}

// 从 repeats 中删除指定下标的元素（从后往前删除以避免索引变化）
void RemoveIndicesFromRepeats(std::vector<Expr> &repeats, const std::vector<size_t> &indices_to_remove) {
  for (auto it = indices_to_remove.rbegin(); it != indices_to_remove.rend(); ++it) {
    size_t index_to_remove = *it;
    if (index_to_remove < repeats.size()) {
      repeats.erase(repeats.begin() + index_to_remove);
    }
  }
}

// 创建动态shape的StrideResult，处理非连续分支选择
StrideResult CreateDynamicStrideResult(const Expr &last_stride, const Expr &non_continuous_stride,
                                       const Expr &normal_stride, int32_t block_count_idx, bool is_ub_stride) {
  const char *var_prefix = is_ub_stride ? "ub_stride_select" : "gm_stride_select";
  Expr res = CreateExpr(var_prefix);

  auto normal_case = std::make_shared<IfCase>(normal_stride);
  auto non_continuous_case = std::make_shared<IfCase>(non_continuous_stride);

  constexpr int64_t kContinueLastStrideVal = 1L;
  TernaryOp ternary_op = TernaryOp(CondType::K_GT, last_stride, CreateExpr(kContinueLastStrideVal),
                                   std::move(non_continuous_case), std::move(normal_case));
  ternary_op.SetVariable(res);

  StrideResult result(res, block_count_idx);
  result.ternary_ops[res] = ternary_op;
  return result;
}

// 计算 block_count_idx：用于性能模型计算
int32_t CalculateBlockCountIdx(int32_t filtered_dim_size, bool actually_swap) {
  constexpr int32_t kBlockLenCountDim = 2;
  return actually_swap ? (filtered_dim_size - kBlockLenCountDim - 1) : (filtered_dim_size - kBlockLenCountDim);
}
}  // namespace

// 为性能模型参数添加额外参数（从 node_perf_info 中获取）
std::map<std::string, float> EnrichParamMapForModel(const std::map<std::string, float> &base_params,
                                                    const NodePerfInfo &node_perf_info,
                                                    const std::string &input_dtype) {
  std::map<std::string, float> param_map = base_params;
  // 添加 block_count_idx（从 node_perf_info 获取）
  param_map["block_count_idx"] = static_cast<float>(node_perf_info.block_count_idx);
  param_map["loop_mode_dim_count"] = static_cast<float>(node_perf_info.loop_mode_dim_count);
  GELOGD("Enriched param_map with block_count_idx=%d for optype=%s", node_perf_info.block_count_idx,
         node_perf_info.optype.c_str());
  // 添加 data_type_size（从 kDataTypeSizeMap 获取）
  const auto &data_type_size_iter = kDataTypeSizeMap.find(input_dtype);
  GE_ASSERT_TRUE(data_type_size_iter != kDataTypeSizeMap.end(), "Data type[%s] not found in kDataTypeSizeMap!",
                 input_dtype.c_str());
  int32_t data_type_size_val = 4;  // default value
  if (data_type_size_iter->second.IsConstExpr()) {
    data_type_size_iter->second.GetConstValue(data_type_size_val);
  }
  param_map["data_type_size"] = static_cast<float>(data_type_size_val);
  GELOGD("Enriched param_map with data_type_size=%d for dtype=%s", data_type_size_val, input_dtype.c_str());

  return param_map;
}

af::Status LoadStoreStrideV2Func(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                                 const Expr &stride, Expr &res) {
  GE_ASSERT_TRUE(!dims.empty(), "Dims are empty.");
  const auto k_iter = param_map.find("k");
  const auto u_iter = param_map.find("u");
  GE_ASSERT_TRUE(k_iter != param_map.end(), "Param k not found in param_map.");
  GE_ASSERT_TRUE(u_iter != param_map.end(), "Param u not found in param_map.");
  Expr k = CreateExpr(k_iter->second);
  Expr upper_val = CreateExpr(u_iter->second);
  // 首次发生Stride的轴索引（默认 0）
  int32_t block_count_idx = static_cast<int32_t>(ParseOptionalFloatParam(param_map, "block_count_idx", 0.0f));
  int32_t loop_mode_dim_count = static_cast<int32_t>(ParseOptionalFloatParam(param_map, "loop_mode_dim_count", 0.0f));
  // Stride惩罚项：k * block_count * stride * data_type_size
  const int32_t effective_block_count_idx = std::max(0, block_count_idx - loop_mode_dim_count);
  const size_t loop_mode_offset = std::min(static_cast<size_t>(std::max(0, loop_mode_dim_count)), dims.size());
  const std::vector<Expr> dma_dims(dims.begin() + loop_mode_offset, dims.end());
  Expr block_count = CalculateBlockCountByIndex(dma_dims, effective_block_count_idx);
  Expr stride_used = LimitedStrideUpperBound(stride * GetDataTypeSizeExpr(param_map), upper_val);
  stride_used = af::sym::Min(stride_used, upper_val);
  res = af::sym::Mul(k, af::sym::Mul(block_count, stride_used));
  GELOGD(
      "LoadStoreStrideV2: dims[%s], k=%s, block_count=%s, block_count_idx=%d, "
      "stride_used=%s, data_type_size=%s, loop_mode_dim_count=%d, res=%s",
      GetVecString(dims).c_str(), Str(k).c_str(), Str(block_count).c_str(), block_count_idx, Str(stride_used).c_str(),
      Str(GetDataTypeSizeExpr(param_map)).c_str(), loop_mode_dim_count, Str(res).c_str());

  return af::SUCCESS;
}

af::Status LoadStoreStrideV2WithPenaltyFunc(const std::map<std::string, float> &param_map,
                                            const std::vector<Expr> &dims, const Expr &stride, Expr &res) {
  GE_ASSERT_TRUE(!dims.empty(), "Dims are empty.");
  const auto k_iter = param_map.find("k");
  const auto u_iter = param_map.find("u");
  GE_ASSERT_TRUE(k_iter != param_map.end(), "Param k not found in param_map.");
  GE_ASSERT_TRUE(u_iter != param_map.end(), "Param u not found in param_map.");

  int32_t block_count_idx = static_cast<int32_t>(ParseOptionalFloatParam(param_map, "block_count_idx", 0.0f));
  float penalty_coeff = ParseOptionalFloatParam(param_map, "penalty_coeff", 0.0f);

  af::Status status = LoadStoreStrideV2Func(param_map, dims, stride, res);
  GE_ASSERT_TRUE(status == af::SUCCESS, "LoadStoreStrideV2Func failed.");

  Expr stride_used = LimitedStrideUpperBound(stride * GetDataTypeSizeExpr(param_map), CreateExpr(u_iter->second));
  Expr penalty = CalculateStridePenalty(block_count_idx, stride_used, penalty_coeff);
  res = af::sym::Add(res, penalty);

  GELOGD("LoadStoreStrideV2WithPenalty: penalty_coeff=%f, penalty=%s, final_res=%s", penalty_coeff,
         Str(penalty).c_str(), Str(res).c_str());

  return af::SUCCESS;
}

af::Status LoadStoreFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                         const Expr &stride, Expr &res, bool expand_data) {
  (void)stride;
  GE_ASSERT_TRUE(!dims.empty(), "Dims are empty.");
  GE_ASSERT_TRUE(param_map.find("h") != param_map.end(), "Param h not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("a") != param_map.end(), "Param a not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("b") != param_map.end(), "Param b not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("hl") != param_map.end(), "Param hl not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("data_type_size") != param_map.end(), "Param data_type_size not found in param_map.");
  Expr h = CreateExpr(param_map.at("h"));
  Expr a = CreateExpr(param_map.at("a"));
  Expr b = CreateExpr(param_map.at("b"));
  Expr hl = CreateExpr(param_map.at("hl"));
  Expr data_type_size = CreateExpr(static_cast<int32_t>(param_map.at("data_type_size")));
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  Expr ub_align = CreateExpr(32);
  Expr blockdim = CreateExpr("block_dim");
  Expr t = af::sym::Add(a, af::sym::Div(b, blockdim));
  Expr data_size = af::sym::Mul(dim_product, data_type_size);
  if (expand_data) {
    Expr blocklen = af::sym::Mul(dims[dims.size() - 1UL], data_type_size);
    data_size = af::sym::Mul(af::sym::Div(data_size, blocklen), kMte2UbStrideAlignSize);
  }
  auto cycles = af::sym::Div(data_size, t);
  // 32B非对齐的惩罚项，hl=1时需要加该惩罚项
  // penalty = (512 / 2 - (h + 512 / t)) * hl
  Expr penalty = hl * (CreateExpr(256) - (h + af::sym::Div(CreateExpr(512), t)));
  res = af::sym::Add(af::sym::Add(cycles, h), penalty);
  return af::SUCCESS;
}

af::Status LoadStoreFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                         const Expr &stride, Expr &res) {
  return LoadStoreFunc(param_map, dims, stride, res, false);
}

af::Status LoadUbStride(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                        const Expr &stride, Expr &res) {
  return LoadStoreFunc(param_map, dims, stride, res, true);
}

af::Status StoreFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims, const Expr &stride,
                     Expr &res) {
  (void)stride;
  GE_ASSERT_TRUE(!dims.empty(), "Dims are empty.");
  GE_ASSERT_TRUE(param_map.find("h") != param_map.end(), "Param h not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("ak") != param_map.end(), "Param ak not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("ab") != param_map.end(), "Param ab not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("bk") != param_map.end(), "Param bk not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("bb") != param_map.end(), "Param bb not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("data_type_size") != param_map.end(), "Param data_type_size not found in param_map.");
  Expr h = CreateExpr(param_map.at("h"));
  Expr ak = CreateExpr(param_map.at("ak"));
  Expr ab = CreateExpr(param_map.at("ab"));
  Expr bk = CreateExpr(param_map.at("bk"));
  Expr bb = CreateExpr(param_map.at("bb"));
  Expr data_type_size = CreateExpr(static_cast<int32_t>(param_map.at("data_type_size")));
  Expr blocklen = af::sym::Mul(dims[dims.size() - 1UL], data_type_size);
  Expr blockcount = accumulate(dims.begin(), dims.end() - 1, CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  GELOGD("Special case for store. Blocklen = [%s]Byte, Blockcount = [%s].", Str(blocklen).c_str(),
         Str(blockcount).c_str());
  Expr blockdim = CreateExpr("block_dim");
  Expr a = ak * blockdim + ab;
  Expr b = bk * blockdim + bb;
  res = (a * blocklen + b) * blockcount + h;
  return af::SUCCESS;
}
const std::map<std::string, AscendCApiPerfFunc> kModelMap = {
    {"SimpleLinear", LinearFunc},
    {"LoadUbStride", LoadUbStride},
    {"LoadStoreFunc", LoadStoreFunc},
    {"LoadStoreStrideFunc", LoadStoreStrideFunc},
    {"LoadStoreStrideV2Func", LoadStoreStrideV2Func},
    {"LoadStoreStrideV2WithPenaltyFunc", LoadStoreStrideV2WithPenaltyFunc},
    {"StoreFunc", StoreFunc},
};

const PerfParamTable *GetParamPerfTable() {
  const auto asc_ir_att_impl = ascgen_utils::GetAscIrAttImpl(kDefaultApi);
  GE_ASSERT_NOTNULL(asc_ir_att_impl);
  const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
  GE_ASSERT_NOTNULL(api_perf_name);
  auto api_perf = ApiPerfFactory::Instance().Create(api_perf_name);
  GE_ASSERT_NOTNULL(api_perf);
  return api_perf->GetPerfParam();
}

static af::Status IsSocPerfTarget(const SocPerfTarget &target, bool &is_soc_perf_target) {
  is_soc_perf_target = false;
  return RequiresFullSocVersion(target.npu_arch, is_soc_perf_target);
}

static af::Status GetTargetApiParams(const SocPerfTarget &target, std::shared_ptr<const Json> &params) {
  const auto asc_ir_att_impl = af::ascir::AscirRegistry::GetInstance().GetIrAttImpl(target.npu_arch, kDefaultApi);
  GE_ASSERT_NOTNULL(asc_ir_att_impl);
  const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
  GE_ASSERT_NOTNULL(api_perf_name);
  auto api_perf = ApiPerfFactory::Instance().Create(api_perf_name);
  GE_ASSERT_NOTNULL(api_perf);
  const auto param_table = api_perf->GetPerfParam();
  GE_ASSERT_NOTNULL(param_table);
  const auto ascend_api_perf_table = param_table->GetAscendCApiPerfTable();
  GE_ASSERT_NOTNULL(ascend_api_perf_table);
  return GetSocApiParams(target, *ascend_api_perf_table, params);
}

void UpdateContinuousIdx(size_t idx, const std::vector<Expr> &strides, const std::vector<Expr> &repeats,
                         std::vector<bool> &is_continuous) {
  size_t cur = idx - 1UL;
  Expr cur_axis_stride = repeats[idx] * strides[idx];
  while (cur >= 1UL && strides[cur] == 0) {
    --cur;
  }
  if (strides[cur] != 0 && cur_axis_stride != strides[cur]) {
    is_continuous[idx] = false;
  }
}

af::Status UpdateIsContinous(const std::vector<Expr> &repeats, const std::vector<Expr> &strides,
                             std::vector<bool> &is_continuous, const char *prefix_name) {
  GELOGD("tensor repeats is {%s}, %s strides is {%s}.", GetVecString(repeats).c_str(), prefix_name,
         GetVecString(strides).c_str());
  GE_WARN_ASSERT(repeats.size() == strides.size(), "repeats size %zu != strides size %zu", repeats.size(),
                 strides.size());
  for (size_t i = (repeats.size() - 1UL); i >= 1UL; --i) {
    // 判断哪些轴是非连续轴
    if (strides[i] == 0) {
      is_continuous[i] = false;
      continue;
    }
    UpdateContinuousIdx(i, strides, repeats, is_continuous);
  }
  return af::SUCCESS;
}

void UpdateTensorDim(af::AscNode *node, const std::vector<bool> &is_continuous, const std::vector<Expr> &org_dims,
                     TensorShapeInfo &tensor) {
  for (size_t i = 1UL; i <= (tensor.repeats.size() - 1UL); ++i) {
    Expr product = tensor.dims.back();
    if (is_continuous[i]) {
      // 连续轴的dim使用合并原始轴后的两个dim乘积
      product = org_dims[i] * product;
      GELOGD("Got continuous axis[%zu], dim[%s] * [%s] = [%s]", i, af::SymbolicUtils::ToString(org_dims[i]).c_str(),
             af::SymbolicUtils::ToString(tensor.dims.back()).c_str(), af::SymbolicUtils::ToString(product).c_str());
      tensor.dims.pop_back();
    } else {
      // 非连续轴的dim使用stride
      size_t cur = i - 1UL;
      while (cur >= 1UL && tensor.strides[cur] == 0) {
        --cur;
      }
      if (AttUtils::IsLoadStoreNode(node)) {
        product = org_dims[i];
      } else {
        product = tensor.strides[cur];
      }
      GELOGD("Got non-continuous axis[%zu], dim[%s]", i, af::SymbolicUtils::ToString(product).c_str());
    }
    tensor.dims.emplace_back(product);
  }
}

af::Status GetDMAActualPerf(const NodeDetail &node_info, const Expr &swap_outer_repeat, const std::vector<Expr> &dims,
                            PerfOutputInfo &perf) {
  NodeDetail cur_node;
  cur_node.name = node_info.name;
  cur_node.optype = node_info.optype;
  cur_node.input_dtype = {node_info.input_dtype[0]};
  cur_node.output_dtype = {node_info.output_dtype[0]};
  cur_node.gm_stride = node_info.gm_stride;
  cur_node.ub_stride = node_info.ub_stride;
  cur_node.block_count_idx = node_info.block_count_idx;
  cur_node.repeats = node_info.repeats;
  cur_node.vectorized_shape = node_info.vectorized_shape;
  cur_node.ternary_ops = node_info.ternary_ops;
  cur_node.nddma_descriptor = node_info.nddma_descriptor;
  GE_ASSERT_SUCCESS(SetDims(dims, cur_node));
  std::string registered_key_name;
  GE_ASSERT_SUCCESS(GetApiRegisterVerName(registered_key_name));
  const auto perf_func = GetAscendCPerfFunc(node_info.optype + registered_key_name);
  GE_ASSERT_NOTNULL(perf_func);
  GE_ASSERT_SUCCESS(perf_func(cur_node, perf), "Failed to get perf for node %s", cur_node.ToString().c_str());
  if (cur_node.optype == kMoveUbToGm) {
    perf.pipe_res[PipeType::AIV_MTE3] = swap_outer_repeat * GetPipeCost(perf, PipeType::AIV_MTE3);
  } else {
    perf.pipe_res[PipeType::AIV_MTE2] = swap_outer_repeat * GetPipeCost(perf, PipeType::AIV_MTE2);
  }
  return af::SUCCESS;
}

af::Status UpdateSwapPerf(const std::vector<Expr> &dma_dims, const NodeDetail &node_info,
                          const int32_t supported_max_dma_len, PerfOutputInfo &swap_perf, PerfOutputInfo &non_swap_perf,
                          PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(supported_max_dma_len > 0 && dma_dims.size() > static_cast<size_t>(supported_max_dma_len));
  const size_t dim_size = dma_dims.size();
  const size_t max_dma_len = static_cast<size_t>(supported_max_dma_len);
  if (dma_dims[dim_size - max_dma_len].IsConstExpr() && dma_dims[dim_size - max_dma_len - 1U].IsConstExpr()) {
    if (af::SymbolicUtils::StaticCheckLt(dma_dims[dim_size - max_dma_len], dma_dims[dim_size - max_dma_len - 1U]) ==
        af::TriBool::kTrue) {
      GE_ASSERT_SUCCESS(UpdateTenary(swap_perf, perf_res));
    } else {
      GE_ASSERT_SUCCESS(UpdateTenary(non_swap_perf, perf_res));
    }
  } else {
    Expr res;
    PipeType pipe_type = node_info.optype == kMoveUbToGm ? PipeType::AIV_MTE3 : PipeType::AIV_MTE2;
    GE_ASSERT_SUCCESS(UpdateTenary(swap_perf, perf_res));
    GE_ASSERT_SUCCESS(UpdateTenary(non_swap_perf, perf_res));
    GetPerfVar(att::SanitizeNodeName(node_info.name), res, perf_res.ternary_ops);
    perf_res.ternary_ops[res] =
        TernaryOp(CondType::K_LT, dma_dims[dim_size - max_dma_len], dma_dims[dim_size - max_dma_len - 1U],
                  GetPipeCost(swap_perf, pipe_type), GetPipeCost(non_swap_perf, pipe_type));
    perf_res.ternary_ops[res].SetVariable(res);
    perf_res.pipe_res[pipe_type] = res;
  }
  return af::SUCCESS;
}

// 获取dma参数，获取外派循环轴/使用的轴，支持根据内轴大小比较进行交换
// need_swap=true 时的语义（与 codegen SetDmaParams/CreateDmaCallInner 对齐）：
//   将倒数第(supported_max_dma_len+1)维换入，倒数第supported_max_dma_len维外抛
//   used_dims = [倒数第(kMaxDmaLen+1)维, 倒数第(kMaxDmaLen-1)维, ..., 倒数第1维]
//   例：dims=[A,B,C], kMaxDmaLen=2, swap=true
//     used_dims=[A, C]（跳过B，外抛B），outer_repeat *= B
af::Status GetDmaParams(const vector<Expr> &dims, Expr &outer_repeat, vector<Expr> &used_dims,
                        const int32_t supported_max_dma_len, bool need_swap) {
  used_dims.clear();
  GE_ASSERT_TRUE(!dims.empty());
  if (static_cast<int32_t>(dims.size()) <= supported_max_dma_len) {
    used_dims = dims;
    outer_repeat = af::sym::kSymbolOne;
    GELOGD("Got outer loop %s, used dims %s", outer_repeat.Serialize().get(), GetVecString(used_dims).c_str());
  } else {
    size_t dim_size = dims.size();
    outer_repeat = accumulate(dims.begin(), (dims.end() - supported_max_dma_len - 1), CreateExpr(1),
                              [](const Expr &a, const Expr &b) { return af::sym::Mul(a, b); });
    size_t first_dim_index;
    if (need_swap) {
      // 将倒数第(kMaxDmaLen+1)维换入 DMA，倒数第kMaxDmaLen维外抛
      // used_dims = [dims[n-kMaxDmaLen-1], dims[n-kMaxDmaLen+1], ..., dims[n-1]]
      first_dim_index = dim_size - supported_max_dma_len - 1;
      used_dims.emplace_back(dims[first_dim_index]);
      outer_repeat = outer_repeat * dims[dim_size - supported_max_dma_len];  // 外抛倒数第kMaxDmaLen维
      for (int32_t i = 2; i <= supported_max_dma_len; i++) {
        used_dims.emplace_back(dims[first_dim_index + i]);  // 跳过 dims[first+1]（已外抛）
      }
    } else {
      first_dim_index = dim_size - supported_max_dma_len;
      used_dims.emplace_back(dims[first_dim_index]);
      outer_repeat = outer_repeat * dims[first_dim_index - 1];
      for (int32_t i = 1; i < supported_max_dma_len; i++) {
        used_dims.emplace_back(dims[first_dim_index + i]);
      }
    }

    GELOGD(
        "[DMA_PARAMS] need_swap=%d, dim_size=%zu, kMaxDmaLen=%d, first_dim_index=%zu, used_dims=[%s], outer_repeat=%s",
        need_swap, dim_size, supported_max_dma_len, first_dim_index, GetVecString(used_dims).c_str(),
        outer_repeat.Str().get());
  }
  return af::SUCCESS;
}

PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  return param_table->GetPipeHeadPerfFunc(pipe_type);
}

af::Status GetApiRegisterVerName(std::string &registered_key_name) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  registered_key_name = param_table->GetApiRegisterVerName();
  return af::SUCCESS;
}

af::Status GetOpHeadCost(Expr &head_cost) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  head_cost = param_table->GetOpHeadCost();
  return af::SUCCESS;
}

static std::unique_ptr<ApiPerf> CreateSocApiPerf(const SocPerfTarget &target, const std::string &node_type) {
  const auto asc_ir_att_impl = af::ascir::AscirRegistry::GetInstance().GetIrAttImpl(target.npu_arch, node_type);
  if (asc_ir_att_impl == nullptr) {
    return nullptr;
  }
  const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
  auto api_perf = api_perf_name == nullptr ? ApiPerfFactory::Instance().Create(node_type)
                                           : ApiPerfFactory::Instance().Create(api_perf_name);
  if (api_perf == nullptr) {
    return nullptr;
  }

  Perf selected_func = nullptr;
  if (ResolveSocAscIrPerf(target, api_perf->GetApiName(), api_perf->GetPerfFunc(), selected_func) != af::SUCCESS) {
    return nullptr;
  }
  return std::make_unique<ApiPerf>(api_perf->GetApiName(), selected_func, api_perf->GetMicroPerfFunc(),
                                   api_perf->GetPerfParam(), api_perf->GetTilingScheduleConfigTable());
}

std::unique_ptr<ApiPerf> GetApiPerf(const std::string &node_type) {
  GELOGD("Begin to get api name for node[%s]", node_type.c_str());
  SocPerfTarget target;
  if (ReadSocPerfTarget(target) == af::SUCCESS) {
    bool is_soc_perf_target = false;
    if (IsSocPerfTarget(target, is_soc_perf_target) != af::SUCCESS) {
      return nullptr;
    }
    if (is_soc_perf_target) {
      return CreateSocApiPerf(target, node_type);
    }
  }

  const auto asc_ir_att_impl = ascgen_utils::GetAscIrAttImpl(node_type);
  GE_ASSERT_NOTNULL(asc_ir_att_impl, "Get ascir att impl failed, node_type[%s]", node_type.c_str());
  const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
  if (api_perf_name != nullptr) {
    GELOGD("Found api name for node[%s]: [%s]", node_type.c_str(), api_perf_name);
    return ApiPerfFactory::Instance().Create(api_perf_name);
  }
  return ApiPerfFactory::Instance().Create(node_type);
}

static af::Status CalculatePerfFromParams(const NodePerfInfo &node_perf_info, const Json &params_data, Expr &res) {
  const std::string optype = node_perf_info.optype;
  const std::string input_dtype = node_perf_info.input_dtype;
  const std::string output_dtype = node_perf_info.output_dtype;
  const auto &opt_type_param = params_data.find(optype);
  GE_ASSERT_TRUE(opt_type_param != params_data.end(), "Optype[%s] not found! json: %s", optype.c_str(),
                 params_data.dump().c_str());
  auto op_model = opt_type_param.value();
  const auto &model_type_param = op_model.find("model_type");
  std::string model_type = model_type_param.value();
  auto model_params_it = op_model.find("model_params");
  GE_ASSERT_TRUE(model_params_it != op_model.end(), "model_params not found in op_model for optype[%s]!",
                 optype.c_str());
  auto model_params = model_params_it.value();
  std::string dtype = input_dtype + "to" + output_dtype;
  if (model_params.find(dtype) == model_params.end()) {
    GELOGW("Dtype[%s] is not registered for optype[%s]!", dtype.c_str(), optype.c_str());
    dtype = GetDefaultDataType(input_dtype) + "to" + GetDefaultDataType(output_dtype);
    GELOGD("Use default dtype[%s] for optype[%s].", dtype.c_str(), optype.c_str());
  }
  const auto used_params = model_params.find(dtype);
  GE_ASSERT_TRUE(used_params != model_params.end(), "Dtype[%s] for optype[%s] not found! json: %s", dtype.c_str(),
                 optype.c_str(), params_data.dump().c_str());
  const auto &perf_func = kModelMap.find(model_type);
  GE_ASSERT_TRUE(perf_func != kModelMap.end(), "Model type[%s] for optype[%s] not found! json: %s", model_type.c_str(),
                 optype.c_str(), params_data.dump().c_str());

  // 获取 JSON 参数并使用辅助函数进行参数增强
  std::map<std::string, float> base_params = used_params.value();
  std::map<std::string, float> param_map = EnrichParamMapForModel(base_params, node_perf_info, input_dtype);

  GE_ASSERT_SUCCESS(perf_func->second(param_map, node_perf_info.dims, node_perf_info.gm_stride, res),
                    "Get Perf Func failed! json: %s", params_data.dump().c_str());
  return af::SUCCESS;
}

af::Status GetPerf(const NodePerfInfo &node_perf_info, Expr &res) {
  GELOGD("GetPerf: optype=[%s], input_dtype=[%s], output_dtype=[%s], block_count_idx=%d", node_perf_info.optype.c_str(),
         node_perf_info.input_dtype.c_str(), node_perf_info.output_dtype.c_str(), node_perf_info.block_count_idx);

  SocPerfTarget target;
  std::shared_ptr<const Json> soc_params;
  if (ReadSocPerfTarget(target) == af::SUCCESS) {
    bool is_soc_perf_target = false;
    GE_ASSERT_SUCCESS(IsSocPerfTarget(target, is_soc_perf_target));
    if (is_soc_perf_target) {
      GE_ASSERT_SUCCESS(GetTargetApiParams(target, soc_params));
      return CalculatePerfFromParams(node_perf_info, *soc_params, res);
    }
  }

  const auto perf_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(perf_table);
  const auto ascend_api_perf_table = perf_table->GetAscendCApiPerfTable();
  GE_ASSERT_NOTNULL(ascend_api_perf_table);
  Json params_data;
  GE_ASSERT_SUCCESS(StringToJson(*ascend_api_perf_table, params_data));
  return CalculatePerfFromParams(node_perf_info, params_data, res);
}

inline void FilterStridesAndRepeats(const std::vector<Expr> &strides, const std::vector<Expr> &repeats,
                                    std::vector<Expr> &filtered_strides,  // 输出：过滤后的strides
                                    std::vector<Expr> &filtered_repeats   // 输出：过滤后的repeats
) {
  size_t loop_max = 0;
  if (!strides.empty() && !repeats.empty()) {
    loop_max = std::min(strides.size() - 1, repeats.size() - 1);
  }
  // 过滤出非尾轴的非广播维度
  for (size_t i = 0UL; i < loop_max; ++i) {
    if (!(af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue &&
          af::SymbolicUtils::StaticCheckEq(repeats[i], af::sym::kSymbolOne) == af::TriBool::kTrue)) {
      filtered_strides.emplace_back(strides[i]);
      filtered_repeats.emplace_back(repeats[i]);
    }
  }
  // 保存尾轴
  if (!repeats.empty() && repeats.size() == strides.size()) {
    filtered_strides.emplace_back(strides.back());
    filtered_repeats.emplace_back(repeats.back());
  }
  GELOGD("Repeats, strides filter broadcast axis: filtered_strides=[%s], filtered_repeats=[%s]",
         GetVecString(filtered_strides).c_str(), GetVecString(filtered_repeats).c_str());
}

StrideResult CalculateMultiDimStride(const std::vector<Expr> &filtered_strides,
                                     const std::vector<Expr> &filtered_repeats, const NodeDetail &node_info,
                                     const char *stride_type, const int32_t supported_max_dma_len, bool need_swap,
                                     bool is_ub_stride) {
  const auto filtered_dim_size = static_cast<int32_t>(filtered_repeats.size());
  const bool actually_swap = need_swap && (filtered_dim_size > supported_max_dma_len);
  const int32_t block_count_idx = CalculateBlockCountIdx(filtered_dim_size, actually_swap);
  // 一般而言不应该出现下面异常分支
  GE_WARN_ASSERT((block_count_idx < static_cast<int32_t>(filtered_strides.size())) && (block_count_idx >= 0),
                 "%s, block_count_idx is %d over size(%zu).", node_info.ToString().c_str(), block_count_idx,
                 filtered_strides.size());
  const auto &last_stride = filtered_strides.back();
  constexpr int64_t kContinueLastStrideVal = 1L;
  int64_t last_stride_val = kContinueLastStrideVal;
  // 尾轴stride > 1时存在严重非连续，特殊处理
  if (last_stride.IsConstExpr()) {
    // 静态shape：原有逻辑
    if (last_stride.GetConstValue(last_stride_val) && (last_stride_val > kContinueLastStrideVal)) {
      GELOGD("%s, block_count_idx=%d, last_stride[%s=%ld], need_swap=%d, actually_swap=%d, filtered_dim_size=%d",
             node_info.ToString().c_str(), block_count_idx, stride_type, last_stride_val, need_swap, actually_swap,
             filtered_dim_size);
      // codegen 为补出的 repeat=1 尾轴生成 DataCopy 的 src_stride=(GM stride - 1)。
      // ATT 仍按原始 repeats 计算 stride 惩罚次数，但 stride 值要与该 API 入参一致。
      if (node_info.optype == kLoad && !node_info.vectorized_shape.empty()) {
        return StrideResult(last_stride - CreateExpr(1), block_count_idx);
      }
      return StrideResult(last_stride, block_count_idx);
    }
  } else {
    // 动态shape：使用TernaryOp
    auto normal_stride =
        af::sym::Abs(af::sym::Sub(filtered_strides[block_count_idx], filtered_repeats[filtered_dim_size - 1]));
    const auto non_continuous_stride =
        (node_info.optype == kLoad && !node_info.vectorized_shape.empty()) ? last_stride - CreateExpr(1) : last_stride;
    auto result =
        CreateDynamicStrideResult(last_stride, non_continuous_stride, normal_stride, block_count_idx, is_ub_stride);
    GELOGD("%s, dynamic stride select, last_stride=%s", node_info.ToString().c_str(), last_stride.Str().get());
    return result;
  }
  auto expr = af::sym::Abs(af::sym::Sub(filtered_strides[block_count_idx], filtered_repeats[filtered_dim_size - 1]));
  GELOGD("[STRIDE_CALC] %s: %s, block_count_idx=%d, stride=%s, need_swap=%d, actually_swap=%d, filtered_dim_size=%d",
         node_info.ToString().c_str(), stride_type, block_count_idx, Str(expr).c_str(), need_swap, actually_swap,
         filtered_dim_size);
  return StrideResult(expr, block_count_idx);
}

StrideResult CalculateStride(const TensorShapeInfo &shape_info, const bool is_ub_stride, const NodeDetail &node_info,
                             const int32_t supported_max_dma_len, bool need_swap) {
  const std::vector<Expr> &strides = is_ub_stride ? shape_info.strides : shape_info.gm_strides;
  const std::vector<Expr> &repeats = is_ub_stride ? shape_info.origin_repeats : shape_info.repeats;
  const char *stride_type = is_ub_stride ? "ub_stride" : "gm_stride";
  GELOGD("%s(%s): %s=[%s], repeats=[%s], need_swap=%d", node_info.name.c_str(), node_info.optype.c_str(), stride_type,
         GetVecString(strides).c_str(), GetVecString(repeats).c_str(), need_swap);
  std::vector<Expr> filtered_strides;
  std::vector<Expr> filtered_repeats;
  FilterStridesAndRepeats(strides, repeats, filtered_strides, filtered_repeats);
  auto filtered_dim_size = static_cast<int32_t>(filtered_repeats.size());
  if ((filtered_dim_size <= supported_max_dma_len) && need_swap) {
    GELOGW("%s, can't swap %s. filtered_dim_size=%d, supported_max_dma_len=%d", node_info.ToString().c_str(),
           stride_type, filtered_dim_size, supported_max_dma_len);
    return StrideResult(CreateExpr(0), 0);
  }
  if (filtered_dim_size == 1 || filtered_strides.empty()) {
    bool is_stride_zero = filtered_strides.empty() || (filtered_strides.back() == 0);
    if (!is_stride_zero && node_info.optype == kLoad && !node_info.vectorized_shape.empty()) {
      return StrideResult(filtered_strides.back() - CreateExpr(1), 0);
    }
    auto stride = is_stride_zero ? CreateExpr(0) : (filtered_strides.back() - CreateExpr(1)) * filtered_repeats.back();
    GELOGD("%s, total len is %d, is_stride_zero=%d, stride=%s", node_info.ToString().c_str(), filtered_dim_size,
           is_stride_zero, Str(stride).c_str());
    return StrideResult(stride, 0);
  }
  return CalculateMultiDimStride(filtered_strides, filtered_repeats, node_info, stride_type, supported_max_dma_len,
                                 need_swap, is_ub_stride);
}

af::Status SetStride(const TensorShapeInfo &shape_info, NodeDetail &node_info, const int32_t supported_max_dma_len,
                     bool need_swap) {
  auto gm_stride_result = CalculateStride(shape_info, false, node_info, supported_max_dma_len, need_swap);
  auto ub_stride_result = CalculateStride(shape_info, true, node_info, supported_max_dma_len, need_swap);
  node_info.gm_stride = (gm_stride_result.stride.Serialize() == nullptr) ? CreateExpr(0) : gm_stride_result.stride;
  node_info.ub_stride = (ub_stride_result.stride.Serialize() == nullptr) ? CreateExpr(0) : ub_stride_result.stride;
  // 保存 gm_stride 的 block_count_idx，用于性能函数计算
  node_info.block_count_idx = gm_stride_result.block_count_idx;
  // 合并ternary_ops到node_info
  for (const auto &ternary_op : gm_stride_result.ternary_ops) {
    node_info.ternary_ops[ternary_op.first] = ternary_op.second;
  }
  for (const auto &ternary_op : ub_stride_result.ternary_ops) {
    node_info.ternary_ops[ternary_op.first] = ternary_op.second;
  }
  GELOGD("%s: repeats=%s origin_repeats=%s need_swap=%d block_count_idx=%d", node_info.ToString().c_str(),
         GetVecString(shape_info.repeats).c_str(), GetVecString(shape_info.origin_repeats).c_str(), need_swap,
         node_info.block_count_idx);
  return af::SUCCESS;
}

af::Status SetNodeDetail(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, NodeDetail &node_info) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  for (const auto &input_shape : input_shapes) {
    node_info.input_dtype.emplace_back(input_shape.data_type);
  }
  for (const auto &output_shape : output_shapes) {
    node_info.output_dtype.emplace_back(output_shape.data_type);
  }
  GE_ASSERT_SUCCESS(SetDims(input_shapes[0].dims, output_shapes[0].dims, node_info));
  return af::SUCCESS;
}

af::Status SetDims(const TensorShapeInfo &tensor_shape_info, NodeDetail &node_info) {
  GE_ASSERT_SUCCESS(SetDims(tensor_shape_info.dims, node_info));
  node_info.repeats = tensor_shape_info.repeats;
  SetLoadVectorizedShape(tensor_shape_info, node_info);
  return af::SUCCESS;
}

af::Status SetDims(const std::vector<Expr> &dims, NodeDetail &node_info) {
  node_info.input_dims.clear();
  node_info.output_dims.clear();
  for (const auto &dim : dims) {
    node_info.input_dims.emplace_back(dim);
    node_info.output_dims.emplace_back(dim);
  }
  return af::SUCCESS;
}

af::Status SetDims(const std::vector<Expr> &input_dims, const std::vector<Expr> &output_dims, NodeDetail &node_info) {
  node_info.input_dims.clear();
  node_info.output_dims.clear();
  for (const auto &dim : input_dims) {
    node_info.input_dims.emplace_back(dim);
  }
  for (const auto &dim : output_dims) {
    node_info.output_dims.emplace_back(dim);
  }
  return af::SUCCESS;
}

NodeDetail GenNodeDetail(const std::string &input_dtype, const std::string &output_dtype,
                         const std::vector<Expr> &dims) {
  NodeDetail ret;
  ret.input_dtype.emplace_back(input_dtype);
  ret.output_dtype.emplace_back(output_dtype);
  if (dims.empty()) {
    ret.input_dims.emplace_back(af::sym::kSymbolOne);
    ret.output_dims.emplace_back(af::sym::kSymbolOne);
  } else {
    for (const auto &dim : dims) {
      ret.input_dims.emplace_back(dim);
      ret.output_dims.emplace_back(dim);
    }
  }
  return ret;
}

Expr GetPipeCost(const PerfOutputInfo &perf_res, const PipeType &pipe_type) {
  auto iter = perf_res.pipe_res.find(pipe_type);
  if (iter != perf_res.pipe_res.end()) {
    return iter->second;
  } else {
    return af::sym::kSymbolZero;
  }
}

bool HasSmallBlockLenWithUbStride(const NodeDetail &node_info) {
  constexpr int32_t kOneBlkSize = 32;
  uint32_t ub_stride_val = 0;
  uint32_t block_len_val = kOneBlkSize;
  auto data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(data_type_size != kDataTypeSizeMap.end());
  Expr blocklen = af::sym::Mul(node_info.input_dims[node_info.input_dims.size() - 1UL], data_type_size->second);
  if (blocklen.GetConstValue(block_len_val) && node_info.ub_stride.GetConstValue(ub_stride_val)) {
    if (block_len_val < static_cast<uint32_t>(kOneBlkSize) && ub_stride_val > 0U) {
      GELOGD("%s: ub stride(%u) and block len(%u) less than %d", node_info.ToString().c_str(), ub_stride_val,
             block_len_val, kOneBlkSize);
      return true;
    }
  }
  return false;
}

/**
 * 根据转置信息重排 gm_strides
 * 当 Load/Store 操作发生转置时，将 gm_strides 的轴顺序转置回 vectorized 轴的顺序
 *
 * @param node 节点指针，用于访问 input tensor 属性
 * @param tensor 包含 gm_strides 的张量信息
 * @return af::SUCCESS 成功，其他值表示失败
 */
af::Status ReorderGmStrideByTranspose(const af::AscNodePtr &node, TensorShapeInfo &tensor) {
  GE_ASSERT_NOTNULL(node);
  // 检查是否为空
  if (ShouldSkipReorderForEmptyOutputs(node)) {
    return af::SUCCESS;
  }
  // 访问 tensor 属性
  const auto &attr = node->outputs[0].attr;
  const std::vector<int64_t> &vectorized_axis = attr.vectorized_axis;
  const std::vector<int64_t> &axis = attr.axis;
  std::vector<Expr> repeats = attr.repeats;
  // 检查相关数据是否为空
  if (ShouldSkipReorderForEmptyData(node, tensor, vectorized_axis, axis)) {
    return af::SUCCESS;
  }
  // 检查是否需要重排
  if (!NeedsReordering(vectorized_axis, axis)) {
    GELOGD("No transpose detected, skipping repeats reordering, repeats[%s]", GetVecString(tensor.repeats).c_str());
    return af::SUCCESS;
  }
  // 找出需要删除的轴下标
  // 例如：vectorized_axis={0,1,3}, axis={4,3,2,1,0}
  // axis 下标 0 对应轴 4，不在 vectorized_axis 中
  // axis 下标 2 对应轴 2，不在 vectorized_axis 中
  // 需要从 repeats 中删除下标 0 和 2 的元素
  std::vector<size_t> indices_to_remove = FindIndicesToRemove(vectorized_axis, axis);
  // 从后往前删除 repeats 中对应下标的元素（避免索引变化问题）
  if (!indices_to_remove.empty()) {
    for (size_t idx : indices_to_remove) {
      if (idx < repeats.size()) {
        GELOGD("Node[%s,%s] axis_id[%d] at axis position[%zu] not found in vectorized_axis, will remove repeats[%zu]",
               node->GetNamePtr(), node->GetType().c_str(), axis[idx], idx, idx);
        GELOGD("Node[%s,%s] removing repeats[%zu]=%s", node->GetNamePtr(), node->GetType().c_str(), idx,
               Str(repeats[idx]).c_str());
      }
    }
    RemoveIndicesFromRepeats(repeats, indices_to_remove);
    GELOGD("Node[%s,%s] after removing %zu elements, strides[%s]", node->GetNamePtr(), node->GetType().c_str(),
           indices_to_remove.size(), GetVecString(repeats).c_str());
  }
  GELOGD("Reordered node[%s,%s] repeats: original[%s] -> reordered[%s], vectorized_axis[%s], axis[%s]",
         node->GetNamePtr(), node->GetType().c_str(), GetVecString(tensor.repeats).c_str(),
         GetVecString(repeats).c_str(), DebugString(vectorized_axis).c_str(), DebugString(axis).c_str());
  tensor.repeats.swap(repeats);
  return af::SUCCESS;
}

/** 举例:
  repeats为{z0Tb_size, z0t_size, 2, 10,};
  strides为{(20 * z0t_size, 20, 10, 1,};
  vectorized_axis为{z0t, z2, z3};
  考虑非连续场景:
  vectorized_stride为{32, 16, 1}  非连续场景
  连续标记为{true, true, false} {跳过,    32等于16*2,        16不等于10*2}
                                        z0t_size与z1合轴  z1与z2不合轴
  考虑连续场景:
  vectorized_stride为{20, 10, 1}  连续场景
  连续标记为{true, true, true}  {跳过     20等于10*2,        10不等于1*2}
                                        z0t_size与z1合轴  z1与z2合轴
  reduce场景:reduce轴会断掉连续轴的合并
 **/
af::Status MergeTensorContinuousDims(const af::AscNodePtr &node, const std::string &tensor_name,
                                     TensorShapeInfo &tensor) {
  Expr axis_size;
  std::vector<Expr> org_dims;  // 原始的dim
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(!tensor.repeats.empty());
  GE_ASSERT_TRUE(tensor.repeats.size() == tensor.strides.size(), "repeats size[%zu] not equal strides size[%zu]",
                 tensor.repeats.size(), tensor.strides.size());
  // 使用repeat和stride重新初始化dims
  tensor.dims.clear();
  size_t vectorized_size = tensor.repeats.size();
  std::vector<bool> is_continuous(vectorized_size, true);
  for (size_t i = 0UL; i < vectorized_size; ++i) {
    org_dims.emplace_back(tensor.repeats[i]);
  }
  if (org_dims.empty()) {
    org_dims.emplace_back(af::sym::kSymbolOne);
  } else {
    (void)UpdateIsContinous(tensor.repeats, tensor.strides, is_continuous, "UB");
    if (AttUtils::IsLoadNode(node.get())) {
      // 根据转置信息重排 gm_strides
      GE_ASSERT_SUCCESS(ReorderGmStrideByTranspose(node, tensor), "Failed to reorder gm_strides by transpose");
    }
    if (AttUtils::IsLoadStoreNode(node.get())) {
      (void)UpdateIsContinous(tensor.repeats, tensor.gm_strides, is_continuous, "GM");
    }
    tensor.dims.emplace_back(org_dims[0]);
    UpdateTensorDim(node.get(), is_continuous, org_dims, tensor);
  }
  GELOGD("Tensor[%s] shape info: merge dims:%s org dims:%s continuous:%s", tensor_name.c_str(),
         tensor.GetDimExpr().c_str(), GetVecString(org_dims).c_str(), af::ToString(is_continuous).c_str());
  return af::SUCCESS;
}

af::Status GetOuterParams(const vector<Expr> &dims, Expr &outer_repeat, vector<Expr> &used_dims,
                          const uint32_t dma_max_len) {
  // 若dma_max_len为2:
  // (2, 2, 2)
  //     used_dims = {2, 2}
  // repeat=2
  GE_ASSERT_TRUE(!dims.empty());
  if (dims.size() == 1) {
    used_dims = {dims[0]};
    outer_repeat = af::sym::kSymbolOne;
  } else {
    size_t dim_size = dims.size();
    for (int64_t i = 0L; i < static_cast<int64_t>(dma_max_len); ++i) {
      used_dims.emplace_back(dims[dim_size - static_cast<int64_t>(dma_max_len) + i]);
    }
    outer_repeat = accumulate(dims.begin(), (dims.end() - dma_max_len), CreateExpr(1),
                              [](Expr a, Expr b) { return af::sym::Mul(a, b); });
    GELOGD("Got outer loop %s, used dims %s", outer_repeat.Serialize().get(), GetVecString(used_dims).c_str());
  }
  return af::SUCCESS;
}

af::Status UpdateTenary(PerfOutputInfo &perf_res, PerfOutputInfo &output_res) {
  Expr cur_var;
  std::vector<std::pair<Expr, Expr>> replace_vars;
  for (const auto &pair : perf_res.ternary_ops) {
    std::string cur_name = Str(pair.first);
    GetPerfVar(cur_name, cur_var, output_res.ternary_ops);
    replace_vars.emplace_back(std::make_pair(pair.first, cur_var));
    output_res.ternary_ops[cur_var] = pair.second;
    output_res.ternary_ops[cur_var].SetVariable(cur_var);
  }
  for (const auto &pair : perf_res.pipe_res) {
    output_res.pipe_res[pair.first] = pair.second.Replace(replace_vars);
  }
  perf_res.pipe_res = output_res.pipe_res;
  return af::SUCCESS;
}

af::Status GetDmaPerf(const TensorShapeInfo &tensor_info, NodeDetail &node_info, PerfOutputInfo &perf_res,
                      int32_t supported_max_dma_len, bool need_swap) {
  PerfOutputInfo non_swap_perf;
  vector<Expr> used_dims;
  vector<Expr> swap_used_dims;  // 交换后的 used_dims，避免覆盖
  PerfOutputInfo swap_perf;
  Expr non_swap_outer_repeat;
  Expr swap_outer_repeat;
  non_swap_perf.cache_line_config = perf_res.cache_line_config;
  swap_perf.cache_line_config = perf_res.cache_line_config;
  const size_t dim_size = node_info.input_dims.size();
  GELOGD("[DMA_PERF] %s: BEGIN GetDmaPerf, dim_size=%zu, input_dims=[%s], supported_max_dma_len=%d, need_swap=%d",
         node_info.name.c_str(), dim_size, GetVecString(node_info.input_dims).c_str(), supported_max_dma_len,
         need_swap);
  const auto &dma_dims = node_info.vectorized_shape.empty() ? node_info.input_dims : node_info.vectorized_shape;
  const size_t dma_dim_size = dma_dims.size();
  if (static_cast<int32_t>(dma_dim_size) <= supported_max_dma_len) {
    GE_ASSERT_SUCCESS(SetStride(tensor_info, node_info, supported_max_dma_len));
    GE_ASSERT_SUCCESS(GetDMAActualPerf(node_info, af::sym::kSymbolOne, dma_dims, non_swap_perf));
    GE_ASSERT_SUCCESS(UpdateTenary(non_swap_perf, perf_res));
  } else {
    GE_ASSERT_SUCCESS(GetDmaParams(dma_dims, non_swap_outer_repeat, used_dims, supported_max_dma_len));
    GE_ASSERT_SUCCESS(SetStride(tensor_info, node_info, supported_max_dma_len));
    GE_ASSERT_SUCCESS(GetDMAActualPerf(node_info, non_swap_outer_repeat, used_dims, non_swap_perf));
    GELOGD("Perf without swap is [%s]", non_swap_perf.ToString().c_str());
    if (need_swap) {
      GELOGD("[DMA_PERF] %s: swap path start, swap_used_dims will be calculated", node_info.name.c_str());
      GE_ASSERT_SUCCESS(GetDmaParams(dma_dims, swap_outer_repeat, swap_used_dims, supported_max_dma_len, true));
      GE_ASSERT_SUCCESS(SetStride(tensor_info, node_info, supported_max_dma_len, true));
      GE_ASSERT_SUCCESS(GetDMAActualPerf(node_info, swap_outer_repeat, swap_used_dims, swap_perf));
      GELOGD("[DMA_PERF] %s: Perf with swap is [%s], swap_used_dims=[%s], swap_outer_repeat=%s", node_info.name.c_str(),
             swap_perf.ToString().c_str(), GetVecString(swap_used_dims).c_str(), swap_outer_repeat.Str().get());
    } else {
      GE_ASSERT_SUCCESS(UpdateTenary(non_swap_perf, perf_res), "Update non_swap perf failed, node=%s",
                        node_info.ToString().c_str());
    }
    GELOGD("The input dim is [%s]", GetVecString(node_info.input_dims).c_str());
    if (need_swap) {
      GE_ASSERT_SUCCESS(UpdateTenary(swap_perf, perf_res), "Update swap perf failed, node=%s",
                        node_info.ToString().c_str());
      GE_ASSERT_SUCCESS(UpdateSwapPerf(dma_dims, node_info, supported_max_dma_len, swap_perf, non_swap_perf, perf_res),
                        "Update swap perf failed, node=%s", node_info.ToString().c_str());
    }
  }
  // 合并node_info中的ternary_ops到perf_res
  for (const auto &ternary_op : node_info.ternary_ops) {
    perf_res.ternary_ops[ternary_op.first] = ternary_op.second;
  }
  return af::SUCCESS;
}

}  // namespace att
