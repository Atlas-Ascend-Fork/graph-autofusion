/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
#define __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "codegen_kernel.h"
#include "indirect_load_utils.h"

namespace codegen {
struct SimtOutputChain {
  std::vector<af::AscNodePtr> nodes;
  ascir::TensorId result_tensor_id{af::kIdNone};
  ascir::TensorId target_tensor_id{af::kIdNone};
  af::DataType dtype{af::DT_UNDEFINED};
  bool local_target{false};
};

struct LogicalTensorInfo {
  LogicalTensorInfo() = default;
  explicit LogicalTensorInfo(const ascgen_utils::indirect_load::LogicalTensorView &view)
      : sizes(view.sizes), strides(view.strides) {}
  LogicalTensorInfo(const std::vector<ascir::SizeExpr> &sizes_in, const std::vector<ascir::SizeExpr> &strides_in)
      : sizes(sizes_in), strides(strides_in) {}

  std::vector<ascir::SizeExpr> sizes;
  std::vector<ascir::SizeExpr> strides;
};

using SimtLoadMetadataMap = std::unordered_map<std::string, ascgen_utils::indirect_load::SimtLoadMetadata>;
using SimtLoadIndexOffsetExpressions = std::unordered_map<std::string, std::string>;
using SimtGraphNodeMap = std::unordered_map<std::string, af::AscNodePtr>;

class IndirectLoadRegApiCall final : public ApiCall {
 public:
  explicit IndirectLoadRegApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~IndirectLoadRegApiCall() final = default;
  Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const override;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  std::string &result) const override;

 protected:
  Status ParseAttr(const ascir::NodeView &node) override;

 private:
  Status ResolveSimtMetadata(const ascir::NodeView &node,
                             const ascgen_utils::indirect_load::SimtLoweringMetadata &metadata,
                             const SimtGraphNodeMap &node_map);
  // [SIMT index 闭包预计算] 判定：链计划可行 + tmp scratch 已申请 + tile 整行对齐。
  // 函数级 Index()/Context 与调用点 pre-pass 发射共用同一位点，避免形态漂移。
  bool CanUseSimtIndexPrepass(const TPipe &tpipe, ascgen_utils::indirect_load::SimtIndexPrepassPlan &plan) const;
  // 后置 Reduce 的 local target 分配（预计算开启时提升到 Context 初始化之前）。
  Status GenerateSimtLocalTargetAlloc(const TPipe &tpipe, std::stringstream &ss) const;
  // [SIMT 动态 shape 坐标分解] 空跑三个求值器收集静态方法体引用的运行期符号尺寸
  // （t->ksN 不能内联进静态方法，须走 Context）。返回按序去重的 tiler 表达式列表
  // （Context 字段 rt_size_N 的初始化表达式）；prepass 行宽表达式经 idx_inner_expr
  // 带出（首注册项）。Context 字段声明（GenerateFuncDefinition）与初始化
  // （GenerateSimt）两处按同参调用，保证字段序一致。
  // [SIMT 32 位双实例] 偏移域运行时判定可 32 位的形态才双发 body：kStrided
  // Rank3/Axis1（gather 轴内单维、共享坐标差分路径、无 post Reduce/prepass）。
  bool SimtSupportsDualOffsetBody() const;
  std::vector<std::string> CollectSimtRuntimeSizeExprs(
      const std::string &input_dtype, const std::string &index_dtype, const std::string &offset_type,
      const TPipe &tpipe, bool use_prepass, const ascgen_utils::indirect_load::SimtIndexPrepassPlan &prepass_plan,
      std::string &idx_inner_expr) const;
  Status GenerateSk(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                    const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimd(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                      const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimtInvocation(const TPipe &tpipe, const std::string &input_dtype, const std::string &outer_tb_var,
                                const ascgen_utils::indirect_load::SimtIndexPrepassPlan *prepass_plan,
                                std::stringstream &ss) const;
  Status GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &input,
                      std::string &result) const;

  int64_t axis_ = 0;
  ascir::AxisId outer_axis_ = af::kIdNone;
  ascir::TemplateId template_id_{ascir::TemplateId::kDefault};
  ascgen_utils::indirect_load::TemplateLogicalView logical_view_;
  ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info_;
  mutable ascgen_utils::indirect_load::SimdLoweringMetadata simd_metadata_;
  bool has_post_reduce_{false};
  std::vector<af::AscNodePtr> index_nodes_;
  std::vector<af::AscNodePtr> output_nodes_;
  std::vector<SimtOutputChain> output_chains_;
  ascgen_utils::indirect_load::SimtPolicyMetadata simt_policy_;
  SimtLoadMetadataMap simt_index_loads_;
  SimtLoadMetadataMap simt_output_loads_;
  std::vector<ascgen_utils::indirect_load::SimtGmTensorMetadata> simt_gm_tensors_;
  ascir::TensorId index_result_tensor_id_{af::kIdNone};
  ascir::TensorId simt_value_tensor_id_{af::kIdNone};
  ascir::TensorId output_result_tensor_id_{af::kIdNone};
  af::DataType index_dtype_{af::DT_UNDEFINED};
  ascgen_utils::indirect_load::Implementation implementation_{ascgen_utils::indirect_load::Implementation::kDefault};
};
}  // namespace codegen

#endif  // __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
