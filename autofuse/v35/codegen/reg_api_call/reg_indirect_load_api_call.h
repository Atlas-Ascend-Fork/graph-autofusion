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
  std::vector<af::Expression> target_sizes;
  std::vector<af::Expression> target_strides;
  af::Expression target_offset{af::ops::Zero};
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

// [fused 成员 value 输入张量解析] 成员 value 输入链上的 direct-GM 边界节点
// （kSimtInputBoundary/kSimtInlineTransform，skips_api_emit）不注册 tpipe、不生成
// api call——按 L4 接线约定沿【单输入】生产者链穿透回溯到 GM 源（Data 输出张量）。
// 多输入节点（Select/Mul 等）出现在 value 链上时返回 nullptr（该形态无单一 GM 值表，
// fused 单指针模型不支持，保持显式失败）。实现见 cpp 注释。
const Tensor *ResolveFusedMemberInputTensor(const TPipe &tpipe, const af::AscNodePtr &member);

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
  // [双发判定唯一入口·结果缓存] 定义侧（GenerateFuncDefinition 决定发几套 body）
  // 与调用侧（GenerateSimtInvocation 决定调用点形状）必须同答案，否则生成代码
  // "定义一套、调用另一套"直接编译失败。prepass_active 依赖 CanUseSimtIndexPrepass
  // （图内容 + tmp_buf_id 时机），两处独立求值存在漂移窗口（图副本改写、tmp 填充
  // 时机差异）——首次求值缓存（-1 未算 / 0 不双发 / 1 双发），两侧物理一致。
  // 漂移最坏退化为 body 多发一套（白定义，无编译失败），自洽优先。缓存按
  // ApiCall 对象生命周期有效（工厂每任务新建实例，无跨任务残留；重解析随
  // ResolveSimtMetadata 重置）。public 供 UT 直接钉死首答生效语义（P3-c）。
  bool DecideDualOffsetBody(bool prepass_active) const;

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
  // Rank3/Axis1（gather 轴内单维、共享坐标差分路径、无 post Reduce/prepass）；
  // kEmbedding Rank2/Axis0（真偏移上界 guard 逐视图逐链判定，含 post Reduce
  // 且无 prepass 的形态——prepass_active 复用 CanUseSimtIndexPrepass 结果，
  // prepass 场景首版保守排除，第二阶段单独评估）。
  bool SimtSupportsDualOffsetBody(bool prepass_active) const;
  // SIMT evaluator definitions are emitted into one source file.  Post-reduce and
  // GM-output variants of the same graph node have different Context layouts and
  // must not share generated type names or preprocessor guards.
  std::string SimtDefinitionSuffix() const;
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
  mutable int8_t dual_offset_body_{-1};  // DecideDualOffsetBody 缓存：-1 未算 / 0 不双发 / 1 双发
};
}  // namespace codegen

#endif  // __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
