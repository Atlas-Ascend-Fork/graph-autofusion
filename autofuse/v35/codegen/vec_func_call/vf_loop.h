/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_VF_LOOP_H__
#define __AUTOFUSE_VF_LOOP_H__
#include <map>
#include <set>

#include "codegen_kernel.h"
#include "../micro_api_call/micro_api_call.h"

namespace codegen {

struct ArangeParam {
  ascir::TensorId tensor_id;
  std::string base;
  std::string step;
  std::string offset;
  std::vector<std::string> logical_strides;
};

using ArangeOffsetMap = std::map<ascir::TensorId, std::string>;
using VFInputMapping = std::map<int64_t, uint32_t>;
Status GetArangeLogicalStride(const TPipe &tpipe, const MicroApiTensor *reg_tensor,
                              const std::vector<ascir::AxisId> &axis_ids, std::string &result);
std::string GenCvUbFuseVfFuncDimParams();
std::string GenCvUbFuseVfCallDimParams();
std::string GenCvUbFuseRowStride(const TPipe &tpipe, const Tensor &ub_tensor);
std::string GenCvUbFuseAddrOffset(const TPipe &tpipe, const Tensor &ub_tensor);

class VFLoop;
struct VFLoopBody {
  LoopType type_;
  union {
    MicroApiCall *call_;
    VFLoop *loop_;
  };
};

class VFLoop {
 public:
  explicit VFLoop(const ascir::AxisId axis);
  void AddLoop(VFLoop *loop);
  void AddCall(MicroApiCall *call);

  /* 图解析阶段调用 */
  Status ConstructFromNodes(ascir::NodeViewVisitorConst nodes, const ascir::NodeView &vf_node,
                            const VFInputMapping &input_mapping = {});
  void Destruct();

  /* kernel生成阶段调用 */
  Status Generate(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth, std::string &result,
                  std::string &loop_size_result, int32_t &only_loop_max_depth, std::vector<std::string> &loop_size_vec,
                  const ArangeOffsetMap &arange_offsets = {}) const;
  Status GenerateCvUbFuse(const TPipe &tpipe, const TensorManager &tensor_mgr, std::string &result,
                          std::string &loop_size_result) const;
  void SetMaxDtypeSize(std::string dtype);
  void SetLoopDtype(std::string dtype) const;
  void CollectMaskRegTempTensors(const TPipe &tpipe, const TensorManager &tensor_mgr,
                                 std::vector<std::string> &temp_tensors) const;
  void CollectArangeParams(const TPipe &tpipe, std::vector<ArangeParam> &params) const;

 private:
  ascir::AxisId axis_id_;
  VFLoop *parent_;
  std::vector<VFLoopBody> bodys_;
  std::string max_dtype_size_;
  mutable std::string loop_dtype_;

  Status GenerateLoop(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth,
                      std::vector<ascir::AxisId> &current_axis, std::stringstream &ss, std::stringstream &loop_size_ss,
                      int32_t &only_loop_max_depth, std::vector<std::string> &loop_size_vec,
                      std::set<std::string> &emitted_declarations, const ArangeOffsetMap &arange_offsets) const;
  Status GenerateBody(const TPipe &tpipe, const TensorManager &tensor_mgr, int32_t depth,
                      std::vector<ascir::AxisId> &current_axis, std::stringstream &ss, std::stringstream &loop_size_ss,
                      int32_t &only_loop_max_depth, std::vector<std::string> &loop_size_vec,
                      std::set<std::string> &emitted_declarations, const ArangeOffsetMap &arange_offsets) const;
  Status GenerateCvUbFuseBody(const TPipe &tpipe, const TensorManager &tensor_mgr,
                              std::vector<ascir::AxisId> &current_axis, std::stringstream &ss) const;
};

}  // namespace codegen
#endif  // __AUTOFUSE_VF_LOOP_H__
