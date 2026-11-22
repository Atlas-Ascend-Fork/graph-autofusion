/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_load_api_call.h"

#include <sstream>

#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "reg_api_call_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen_api_param/codegen_api_param.h"

using namespace af::ops;
using namespace af::ascir_op;

namespace codegen {
Status LoadRegApiCall::ParseAttr(const ascir::NodeView &node) {
  (void)node->attr.ir_attr->GetAttrValue("offset", offset_);
  return af::SUCCESS;
}

Status LoadRegApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                     const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                     const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  const auto &gm = inputs[0].get();
  const auto &ub = outputs[0].get();
  auto api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);
  api_param->api_name = api_name_;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(gm.dtype, dtype_name), "data type:%d failed", static_cast<int32_t>(gm.dtype));
  api_param->template_params.emplace_back(dtype_name);
  DmaSpecificParams dma_specific_params;
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && !ub.is_ub_scalar) {
    BuildDataCopyApiParamInCVFusion(tpipe, *api_param, dma_specific_params, gm, ub, dtype_name, true);
  } else {
    std::string gm_offset = ub.is_ub_scalar ? "0" : tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides);
    gm_offset = gm_offset + " + " + tpipe.tiler.Size(offset_);
    auto has_transpose = IsGraphHasTransposeNode(this->node);
    BuildDataCopyApiParamInNormal(tpipe, *api_param, dma_specific_params, gm, ub, gm_offset, true, has_transpose);
  }
  api_param->specific_params = dma_specific_params;
  if (ascgen_utils::GetTemplateSemanticTransform(*this->node) == ascgen_utils::template_semantic::kTranspose) {
    api_param->semantic_feature = ascgen_utils::template_semantic::kLoad;
  }

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param));
  return af::SUCCESS;
}

Status LoadRegApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler,
                                         std::stringstream &ss) const {
  return GenDataCopyDimParam(api_param, tiler, graph_name, node_name, ss);
}
static ApiCallRegister<LoadRegApiCall> register_load_reg_api_call("LoadRegApiCall");
}  // namespace codegen
