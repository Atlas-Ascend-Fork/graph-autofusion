/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &x = context->GetInputShape(0)->GetOriginShape();
  const auto &y = context->GetInputShape(1)->GetOriginShape();
  if (x.GetDimNum() != 1 || y.GetDimNum() != 1 || x.GetDim(0) != 1024 || y.GetDim(0) != 1024) {
    return ge::GRAPH_FAILED;
  }
  auto *tiling = context->GetTilingData<AddCustomTilingData>();
  size_t *workspace = context->GetWorkspaceSizes(1);
  if (tiling == nullptr || workspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tiling->totalLength = 1024;
  tiling->tileNum = 8;
  if (context->SetSimdNumBlocks(8) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  workspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
  if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
    return GRAPH_FAILED;
  }
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }
  return context->SetOutputDataType(0, context->GetInputDataType(0));
}
}  // namespace ge

namespace ops {
class AddCustom : public OpDef {
 public:
  explicit AddCustom(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("z")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore()
        .SetTiling(optiling::TilingFunc)
        .AddConfig("ascend910b")
        .AddConfig("ascend910_93")
        .AddConfig("ascend950");
  }
};
OP_ADD(AddCustom);
}  // namespace ops
