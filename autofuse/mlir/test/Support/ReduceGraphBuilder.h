/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_SUPPORT_REDUCE_GRAPH_BUILDER_H
#define AUTOFUSE_MLIR_TEST_SUPPORT_REDUCE_GRAPH_BUILDER_H

#include "../../../tests/framework/easy_asc_graph/asc_graph_builder.h"

namespace mlir::test {
// Keep reduction-specific dtype and shape support local to MLIR tests.
template <typename Op>
af::testing::AscGraphBuilder &AddReduce(af::testing::AscGraphBuilder &builder, const std::string &name,
                                        const std::string &input, const std::vector<af::Expression> &inputShape,
                                        const std::vector<size_t> &axes, af::DataType outputType) {
  auto shape = inputShape;
  for (size_t axis : axes) {
    assert(axis < shape.size());
    shape[axis] = af::sym::kSymbolOne;
  }
  return builder.Op<Op>(name, {input}, shape, {}, outputType);
}
}  // namespace mlir::test
#endif
