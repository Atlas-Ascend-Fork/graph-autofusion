/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCIR_REDUCE_API_PERF_V2_H_
#define AUTOFUSE_ASCIR_REDUCE_API_PERF_V2_H_

#include "api_perf_register/ascendc_api_perf.h"

namespace att {
namespace ascir_reduce_v2 {
af::Status ElementwiseMaxApi(const std::vector<TensorShapeInfo> &input_shapes,
                             const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                             PerfOutputInfo &perf_res);
af::Status ElementwiseMinApi(const std::vector<TensorShapeInfo> &input_shapes,
                             const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                             PerfOutputInfo &perf_res);
af::Status ReduceMaxApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res);
af::Status ReduceMinApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res);
af::Status ReduceAnyApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res);
af::Status ReduceAllApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res);
af::Status ReduceSumApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res);
af::Status ReduceMeanApi(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                         PerfOutputInfo &perf_res);
af::Status ReduceProdApi(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                         PerfOutputInfo &perf_res);
af::Status MaxApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status MinApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status AnyApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status AllApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status SumApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status MeanApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                   const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status ProdApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                   const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status ArgmaxApi(const std::vector<TensorShapeInfo> &input_shapes,
                     const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status ArgminApi(const std::vector<TensorShapeInfo> &input_shapes,
                     const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node, PerfOutputInfo &perf_res);
af::Status ArgMaxMultiRPhase1Api(const std::vector<TensorShapeInfo> &input_shapes,
                                 const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                 PerfOutputInfo &perf_res);
af::Status ArgMaxMultiRPhase2Api(const std::vector<TensorShapeInfo> &input_shapes,
                                 const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                 PerfOutputInfo &perf_res);
af::Status ArgmaxWithValueApi(const std::vector<TensorShapeInfo> &input_shapes,
                              const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                              PerfOutputInfo &perf_res);
af::Status ArgMaxMultiRPhase2WithValueApi(const std::vector<TensorShapeInfo> &input_shapes,
                                          const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                          PerfOutputInfo &perf_res);
}  // namespace ascir_reduce_v2
}  // namespace att

#endif  // AUTOFUSE_ASCIR_REDUCE_API_PERF_V2_H_
