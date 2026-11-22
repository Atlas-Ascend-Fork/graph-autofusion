/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_api_perf.h"
#include <string>
#include <algorithm>
#include <numeric>
#include "common/checker.h"
#include "base/att_const_values.h"
#include "api_perf_register/soc_perf_config.h"

using namespace af::sym;
namespace att {
const uint64_t kRptSize = 512U;

namespace ascendcperf {
af::Status GetAlignedCase(const NodeDetail &node_info, Expr &aligned_case, int32_t &use_case) {
  auto dims = node_info.input_dims;
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  auto iter1 = kRptEleMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(iter1 != kRptEleMap.end());
  Expr data_size = dim_product * iter1->second;
  if (data_size.IsConstExpr()) {
    int32_t datasize;
    data_size.GetConstValue(datasize);
    if (datasize % kRptSize == 0) {
      use_case = kCaseOne;
    } else {
      use_case = kCaseTwo;
    }
  } else {
    use_case = kCaseDefault;
    aligned_case = af::sym::Mod(data_size, kSymPowerofEight);
  }
  return af::SUCCESS;
}

af::Status RptElementwisePerf(const NodeDetail &node_info, const Expr &aligned_res, const Expr &unaligned_res,
                              PerfOutputInfo &perf) {
  int32_t use_case;
  Expr aligned_case;
  GE_ASSERT_SUCCESS(GetAlignedCase(node_info, aligned_case, use_case));
  if (use_case == kCaseOne) {
    perf.pipe_res[PipeType::AIV_VEC] = aligned_res;
  } else if (use_case == kCaseTwo) {
    perf.pipe_res[PipeType::AIV_VEC] = unaligned_res;
  } else {
    Expr res = CreateExpr("compare_if_aligned_node");
    std::shared_ptr<IfCase> branch_a = std::make_shared<IfCase>(aligned_res);
    std::shared_ptr<IfCase> branch_b = std::make_shared<IfCase>(unaligned_res);
    GE_ASSERT_NOTNULL(branch_a);
    GE_ASSERT_NOTNULL(branch_b);
    TernaryOp ternary_op =
        TernaryOp(CondType::K_EQ, aligned_case, CreateExpr(0), std::move(branch_a), std::move(branch_b));
    ternary_op.SetVariable(res);
    perf.ternary_ops[res] = ternary_op;
    perf.pipe_res[PipeType::AIV_VEC] = res;
  }
  return af::SUCCESS;
}

af::Status GetDatasize(const TensorShapeInfo &shape, Expr &dim_product) {
  auto dims = shape.dims;
  GE_ASSERT_TRUE(!dims.empty());
  dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return af::sym::Mul(a, b); });
  return af::SUCCESS;
}

af::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  std::string registered_key_name;
  GE_ASSERT_SUCCESS(GetApiRegisterVerName(registered_key_name));
  const auto load_perf_func = GetAscendCPerfFunc(kLoad + registered_key_name);
  GE_ASSERT_NOTNULL(load_perf_func);
  return load_perf_func(node_info, perf);
}

/*
Absapi的性能公式：
  float16: (0.0077 * data_size + 20.0153) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0147 * data_size + 20.0592) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kAbs, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Addsapi的性能公式：
  float16: (0.0071 * data_size + 22.0938) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0141 * data_size + 22.0936) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status AddsPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kAdds, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Addapi的性能公式：
  float16: (0.0103 * data_size + 22.2173) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0206 * data_size + 23.2225) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kAdd, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Andapi的性能公式：
  float16: (0.0107 * data_size + 17.1393) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0112 * data_size + 17.5611) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status AndPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kAnd, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
BlockReduceMaxapi的性能公式：
  float16: (0.0547 * data_size + 18.0042) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0198 * data_size + 16.7401) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status BlockReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kBlockReduceMax, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
BlockReduceMinapi的性能公式：
  float16: (0.0547 * data_size + 18.0092) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0198 * data_size + 16.7413) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status BlockReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kBlockReduceMin, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Brcbapi的性能公式：
  float16: (0.0074 * data_size + 13.0572) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0146 * data_size + 13.0732) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status BrcbPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kBrcb, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Castapi的性能公式：
  fp16touint8: (0.0083 * data_size + 19.9408) * 调用次数 + 37.37(PIPE头开销)
  fp16tofp32: (0.0147 * data_size + 20.1204) * 调用次数 + 37.37(PIPE头开销)
  fp32tofp16: (0.0087 * data_size + 20.4393) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kCast, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CopyUbtoUbapi的性能公式：
  fp16: (0.0076 * data_size + 11.6372) * 调用次数 + 37.37(PIPE头开销)
  fp32: (0.0152 * data_size + 11.6372) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CopyUbtoUbPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kUb2ub, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CopyUbapi的性能公式：
  fp16: (0.0078 * data_size + 13.0049) * 调用次数 + 37.37(PIPE头开销)
  fp32: (0.0157 * data_size + 12.9966) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CopyPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kCopy, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CompareScalarEQapi的性能公式：
  float16: (0.0084 * data_size + 21.9204) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0160 * data_size + 21.9749) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CompareScalarEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareScalarEQ, node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CompareEQapi的性能公式：
  float16: (0.0155 * data_size + 21.0340) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0081 * data_size + 21.0340) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
  float32: (0.0310 * data_size + 21.0316) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0157 * data_size + 21.0316) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
*/
af::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr aligned_res;
  Expr unaligned_res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareEQ + "Aligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            aligned_res));
  GE_ASSERT_SUCCESS(GetPerf({kCompareEQ + "Unaligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            unaligned_res));
  GE_ASSERT_SUCCESS(RptElementwisePerf(node_info, aligned_res, unaligned_res, perf));
  return af::SUCCESS;
}

/*
CompareScalarGEapi的性能公式：
  float16: (0.0086 * data_size + 21.9025) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0161 * data_size + 21.9643) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CompareScalarGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareScalarGE, node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CompareGEapi的性能公式：
  float16: (0.0155 * data_size + 21.0279) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0079 * data_size + 21.0279) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
  float32: (0.0310 * data_size + 20.9711) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0156 * data_size + 20.9711) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
*/
af::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr aligned_res;
  Expr unaligned_res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareGE + "Aligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            aligned_res));
  GE_ASSERT_SUCCESS(GetPerf({kCompareGE + "Unaligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            unaligned_res));
  GE_ASSERT_SUCCESS(RptElementwisePerf(node_info, aligned_res, unaligned_res, perf));
  return af::SUCCESS;
}

/*
CompareScalarGTapi的性能公式：
  float16: (0.0084 * data_size + 21.9200) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0160 * data_size + 21.9712) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CompareScalarGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareScalarGT, node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CompareGTapi的性能公式：
  float16: (0.0160 * data_size + 20.9711) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0114 * data_size + 20.9711) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
  float32: (0.0310 * data_size + 21.0319) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0157 * data_size + 21.0319) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
*/
af::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr aligned_res;
  Expr unaligned_res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareGT + "Aligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            aligned_res));
  GE_ASSERT_SUCCESS(GetPerf({kCompareGT + "Unaligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            unaligned_res));
  GE_ASSERT_SUCCESS(RptElementwisePerf(node_info, aligned_res, unaligned_res, perf));
  return af::SUCCESS;
}

/*
CompareScalarLEapi的性能公式：
  float16: (0.0084 * data_size + 21.9210) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0161 * data_size + 21.9722) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CompareScalarLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareScalarLE, node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CompareLEapi的性能公式：
  float16: (0.0155 * data_size + 21.9556) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0079 * data_size + 21.9556) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
  float32: (0.0310 * data_size + 21.0303) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0156 * data_size + 21.0303) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
*/
af::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr aligned_res;
  Expr unaligned_res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareLE + "Aligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            aligned_res));
  GE_ASSERT_SUCCESS(GetPerf({kCompareLE + "Unaligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            unaligned_res));
  GE_ASSERT_SUCCESS(RptElementwisePerf(node_info, aligned_res, unaligned_res, perf));
  return af::SUCCESS;
}

/*
CompareScalarLTapi的性能公式：
  float16: (0.0084 * data_size + 21.9381) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0161 * data_size + 22.9860) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CompareScalarLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareScalarLT, node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CompareLTapi的性能公式：
  float16: (0.0155 * data_size + 21.0297) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0079 * data_size + 21.0297) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
  float32: (0.0316 * data_size + 20.9940) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0173 * data_size + 20.9940) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
*/
af::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr aligned_res;
  Expr unaligned_res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareLT + "Aligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            aligned_res));
  GE_ASSERT_SUCCESS(GetPerf({kCompareLT + "Unaligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            unaligned_res));
  GE_ASSERT_SUCCESS(RptElementwisePerf(node_info, aligned_res, unaligned_res, perf));
  return af::SUCCESS;
}

/*
CompareScalarNEapi的性能公式：
  float16: (0.0084 * data_size + 21.9114) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0161 * data_size + 22.9690) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status CompareScalarNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareScalarNE, node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
CompareNEapi的性能公式：
  float16: (0.0164 * data_size + 20.9123) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0115 * data_size + 20.9123) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
  float32: (0.0310 * data_size + 21.0313) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B == 0)
           (0.0156 * data_size + 21.0313) * 调用次数 + 37.37(PIPE头开销) (data_size % 512B != 0)
*/
af::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr aligned_res;
  Expr unaligned_res;
  GE_ASSERT_SUCCESS(GetPerf({kCompareNE + "Aligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            aligned_res));
  GE_ASSERT_SUCCESS(GetPerf({kCompareNE + "Unaligned", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride},
                            unaligned_res));
  GE_ASSERT_SUCCESS(RptElementwisePerf(node_info, aligned_res, unaligned_res, perf));
  return af::SUCCESS;
}

/*
Powerapi的性能公式：
 全Tensor:
  float16: (0.6647 * data_size + 2516.53) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.6199 * data_size + 730.11) * 调用次数 + 37.37(PIPE头开销)
 含Scalar:
  float16: (0.64965 * data_size + 2517.18) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.69805 * data_size + 735.05) * 调用次数 + 37.37(PIPE头开销)
*/

af::Status PowerPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  if (node_info.input_dtype.size() == kNumTwo) {
    GE_ASSERT_SUCCESS(GetPerf({kPower + "AllTensor", node_info.input_dtype[0], node_info.output_dtype[0],
                               node_info.input_dims, node_info.gm_stride},
                              res));
  } else {
    GE_ASSERT_SUCCESS(GetPerf({kPower + "WithScalar", node_info.input_dtype[0], node_info.output_dtype[0],
                               node_info.input_dims, node_info.gm_stride},
                              res));
  }
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Divapi的性能公式：
  float16: (0.0460 * data_size + 29.1233) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0454 * data_size + 29.0892) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kDiv, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Duplicateapi的性能公式：
  float16: (0.0078 * data_size + 16.9993) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0156 * data_size + 16.9965) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status DuplicatePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kDuplicate, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Erfapi的性能公式：
  注意：Erf为高阶api，可以被视作为多个mins,maxs,mul,adds,muls算子的组合
  float16: (0.6996 * data_size + 478.4175) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.6038 * data_size + 458.2933) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kErf, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Expapi的性能公式：
  float16: (0.0311 * data_size + 28.0144) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0307 * data_size + 28.0376) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kExp, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Gatherapi的性能公式：
  float16: (0.1873 * data_size + 17.0248) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1875 * data_size + 15.0000) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status GatherPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kGather, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Gathermaskapi的性能公式：
  float16: (0.0156 * data_size + 14.0242) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0313 * data_size + 14.0207) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status GatherMaskPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kGatherMask, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Maxsapi的性能公式：
  float16: (0.0071 * data_size + 20.0912) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0141 * data_size + 20.0887) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status MaxsPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kMaxs, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Maxapi的性能公式：
  float16: (0.0111 * data_size + 20.1200) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0215 * data_size + 20.1333) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kMax, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Minsapi的性能公式：
  float16: (0.0071 * data_size + 20.0896) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0142 * data_size + 20.0876) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status MinsPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kMins, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Minapi的性能公式：
  float16: (0.0111 * data_size + 20.1104) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0215 * data_size + 20.1271) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kMin, node_info.input_dtype[0], node_info.output_dtype[0], node_info.output_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Mulsapi的性能公式：
  float16: (0.0071 * data_size + 23.1006) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0142 * data_size + 23.0966) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status MulsPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kMuls, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Mulapi的性能公式：
  float16: (0.0110 * data_size + 23.1243) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0206 * data_size + 23.2291) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kMul, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Orapi的性能公式：
  uint16: (0.0132 * data_size + 12.4018) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status OrPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kOr, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
PairReduceSumapi的性能公式：
  float16: (0.0547 * data_size + 37.159) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1094 * data_size + 36.964) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status PairReduceSumPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kPairReduceSum, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Reciprocalapi的性能公式：
  float16: (0.0078 * data_size + 21.0076) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0146 * data_size + 21.0639) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kReciprocal, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Reluapi的性能公式：
  float16: (0.0077 * data_size + 20.0173) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0154 * data_size + 20.0189) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kRelu, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Rsqrtapi的性能公式：
  float16: (0.0071 * data_size + 21.0970) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0143 * data_size + 21.0979) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kRsqrt, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Selectapi的性能公式：
  float16: (0.0118 * data_size + 45.9656) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0229 * data_size + 43.9906) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status SelectPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kSelect, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Sigmoidapi的性能公式：
  Sigmoid是一个高阶公式，本质上调用了多次Muls,Exp,Adds,Duplicate,Div
  float16: (0.1011 * data_size + 116.0436) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1256 * data_size + 115.9747) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kSigmoid, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Signapi的性能公式：
  调用了muls, abs, adds, div四个基础算子
  float16: (0.0855 * data_size + 119.0821) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1701 * data_size + 119.0656) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kSign, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Sqrtapi的性能公式：
  float16: (0.0312 * data_size + 29.0056) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0313 * data_size + 28.9961) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kSqrt, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Subapi的性能公式：
  float16: (0.0107 * data_size + 22.1226) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.0213 * data_size + 22.1254) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kSub, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
Tanhapi的性能公式：
  高阶api，half额外包含两个Cast算子，包含mins,maxs,muls,exp,两次adds, 一次div
  float16: (0.1976 * data_size + 181.6919) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1570 * data_size + 153.9298) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kTanh, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride}, res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
WholeReduceSumapi的性能公式：
  float16: (0.0547 * data_size + 35.0021) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1094 * data_size + 32.0029) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status WholeReduceSumPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kWholeReduceSum, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
WholeReduceMaxapi的性能公式：
  float16: (0.0547 * data_size + 21.0027) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1094 * data_size + 20.0051) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status WholeReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kWholeReduceMax, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
WholeReduceMinapi的性能公式：
  float16: (0.0547 * data_size + 21.0056) * 调用次数 + 37.37(PIPE头开销)
  float32: (0.1094 * data_size + 20.0068) * 调用次数 + 37.37(PIPE头开销)
*/
af::Status WholeReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GE_ASSERT_SUCCESS(GetPerf(
      {kWholeReduceMin, node_info.input_dtype[0], node_info.output_dtype[0], node_info.input_dims, node_info.gm_stride},
      res));
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

af::Status VectorCompute([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  auto dims =
      input_shapes[0].dims.size() >= output_shapes[0].dims.size() ? input_shapes[0].dims : output_shapes[0].dims;
  auto data_type_size = input_shapes[0].data_type_size > output_shapes[0].data_type_size
                            ? input_shapes[0].data_type_size
                            : output_shapes[0].data_type_size;
  GE_ASSERT_TRUE(!dims.empty());
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  Expr data_size = Mul(dim_product, CreateExpr(data_type_size));
  auto cycles = Div(data_size, kSymPowerofSeven);
  Expr dens = Sub(Mul(dims.back(), kInitA), kInitB);
  if (dens == 0) {
    dens = CreateExpr(1);
  }
  auto weight = Div(kSymPowerofEight, dens);
  cycles = Mul(cycles, weight);
  const auto kHeadCost = CreateExpr(4U);
  perf_res.pipe_res[PipeType::AIV_VEC] = Add(cycles, kHeadCost);
  return af::SUCCESS;
}

af::Status MatmulCompute([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(input_shapes.size() >= 2U && !output_shapes.empty());
  auto a_shape = input_shapes[0].dims;
  auto b_shape = input_shapes[1].dims;
  auto c_shape = output_shapes[0].dims;
  auto input_size = a_shape.size();
  auto output_size = c_shape.size();
  GE_ASSERT_TRUE((input_size >= 2U) && (output_size >= 2U), "MatMul input/output shape is not valid.");
  auto m = c_shape[output_size - 2U];
  auto k = a_shape[input_size - 1U];
  auto n = c_shape[output_size - 1U];
  m = Div(m, kSymSixteen);
  k = Div(k, kSymSixteen);
  n = Div(n, kSymSixteen);
  auto bw_gm = CreateExpr(4.7087f);
  auto bw_l2 = CreateExpr(62.2890f);
  auto alpha_a = CreateExpr(0.9504f);
  auto beta_a = CreateExpr(20.0944f);
  auto gamma_a = CreateExpr(1.2069f);
  auto alpha_b = CreateExpr(0.8999f);
  auto beta_b = CreateExpr(0.3985f);
  auto gamma_b = CreateExpr(0.9217f);
  auto factor_a = Add(Mul(k, gamma_a), CreateExpr(0.0000000014f));
  auto factor_b = Add(Mul(n, gamma_b), CreateExpr(0.0000000026f));
  auto repeat_a = Add(Mul(n, alpha_a), beta_a);
  auto repeat_b = Add(Mul(m, alpha_b), beta_b);
  auto load_a_gm = Div(Mul(m, k), bw_gm);
  auto load_a_l2 = Mul(Div(Mul(m, k), bw_l2), repeat_a);
  auto load_b_gm = Div(Mul(k, n), bw_gm);
  auto load_b_l2 = Mul(Div(Mul(k, n), bw_l2), repeat_b);
  auto mte2_cycle = Add(Mul(Add(load_a_gm, load_a_l2), factor_a), Mul(Add(load_b_gm, load_b_l2), factor_b));
  auto ratio_m = Mul(m, CreateExpr(0.005784f));
  auto ratio_n = Mul(n, CreateExpr(0.003456f));
  auto ratio_k = Mul(k, CreateExpr(0.009211f));
  auto ratio_a = Div(Mul(m, k), CreateExpr(37316800400.0f));
  auto ratio_b = Div(Mul(n, k), CreateExpr(172892216000.0f));
  auto ratio_c = Div(Mul(m, n), CreateExpr(87941155700.0f));
  auto ratio = Add(Add(ratio_a, ratio_b), ratio_c);
  ratio = Add(ratio, Add(Add(ratio_m, ratio_n), ratio_k));
  perf_res.pipe_res[PipeType::AIC_MTE2] = Div(mte2_cycle, ratio);
  return af::SUCCESS;
}

af::Status AndApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::AndPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status AddsApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::AddsPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status BlockReduceMaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::BlockReduceMaxPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status BlockReduceMinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::BlockReduceMinPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status BrcbApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::BrcbPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareScalarEQApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarEQPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareScalarGEApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarGEPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareScalarGTApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarGTPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareScalarLEApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarLEPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareScalarNEApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarNEPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareScalarLTApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarLTPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status PowerApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::PowerPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status PairReduceSumApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                            [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                            [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::PairReduceSumPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status DuplicateApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::DuplicatePerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status DropoutCompute([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  (void)output_shapes;
  Expr dim_product;
  Expr t = CreateExpr(50.05f);
  Expr c = CreateExpr(0.99f);
  Expr h = CreateExpr(74.52f);
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  GE_ASSERT_SUCCESS(GetDatasize(input_shapes[0], dim_product));
  auto cycles = Div(dim_product, t);
  cycles = Mul(cycles, c);
  perf_res.pipe_res[PipeType::AIV_VEC] = Add(cycles, h);
  return af::SUCCESS;
}

af::Status DefaultMTE1Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  perf_res.pipe_res[PipeType::AICORE_MTE1] = af::sym::kSymbolOne;
  return af::SUCCESS;
}

af::Status DefaultMTE2Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  perf_res.pipe_res[PipeType::AICORE_MTE2] = af::sym::kSymbolOne;
  return af::SUCCESS;
}

af::Status DefaultMTE3Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  perf_res.pipe_res[PipeType::AICORE_MTE3] = af::sym::kSymbolOne;
  return af::SUCCESS;
}

af::Status DefaultVECApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  // rand/randn have no input shape, so check output shape only
  GE_ASSERT_TRUE(!output_shapes.empty());
  perf_res.pipe_res[PipeType::AIV_VEC] = af::sym::kSymbolOne;
  return af::SUCCESS;
}

af::Status DefaultCUBEApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  perf_res.pipe_res[PipeType::AICORE_CUBE] = af::sym::kSymbolOne;
  return af::SUCCESS;
}

af::Status WholeReduceMaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::WholeReduceMaxPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status WholeReduceMinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::WholeReduceMinPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status WholeReduceSumApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::WholeReduceSumPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CubeCompute([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                       [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                       [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(input_shapes.size() >= 2U && !output_shapes.empty());
  auto a_shape = input_shapes[0].dims;
  auto b_shape = input_shapes[1].dims;
  auto output_dims = output_shapes[0].dims;
  Expr res = CreateExpr(1);
  if (a_shape.size() >= 2U) {
    for (uint32_t i = 0U; i < a_shape.size(); i++) {
      auto length = input_shapes[0].dims[i];
      res = Mul(res, Div(Add(length, kSymFifteen), kSymSixteen));
    }
    for (uint32_t i = 1U; i < b_shape.size(); i++) {
      auto length = input_shapes[1].dims[i];
      res = Mul(res, Div(Add(length, kSymFifteen), kSymSixteen));
    }
  }
  perf_res.pipe_res[PipeType::AICORE_CUBE] = res;
  return af::SUCCESS;
}

af::Status GatherMaskApi(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, [[maybe_unused]] const NodeInfo &node,
                         PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::GatherMaskPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status MaxsApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::MaxsPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status MinsApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::MinsPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status MulsApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::MulsPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status MulApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::MulPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status OrApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                 [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                 [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::OrPerf(node_info, perf_res));
  return af::SUCCESS;
}

/*
SetVectorMaskApi与shape无关，性能为0
*/
af::Status SetVectorMaskApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                            [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                            [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  (void)input_shapes;
  (void)output_shapes;
  perf_res.pipe_res[PipeType::AIV_VEC] = af::sym::kSymbolZero;
  return af::SUCCESS;
}

af::Status SigmoidApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                      [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                      [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::SigmoidPerf(node_info, perf_res));
  return af::SUCCESS;
}
}  // namespace ascendcperf

Perf GetPerfFunc(const std::string &op_type) {
  return EvalCosts::Instance().GetFunc(op_type);
}

AscendCPerf GetAscendCPerfFunc(const std::string &op_type) {
  AscendCPerf default_func = EvalCosts::Instance().GetAscendCFunc(op_type);
  if (default_func == nullptr) {
    return nullptr;
  }

  SocPerfTarget target;
  bool required = false;
  if (ReadSocPerfTarget(target) != af::SUCCESS || RequiresFullSocVersion(target.npu_arch, required) != af::SUCCESS ||
      !required) {
    return default_func;
  }
  AscendCPerf selected_func = nullptr;
  if (ResolveSocAscendCPerf(target, op_type, default_func, selected_func) != af::SUCCESS) {
    return nullptr;
  }
  return selected_func;
}

EvalCosts &EvalCosts::Instance() {
  static EvalCosts eval_costs;
  return eval_costs;
}

REGISTER_EVAL_FUNC(kComputeVector, ascendcperf::VectorCompute);
REGISTER_EVAL_FUNC(kComputeCube, ascendcperf::CubeCompute);
REGISTER_EVAL_FUNC(kAdds, ascendcperf::AddsApi);
REGISTER_EVAL_FUNC(kAnd, ascendcperf::AndApi);
REGISTER_EVAL_FUNC(kBlockReduceMax, ascendcperf::BlockReduceMaxApi);
REGISTER_EVAL_FUNC(kBlockReduceMin, ascendcperf::BlockReduceMinApi);
REGISTER_EVAL_FUNC(kBrcb, ascendcperf::BrcbApi);
REGISTER_EVAL_FUNC(kCompareScalarEQ, ascendcperf::CompareScalarEQApi);
REGISTER_EVAL_FUNC(kCompareScalarGE, ascendcperf::CompareScalarGEApi);
REGISTER_EVAL_FUNC(kCompareScalarGT, ascendcperf::CompareScalarGTApi);
REGISTER_EVAL_FUNC(kCompareScalarLE, ascendcperf::CompareScalarLEApi);
REGISTER_EVAL_FUNC(kCompareScalarNE, ascendcperf::CompareScalarNEApi);
REGISTER_EVAL_FUNC(kCompareScalarLT, ascendcperf::CompareScalarLTApi);
REGISTER_EVAL_FUNC(kPower, ascendcperf::PowerApi);
REGISTER_EVAL_FUNC(kPairReduceSum, ascendcperf::PairReduceSumApi);
REGISTER_EVAL_FUNC(kDuplicate, ascendcperf::DuplicateApi);
REGISTER_EVAL_FUNC(kGatherMask, ascendcperf::GatherMaskApi);
REGISTER_EVAL_FUNC(kMaxs, ascendcperf::MaxsApi);
REGISTER_EVAL_FUNC(kMax,
                   ascendcperf::WholeReduceMaxApi);  // 这里临时注册，需要把注册放在ascir中，并且替换reduce sum/min/max
REGISTER_EVAL_FUNC(kMins, ascendcperf::MinsApi);
REGISTER_EVAL_FUNC(kMin,
                   ascendcperf::WholeReduceMinApi);  // 这里临时注册，需要把注册放在ascir中，并且替换reduce sum/min/max
REGISTER_EVAL_FUNC(kMuls, ascendcperf::MulsApi);
REGISTER_EVAL_FUNC(kMul, ascendcperf::MulApi);
REGISTER_EVAL_FUNC(kNeg, ascendcperf::MulsApi);
REGISTER_EVAL_FUNC(kOr, ascendcperf::OrApi);
REGISTER_EVAL_FUNC(kSetVectorMask, ascendcperf::SetVectorMaskApi);
REGISTER_EVAL_FUNC(kSigmoid, ascendcperf::SigmoidApi);
REGISTER_EVAL_FUNC(kSum,
                   ascendcperf::WholeReduceSumApi);  // 这里临时注册，需要把注册放在ascir中，并且替换reduce sum/min/max
REGISTER_EVAL_FUNC(kWholeReduceMax, ascendcperf::WholeReduceMaxApi);
REGISTER_EVAL_FUNC(kWholeReduceMin, ascendcperf::WholeReduceMinApi);
REGISTER_EVAL_FUNC(kWholeReduceSum, ascendcperf::WholeReduceSumApi);
REGISTER_EVAL_FUNC(kConstant, ascendcperf::VectorCompute);
// Cube
REGISTER_EVAL_FUNC(kDropOut, ascendcperf::DropoutCompute);
REGISTER_EVAL_FUNC(kMatMul, ascendcperf::MatmulCompute);
// Pipe default
REGISTER_EVAL_FUNC(kUnitMTE1, ascendcperf::DefaultMTE1Api);
REGISTER_EVAL_FUNC(kUnitMTE2, ascendcperf::DefaultMTE2Api);
REGISTER_EVAL_FUNC(kUnitMTE3, ascendcperf::DefaultMTE3Api);
REGISTER_EVAL_FUNC(kUnitVector, ascendcperf::DefaultVECApi);
REGISTER_EVAL_FUNC(kUnitCube, ascendcperf::DefaultCUBEApi);

}  // namespace att
