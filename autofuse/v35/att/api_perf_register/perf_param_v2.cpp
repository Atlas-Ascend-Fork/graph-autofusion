/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "perf_param_v2.h"
#include "base/att_const_values.h"

namespace att {
namespace {
const uint64_t kLoadThreshold = 256U;
const Expr kMTE2Thres = CreateExpr(256U);
const Expr kMte2PipeHeadNormalCost = CreateExpr(1174.3f);
const Expr kMte2PipeHeadSmallCost = CreateExpr(775.0f);
const Expr kMte3PipeHeadCost = CreateExpr(571.0f);
}  // namespace
Expr PerfParamTableV2::GetMTE2PipeHead(const std::vector<NodeInfo> &node_infos,
                                       std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) {
  Expr mte2_pipe_head;
  Expr max_blk = af::sym::kSymbolZero;
  for (const auto &node : node_infos) {
    if (node.node_type.find(kLoad) == std::string::npos) {
      continue;
    }
    auto dims = node.inputs[0]->repeat;
    auto dim_product = dims.empty() ? CreateExpr(1) : dims[dims.size() - 1];
    Expr blk = dim_product * CreateExpr(node.inputs[0]->data_type_size);
    if (max_blk != af::sym::kSymbolZero) {
      max_blk = af::sym::Max(blk, max_blk);
    } else {
      max_blk = blk;
    }
  }
  const auto block_dim = CreateExpr("block_dim");
  if (max_blk.IsConstExpr()) {
    uint64_t blocklen;
    max_blk.GetConstValue(blocklen);
    if (blocklen >= kLoadThreshold) {
      mte2_pipe_head = kMte2PipeHeadNormalCost;
    } else {
      mte2_pipe_head = kMte2PipeHeadSmallCost;
    }
  } else {
    GetPerfVar("mte2_pipe_head", mte2_pipe_head, ternary_ops);
    TernaryOp ternary_op =
        TernaryOp(CondType::K_LT, max_blk, kMTE2Thres, kMte2PipeHeadSmallCost, kMte2PipeHeadNormalCost);
    ternary_op.SetVariable(mte2_pipe_head);
    ternary_ops[mte2_pipe_head] = ternary_op;
  }
  return mte2_pipe_head;
}
// 当前仅根据输出类型来判断
const std::map<std::string, std::vector<VfInstructPerf>> &PerfParamTableV2::GetVfInstructPerfTable() const {
  static const std::map<std::string, std::vector<VfInstructPerf>> kVfInstructPerfTable = {
      {kAdd,
       {{VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32}, 4, 2}},
        {VfInstructPerf{{kFloat32, kFloat16, kBfloat16}, 4, 1}}}},
      {kSub,
       {{VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32}, 4, 2}},
        {VfInstructPerf{{kFloat32, kFloat16, kBfloat16}, 4, 1}}}},
      {kMax,
       {
           {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
       }},
      {
          kMin,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kReduceMax,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kReduceMin,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kAbs,
          {
              {VfInstructPerf{{kFloat16, kFloat32}, 2, 1}},
              {VfInstructPerf{{kInt8, kInt16, kInt32}, 4, 2}},
          },
      },
      {
          kEq,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kNe,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kLt,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kGt,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kGe,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kLe,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16, kUInt64, kInt64},
                  3,
                  1}},
          },
      },
      {
          kAdds,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 4, 2}},
          },
      },
      {
          kMaxs,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kMins,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kNot,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kFloat16, kBfloat16, kUInt32, kInt32, kFloat32}, 3, 1}},
          },
      },
      {
          kExp,
          {
              {VfInstructPerf{{kFloat16}, 18, 8}},
              {VfInstructPerf{{kFloat32}, 13, 4}},
          },
      },
      {
          kLn,
          {
              {VfInstructPerf{{kFloat16}, 20, 8}},
              {VfInstructPerf{{kFloat32}, 15, 4}},
          },
      },
      {
          kSqrt,
          {
              {VfInstructPerf{{kFloat16}, 19, 8}},
              {VfInstructPerf{{kFloat32}, 14, 4}},
          },
      },
      {
          kMul,
          {
              {VfInstructPerf{{kUInt16, kInt16, kUInt32, kInt32}, 5, 2}},
              {VfInstructPerf{{kFloat16, kFloat32, kBfloat16}, 5, 1}},
          },
      },
      {
          kMuls,
          {
              {VfInstructPerf{{kUInt16, kInt16, kUInt32, kInt32}, 5, 2}},
              {VfInstructPerf{{kFloat16, kFloat32}, 5, 1}},
          },
      },
      {
          kRelu,
          {
              {VfInstructPerf{{kInt32, kFloat16, kFloat32}, 2, 1}},
          },
      },
      {
          kOr,
          {
              {VfInstructPerf{{kUInt8, kInt16, kInt32, kFloat16, kFloat32}, 3, 1}},
          },
      },
      {
          kNeg,
          {
              {VfInstructPerf{{kInt8, kInt16, kInt32}, 4, 2}},
              {VfInstructPerf{{kFloat16, kFloat32}, 2, 1}},
          },
      },
      {
          kDiv,
          {
              {VfInstructPerf{{kFloat16}, 19, 8}},
              {VfInstructPerf{{kFloat32}, 14, 4}},
          },
      },
      {
          kLeakyRelu,
          {
              {VfInstructPerf{{kInt32, kFloat16, kFloat32}, 2, 1}},
          },
      },
      // Cast的数据需要根据输入输出类型分别判断，先粗略计算
      {
          kCast,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32}, 4, 2}},
              {VfInstructPerf{{kFloat16, kFloat32, kBfloat16}, 5, 2}},
          },
      },
      {
          kMaskSel,
          {
              {VfInstructPerf{{kUInt8}, 4, 2}},
          },
      },
      {
          kVcadd,
          {
              {VfInstructPerf{{kUInt16, kInt16, kUInt32, kInt32}, 13, 1}},
              {VfInstructPerf{{kFloat16}, 21, 1}},
              {VfInstructPerf{{kFloat32}, 19, 1}},
          },
      },
      {
          kVsqz,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 15, 5}},
          },
      },
      {
          kDuplicate,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 8, 1}},
          },
      },
      // VCI
      {
          kArange,
          {
              {VfInstructPerf{{kInt8, kInt16, kInt32}, 4, 2}},
              {VfInstructPerf{{kFloat16, kFloat32}, 4, 1}},
          },
      },
      {
          kCompareScalarNE,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kCompareScalarEQ,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kCompareScalarLT,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kCompareScalarGE,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kCompareScalarGT,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kAnd,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kSelect,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kXor,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kVshrs,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      {
          kTruncate,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 5, 2}},
          },
      },
      {
          kMulAddDst,
          {
              {VfInstructPerf{{kUInt16, kInt16, kUInt32, kInt32}, 6, 2}},
              {VfInstructPerf{{kFloat16, kFloat32}, 6, 1}},
          },
      },
      {
          kUpdateMask,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 3, 1}},
          },
      },
      // VPACK
      {
          kPack,
          {
              {VfInstructPerf{{kUInt16, kInt16, kFloat16, kBfloat16, kUInt32, kInt32, kFloat32}, 8, 1}},
          },
      },
      // VUNPACK
      {
          kUnPack,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kFloat16, kBfloat16}, 8, 1}},
          },
      },
      // VINTLV
      {
          kInterleave,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kFloat16, kBfloat16, kUInt32, kInt32, kFloat32}, 8, 1}},
          },
      },
      // VDINTLV
      {
          kDeInterleave,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kFloat16, kBfloat16, kUInt32, kInt32, kFloat32}, 8, 1}},
          },
      },
      // PPACK
      {
          kMaskPack,
          {
              {VfInstructPerf{{kUInt8, kInt8}, 8, 1}},
          },
      },
      // PUNPACK
      {
          kMaskUnPack,
          {
              {VfInstructPerf{{kUInt8, kInt8}, 8, 1}},
          },
      },
      {
          kMaskAnd,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 4, 2}},
          },
      },
      {
          kMaskOr,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kUInt32, kInt32, kFloat16, kFloat32, kBfloat16}, 4, 2}},
          },
      },
      {
          // 当前已覆盖 DIST_NORM,DIST_BRC_B8,DIST_BRC_B16,DIST_UNPACK_B8,DIST_UNPACK_B16,DIST_UNPACK_B32,
          // DataCopyGather
          kLoad,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kBfloat16, kFloat16, kUInt32, kInt32, kFloat32, kUInt64, kInt64},
                  0,
                  0}},
          },
      },
      {
          // 当前已覆盖
          // DIST_NORM_B8,DIST_NORM_B16,DIST_NORM_B32,DIST_FIRST_ELEMENT_B8,DIST_FIRST_ELEMENT_B16
          // DIST_FIRST_ELEMENT_B32,DIST_PACK_B16,DIST_PACK_B32,DIST_PACK_B64,DIST_PACK4_B32
          kStore,
          {
              {VfInstructPerf{{kUInt8, kInt8, kUInt16, kInt16, kBfloat16, kFloat16}, 0, 5}},
              {VfInstructPerf{{kUInt32, kInt32, kFloat32, kUInt64, kInt64}, 0, 1}},
          },
      },
      {
          kVmAdd,
          {
              {VfInstructPerf{{kBfloat16, kFloat16, kFloat32}, 6, 1}},
          },
      },
      // vgather2
      {
          kGather2,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kBfloat16, kFloat16, kUInt32, kInt32, kFloat32, kUInt64, kInt64},
                  0,
                  1}},
          },
      },
      {
          kPlaceholder,
          {
              {VfInstructPerf{
                  {kUInt8, kInt8, kUInt16, kInt16, kBfloat16, kFloat16, kUInt32, kInt32, kFloat32, kUInt64, kInt64},
                  0,
                  0}},
          },
      },
  };
  return kVfInstructPerfTable;
}

const std::map<std::string, std::vector<VfInstructDtypeMappingPerf>> &
PerfParamTableV2::GetVfInstructDtypeMappingPerfTable() const {
  static const std::map<std::string, std::vector<VfInstructDtypeMappingPerf>> kVfInstructDtypeMappingPerfTable = {
      {
          kCast,
          {
              {VfInstructDtypeMappingPerf{{kFloat16, kBfloat16, kFloat32}, {kFloat16, kBfloat16, kFloat32}, 4, 1}},
              {VfInstructDtypeMappingPerf{{kDefault}, {kDefault}, 5, 2}},
          },
      },
      {
          kStore,
          {
              {VfInstructDtypeMappingPerf{{kUInt8}, {kInt64}, 0, 5}},
              {VfInstructDtypeMappingPerf{{kDefault}, {kDefault}, 0, 1}},
          },
      },
  };
  return kVfInstructDtypeMappingPerfTable;
}

PerfParamTableV2::PerfParamTableV2() {
  pipes_head_perf_.emplace(PipeType::AIV_MTE2, &PerfParamTableV2::GetMTE2PipeHead);
  pipes_head_perf_.emplace(
      PipeType::AIV_MTE3,
      [](const std::vector<NodeInfo> &node_infos, std::map<Expr, TernaryOp, ExprCmp> &ternary_ops) -> Expr {
        (void)node_infos;
        (void)ternary_ops;
        return kMte3PipeHeadCost;
      });
  vf_instruct_type_2_api_perf_ = GetVfInstructPerfTable();
  vf_instruct_type_2_dtype_mapping_api_perf_ = GetVfInstructDtypeMappingPerfTable();
}

const std::vector<VfInstructPerf> &PerfParamTableV2::GetVfInstructPerfTable(const std::string &vf_instruct_type) const {
  const auto &iter = vf_instruct_type_2_api_perf_.find(vf_instruct_type);
  if (iter == vf_instruct_type_2_api_perf_.end()) {
    return PerfParamTable::GetVfInstructPerfTable(vf_instruct_type);
  }
  return iter->second;
}

const std::vector<VfInstructDtypeMappingPerf> &PerfParamTableV2::GetVfInstructDtypeMappingPerfTable(
    const std::string &vf_instruct_type) const {
  const auto &iter = vf_instruct_type_2_dtype_mapping_api_perf_.find(vf_instruct_type);
  if (iter == vf_instruct_type_2_dtype_mapping_api_perf_.end()) {
    return PerfParamTable::GetVfInstructDtypeMappingPerfTable(vf_instruct_type);
  }
  return iter->second;
}

Expr PerfParamTableV2::GetVectorFunctionHeadCost() const {
  constexpr int32_t kVectorFunctionHeadCost = 20;
  return CreateExpr(kVectorFunctionHeadCost);
}

Expr PerfParamTableV2::GetOpHeadCost() const {
  constexpr auto kOpHeadCost = 0;
  return CreateExpr(kOpHeadCost);
}

std::string PerfParamTableV2::GetApiRegisterVerName() const {
  return "V2";
}

// 暂时通过extern的方式，待注册机制Ready，通过AscIrAtt类注册
const std::string kParamV2Info = R"(
{
    "Abs": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0077, "b": 20.0153},
            "float32tofloat32": {"k": 0.0147, "b": 20.0592}
        }
    },
    "Adds": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 22.0938},
            "float32tofloat32": {"k": 0.0141, "b": 22.0936}
        }
    },
    "Add": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0103, "b": 22.2173},
            "float32tofloat32": {"k": 0.0206, "b": 23.2225}
        }
    },
    "And": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0107, "b": 17.1393},
            "float32tofloat32": {"k": 0.0112, "b": 17.5611}
        }
    },
    "BlockReduceMax": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 18.0042},
            "float32tofloat32": {"k": 0.0198, "b": 16.7401}
        }
    },
    "BlockReduceMin": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 18.0092},
            "float32tofloat32": {"k": 0.0198, "b": 16.7413}
        }
    },
    "Brcb": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0074, "b": 13.0572},
            "float32tofloat32": {"k": 0.0146, "b": 13.0732}
        }
    },
    "Cast": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0, "b": 0},
            "float32tofloat32": {"k": 0, "b": 0},
            "float16tofloat32": {"k": 0.0147, "b": 20.1204},
            "float32tofloat16": {"k": 0.0087, "b": 20.4393},
            "float16touint8": {"k": 0.0083, "b": 19.9408},
            "uint8tofloat16": {"k": 0.0102, "b": 46.512},
            "int64toint32": {"k": 0.0332, "b": 18.8690}
        }
    },
    "Ub2ub": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0076, "b": 11.6372},
            "float32tofloat32": {"k": 0.0152, "b": 11.6372}
        }
    },
    "Copy": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0078, "b": 13.0049},
            "float32tofloat32": {"k": 0.0157, "b": 12.9966}
        }
    },
    "CompareScalarEQ": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9204},
            "float32tofloat32": {"k": 0.0160, "b": 21.9749}
        }
    },
    "CompareScalarGE": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0086, "b": 21.9025},
            "float32tofloat32": {"k": 0.0161, "b": 21.9643}
        }
    },
    "CompareScalarGT": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9200},
            "float32tofloat32": {"k": 0.0160, "b": 21.9712}
        }
    },
    "CompareScalarLE": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9210},
            "float32tofloat32": {"k": 0.0161, "b": 21.9722}
        }
    },
    "CompareScalarNE": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9114},
            "float32tofloat32": {"k": 0.0161, "b": 22.9690}
        }
    },
    "CompareScalarLT": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0084, "b": 21.9381},
            "float32tofloat32": {"k": 0.0161, "b": 22.9860}
        }
    },
    "CompareEQAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.0340},
            "float32tofloat32": {"k": 0.0310, "b": 21.0316}
        }
    },
    "CompareEQUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0081, "b": 21.0340},
            "float32tofloat32": {"k": 0.0157, "b": 21.0316}
        }
    },
    "CompareGEAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.0279},
            "float32tofloat32": {"k": 0.0310, "b": 21.0309}
        }
    },
    "CompareGEUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0079, "b": 21.0279},
            "float32tofloat32": {"k": 0.0156, "b": 21.0309}
        }
    },
    "CompareGTAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0160, "b": 20.9711},
            "float32tofloat32": {"k": 0.0310, "b": 21.0319}
        }
    },
    "CompareGTUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0114, "b": 20.9711},
            "float32tofloat32": {"k": 0.0157, "b": 21.0319}
        }
    },
    "CompareLEAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.9556},
            "float32tofloat32": {"k": 0.0310, "b": 21.0303}
        }
    },
    "CompareLEUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0080, "b": 21.9556},
            "float32tofloat32": {"k": 0.0158, "b": 21.0303}
        }
    },
    "CompareLTAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0155, "b": 21.0297},
            "float32tofloat32": {"k": 0.0316, "b": 20.9940}
        }
    },
    "CompareLTUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0079, "b": 21.0297},
            "float32tofloat32": {"k": 0.0173, "b": 20.9940}
        }
    },
    "CompareNEAligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0164, "b": 20.9123},
            "float32tofloat32": {"k": 0.0310, "b": 21.0313}
        }
    },
    "CompareNEUnaligned": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0115, "b": 20.9123},
            "float32tofloat32": {"k": 0.0156, "b": 21.0313}
        }
    },
    "PowerAllTensor": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.6647, "b": 2516.53},
            "float32tofloat32": {"k": 0.6199, "b": 730.11}
        }
    },
    "PowerWithScalar": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.64965, "b": 2517.18},
            "float32tofloat32": {"k": 0.69805, "b": 735.05}
        }
    },
    "Div": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0460, "b": 29.1233},
            "float32tofloat32": {"k": 0.0454, "b": 29.0892}
        }
    },
    "Duplicate": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0078, "b": 16.9993},
            "float32tofloat32": {"k": 0.0156, "b": 16.9965}
        }
    },
    "Erf": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.6996, "b": 478.4175},
            "float32tofloat32": {"k": 0.6038, "b": 458.2933}
        }
    },
    "Exp": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0311, "b": 28.0144},
            "float32tofloat32": {"k": 0.0307, "b": 28.0376}
        }
    },
    "Gather": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.1873, "b": 17.0248},
            "float32tofloat32": {"k": 0.1875, "b": 15.0000}
        }
    },
    "GatherMask": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0156, "b": 14.0242},
            "float32tofloat32": {"k": 0.0313, "b": 14.0207}
        }
    },
    "Maxs": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 20.0912},
            "float32tofloat32": {"k": 0.0141, "b": 20.0887}
        }
    },
    "Max": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0111, "b": 20.1200},
            "float32tofloat32": {"k": 0.0215, "b": 20.1333}
        }
    },
    "Mins": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 20.0896},
            "float32tofloat32": {"k": 0.0142, "b": 20.0876}
        }
    },
    "Min": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0111, "b": 20.1104},
            "float32tofloat32": {"k": 0.0215, "b": 20.1271}
        }
    },
    "Muls": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 23.1006},
            "float32tofloat32": {"k": 0.0142, "b": 23.0966},
            "int32toint32": {"k": 0.0180, "b": 20.7457}
        }
    },
    "Mul": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0110, "b": 23.1243},
            "float32tofloat32": {"k": 0.0206, "b": 23.2291}
        }
    },
    "Or": {
        "model_type": "SimpleLinear",
        "model_params": {
            "uint16touint16": {"k": 0.0132, "b": 12.4018}
        }
    },
    "PairReduceSum": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 37.159},
            "float32tofloat32": {"k": 0.1094, "b": 36.964}
        }
    },
    "Reciprocal": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0078, "b": 21.0076},
            "float32tofloat32": {"k": 0.0146, "b": 21.0639}
        }
    },
    "Relu": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0077, "b": 20.0173},
            "float32tofloat32": {"k": 0.0154, "b": 20.0189}
        }
    },
    "Rsqrt": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0071, "b": 21.0970},
            "float32tofloat32": {"k": 0.0143, "b": 21.0979}
        }
    },
    "Select": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0118, "b": 45.9656},
            "float32tofloat32": {"k": 0.0229, "b": 43.9906}
        }
    },
    "Sigmoid": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.1011, "b": 116.0436},
            "float32tofloat32": {"k": 0.1256, "b": 115.9747}
        }
    },
    "Sign": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0855, "b": 119.0821},
            "float32tofloat32": {"k": 0.1701, "b": 119.0656}
        }
    },
    "Sqrt": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0312, "b": 29.0056},
            "float32tofloat32": {"k": 0.0313, "b": 28.9961}
        }
    },
    "Sub": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0107, "b": 22.1226},
            "float32tofloat32": {"k": 0.0213, "b": 22.1254}
        }
    },
    "Tanh": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.1976, "b": 181.6919},
            "float32tofloat32": {"k": 0.1570, "b": 153.9298}
        }
    },
    "WholeReduceMax": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 21.0027},
            "float32tofloat32": {"k": 0.1094, "b": 20.0051}
        }
    },
    "WholeReduceMin": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 21.0056},
            "float32tofloat32": {"k": 0.1094, "b": 20.0068}
        }
    },
    "WholeReduceSum": {
        "model_type": "SimpleLinear",
        "model_params": {
            "float16tofloat16": {"k": 0.0547, "b": 35.0021},
            "float32tofloat32": {"k": 0.1094, "b": 32.0029}
        }
    },
    "LoadSmallBlk": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 160, "b": 6.4088, "a": 13.1355, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 160, "b": 6.4088, "a": 13.1355, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 160, "b": 6.4088, "a": 13.1355, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 160, "b": 6.4088, "a": 13.1355, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 160, "b": 6.4088, "a": 13.1355, "hl": 0, "data_type_size": 2}
        }
    },
    "Load": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 160, "b": 6.6155, "a": 11.8292, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 160, "b": 6.6155, "a": 11.8292, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 160, "b": 6.6155, "a": 11.8292, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 160, "b": 6.6155, "a": 11.8292, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 160, "b": 6.6155, "a": 11.8292, "hl": 0, "data_type_size": 2}
        }
    },
    "LoadStride": {
        "model_type": "LoadStoreStrideV2Func",
        "model_params": {
            "int8toint8": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "int16toint16": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "uint16touint16": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "int32toint32": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "uint32touint32": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "uint64touint64": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "booltobool": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "int64toint64": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "float16tofloat16": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "float32tofloat32": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "uint8touint8": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0},
            "bfloat16tobfloat16": {"k": 0.005, "u": 4096.0, "penalty_coeff": 0.0}
        }
    },
    "NddmaStride": {
        "model_type": "LoadStoreStrideV2WithPenaltyFunc",
        "model_params": {
            "int64toint64": {"k": 0.005, "u": 4096.0, "penalty_coeff": 4.0},
            "float16tofloat16": {"k": 0.005, "u": 4096.0, "penalty_coeff": 4.0},
            "float32tofloat32": {"k": 0.005, "u": 4096.0, "penalty_coeff": 4.0},
            "uint8touint8": {"k": 0.005, "u": 4096.0, "penalty_coeff": 4.0},
            "bfloat16tobfloat16": {"k": 0.005, "u": 4096.0, "penalty_coeff": 4.0}
        }
    },
    "Store": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 160, "b": 10.265, "a": 11.774, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 160, "b": 10.265, "a": 11.774, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 160, "b": 10.265, "a": 11.774, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 160, "b": 10.265, "a": 11.774, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 160, "b": 10.265, "a": 11.774, "hl": 0, "data_type_size": 2}
        }
    },
    "StoreStride": {
        "model_type": "LoadStoreStrideV2Func",
        "model_params": {
            "int8toint8": {"k": 0.0050441522125085, "u": 4096.0, "penalty_coeff": 0.0},
            "int16toint16": {"k": 0.0050219521009849, "u": 4096.0, "penalty_coeff": 0.0},
            "uint16touint16": {"k": 0.0049981859819869, "u": 4096.0, "penalty_coeff": 0.0},
            "int32toint32": {"k": 0.0050058211763464, "u": 4096.0, "penalty_coeff": 0.0},
            "uint32touint32": {"k": 0.0050046751614247, "u": 4096.0, "penalty_coeff": 0.0},
            "uint64touint64": {"k": 0.0049785878890068, "u": 4096.0, "penalty_coeff": 0.0},
            "booltobool": {"k": 0.0050337712308162, "u": 4096.0, "penalty_coeff": 0.0},
            "int64toint64": {"k": 0.0050306649073631, "u": 4096.0, "penalty_coeff": 0.0},
            "float16tofloat16": {"k": 0.005009566773717, "u": 4096.0, "penalty_coeff": 0.0},
            "float32tofloat32": {"k": 0.0050093180829704, "u": 4096.0, "penalty_coeff": 0.0},
            "uint8touint8": {"k": 0.0049922306718019, "u": 4096.0, "penalty_coeff": 0.0},
            "bfloat16tobfloat16": {"k": 0.0050004678761787, "u": 4096.0, "penalty_coeff": 0.0}
        }
    },
    "Nddma": {
        "model_type": "LoadStoreFunc",
        "model_params": {
            "int64toint64": {"h": 418.9789, "b": 6.39, "a": 7.61, "hl": 0, "data_type_size": 8},
            "float16tofloat16": {"h": 418.9789, "b": 6.39, "a": 7.61, "hl": 0, "data_type_size": 2},
            "float32tofloat32": {"h": 418.9789, "b": 6.39, "a": 7.61, "hl": 0, "data_type_size": 4},
            "uint8touint8": {"h": 418.9789, "b": 6.39, "a": 7.61, "hl": 0, "data_type_size": 1},
            "bfloat16tobfloat16": {"h": 418.9789, "b": 6.39, "a": 7.61, "hl": 0, "data_type_size": 2}
        }
    }
}
)";

const std::string *PerfParamTableV2::GetAscendCApiPerfTable() const {
  return &kParamV2Info;
}

PipeHeadPerfFunc PerfParamTableV2::GetPipeHeadPerfFunc(PipeType pipe_type) const {
  auto it = pipes_head_perf_.find(pipe_type);
  if (it == pipes_head_perf_.end()) {
    return &PerfParamTableV2::GetMTE2PipeHead;
  }
  return it->second;
}

}  // namespace att
