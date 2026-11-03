/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __V1_ASCIR_CODEGEN_IMPL__
#define __V1_ASCIR_CODEGEN_IMPL__

#include "ascendc_ir.h"
#include "graph/ascendc_ir/ascir_registry.h"
#include "../reg_func/default_reg_func.h"
#include "ascir_common.h"

namespace af {
namespace ascir {

/*********************************************************************************/
class DataAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Data";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class ScalarAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Scalar";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class IndexExprAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "IndexExpr";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class OutputAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Output";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class WorkspaceAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Workspace";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class LoadAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "LoadApiCall";
  }
  std::string GetApiName() const override {
    return "Load";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"datacopy.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class BroadcastAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcBroadCastTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "BroadcastApiCall";
  }
  std::string GetApiName() const override {
    return "Broadcast";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"duplicate.h", "broadcast.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "adv_api/pad/broadcast.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_transpose_intf.h",
    };
  }
};

class NopAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "ApiCall";
  }
  std::string GetApiName() const override {
    return "Nop";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
};

class CastAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcCastTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CastApiCall";
  }
  std::string GetApiName() const override {
    return "Cast";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"cast.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] has inconsistent shape",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AbsAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAbsTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "AbsExtend";
  }
  bool IsInplaceSupported(const AscNode &abs_node) const override {
    (void)abs_node;
    return true;
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"abs.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ExpAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Exp";
  }
  bool IsInplaceSupported(const AscNode &exp_node) const override {
    (void)exp_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class RemovePadAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "RemovePadApiCall";
  }
  std::string GetApiName() const override {
    return "RemovePad";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"removepad.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class PadAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcPadTmpSize(node);
  }

  std::string GetApiTilingTypeName() const override {
    return "PadTiling";
  }

  std::string GetApiCallName() const override {
    return "PadApiCall";
  }
  std::string GetApiName() const override {
    return "Pad";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/pad/pad.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class LnAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Ln";
  }
  bool IsInplaceSupported(const AscNode &ln_node) const override {
    (void)ln_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SqrtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Sqrt";
  }
  bool IsInplaceSupported(const AscNode &sqrt_node) const override {
    (void)sqrt_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class RsqrtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcRsqrtTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "RsqrtApiCall";
  }
  std::string GetApiName() const override {
    return "RsqrtExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"rsqrt.h"};
  }
  bool IsInplaceSupported(const AscNode &rsqrt_node) const override {
    (void)rsqrt_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class NegAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "NegExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"neg.h"};
  }
  bool IsInplaceSupported(const AscNode &neg_node) const override {
    (void)neg_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ReluAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "UnaryApiCall";
  }
  std::string GetApiName() const override {
    return "Relu";
  }
  bool IsInplaceSupported(const AscNode &relu_node) const override {
    (void)relu_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class ReciprocalAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcDefaultTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "ReciprocalExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reciprocal.h"};
  }
  bool IsInplaceSupported(const AscNode &reciprocal_node) const override {
    (void)reciprocal_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SignAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSignTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "SignExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"cast.h", "sign.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "adv_api/math/sign.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class IsnanAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcIsnanTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCall";
  }
  std::string GetApiName() const override {
    return "IsnanExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"isnan.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class IsFiniteAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcIsFiniteTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCall";
  }
  std::string GetApiName() const override {
    return "IsFiniteExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"isfinite.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h", "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class IsInfAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcIsInfTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryBitWidthChangeApiCall";
  }
  std::string GetApiName() const override {
    return "IsInfExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"is_inf.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h", "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_unary_intf.h",     "basic_api/kernel_operator_vec_cmpsel_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class LogicalNotAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcLogicalNotTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "LogicalNotApiCall";
  }
  std::string GetApiName() const override {
    return "LogicalNot";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"logical_not.h"};
  }
  bool IsInplaceSupported(const AscNode &not_node) const override {
    (void)not_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MaxAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ReduceMax";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class ArgMaxAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ArgMax";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h",    "argmax.h", "compare.h",
            "compare_v2.h",  "duplicate.h", "where.h",  "argmax_with_value.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",   "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",      "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h", "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
};

class ArgMaxMultiRPhase1AscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ArgMaxMultiRPhase1";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h",    "argmax.h", "compare.h",
            "compare_v2.h",  "duplicate.h", "where.h",  "argmax_with_value.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
};

class ArgMaxMultiRPhase2AscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ArgMaxMultiRPhase2";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h",    "argmax.h", "compare.h",
            "compare_v2.h",  "duplicate.h", "where.h",  "argmax_with_value.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",   "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
};

class SumAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ReduceSum";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class MinAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ReduceMin";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class MeanAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ReduceMean";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class ProdAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ReduceProd";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class AnyAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ReduceAny";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class AllAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcReduceTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ReduceApiCall";
  }
  std::string GetApiName() const override {
    return "ReduceAll";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"reduce_init.h", "reduce.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "adv_api/reduce/reduce.h",
        "basic_api/kernel_operator_vec_brcb_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class GeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcGeTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "GE";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);  // 不支持调换
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",         "basic_api/kernel_operator_vec_cmpsel_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class EqAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcEqTmpSize(node);
  }

  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "EQ";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"compare.h", "compare_v2.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",         "basic_api/kernel_operator_vec_cmpsel_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class NeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcNeTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "NE";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",         "basic_api/kernel_operator_vec_cmpsel_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class GtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcGtTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "GT";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);  // 不支持调换
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",         "basic_api/kernel_operator_vec_cmpsel_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcLeTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "LE";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);  // 不支持调换
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",         "basic_api/kernel_operator_vec_cmpsel_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class LtAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcLtTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "CompareApiCall";
  }
  std::string GetApiName() const override {
    return "LT";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"compare.h", "compare_v2.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);  // 不支持调换
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",     "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",         "basic_api/kernel_operator_vec_cmpsel_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_reduce_intf.h",        "basic_api/kernel_operator_vec_unary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SigmoidAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSigmoidTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "SigmoidExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"sigmoid.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_unary_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class Ub2ubAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "Ub2ubApiCall";
  }
  std::string GetApiName() const override {
    return "DataCopy";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

/**************************************************************/
class DivAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcDivTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Div";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_div.h"};
  }

  bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list;  // 支持任意输入是scalar
    return true;
  }
  bool IsInplaceSupported(const AscNode &div_node) const override {
    (void)div_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SubAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSubTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Sub";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_sub.h"};
  }

  bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list;  // 支持任意输入是scalar
    return true;
  }
  bool IsInplaceSupported(const AscNode &sub_node) const override {
    (void)sub_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AddAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Add";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_add.h"};
  }
  bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const AscNode &add_node) const override {
    (void)add_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MulAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "Mul";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_mul.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }
  bool IsInplaceSupported(const AscNode &mul_node) const override {
    (void)mul_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class TrueDivAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcTrueDivTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "TrueDivApiCall";
  }
  std::string GetApiName() const override {
    return "TrueDivExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_div.h", "true_div.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list;  // 支持任意输入是scalar
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class RemainderAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcRemainderTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "RemainderExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"remainder.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MinimumAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "AscendC::Min";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_minimum.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const AscNode &minimum_node) const override {
    (void)minimum_node;
    return true;
  }
  bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MaximumAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "BinaryApiCall";
  }
  std::string GetApiName() const override {
    return "AscendC::Max";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"scalar_maximum.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsBrcInlineSupported(const AscNode &node) const override {
    return !IsAnyInputNodeScalar(node);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const AscNode &maximum_node) const override {
    (void)maximum_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {true, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/

class WhereAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcWhereTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "WhereApiCall";
  }
  std::string GetApiName() const override {
    return "Where";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"duplicate.h", "where.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return is_scalar_list[0] == false;  // 除第1个外都支持Scalar
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",     "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_cmpsel_intf.h",        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_transpose_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1, 2}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class SelectAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcSelectTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "WhereApiCall";
  }
  std::string GetApiName() const override {
    return "Where";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"duplicate.h", "where.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return is_scalar_list[0] == false;  // 除第1个外都支持Scalar
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_duplicate_intf.h",     "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_cmpsel_intf.h",        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h", "basic_api/kernel_operator_vec_transpose_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeFirstInputScalar(node), "Node %s[%s] does not support first input scalar", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {1, 2}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

// MaskedFill reuses Select codegen. IsScalarInputSupported checks original input order
// (before MaskedFillInputReorderPass): inputs[1]=mask cannot be scalar.
class MaskedFillAscIrCodegenImpl : public SelectAscIrCodegenImpl {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcMaskedFillTmpSize(node);
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 3UL);
    return !is_scalar_list[1];
  }
};
/*********************************************************************************/
class LeakyReluAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "LeakyReluApiCall";
  }
  std::string GetApiName() const override {
    return "LeakyRelu";
  }
  bool IsInplaceSupported(const AscNode &leaky_relu_node) const override {
    (void)leaky_relu_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class ClipByValueAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcClipByValueTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ClipByValueApiCall";
  }
  std::string GetApiName() const override {
    return "ClipByValue";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"clipbyvalue.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    (void)is_scalar_list;  // 支持任意输入是scalar
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "adv_api/math/clamp.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {0, 1, 2}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class StoreAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "StoreApiCall";
  }
  std::string GetApiName() const override {
    return "Store";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"datacopy.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class ConcatAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcConcatTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "ConcatApiCall";
  }
  std::string GetApiName() const override {
    return "Concat";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"concat.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",
        "basic_api/kernel_operator_vec_gather_mask_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class GatherAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcGatherTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "GatherApiCall";
  }
  std::string GetApiName() const override {
    return "GatherExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"gather.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_gather_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class TransposeAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcDefaultTmpSize(node);
  }

  std::string GetApiTilingTypeName() const override {
    return "ConfusionTransposeTiling";
  }
  std::string GetApiCallName() const override {
    return "TransposeApiCall";
  }
  std::string GetApiName() const override {
    return "Transpose";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"transpose_base_type.h", "transpose.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_transpose_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/

class ErfAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcErfTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "Erf";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/erf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class TanhAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcTanhTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "Tanh";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/tanh.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class GeluAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return GetInputDataSizeTmpBuffer(node);
  }

  std::string GetApiCallName() const override {
    return "UnaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "Gelu";
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/activation/gelu.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/
class LogicalOrAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcLogicalOrTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "LogicalOr";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"logical.h"};
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsInplaceSupported(const AscNode &or_node) const override {
    (void)or_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "utils/std/type_traits.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_scalar_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
};

class LogicalAndAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcLogicalAndTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "LogicalAnd";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"logical.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    return OnlySecondInputSupportScalar(is_scalar_list);
  }
  bool IsScalarInputSupportedIfExchangeInputs(const std::vector<bool> &is_scalar_list) const override {
    GE_ASSERT_EQ(is_scalar_list.size(), 2UL);
    return OnlySecondInputSupportScalar({is_scalar_list[1], is_scalar_list[0]});
  }
  bool IsInplaceSupported(const AscNode &and_node) const override {
    (void)and_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "utils/std/type_traits.h",
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_scalar_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
    };
  }
};

class BitwiseAndAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcDefaultTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "BitwiseAndExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"bitwise_and.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class FloorDivAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return GetInputDataSizeTmpBuffer(node);
  }

  std::string GetApiCallName() const override {
    return "BinaryApiTmpCall";
  }
  std::string GetApiName() const override {
    return "FloorDivExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"floor_div.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_binary_intf.h",
        "adv_api/math/floor.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};
/*********************************************************************************/

class PowAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcPowTmpSize(node);
  }

  std::string GetApiCallName() const override {
    return "PowApiCall";
  }
  std::string GetApiName() const override {
    return "Pow";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"pow.h"};
  }
  bool IsScalarInputSupported(const std::vector<bool> &is_scalar_list) const override {
    // 不支持全scalar输入
    return !std::all_of(is_scalar_list.begin(), is_scalar_list.end(), [](bool i) { return i; });
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "adv_api/math/power.h",
        "basic_api/kernel_operator_vec_duplicate_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node, {false, {0, 1}}),
                      "Node %s[%s] check shape consistency failed", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class AxpyAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::vector<std::unique_ptr<TmpBufDesc>> CalcTmpBufSize(const AscNode &node) override {
    return CalcAxpyTmpSize(node);
  }
  std::string GetApiCallName() const override {
    return "AxpyApiCall";
  }
  std::string GetApiName() const override {
    return "AxpyExtend";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"axpy.h"};
  }
  bool IsInplaceSupported(const AscNode &axpy_node) const override {
    (void)axpy_node;
    return true;
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "basic_api/kernel_operator_vec_vconv_intf.h",
        "basic_api/kernel_operator_vec_binary_scalar_intf.h",
        "basic_api/kernel_operator_vec_binary_intf.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    GE_ASSERT_SUCCESS(ValidateShapeConsistencyWithSingleOutput(node), "Node %s[%s] check shape consistency failed",
                      node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class MatMulAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "MatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles(bool is_dynamic) const override {
    if (is_dynamic) {
      return {"mat_mul_tiling_key_dynamic.h", "matmul_include_headers.h", "mat_mul_pingpong_basic_cmct_dynamic.h",
              "matmul_dynamic.h"};
    } else {
      return {"mat_mul_tiling_key.h", "matmul_include_headers.h", "mat_mul_pingpong_basic_cmct.h", "matmul.h"};
    }
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "utils/std/algorithm.h",
        "basic_api/kernel_operator_common_intf.h",
        "basic_api/kernel_operator_set_atomic_intf.h",
        "adv_api/matmul/matmul.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class BatchMatMulAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "MatmulApiCall";
  }
  std::string GetApiName() const override {
    return "BatchMatMul";
  }
  std::vector<std::string> LoadApiHeaderFiles(bool is_dynamic) const override {
    if (is_dynamic) {
      return {"batch_mat_mul_v3_tiling_key_dynamic.h", "batch_matmul_include_headers.h",
              "mat_mul_pingpong_basic_cmct_dynamic.h", "batch_matmul_dynamic.h"};
    } else {
      return {"batch_mat_mul_v3_tiling_key.h", "batch_matmul_include_headers.h", "mat_mul_pingpong_basic_cmct.h",
              "batch_matmul.h"};
    }
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {
        "utils/std/algorithm.h",
        "basic_api/kernel_operator_common_intf.h",
        "basic_api/kernel_operator_set_atomic_intf.h",
        "adv_api/matmul/matmul.h",
    };
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    GE_ASSERT_TRUE(!IsNodeHasScalarInput(node), "Node %s[%s] does not support scalar input", node.GetTypePtr(),
                   node.GetNamePtr());
    return true;
  }
};

class Conv2DAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "Conv2DApiCall";
  }
  std::string GetApiName() const override {
    return "Conv2D";
  }
  std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
    return {"conv2d_include_headers.h", "conv2d_v2_tilingkey_cv.h", "conv_pingpong_basic_atcos.h", "conv2d.h"};
  }
  std::vector<std::string> IncludeApiHeaderFiles() const override {
    return {"basic_api/kernel_operator_common_intf.h"};
  }
  [[nodiscard]] bool IsNodeValid(const AscNode &node) const override {
    std::string node_type = node.GetType();
    const auto &inputs = node.GetInDataNodes();
    GE_ASSERT_TRUE(!(inputs.size() > 0 && inputs.at(0)->GetType() == "Scalar"),
                   "Node %s[%s] does not support scalar input at index 0", node.GetTypePtr(), node.GetNamePtr());
    GE_ASSERT_TRUE(!(inputs.size() > 1 && inputs.at(1)->GetType() == "Scalar"),
                   "Node %s[%s] does not support scalar input at index 1", node.GetTypePtr(), node.GetNamePtr());
    return true;
  }
};

class UnsupportedAscIrCodegenImpl : public AscIrCodegen {
 public:
  std::string GetApiCallName() const override {
    return "";
  }
  std::string GetApiName() const override {
    return "Unsupported";
  }
};
}  // namespace ascir
}  // namespace af

#endif  //__ASCIR_CODEGEN_IMPL__
