/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ScalarLiteral.cpp - afir.scalar -> backend constant -----------------===//
//
// `afir.scalar` 是标量字面量在 AFIR 层的表达：带符号性的整数（ui16 / ui32）
// `arith.constant` 建不出来，只能由方言自己的 op 承载。到这里已经离开 AFIR 层，用哪种
// 后端常量不再受完备性判据 C3 约束——带符号性的整数交给 `emitc.constant`，其余交给
// `arith.constant`，打印机把两者都印成 C++ 字面量。
//
// 只收 buffer 形态（结果是单个值）。tensor 形态属于逻辑形态的算子，不进这条下降流水线：
// 它今天没有生产者（bridge 只建单个值的那一种），下面两处 `isa<ShapedType>` 守卫只有手写的
// IR 才走得到——留着是沿用「逻辑形态不进这条流水线」的既有契约，不是为了新接一种输入。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/ScalarLiteral.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {

using ascend::instruction::CapabilityReason;
using ascend::instruction::CapabilitySupport;
using ascend::instruction::ImplementationKey;
using ascend::instruction::ImplementationRegistry;
using ascend::instruction::PatternBuildContext;

struct ScalarLiteralMapping {};

// 字面量不读内存，也不依赖任何目标特征。
class ScalarLiteralCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit ScalarLiteralCapabilityProvider(std::shared_ptr<const ScalarLiteralMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation, const ascend::instruction::QueryContext &) const override {
    if (!isa<ScalarOp>(operation) || isa<ShapedType>(operation->getResult(0).getType())) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    return CapabilitySupport::supported();
  }

 private:
  std::shared_ptr<const ScalarLiteralMapping> mapping_;
};

class LowerScalarLiteralOp final : public OpConversionPattern<ScalarOp> {
 public:
  LowerScalarLiteralOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                       std::shared_ptr<const ScalarLiteralMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(ScalarOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (isa<ShapedType>(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "only the buffer form of a scalar literal is lowered");
    }
    TypedAttr value = cast<TypedAttr>(op.getValue());
    Type type = value.getType();
    // `arith.constant` 只收 signless 整数。
    const auto integerType = dyn_cast<IntegerType>(type);
    if (integerType && !integerType.isSignless()) {
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, type, value);
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, type, value);
    }
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const ScalarLiteralMapping> mapping_;
};

}  // namespace

LogicalResult registerScalarLiteralImplementations(ImplementationRegistry &registry,
                                                   ascend::instruction::SelectionPolicyRegistry *) {
  return registry.registerImplementation<ScalarOp, ScalarLiteralMapping, ScalarLiteralCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, {"scalar.literal"}, ScalarLiteralMapping{},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const ScalarLiteralMapping> mapping) {
        patterns.add<LowerScalarLiteralOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
      });
}

}  // namespace mlir::ascend::kernel
