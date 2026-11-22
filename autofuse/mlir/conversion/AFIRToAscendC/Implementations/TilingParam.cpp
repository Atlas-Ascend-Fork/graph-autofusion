/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- TilingParam.cpp - afir.tiling_param -> AscendC TilingData member ---===//
//
// `afir.tiling_param "q0_size"` 降成 `tiling.q0_size`：一次 `emitasc.member` 加一次
// index cast。host ABI 逐字节不变——字段名与 schema 都还是原来那一份，变的只是
// 「谁在什么时候建这个访问」：此前是循环生成与资源物化各建一次，现在只在这里建。
//
// 名字在目标 schema 里不存在时由这里报错，而不是由 AFIR 层的 verifier：AFIR 只说
// 「我要这个名字的值」，一个名字合不合法取决于目标的 ABI，换个后端答案就不一样。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/TilingParam.h"

#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

struct TilingParamMapping {};

// 能不能降只取决于「本函数有没有 canonical 的 TilingData 实参」。字段在不在 schema 里
// 留给 pattern 报：那是一条要指名道姓说清缺了哪个字段的诊断，而 capability 查询只回答
// 支持与否，没有地方带上这个名字。
class TilingParamCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit TilingParamCapabilityProvider(std::shared_ptr<const TilingParamMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation, const ascend::instruction::QueryContext &) const override {
    if (!isa<TilingParamOp>(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget);
    }
    return CapabilitySupport::supported();
  }

 private:
  std::shared_ptr<const TilingParamMapping> mapping_;
};

class LowerTilingParamOp final : public OpConversionPattern<TilingParamOp> {
 public:
  LowerTilingParamOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                     std::shared_ptr<const TilingParamMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(TilingParamOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    // 实现选择只标注当时存在的算子。conversion 的 materializeSizeAt 仍可能为 descriptor
    // 中的非字面量 stride / extent 创建 tiling_param，因此新算子也需要在此处理。
    // 不能假设所有符号读取都已在选择阶段之前生成。
    if (op->hasAttr(ascend::instruction::kImplementationAttrName)) {
      auto selected = ascend::instruction::getSelectedImplementation(op);
      if (failed(selected) || *selected != key_) {
        return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
      }
    }
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func) {
      return rewriter.notifyMatchFailure(op, "a tiling parameter is only defined inside a kernel function");
    }
    FailureOr<Value> tiling = getCanonicalTilingArgument(func);
    if (failed(tiling)) {
      return failure();
    }
    TilingSchema schema = getTilingSchema(tiling->getType());
    Type fieldType = schema ? schema.getField(op.getSymbol()) : Type();
    if (!fieldType) {
      return op.emitOpError() << "TilingData is missing canonical size expression field '" << op.getSymbol() << "'";
    }
    if (!isa<IntegerType, IndexType>(fieldType)) {
      return op.emitOpError() << "TilingData field '" << op.getSymbol() << "' must have an integer type";
    }
    Value member = createTilingSchemaMember(rewriter, op.getLoc(), schema, *tiling, op.getSymbol());
    if (!member.getType().isIndex()) {
      member = arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), member);
    }
    rewriter.replaceOp(op, member);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const TilingParamMapping> mapping_;
};

}  // namespace

LogicalResult registerTilingParamImplementations(ImplementationRegistry &registry,
                                                 ascend::instruction::SelectionPolicyRegistry *) {
  return registry.registerImplementation<TilingParamOp, TilingParamMapping, TilingParamCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, {"tiling_param"}, TilingParamMapping{},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const TilingParamMapping> mapping) {
        patterns.add<LowerTilingParamOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
      });
}

}  // namespace mlir::ascend::kernel
