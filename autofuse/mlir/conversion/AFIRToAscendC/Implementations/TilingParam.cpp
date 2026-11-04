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
// index cast。**host ABI 逐字节不变**——字段名与 schema 都还是原来那一份，变的只是
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
    // **没有 `ascend.implementation` 属性时照样接。**
    //
    // 选择 pass 只 annotate 过它跑那一遍时图上的算子，而这一条可能是 conversion 阶段**新建**
    // 的：descriptor 里非字面量的 stride / extent 由 `materializeSizeAt` 物化，它把自由符号
    // 绑到 `getOrCreateTilingParam`，缺的时候就地建一条。缺的时候是真会发生的——
    // `AFIRMaterializeTilingParamsPass` 收的是 `size_var` 与 `afir.buffer_desc` 的符号，
    // descriptor 里的 stride / extent 不在其中。
    //
    // 放行不会选错：`afir.tiling_param` 全仓只有这一个 implementation（文件末尾那条注册是
    // 唯一一条），所以「没人选过」与「选的是我」在这里是同一件事。真有了第二个，这里要改成
    // 「没属性就去 registry 查这个 op 是不是只有一个候选」，而不是继续无条件放行。
    //
    // 少了这一条的后果与 `afir.symbolic_eval` 在 conversion 里没有 lowering 是同一类
    // （d0b1971f 修的是那一半）：新建的 op 没人 legalize，dialect conversion 把**整个**
    // pattern 回滚、报在根算子上，看起来像判据写错了。
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
