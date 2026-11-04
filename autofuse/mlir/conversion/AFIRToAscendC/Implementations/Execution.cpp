/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/Execution.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
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

struct BlockIdMapping {
  uint64_t dimension = 0;
};

class BlockIdCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit BlockIdCapabilityProvider(std::shared_ptr<const BlockIdMapping> mapping) : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation, const ascend::instruction::QueryContext &) const override {
    auto op = dyn_cast<BlockIdOp>(operation);
    // GetBlockIdx is a one-dimensional logical block query on AscendC targets;
    // it does not require tensor descriptors or per-chip instruction features.
    if (!op || op.getDimension() != mapping_->dimension) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget);
    }
    return CapabilitySupport::supported();
  }

 private:
  std::shared_ptr<const BlockIdMapping> mapping_;
};

class LowerBlockIdOp final : public OpConversionPattern<BlockIdOp> {
 public:
  LowerBlockIdOp(TypeConverter &converter, MLIRContext *context, ImplementationKey key,
                 std::shared_ptr<const BlockIdMapping> mapping)
      : OpConversionPattern(converter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(BlockIdOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
    auto selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (op.getDimension() != mapping_->dimension) {
      return rewriter.notifyMatchFailure(op, "only block dimension 0 is supported by AscendC lowering");
    }
    rewriter.replaceOpWithNewOp<ascendc::GetBlockIdxOp>(op, rewriter.getIndexType());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BlockIdMapping> mapping_;
};

}  // namespace

LogicalResult registerExecutionImplementations(ImplementationRegistry &registry,
                                               ascend::instruction::SelectionPolicyRegistry *) {
  return registry.registerImplementation<BlockIdOp, BlockIdMapping, BlockIdCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, {"block_id"}, BlockIdMapping{},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BlockIdMapping> mapping) {
        patterns.add<LowerBlockIdOp>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
      });
}

}  // namespace mlir::ascend::kernel
