/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/CastConversion.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseLocalView.h"
#include "conversion/AFIRToAscendC/VectorizedLoops.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "target/Ascend/Capability/ImplementationSelection.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::ascend::kernel {
namespace {
using elementwise::ApiFamily;
using instruction::ImplementationKey;

constexpr unsigned kPackedInt4BitWidth = 4;
constexpr unsigned kInt8BitWidth = 8;
constexpr unsigned kInt64BitWidth = 64;
constexpr unsigned kDecimalRadix = 10;
constexpr unsigned kCastTensorCount = 2;

struct CastWindow {
  Value source;
  Value destination;
  Value count;
  Value rows;
  Value inputStride;
  Value outputStride;
};

// Only direct mask-mode conversions may batch without enlarging scratch storage.
// The existing Cast pair check already establishes support for the selected API.
bool supportsStridedCast(ApiFamily family, Type input, Type output, const afir::VectorPlan &plan) {
  if (family == ApiFamily::RegBased) {
    return true;
  }
  if (auto integer = dyn_cast<IntegerType>(input)) {
    if (integer.getWidth() == kInt8BitWidth && (!integer.isUnsigned() || !output.isF16())) {
      return false;
    }
    if (auto result = dyn_cast<IntegerType>(output)) {
      if (integer.getWidth() == result.getWidth()) {
        return false;
      }
    }
  }
  // Packed sub-byte rows and two-step casts keep the established scalar path.
  if (input.isInteger(kPackedInt4BitWidth) || output.isInteger(kPackedInt4BitWidth) ||
      (input.isF16() && output.isInteger(kInt64BitWidth))) {
    return false;
  }
  // The legacy equal-stride fast path also writes padding. Only use its batched
  // overload when constant row strides prove that this fast path cannot run.
  const auto &strides = plan.segments.back().strides;
  int64_t inputStride = 0;
  int64_t outputStride = 0;
  return strides.size() == kCastTensorCount &&
         !strides.front().getValue().getValue().getAsInteger(kDecimalRadix, inputStride) &&
         !strides.back().getValue().getValue().getAsInteger(kDecimalRadix, outputStride) && inputStride != outputStride;
}

LogicalResult materializeCastRowStrides(Operation *op, ConversionPatternRewriter &rewriter,
                                        const afir::VectorPlan &plan, CastWindow &tiles) {
  for (auto [tensor, stride] : llvm::zip(plan.tensors, plan.segments.back().strides)) {
    auto value = afir::materializeStrideValue(op, rewriter, stride);
    if (failed(value)) {
      return failure();
    }
    (tensor.side == afir::VectorTensor::Side::Destination ? tiles.outputStride : tiles.inputStride) = *value;
  }
  return success();
}

bool selectCastRowWindow(ApiFamily family, const CastWindow &tiles, elementwise::ElementwiseLocalView &view) {
  auto inputType = cast<ascendc::LocalTensorType>(tiles.source.getType()).getElementType();
  auto outputType = cast<ascendc::LocalTensorType>(tiles.destination.getType()).getElementType();
  const bool batchRows = view.apiWindowSize == 0 && !view.vectorPlan->segments.empty() &&
                         supportsStridedCast(family, inputType, outputType, *view.vectorPlan);
  if (batchRows) {
    // Preserve the last effective axis as rows, with a virtual unit inner axis.
    // Local-view validation has checked every row start for alignment and aliasing.
    view.apiWindowSize = 1;
  }
  return batchRows;
}

FailureOr<CastWindow> materializeCastWindow(afir::CastOp op, afir::CastOp::Adaptor adaptor,
                                            ConversionPatternRewriter &rewriter, bool compactTail, ApiFamily family) {
  CastWindow tiles;
  tiles.source = adaptor.getInput();
  tiles.destination = adaptor.getDestination();
  elementwise::ElementwiseLocalView view;
  if (elementwise::prepareElementwiseLocalView(op, {}, compactTail, view, /*allowStridedElements=*/true) !=
      elementwise::LocalViewCompatibility::Compatible) {
    return failure();
  }
  if (!view.vectorPlan) {
    tiles.count = rewriter.create<ascendc::LocalTensorGetSizeOp>(op.getLoc(), rewriter.getI32Type(), tiles.destination);
    return tiles;
  }
  const bool batchRows = selectCastRowWindow(family, tiles, view);
  const auto &segments = view.vectorPlan;
  FailureOr<SmallVector<Value>> ordered = orderTilesForPlan(*segments, {tiles.source}, tiles.destination);
  if (failed(ordered)) {
    return failure();
  }
  FailureOr<VectorizedWindow> window = materializeVectorizedSegments(op, rewriter, *segments, view.extentSource,
                                                                     *ordered, view.apiWindowSize, view.physicalSpan);
  if (failed(window) || window->tiles.size() != segments->tensors.size() || window->extents.size() != 1) {
    return failure();
  }
  for (auto [tensor, tile] : llvm::zip(segments->tensors, window->tiles)) {
    (tensor.side == afir::VectorTensor::Side::Destination ? tiles.destination : tiles.source) = tile;
  }
  tiles.count = window->extents.front();
  if (batchRows) {
    tiles.rows = tiles.count;
    tiles.count = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    if (failed(materializeCastRowStrides(op, rewriter, *segments, tiles))) {
      return failure();
    }
  }
  return tiles;
}

class CastConversionPattern final : public OpConversionPattern<afir::CastOp> {
 public:
  CastConversionPattern(TypeConverter &converter, MLIRContext *context, const ImplementationKey &key, ApiFamily family,
                        const ExecutionSpec &executionSpec, bool compactTail)
      : OpConversionPattern(converter, context),
        key_(key),
        family_(family),
        executionSpec_(executionSpec),
        compactTail_(compactTail) {}

  LogicalResult matchAndRewrite(afir::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto selected = instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination() || (family_ == ApiFamily::MemBased && !adaptor.getTmpBuffer())) {
      return rewriter.notifyMatchFailure(op, "Cast requires destination and API-specific scratch operands");
    }
    if (elementwise::checkElementwiseLocalViews(op, nullptr, /*allowStridedElements=*/true) !=
        elementwise::LocalViewCompatibility::Compatible) {
      return rewriter.notifyMatchFailure(op, "incompatible Cast local views");
    }
    auto window = materializeCastWindow(op, adaptor, rewriter, compactTail_, family_);
    if (failed(window)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported Cast window");
    }
    Operation *target = createCast(op, adaptor, rewriter, *window);
    if (failed(setExecutionSpec(target, executionSpec_))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, window->destination);
    return success();
  }

 private:
  Operation *createCast(afir::CastOp op, afir::CastOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
                        const CastWindow &tiles) const {
    if (family_ == ApiFamily::MemBased) {
      if (tiles.rows) {
        return rewriter.create<ascendc::CastExtendStridedOp>(op.getLoc(), tiles.destination, tiles.source,
                                                             adaptor.getTmpBuffer(), tiles.rows, tiles.count,
                                                             tiles.inputStride, tiles.outputStride);
      }
      return rewriter.create<ascendc::CastExtendContiguousOp>(op.getLoc(), tiles.destination, tiles.source,
                                                              adaptor.getTmpBuffer(), tiles.count);
    } else {
      if (tiles.rows) {
        return rewriter.create<ascendc::CastExtendStridedRegBaseOp>(op.getLoc(), tiles.destination, tiles.source,
                                                                    tiles.rows, tiles.count, tiles.inputStride,
                                                                    tiles.outputStride);
      }
      return rewriter.create<ascendc::CastExtendContiguousRegBaseOp>(op.getLoc(), tiles.destination, tiles.source,
                                                                     tiles.count);
    }
  }

  ImplementationKey key_;
  ApiFamily family_;
  ExecutionSpec executionSpec_;
  bool compactTail_;
};
}  // namespace

void populateCastConversionPattern(RewritePatternSet &patterns, TypeConverter &converter,
                                   const instruction::ImplementationKey &key, elementwise::ApiFamily family,
                                   const ExecutionSpec &executionSpec, bool compactTail) {
  patterns.add<CastConversionPattern>(converter, patterns.getContext(), key, family, executionSpec, compactTail);
}
}  // namespace mlir::ascend::kernel
