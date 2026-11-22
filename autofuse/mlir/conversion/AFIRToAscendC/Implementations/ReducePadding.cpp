/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/ReducePadding.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdint>

namespace mlir::ascend::kernel {
namespace {
// Keep nominal pitches independent of actual extents. Only padding is written;
// the source's valid elements may still have other fused consumers.
class PaddingEmitter {
 public:
  PaddingEmitter(OpBuilder &builder, Location location, Value tensor, Value identity, int64_t alignment)
      : b(builder), loc(location), dst(tensor), pad(identity), align(alignment) {}

  Value c(int64_t n) {
    auto found = indexConstants.find(n);
    if (found != indexConstants.end()) {
      return found->second;
    }
    // Constants must dominate both arms and all nested loops. Sharing them
    // also permits direct C++ emission without relying on aggregate CSE.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(pad.getDefiningOp());
    Value value = b.create<arith::ConstantIndexOp>(loc, n);
    indexConstants[n] = value;
    return value;
  }
  Value maskConstant(int64_t n) {
    auto found = maskConstants.find(n);
    if (found != maskConstants.end()) {
      return found->second;
    }
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(pad.getDefiningOp());
    Value value = b.create<arith::ConstantIntOp>(loc, n, 64);
    maskConstants[n] = value;
    return value;
  }
  Value add(Value x, Value y) {
    return b.create<arith::AddIOp>(loc, x, y);
  }
  Value sub(Value x, Value y) {
    return b.create<arith::SubIOp>(loc, x, y);
  }
  Value mul(Value x, Value y) {
    return b.create<arith::MulIOp>(loc, x, y);
  }
  Value div(Value x, Value y) {
    return b.create<arith::DivUIOp>(loc, x, y);
  }
  Value rem(Value x, Value y) {
    return b.create<arith::RemSIOp>(loc, x, y);
  }
  Value cmp(arith::CmpIPredicate p, Value x, Value y) {
    return b.create<arith::CmpIOp>(loc, p, x, y);
  }
  Value minimum(Value x, Value y) {
    return b.create<arith::SelectOp>(loc, cmp(arith::CmpIPredicate::ult, x, y), x, y);
  }
  Value view(Value offset) {
    return b.create<ascendc::LocalTensorSubIndexOp>(loc, dst.getType(), dst, offset);
  }
  void when(Value condition, function_ref<void()> body) {
    auto op = b.create<scf::IfOp>(loc, condition, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&op.getThenRegion().front());
    body();
  }
  void loop(Value count, Value step, function_ref<void(Value)> body) {
    auto op = b.create<scf::ForOp>(loc, c(0), count, step);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.getBody());
    body(op.getInductionVar());
  }
  Value maskCast(Value v) {
    return b.create<emitc::CastOp>(loc, b.getIntegerType(64, false), v);
  }
  void masked(Value offset, Value low, Value high, Value repeats, Value blockStride, Value repeatStride) {
    b.create<ascendc::DuplicateL1Op>(loc, view(offset), pad, ValueRange{maskCast(low), maskCast(high)}, repeats,
                                     blockStride, repeatStride, b.getUnitAttr());
  }
  // One partial 32B block per row. Pack eight rows per repeat when both
  // encoded strides fit; otherwise use bounded repeats, or single-row calls.
  void partialRows(Value rows, Value pitch, Value actual) {
    Value tail = rem(actual, c(align));
    when(cmp(arith::CmpIPredicate::ne, tail, c(0)), [&] {
      // The C++ backend represents index as uint32_t. Masks need an
      // explicit 64-bit type before shifting, not just a cast at the API call.
      Value tail64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), tail);
      Value bits = sub(maskConstant(1 << align), b.create<arith::ShLIOp>(loc, maskConstant(1), tail64));
      Value block = sub(actual, tail);
      Value stride = div(pitch, c(align));
      Value packedStride = mul(stride, c(8));
      auto branch = b.create<scf::IfOp>(loc, cmp(arith::CmpIPredicate::ule, packedStride, c(255)), true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        Value packedMask = bits;
        for (int64_t shift = align; shift < 64; shift += align) {
          packedMask = b.create<arith::OrIOp>(loc, packedMask, b.create<arith::ShLIOp>(loc, bits, maskConstant(shift)));
        }
        Value groups = div(rows, c(8));
        loop(groups, c(255), [&](Value group) {
          masked(add(mul(mul(group, c(8)), pitch), block), packedMask, align == 16 ? packedMask : c(0),
                 minimum(sub(groups, group), c(255)), stride, packedStride);
        });
        Value remainder = rem(rows, c(8));
        when(cmp(arith::CmpIPredicate::ne, remainder, c(0)),
             [&] { masked(add(mul(mul(groups, c(8)), pitch), block), bits, c(0), remainder, c(1), stride); });
      }
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getElseRegion().front());
        Value fits = cmp(arith::CmpIPredicate::ule, stride, c(255));
        Value batch = b.create<arith::SelectOp>(loc, fits, c(255), c(1));
        Value encodedStride = b.create<arith::SelectOp>(loc, fits, stride, c(1));
        loop(rows, batch, [&](Value row) {
          masked(add(mul(row, pitch), block), bits, c(0), minimum(sub(rows, row), batch), c(1), encodedStride);
        });
      }
    });
  }
  // Both boundaries are block aligned. Batch short segments across rows when
  // the repeat stride is encodable; long segments use the native count API.
  void fullRows(Value rows, Value pitch, Value start) {
    when(cmp(arith::CmpIPredicate::ult, start, pitch), [&] {
      Value count = sub(pitch, start);
      Value stride = div(pitch, c(align));
      Value segments = div(add(count, c(8 * align - 1)), c(8 * align));
      Value batch = b.create<arith::AndIOp>(loc, cmp(arith::CmpIPredicate::ult, segments, rows),
                                            cmp(arith::CmpIPredicate::ule, stride, c(255)));
      auto branch = b.create<scf::IfOp>(loc, batch, true);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        loop(count, c(8 * align), [&](Value segment) {
          Value mask = minimum(sub(count, segment), c(8 * align));
          loop(rows, c(255), [&](Value row) {
            b.create<ascendc::DuplicateL0Op>(loc, view(add(add(mul(row, pitch), start), segment)), pad, mask,
                                             minimum(sub(rows, row), c(255)), c(1), stride, b.getUnitAttr());
          });
        });
      }
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&branch.getElseRegion().front());
        loop(rows, c(1),
             [&](Value row) { b.create<ascendc::DuplicateL2Op>(loc, view(add(mul(row, pitch), start)), pad, count); });
      }
    });
  }

 private:
  OpBuilder &b;
  Location loc;
  Value dst;
  Value pad;
  int64_t align;
  llvm::DenseMap<int64_t, Value> indexConstants;
  llvm::DenseMap<int64_t, Value> maskConstants;
};

class ReducePaddingPattern : public OpConversionPattern<ascendc::ReduceInitOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ascendc::ReduceInitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type element = cast<ascendc::LocalTensorType>(adaptor.getDst().getType()).getElementType();
    if (!element.isF16() && !element.isF32() && !element.isSignlessInteger(32)) {
      return rewriter.notifyMatchFailure(op, "reduce padding requires f16, f32 or i32");
    }
    Location loc = op.getLoc();
    bool minimum = op.getReduceType() == ascendc::ReduceInitType::Min;
    bool maximum = op.getReduceType() == ascendc::ReduceInitType::Max;
    int64_t unit =
        op.getReduceType() == ascendc::ReduceInitType::Prod || op.getReduceType() == ascendc::ReduceInitType::All ? 1
                                                                                                                  : 0;
    Value identity;
    if (auto f = dyn_cast<FloatType>(element)) {
      if (element.isF16() && (minimum || maximum)) {
        // Use the SDK's typed half constant. A folded f16 arith constant is
        // printed as a float infinity and can saturate during CPU conversion.
        auto literal =
            emitc::OpaqueAttr::get(rewriter.getContext(), maximum ? "AscendC::NumericLimits<half>::NegativeInfinity()"
                                                                  : "AscendC::NumericLimits<half>::Infinity()");
        identity = rewriter.create<emitc::ConstantOp>(loc, element, literal);
      } else {
        auto attr = minimum || maximum ? rewriter.getFloatAttr(f, llvm::APFloat::getInf(f.getFloatSemantics(), maximum))
                                       : rewriter.getFloatAttr(f, static_cast<double>(unit));
        identity = rewriter.create<arith::ConstantOp>(loc, attr);
      }
    } else {
      identity = rewriter.create<arith::ConstantIntOp>(loc, minimum ? INT32_MAX : maximum ? INT32_MIN : unit, 32);
    }
    auto index = [&](Value v) -> Value {
      if (v.getType().isIndex()) {
        return v;
      }
      return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), v);
    };
    Value a = index(adaptor.getDimA()), r = index(adaptor.getDimR());
    Value current = index(adaptor.getDimRCurrent()), inner = index(adaptor.getInnerR());
    Value stride = index(adaptor.getInnerRStride());
    int64_t alignment = 32 / (element.getIntOrFloatBitWidth() / 8);
    PaddingEmitter emit(rewriter, loc, adaptor.getDst(), identity, alignment);
    if (op.getIsTailLast()) {
      emit.partialRows(a, r, inner);
    } else {
      Value rows = emit.mul(a, emit.div(r, stride));
      emit.partialRows(rows, stride, inner);
      Value aligned = emit.mul(emit.div(emit.add(inner, emit.c(alignment - 1)), emit.c(alignment)), emit.c(alignment));
      emit.fullRows(rows, stride, aligned);
    }
    emit.fullRows(a, r, current);
    rewriter.eraseOp(op);
    return success();
  }
};
}  // namespace
void populateReducePaddingPatterns(RewritePatternSet &patterns) {
  patterns.add<ReducePaddingPattern>(patterns.getContext());
}
}  // namespace mlir::ascend::kernel
