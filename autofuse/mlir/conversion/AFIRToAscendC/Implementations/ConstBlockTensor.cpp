/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/ConstBlockTensor.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ascend::kernel::elementwise {
namespace {

// 一个 block 的字节数。包装读这块张量时把 `src0BlkStride` / `src0RepStride` 都设成 0,
// 也就是每个 repeat 都回到同一个 block 的开头——所以一个 block 就够，与 tile 多大无关。
// legacy 这里写的是 `ONE_BLK_SIZE`，同一个数。
constexpr int64_t kBlockBytes = 32;

// 已经造过的那一块。查的是 `ascendc.tbuf.get_tensor` 上的认领标记，不是别的：认领标记
// 的取值是带类型的常量本身，一条属性同时说清「值是几」与「什么 dtype」。
Value findExistingConstBlock(func::FuncOp func, TypedAttr wanted) {
  Value found;
  func.walk([&](ascendc::TBufGetTensorOp get) {
    if (get->getAttr(kConstBlockAttrName) == wanted) {
      found = get.getTensor();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// 入口块里 `MemoryAllocate` 建的那个 `ascendc.pipe`。只喂 `!ascendc.local_tensor` 参数的
// lit 用例里一个资源都没有，于是也没有 pipe——那时候返回空，调用方自己建一个。
Value findPipe(func::FuncOp func) {
  for (Operation &operation : func.front()) {
    if (auto pipe = dyn_cast<ascendc::PipeOp>(operation)) {
      return pipe.getPipe();
    }
  }
  return {};
}

}  // namespace

FailureOr<Value> getOrCreateConstBlockTensor(ConversionPatternRewriter &rewriter, Operation *op, Type elementType,
                                             const llvm::APFloat &value) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || func.getBody().empty()) {
    return op->emitOpError("a pre-filled constant block requires an enclosing func.func");
  }
  auto floatType = dyn_cast<FloatType>(elementType);
  if (!floatType) {
    // 今天两块常量都是浮点（`RsqrtExtend` 的 float、`LogicalNot` 的 half）。整数那一支
    // 一个实例都没有，先让它报出来，别先摆一条没人走的分支在这里。
    return op->emitOpError("a pre-filled constant block currently requires a floating-point element type");
  }
  llvm::APFloat converted = value;
  bool losesInfo = false;
  converted.convert(floatType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  auto marker = cast<TypedAttr>(rewriter.getFloatAttr(floatType, converted));

  if (Value existing = findExistingConstBlock(func, marker)) {
    return existing;
  }

  // 位置：入口块的 pipe 之后。这块常量的字节数是编译期的 32，不引用任何 TilingData
  // 符号，所以不必排在 `afir.tiling_param` 之后——排在 pipe 之后就够，而 pipe 本身已经
  // 在那一段之后了。
  OpBuilder::InsertionGuard guard(rewriter);
  const Location loc = func.getLoc();
  Value pipe = findPipe(func);
  if (pipe) {
    rewriter.setInsertionPointAfterValue(pipe);
  } else {
    rewriter.setInsertionPointToStart(&func.front());
    pipe = rewriter.create<ascendc::PipeOp>(loc);
  }

  auto bufferType = ascendc::TBufType::get(rewriter.getContext(), ascendc::TPosition::VECCALC);
  Value buffer = rewriter.create<ascendc::TBufOp>(loc, bufferType);
  Value bytes = rewriter.create<arith::ConstantIndexOp>(loc, kBlockBytes);
  rewriter.create<ascendc::TPipeInitBufferOp>(loc, pipe, buffer, bytes);

  auto tensorType = ascendc::LocalTensorType::get({ShapedType::kDynamic}, floatType);
  auto tensor = rewriter.create<ascendc::TBufGetTensorOp>(loc, tensorType, buffer, /*len=*/Value());
  tensor->setAttr(kConstBlockAttrName, marker);

  Value scalar = rewriter.create<arith::ConstantOp>(loc, floatType, marker);
  const int64_t elements = kBlockBytes / static_cast<int64_t>(floatType.getWidth() / 8);
  Value count = rewriter.create<arith::ConstantIntOp>(loc, elements, 32);
  rewriter.create<ascendc::DuplicateL2Op>(loc, tensor.getTensor(), scalar, count);
  // `Duplicate` 与随后读这块张量的那个包装都在 V 上，中间没有屏障就是一次 RAW 竞争。
  // legacy 的 `BlkTensorAllocAndInit` 也是紧跟一条 `PipeBarrier<PIPE_V>`。
  rewriter.create<ascendc::PipeBarrierOp>(loc, ascendc::Pipe::PIPE_V);
  return tensor.getTensor();
}

}  // namespace mlir::ascend::kernel::elementwise
