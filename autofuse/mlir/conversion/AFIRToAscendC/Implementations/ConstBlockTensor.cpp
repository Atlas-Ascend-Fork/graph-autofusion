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

// 包装将 src0BlkStride / src0RepStride 设为 0，每个 repeat 都读取同一 block，
// 因此只需分配一个 block，不随 tile 大小增长。字节数与 legacy ONE_BLK_SIZE 一致。
constexpr int64_t kBlockBytes = 32;

// 按 ascendc.tbuf.get_tensor 的标记查找已创建的常量块。
// 标记使用带类型的常量，同时区分数值与 dtype。
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

// 查找 MemoryAllocate 在入口块创建的 ascendc.pipe。
// 仅带 local_tensor 参数的 LIT 可能没有资源；此时返回空，由调用方创建 pipe。
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
    // 当前使用者仅需浮点常量：RsqrtExtend 用 float，LogicalNot 用 half；整数请求时报错。
    return op->emitOpError("a pre-filled constant block currently requires a floating-point element type");
  }
  llvm::APFloat converted = value;
  bool losesInfo = false;
  converted.convert(floatType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  auto marker = cast<TypedAttr>(rewriter.getFloatAttr(floatType, converted));

  if (Value existing = findExistingConstBlock(func, marker)) {
    return existing;
  }

  // 在入口块的 pipe 后创建常量块。大小固定为 32 字节，不依赖 TilingData 符号；
  // pipe 已位于 tiling 参数序言之后，无需另行安排位置。
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
