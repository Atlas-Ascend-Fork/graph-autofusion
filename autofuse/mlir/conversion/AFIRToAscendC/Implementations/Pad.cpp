/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Pad.cpp - AFIR pad lowering ---------------------------*- C++ -*-===//
//
// `afir.pad` 的 lowering。**一档**：legacy 两侧 codegen 走同一个 `PadApiCall`
// （`codegen/api_call/elewise/pad_api_call.cpp:26`），发同一个官方 `AscendC::Pad`
// （矩阵 tier D1），所以 `ApiFamily::Shared`、只有一个 implementation。
//
// legacy 是**运行期二选一**（`pad_api_call.cpp:61-83`）：条件
// `Size(y.vectorized_strides[n-2]) != ActualSize(y.axis_size[尾轴])`——倒数第二根
// 向量化轴的对齐行距与尾轴实际长度，布局里真的有补位吗。有补位走
// `AscendC::Pad(y, x, PadParams{0, S-E, 0}, tmp, tiling)`，没补位退化为一条
// `AscendC::DataCopy(y, x, BlkAlign<T>(元素数连乘))`。这里移植为 `scf.if` 二分支，
// 条件、两支的参数算式与 legacy 同构（legacy 只在「编译期就可证无补位」时省掉 if——
// 运行期结果相同，少一个特例分支）。
//
// 尺寸全部按 legacy 读**输出（y）**：S 是 y 的对齐行距、E 是 y 的尾轴实际长度、元素数
// 连乘是 y 各向量化轴实际长度之积。输入（x）只贡献两侧恒 0 的向量化偏移（UB→UB 的
// tile 内调用，与 `RemovePad.cpp` / `UnaryElementwise.cpp` 同一条判据）。
//
// 收的形态边界（不满足一律 unsupported）：
//   - 两份向量化清单齐：两侧 `vectorized_axis` 是同一串、至少两根（legacy 的条件读
//     `strides[n-2]`，一根轴发不出条件——legacy 那里是 vector 负下标，直接 UB）；
//   - 两侧最内向量化 stride 可证为 1：两条分支都按「行内连续」读写；
//   - **源整窗紧凑**：源行距可证等于源尾轴实际长度。两条分支都按连续块读源
//     （`Pad` 的 srcHeight×srcWidth 稠密读、`DataCopy` 的整块拷），行距对不上是
//     静默读错，legacy 不查这条（调度结构性保证），这里把它变成可判的判据；
//   - 尾轴实际长度两侧一致：host tiling 按**输入**的尾轴算宽度（ATT
//     `api_tiling_gen/api/pad.cpp:25` 的 `GetPadHeightAndWidth(node->inputs[0])`），
//     device 按**输出**的布局写，两侧说岔了 host 写的 tiling 与 device 的布局对不上。
//
// 调度范围统一由 ComputeExtent 按轴身份读取；需要尾块修正却缺少 ActualSize
// 绑定时必须失败，不能退回名义长度。独立 local view 使用 descriptor repeats。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/Pad.h"
#include "dialect/AFIR/Analysis/ComputeExtent.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
// `kIsStockCall<ascendc::PadOp>` 的特化在 `ElementwiseStockCalls.h` 上——这张表不随
// `ElementwiseApiFamily.h` 下发，谁问谁自带，漏了的话这里读到主模板的 false，注册期
// `checkFamilyMatchesHeader` 当场判 `PadOp` 不是标准调用，整张注册表建不起来。
#include "conversion/AFIRToAscendC/Implementations/ElementwiseStockCalls.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <memory>
#include <optional>
#include <utility>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {
constexpr size_t kMinimumPadRank = 2;

using ascend::instruction::CapabilityReason;
using ascend::instruction::CapabilitySupport;
using ascend::instruction::ImplementationKey;
using ascend::instruction::ImplementationRegistry;
using ascend::instruction::PatternBuildContext;
using elementwise::ApiFamily;
using elementwise::isSupportedElementType;
using elementwise::kF16;
using elementwise::kF32;
using elementwise::kI16;
using elementwise::kI32;
using elementwise::kUI16;

// 与 `kRemovePadElementTypes` 同一列：legacy 两侧供给表的 dtype 集相同，`ui32` / `bf16`
// 在 `ElementTypes` 里没有位（`elementTypeBit` 答 0），够不着也就谈不上收。
constexpr uint32_t kPadElementTypes = kF16 | kF32 | kI16 | kI32 | kUI16;

struct PadMapping {
  ApiFamily family = ApiFamily::Shared;
  uint32_t elementTypes = kPadElementTypes;
  // 官方 `AscendC::Pad` 的形参表带 `sharedTmpBuffer`，两侧 `CalcPadTmpSize` 都算尺寸，
  // 调度因此必给 tmp——`checkTmpBuffer` 拿这条与调度严格对等。
  bool needsTmpBuffer = true;
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
};

// 一个 `afir.pad` 的可判形态。只记判据用得上的两份 descriptor 与两个表达式，
// 不记物化出来的值——capability 只判形态、不建 IR，pattern 才物化。
struct PadShape {
  TensorDescAttr source;       // x：紧凑侧（legacy 的 inputs[0]）
  TensorDescAttr destination;  // y：带填充侧（legacy 的 outputs[0]，全部尺寸按它读）
  ExprAttr pitch;              // S：y 倒数第二根向量化轴的对齐行距
  ExprAttr width;              // E：y 尾轴实际长度
};

// descriptor 的数组自洽：axis / repeats / strides 等长，vectorized 两列等长。
bool descriptorArraysConsistent(TensorDescAttr descriptor) {
  if (!descriptor) {
    return false;
  }
  const size_t axes = descriptor.getAxis().size();
  if (axes == 0 || descriptor.getRepeats().size() != axes || descriptor.getStrides().size() != axes) {
    return false;
  }
  return descriptor.getVectorizedAxis().size() == descriptor.getVectorizedStrides().size();
}

// `op` 是不是本档收的形状；是就把判据用得上的那几项记下来（文件头的四条边界）。
FailureOr<PadShape> analyzePad(Operation *op) {
  auto operandDescs = op->getAttrOfType<ArrayAttr>(kOperandDescsAttrName);
  TensorDescAttr source =
      (operandDescs && !operandDescs.empty()) ? dyn_cast<TensorDescAttr>(operandDescs[0]) : TensorDescAttr();
  auto destinationDescs = op->getAttrOfType<ArrayAttr>(kDestinationDescsAttrName);
  TensorDescAttr destination = (destinationDescs && !destinationDescs.empty())
                                   ? dyn_cast<TensorDescAttr>(destinationDescs[0])
                                   : TensorDescAttr();
  if (!descriptorArraysConsistent(source) || !descriptorArraysConsistent(destination)) {
    return failure();
  }
  if (source.getVectorizedAxis().empty() || source.getVectorizedAxis() != destination.getVectorizedAxis()) {
    return failure();
  }
  const size_t count = destination.getVectorizedAxis().size();
  if (count < kMinimumPadRank) {
    return failure();
  }
  if (!provablyOne(destination.getVectorizedStrides()[count - 1]) ||
      !provablyOne(source.getVectorizedStrides()[count - 1])) {
    return failure();
  }
  FailureOr<size_t> lastPosition = axisPosition(destination, destination.getVectorizedAxis()[count - 1]);
  FailureOr<size_t> sourceLastPosition = axisPosition(source, source.getVectorizedAxis()[count - 1]);
  if (failed(lastPosition) || failed(sourceLastPosition)) {
    return failure();
  }
  PadShape shape{source, destination, destination.getVectorizedStrides()[count - 2],
                 destination.getRepeats()[*lastPosition]};
  auto hasDenseOuterStrides = [](TensorDescAttr descriptor, bool padded) {
    const auto axes = descriptor.getVectorizedAxis();
    const auto strides = descriptor.getVectorizedStrides();
    // Pad 操作的是 UB 向量化窗口；逻辑 strides 还包含 GM 的切分、块循环布局。
    // 目的侧最内行距允许 padding，其余外层步长必须连续。
    for (size_t axis = axes.size() - (padded ? 2 : 1); axis > 0; --axis) {
      FailureOr<size_t> position = axisPosition(descriptor, axes[axis]);
      if (failed(position)) {
        return false;
      }
      FailureOr<ExprAttr> expected =
          multiplyExprs(descriptor.getContext(), strides[axis], descriptor.getRepeats()[*position]);
      if (failed(expected) || !provablyEqual(strides[axis - 1], *expected)) {
        return false;
      }
    }
    return true;
  };
  // Pad 发的是一次扁平窗口调用，因此不仅行内、还必须证明整个源窗口是连续的。
  if (!hasDenseOuterStrides(source, false)) {
    return failure();
  }
  // 目的侧允许尾轴补齐，但每个外层平面必须按统一 padded 行距推进；否则一次从零偏移
  // 的 Pad 会把第二个及后续平面写到错误位置。
  if (!hasDenseOuterStrides(destination, true)) {
    return failure();
  }
  // 尾轴实际长度两侧一致（host 按输入算宽度、device 按输出布局写）。
  if (!provablyEqual(shape.width, source.getRepeats()[*sourceLastPosition])) {
    return failure();
  }
  return shape;
}

// Scheduled axes use the shared ActualSize bindings; missing bindings must not
// fall back to nominal extents. Standalone local views use descriptor repeats.
FailureOr<Value> materializeAxisExtent(Operation *op, OpBuilder &builder, TensorDescAttr descriptor, int64_t axisId) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (func && func->hasAttr(kAscGraphAttrName)) {
    auto extents = materializeScheduledAxisExtents(op, builder, {axisId});
    if (failed(extents)) {
      return failure();
    }
    return (*extents)[0].actual;
  }
  FailureOr<size_t> position = axisPosition(descriptor, axisId);
  if (failed(position)) {
    return failure();
  }
  return materializeExtentValue(op, builder, descriptor.getRepeats()[*position]);
}

class PadCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit PadCapabilityProvider(std::shared_ptr<const PadMapping> mapping) : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_ || !elementwise::hasCanonicalDescriptors(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }

    ascend::FeatureState executionSupport = context.target.getExecutionSupport(mapping_->executionSpec);
    if (executionSupport == ascend::FeatureState::Unavailable) {
      return CapabilitySupport::unsupported(CapabilityReason::TargetFeatureUnavailable);
    }
    if (executionSupport == ascend::FeatureState::Unknown) {
      return CapabilitySupport::undetermined(CapabilityReason::TargetFeatureNotModeled);
    }

    if (std::optional<CapabilitySupport> family = elementwise::checkApiFamily(mapping_->family, context)) {
      return *family;
    }
    if (std::optional<CapabilitySupport> tmpBuffer = elementwise::checkTmpBuffer(operation, mapping_->needsTmpBuffer)) {
      return *tmpBuffer;
    }
    // host tiling 字段名必须在：`PadTiling` 的 26 个成员由 ATT 在 host 侧填，这一层只按
    // 字段名取整个 struct，没有它取不到（`Transpose` mem 档同一条判据）。
    auto field = operation->getAttrOfType<StringAttr>(kApiTilingFieldAttrName);
    if (!field || !getApiTilingFieldType(operation, field.getValue())) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }

    FailureOr<OperandPartition> partition = getOperandPartition(operation);
    if (failed(partition) || partition->inputCount != 1 || partition->destinationCount != 1) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    auto sourceType = dyn_cast<ascendc::LocalTensorType>(partition->inputs(operation)[0].getType());
    auto destinationType = dyn_cast<ascendc::LocalTensorType>(partition->destinations(operation)[0].getType());
    if (!sourceType || !destinationType) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    // 两端同 dtype 由 ODS 的 `AllElementTypesMatch` 钉住，这里只查集合。
    if (!isSupportedElementType(mapping_->elementTypes, sourceType.getElementType()) ||
        sourceType.getElementType() != destinationType.getElementType()) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }

    if (failed(analyzePad(operation))) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *, const ascend::instruction::InplaceQuery &,
                                 const ascend::instruction::QueryContext &) const override {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryScalarInput(Operation *, const ascend::instruction::ScalarInputQuery &,
                                     const ascend::instruction::QueryContext &) const override {
    // legacy 的 `IsNodeValid` 也拒标量输入（`v1_ascir_codegen_impl.h:287`）。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    // `Pad` 的输入输出差的是补零的范围，不是广播；形状关系与 `(1, A) op (B, A)` 无关。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const PadMapping> mapping_;
};

FailureOr<Value> materializePadCount(Operation *op, OpBuilder &builder, TensorDescAttr destination) {
  Value count;
  for (int64_t axis : destination.getVectorizedAxis()) {
    auto extent = materializeAxisExtent(op, builder, destination, axis);
    if (failed(extent)) {
      return failure();
    }
    count = count ? builder.create<arith::MulIOp>(op->getLoc(), count, *extent).getResult() : *extent;
  }
  return count;
}

LogicalResult emitPadBranches(PadOp op, PadOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
                              const PadShape &shape, const ascend::ExecutionSpec &execution) {
  auto tiling = getCanonicalTilingArgument(op->getParentOfType<func::FuncOp>());
  if (failed(tiling)) {
    return failure();
  }
  Value apiTiling = createTilingSchemaMember(rewriter, op.getLoc(), *tiling, *op.getApiTilingField());
  auto pitch = materializeStrideValue(op, rewriter, shape.pitch);
  auto width = materializeAxisExtent(op, rewriter, shape.destination, shape.destination.getVectorizedAxis().back());
  if (failed(pitch) || failed(width)) {
    return failure();
  }
  Value padNum = rewriter.create<arith::SubIOp>(op.getLoc(), *pitch, *width);
  Value padded = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::ne, *pitch, *width);
  Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value source = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), adaptor.getInput().getType(),
                                                                 adaptor.getInput(), zero);
  Value destination = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), adaptor.getDestination().getType(),
                                                                      adaptor.getDestination(), zero);
  auto branch = rewriter.create<scf::IfOp>(op.getLoc(), padded, /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(branch.thenBlock());
  Operation *call =
      rewriter.create<ascendc::PadOp>(op.getLoc(), destination, source, padNum, adaptor.getTmpBuffer(), apiTiling);
  if (failed(setExecutionSpec(call, execution))) {
    return failure();
  }
  rewriter.setInsertionPointToStart(branch.elseBlock());
  auto count = materializePadCount(op, rewriter, shape.destination);
  if (failed(count)) {
    return failure();
  }
  rewriter.create<ascendc::DataCopyBlkAlignedOp>(op.getLoc(), destination, source, *count);
  rewriter.setInsertionPointAfter(branch);
  return success();
}

class PadConversionPattern final : public OpConversionPattern<PadOp> {
 public:
  PadConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                       std::shared_ptr<const PadMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(PadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination() || !adaptor.getTmpBuffer()) {
      return rewriter.notifyMatchFailure(op, "buffer-form pad requires a destination and a tmp buffer");
    }
    FailureOr<PadShape> shape = analyzePad(op);
    if (failed(shape)) {
      return rewriter.notifyMatchFailure(op, "not a supported pad form");
    }

    std::optional<llvm::StringRef> field = op.getApiTilingField();
    if (!field || !getApiTilingFieldType(op, *field) ||
        failed(emitPadBranches(op, adaptor, rewriter, *shape, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "cannot materialize pad tiling or branches");
    }
    rewriter.replaceOp(op, adaptor.getDestination());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const PadMapping> mapping_;
};

}  // namespace

LogicalResult registerPadImplementations(ImplementationRegistry &registry,
                                         ascend::instruction::SelectionPolicyRegistry *) {
  // 两侧共用同一个官方调用（tier D1），所以 `Shared`、只有这一个 implementation。
  // id 说的是供给表，不是芯片。
  constexpr llvm::StringLiteral kId = "pad.ub_ub_extend";
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::PadOp>(ApiFamily::Shared)) ||
      failed(elementwise::checkFamilyMatchesHeader<ascendc::DataCopyBlkAlignedOp>(ApiFamily::Shared))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(PadOp::getOperationName(), kId, ApiFamily::Shared, kPadElementTypes);
  return registry.registerImplementation<PadOp, PadMapping, PadCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{kId.str()},
      PadMapping{ApiFamily::Shared, kPadElementTypes, /*needsTmpBuffer=*/true},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const PadMapping> mapping) {
        patterns.add<PadConversionPattern>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
      });
}

}  // namespace mlir::ascend::kernel
