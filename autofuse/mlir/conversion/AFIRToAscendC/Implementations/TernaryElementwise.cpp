/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- TernaryElementwise.cpp - AFIR ternary elementwise lowering -*- C++ -*-===//
//
// 这里放三元 elementwise 的两个族：选择族（`afir.select` / `afir.where`）和 clip 族
// （`afir.clip_by_value`）。结构与一元、二元两条路一致：
// 一个 implementation 由「建哪个 AscendC op」「属于哪张扩展 API 支持表」「要不要
// tmp buffer」三件独立的事确定，判据写在 `ElementwiseApiFamily.h`，三条路共用一份。
//
// 和二元路径真正的差别有两条，都不只是「多一个输入」：
//
//   1. 第一个输入的元素类型不一定和另外两个相同。 选择族的它是 ui8 掩码，clip 族
//      的它是 x，与另外两个同 dtype。如果一律要求三个输入元素类型相同，选择族就只剩
//      ui8 能过；如果一律不要求，又会放过 `f16 select f32`。所以这里用
//      `conditionIsMask` 分两种情形，而不是写一条固定判据。
//   2. 两侧生成的都是 Autofuse 的包装，不存在「标准调用 vs 扩展调用」的分法。
//      mem 侧的 `Where` / `ClipByValue` 要调用方给 tmp buffer，reg 侧自己开寄存器、
//      不要。两个 implementation 的 `ApiFamily` 因此分别是 MemBased 和 RegBased，
//      没有 `Any` 这一档。
//
// 这条路只生成连续 tile 形态：`Where(dst, cond, src0, src1, calCount[, tmp])` 和
// `ClipByValue(dst, x, min, max, calCount[, tmp])` 形状一致，共用一个 emitter。
// 选择族两侧其实还有另一套调用形态，带 firstAxis / lastAxis 和 second-to-last stride
// 的循环（legacy 的 `GenerateNormalCase` 等，以及上游那个 12 操作数的
// `ascendc.where_extend`）。AFIR 现在的「连续 tile + 元素个数」模型表达不了这套参数，
// 和比较族偏差 8 的第一条是同一件事，将来一起解。clip 族两侧都没有循环形态，不受影响。
//
// 操作数用 `afir::getOperandPartition()` 切，不用位置下标：多一块 tmp buffer 布局就变，
// 而下标写死不会报错，只会把 tmp buffer 当成第三个输入算进去。
//
// 本文件的 stable id 与注释都不出现芯片名。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/TernaryElementwise.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseLocalView.h"
#include "conversion/AFIRToAscendC/Implementations/Synchronization.h"
#include "conversion/AFIRToAscendC/VectorizedLoops.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseStockCalls.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <memory>
#include <optional>
#include <utility>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {

using ascend::instruction::CapabilityReason;
using ascend::instruction::CapabilitySupport;
using ascend::instruction::ImplementationKey;
using ascend::instruction::ImplementationRegistry;
using ascend::instruction::PatternBuildContext;
using elementwise::ApiFamily;
using elementwise::kBF16;
using elementwise::kF16;
using elementwise::kF32;
using elementwise::kI16;
using elementwise::kI32;
using elementwise::kUI16;
using elementwise::kUI8;

constexpr unsigned kSource0InputIndex = 1;
constexpr unsigned kSource1InputIndex = 2;

struct TernaryElementwiseMapping {
  ApiFamily family = ApiFamily::MemBased;
  // 选中的 API 要不要调用方给的 tmp buffer。与家族不是同一件事：mem 侧的 `Where` 要，
  // reg 侧的 `WhereExtend` 不要，而两者都是包装。
  bool needsTmpBuffer = false;
  // 第一个操作数是不是 ui8 真值掩码。选择族是，clip 族不是——那一位是 x，与另外两个输入
  // 同 dtype。这与 ODS 上 `AllElementTypesMatch` 绑不绑第一个操作数是同一件事的两处
  // 表述：`AFIR_BufferizableTernaryElementwiseOp` 不绑、`...HomoElementwiseOp` 绑。
  bool conditionIsMask = true;
  // 后两个输入是不是标量形态。三元这一类只需要一位（不像二元那边三档）：两类的第一个
  // 操作数恒是 tile，后两个各自可以是标量、位置不能换——可用实现是同名重载，标量在哪个位置
  // 由 C++ 重载决议挑。`false` 要求三个输入都是 tile，`true` 要求后两个里至少一个
  // 是标量。理由与出处见 `Elementwise.td` 的 `scalarInputs`。
  bool scalarInputs = false;
  // 数据输入收哪几种元素类型。原先这条是本文件里一个写死的
  // `isSupportedElementType(Type)`，三条路里只有三元这样——集合写死在函数体里，注册处
  // 看不见，也就没法与 legacy 的 `REG_ASC_IR` 比对。改成与一元、二元同一个形状：集合
  // 写在注册处，判据用共用的那个带掩码的重载。取值与原先那个函数逐字等价。
  uint32_t elementTypes = kF16 | kF32;
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
  unsigned resultIndex = 0;
};

// 施加过调度偏移之后的三个输入 tile、目的地 tile 与元素个数。
// `condition` 与 `destination` 恒是 tile；后两个输入**可以是标量**，那时对应的
// `LocalTensorType` 为空、元素类型落在 `source0Element` / `source1Element` 上。
// 与二元那条路的 `BinaryTensorTypes::rhsElement` 同一个形状。
struct TernaryTensorTypes {
  ascendc::LocalTensorType condition;
  ascendc::LocalTensorType source0;
  ascendc::LocalTensorType source1;
  Type source0Element;
  Type source1Element;
  ascendc::LocalTensorType destination;
};

// `scalarInputs` 与实际操作数形态必须对上，两种走法一次挡掉：tensor 形态的 implementation
// 收到标量会建出一个实参对不上的调用，标量形态收到三块 tile 会选走本该给 tensor 形态的图。
// 这与二元那条路的 `emissionOrder` 是同一件事的三元版本——只是三元不需要算次序，位置不换。
FailureOr<TernaryTensorTypes> getTensorTypes(Operation *op, bool scalarInputs) {
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(op);
  if (failed(partition) || partition->inputCount != 3 || partition->destinationCount != 1) {
    return failure();
  }
  TernaryTensorTypes types;
  types.condition = dyn_cast<ascendc::LocalTensorType>(partition->inputs(op)[0].getType());
  const Type source0Type = partition->inputs(op)[kSource0InputIndex].getType();
  const Type source1Type = partition->inputs(op)[kSource1InputIndex].getType();
  types.source0 = dyn_cast<ascendc::LocalTensorType>(source0Type);
  types.source1 = dyn_cast<ascendc::LocalTensorType>(source1Type);
  types.source0Element = types.source0 ? types.source0.getElementType() : source0Type;
  types.source1Element = types.source1 ? types.source1.getElementType() : source1Type;
  types.destination = dyn_cast<ascendc::LocalTensorType>(partition->destinations(op)[0].getType());
  if (!types.condition || !types.destination) {
    return failure();
  }
  if (scalarInputs) {
    if (elementwise::isSingletonTensorInput(op, kSource0InputIndex)) {
      types.source0 = {};
    }
    if (elementwise::isSingletonTensorInput(op, kSource1InputIndex)) {
      types.source1 = {};
    }
  }
  const bool bothAreTiles = types.source0 && types.source1;
  if (scalarInputs ? bothAreTiles : !bothAreTiles) {
    return failure();
  }
  return types;
}

bool hasRankOneTernaryTiles(const TernaryTensorTypes &types) {
  auto isRankOne = [](ascendc::LocalTensorType tile) { return !tile || (tile.hasRank() && tile.getRank() == 1); };
  return types.condition.hasRank() && types.condition.getRank() == 1 && isRankOne(types.source0) &&
         isRankOne(types.source1) && types.destination.hasRank() && types.destination.getRank() == 1;
}

bool hasMissingScalarInputInformation(Operation *operation) {
  for (unsigned index : {kSource0InputIndex, kSource1InputIndex}) {
    if (elementwise::isSingletonTensorInput(operation, index) &&
        (!scalarReadEvent(operation, operation->getOperand(index)) || !findFunctionPipe(operation))) {
      return true;
    }
  }
  return false;
}

bool hasUnsupportedTernaryElementTypes(const TernaryElementwiseMapping &mapping, const TernaryTensorTypes &types) {
  // 选择族的条件是一个字节的真值，另外两个输入与结果同 dtype。两条分开查：只查
  // 「三者相同」会把条件也拖成数据类型，只查「条件是 ui8」会放过 f16 select f32。
  if (mapping.conditionIsMask && !types.condition.getElementType().isUnsignedInteger(8)) {
    return true;
  }
  // 元素类型对标量取它自己，对 tile 取元素类型——`getTensorTypes` 已经归一化过。
  const Type elementType = types.source0Element;
  if (types.source1Element != elementType || types.destination.getElementType() != elementType ||
      !elementwise::isSupportedElementType(mapping.elementTypes, elementType)) {
    return true;
  }
  // clip 族的第一个操作数是 x，不是掩码——「ui8」那一条对它不成立，换成与另外两个同
  // dtype。legacy 在 codegen 里用 `GE_ASSERT_TRUE(x1_dtype_name == x2_dtype_name)`
  // 兜的就是这一条。
  return !mapping.conditionIsMask && types.condition.getElementType() != elementType;
}

class TernaryElementwiseCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit TernaryElementwiseCapabilityProvider(std::shared_ptr<const TernaryElementwiseMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_ || !elementwise::hasCanonicalDescriptors(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    SmallVector<unsigned> scalars;
    if (mapping_->scalarInputs) {
      for (unsigned index : {kSource0InputIndex, kSource1InputIndex}) {
        if (elementwise::isSingletonTensorInput(operation, index)) {
          scalars.push_back(index);
        }
      }
    }
    elementwise::ElementwiseLocalView view;
    auto compatibility = elementwise::prepareElementwiseLocalView(operation, scalars, false, view);
    if (compatibility != elementwise::LocalViewCompatibility::Compatible) {
      return CapabilitySupport::unsupported(compatibility == elementwise::LocalViewCompatibility::MissingInformation
                                                ? CapabilityReason::MissingInformation
                                                : CapabilityReason::UnsupportedShape);
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

    return queryTensorContract(operation, !view.vectorPlan);
  }

  CapabilitySupport queryTensorContract(Operation *operation, bool wholeTensor) const {
    FailureOr<TernaryTensorTypes> types = getTensorTypes(operation, mapping_->scalarInputs);
    if (failed(types)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    if (mapping_->scalarInputs && hasMissingScalarInputInformation(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // Rank applies to tiles only; scalar inputs have no rank.
    if (wholeTensor && !hasRankOneTernaryTiles(*types)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    if (hasUnsupportedTernaryElementTypes(*mapping_, *types)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *operation, const ascend::instruction::InplaceQuery &query,
                                 const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_ || query.outputResultIndex != mapping_->resultIndex) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
    if (failed(partition) || partition->inputCount != 3 || partition->destinationCount != 1) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // 只有输入可以与结果重叠。目的地本来就是结果，拿它来问是问错了问题。
    if (query.inputOperandIndex >= partition->inputCount) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (query.accessMode != ascend::instruction::AliasAccessMode::ReadWriteInput) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    if (mapping_->scalarInputs && elementwise::isSingletonTensorInput(operation, query.inputOperandIndex)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    if (!elementwise::hasSameLocalLayout(operation, query.inputOperandIndex)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    CapabilitySupport lowering = queryLowering(operation, context);
    if (lowering.getState() != ascend::instruction::SupportState::Supported) {
      return lowering;
    }
    llvm::SmallVector<ascend::instruction::CapabilityConstraint, 4> constraints;
    constraints.emplace_back(
        ascend::instruction::AliasConstraint{query.inputOperandIndex, query.outputResultIndex, query.accessMode});
    return CapabilitySupport::supported(std::move(constraints));
  }

  CapabilitySupport queryScalarInput(Operation *, const ascend::instruction::ScalarInputQuery &,
                                     const ascend::instruction::QueryContext &) const override {
    // 标量形态是另一组 implementation（`<op>.tensor_scalar.*`），不是这个查询的答案
    // ——与二元那条路逐字相同的规则。
    //
    // **原先这里写着「那走的是另一套调用形态（`Where<true, true>` 加 second-to-last
    // stride）」，那是错的**：`where.h` 的三个标量重载（场景 1/2/3）就是
    // `template <typename T> Where(dst, mask, src0, src1, size, tmp)`，与 tensor 形态同名
    // 同位，没有 bool 模板实参也没有 stride；`clipbyvalue.h` 的场景 2/3/4 同理。
    // 又一次「文档说卡住了」不成立。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    // 内联广播包装只收两个源操作数（`BinaryBrcInlineApiWithTwoVectorizedAxisV2`），三元 op
    // 有三个源，形状都对不上这个接口。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const TernaryElementwiseMapping> mapping_;
};

struct TernaryOperandTiles {
  Value condition;
  Value source0;
  Value source1;
  Value destination;
  Value count;
};

template <typename SourceOp>
FailureOr<SmallVector<unsigned>> materializeScalarInputs(SourceOp op, TernaryOperandTiles &tiles,
                                                         ConversionPatternRewriter &rewriter) {
  SmallVector<unsigned> scalarIndices;
  Value *const sources[] = {&tiles.condition, &tiles.source0, &tiles.source1};
  for (unsigned index : {kSource0InputIndex, kSource1InputIndex}) {
    if (!elementwise::isSingletonTensorInput(op, index)) {
      continue;
    }
    auto scalar = materializeScalarRead(op, *sources[index], scalarReadEvent(op, op->getOperand(index)), rewriter);
    if (failed(scalar)) {
      return failure();
    }
    *sources[index] = *scalar;
    scalarIndices.push_back(index);
  }
  return scalarIndices;
}

FailureOr<Value> applyVectorizedWindow(TernaryOperandTiles &tiles, const afir::VectorPlan &segments,
                                       const VectorizedWindow &window) {
  if (window.tiles.size() != segments.tensors.size() || window.extents.size() != 1) {
    return failure();
  }
  Value *const byIndex[] = {&tiles.condition, &tiles.source0, &tiles.source1};
  for (auto [tensor, tile] : llvm::zip(segments.tensors, window.tiles)) {
    if (tensor.side == afir::VectorTensor::Side::Destination) {
      tiles.destination = tile;
      continue;
    }
    if (tensor.index >= std::size(byIndex)) {
      return failure();
    }
    *byIndex[tensor.index] = tile;
  }
  return window.extents.front();
}

template <typename SourceOp, typename OpAdaptor>
FailureOr<TernaryOperandTiles> materializeOperandTiles(SourceOp op, OpAdaptor adaptor,
                                                       ConversionPatternRewriter &rewriter, bool scalarInputs) {
  TernaryOperandTiles tiles;
  tiles.condition = adaptor.getCondition();
  tiles.source0 = adaptor.getInput1();
  tiles.source1 = adaptor.getInput2();
  tiles.destination = adaptor.getDestination();
  SmallVector<unsigned> scalarIndices;
  if (scalarInputs) {
    FailureOr<SmallVector<unsigned>> materialized = materializeScalarInputs(op, tiles, rewriter);
    if (failed(materialized)) {
      return failure();
    }
    scalarIndices = *materialized;
  }
  elementwise::ElementwiseLocalView view;
  if (elementwise::prepareElementwiseLocalView(op, scalarIndices, false, view) !=
      elementwise::LocalViewCompatibility::Compatible) {
    return failure();
  }
  if (!view.vectorPlan) {
    tiles.count = rewriter.create<ascendc::LocalTensorGetSizeOp>(op.getLoc(), rewriter.getI32Type(), tiles.destination);
    return tiles;
  }
  const auto &segments = view.vectorPlan;
  FailureOr<SmallVector<Value>> ordered =
      orderTilesForPlan(*segments, {tiles.condition, tiles.source0, tiles.source1}, tiles.destination);
  if (failed(ordered)) {
    return failure();
  }
  FailureOr<VectorizedWindow> window =
      materializeVectorizedSegments(op, rewriter, *segments, view.extentSource, *ordered, /*apiWindowSize=*/1);
  if (failed(window)) {
    return failure();
  }
  FailureOr<Value> count = applyVectorizedWindow(tiles, *segments, *window);
  if (failed(count)) {
    return failure();
  }
  tiles.count = *count;
  return tiles;
}

template <typename SourceOp, typename TargetOp, bool TakesTmpBuffer>
class TernaryElementwiseConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  TernaryElementwiseConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                      std::shared_ptr<const TernaryElementwiseMapping> mapping)
      : OpConversionPattern<SourceOp>(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form ternary elementwise op requires a destination");
    }

    FailureOr<TernaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, rewriter, mapping_->scalarInputs);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    Operation *targetOp = nullptr;
    if constexpr (TakesTmpBuffer) {
      if (!adaptor.getTmpBuffer()) {
        return rewriter.notifyMatchFailure(op, "the selected API requires a tmp buffer operand");
      }
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->condition, tiles->source0,
                                           tiles->source1, tiles->count, adaptor.getTmpBuffer());
    } else {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->condition, tiles->source0,
                                           tiles->source1, tiles->count);
    }
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const TernaryElementwiseMapping> mapping_;
};

template <typename SourceOp, typename TargetOp, bool TakesTmpBuffer>
ImplementationRegistry::PatternRegistration<TernaryElementwiseMapping> createPatternRegistration() {
  return [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
            std::shared_ptr<const TernaryElementwiseMapping> mapping) {
    patterns.add<TernaryElementwiseConversionPattern<SourceOp, TargetOp, TakesTmpBuffer>>(
        context.typeConverter, patterns.getContext(), key, std::move(mapping));
  };
}

template <typename SourceOp, typename TargetOp, bool TakesTmpBuffer>
LogicalResult registerTernary(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family,
                              bool conditionIsMask, uint32_t elementTypes, bool scalarInputs = false) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(family))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, family, elementTypes);
  return registry.registerImplementation<SourceOp, TernaryElementwiseMapping, TernaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      TernaryElementwiseMapping{family, TakesTmpBuffer, conditionIsMask, scalarInputs, elementTypes},
      createPatternRegistration<SourceOp, TargetOp, TakesTmpBuffer>());
}

// 选择族的两侧：mem 走 `Where` 且要 tmp buffer,reg 走 `WhereExtend` 且不要。这一对在
// 三个 op 上逐字相同，所以在这里生成一次。
template <typename SourceOp>
LogicalResult registerSelection(ImplementationRegistry &registry, llvm::StringRef op, uint32_t memElementTypes,
                                uint32_t regElementTypes) {
  // i16 原先漏了：ODS 的 `AFIR_ConditionElementTypes` 有它、legacy 两侧的 `REG_ASC_IR`
  // 也有它，只有注册处没写——`ElementTypeMatrixTest` 把这一条照了出来。
  // legacy 两侧还收 i64、reg 侧还收 ui8 / ui16，AFIR 这一层表达不出来。
  if (failed(registerTernary<SourceOp, ascendc::WhereContiguousOp, /*TakesTmpBuffer=*/true>(
          registry, (op + ".tensor.extend").str(), ApiFamily::MemBased, /*conditionIsMask=*/true, memElementTypes)) ||
      failed(registerTernary<SourceOp, ascendc::WhereContiguousRegBaseOp, /*TakesTmpBuffer=*/false>(
          registry, (op + ".tensor.extend_reg_base").str(), ApiFamily::RegBased, /*conditionIsMask=*/true,
          regElementTypes))) {
    return failure();
  }
  // 标量形态：`then` / `else` 里至少一个是标量（`where.h` 的场景 1/2/3）。**用的是与
  // tensor 形态同一个 def**——同名重载、实参同位，只有那个位置上的 SSA 值从 tile 变成标量。
  // 元素类型与 tmp 需求都跟着各自那一侧，一个字不改。
  if (failed(registerTernary<SourceOp, ascendc::WhereContiguousOp, /*TakesTmpBuffer=*/true>(
          registry, (op + ".tensor_scalar.extend").str(), ApiFamily::MemBased, /*conditionIsMask=*/true,
          memElementTypes, /*scalarInputs=*/true))) {
    return failure();
  }
  return registerTernary<SourceOp, ascendc::WhereContiguousRegBaseOp, /*TakesTmpBuffer=*/false>(
      registry, (op + ".tensor_scalar.extend_reg_base").str(), ApiFamily::RegBased, /*conditionIsMask=*/true,
      regElementTypes, /*scalarInputs=*/true);
}

}  // namespace

LogicalResult registerTernaryElementwiseImplementations(ImplementationRegistry &registry,
                                                        ascend::instruction::SelectionPolicyRegistry *) {
  // `select` 与 `where` 在 legacy 里是同一个包装的两个 AscIR 节点类型(`SelectAscIrCodegenImpl`
  // 与 `WhereAscIrCodegenImpl` 逐字相同，只有 tmp 尺寸函数换了个名字而两者互相调用),
  // 差别只在 legacy 的 `IsScalarInputSupported` 允许哪个输入是标量，而两者允许的位置相同
  // （后两个），所以两者在 AFIR 这一层完全一样——标量形态也一样，各自多两个 implementation。
  // 两者的集合不一样：legacy 的 mem 侧相同，reg 侧只有 `Where` 多收 ui8 / ui16。
  // i16 原先两个都漏了——ODS 有、legacy 两侧也有，只有注册处没写。
  constexpr uint32_t kSelectionElementTypes = kF16 | kF32 | kI16 | kI32;
  constexpr uint32_t kWhereRegElementTypes = kSelectionElementTypes | kUI8 | kUI16 | kBF16;
  if (failed(registerSelection<SelectOp>(registry, "select", kSelectionElementTypes, kSelectionElementTypes)) ||
      failed(registerSelection<WhereOp>(registry, "where", kSelectionElementTypes, kWhereRegElementTypes))) {
    return failure();
  }

  // 选择族只有这两个，没有 `masked_fill`。 理由见 `Ops/Condition.td` 上那段：
  // `MaskedFillInputReorderPass` 在调度之前就把 `MaskedFill` 改写成了 `Select`，这条路
  // 上看不到那个节点类型。曾经注册过一个只有 mem 侧的 `masked_fill.tensor.extend`，没有
  // 任何用例走得到，已删。

  // clip 族。与选择族同形——两侧同名 `ClipByValue`，mem 侧要 tmp buffer、reg 侧不要
  // （它转发给 `Clamp`，自己开寄存器）——差别只有一条：第一个操作数是 x 而不是掩码，
  // 与另外两个输入同 dtype。所以这里不能走 `registerSelection`，`conditionIsMask` 得
  // 是 false。
  // 只收 f16 / f32：ODS 的 `AFIR_BufferizableTernaryHomoElementwiseOp<"clip_by_value", [F16, F32]>`
  // 与 legacy 的 mem 侧一致。原先注册处多写了 i32，ODS 本来就拒，是一句没人读的假话。
  // reg 侧 legacy 宽得多（到 ui64），AFIR 这一层表达不出来。
  constexpr uint32_t kClipMemElementTypes = kF16 | kF32;
  constexpr uint32_t kClipRegElementTypes = kClipMemElementTypes | kI16 | kI32 | kUI8 | kUI16 | kBF16;
  if (failed(registerTernary<ClipByValueOp, ascendc::ClipByValueContiguousOp, /*TakesTmpBuffer=*/true>(
          registry, "clip_by_value.tensor.extend", ApiFamily::MemBased, /*conditionIsMask=*/false,
          kClipMemElementTypes)) ||
      failed(registerTernary<ClipByValueOp, ascendc::ClipByValueContiguousRegBaseOp, /*TakesTmpBuffer=*/false>(
          registry, "clip_by_value.tensor.extend_reg_base", ApiFamily::RegBased, /*conditionIsMask=*/false,
          kClipRegElementTypes))) {
    return failure();
  }

  // clip 的标量形态：`min` / `max` 里至少一个是标量（两份头的场景 2/3/4）。**`x` 不能是
  // 标量**——只有「三个全是标量」那一个重载收它，而那个重载没有 count / tmp 形参，
  // `verifyBufferizedElementwise` 的「至少一个输入有形状」本来就挡掉了全标量。这一条在
  // ODS 那一层就定死了（`condition` 位不给标量类型），这里不用再判一次。
  if (failed(registerTernary<ClipByValueOp, ascendc::ClipByValueContiguousOp, /*TakesTmpBuffer=*/true>(
          registry, "clip_by_value.tensor_scalar.extend", ApiFamily::MemBased, /*conditionIsMask=*/false,
          kClipMemElementTypes, /*scalarInputs=*/true))) {
    return failure();
  }
  if (failed(registerTernary<ClipByValueOp, ascendc::ClipByValueContiguousRegBaseOp, /*TakesTmpBuffer=*/false>(
          registry, "clip_by_value.tensor_scalar.extend_reg_base", ApiFamily::RegBased, /*conditionIsMask=*/false,
          kClipRegElementTypes, /*scalarInputs=*/true))) {
    return failure();
  }

  // step 3。`fma` 与 clip 族同形——三个输入同 dtype，所以 `conditionIsMask` 是 false
  // （躯干那一位在这个 op 上读作 x1，不是掩码）——差别有两条，都不在这条 pattern 上：
  //
  //   1. 可用实现是标准调用 `AscendC::Fma`，所以 target op 是 `FmaL2Op`（不声明
  //      `ApiExtensionOpInterface`、进 `kIsStockCall`），与 `hypot` 同一种情况；
  //   2. tmp 在 count 之前。这条 pattern 建的是 `(dst, 三个源, count, tmp)`，而
  //      `FmaL2Op` 的 `arguments` 正是按它声明的，真实次序由 emitter 打
  //      （`Autofuse/VecTernary.h`）。所以这里不需要新 pattern。
  //
  // 只有 reg 一侧（`REG_ASC_IR(Fma)` 只在 v2 那张表上），标量形态一个都没有
  // （`IsNodeValid` 里的 `!IsNodeHasScalarInput`）。元素类型取 legacy 的 `T` 去掉 i8。
  constexpr uint32_t kFmaElementTypes = kF16 | kF32 | kI16 | kUI8 | kBF16;
  return registerTernary<FmaOp, ascendc::FmaL2Op, /*TakesTmpBuffer=*/true>(registry, "fma.tensor", ApiFamily::RegBased,
                                                                           /*conditionIsMask=*/false, kFmaElementTypes);
}

}  // namespace mlir::ascend::kernel
