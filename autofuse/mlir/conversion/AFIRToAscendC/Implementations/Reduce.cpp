/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/Reduce.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "dialect/AFIR/Analysis/ReducePlan.h"
#include "dialect/AFIR/Analysis/ReduceScratchLifetime.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "conversion/AFIRToAscendC/Implementations/ReduceGeometry.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {

using ascend::instruction::CapabilityReason;
using ascend::instruction::CapabilitySupport;
using ascend::instruction::ImplementationKey;
using ascend::instruction::ImplementationRegistry;
using ascend::instruction::PatternBuildContext;

struct ReduceMapping {
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
};

// Match the memory-based legacy operator contracts; API selection is separate
// because integer Sum uses Autofuse's exact integer implementation.
bool isSupportedElementType(Operation *op, Type type) {
  return type.isF32() || (isa<afir::SumOp>(op) && type.isSignlessInteger(32)) ||
         (isa<afir::MaxOp, afir::MinOp>(op) && type.isF16());
}

// 各 op 的初始化单位元、归约 API 和跨轮合并指令。
// Any/All 的 f32 值域为 {0, 1}，分别用 Max/Min 合并。
template <typename OpT>
struct ReduceTraits;

template <>
struct ReduceTraits<afir::SumOp> {
  using ApiOp = ascendc::ReduceSum2DL2Op;
  using AccumulateOp = ascendc::AddL2Op;
  static constexpr ascendc::ReduceInitType kInitType = ascendc::ReduceInitType::Sum;
  static constexpr bool kScalesByReducedCount = false;
  static constexpr const char *kImplementationId = "reduce_sum.tensor_ar_ra";
};

template <>
struct ReduceTraits<afir::MaxOp> {
  using ApiOp = ascendc::ReduceMax2DL2Op;
  using AccumulateOp = ascendc::MaxL2Op;
  static constexpr ascendc::ReduceInitType kInitType = ascendc::ReduceInitType::Max;
  static constexpr bool kScalesByReducedCount = false;
  static constexpr const char *kImplementationId = "reduce_max.tensor_ar_ra";
};

template <>
struct ReduceTraits<afir::MinOp> {
  using ApiOp = ascendc::ReduceMin2DL2Op;
  using AccumulateOp = ascendc::MinL2Op;
  static constexpr ascendc::ReduceInitType kInitType = ascendc::ReduceInitType::Min;
  static constexpr bool kScalesByReducedCount = false;
  static constexpr const char *kImplementationId = "reduce_min.tensor_ar_ra";
};

template <>
struct ReduceTraits<afir::ProdOp> {
  using ApiOp = ascendc::ReduceProd2DL2Op;
  using AccumulateOp = ascendc::MulL2Op;
  static constexpr ascendc::ReduceInitType kInitType = ascendc::ReduceInitType::Prod;
  static constexpr bool kScalesByReducedCount = false;
  static constexpr const char *kImplementationId = "reduce_prod.tensor_ar_ra";
};

template <>
struct ReduceTraits<afir::AnyOp> {
  using ApiOp = ascendc::ReduceAny2DL2Op;
  using AccumulateOp = ascendc::MaxL2Op;
  static constexpr ascendc::ReduceInitType kInitType = ascendc::ReduceInitType::Any;
  static constexpr bool kScalesByReducedCount = false;
  static constexpr const char *kImplementationId = "reduce_any.tensor_ar_ra";
};

template <>
struct ReduceTraits<afir::AllOp> {
  using ApiOp = ascendc::ReduceAll2DL2Op;
  using AccumulateOp = ascendc::MinL2Op;
  static constexpr ascendc::ReduceInitType kInitType = ascendc::ReduceInitType::All;
  static constexpr bool kScalesByReducedCount = false;
  static constexpr const char *kImplementationId = "reduce_all.tensor_ar_ra";
};

// Mean 用 ReduceSum 求和，再乘完整逻辑归约域的 1/N。
// 核内跨轮累加仅在末轮缩放一次；RCore 各核产生 S_c/N，第二阶段用 Sum 合并。
template <>
struct ReduceTraits<afir::MeanOp> {
  using ApiOp = ascendc::ReduceSum2DL2Op;
  using AccumulateOp = ascendc::AddL2Op;
  static constexpr ascendc::ReduceInitType kInitType = ascendc::ReduceInitType::Sum;
  static constexpr bool kScalesByReducedCount = true;
  static constexpr const char *kImplementationId = "reduce_mean.tensor_ar_ra";
};

FailureOr<ascendc::LocalTensorType> getDestinationType(Operation *op) {
  auto destinationStyle = dyn_cast<afir::AFIRDestinationStyleInterface>(op);
  if (!destinationStyle || destinationStyle.getNumDestinations() != 1 || op->getNumOperands() < 2) {
    return failure();
  }
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(op);
  if (failed(partition)) {
    return failure();
  }
  auto type = dyn_cast<ascendc::LocalTensorType>(partition->destinations(op).front().getType());
  if (!type) {
    return failure();
  }
  return type;
}

// 先校验元素类型再计算对齐宽度，避免小于 8 位的类型使字节数为 0。
FailureOr<int64_t> getReduceAlignNum(Operation *op) {
  FailureOr<ascendc::LocalTensorType> destination = getDestinationType(op);
  if (failed(destination) || !isSupportedElementType(op, destination->getElementType())) {
    return failure();
  }
  return kBlockBytes / (destination->getElementType().getIntOrFloatBitWidth() / 8);
}

// Mean 读取 bridge 签生成的 reduction_domain，并校验绑定。
// 内存物化后 SSA 来源已不完整，不能在此重新证明归约域。
template <typename OpT>
class ReduceCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit ReduceCapabilityProvider(std::shared_ptr<const ReduceMapping> mapping) : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (!hasCanonicalDescriptors(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // 布局或 scratch 不满足完整归约计划时，报告 UnsupportedShape。
    FailureOr<afir::ReducePlan> plan = afir::buildReducePlan(operation);
    if (failed(plan)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    // 布局证明依赖元素类型决定的对齐宽度。
    FailureOr<int64_t> alignNum = getReduceAlignNum(operation);
    if (failed(alignNum)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    if (failed(checkReduceLayout(operation, *plan, *alignNum))) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    // 能力查询与下降共用完整逻辑归约域的证明。
    if constexpr (ReduceTraits<OpT>::kScalesByReducedCount) {
      if (failed(afir::getProvenReductionDomain(operation))) {
        return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
      }
    }
    auto typedOp = dyn_cast<OpT>(operation);
    switch (getReduceResidency(typedOp ? typedOp.getInput() : Value(), typedOp ? typedOp.getDestination() : Value())) {
      case ReduceResidency::Unknown:
        return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
      case ReduceResidency::NotUb:
        return CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget);
      case ReduceResidency::Ok:
        break;
    }
    ascend::FeatureState executionSupport = context.target.getExecutionSupport(mapping_->executionSpec);
    if (executionSupport == ascend::FeatureState::Unavailable) {
      return CapabilitySupport::unsupported(CapabilityReason::TargetFeatureUnavailable);
    }
    if (executionSupport == ascend::FeatureState::Unknown) {
      return CapabilitySupport::undetermined(CapabilityReason::TargetFeatureNotModeled);
    }

    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *, const ascend::instruction::InplaceQuery &,
                                 const ascend::instruction::QueryContext &) const override {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryScalarInput(Operation *, const ascend::instruction::ScalarInputQuery &,
                                     const ascend::instruction::QueryContext &) const override {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    // 归约的输出比输入少几根轴（归约轴被折掉），不是同秩的 `(1, A)` 对 `(B, A)`——形状关系
    // 不是广播，是降维。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const ReduceMapping> mapping_;
};

// 七个归约 op 通过 ReduceTraits 提供差异，共用同一 lowering pattern。
template <typename OpT>
class ReduceConversionPattern final : public OpConversionPattern<OpT> {
 public:
  using OpAdaptor = typename OpConversionPattern<OpT>::OpAdaptor;
  using Traits = ReduceTraits<OpT>;

  ReduceConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                          std::shared_ptr<const ReduceMapping> mapping)
      : OpConversionPattern<OpT>(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(OpT op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }

    FailureOr<afir::ReducePlan> plan = afir::buildReducePlan(op);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR reduce plan");
    }
    // 独立驱动 pattern 时也须在改写前验证 Mean 的完整归约域。
    if constexpr (Traits::kScalesByReducedCount) {
      std::string domainReason;
      if (failed(afir::getProvenReductionDomain(op, &domainReason))) {
        return rewriter.notifyMatchFailure(
            op, "cannot recover the full logical reduction domain for mean: " + domainReason);
      }
    }
    Value destination = adaptor.getDestination();
    ValueRange scratch = adaptor.getScratch();
    if (!destination || plan->apiScratchIndex >= scratch.size()) {
      return rewriter.notifyMatchFailure(op, "invalid AFIR reduce operand mapping");
    }
    for (Value value : op.getScratch()) {
      if (failed(afir::verifyReduceScratchScope(value, op))) {
        return failure();
      }
    }
    auto destinationType = dyn_cast<ascendc::LocalTensorType>(destination.getType());
    if (!destinationType) {
      return rewriter.notifyMatchFailure(op, "AFIR reduce destination is not a LocalTensor");
    }
    // 独立驱动 pattern 时也须先校验元素类型，避免计算对齐宽度时除零。
    if (!isSupportedElementType(op, destinationType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "unsupported element type for this AFIR reduce");
    }
    const int64_t alignNum = kBlockBytes / (destinationType.getElementType().getIntOrFloatBitWidth() / 8);
    // 独立驱动 pattern 时仍须用实际对齐宽度校验布局。
    std::string layoutReason;
    if (failed(checkReduceLayout(op, *plan, alignNum, &layoutReason))) {
      return rewriter.notifyMatchFailure(op, layoutReason);
    }

    return emitReduce(op, adaptor, rewriter, *plan, destination, scratch, destinationType, alignNum);
  }

  LogicalResult emitReduce(OpT op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter, const afir::ReducePlan &plan,
                           Value destination, ValueRange scratch, ascendc::LocalTensorType destinationType,
                           int64_t alignNum) const {
    const Location location = op.getLoc();
    FailureOr<PhysicalReduceShape> shape = materializePhysicalReduceShape(op, rewriter, plan, alignNum);
    if (failed(shape)) {
      return rewriter.notifyMatchFailure(op, "scheduled reduce is missing runtime axis extents");
    }

    Value source = adaptor.getInput();
    Value apiScratch = scratch[plan.apiScratchIndex];

    // UB 输入、destination 和 scratch 均从当前缓冲起点取用。
    // 队列 tile 偏移由 DeQue 处理，非队列 TBuf 容纳一个 tile；缺 descriptor 不可默认成 UB。
    switch (getReduceResidency(op.getInput(), op.getDestination())) {
      case ReduceResidency::Unknown:
        return rewriter.notifyMatchFailure(op, "AFIR reduce operand has no 'afir.buffer_desc' at its definition");
      case ReduceResidency::NotUb:
        return rewriter.notifyMatchFailure(op, "AFIR reduce currently requires UB-resident input and destination");
      case ReduceResidency::Ok:
        break;
    }

    // 仅 AR 需要把内侧 R 的 padding 填成单位元；RA 的内侧 padding 属于不写回的 A 槽位。
    // ReduceInit 按行主序填充，不能直接用于 RA。
    // dim_r_current 取对齐后的段长供整块填充，inner_r 取原始长度供末块掩码填充。
    if (plan.pattern == afir::ReducePattern::AR) {
      rewriter.create<ascendc::ReduceInitOp>(location, source, shape->outerActual, shape->innerFootprint,
                                             shape->innerActualFootprint, shape->innermostInnerActual,
                                             shape->innermostInnerStride, Traits::kInitType, plan.tailLast);
      rewriter.create<ascendc::PipeBarrierOp>(location, ascendc::Pipe::PIPE_V);
    }

    // 累加时先把本轮的段结果写进自己的 scratch，再与累加器合并；不累加时直接写 destination。
    Value reduceDestination = destination;
    Value accumulator;
    if (plan.isAccumulating()) {
      if (!plan.accumulationScratchIndex.has_value() || *plan.accumulationScratchIndex >= scratch.size()) {
        return rewriter.notifyMatchFailure(op, "accumulating reduce has no accumulation scratch");
      }
      accumulator = rewriter.create<ascendc::LocalTensorReinterpretCastOp>(location, destinationType,
                                                                           scratch[*plan.accumulationScratchIndex]);
      reduceDestination = accumulator;
    }

    // srcShape 按物理外维、内维排列。
    SmallVector<Value, 2> srcShape{toI32(rewriter, location, shape->outerActual),
                                   toI32(rewriter, location, shape->innerFootprint)};
    const ascendc::ReduceLayout layout =
        plan.pattern == afir::ReducePattern::AR ? ascendc::ReduceLayout::AR : ascendc::ReduceLayout::RA;
    Operation *reduceOp;
    if (std::is_same_v<OpT, afir::SumOp> && destinationType.getElementType().isSignlessInteger(32)) {
      reduceOp = rewriter.create<ascendc::ReduceSumInt32Op>(location, reduceDestination, source, apiScratch, layout,
                                                            srcShape, true, false);
    } else {
      reduceOp = rewriter.create<typename Traits::ApiOp>(
          location, reduceDestination, source, apiScratch, layout, srcShape,
          // API 要求 srcInnerPad 为 true。AR 的 R padding 已初始化，RA 的 A padding 不写回 GM。
          /*srcInnerPad=*/true, /*isReuseSource=*/false);
    }
    if (failed(setExecutionSpec(reduceOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }

    if (plan.isAccumulating()) {
      if (failed(emitAccumulation(op, rewriter, plan, destination, accumulator, shape->resultSpan, alignNum))) {
        return failure();
      }
    }

    if constexpr (Traits::kScalesByReducedCount) {
      if (failed(emitMeanScale(op, rewriter, plan, destination, destinationType, shape->resultSpan, alignNum))) {
        return failure();
      }
    }

    // End scratch use after hardware completion, including the merge that reads
    // round_result in accumulating reductions. This also protects loop-backedge reuse.
    // Apply this completion contract to every bufferized Reduce, even if view
    // canonicalization removed its scope annotations.
    rewriter.create<ascendc::PipeBarrierOp>(location, ascendc::Pipe::PIPE_V);
    rewriter.replaceOp(op, destination);
    return success();
  }

 private:
  // 全部累加轴的归纳变量为 0 时拷贝首轮结果，其余轮次按 op 的合并指令累加。
  LogicalResult emitAccumulation(OpT op, ConversionPatternRewriter &rewriter, const afir::ReducePlan &plan,
                                 Value destination, Value accumulator, Value resultSpan, int64_t alignNum) const {
    const Location location = op.getLoc();
    rewriter.create<ascendc::PipeBarrierOp>(location, ascendc::Pipe::PIPE_V);

    Value zero = rewriter.create<arith::ConstantIndexOp>(location, 0);
    Value isFirst;
    for (int64_t axisId : plan.accumulationAxes) {
      scf::ForOp loop = findAccumulationLoop(op, axisId);
      if (!loop) {
        return rewriter.notifyMatchFailure(op, "cannot find the loop of an accumulation axis");
      }
      Value atZero = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, loop.getInductionVar(), zero);
      isFirst = isFirst ? rewriter.create<arith::AndIOp>(location, isFirst, atZero).getResult() : atZero;
    }
    if (!isFirst) {
      return rewriter.notifyMatchFailure(op, "accumulating reduce has no accumulation axis");
    }

    // 合并覆盖 A 槽位的 resultSpan，并按 32B 对齐，不能用 A 元素数替代跨步槽位跨度。
    // 对齐后仍受 nominal 缓冲容量约束；跨度内的补白槽位不写回 GM。
    Value count = toI32(rewriter, location, alignUp(rewriter, location, resultSpan, alignNum));

    auto ifOp = rewriter.create<scf::IfOp>(location, isFirst, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      rewriter.create<ascendc::DataCopyL2Op>(location, destination, accumulator, count);
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      auto mergeOp =
          rewriter.create<typename Traits::AccumulateOp>(location, destination, destination, accumulator, count);
      if (failed(setExecutionSpec(mergeOp, mapping_->executionSpec))) {
        return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
      }
    }
    return success();
  }

  // 所有累加循环满足 upperBound - iv <= step 时为末轮；非累加返回空 Value。
  // 直接使用 scf.for 的上界和正字面量步长，避免 iv + step 溢出及重复物化边界。
  FailureOr<Value> buildAtLastRound(OpT op, ConversionPatternRewriter &rewriter, const afir::ReducePlan &plan) const {
    if (!plan.isAccumulating()) {
      return Value();
    }
    const Location location = op.getLoc();
    Value isLast;
    for (int64_t axisId : plan.accumulationAxes) {
      scf::ForOp loop = findAccumulationLoop(op, axisId);
      if (!loop) {
        return rewriter.notifyMatchFailure(op, "cannot find the loop of an accumulation axis");
      }
      // 步长必须是正的字面量。
      APInt step;
      if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isStrictlyPositive()) {
        return rewriter.notifyMatchFailure(op, "an accumulation loop has a non-positive or non-constant step");
      }
      Value remaining = rewriter.create<arith::SubIOp>(location, loop.getUpperBound(), loop.getInductionVar());
      Value atLast = rewriter.create<arith::CmpIOp>(location, arith::CmpIPredicate::sle, remaining, loop.getStep());
      isLast = isLast ? rewriter.create<arith::AndIOp>(location, isLast, atLast).getResult() : atLast;
    }
    if (!isLast) {
      return rewriter.notifyMatchFailure(op, "accumulating reduce has no accumulation axis");
    }
    return isLast;
  }

  // 求和完成后乘 1/N，N 为完整逻辑归约域的 Original 轴尺寸之积。
  // 非累加时执行一次，跨轮时在末轮累加之后执行。
  // 先求和再缩放可能使 f32 中间和溢出，即使真实均值可表示。
  LogicalResult emitMeanScale(OpT op, ConversionPatternRewriter &rewriter, const afir::ReducePlan &plan,
                              Value destination, ascendc::LocalTensorType destinationType, Value resultSpan,
                              int64_t alignNum) const {
    const Location location = op.getLoc();
    auto elementType = dyn_cast<FloatType>(destinationType.getElementType());
    if (!elementType) {
      return rewriter.notifyMatchFailure(op, "AFIR reduce mean requires a floating-point destination");
    }
    std::string domainReason;
    FailureOr<afir::LogicalReductionDomain> domain = afir::getProvenReductionDomain(op, &domainReason);
    if (failed(domain)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot recover the full logical reduction domain for mean: " + domainReason);
    }
    FailureOr<Value> atLastRound = buildAtLastRound(op, rewriter, plan);
    if (failed(atLastRound)) {
      return failure();
    }

    SmallVector<Value, 4> extents;
    for (afir::ExprAttr size : domain->sizes) {
      FailureOr<Value> extent = afir::materializeExtentValue(op, rewriter, size);
      if (failed(extent)) {
        return rewriter.notifyMatchFailure(op, "an original reduction axis size cannot be materialized");
      }
      extents.push_back(*extent);
    }
    // N 经 index -> i64 -> 浮点转换，避免 i32 截断；乘法溢出由 multiplyAll 检查。
    Value total = multiplyAll(rewriter, location, extents);
    Value totalI64 = rewriter.create<arith::IndexCastOp>(location, rewriter.getI64Type(), total);
    Value denominator = rewriter.create<arith::SIToFPOp>(location, elementType, totalI64);
    Value one = rewriter.create<arith::ConstantOp>(location, elementType, rewriter.getFloatAttr(elementType, 1.0));
    Value scale = rewriter.create<arith::DivFOp>(location, one, denominator);

    // 缩放覆盖按 32B 对齐的 A 槽位跨度；补白槽位不写回 GM。
    Value count = toI32(rewriter, location, alignUp(rewriter, location, resultSpan, alignNum));

    // 非累加形态直接缩放，无需条件分支。
    std::optional<OpBuilder::InsertionGuard> guard;
    if (*atLastRound) {
      auto ifOp = rewriter.create<scf::IfOp>(location, *atLastRound, /*withElseRegion=*/false);
      guard.emplace(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
    }

    // 屏障与 Muls 同处末轮分支，保证前面的 Add/DataCopy 完成后再缩放。
    rewriter.create<ascendc::PipeBarrierOp>(location, ascendc::Pipe::PIPE_V);
    auto scaleOp = rewriter.create<ascendc::MulsL2Op>(location, destination, destination, scale, count);
    if (failed(setExecutionSpec(scaleOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    return success();
  }

  ImplementationKey key_;
  std::shared_ptr<const ReduceMapping> mapping_;
};

template <typename OpT>
ImplementationRegistry::PatternRegistration<ReduceMapping> createReducePatternRegistration() {
  return [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
            std::shared_ptr<const ReduceMapping> mapping) {
    patterns.add<ReduceConversionPattern<OpT>>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
  };
}

template <typename OpT>
LogicalResult registerOneReduceImplementation(ImplementationRegistry &registry) {
  return registry.registerImplementation<OpT, ReduceMapping, ReduceCapabilityProvider<OpT>>(
      ascend::instruction::ImplementationKind::AscendC,
      ascend::instruction::ImplementationId{ReduceTraits<OpT>::kImplementationId}, ReduceMapping{},
      createReducePatternRegistration<OpT>());
}

}  // namespace

LogicalResult registerReduceImplementations(ImplementationRegistry &registry,
                                            ascend::instruction::SelectionPolicyRegistry *) {
  if (failed(registerOneReduceImplementation<afir::SumOp>(registry)) ||
      failed(registerOneReduceImplementation<afir::MaxOp>(registry)) ||
      failed(registerOneReduceImplementation<afir::MinOp>(registry)) ||
      failed(registerOneReduceImplementation<afir::MeanOp>(registry)) ||
      failed(registerOneReduceImplementation<afir::ProdOp>(registry)) ||
      failed(registerOneReduceImplementation<afir::AnyOp>(registry)) ||
      failed(registerOneReduceImplementation<afir::AllOp>(registry))) {
    return failure();
  }
  return success();
}

}  // namespace mlir::ascend::kernel
