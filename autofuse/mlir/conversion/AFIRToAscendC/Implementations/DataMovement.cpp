/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/Implementations/DataMovement.h"
#include "target/Ascend/AscendTargetProfile.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {
constexpr size_t kTwoDimensions = 2;

using ascend::instruction::CapabilityReason;
using ascend::instruction::CapabilitySupport;
using ascend::instruction::ImplementationKey;
using ascend::instruction::ImplementationRegistry;
using ascend::instruction::PatternBuildContext;

struct DataMovementMapping {
  bool load = false;
};

bool hasCompactRows(const TransferPlan &plan) {
  return plan.kind == TransferPlanKind::Scheduled && plan.dims.size() >= 2 &&
         proveEqual(plan.dims[plan.dims.size() - 2].localStride, plan.dims.back().extent) ==
             afir::detail::TruthValue::True;
}

bool hasCanonicalDescriptors(Operation *op, bool load) {
  if (!op->hasAttr(afir::kOperandDescsAttrName) || !op->hasAttr(afir::kDestinationDescsAttrName)) {
    return false;
  }
  if (!load) {
    return true;
  }
  return op->hasAttr(afir::kResultDescsAttrName);
}

bool hasExpectedTensorTypes(Operation *op, bool load) {
  if (op->getNumOperands() != 2 || (load && op->getNumResults() != 1) || (!load && op->getNumResults() != 0)) {
    return false;
  }
  Type source = op->getOperand(0).getType();
  Type destination = op->getOperand(1).getType();
  if (load) {
    return isa<ascendc::GlobalTensorType>(source) && isa<ascendc::LocalTensorType>(destination) &&
           op->getResult(0).getType() == destination;
  }
  return isa<ascendc::LocalTensorType>(source) && isa<ascendc::GlobalTensorType>(destination);
}

bool hasTwoIndexingMaps(Operation *op) {
  auto maps = op->getAttrOfType<ArrayAttr>(afir::kIndexingMapsAttrName);
  return maps && maps.size() == kTwoDimensions;
}

class DataMovementCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit DataMovementCapabilityProvider(std::shared_ptr<const DataMovementMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (!hasCanonicalDescriptors(operation, mapping_->load) || !hasTwoIndexingMaps(operation) ||
        !hasExpectedTensorTypes(operation, mapping_->load)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // 建不出计划一律报 `MissingInformation`：寻址这一侧没有形状天花板，失败说的只有
    // 「这条搬运本该走调度路径，而它的标注是破的」。不走调度路径的那一档不会失败，
    // `buildTransferPlan` 对它返回 Identity 计划。段几何防御判据（负跨度/不整除）同桶：
    // 调度器产出的 desc 恒满足约束，违反就是标注破了——违例形态会一路发到
    // `DataCopyPadExtend`，零报错、结果错。
    FailureOr<TransferPlan> plan = buildTransferPlan(operation);
    const bool regBased =
        context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) == ascend::FeatureState::Available;
    if (failed(plan) || (!(regBased && hasCompactRows(*plan)) && !blockDmaSegmentGeometryIsSound(operation, *plan))) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    auto source = dyn_cast<ShapedType>(operation->getOperand(0).getType());
    auto destination = dyn_cast<ShapedType>(operation->getOperand(1).getType());
    if (!source || !destination || source.getElementType() != destination.getElementType()) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }

    // 搬运的两端就是它的两个操作数，空间从各自的定义点取。读实际携带的空间而不是
    // 硬编码 GM<->UB，是因为 `ascendc::LocalTensorType` 同时覆盖 L1/L0 与 UB：写死
    // 一对会给另一条搬运边悄悄答错。
    if (operation->getNumOperands() != 2) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    auto sourceSpace = getValueMemorySpace(operation->getOperand(0));
    auto destinationSpace = getValueMemorySpace(operation->getOperand(1));
    if (failed(sourceSpace) || failed(destinationSpace)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (!context.target.hasDirectTransfer(*sourceSpace, *destinationSpace)) {
      return CapabilitySupport::unsupported(CapabilityReason::TargetFeatureUnavailable);
    }
    return CapabilitySupport::supported();
  }

 private:
  std::shared_ptr<const DataMovementMapping> mapping_;
};

// 一次搬运的操作数。`Scheduled` 会把源与目的换成带运行期偏移的子视图，并填上 DMA 的四个
// 参数；identity 只填 `blockLen`，其余保持一维的退化值。
//
// 缝隙按**空间**存（global / local），方向映射在 `emitTransfer` 里做。理由不只是
// 「计划拿不到方向」：`DataCopyPadExtend` 的两个重载单位换算不同——GM 侧那个参数乘
// `sizeof(T)`、UB 侧那个除 `align_num`（`autofuse/ascendc/api/datacopy.h`）——按 src/dst
// 存等于把量纲一起交给猜测，接反了不只是位置错。
struct TransferOperands {
  Value source;
  Value destination;
  Value blockCount;
  Value blockLen;
  Value globalStride;
  Value localStride;
};

template <typename SourceOp>
class DataMovementConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  DataMovementConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                std::shared_ptr<const DataMovementMapping> mapping, bool regBased)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        key_(std::move(key)),
        mapping_(std::move(mapping)),
        regBased_(regBased) {}

  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    FailureOr<TransferPlan> plan = buildTransferPlan(op);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR transfer plan");
    }
    const bool compact = regBased_ && hasCompactRows(*plan);
    if (!compact && !blockDmaSegmentGeometryIsSound(op, *plan)) {
      return rewriter.notifyMatchFailure(op, "segment geometry is not expressible as a block DMA");
    }

    const TransferOperands buffers{
        adaptor.getOperands()[0], adaptor.getOperands()[1], Value(), Value(), Value(), Value()};
    if (plan->kind == TransferPlanKind::Scheduled) {
      // 计划多于两项时发射器会先建外层 `scf.for`，回调在最内层执行——DMA 与它的两个子
      // 视图因此都落在循环体里。
      LogicalResult emitted = emitScheduledTransfer(
          op, rewriter, *plan,
          [&](OpBuilder &builder, const ScheduledTileIndices &indices) {
            TransferOperands tile = buffers;
            applyScheduledTileOffsets(op, builder, indices, tile);
            emitTransfer(op, builder, tile, compact);
            return success();
          },
          compact);
      if (failed(emitted)) {
        return rewriter.notifyMatchFailure(op, "cannot emit the scheduled transfer");
      }
    } else {
      TransferOperands whole = buffers;
      whole.blockLen =
          mapping_->load
              ? rewriter.create<ascendc::LocalTensorGetSizeOp>(op.getLoc(), rewriter.getI32Type(), whole.destination)
              : rewriter.create<ascendc::LocalTensorGetSizeOp>(op.getLoc(), rewriter.getI32Type(), whole.source);
      emitTransfer(op, rewriter, whole);
    }

    if (mapping_->load) {
      // 替换值是**整块**缓冲，不是循环体里那个带行内偏移的子视图：Load 的结果代表整个
      // tile，而子视图只是本轮 DMA 的落点，它在循环里，外面的使用者够不着（直接违反支配
      // 关系）。两项及以内没有循环，两者取值相同——`LocalTensorSubIndexOp` 在下标为 0 时
      // 本来就折叠回张量自身。
      rewriter.replaceOp(op, buffers.destination);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

 private:
  // 把源与目的换成带偏移的子视图，count 换成该 tile 的长度。
  //
  // GM 侧的偏移是「tile 起点 + 本轮行内偏移」；本地侧只有行内偏移——tile 在缓冲里的起点
  // 恒为基址（队列按 slot 自己轮转，TBuf 按 `b{id}_size` 只装一个 tile、每个迭代整块重写，
  // legacy 两处都是每趟下标 0）。没有外层循环时本地侧那一项是常量 0，与改动前一致。
  void applyScheduledTileOffsets(SourceOp op, OpBuilder &builder, const ScheduledTileIndices &indices,
                                 TransferOperands &operands) const {
    if (mapping_->load) {
      operands.source = builder.create<ascendc::GlobalTensorSubIndexOp>(op.getLoc(), operands.source.getType(),
                                                                        operands.source, indices.globalOffset);
      operands.destination = builder.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), operands.destination.getType(),
                                                                            operands.destination, indices.localOffset);
    } else {
      operands.source = builder.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), operands.source.getType(),
                                                                       operands.source, indices.localOffset);
      operands.destination = builder.create<ascendc::GlobalTensorSubIndexOp>(
          op.getLoc(), operands.destination.getType(), operands.destination, indices.globalOffset);
    }
    operands.blockCount = indices.blockCount;
    operands.blockLen = indices.blockLen;
    operands.globalStride = indices.globalStride;
    operands.localStride = indices.localStride;
  }

  // 一律发 `DataCopyPadExtend`，与 legacy 的 `load_api_call.cpp` / `store_api_call.cpp` 一致：
  // 那两处也没有第二条路。**不能按 count 对齐与否分流**——tile 的元素个数是运行期的
  // TilingData 字段（尾块与整块走同一条语句），lowering 这一刻看不出它对不对齐；而裸
  // `AscendC::DataCopy` 要求 `count * sizeof(T)` 32B 对齐，不对齐时 NPU 上不报错、
  // 按 32B 向下取整，尾巴静默丢掉。
  void emitTransfer(SourceOp op, OpBuilder &builder, const TransferOperands &operands, bool compact = false) const {
    Value blockCount = operands.blockCount;
    if (!blockCount) {
      blockCount = builder.create<arith::ConstantOp>(op.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1));
    }
    Value globalStride = operands.globalStride;
    Value localStride = operands.localStride;
    if (!globalStride || !localStride) {
      Value zeroStride =
          builder.create<arith::ConstantOp>(op.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
      globalStride = zeroStride;
      localStride = zeroStride;
    }
    // 方向映射，这一行是全流程里唯一知道方向的地方。
    //
    // Load 是 `LocalTensor ← GlobalTensor`：src 是 GM、dst 是 UB。Store 反过来。
    // 与 legacy `SetDmaParams` 的 `src = copy_in ? gm : ub` 逐字对应。写反的表现是
    // 二维 Store 把 UB 的行距当成 GM 的——位置错，而且因为两个重载的单位换算相反
    // （GM 乘 `sizeof(T)`、UB 除 `align_num`），量纲也跟着错。
    Value sourceStride = mapping_->load ? globalStride : localStride;
    Value destinationStride = mapping_->load ? localStride : globalStride;
    if (regBased_) {
      builder.create<ascendc::DataCopyPadExtendRegBaseOp>(op.getLoc(), operands.destination, operands.source,
                                                          blockCount, operands.blockLen, sourceStride,
                                                          destinationStride, builder.getBoolAttr(compact));
      return;
    }
    builder.create<ascendc::DataCopyPadExtendOp>(op.getLoc(), operands.destination, operands.source, blockCount,
                                                 operands.blockLen, sourceStride, destinationStride);
  }

  ImplementationKey key_;
  std::shared_ptr<const DataMovementMapping> mapping_;
  bool regBased_;
};

template <typename SourceOp>
ImplementationRegistry::PatternRegistration<DataMovementMapping> createPatternRegistration() {
  return [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
            std::shared_ptr<const DataMovementMapping> mapping) {
    patterns.add<DataMovementConversionPattern<SourceOp>>(
        context.typeConverter, patterns.getContext(), key, std::move(mapping),
        context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) == ascend::FeatureState::Available);
  };
}

template <typename SourceOp>
LogicalResult registerDataMovement(ImplementationRegistry &registry, bool load, StringRef id) {
  return registry.registerImplementation<SourceOp, DataMovementMapping, DataMovementCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      DataMovementMapping{load}, createPatternRegistration<SourceOp>());
}

}  // namespace

LogicalResult registerDataMovementImplementations(ascend::instruction::ImplementationRegistry &registry,
                                                  ascend::instruction::SelectionPolicyRegistry *) {
  if (failed(registerDataMovement<LoadOp>(registry, true, "load.gm_ub")) ||
      failed(registerDataMovement<StoreOp>(registry, false, "store.ub_gm"))) {
    return failure();
  }
  return success();
}

}  // namespace mlir::ascend::kernel
