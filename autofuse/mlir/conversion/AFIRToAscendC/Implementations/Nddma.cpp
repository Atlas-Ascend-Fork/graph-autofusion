/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Nddma.cpp - AFIR nddma lowering --------------------------*- C++ -*-===//
//
// `afir.nddma` 的 lowering。**一档，只在 reg 侧**：生成 `DataCopyNddma<T, dim>`
// （`v35/ascendc/api_regbase/datacopy_nddma.h`），legacy 的 `NddmaApiCall` 逐字生成的也是它
// （`v35/codegen/reg_api_call/reg_nddma_api_call.cpp`）。mem 侧那张支持表上没有这个包装，
// 所以那一侧这个 op 一律报不支持、整张图回落 legacy。
//
// 它与 `DataMovement.cpp` 的 Load 是同一条搬运边的两种落法，差别只有一处：那边一次
// 搬一段连续的元素（`DataCopyPadExtend`，`localTileIsSingleSegment` 把多段挡在外面），
// 这边把合轴之后的每一段各交一维给 ND-DMA 硬件，所以多段恰好是它存在的理由。
//
// 三组尺寸与维的对应（外→内，与包装里 `for (i < dim)` 反序填 `MultiCopyLoopInfo` 的次序对上）：
//
//   - `output_dims[i]`   —— 第 i 维的元素数，`materializeSegmentExtent`；
//   - `output_stride[i]` —— 第 i 维在片上张量上的步长；
//   - `input_stride[i]`  —— 第 i 维在 GM 张量上的步长。
//
// 「分成哪几维」不等于 `collapseVectorizedAxes` 分出来的段，这是本文件唯一的设计内容。
// 那边按 descriptor 的 role 过滤，只收片上侧的张量（GM 侧 role 是 `input`、一根向量化轴都
// 没有），于是它合段时只证了片上连续。ND-DMA 的一维要两侧都连续，所以这里在它的段之上
// 再切一刀（`classifyNddmaDims`），判据与合轴同一个函数（`afir::provablyContiguous`）。
// 不切的话最基本的广播就会算错——反例写在那个函数的注释里。
//
// GM 侧的步长还得逐轴问 `afir::globalAxisUnitStride`：那一列 `VectorSegment::strides` 里
// 根本没有，而且键是原始轴、要穿过图级合轴与分块才对得上调度轴。那套换算
// `TransferPlan.cpp` 里已经有一份，按「一条判据只写一份」开成了接口，没有在这里重算一遍。
//
// 不像 legacy 那样恒生成 5 维再左补 1：`dim` 是模板实参、由数组长度推出来，补出来的那些 1
// 只让包装多转几圈空循环。维数超过包装能收的上限时报不支持（见 `kNddmaMaxDims`），
// 由 legacy 按它的 `CreateNddmaCall` 拆外层循环——那条路这里没有，拆循环是独立一件事。
//
// 本文件的 stable id 与注释都不出现芯片名。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/Nddma.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <memory>
#include <optional>
#include <utility>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
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

// 包装一次调用能收的维数上限，也就是 `MultiCopyLoopInfo<dim>` 里那个 `dim`。legacy 查的是
// 同一个数（`v35/codegen/reg_api_call/reg_nddma_api_call.cpp` 的 `kNddmaMaxLen`，超了就走 `CreateNddmaCall` 生成外层
// 循环），数的也是同一个量：它的 `param.repeats` 是按两侧 stride 一起合出来的维，与这里
// 分出来的维对应。
constexpr size_t kNddmaMaxDims = 5;

struct NddmaMapping {
  ApiFamily family = ApiFamily::RegBased;
};

bool hasCanonicalDescriptors(Operation *op) {
  return op->hasAttr(afir::kOperandDescsAttrName) && op->hasAttr(afir::kDestinationDescsAttrName) &&
         op->hasAttr(afir::kResultDescsAttrName);
}

bool hasTwoIndexingMaps(Operation *op) {
  auto maps = op->getAttrOfType<ArrayAttr>(afir::kIndexingMapsAttrName);
  return maps && maps.size() == 2;
}

// 两端的类型。与 `DataMovement.cpp` 的 Load 那一支逐字相同：源是 GM、落点是片上，结果就是
// 落点（destination-style）。不查元素数——一边搬一边广播，两端元素数本来就不同，那是
// 这个 op 与 Load 的唯一形态差别（`afir.nddma` 的 verifier 处有说明）。
bool hasExpectedTensorTypes(Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
    return false;
  }
  Type source = op->getOperand(0).getType();
  Type destination = op->getOperand(1).getType();
  return isa<ascendc::GlobalTensorType>(source) && isa<ascendc::LocalTensorType>(destination) &&
         op->getResult(0).getType() == destination;
}

// 这个 op 的 `strides` 与 `vectorized_strides` 说的不是同一件事，不要拿它们互相校验。
//
// 搬运节点的 descriptor 把两侧合在一份里：`strides` 是它自己的 GM 侧布局，
// `vectorized_strides` 是片上那一侧。`afir.nddma` 的两条来路都是这样——
//
//   - `NddmaTemplate::GenNddmaNode`（`v35/optimize/template/nddma_template.cpp:303`）把
//     Broadcast 的 `repeats` 与 `vectorized_strides` 搬到 Load 节点上，有意不动 `strides`：
//     legacy 的 `NddmaApiCall` 正是从这一列取 `input_stride` 与 GM 偏移
//     （`v35/codegen/reg_api_call/reg_nddma_api_call.cpp:74/76` →
//     `codegen/api_call/utils/api_call_utils.cpp:116` 的 `gm_tensor.axis_strides`），
//     该约定写在 `nddma_template.cpp:858` 的注释里；
//   - `GenLoadToGenNddmaNode`（`v35/optimize/un_alignment_strategy.cpp:261`）只改节点类型、
//     一个字段都不抄，它的两列本来就不等——不等正是它要转 nddma 的原因。
//
// 实测（`[1024,1] → [1024,128]`，`template2`）：
//
//   strides = [(z0Tb_size * z0t_size), z0t_size, 1, 0]   ← 输入 `[1024,1]` 的 GM 布局
//   vectorized_strides = [128, 1]                        ← 片上这块 tile
//
// 曾经这里有一条 `resultDescriptorIsSelfConsistent`，要求两列在向量化轴上相等，把这一类
// 整个判成「上游给错了」。它答错的是归属：真正出问题的是下游 `afir.store` 的 GM 偏移
// 回落到这一列（见 `TransferPlan.cpp` 的 `localStridesAreLayoutProxy`），而不是本档的输入
// 不可信。两列的含义各自清楚，本档要的 GM 步长逐轴问 `afir::globalAxisUnitStride`、片上步长
// 从 `collapseVectorizedAxes` 的段里拿，谁也不需要那条等式。

// ND-DMA 的一维：段内片上与 GM 两侧都连续，所以整段等价于一根轴。
struct NddmaDim {
  // 本维覆盖的调度轴与各自的 extent，外→内。借 `VectorSegment` 装是为了直接喂给
  // `materializeSegmentExtent`——尾块修正那条判据只写在它里面。**只填 `axes` 与
  // `axisExtents`**，那两个就是它读的全部；`extent` 与 `strides` 留空，它们是合轴那一层
  // 的产物，在这一层已经被下面两个字段取代。
  afir::VectorSegment extent;
  // 本维最内那根轴上的步长，两侧各一个。取最内轴而不是最外轴：这个量的用途是
  // 「本维前进一格，地址前进多少」，而维内各轴已证连续，所以整维等价于一根 stride 为最内轴
  // stride 的轴（与 `VectorSegment::strides` 同一条约定）。
  afir::ExprAttr localStride;
  afir::ExprAttr globalStride;
};

// 一个 `afir.nddma` 落成一条 `DataCopyNddma` 的形态：它的那几维。
//
// 分维要问两侧，而 `collapseVectorizedAxes` 只问得到片上那一侧。 那边合出来的段里，
// 片上连续是证过的；GM 侧没有参与判据，所以同一段里 GM 可能并不连续。最直接的反例就是最
// 基本的广播 `[8,1] → [8,128]`：片上 `stride(z0) = 128 == 1 × 128 = stride(z1) × extent(z1)`
// 成立、两轴并成一段，而 GM 侧 `stride(z0) = 1`、`stride(z1) = 0`，`1 != 0 × 128`。那一段
// 的 GM 位移不是「段内下标 × 某个步长」，拿一个 `input_stride` 去代表它就是静默搬错。
//
// 所以这里在片上的段之上再切一刀：段内逐对查 GM 连续性，不连续处断开。方向是多切，
// 与合轴同一个保守方向——多一维的后果是包装多转一层循环，少一维的后果是算错地址。两段之间
// 从不合并：那里片上就不连续。
//
// 维内各轴的片上单位步长不必再求：片上在段内是连续的，所以
// `localUnit(a_j) = segment.strides[片上] × Π extent(a_m), m > j`，由内往外累乘即得。
// GM 侧那一列只能逐轴问 `afir::globalAxisUnitStride`（见该处注释：方向与 `axisUnitStride` 相反）。
//
// 收的边界（不满足一律 `failure()`，调用方翻成 unsupported、整图回落 legacy）：
//
//   - 参与合轴判据的张量恰好一个，而且是片上那一侧。这不是在挑一列，是在确认
//     `collapseVectorizedAxes` 的 role 过滤确实只留下了片上侧：留下两个说明这个 op 身上
//     多了一个带向量化轴的 compute 张量，那时按下标取片上步长会取到别人那一列。
//   - 切完之后的维数在 `[1, kNddmaMaxDims]` 内。
//   - 每根轴的 GM 单位步长都求得出来。
//
// 不要求有广播维。 `afir.nddma` 有两条来路：广播融进搬入（`NddmaTemplate`）与尾轴不连续
// 的搬入（`LoadToNddmaTemplate`），后者一个广播维都没有，生成的是同一条包装。
FailureOr<SmallVector<NddmaDim>> classifyNddmaDims(Operation *op, const afir::VectorPlan &plan) {
  if (plan.tensors.size() != 1 || plan.tensors.front().side != afir::VectorTensor::Side::Destination) {
    return failure();
  }
  MLIRContext *context = op->getContext();
  if (plan.segments.empty()) {
    return failure();
  }

  SmallVector<NddmaDim> dims;
  for (const afir::VectorSegment &segment : plan.segments) {
    const size_t count = segment.axes.size();
    if (count == 0 || count != segment.axisExtents.size() || segment.strides.size() != 1) {
      return failure();
    }
    // 维内各轴在两侧的单位步长，与 `segment.axes` 一一对应。
    SmallVector<afir::ExprAttr> localUnits(count);
    SmallVector<afir::ExprAttr> globalUnits(count);
    afir::ExprAttr running = segment.strides.front();
    for (size_t position = count; position-- > 0;) {
      localUnits[position] = running;
      if (position > 0) {
        FailureOr<afir::ExprAttr> next = afir::multiplyExprs(context, running, segment.axisExtents[position]);
        if (failed(next)) {
          return failure();
        }
        running = *next;
      }
      FailureOr<afir::ExprAttr> global = afir::globalAxisUnitStride(op, segment.axes[position]);
      if (failed(global)) {
        return failure();
      }
      globalUnits[position] = *global;
    }

    // 从最外那根轴开段，GM 侧一断就收一维。段内片上连续是证过的，所以只查 GM。
    size_t dimStart = 0;
    for (size_t position = 1; position <= count; ++position) {
      const bool breakHere =
          position == count || !afir::provablyContiguous(context, globalUnits[position - 1], globalUnits[position],
                                                         segment.axisExtents[position]);
      if (!breakHere) {
        continue;
      }
      NddmaDim dim;
      dim.extent.axes.assign(segment.axes.begin() + dimStart, segment.axes.begin() + position);
      dim.extent.axisExtents.assign(segment.axisExtents.begin() + dimStart, segment.axisExtents.begin() + position);
      dim.localStride = localUnits[position - 1];
      dim.globalStride = globalUnits[position - 1];
      dims.push_back(std::move(dim));
      dimStart = position;
    }
  }
  if (dims.empty() || dims.size() > kNddmaMaxDims) {
    return failure();
  }
  return dims;
}

// capability 与 pattern 问的是同一件事，各算一遍。不缓存在 op 上：两次之间 IR 不变，
// 而缓存意味着再加一条「这份缓存还新鲜吗」的判据，那条判据答错时的后果比多算一遍重得多
// （理由同 `Broadcast.cpp` 的 `analyzeExtendBroadcast`）。
FailureOr<SmallVector<NddmaDim>> analyzeNddma(Operation *op, afir::VectorPlan &plan) {
  FailureOr<afir::VectorPlan> collapsed = afir::collapseVectorizedAxes(op);
  if (failed(collapsed)) {
    return failure();
  }
  plan = std::move(*collapsed);
  return classifyNddmaDims(op, plan);
}

class NddmaCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit NddmaCapabilityProvider(std::shared_ptr<const NddmaMapping> mapping) : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (!hasCanonicalDescriptors(operation) || !hasTwoIndexingMaps(operation) || !hasExpectedTensorTypes(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // 这个包装只在 reg-base 那张支持表上。profile 没描述这个 feature 时停在 undetermined，
    // 不默认倒向任何一边——猜「有」的后果是 mem 侧的 kernel 里内联一份它找不到的头。
    if (std::optional<CapabilitySupport> family = elementwise::checkApiFamily(mapping_->family, context)) {
      return *family;
    }

    auto source = dyn_cast<ShapedType>(operation->getOperand(0).getType());
    auto destination = dyn_cast<ShapedType>(operation->getOperand(1).getType());
    if (!source || !destination || source.getElementType() != destination.getElementType()) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    // 不按元素类型集合过滤，与 `DataMovement.cpp` 同一条：`DataCopyNddma<T, dim>` 是
    // 个模板，往下落到 `AscendC::DataCopy<T, dim, config>`，收什么由那一层说，这一层
    // 没有证据能写出一个更窄的集合。两端同 dtype 由 ODS 之外再查一遍，因为搬运的两端
    // 类型是两个独立的操作数。

    // 搬运的两端就是它的两个操作数，空间从各自的定义点取——理由同 `DataMovement.cpp`：
    // `ascendc::LocalTensorType` 同时覆盖 L1/L0 与 UB，写死一对会给另一条搬运边答错。
    auto sourceSpace = getValueMemorySpace(operation->getOperand(0));
    auto destinationSpace = getValueMemorySpace(operation->getOperand(1));
    if (failed(sourceSpace) || failed(destinationSpace)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (!context.target.hasDirectTransfer(*sourceSpace, *destinationSpace)) {
      return CapabilitySupport::unsupported(CapabilityReason::TargetFeatureUnavailable);
    }

    // 三组尺寸与 GM 偏移都要调度给的那一套（循环轴、分块轴、切分轴），所以只收
    // `Scheduled`。identity 那一档没有循环、也没有 tile，`globalAxisUnitStride`
    // 与 `materializeScheduledGlobalOffset` 都答不了。
    FailureOr<afir::TransferPlan> plan = buildTransferPlan(operation);
    if (failed(plan)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (plan->kind != afir::TransferPlanKind::Scheduled) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    afir::VectorPlan vectorPlan;
    if (failed(analyzeNddma(operation, vectorPlan))) {
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
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const NddmaMapping> mapping_;
};

class NddmaConversionPattern final : public OpConversionPattern<afir::NddmaOp> {
 public:
  NddmaConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                         std::shared_ptr<const NddmaMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(afir::NddmaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form nddma requires a destination");
    }
    FailureOr<TransferPlan> plan = buildTransferPlan(op);
    if (failed(plan) || plan->kind != TransferPlanKind::Scheduled) {
      return rewriter.notifyMatchFailure(op, "cannot build a scheduled AFIR transfer plan");
    }
    afir::VectorPlan vectorPlan;
    FailureOr<SmallVector<NddmaDim>> dims = analyzeNddma(op, vectorPlan);
    if (failed(dims)) {
      return rewriter.notifyMatchFailure(op, "not a supported nddma segmentation");
    }

    // 尾长只有最外那一维用得上，切分轴不落在它里面时连算都不算——算了就是一条没有
    // 使用者的 `afir.symbolic_eval` 留在函数里。「切分轴恒在最外」是上游的不变量
    // （`getTileInnerAxis` 取局部 descriptor `vectorized_axis` 的 `front()`，合轴与这里的
    // 再切分都保持次序），这里跟 `Broadcast.cpp` 的 reg 侧一档用同一条判据。
    Value tailAwareSize;
    if (llvm::is_contained(dims->front().extent.axes, plan->tileInnerAxis)) {
      FailureOr<Value> size = afir::materializeTailAwareTileSize(op, rewriter, *plan);
      if (failed(size)) {
        return rewriter.notifyMatchFailure(op, "cannot materialize the tail-aware tile size");
      }
      tailAwareSize = *size;
    }

    SmallVector<Value> outputDims;
    SmallVector<Value> outputStride;
    SmallVector<Value> inputStride;
    outputDims.reserve(dims->size());
    outputStride.reserve(dims->size());
    inputStride.reserve(dims->size());
    for (auto [index, dim] : llvm::enumerate(*dims)) {
      // 尾块修正只给最外那一维。 其余维的 extent 在这里只当趟数用，而最外之外不含
      // 切分轴，所以 `materializeSegmentExtent` 对它们本来就取不到那一项——传与不传生成
      // 的 Op 一样。写成 `index == 0` 是为了与 `Broadcast.cpp` 的
      // `materializeExtendShapeEntry` 保持同一个写法，那边放宽上游不变量时两处要一起补。
      FailureOr<Value> extent = afir::materializeSegmentExtent(op, rewriter, dim.extent, plan->tileInnerAxis,
                                                               index == 0 ? tailAwareSize : Value{});
      FailureOr<Value> local = afir::materializeStrideValue(op, rewriter, dim.localStride);
      FailureOr<Value> global = afir::materializeStrideValue(op, rewriter, dim.globalStride);
      if (failed(extent) || failed(local) || failed(global)) {
        return rewriter.notifyMatchFailure(op, "cannot materialize an nddma dimension extent or stride");
      }
      outputDims.push_back(*extent);
      outputStride.push_back(*local);
      inputStride.push_back(*global);
    }

    // GM 侧这一趟从哪里开始读。段之内与段之间的位移都由上面三组尺寸承担，留给调用方的
    // 只有 tile 基址。本地侧的偏移恒为 0——队列按 slot 自己轮转，TBuf 按 `b{id}_size`
    // 只装一个 tile（与 `DataMovement.cpp` 同一条）。
    FailureOr<Value> globalOffset = afir::materializeScheduledGlobalOffset(op, rewriter, *plan);
    if (failed(globalOffset)) {
      return rewriter.notifyMatchFailure(op, "scheduled nddma is missing its global tile offset");
    }
    Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value source = rewriter.create<ascendc::GlobalTensorSubIndexOp>(op.getLoc(), adaptor.getInput().getType(),
                                                                    adaptor.getInput(), *globalOffset);
    Value destination = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), adaptor.getDestination().getType(),
                                                                        adaptor.getDestination(), zeroOffset);

    rewriter.create<ascendc::DataCopyNddmaRegBaseOp>(op.getLoc(), destination, source, outputDims, outputStride,
                                                     inputStride);
    // 结果是整块目的地 tile，不是偏移过的那个值：下游的 `afir.store` 要的是整块
    // （与 `Broadcast.cpp` 两档同一个理由）。
    rewriter.replaceOp(op, adaptor.getDestination());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const NddmaMapping> mapping_;
};

}  // namespace

LogicalResult registerNddmaImplementations(ascend::instruction::ImplementationRegistry &registry,
                                           ascend::instruction::SelectionPolicyRegistry *) {
  constexpr ApiFamily kFamily = ApiFamily::RegBased;
  // 注册期就把「选不选中我」与「往 kernel 里内联哪份头文本」这两份编码对上，对不上让整张
  // 注册表建不起来。`datacopy_nddma_reg_base.h` 带 `_reg_base.h` 后缀，正是这条判据认的
  // 那个标记（`ElementwiseApiFamily.h` 的 `headersMatchFamily`）。
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::DataCopyNddmaRegBaseOp>(kFamily))) {
    return failure();
  }
  // 不进 `registeredFamilies()` 那张侧表。 那张表是给 elementwise 的（读点是
  // `BridgeOperationRegistryTest` 的按族覆盖断言与 `ElementTypeMatrixTest` 的元素类型集合
  // 断言），而搬运族不在里面——`Load` / `Store` 同样不登记。登记过一次，后果是这一档
  // 「不按元素类型集合判」被 `OnlyCastPairImplementationsOptOutOfElementTypeSets` 当成
  // 「忘了传集合」（那条断言钉的是只有 `CastPair` 可以不传），全量 `check-ut` 当场红。
  //
  // 代价是「这一档只有 reg 侧」这件事没有断言钉着。它与 `Load` / `Store` 同一处境，而理由
  // 写在这里：mem 侧那张支持表上没有 `DataCopyNddma` 这个包装（`autofuse/ascendc/api/`
  // 下没有对应文件），legacy 自己也只在 reg 侧的 `NddmaApiCall` 里生成它。
  return registry.registerImplementation<afir::NddmaOp, NddmaMapping, NddmaCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{"nddma.gm_ub_reg_base"},
      NddmaMapping{kFamily},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const NddmaMapping> mapping) {
        patterns.add<NddmaConversionPattern>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
      });
}

}  // namespace mlir::ascend::kernel
