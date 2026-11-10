/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- RemovePad.cpp - AFIR remove_pad lowering -----------------*- C++ -*-===//
//
// `afir.remove_pad` 的 lowering。**一档**：两张供给表共用同一个扩展包装
// `RemovePadExtend`（`autofuse/ascendc/api/removepad.h:14`，矩阵 tier D1），所以
// `ApiFamily::Shared`、只有一个 implementation。
//
// 包装内部是一条 `AscendC::GatherMask`，五个实参的实际含义按它反推：
//
//   RemovePadExtend(dst, src, R, P, N)
//     = 从 src 取 N 行（进 `repeatTimes`），行距 P 个元素（只经 `P * sizeof(T) / 32`
//       进 `src0RepeatStride`），每行取前 R 个元素（counter 模式的 `mask`），
//       连续写到 dst。
//
// **dst 侧一个跨度参数都没有**，`GatherMask` 的写是连续的——所以「目的地在整串段上稠密」
// 是这一档的硬判据，违反它是静默写错位置。
//
// 段语言与 legacy 的两条分支：`remove_pad_api_call.cpp:78` 的 `is_last_axis_align` 判的是
// 「x 的尾向量化 stride 是不是既非 1 也非 0」，与这里「最内段的源 stride 是不是 1」是同一件事，
// 只是这边问的是**合轴之后的段**。四种收的形态：
//
//   1 段 + 源 stride 1（且只有一根向量化轴）
//     → R = 段 extent, N = 1                          【纯搬运，legacy 的 `!is_last_axis_align`】
//   1 段 + 源 stride P
//     → R = 1, N = 段 extent, P = 该段源 stride       【逐元素收集，legacy 的 `is_last_axis_align`】
//   2 段 + 源 stride 1
//     → R = 内段 extent, N = 外段 extent, P = 外段源 stride  【legacy 的 `!is_last_axis_align`
//       多轴形态】
//   2 段 + 源槽距（gather）
//     → R = 1, N = 外段实际 extent × 内段名义 extent, P = 槽距（内段源 stride）
//       【整窗口逐元素收集，legacy 的 `is_last_axis_align` 多段形态——转置折进 Load 布局后
//       源按 `Align(1, 32/sizeof(T))` 逐元素开槽、向量化区又被 TileInner 边界切成两段时
//       走这里；legacy 由 `GetOneAxisSize` 逐轴连乘承载，本档见 `classifyRemovePadSegments`
//       的整窗线性判据与 pattern 的 `gatherWholeWindow` 趟数】
//
// **段数 > 2 一律报不支持**，而且不建外层 `scf.for`：legacy 把所有向量化轴连乘进一次调用
// （`GetNonLastAxisMergeSize`），隐含假设「行距处处相同」；合轴之后段数 ≥ 3 恰好说明那个假设
// 不成立，legacy 在那里是静默算错。建循环技术上做得到（广播那一档就建了），但循环会把 dst
// 的起始地址推到 `k × 内侧元素数` 处，那个地址不是 32B 整数倍时 `GatherMask` 的目的地地址
// 检查会失败。要收它得先有一条那个形状的用例证明地址落得对。
//
// **32B 整除判据**（`isProvablyMultipleOf`）是本档与 legacy 唯一的实质差别。legacy 不读
// descriptor 的 stride，而是自己重算 `SizeAlign(行长, 32/sizeof(T))`
// （`remove_pad_api_call.cpp:88`），那个值必然是 32B 的整数倍，所以它结构性地躲开了包装里
// `src0RepeatStride = P * sizeof(T) / 32` 的整除截断。本档发的是 descriptor 里那条**真实行距**
// ——真实布局，而重算出来的只是「应该是」——代价是要自己挡住截断。机理与设计文档 P3 那条
// `dst_rpt_stride` 逐字相同：截断之后 repeat 前进的距离小于一行，**零报错、结果错**。
//
// 设计与全部证据见 `docs/development/2026-09-30-afir-remove-pad-design.md`。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/RemovePad.h"
#include "conversion/AFIRToAscendC/VectorizedLoops.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
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
using elementwise::isSupportedElementType;
using elementwise::kBF16;
using elementwise::kF16;
using elementwise::kF32;
using elementwise::kI16;
using elementwise::kI32;
using elementwise::kUI16;

// UB 的访问粒度，也是 `removepad.h` 里那条整除的除数。与 bridge 的
// `kTmpBufferAlignmentBytes` 使用同一个常量，不是芯片判别。
constexpr int64_t kUbAccessGranularityBytes = 32;

// legacy 两侧的支持表逐字相同：`{i16, ui16, i32, ui32, f16, f32, bf16}`
// （`ascir_builtin_ops_v1.cpp:163`）。这里少两个——`ui32` 与 `bf16` 在
// `ElementTypes` 里没有位（`elementTypeBit` 答 0），够不着也就谈不上收。
// ODS 的 `AFIR_RemovePadElementTypes` 比这里宽是对的：那是 op 收什么。
// bf16 在这一条上两侧都有（`mem_dtypes` 与 `reg_dtypes` 都列着它），所以它与那 36 个
// reg 侧独有的位不同——这个包装两张支持表共用一份，加在这里不会对任何一侧说假话。
constexpr uint32_t kRemovePadElementTypes = kF16 | kF32 | kI16 | kI32 | kUI16 | kBF16;

struct RemovePadMapping {
  ApiFamily family = ApiFamily::Shared;
  uint32_t elementTypes = kRemovePadElementTypes;
  // legacy 的两份 codegen impl 都没有覆写 `CalcTmpBufSize`，基类默认返回空
  // （`ascir_registry.h:118`），所以调度不给这个节点 tmp buffer，包装的形参表里也没有它。
  // `checkTmpBuffer` 拿这个值与调度严格对等。
  bool needsTmpBuffer = false;
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
};

// `expression` 可证等于 `literal` 吗。证不了返回 false（保守方向）。
//
// 与 Broadcast.cpp 的同名函数逐字相同。 提到公共头不属于本需求：`SymbolicExpression.h`
// 在 `ProveExpressionsEqual` 上已经记着同一类去重（bridge 里还有第三份
// `proveExpressionsEqual`），那三份应当一起收，不该由本需求单独挪走其中一份。
bool provablyLiteral(afir::ExprAttr expression, llvm::StringRef literal) {
  if (!expression) {
    return false;
  }
  llvm::StringRef text = expression.getValue().getValue();
  return text == literal || afir::ProveExpressionsEqual(text, literal);
}

// `provablyEqual` 直接用 `mlir::afir` 导出的那份（`VectorSegmentation.cpp`）：
// 文本短路 + `ProveExpressionsEqual`，空侧答 false，语义与这里原先的局部副本一致。

// 一个 `afir.remove_pad` 在分段之后的形态。三个尺寸各记「哪一段的哪一项」，不记物化出来的
// 值——capability 只判形态、不建 IR，pattern 才物化。
struct RemovePadShape {
  // `VectorPlan::tensors` 里源与目的地的下标。两侧各恰好一个。
  unsigned source = 0;
  unsigned destination = 0;
  // 行长 R 取 `segments[rowSegment].extent`；`rowIsUnit` 为真时 R 是字面量 1。
  size_t rowSegment = 0;
  bool rowIsUnit = false;
  // 趟数 N 取 `segments[repeatSegment].extent`；`repeatIsUnit` 为真时 N 是字面量 1。
  size_t repeatSegment = 0;
  bool repeatIsUnit = false;
  // 整窗口逐元素收集（两段 gather）：R 是「外段实际元素数 × 内段名义元素数」，不是单段
  // 的元素数。两段的行距都是槽距（最内段源 stride），见 `classifyRemovePadSegments`。
  bool gatherWholeWindow = false;
  // 源侧行距 P。`repeatIsUnit` 时它不参与（见 `classifyRemovePadSegments`）。
  afir::ExprAttr pitch;
};

// `plan` 是不是本档收的形状；是就把判据用得上的那几项算出来。
//
// 收的边界（不满足一律 `failure()`，调用方翻成 unsupported）：
//
//   - 源与目的地各恰好一个；段数在 `[1, 2]` 内（段数 > 2 的理由见文件头）。
//   - 目的地稠密：最内段 `strides[dst]` 可证为 1；两段时外段 `strides[dst]` 可证等于内段
//     `extent`。`GatherMask` 的写是连续的，目的地有任何跨度就是静默写错位置。
//   - **行形态**（两段、最内段源 stride 可证为 1）：一行内部也有跨度的话，
//     `GatherMask` 的 `src0BlockStride` 表达不了（包装写死 1）。
//   - **整窗口逐元素收集**（两段、最内段源 stride 是对齐槽距）：外段源步长可证等于
//     「槽距 × 内段元素数（名义）」，源侧整窗线性。这是 legacy
//     `remove_pad_api_call.cpp` `is_last_axis_align` 分支的同款形态：转置折进 Load
//     布局后，源按 `Align(1, 32/sizeof(T))` 逐元素开槽。单段时同款收集见下面的
//     else 分支。
//   - 行距/槽距是 32B 的整数倍（`isProvablyMultipleOf`）。
//
// `N` 可证为 1 时单独放行整除判据，这是一条有前置条件的放行，前置条件在这里查：
// `src0RepeatStride` 只被第 2..N 趟 repeat 读，`repeatTimes == 1` 时它读不到。这一格的 `N`
// 是字面量 1（段只有一个、源 stride 又是 1，趟数就是 1），不是符号，所以前置条件是判得死的。
FailureOr<RemovePadShape> classifyRemovePadSegments(const afir::VectorPlan &plan, int64_t elementBytes) {
  RemovePadShape shape;
  std::optional<unsigned> source;
  std::optional<unsigned> destination;
  for (auto [index, tensor] : llvm::enumerate(plan.tensors)) {
    std::optional<unsigned> &slot = tensor.side == afir::VectorTensor::Side::Destination ? destination : source;
    if (slot) {
      return failure();
    }
    slot = static_cast<unsigned>(index);
  }
  if (!source || !destination) {
    return failure();
  }
  shape.source = *source;
  shape.destination = *destination;

  const size_t count = plan.segments.size();
  if (count == 0 || count > 2) {
    return failure();
  }
  for (const afir::VectorSegment &segment : plan.segments) {
    if (segment.strides.size() != plan.tensors.size()) {
      return failure();
    }
  }
  const afir::VectorSegment &inner = plan.segments[count - 1];
  if (!provablyLiteral(inner.strides[shape.destination], "1")) {
    return failure();
  }
  const bool contiguousRow = provablyLiteral(inner.strides[shape.source], "1");
  if (count == 2) {
    const afir::VectorSegment &outer = plan.segments[0];
    // 目的地稠密两种形态共用：外段步长必须等于内段元素数（名义），`GatherMask` 的写
    // 才是线性的。
    if (!provablyEqual(outer.strides[shape.destination], inner.extent)) {
      return failure();
    }
    if (contiguousRow) {
      // 行形态：外段是行距，内段是整行。
      shape.rowSegment = count - 1;
      shape.repeatSegment = 0;
      shape.pitch = outer.strides[shape.source];
    } else {
      // 整窗口逐元素收集（两段 gather）：最内段源 stride 是槽距（对齐粒度，见下面
      // `isProvablyMultipleOf`），外段源步长必须可证等于「槽距 × 内段元素数（名义）」
      // ——源侧整窗线性，`GatherMask` 才能用一条线性寻址扫过两段。对应 legacy
      // `remove_pad_api_call.cpp` 的 `is_last_axis_align` 分支：那边的 R 由
      // `GetOneAxisSize` 逐轴相乘，非首 TileInner 轴取名义 `Size`、其余取
      // `ActualSize`，与本处「外段实际 × 内段名义」同一口径。
      FailureOr<ExprAttr> expected =
          multiplyExprs(inner.strides[shape.source].getContext(), inner.strides[shape.source], inner.extent);
      if (failed(expected) || !provablyEqual(outer.strides[shape.source], *expected)) {
        return failure();
      }
      shape.rowIsUnit = true;
      shape.gatherWholeWindow = true;
      // Whole-window gathering consumes both extents; repeatSegment is unused.
      shape.pitch = inner.strides[shape.source];
    }
  } else if (contiguousRow) {
    // 纯搬运：一行就是整段，趟数 1。
    shape.rowSegment = 0;
    shape.repeatIsUnit = true;
  } else {
    // 逐元素收集：每个元素单独占一段行距（调度给的是 `Align(1, 32/sizeof(T))`）。
    shape.rowIsUnit = true;
    shape.repeatSegment = 0;
    shape.pitch = inner.strides[shape.source];
  }
  if (!shape.repeatIsUnit &&
      !ProveExpressionMultipleOf(shape.pitch.getValue().getValue(), kUbAccessGranularityBytes / elementBytes)) {
    return failure();
  }
  return shape;
}

// 元素类型的字节数。位宽不是 8 的整数倍（`i4`）时返回 0——那种 dtype 不在本档收的集合里，
// 但判据不该依赖那件事成立。
int64_t elementByteWidth(Type elementType) {
  if (!elementType.isIntOrFloat()) {
    return 0;
  }
  const unsigned bits = elementType.getIntOrFloatBitWidth();
  return bits % 8 == 0 ? static_cast<int64_t>(bits / 8) : 0;
}

// capability 与 pattern 问的是同一件事，各算一遍而不是缓存——与 `analyzeSegmentedBroadcast`
// 同一个理由：两处问的是同一个值上的同一件事，缓存反而多一条要对齐的路。
FailureOr<RemovePadShape> analyzeRemovePad(Operation *op, int64_t elementBytes, afir::VectorPlan &plan) {
  if (elementBytes <= 0) {
    return failure();
  }
  FailureOr<afir::VectorPlan> collapsed = afir::collapseVectorizedAxes(op);
  if (failed(collapsed)) {
    return failure();
  }
  plan = std::move(*collapsed);
  return classifyRemovePadSegments(plan, elementBytes);
}

class RemovePadCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit RemovePadCapabilityProvider(std::shared_ptr<const RemovePadMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_ || !elementwise::hasCanonicalDescriptors(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    FailureOr<TransferPlan> plan = buildTransferPlan(operation);
    if (failed(plan)) {
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

    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
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

    afir::VectorPlan vectorPlan;
    if (failed(analyzeRemovePad(operation, elementByteWidth(sourceType.getElementType()), vectorPlan))) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *, const ascend::instruction::InplaceQuery &,
                                 const ascend::instruction::QueryContext &) const override {
    // `GatherMask` 的源与目的地重叠没有核过，而这个 op 的两端本来就是两块（一块带填充、
    // 一块紧凑），调度产不出别名的形态。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryScalarInput(Operation *, const ascend::instruction::ScalarInputQuery &,
                                     const ascend::instruction::QueryContext &) const override {
    // legacy 的 `IsNodeValid` 也拒标量输入（`v1_ascir_codegen_impl.h:262`）。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    // `RemovePad` 的输入输出差的是 padding 的有无，不是广播；两者的轴数、步长关系与
    // `(1, A) op (B, A)` 无关。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const RemovePadMapping> mapping_;
};

class RemovePadConversionPattern final : public OpConversionPattern<afir::RemovePadOp> {
 public:
  RemovePadConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                             std::shared_ptr<const RemovePadMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(afir::RemovePadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form remove_pad requires a destination");
    }
    auto sourceType = dyn_cast<ascendc::LocalTensorType>(adaptor.getInput().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "source is not a target-dialect tile");
    }
    FailureOr<TransferPlan> plan = buildTransferPlan(op);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR transfer plan");
    }
    afir::VectorPlan vectorPlan;
    FailureOr<RemovePadShape> shape = analyzeRemovePad(op, elementByteWidth(sourceType.getElementType()), vectorPlan);
    if (failed(shape)) {
      return rewriter.notifyMatchFailure(op, "not a supported remove_pad segmentation");
    }

    FailureOr<SmallVector<Value>> ordered =
        orderTilesForPlan(vectorPlan, {adaptor.getInput()}, adaptor.getDestination());
    if (failed(ordered)) {
      return rewriter.notifyMatchFailure(op, "cannot order the tiles the segmentation talks about");
    }
    // 窗口是整串段，所以一层循环都不建：段数上限 2 是这一档的判据，`materializeVectorizedSegments`
    // 在 `apiWindowSize == segments.size()` 时只做两件事——给两个 tile 各生成一条偏移为常量 0 的
    // subindex（本地侧偏移恒为 0，与 `UnaryElementwise.cpp` 同一条判据），以及按逐轴 extent
    // 算出各段的元素数（含切分轴的尾块修正）。
    const auto windowSize = static_cast<unsigned>(vectorPlan.segments.size());
    FailureOr<VectorizedWindow> window = materializeVectorizedSegments(
        op, rewriter, vectorPlan,
        plan->kind == TransferPlanKind::Scheduled ? ExtentSource::Actual : ExtentSource::Nominal, *ordered, windowSize);
    if (failed(window) || window->tiles.size() != vectorPlan.tensors.size() || window->extents.size() != windowSize) {
      return rewriter.notifyMatchFailure(op, "cannot materialize the vectorized segment extents");
    }

    // 行长与趟数：字面量 1 的那一项建常量，其余取段的元素数。
    Value one;
    if (shape->rowIsUnit || shape->repeatIsUnit) {
      one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    }
    Value rowSize = shape->rowIsUnit ? one : window->extents[shape->rowSegment];
    Value repeatCount;
    if (shape->repeatIsUnit) {
      repeatCount = one;
    } else if (shape->gatherWholeWindow) {
      // 整窗口收集的趟数是「外段实际元素数 × 内段名义元素数」。目的地行距按名义布局
      // （稠密判据钉的是名义 extent），尾块的内段行间空槽要一并线性扫过、写进 tile 缓冲
      // 的名义空位——消费方按 ActualSize 前缀读，空位不读；外段是本趟真正的行数，取
      // 实际值，乘积不超过 tile 缓冲的名义容量。与 legacy `GetOneAxisSize` 对非首
      // TileInner 轴取名义 `Size`、其余取 `ActualSize` 的口径逐字对齐。
      FailureOr<Value> innerCount =
          afir::materializeSegmentExtent(op, rewriter, vectorPlan.segments[1], /*scheduledAxis=*/-1, {});
      if (failed(innerCount)) {
        return rewriter.notifyMatchFailure(op, "cannot materialize the nominal inner segment extent");
      }
      repeatCount = rewriter.create<arith::MulIOp>(op.getLoc(), window->extents[0], *innerCount).getResult();
    } else {
      repeatCount = window->extents[shape->repeatSegment];
    }
    // 行距。`repeatIsUnit` 那一格发的是行长本身：形参在（包装的签名里），但
    // `src0RepeatStride` 只被第 2..N 趟读，`repeatTimes == 1` 时读不到它。
    Value pitch = rowSize;
    if (!shape->repeatIsUnit) {
      FailureOr<Value> stride = afir::materializeStrideValue(op, rewriter, shape->pitch);
      if (failed(stride)) {
        return rewriter.notifyMatchFailure(op, "cannot materialize the source row pitch");
      }
      pitch = *stride;
    }

    Value source = window->tiles[shape->source];
    Value destination = window->tiles[shape->destination];
    Operation *call =
        rewriter.create<ascendc::RemovePadExtendOp>(op.getLoc(), destination, source, rowSize, pitch, repeatCount);
    if (failed(setExecutionSpec(call, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const RemovePadMapping> mapping_;
};

}  // namespace

LogicalResult registerRemovePadImplementations(ImplementationRegistry &registry,
                                               ascend::instruction::SelectionPolicyRegistry *) {
  // 两侧共用同一个包装（tier D1），所以 `Shared`、只有这一个 implementation。
  // id 说的是支持表，不是芯片。
  constexpr llvm::StringLiteral kId = "remove_pad.tensor.extend";
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::RemovePadExtendOp>(ApiFamily::Shared))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(afir::RemovePadOp::getOperationName(), kId, ApiFamily::Shared,
                                      kRemovePadElementTypes);
  return registry.registerImplementation<afir::RemovePadOp, RemovePadMapping, RemovePadCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{kId.str()},
      RemovePadMapping{ApiFamily::Shared, kRemovePadElementTypes, /*needsTmpBuffer=*/false},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const RemovePadMapping> mapping) {
        patterns.add<RemovePadConversionPattern>(context.typeConverter, patterns.getContext(), key, std::move(mapping));
      });
}

}  // namespace mlir::ascend::kernel
