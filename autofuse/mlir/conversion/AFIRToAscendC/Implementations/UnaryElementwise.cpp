/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- UnaryElementwise.cpp - AFIR unary elementwise lowering ---*- C++ -*-===//
//
// 每个 AFIR 一元 op 在这里注册一到两个 implementation。一个 implementation 由下面三件
// 事确定，它们彼此独立，不能合并成一个开关。合并过一次，就一定会在某个 op 上答反。
//
//   1. 建哪个 AscendC op：生成标准调用，还是生成 Autofuse 的包装。由模板实参决定。
//   2. 属于哪张扩展 API 支持表（`ApiFamily`）：mem-based 的 `xxx.h`、reg-base 的
//      `xxx_reg_base.h`，或者两张表给的是同一个调用（`Any`，不查 feature）。
//      它问的不是「标准 API 在哪一侧」：`afir.abs` 的标准调用在 reg-base 一侧，
//      而 `afir.neg` 两侧都只有包装，区别仅在头文件。
//   3. 要不要 tmp buffer：`afir.sigmoid` 两侧都要，`afir.neg` 两侧都不要，
//      `afir.abs` 和 `afir.reciprocal` 只有走包装的那一侧要。
//
// 第 3 条与 feature 是「且」的关系，而且两个方向都要查。调度多给一块要报 unsupported，
// 否则那块 UB 就被悄悄丢掉；调度少给一块同样要报，因为 AFIR 造不出这块 buffer，它的
// 尺寸由 ATT 聚合得出。
//
// 本文件的 stable id 与注释都不出现芯片名。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/UnaryElementwise.h"
#include "conversion/AFIRToAscendC/VectorizedLoops.h"
#include "target/Ascend/AscendTargetProfile.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/Implementations/ConstBlockTensor.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseStockCalls.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseTransferPlan.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/AFIRElementwiseSupport.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <memory>
#include <optional>
#include <type_traits>
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

using elementwise::isSupportedElementType;
using elementwise::kBF16;
using elementwise::kF16;
using elementwise::kF32;
using elementwise::kI16;
using elementwise::kI32;
using elementwise::kUI16;
using elementwise::kUI8;

// 这条路本来只有一个全局判据，收 {f16, f32, i32}。改成每个 implementation 各说各的之后，
// 已经接通的那些仍然沿用它——它们各自到底该收哪几种，没有逐个对过 legacy，这里只是
// 把原来的行为原样留下，不是查证的结果。新接的 op 别用这个常量，照着自己的 `REG_ASC_IR`
// 写。
// 一元里收 {f16, f32, i32} 的那几个：`abs` / `neg` / `relu`。
//
// 这不是「一元的缺省」，原先它叫 `kLegacyUnaryElementTypes` 并且是 `registerUnary`
// 的缺省实参，于是 `exp` / `ln` / `sqrt` / `sigmoid` / `reciprocal` / 谓词族这些
// legacy 只收 f16 / f32 的 op 也一并领到了 i32——`ElementTypeMatrixTest` 第一次跑就把
// 这 16 处照了出来。缺省实参因此去掉：集合是每个 op 自己的事实，不该有缺省。
constexpr uint32_t kFloatAndI32ElementTypes = kF16 | kF32 | kI32;
constexpr uint32_t kFloatElementTypes = kF16 | kF32;
// `abs` 的 reg 侧：legacy 比 mem 侧多收 i16 / ui8。
constexpr uint32_t kAbsRegElementTypes = kF16 | kF32 | kI16 | kI32 | kUI8 | kBF16;
// `neg`：legacy 两侧都收 i16，原先注册处漏了。
constexpr uint32_t kNegElementTypes = kF16 | kF32 | kI16 | kI32;

// 结果元素类型与输入的关系。绝大多数是 `Same`；`Uint8Predicate` 是输入浮点、结果一个
// 字节真值的那一类（isnan / isfinite / isinf）；`CastPair` 只有 `afir.cast`——它的输入
// 与结果必须不同 dtype，而合法的组合是一张逐行的对表，不是一个集合。
//
// 三档都不是「一个集合加一条关系」能合并的：`Same` 要查 `elementTypes`，另外两档各自
// 把输入与结果一起钉死，再给一份集合就是两处说同一件事。
// 第四档 `Uint8Logical` 是 `afir.logical_not` 的 reg 侧：结果恒为 `ui8`，输入是
// `elementTypes` 说的那个集合。它与 `Uint8Predicate` 差的不是一句话——谓词族把输入钉死
// 成浮点、不读 `elementTypes`（`isnan` 对 int32 没有意义），而这一档收的正是整数与浮点
// 混着的那一串（reg 侧的对表里 i8 / i16 / i32 / i64 / ui* / bf16 / f16 / f32 都在）。
// 合并成一档就要么把谓词族放宽到整数，要么把这一档收窄到浮点，两边都错。
//
// mem 侧的 `logical_not` 不走这一档，它是 `Same`——两侧关系不同，这正是它不声明
// `InferTypeOpInterface` 的原因（见 `Ops/Elementwise.td` 的 `AFIR_BufferizableLogicalNotOp`）。
// 结果元素类型与输入的关系：共用 `AFIRElementwiseSupport.h` 的那一个枚举，ODS 的
// `resultRule` 读的也是它。原先这里是一个本地枚举，与 ODS 的代码块各写各的。
//
// 原先的 `Uint8Predicate` 与 `Uint8Logical` 合成了一档：两者说的结果 dtype 是同一个
// （ui8），差别只在输入怎么查——前者把「f16 或 f32」写死在判据里，后者查注册处的集合。
// 谓词族现在显式注册 `kFloatElementTypes`，那条写死的就成了同一个答案的第二份编码。
using afir::elementwise::ResultElementRule;

// 这个 implementation 要不要一块预先填好 1 的常量 block，以及那一块的元素类型。
//
// 它不是 `needsTmpBuffer` 的第二个取值：tmp buffer 是调度分配的、AFIR 上带得到的一个
// 操作数，而这一块在 AFIR 上根本不存在，是 lowering 在 kernel 序言里现造的
// （见 `ConstBlockTensor.h`）。元素类型由包装的形参钉死，与 dst / src 的 dtype 无关——
// `RsqrtExtend` 收 `LocalTensor<float>`，`LogicalNot` 收 `LocalTensor<half>`，哪怕那两个
// op 正在处理的是别的 dtype。
//
// 加一个枚举值之前先看 legacy 的 `api_extract_dup_map`（`common/common_utils.cpp`）。
// 这 32 字节是 ATT 预留、MLIR 分配的：ATT 按那张表去重算出
// `tuning_space_->builtin_tmp_buffer` 并算进 UB 占用，本文件按 `ascendc.const_block`
// 去重把它造出来，两处各算各的、没有共享断言。表里没有的 (值, dtype) 一旦在这里出现，
// MLIR 就多分配一块而 ATT 不知道——失败是运行期 UB 越界，不是编译报错。见进度文档 L-5。
enum class ConstBlockOnes {
  None,
  F32,
  F16,
};

struct UnaryElementwiseMapping {
  ApiFamily family = ApiFamily::Any;
  ResultElementRule resultElementType = ResultElementRule::Same;
  // 只有 `CastPair` 那一档读它：查哪一侧的 (输入, 输出) 对表。其余档次下它是死字段，
  // 取什么值都不影响结果——所以它跟着 `family` 走，而不是再开一个注册处实参。
  afir::elementwise::CastSupply castSupply = afir::elementwise::CastSupply::MemBased;
  // 这个 implementation 收哪几种元素类型。位的拼写在 `ElementwiseApiFamily.h`,
  // 集合写在注册处——与二元那条路同一个办法。
  uint32_t elementTypes = kFloatElementTypes;
  // 选中的 API 要不要调用方给的 tmp buffer。它不是 `family` 的函数，也不是「标准还是
  // 扩展」的函数：见文件头。唯一依据是 pattern 的模板实参，注册处由同一个实参同时
  // 决定这里和建 op 的实参个数。
  bool needsTmpBuffer = false;
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
};

// 输入与结果的元素类型关系，以及输入自己的取值范围。两件事写在一起，是因为对谓词族
// 它们互相约束：结果必须是一个字节宽的整数，输入必须是浮点——`isnan` 对 int32 没有
// 意义，而只查「结果是 ui8」会把它放进来。
//
// 谓词族不看 `allowed`：它的输入集合由这条判据自己钉死，再给一份集合就是两处说同一件事。
bool areElementTypesSupported(ResultElementRule relation, uint32_t allowed, afir::elementwise::CastSupply supply,
                              Type sourceElement, Type destinationElement) {
  if (relation == ResultElementRule::CastPair) {
    // 也不看 `allowed`：合法的组合是一张对表，不是「输入收哪些」乘「结果能是哪些」。
    // 同 dtype 那条单独挡——legacy 的 `CastApiCall` 开头就 `GE_ASSERT_TRUE(x.dtype !=
    // y.dtype)`，而 mem 侧的对表里偏偏有一行 `{f32, f32}`（那是给 VF 融合留的），
    // 照收会生成一条 legacy 永远不会生成的调用。
    return sourceElement != destinationElement &&
           afir::elementwise::isSupportedCastPair(supply, sourceElement, destinationElement);
  }
  return isSupportedElementType(allowed, sourceElement) &&
         afir::elementwise::matchesResultElementRule(sourceElement, destinationElement, relation);
}

class UnaryElementwiseCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit UnaryElementwiseCapabilityProvider(std::shared_ptr<const UnaryElementwiseMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_ || !elementwise::hasCanonicalDescriptors(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    TransferPlanKind planKind = TransferPlanKind::Identity;
    const elementwise::LocalViewCompatibility compatibility =
        elementwise::checkElementwiseLocalViews(operation, &planKind);
    if (compatibility == elementwise::LocalViewCompatibility::UnsupportedShape) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    if (compatibility == elementwise::LocalViewCompatibility::MissingInformation) {
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

    return queryTensorContract(operation, planKind);
  }

  CapabilitySupport queryTensorContract(Operation *operation, TransferPlanKind planKind) const {
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
    if (failed(partition) || partition->inputCount != 1 || partition->destinationCount != 1) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    auto sourceType = dyn_cast<ascendc::LocalTensorType>(partition->inputs(operation)[0].getType());
    auto destinationType = dyn_cast<ascendc::LocalTensorType>(partition->destinations(operation)[0].getType());
    if (!sourceType || !destinationType) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    // 未调度那条路（`Identity`）只收秩为 1 的 tile。这条判据二元和三元一直都有，一元
    // 这边原先漏了：同一份 `2x8xf16`，`afir.add` 当场报 unsupported，而 `afir.exp` 会被
    // 放行、按 `GetSize()` 生成整块的元素数。三条路对同一份 IR 给不同答案，不是有意的
    // 放宽，所以在这里补齐。三条一起钉在
    // `afir-elementwise-identity-rank-invalid.mlir`（二元那条原先拦住它的其实是缺
    // descriptor，不是秩，那份用例一并重做了）。
    //
    // 走 `Scheduled` 时不查秩：元素个数由有效视图折算出的局部跨度说了算，多维 tile 上
    // LocalTensor 的秩跟着 UB buffer 的 memref 走，与它无关。
    if (planKind == TransferPlanKind::Identity && !(sourceType.hasRank() && sourceType.getRank() == 1 &&
                                                    destinationType.hasRank() && destinationType.getRank() == 1)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    if (!areElementTypesSupported(mapping_->resultElementType, mapping_->elementTypes, mapping_->castSupply,
                                  sourceType.getElementType(), destinationType.getElementType())) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    // 形状默认逐字相同；唯一例外是 scheduler 删除 redundant Broadcast 后留下的
    // unary inline broadcast。它必须走 AFIR 层的窄判据：差异轴 stride 为 0 且不在
    // 向量化轴上，AscendC API 才能按 destination 窗口重复读取 source。
    if (sourceType.getShape() != destinationType.getShape() &&
        !afir::elementwise::sourceBroadcastsToDestination(operation, partition->inputs(operation)[0],
                                                          partition->destinations(operation)[0])) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *, const ascend::instruction::InplaceQuery &,
                                 const ascend::instruction::QueryContext &) const override {
    // The extension wrapper handles dst == src explicitly, but nothing in this
    // slice produces an aliased operand yet, so it stays unclaimed rather than
    // claimed-and-untested.
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
  std::shared_ptr<const UnaryElementwiseMapping> mapping_;
};

// The operand tile and element count both lowerings need, after the scheduled
// offsets have been applied. Identical in shape to the binary elementwise path;
// the two differ only in how many inputs they carry.
struct UnaryOperandTiles {
  Value source;
  Value destination;
  Value count;
};

template <typename SourceOp>
LogicalResult checkConversionLocalViews(SourceOp op) {
  const elementwise::LocalViewCompatibility compatibility = elementwise::checkElementwiseLocalViews(op);
  if (compatibility != elementwise::LocalViewCompatibility::Compatible) {
    return failure();
  }
  return success();
}

template <typename SourceOp, typename OpAdaptor>
FailureOr<UnaryOperandTiles> materializeOperandTiles(SourceOp op, OpAdaptor adaptor,
                                                     ConversionPatternRewriter &rewriter, bool compactTail) {
  UnaryOperandTiles tiles;
  tiles.source = adaptor.getInput();
  tiles.destination = adaptor.getDestination();
  FailureOr<TransferPlan> plan = buildTransferPlan(op, TransferPlanSide::Operand);
  if (failed(plan)) {
    return failure();
  }
  if (plan->kind != TransferPlanKind::Scheduled) {
    tiles.count = rewriter.create<ascendc::LocalTensorGetSizeOp>(op.getLoc(), rewriter.getI32Type(), tiles.destination);
    return tiles;
  }

  // 向量化轴的分段（阶段 2 合轴）。一元 elementwise 把最内一段交给 API，其余段变循环。
  //
  // 今天所有图在这里都只有一段——图级合轴已经合到底——所以走的是「不建循环、偏移恒为
  // 常量 0」那条分支，生成的 Op 与此前逐字相同。多段此前是被
  // `materializeScheduledTileIndices` 的「stride 必须是 0 或 1」挡回去的（**安全地拒绝**，
  // 不是算错），现在变成支持。
  FailureOr<afir::VectorPlan> segments = afir::collapseVectorizedAxes(op, compactTail);
  if (failed(segments)) {
    return failure();
  }
  FailureOr<SmallVector<Value>> ordered = orderTilesForPlan(*segments, {tiles.source}, tiles.destination);
  if (failed(ordered)) {
    return failure();
  }
  FailureOr<VectorizedWindow> window =
      materializeVectorizedSegments(op, rewriter, *segments, *plan, *ordered, /*apiWindowSize=*/1);
  if (failed(window) || window->tiles.size() != segments->tensors.size() || window->extents.size() != 1) {
    return failure();
  }
  for (auto [tensor, tile] : llvm::zip(segments->tensors, window->tiles)) {
    (tensor.side == afir::VectorTensor::Side::Destination ? tiles.destination : tiles.source) = tile;
  }
  tiles.count = window->extents.front();
  return tiles;
}

// `MaterializesReuseSource` 是第四根轴，只有 `MathLibraryOp` 一类用得上：那一类的
// ODS（`ascendc/Base.td` 的 `UnaryMathOp`）带一个必填的 `isReuseSource` 操作数，
// 被 emitter 当模板实参打印（`Target/Asc/Adv/Math.h`）。dialect 里产得出它：
// `arith.constant false` 经 `printConstantOp` 打成 `constexpr bool c0_i1 = false;`，
// 而 `constexpr bool` 是合法的模板实参，于是生成的是
// `AscendC::Tanh<float, c0_i1>(dst, src, tmp, count)`。
//
// legacy 的 `UnaryApiTmpCall` 一个模板实参都不打（`Tanh(y[off], x[off], tmp, size);`），
// 靠 AscendC 自己的默认值 `isReuseSource = false`。我们打出来的与它语义相同、文本不同
// ——与「已知覆盖空洞」第 0 条的 `isSetMask` 同一类偏差，是已被接受的先例。
//
// 它与 `TakesTmpBuffer` 不是独立的两位：今天这一类的 op（tanh / erf）都要 tmp buffer，
// 「不要 tmp 但要 isReuseSource」那个组合一个实例都没有，所以不写那一支，由
// `static_assert` 挡住。真出现了再补，别先摆一条没人走的分支在这里。
template <typename SourceOp, typename TargetOp, bool TakesTmpBuffer, bool MaterializesReuseSource = false,
          ConstBlockOnes ConstBlock = ConstBlockOnes::None>
class UnaryElementwiseConversionPattern final : public OpConversionPattern<SourceOp> {
  static_assert(!MaterializesReuseSource || TakesTmpBuffer,
                "a MathLibraryOp without a tmp buffer has no instance yet; add the branch when one appears");
  // 两个包装都是「常量 block + tmp buffer」一起要的。「只要常量不要 tmp」那个组合一个
  // 实例都没有，所以不写那一支——真出现了再补。
  static_assert(ConstBlock == ConstBlockOnes::None || TakesTmpBuffer,
                "a const-block wrapper without a tmp buffer has no instance yet; add the branch when one appears");
  static_assert(ConstBlock == ConstBlockOnes::None || !MaterializesReuseSource,
                "no wrapper takes both a const block and an isReuseSource template argument");

 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  UnaryElementwiseConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                    std::shared_ptr<const UnaryElementwiseMapping> mapping, bool compactTail)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        key_(std::move(key)),
        mapping_(std::move(mapping)),
        compactTail_(compactTail) {}

  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form unary elementwise op requires a destination");
    }
    if (failed(checkConversionLocalViews(op))) {
      return rewriter.notifyMatchFailure(op, "incompatible elementwise local views");
    }

    FailureOr<UnaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    Operation *targetOp = nullptr;
    if constexpr (TakesTmpBuffer) {
      if (!adaptor.getTmpBuffer()) {
        return rewriter.notifyMatchFailure(op, "the selected API requires a tmp buffer operand");
      }
      if constexpr (ConstBlock != ConstBlockOnes::None) {
        // 常量 block 排在 tmp buffer 之前，与包装的形参顺序一致
        // （`RsqrtExtend(dst, src, blk, tmp, size)`）。它按 (值, dtype) 在整个 kernel 里
        // 共用一份，所以这里拿到的可能是别的 op 已经造好的那一块。
        Type blockElement = ConstBlock == ConstBlockOnes::F32 ? Type(Float32Type::get(rewriter.getContext()))
                                                              : Type(Float16Type::get(rewriter.getContext()));
        FailureOr<Value> constBlock =
            elementwise::getOrCreateConstBlockTensor(rewriter, op.getOperation(), blockElement, llvm::APFloat(1.0));
        if (failed(constBlock)) {
          return rewriter.notifyMatchFailure(op, "cannot materialize the pre-filled constant block");
        }
        targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->source, *constBlock,
                                             adaptor.getTmpBuffer(), tiles->count);
      } else if constexpr (MaterializesReuseSource) {
        // 值恒为 false，与 legacy 依赖的那个 AscendC 默认值一致。建成 `arith.constant`
        // 而不是别的，是因为只有它会被 emit 成 `constexpr bool`——一个模板实参位上
        // 用得了的东西。
        Value reuseSource =
            rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
        targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->source, adaptor.getTmpBuffer(),
                                             tiles->count, reuseSource);
      } else {
        targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->source, adaptor.getTmpBuffer(),
                                             tiles->count);
      }
    } else {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->source, tiles->count);
    }
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const UnaryElementwiseMapping> mapping_;
  bool compactTail_;
};

// 一元 + 一个由 ODS 属性来的实参。目前只有 `afir.leaky_relu` 是这个形状：生成的
// 调用是 `AscendC::LeakyRelu(dst, src, slope, calCount)`，而 slope 在 AFIR 上是属性不是
// 操作数——它不参与三段划分，也不占一块缓冲。
//
// 常量按 tile 的元素类型建，不是一律 f32：两个包装都是
// `LeakyRelu(const LocalTensor<T>&, const LocalTensor<T>&, const T&, uint32_t)`，给个 f32
// 常量配 `LocalTensor<half>` 会让 T 同时推成 half 和 float、一个重载都对不上。legacy 是
// 把 `(half)` 强转写进调用文本来解决同一件事(`leaky_relu_api_call.cpp`)。
template <typename SourceOp, typename TargetOp>
class UnaryScalarAttrConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  UnaryScalarAttrConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                   std::shared_ptr<const UnaryElementwiseMapping> mapping, bool compactTail)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        key_(std::move(key)),
        mapping_(std::move(mapping)),
        compactTail_(compactTail) {}

  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form unary elementwise op requires a destination");
    }

    FailureOr<UnaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    auto destinationType = dyn_cast<ascendc::LocalTensorType>(tiles->destination.getType());
    if (!destinationType) {
      return rewriter.notifyMatchFailure(op, "destination is not a target-dialect tile");
    }
    auto elementType = dyn_cast<FloatType>(destinationType.getElementType());
    if (!elementType) {
      return rewriter.notifyMatchFailure(op, "the scalar argument requires a floating-point element type");
    }
    // 属性是 f32，目标可能是 f16——一次到位地按目标语义舍入，不经过两次转换。
    llvm::APFloat slope = op.getNegativeSlopAttr().getValue();
    bool losesInfo = false;
    slope.convert(elementType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    Value scalar =
        rewriter.create<arith::ConstantOp>(op.getLoc(), elementType, rewriter.getFloatAttr(elementType, slope));

    Operation *targetOp =
        rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->source, scalar, tiles->count);
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const UnaryElementwiseMapping> mapping_;
  bool compactTail_;
};

// 一元 + 一个落在模板实参位上的 ODS 属性，外加一块 tmp buffer。今天是多项式族那
// 八个：`ChebyshevPolynomialTExtend<float, 3>(dst, src, tmp, count)`。
//
// 它与上面那条 `UnaryScalarAttrConversionPattern` 的差别只有一处，但那一处决定了不能
// 复用：那边把属性现建成一个常量 SSA 值再当实参传（`AscendC::LeakyRelu` 的 slope
// 就是个实参），而模板实参位要的是编译期常量——在那里打一个 SSA 值出来是个变量名，
// 设备编译器才报错。所以 `n` 原样留在 target op 的属性上，由 emitter 打成十进制字面量。
template <typename SourceOp, typename TargetOp>
class UnaryTemplateAttrConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  UnaryTemplateAttrConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                     std::shared_ptr<const UnaryElementwiseMapping> mapping, bool compactTail)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        key_(std::move(key)),
        mapping_(std::move(mapping)),
        compactTail_(compactTail) {}

  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form unary elementwise op requires a destination");
    }
    if (!adaptor.getTmpBuffer()) {
      return rewriter.notifyMatchFailure(op, "the selected API requires a tmp buffer operand");
    }

    FailureOr<UnaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    Operation *targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->source,
                                                    adaptor.getTmpBuffer(), tiles->count, op.getN());
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const UnaryElementwiseMapping> mapping_;
  bool compactTail_;
};

// 一元 + 一个由这个 op 写死的 `RoundMode`，目标是标准调用
// `AscendC::Cast(dst, src, AscendC::RoundMode::CAST_XXX, count)`。
//
// 取整到整数那一类（`floor_to_int` / `round_to_int` / `trunc_to_int`）走这条。它们与
// `afir.cast` 生成的是**同一个** `AscendC::Cast`，差别只在 RoundMode 的来源：cast 那边按
// (输入, 输出) 推，这几个由 op 自己定死。所以这里不新建 AscendC op，直接用现成的
// `ascendc::CastL2Op`（签名 `(dst, src, roundMode, calCount)`）。
//
// 为什么不能复用 `UnaryElementwiseConversionPattern`：它的无 tmp 分支建的是
// `TargetOp(loc, dst, src, count)` 三个实参，装不下 `roundMode`。RoundMode 是模板实参
// `Mode`、不是 mapping 字段，所以「注册的是哪个模式」与「建 op 时打的是哪个模式」不可能
// 对不上——与 `TakesTmpBuffer` 同一条理由。
//
// `ceil2int` **不走这条**：它走 fork 的 `Ceil2IntExtend` 包装，RoundMode 写在包装体里，
// 外面看不见。两个 op 的结果 dtype 规则相同（`Int32`）而可用实现不同，这正是「规则」与
// 「可用实现」是两根轴的例子。
template <typename SourceOp, typename TargetOp, ascendc::RoundMode Mode>
class UnaryFixedRoundModeConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  UnaryFixedRoundModeConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                       std::shared_ptr<const UnaryElementwiseMapping> mapping, bool compactTail)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        key_(std::move(key)),
        mapping_(std::move(mapping)),
        compactTail_(compactTail) {}

  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form unary elementwise op requires a destination");
    }

    FailureOr<UnaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    Operation *targetOp =
        rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->source,
                                  ascendc::RoundModeAttr::get(rewriter.getContext(), Mode), tiles->count);
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const UnaryElementwiseMapping> mapping_;
  bool compactTail_;
};

template <typename SourceOp, typename TargetOp, bool TakesTmpBuffer, bool MaterializesReuseSource,
          ConstBlockOnes ConstBlock>
ImplementationRegistry::PatternRegistration<UnaryElementwiseMapping> createPatternRegistration() {
  return [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
            std::shared_ptr<const UnaryElementwiseMapping> mapping) {
    patterns.add<
        UnaryElementwiseConversionPattern<SourceOp, TargetOp, TakesTmpBuffer, MaterializesReuseSource, ConstBlock>>(
        context.typeConverter, patterns.getContext(), key, std::move(mapping),
        context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) == ascend::FeatureState::Available);
  };
}

// 查哪一侧的 cast 对表，由 `family` 定死——同一个 implementation 不可能一边生成 reg 侧的
// 调用、一边按 mem 侧的对表判支持。只有 `ResultElementRule::CastPair` 那一档读它。
afir::elementwise::CastSupply castSupplyOf(ApiFamily family) {
  return family == ApiFamily::RegBased ? afir::elementwise::CastSupply::RegBased
                                       : afir::elementwise::CastSupply::MemBased;
}

// `TakesTmpBuffer` 同时决定 capability 里的 tmp buffer 判据和建 op 时的实参个数，所以
// 两者不可能对不上——这正是它是模板实参而不是 mapping 字段的原因。
template <typename SourceOp, typename TargetOp, bool TakesTmpBuffer = false, bool MaterializesReuseSource = false,
          ConstBlockOnes ConstBlock = ConstBlockOnes::None>
LogicalResult registerUnary(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family,
                            ResultElementRule resultElementType, uint32_t elementTypes) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(family))) {
    return failure();
  }
  // `CastPair` 那一档根本不读集合（见 `areElementTypesSupported`），记 0 而不是记那个
  // 没人读的缺省值——记下去就等于在侧表里放一个假事实。
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, family,
                                      resultElementType == ResultElementRule::CastPair ? 0U : elementTypes);
  return registry.registerImplementation<SourceOp, UnaryElementwiseMapping, UnaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      UnaryElementwiseMapping{family, resultElementType, castSupplyOf(family), elementTypes, TakesTmpBuffer},
      createPatternRegistration<SourceOp, TargetOp, TakesTmpBuffer, MaterializesReuseSource, ConstBlock>());
}

// `afir.leaky_relu`：两张支持表上是同一个 `AscendC::LeakyRelu`、都不要 tmp buffer,
// 所以家族恒为 `Any`，只有一个 implementation。
template <typename SourceOp, typename TargetOp>
LogicalResult registerUnaryScalarAttr(ImplementationRegistry &registry, llvm::StringRef id) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(ApiFamily::Any))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, ApiFamily::Any, kFloatElementTypes);
  return registry.registerImplementation<SourceOp, UnaryElementwiseMapping, UnaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      UnaryElementwiseMapping{ApiFamily::Any, ResultElementRule::Same, castSupplyOf(ApiFamily::Any), kFloatElementTypes,
                              /*needsTmpBuffer=*/false},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const UnaryElementwiseMapping> mapping) {
        patterns.add<UnaryScalarAttrConversionPattern<SourceOp, TargetOp>>(
            context.typeConverter, patterns.getContext(), key, std::move(mapping),
            context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) ==
                ascend::FeatureState::Available);
      });
}

// 多项式族：八个 op 的形状逐项相同——reg-only、f32 一种 dtype、要一块 tmp、`n` 落在
// 模板实参位。所以这里不开模板轴，参数只剩 id。
template <typename SourceOp, typename TargetOp>
LogicalResult registerUnaryTemplateAttr(ImplementationRegistry &registry, llvm::StringRef id) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(ApiFamily::RegBased))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, ApiFamily::RegBased, kF32);
  return registry.registerImplementation<SourceOp, UnaryElementwiseMapping, UnaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      UnaryElementwiseMapping{ApiFamily::RegBased, ResultElementRule::Same, castSupplyOf(ApiFamily::RegBased), kF32,
                              /*needsTmpBuffer=*/true},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const UnaryElementwiseMapping> mapping) {
        patterns.add<UnaryTemplateAttrConversionPattern<SourceOp, TargetOp>>(
            context.typeConverter, patterns.getContext(), key, std::move(mapping),
            context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) ==
                ascend::FeatureState::Available);
      });
}

// 取整到整数那一类：目标恒是 `ascendc::CastL2Op`，只有 RoundMode 与 id 不同。
// `elementTypes` 要传——这一类的输入域是 legacy 自己的 `{f16, f32, bf16}`，而结果 dtype
// 由 `rule` 说（`Int32` 或 `CastPair`）。
template <typename SourceOp, ascendc::RoundMode Mode>
LogicalResult registerUnaryFixedRoundMode(ImplementationRegistry &registry, llvm::StringRef id, ResultElementRule rule,
                                          uint32_t elementTypes) {
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::CastL2Op>(ApiFamily::RegBased))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, ApiFamily::RegBased,
                                      rule == ResultElementRule::CastPair ? 0U : elementTypes);
  return registry.registerImplementation<SourceOp, UnaryElementwiseMapping, UnaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      UnaryElementwiseMapping{ApiFamily::RegBased, rule, castSupplyOf(ApiFamily::RegBased), elementTypes,
                              /*needsTmpBuffer=*/false},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const UnaryElementwiseMapping> mapping) {
        patterns.add<UnaryFixedRoundModeConversionPattern<SourceOp, ascendc::CastL2Op, Mode>>(
            context.typeConverter, patterns.getContext(), key, std::move(mapping),
            context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) ==
                ascend::FeatureState::Available);
      });
}

// 谓词族：三个 op 的两条路形状完全一样，只有名字不同——标准调用在 reg-base 一侧、不要
// tmp buffer，包装在 mem-based 一侧、要一块。
template <typename SourceOp, typename StockOp, typename ExtendOp>
LogicalResult registerPredicate(ImplementationRegistry &registry, llvm::StringRef op) {
  // 两侧的掩码不是同一个：reg 侧多 bf16（三个谓词的 `reg_dtypes` 都列着它），
  // mem 侧的 `REG_ASC_IR` 没有。共用一个常量就只能二选一——给 mem 侧也开 bf16 是一句
  // 没人读的假话（那一侧的包装编不过），不给 reg 侧开则是真损失（legacy 收的类型上
  // 答 unsupported，整图静默回落）。
  if (failed(registerUnary<SourceOp, StockOp>(registry, (op + ".tensor").str(), ApiFamily::RegBased,
                                              ResultElementRule::Uint8, kFloatElementTypes | kBF16))) {
    return failure();
  }
  return registerUnary<SourceOp, ExtendOp, /*TakesTmpBuffer=*/true>(
      registry, (op + ".tensor.extend").str(), ApiFamily::MemBased, ResultElementRule::Uint8, kFloatElementTypes);
}

}  // namespace

LogicalResult registerUnaryElementwiseImplementations(ImplementationRegistry &registry,
                                                      ascend::instruction::SelectionPolicyRegistry *) {
  // 两张支持表给的是同一个 `AscendC::` 调用，都不要 tmp buffer：一个 id，不问 feature。
  if (failed(registerUnary<ExpOp, ascendc::ExpL2Op>(registry, "exp.tensor", ApiFamily::Any, ResultElementRule::Same,
                                                    kFloatElementTypes)) ||
      failed(registerUnary<LnOp, ascendc::LnL2Op>(registry, "ln.tensor", ApiFamily::Any, ResultElementRule::Same,
                                                  kFloatElementTypes)) ||
      failed(registerUnary<ReluOp, ascendc::ReluL2Op>(registry, "relu.tensor", ApiFamily::Any, ResultElementRule::Same,
                                                      kFloatAndI32ElementTypes)) ||
      failed(registerUnary<SqrtOp, ascendc::SqrtL2Op>(registry, "sqrt.tensor", ApiFamily::Any, ResultElementRule::Same,
                                                      kFloatElementTypes))) {
    return failure();
  }

  // 标准调用只在 reg-base 一侧，mem-based 一侧走包装并要一块 tmp buffer。
  if (failed(registerUnary<AbsOp, ascendc::AbsL2Op>(registry, "abs.tensor", ApiFamily::RegBased,
                                                    ResultElementRule::Same, kAbsRegElementTypes)) ||
      failed(registerUnary<AbsOp, ascendc::AbsExtendOp, /*TakesTmpBuffer=*/true>(
          registry, "abs.tensor.extend", ApiFamily::MemBased, ResultElementRule::Same, kFloatAndI32ElementTypes)) ||
      failed(registerUnary<ReciprocalOp, ascendc::ReciprocalL2Op>(
          registry, "reciprocal.tensor", ApiFamily::RegBased, ResultElementRule::Same, kFloatElementTypes | kBF16)) ||
      failed(registerUnary<ReciprocalOp, ascendc::ReciprocalExtendOp, /*TakesTmpBuffer=*/true>(
          registry, "reciprocal.tensor.extend", ApiFamily::MemBased, ResultElementRule::Same, kFloatElementTypes))) {
    return failure();
  }

  // 两侧都只有包装，签名也一样，只有扩展头不同——id 用 `extend` / `extend_reg_base`
  // 区分，说的是支持表，不是芯片。
  if (failed(registerUnary<NegOp, ascendc::NegExtendOp>(registry, "neg.tensor.extend", ApiFamily::MemBased,
                                                        ResultElementRule::Same, kNegElementTypes)) ||
      failed(registerUnary<NegOp, ascendc::NegExtendRegBaseOp>(registry, "neg.tensor.extend_reg_base",
                                                               ApiFamily::RegBased, ResultElementRule::Same,
                                                               kNegElementTypes | kBF16))) {
    return failure();
  }

  // 两侧都要 tmp buffer，差的只是标准调用还是包装。
  if (failed(registerUnary<afir::SigmoidOp, ascendc::SigmoidOp, /*TakesTmpBuffer=*/true>(
          registry, "sigmoid.tensor", ApiFamily::RegBased, ResultElementRule::Same, kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::SigmoidOp, ascendc::SigmoidExtendOp, /*TakesTmpBuffer=*/true>(
          registry, "sigmoid.tensor.extend", ApiFamily::MemBased, ResultElementRule::Same, kFloatElementTypes))) {
    return failure();
  }

  // 浮点谓词族：结果 dtype 与输入不同（ui8）。标准调用在 reg-base 一侧且不要 tmp buffer。
  if (failed(registerPredicate<IsnanOp, ascendc::IsNanL2Op, ascendc::IsnanExtendOp>(registry, "isnan")) ||
      failed(registerPredicate<IsFiniteOp, ascendc::IsFiniteL2Op, ascendc::IsFiniteExtendOp>(registry, "isfinite")) ||
      failed(registerPredicate<IsInfOp, ascendc::IsInfL2Op, ascendc::IsInfExtendOp>(registry, "isinf"))) {
    return failure();
  }

  // `gelu`：两张支持表给的是同一个 `AscendC::Gelu`，也不引扩展头，所以家族恒为 `Any`、
  // 只有一个 implementation——但它要一块 tmp buffer，这一点与 exp / ln 那一档不同。
  // 两侧 `CalcTmpBufSize` 给的尺寸不同（`GetInputDataSizeTmpBuffer` vs `CalcVoidTmpSizeV2`
  // 的 0），AFIR 只带 id 不带尺寸，这个差别落在 ATT 一侧。
  //
  // 只收 f16 / f32——两侧的 `REG_ASC_IR(Gelu)` 都只注册了这两个。
  if (failed(registerUnary<afir::GeluOp, ascendc::GeluOp, /*TakesTmpBuffer=*/true>(
          registry, "gelu.tensor", ApiFamily::Any, ResultElementRule::Same, kF16 | kF32))) {
    return failure();
  }

  // `sign`：两侧同名（`SignExtend`）、签名也一样、都要 tmp buffer，只有扩展头与收的元素
  // 类型不同——一元里的 `remainder`。
  //
  // 这是第一个声明两个业务头的 op：包装按 dtype 分流，f16 / f32 直接转发
  // `AscendC::Sign`，其余先 `Cast`，所以 `cast.h` 与 `sign*.h` 两个都要。
  //
  // reg 侧多收 ui8；i64 / bf16 它也收，但 AFIR 建模不出来。
  if (failed(registerUnary<afir::SignOp, ascendc::SignExtendOp, /*TakesTmpBuffer=*/true>(
          registry, "sign.tensor.extend", ApiFamily::MemBased, ResultElementRule::Same, kF16 | kF32 | kI32)) ||
      failed(registerUnary<afir::SignOp, ascendc::SignExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "sign.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kF16 | kF32 | kI32 | kUI8 | kBF16))) {
    return failure();
  }

  // `tanh` / `erf`：**`MathLibraryOp` 一类的头两个**，也是这条路上第一次物化
  // `isReuseSource`（见 `UnaryElementwiseConversionPattern` 的第四根轴）。
  //
  // 与 `sigmoid` 同形——两侧都要 tmp buffer，一侧标准调用、一侧包装——但方向是反的：
  // 标准调用在 mem 侧（`AscendC::Tanh` / `AscendC::Erf`，不引任何扩展头），包装在
  // reg 侧（`TanhExtend` / `ErfExtend`，`tanh_reg_base.h` / `erf_reg_base.h`）。
  // `abs` / `reciprocal` / 谓词族都是标准调用在 reg 侧，别照着它们猜。
  //
  // 于是 mem 侧那个 implementation 是**第一个 `ApiFamily::MemBased` 却一个扩展头都不带
  // 的**。`headersMatchFamily` 对「无头」的处理是直接放行（`hasAnyHeader` 两个都空就
  // 返回 true，不看 family），所以它过得去——与 `abs.tensor` / `sigmoid.tensor` 那几个
  // `RegBased` 且无头的是同一条路径，只是换了一边。别因为它无头就写成 `Any`：`Any` 的
  // 语义是「两张表给的是同一个调用」，而 reg 侧这里是另一个函数。
  //
  // 两侧都只收 f16 / f32。reg 侧的 `REG_ASC_IR` 还收 bf16，AFIR 建模不出来。
  if (failed(registerUnary<afir::TanhOp, ascendc::TanhOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "tanh.tensor", ApiFamily::MemBased,
                                                             ResultElementRule::Same, kF16 | kF32)) ||
      failed(registerUnary<afir::TanhOp, ascendc::TanhExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "tanh.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kF16 | kF32 | kBF16)) ||
      failed(registerUnary<afir::ErfOp, ascendc::ErfOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "erf.tensor", ApiFamily::MemBased,
                                                             ResultElementRule::Same, kF16 | kF32)) ||
      failed(registerUnary<afir::ErfOp, ascendc::ErfExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "erf.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF16 | kF32 | kBF16))) {
    return failure();
  }

  // `ndtr`：特殊函数一元族的第一个，形状与上面 `erf` 的 reg 那一半逐项相同——
  // `NdtrExtend(dst, src, tmp, count)`、tmp 尺寸 0 但形参在、不物化 `isReuseSource`。
  //
  // 唯一的新东西是它只有这一侧：legacy 的 mem 侧（`ascir_builtin_ops_v1.cpp`）压根
  // 没有 `Ndtr` 这个注册，所以不存在第二行可写。`EveryElementwiseOpCoversBothApiFamilies`
  // 那条断言因此对它不成立，豁免写在那条用例里、连着「必须真的还缺一侧」的反向断言，
  // 不是在这里补一个假的 mem 侧实现——补了就会在 mem 目标上选中它，生成一个
  // `AscendC::` 里根本不存在的调用。
  //
  // 只收 f32：`REG_ASC_IR(Ndtr)` 的 `TensorType` 就只有 `DT_FLOAT`，不是 AFIR 的支持表
  // 拦的，所以这里不该跟着 `erf` 写 `kF16 | kBF16`——写宽了会比 legacy 宽，
  // `ElementTypeMatrixTest` 那条「ODS ⊆ legacy 两列并集」当场红。
  //
  // 同族其余三个（第 2 批）形态与它逐项相同，只有包装名与扩展头不同；`airy_ai` 的包装要
  // 两个头（utils 在前），那一处的差异全在 ODS，这里看不出来，也不需要看出来。
  if (failed(registerUnary<afir::NdtrOp, ascendc::NdtrExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "ndtr.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::NdtriOp, ascendc::NdtriExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "ndtri.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::LogNdtrOp, ascendc::LogNdtrExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "log_ndtr.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::I0Op, ascendc::I0ExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "i0.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::AiryAiOp, ascendc::AiryAiExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "airy_ai.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::BesselJ0Op, ascendc::BesselJ0ExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "bessel_j0.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::BesselJ1Op, ascendc::BesselJ1ExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "bessel_j1.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::I0eOp, ascendc::I0eExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "i0e.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::I1eOp, ascendc::I1eExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "i1e.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::ModifiedBesselI0Op, ascendc::ModifiedBesselI0ExtendRegBaseOp,
                           /*TakesTmpBuffer=*/true>(registry, "modified_bessel_i0.tensor.extend_reg_base",
                                                    ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::ModifiedBesselI1Op, ascendc::ModifiedBesselI1ExtendRegBaseOp,
                           /*TakesTmpBuffer=*/true>(registry, "modified_bessel_i1.tensor.extend_reg_base",
                                                    ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::ModifiedBesselK0Op, ascendc::ModifiedBesselK0ExtendRegBaseOp,
                           /*TakesTmpBuffer=*/true>(registry, "modified_bessel_k0.tensor.extend_reg_base",
                                                    ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::ModifiedBesselK1Op, ascendc::ModifiedBesselK1ExtendRegBaseOp,
                           /*TakesTmpBuffer=*/true>(registry, "modified_bessel_k1.tensor.extend_reg_base",
                                                    ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      // 第 6 批（一元六个里的六个；`log10` 不在这一批——它是 `MathLibraryOp` 却不要 tmp，
      // 正是 `UnaryElementwiseConversionPattern` 的 `static_assert` 挡住的那个空组合）。
      //
      // `erfinv` 的包装有 3 参和 4 参两个重载，legacy 生成的是 3 参那个，所以不带 tmp。
      failed(registerUnary<afir::ErfinvOp, ascendc::ErfinvExtendRegBaseOp>(
          registry, "erfinv.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::Exp2Op, ascendc::Exp2ExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "exp2.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      // `log2` / `floor` / `lgamma` 没有包装，走标准数学库调用，所以与 `tanh` / `erf` 的
      // mem 那一半同形：要 tmp（`floor` 那块尺寸是 0，形参在就照收）且要物化
      // `isReuseSource`。**但 family 是 `RegBased`**：legacy 只有 reg 侧注册它们。
      failed(registerUnary<afir::Log2Op, ascendc::Log2Op, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "log2.tensor", ApiFamily::RegBased,
                                                             ResultElementRule::Same, kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::FloorOp, ascendc::FloorOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "floor.tensor", ApiFamily::RegBased,
                                                             ResultElementRule::Same, kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::LgammaOp, ascendc::LgammaOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "lgamma.tensor", ApiFamily::RegBased,
                                                             ResultElementRule::Same, kFloatElementTypes | kBF16))) {
    return failure();
  }

  // 第 7 批（一元 24 个），全部 reg-only。三段：
  //
  //   1. 19 个走 fork 的包装、要 tmp。**它们在 `Adv/Math.td` 多半有同名的 `UnaryMathOp`，
  //      但那个是标准调用，不是这里的可用实现**——legacy 的 `GetApiName` 给的是 `XxxExtend`、
  //      `LoadApiHeaderFiles` 给了 `xxx_reg_base.h`。`ceil` / `round` / `cos` 这些照名字
  //      挑就会挑到标准调用那一个，而 reg 侧根本没有它。
  //   2. `digamma` / `erfc` / `trunc` 没有包装，与第 6 批的 `log2` / `floor` / `lgamma`
  //      逐项同形：标准调用 + tmp + 物化 `isReuseSource`，并且要进 `kIsStockCall`。
  //   3. `expm1` / `log1p` 走包装但不要 tmp，与 `erfinv` 同形。
  //
  // dtype 分两档：三角 / 双曲 / 取整那一串与 `digamma` / `erfc` / `trunc` / `expm1` /
  // `log1p` 收 {f16, f32, bf16}，Bessel 那五个 legacy 的域只有 `DT_FLOAT`。
  if (failed(registerUnary<afir::AcosOp, ascendc::AcosExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "acos.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::AcoshOp, ascendc::AcoshExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "acosh.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::AsinOp, ascendc::AsinExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "asin.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::AsinhOp, ascendc::AsinhExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "asinh.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::AtanOp, ascendc::AtanExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "atan.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::AtanhOp, ascendc::AtanhExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "atanh.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::CeilOp, ascendc::CeilExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "ceil.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::CosOp, ascendc::CosExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "cos.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::CoshOp, ascendc::CoshExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "cosh.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::ErfcxOp, ascendc::ErfcxExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "erfcx.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::RoundOp, ascendc::RoundExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "round.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::SinOp, ascendc::SinExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "sin.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::SinhOp, ascendc::SinhExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "sinh.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::TanOp, ascendc::TanExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "tan.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::BesselY0Op, ascendc::BesselY0ExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "bessel_y0.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::BesselY1Op, ascendc::BesselY1ExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
          registry, "bessel_y1.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(
          registerUnary<afir::SphericalBesselJ0Op, ascendc::SphericalBesselJ0ExtendRegBaseOp, /*TakesTmpBuffer=*/true>(
              registry, "spherical_bessel_j0.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Same,
              kF32)) ||
      failed(registerUnary<afir::ScaledModifiedBesselK0Op, ascendc::ScaledModifiedBesselK0ExtendRegBaseOp,
                           /*TakesTmpBuffer=*/true>(registry, "scaled_modified_bessel_k0.tensor.extend_reg_base",
                                                    ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::ScaledModifiedBesselK1Op, ascendc::ScaledModifiedBesselK1ExtendRegBaseOp,
                           /*TakesTmpBuffer=*/true>(registry, "scaled_modified_bessel_k1.tensor.extend_reg_base",
                                                    ApiFamily::RegBased, ResultElementRule::Same, kF32)) ||
      failed(registerUnary<afir::DigammaOp, ascendc::DigammaOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "digamma.tensor", ApiFamily::RegBased,
                                                             ResultElementRule::Same, kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::ErfcOp, ascendc::ErfcOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "erfc.tensor", ApiFamily::RegBased,
                                                             ResultElementRule::Same, kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::TruncOp, ascendc::TruncOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/true>(registry, "trunc.tensor", ApiFamily::RegBased,
                                                             ResultElementRule::Same, kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::Expm1Op, ascendc::Expm1ExtendRegBaseOp>(registry, "expm1.tensor.extend_reg_base",
                                                                         ApiFamily::RegBased, ResultElementRule::Same,
                                                                         kFloatElementTypes | kBF16)) ||
      failed(registerUnary<afir::Log1pOp, ascendc::Log1pExtendRegBaseOp>(registry, "log1p.tensor.extend_reg_base",
                                                                         ApiFamily::RegBased, ResultElementRule::Same,
                                                                         kFloatElementTypes | kBF16))) {
    return failure();
  }

  // 第 8 批（一元 1 个）。`log10` 是第 6 批那段注释点名留下的那个空组合：标准调用、
  // 不要 tmp。那一轮以为它只能是 `MathLibraryOp`，于是撞上
  // `UnaryElementwiseConversionPattern` 的 `static_assert`；这里换的是**可用实现**而不是
  // 那条 `static_assert`——`ascendc::Log10L2Op` 是 Autofuse 子树里按 `UnaryL2Op` 新建
  // 的 def，没有 `isReuseSource` 操作数，所以两个模板位都是缺省，生成的就是
  // `AscendC::Log10(dst, src, count)`，与 legacy 的 `UnaryApiCall` 逐项同形。
  //
  // family 是 `RegBased`：legacy 只有 reg 侧注册 `Log10`（对照表 `mem_present=0`）。
  if (failed(registerUnary<afir::Log10Op, ascendc::Log10L2Op>(registry, "log10.tensor", ApiFamily::RegBased,
                                                              ResultElementRule::Same, kFloatElementTypes | kBF16))) {
    return failure();
  }

  // 第 9 批（一元 8 个，多项式族）。八行逐项同形，只差名字——`n` 走模板实参位那件事
  // 收在 `registerUnaryTemplateAttr` 里，所以这里没有模板轴可写错。
  // dtype 只有 f32，是 legacy 自己的域（八个的 `REG_ASC_IR` 都是 `TensorType{DT_FLOAT}`）。
  if (failed(registerUnaryTemplateAttr<afir::ChebyshevPolynomialTOp, ascendc::ChebyshevPolynomialTExtendRegBaseOp>(
          registry, "chebyshev_polynomial_t.tensor.extend_reg_base")) ||
      failed(registerUnaryTemplateAttr<afir::ChebyshevPolynomialUOp, ascendc::ChebyshevPolynomialUExtendRegBaseOp>(
          registry, "chebyshev_polynomial_u.tensor.extend_reg_base")) ||
      failed(registerUnaryTemplateAttr<afir::ChebyshevPolynomialVOp, ascendc::ChebyshevPolynomialVExtendRegBaseOp>(
          registry, "chebyshev_polynomial_v.tensor.extend_reg_base")) ||
      failed(registerUnaryTemplateAttr<afir::ChebyshevPolynomialWOp, ascendc::ChebyshevPolynomialWExtendRegBaseOp>(
          registry, "chebyshev_polynomial_w.tensor.extend_reg_base")) ||
      failed(registerUnaryTemplateAttr<afir::ShiftedChebyshevPolynomialTOp,
                                       ascendc::ShiftedChebyshevPolynomialTExtendRegBaseOp>(
          registry, "shifted_chebyshev_polynomial_t.tensor.extend_reg_base")) ||
      failed(registerUnaryTemplateAttr<afir::ShiftedChebyshevPolynomialUOp,
                                       ascendc::ShiftedChebyshevPolynomialUExtendRegBaseOp>(
          registry, "shifted_chebyshev_polynomial_u.tensor.extend_reg_base")) ||
      failed(registerUnaryTemplateAttr<afir::ShiftedChebyshevPolynomialVOp,
                                       ascendc::ShiftedChebyshevPolynomialVExtendRegBaseOp>(
          registry, "shifted_chebyshev_polynomial_v.tensor.extend_reg_base")) ||
      failed(registerUnaryTemplateAttr<afir::ShiftedChebyshevPolynomialWOp,
                                       ascendc::ShiftedChebyshevPolynomialWExtendRegBaseOp>(
          registry, "shifted_chebyshev_polynomial_w.tensor.extend_reg_base"))) {
    return failure();
  }

  // step 2a（一元 2 个，取整到 i32）。两个的结果规则相同（`Int32`）而可用实现不同：
  // `ceil2int` 走 fork 的 `Ceil2IntExtend` 包装（RoundMode 写在包装体里），`floor_to_int`
  // 走标准调用 `AscendC::Cast` + 固定 `CAST_FLOOR`。所以一个用 `registerUnary`、一个用
  // `registerUnaryFixedRoundMode`——这正是「结果规则」与「可用实现」是两根独立轴的例子。
  //
  // 输入域取 legacy 自己的（两个的 `T1` 都是 `{DT_FLOAT, DT_FLOAT16, DT_BF16}`）。
  if (failed(registerUnary<afir::Ceil2IntOp, ascendc::Ceil2IntExtendRegBaseOp>(
          registry, "ceil2int.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Int32,
          kFloatElementTypes | kBF16)) ||
      failed(registerUnaryFixedRoundMode<afir::FloorToIntOp, ascendc::RoundMode::CAST_FLOOR>(
          registry, "floor_to_int.tensor", ResultElementRule::Int32, kFloatElementTypes | kBF16))) {
    return failure();
  }

  // `cast`：**这条路上第一个输入与结果必须不同 dtype 的 op**，也是第一个合法组合是一张
  // 逐行对表而不是一个集合的 op（`ResultElementRule::CastPair`，表在 `Ops/Cast.cpp`）。
  //
  // 两侧都走包装、都叫 `CastExtend`，但签名完全不同——这也是第一次：
  //
  //   - mem 侧 `CastExtend(dst, src, tmp, size)`（`cast.h`），要一块 tmp buffer
  //     （`CalcCastTmpSize`；中间要转两道的那些 dtype 对拿它装中间结果）；
  //   - reg 侧 `CastExtend(dst, src, {dims}, {out_stride}, {in_stride})`
  //     （`cast_reg_base.h`），不要 tmp，三个数组实参由 emitter 按连续形态打字面量。
  //
  // 上面那三个 `ascendc.cast_l*` 不是这里的可用实现：它们生成的是标准的 `AscendC::Cast`、
  // 要一个 `RoundMode` 实参，而两份包装都是自己按 (输入, 输出) 推 `RoundMode` 的。
  //
  // 集合传 0：`CastPair` 那一档根本不读它（见 `areElementTypesSupported`），给一个真集合
  // 只会让人以为它参与判据。0 同时是 `ElementTypeMatrixTest` 认的「这一档不按集合判」的
  // 标记，那条断言据此跳过这两个 id，并用一条反向断言盯住跳过的名单不会悄悄变长。
  if (failed(registerUnary<afir::CastOp, ascendc::CastExtendContiguousOp, /*TakesTmpBuffer=*/true>(
          registry, "cast.tensor.extend", ApiFamily::MemBased, ResultElementRule::CastPair, 0U)) ||
      failed(registerUnary<afir::CastOp, ascendc::CastExtendContiguousRegBaseOp>(
          registry, "cast.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::CastPair, 0U))) {
    return failure();
  }

  // `rsqrt` / `logical_not`：**这条路上头两个要「预先填好 1 的常量 block」的 op**
  // （见 `ConstBlockTensor.h`）。那一块不是第二块 tmp buffer，所以它不在 AFIR 的操作数
  // 里，也不进 `checkTmpBuffer` 那条判据——两个 op 的 tmp buffer 仍然各只有一块。
  //
  // `rsqrt` 两侧差得比一元里任何一对都远：reg 侧是标准调用 `AscendC::Rsqrt`，不引头、
  // 不要 tmp、不要常量；mem 侧走 `RsqrtExtend`，三样全要。那块常量是 `float`，与
  // 正在算的 dtype 无关——包装一律先把中间结果转成 float 再 `Div`。
  if (failed(registerUnary<afir::RsqrtOp, ascendc::RsqrtL2Op>(registry, "rsqrt.tensor", ApiFamily::RegBased,
                                                              ResultElementRule::Same, kF16 | kF32 | kBF16)) ||
      failed(registerUnary<afir::RsqrtOp, ascendc::RsqrtExtendOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/false, ConstBlockOnes::F32>(
          registry, "rsqrt.tensor.extend", ApiFamily::MemBased, ResultElementRule::Same, kF16 | kF32))) {
    return failure();
  }

  // `logical_not`：**一元里第一个两侧结果 dtype 关系不同的 op**（二元那边是 `truediv`）。
  // mem 侧两端同 dtype（`ResultElementRule::Same`），reg 侧结果恒为 ui8
  // （`Uint8Logical`）——所以它在 AFIR 那一层不声明 `InferTypeOpInterface`，理由写在
  // `AFIR_BufferizableLogicalNotOp` 头上。
  //
  // 两侧的名字也不是同一个：mem 侧是裸名 `LogicalNot`（不带 `Extend` 后缀，也不是
  // `AscendC::LogicalNot`——包装是全局函数），reg 侧才叫 `LogicalNotExtend`。
  // 收的元素类型 mem 侧 {f16, f32, ui8, i16, i32}，reg 侧多一个 ui16（它还收 i8 / i64 /
  // ui32 / ui64 / bf16 / bool，AFIR 建模不出来）。那块常量是 `half`。
  if (failed(registerUnary<afir::LogicalNotOp, ascendc::LogicalNotExtendOp, /*TakesTmpBuffer=*/true,
                           /*MaterializesReuseSource=*/false, ConstBlockOnes::F16>(
          registry, "logical_not.tensor.extend", ApiFamily::MemBased, ResultElementRule::Same,
          kF16 | kF32 | kUI8 | kI16 | kI32)) ||
      failed(registerUnary<afir::LogicalNotOp, ascendc::LogicalNotExtendRegBaseOp>(
          registry, "logical_not.tensor.extend_reg_base", ApiFamily::RegBased, ResultElementRule::Uint8,
          kF16 | kF32 | kUI8 | kUI16 | kI16 | kI32 | kBF16))) {
    return failure();
  }

  // 一元 + 一个来自 ODS 属性的实参。两张支持表给的是同一个 `AscendC::LeakyRelu`,
  // 都不要 tmp buffer，也不引扩展头。
  if (failed(registerUnaryScalarAttr<LeakyReluOp, ascendc::LeakyReluL2Op>(registry, "leaky_relu.tensor"))) {
    return failure();
  }

  // step 4。`bitwise_not` 只有 reg 一侧（mem 侧 legacy 不注册），走 fork 的包装
  // `BitwiseNotExtend(dst, src, count)`（`api_regbase/bitwise_not.h`），不带 tmp。
  // legacy 另有一条走 `MicroAPI::Not` 的 VF 融合路，本 Op 不建模那一条。
  return registerUnary<afir::BitwiseNotOp, ascendc::BitwiseNotExtendRegBaseOp>(
      registry, "bitwise_not.tensor", ApiFamily::RegBased, ResultElementRule::Same, kUI8 | kI16 | kUI16 | kI32);
}

}  // namespace mlir::ascend::kernel
