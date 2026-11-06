/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- BinaryElementwise.cpp - AFIR binary elementwise lowering -*- C++ -*-===//
//
// 结构与 `UnaryElementwise.cpp` 一致：每个 AFIR 二元 op 在这里注册一到两个
// implementation，一个 implementation 由「建哪个 AscendC op」「属于哪张扩展 API 供给
// 表」「要不要 tmp buffer」三件独立的事确定。判据写在 `ElementwiseApiFamily.h`，一元和
// 二元共用同一份。
//
// 比一元路径多出来的只有「有两个输入」带来的两条检查：两个输入的元素类型要相同，两个
// 输入的形状要和目的地相同。
//
// 操作数用 `afir::getOperandPartition()` 切，不用位置下标。二元 op 今天的布局确实是
// `lhs, rhs, dst`，但那是 ODS 的 `AttrSizedOperandSegments` 算出来的结果，不是这里可以
// 直接假定的前提：多一块 tmp buffer，布局就变了。下标写死不会报错，只会把 tmp buffer
// 当成 rhs 算进去。
//
// 本文件的 stable id 与注释都不出现芯片名。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/BinaryElementwise.h"
#include "conversion/AFIRToAscendC/VectorizedLoops.h"
#include "target/Ascend/AscendTargetProfile.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
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

// 一个 implementation 收哪几种元素类型。位的拼写在 `ElementwiseApiFamily.h`，这里只写
// 集合。
//
// **这是一张集合，不是几个能起名字的族。** 原先想给反复出现的那几个集合各起一个名字，
// 接到第三个 op 就不成立了——`bitwise_and` 是 {ui8, i16, ui16, i32}，`floor_div` 的
// mem 侧是 {f16, f32}、reg 侧是 {f16, f32, i16, i32}，`remainder` 的 mem 侧是
// {f32, i32}。这些集合两两都不是包含关系，也没有共同的概念，起名字只是把集合又拼写
// 一遍。所以注册处直接写出那几种类型，看得见、也对得上 legacy 的 `REG_ASC_IR`——
// **每一处的取值都是从那里抄来的，不是推出来的。**
//
// 同一个 AFIR op 的两个 implementation 可以收不同的集合（`floor_div` / `remainder` 都
// 是）。ODS 那一层写并集，这里各说各的一半。
using elementwise::isSupportedElementType;
using elementwise::kBF16;
using elementwise::kF16;
using elementwise::kF32;
using elementwise::kI16;
using elementwise::kI32;
using elementwise::kUI16;
using elementwise::kUI8;

// add / mul / maximum / minimum 收的那一组。它是唯一一个真的重复出现、也真的有个名字
// 的集合，所以留一个常量；其余的在注册处写全。
constexpr uint32_t kArithmeticElementTypes = kF16 | kF32 | kI16 | kI32;

// reg 侧的 `sub` / `floor_div`：legacy 在那一侧比 mem 多收 ui8 / ui16。两侧写成同一个
// 集合会把 mem 侧放宽到它供不出来的类型上，所以单开一个。
//
// **bf16 不在这里，尽管 `sub` 的 reg 侧收它。** 两个 op 在这一位上不一样：
// `ascir_op_lowering_matrix.csv` 里 `Sub` 的 `reg_dtypes` 有 bf16，`FloorDiv` 的没有。
// 2026-10-20 一度把 `| kBF16` 加在这个共用常量上，于是 `floor_div` 被静默放宽到一个 legacy
// 两侧都不供的类型——两份拷贝（这里和 ODS）同时错，所以「ODS == 注册处」那种等号判据照不
// 出来，是 `OdsElementTypesAgreeWithRegistrationsAndLegacy` 的判据二（ODS ⊆ legacy 两列并集）
// 抓到的。**两个 op 不一致的位要写在各自的注册处，不要塞进共用常量。**
constexpr uint32_t kArithmeticRegElementTypes = kArithmeticElementTypes | kUI8 | kUI16;

// 调度分给这个节点的 tmp buffer 与选中的调用之间是什么关系。
//
// **原先这里是一位布尔，Sub / Div 把它顶成了三态。** 那一位同时说了两件事：capability
// 收不收 tmp 操作数、建 target op 时传不传那个实参。两件事此前从来一致，直到 Sub / Div:
// legacy 的 `CalcBinaryApiTmpSize` 对它们的 tensor-tensor 形态也分了一块（32 字节），
// 而这条路发的是标准的 `AscendC::Sub` / `AscendC::Div`，四个实参，没有这块的位置。
// 真正用它的是标量形态的 `Subs` / `Divs`。
//
// 这块 UB 不能装作不存在：bridge 已经把它挂上了 AFIR op（`node.tmp_buffers` 是调度的
// 答案，不是 bridge 的），capability 那边答「不要 tmp」会直接判成 unsupported。也不能
// 硬塞给调用——那是另一个重载，或者根本不存在。所以第三档是必要的，不是图省事。
enum class TmpBufferUse {
  // 调度不分，调用不收。
  None,
  // 调度分了，调用收——包装的签名里有这个形参（哪怕包装自己没用它，如
  // `BitwiseAndExtend`）。
  Passed,
  // 调度分了，调用不收。今天只有 mem 侧的 `sub` / `div` 落在这一档。
  Unused,
};

constexpr bool schedulerGivesTmpBuffer(TmpBufferUse use) {
  return use != TmpBufferUse::None;
}

constexpr bool callTakesTmpBuffer(TmpBufferUse use) {
  return use == TmpBufferUse::Passed;
}

// **哪些操作数可以是标量**，三个取值与 ODS 上 `scalarOperand` 那三个字符串一一对应
// （Elementwise.td）——同一根轴只此一处说法，两边对不上时 `getTensorTypes` 当场判
// unsupported，不会静默。
//
//   `None` 两个都是 tile；`Rhs` 标量只能在右；`Any` 哪一边都可以，但恰好一个。
//
// `None` 与另外两档建的是不同的 AscendC op（`Add` / `Adds`）。**而 `Any` 的两个方向建
// 的是同一个 op、同一组实参**：矩阵上带 `swap(2nd)` 的 8 个 op 全是可交换的（理由与
// 出处写在 Elementwise.td 的 `scalarOperand` 上），legacy 的 `binary_api_call.cpp` 对它们
// 也是交换完发同一个不带模板实参的 `Adds`。所以标量在左不是第二个 implementation，
// 只是发射时把有形状的那个操作数放到第一个实参上（`emissionOrder`）。
//
// 不可交换的标量形态（Div / Sub / TrueDiv 的 `is_scalar_latter`）今天一个都没接；等接的
// 时候它们要的是第二种调用形状，那才是新的一档。
enum class ScalarOperand {
  None,
  Rhs,
  Any,
};

// 结果元素类型与输入的关系。`Same` 是这条路上此前的全部；`Uint8Logical` 是
// `logical_and` / `logical_or`——结果恒为 `ui8`，与输入 dtype 无关。
//
// **与一元那条路的 `Uint8Predicate` 不是同一个东西，别合并。** 谓词族（isnan 一族）
// 的输入集合被那条判据自己钉死成浮点，于是它不看注册处给的集合；逻辑族的输入集合
// 两侧不同（mem 侧五个，reg 侧更宽），仍然要按 implementation 各说各的。所以这里
// 只接管「结果是什么」，输入照旧走 `elementTypes`。
//
// `SameOrF32ForI32` 是 `truediv`：结果 dtype 既不是「与输入相同」也不是一个常数，而是
// **逐行**的——`{f16->f16, f32->f32, i32->f32}`。这一档只加了 i32 那一行，没有改成
// 一张 (输入, 结果) 对表：对表化要动全部 45 个注册点，是它自己的一件事。
// 与一元那条路同一个枚举，也与 ODS 的 `resultRule` 同一个：见
// `AFIRElementwiseSupport.h` 的 `ResultElementRule`。
using afir::elementwise::ResultElementRule;

struct BinaryElementwiseMapping {
  ApiFamily family = ApiFamily::Any;
  ScalarOperand scalarOperand = ScalarOperand::None;
  ResultElementRule resultElementType = ResultElementRule::Same;
  uint32_t elementTypes = kArithmeticElementTypes;
  // 选中的 API 要不要调用方给的 tmp buffer。唯一的真源是 pattern 的模板实参，注册处由
  // 同一个实参同时决定这里和建 op 的实参个数。
  bool needsTmpBuffer = false;
  uint32_t resultIndex = 0;
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
};

// 两个输入 + 一个目的地的类型，**按发射顺序**取出来而不是按 AFIR 操作数顺序：`lhs`
// 恒是有形状的那一个（发射时的第一个输入实参），`rhs` 是第二个实参——tensor 形态下它
// 是第二块 tile，标量形态下它为空、类型在 `rhsElement` 里。
//
// 按发射顺序而不是操作数顺序，是为了让下游只有一套代码：capability 的形状 / 元素类型
// 判据、lowering 建 op 都不必知道标量落在哪一边，那件事只在 `emissionOrder` 一处分流。
struct BinaryTensorTypes {
  ascendc::LocalTensorType lhs;
  ascendc::LocalTensorType rhs;
  Type rhsElement;
  ascendc::LocalTensorType destination;
};

// AFIR 的两个输入下标，**按发射顺序**：`tile` 恒是有形状的那一个，放到第一个实参上。
// `std::nullopt` 表示实际操作数与本形态对不上——tensor 形态收到标量、标量形态收到两块
// tile、`Any` 收到两个标量，三种都在这里一次挡掉。
struct BinaryOperandOrder {
  unsigned tile;
  unsigned other;
};

std::optional<BinaryOperandOrder> emissionOrder(ScalarOperand form, bool lhsIsTile, bool rhsIsTile) {
  switch (form) {
    case ScalarOperand::None:
      return lhsIsTile && rhsIsTile ? std::optional<BinaryOperandOrder>({0U, 1U}) : std::nullopt;
    case ScalarOperand::Rhs:
      return lhsIsTile && !rhsIsTile ? std::optional<BinaryOperandOrder>({0U, 1U}) : std::nullopt;
    case ScalarOperand::Any:
      if (lhsIsTile != rhsIsTile) {
        return lhsIsTile ? std::optional<BinaryOperandOrder>({0U, 1U}) : std::optional<BinaryOperandOrder>({1U, 0U});
      }
      return std::nullopt;
  }
  return std::nullopt;
}

// 结果类型不在这里比：`verifyBufferizedElementwise` 已经强制「结果类型 == 目的地类型」，
// 而目的地在下面逐条比过了。
FailureOr<BinaryTensorTypes> getTensorTypes(Operation *op, ScalarOperand scalarOperand) {
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(op);
  if (failed(partition) || partition->inputCount != 2 || partition->destinationCount != 1) {
    return failure();
  }
  OperandRange inputs = partition->inputs(op);
  // 形态与实际操作数必须对上，三种走法一次挡掉（见 `emissionOrder`）：tensor 形态的
  // implementation 收到标量会建出一个参数对不上的 `Add`，标量形态收到 tile 会把整块当成
  // 一个值，而「两个都是标量」连一块 tile 都没有。
  const std::optional<BinaryOperandOrder> order =
      emissionOrder(scalarOperand, isa<ascendc::LocalTensorType>(inputs[0].getType()),
                    isa<ascendc::LocalTensorType>(inputs[1].getType()));
  if (!order) {
    return failure();
  }
  BinaryTensorTypes types;
  auto lhs = dyn_cast<ascendc::LocalTensorType>(inputs[order->tile].getType());
  auto destination = dyn_cast<ascendc::LocalTensorType>(partition->destinations(op)[0].getType());
  if (!lhs || !destination) {
    return failure();
  }
  const Type rhsType = inputs[order->other].getType();
  types.lhs = lhs;
  types.rhs = dyn_cast<ascendc::LocalTensorType>(rhsType);
  types.rhsElement = types.rhs ? types.rhs.getElementType() : rhsType;
  types.destination = destination;
  return types;
}

// 两个输入与目的地的元素类型。两个输入之间恒等——legacy 两侧都把它们绑在同一个模板
// 形参上，与结果 dtype 是不是同一个无关。目的地按 `relation` 分：`Same` 跟着输入，
// `Uint8Logical` 恒为 `ui8`，`SameOrF32ForI32` 只有 i32 那一行出 f32。
bool areElementTypesSupported(ResultElementRule relation, uint32_t allowed, Type lhsElement, Type rhsElement,
                              Type destinationElement) {
  if (rhsElement != lhsElement || !isSupportedElementType(allowed, lhsElement)) {
    return false;
  }
  // 三档（同 dtype / 恒 ui8 / i32 出 f32）全部由共用的那一条规则回答，ODS 的结果推导读的
  // 也是它——分开写就会分开错，而错法是静默的。
  return afir::elementwise::matchesResultElementRule(lhsElement, destinationElement, relation);
}

// AscendC binary APIs take one element count for the destination window. Either
// source tile may therefore be smaller only on an outer, non-vectorized axis with
// stride zero: the generated scheduled loop repeats that window while the local
// offset stays zero. The exact safety predicate is shared with unary lowering;
// a size-one dimension on a vectorized axis is not safe to widen there either.
bool shapesAreCompatibleForOuterBroadcast(Operation *op) {
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(op);
  if (failed(partition) || partition->inputCount != 2 || partition->destinationCount != 1) {
    return false;
  }
  OperandRange inputs = partition->inputs(op);
  Value destination = partition->destinations(op)[0];
  return afir::elementwise::sourceBroadcastsToDestination(op, inputs[0], destination) &&
         afir::elementwise::sourceBroadcastsToDestination(op, inputs[1], destination);
}

class BinaryElementwiseCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit BinaryElementwiseCapabilityProvider(std::shared_ptr<const BinaryElementwiseMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_ || !elementwise::hasCanonicalDescriptors(operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // 秩的门禁下面要按计划种类分流，而种类只有计划知道，所以这里顺带把它取出来。
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
    const bool rhsIsTile = mapping_->scalarOperand == ScalarOperand::None;
    FailureOr<BinaryTensorTypes> types = getTensorTypes(operation, mapping_->scalarOperand);
    if (failed(types)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    // 秩的门禁只对 identity 计划成立：那条路径把 `LocalTensorGetSizeOp` 当长度，一维
    // 之外没有第二个下标可解释。scheduled 计划的 LocalTensor 秩跟着 UB buffer 的 memref
    // 走（多维 tile 上就是多维类型），长度由有效视图折算出的局部跨度给，不看这个秩。
    if (planKind == TransferPlanKind::Identity &&
        !(types->lhs.hasRank() && types->lhs.getRank() == 1 && types->destination.hasRank() &&
          types->destination.getRank() == 1 && (!rhsIsTile || (types->rhs.hasRank() && types->rhs.getRank() == 1)))) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    if (!areElementTypesSupported(mapping_->resultElementType, mapping_->elementTypes, types->lhs.getElementType(),
                                  types->rhsElement, types->destination.getElementType())) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    // 标量右操作数没有形状可比——它广播到哪个形状都是它自己。
    const bool sameShape = types->destination.getShape() == types->lhs.getShape() &&
                           (!rhsIsTile || types->rhs.getShape() == types->lhs.getShape());
    if (!sameShape && !(rhsIsTile && shapesAreCompatibleForOuterBroadcast(operation))) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *operation, const ascend::instruction::InplaceQuery &query,
                                 const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_ || query.outputResultIndex != mapping_->resultIndex) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
    if (failed(partition) || partition->inputCount != 2 || partition->destinationCount != 1) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // 只有输入可以与结果重叠。目的地本来就是结果，拿它来问是问错了问题。
    if (query.inputOperandIndex >= partition->inputCount) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    // 标量操作数不是一块存储，谈不上「与目的地共用」。不挡的话这里会给出
    // `AliasConstraint{1, 0}`——声称一个编译期常量可以被写进去。
    if (!isa<ascendc::LocalTensorType>(partition->inputs(operation)[query.inputOperandIndex].getType())) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    if (query.accessMode != ascend::instruction::AliasAccessMode::ReadWriteInput) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
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
    // tensor-tensor 形态没有标量操作数映射；标量变体是另一组 implementation。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    // This lowering requires all tensor operands and the result to have the same shape.  AFIR's
    // broadcast semantics do not imply that this particular AscendC implementation supports inline broadcast.
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const BinaryElementwiseMapping> mapping_;
};

// 施加过调度偏移之后的两个输入 tile、目的地 tile 与元素个数。
struct BinaryOperandTiles {
  Value lhs;
  Value rhs;
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

// `order` 是 `emissionOrder` 算出来的那一份，由调用方传进来——pattern 已经为了校验形态
// 算过一次，这里再算一遍就是第二份编码。
template <typename SourceOp, typename OpAdaptor>
FailureOr<BinaryOperandTiles> materializeOperandTiles(SourceOp op, OpAdaptor adaptor, BinaryOperandOrder order,
                                                      ConversionPatternRewriter &rewriter, bool compactTail) {
  const Value operands[] = {adaptor.getLhs(), adaptor.getRhs()};
  BinaryOperandTiles tiles;
  tiles.lhs = operands[order.tile];
  tiles.rhs = operands[order.other];
  tiles.destination = adaptor.getDestination();
  // Scalar operands may precede the tile. Use the same operand order as API
  // emission instead of implicitly selecting descriptor zero.
  FailureOr<TensorDescAttr> descriptor = elementwise::getOperandDescriptor(op, order.tile);
  if (failed(descriptor)) {
    return failure();
  }
  FailureOr<TransferPlan> plan = buildTransferPlanForDescriptor(op, *descriptor);
  if (failed(plan)) {
    return failure();
  }
  if (plan->kind != TransferPlanKind::Scheduled) {
    tiles.count = rewriter.create<ascendc::LocalTensorGetSizeOp>(op.getLoc(), rewriter.getI32Type(), tiles.destination);
    return tiles;
  }

  // 向量化轴的分段（阶段 2 合轴）。最内一段交给 API，其余段变循环；今天所有图都只有
  // 一段，走的是「不建循环、偏移恒为常量 0」那条分支，发出来的 Op 与此前逐字相同。
  //
  // 本地侧偏移恒为 0 的理由（两类成员都是：队列按 slot 自己轮转；TBuf 按 ATT 的
  // `b{id}_size` 语义只装一个 tile，块内每个迭代整块重写）在单段时仍然成立——单段
  // 没有外层循环，偏移里没有归纳变量。调度偏移只落在 GM 侧。给本地侧再加「归纳变量 ×
  // tileSize」是按多 tile 分配算一块单 tile 的缓冲，块内循环多于一趟时越界写 UB
  // （NPU 上 0x507035）。
  //
  // **标量不必在这里判**：它没有向量化轴，`collapseVectorizedAxes` 不会把它收进来，
  // 于是它拿不到 subindex——与此前按 `isa<LocalTensorType>` 分流的结论相同。
  FailureOr<afir::VectorPlan> segments = afir::collapseVectorizedAxes(op, compactTail);
  if (failed(segments)) {
    return failure();
  }
  // `order` 重排的是**发射次序**，而 `plan.tensors` 的下标说的是 `afir.operand_descs`
  // 里的位置，两者不是一回事。所以这里按 adaptor 的原始次序给值，重排后的 `tiles.lhs` /
  // `tiles.rhs` 再各自从结果里认回去。
  FailureOr<SmallVector<Value>> ordered =
      orderTilesForPlan(*segments, {adaptor.getLhs(), adaptor.getRhs()}, tiles.destination);
  if (failed(ordered)) {
    return failure();
  }
  FailureOr<VectorizedWindow> window =
      materializeVectorizedSegments(op, rewriter, *segments, *plan, *ordered, /*apiWindowSize=*/1);
  if (failed(window) || window->tiles.size() != segments->tensors.size() || window->extents.size() != 1) {
    return failure();
  }
  for (auto [tensor, tile] : llvm::zip(segments->tensors, window->tiles)) {
    if (tensor.side == afir::VectorTensor::Side::Destination) {
      tiles.destination = tile;
      continue;
    }
    // 原始下标对回重排后的位置。
    if (tensor.index == order.tile) {
      tiles.lhs = tile;
    } else if (tensor.index == order.other) {
      tiles.rhs = tile;
    } else {
      return failure();
    }
  }
  tiles.count = window->extents.front();
  return tiles;
}

template <typename SourceOp, typename TargetOp, ScalarOperand Scalar, TmpBufferUse Tmp>
class BinaryElementwiseConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  BinaryElementwiseConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                     std::shared_ptr<const BinaryElementwiseMapping> mapping, bool compactTail)
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
      return rewriter.notifyMatchFailure(op, "buffer-form binary elementwise op requires a destination");
    }
    // 两个输入的实际形态必须与本 pattern 注册的形态一致，并且发射顺序由同一个函数算——
    // capability 那边走的是 `getTensorTypes`，它查的是同一个 `emissionOrder`。走到这里
    // 对不上说明注册写反了：`Scalar` 与建 op 的实参类型不像 `Tmp` 那样由编译器兜住，
    // 两个输入都是一个 `Value`。
    const std::optional<BinaryOperandOrder> order =
        emissionOrder(Scalar, isa<ascendc::LocalTensorType>(adaptor.getLhs().getType()),
                      isa<ascendc::LocalTensorType>(adaptor.getRhs().getType()));
    if (!order) {
      return rewriter.notifyMatchFailure(op, "the selected implementation expects a different operand form");
    }
    if (failed(checkConversionLocalViews(op))) {
      return rewriter.notifyMatchFailure(op, "incompatible elementwise local views");
    }

    FailureOr<BinaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, *order, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    // 这一条对 `Passed` 和 `Unused` 都要查：`Unused` 那档调用虽然不收这块，但调度分了
    // 它，缺了就说明拿到的不是这个 implementation 算过的那张图。
    if constexpr (schedulerGivesTmpBuffer(Tmp)) {
      if (!adaptor.getTmpBuffer()) {
        return rewriter.notifyMatchFailure(op, "the selected implementation expects a tmp buffer operand");
      }
    }
    Operation *targetOp = nullptr;
    if constexpr (callTakesTmpBuffer(Tmp)) {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs,
                                           adaptor.getTmpBuffer(), tiles->count);
    } else {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs, tiles->count);
    }
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BinaryElementwiseMapping> mapping_;
  bool compactTail_;
};

// 标量形态里**方向要发出去**的那一族：`sub` / `div` / `truediv`。
//
// 与上面那个 pattern 只差一件事：建 op 时多带一个 `isScalarLatter`。它不是新算的——
// `emissionOrder` 已经把「有形状的那个是哪一个操作数」算出来了，`order->tile == 0` 就是
// 「标量在右」。**所以方向只有一处编码**，与 capability 查的是同一份。
//
// 为什么这一族要发方向、可交换的那 8 个不用：legacy 的 `binary_api_call.cpp` /
// `truediv_api_call.cpp` 把方向编码成 `IS_SCALAR_LATTER` 这个 bool 模板实参
// （`Subs<T, false>` = 标量在左），因为 `a - s` 与 `s - a` 不是同一个结果；`Adds` 那一族
// 交换完发的是逐字相同的调用。详见 `Elementwise.td` 的 `scalarOperand` 那段。
template <typename SourceOp, typename TargetOp, TmpBufferUse Tmp>
class BinaryScalarDirectionConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  BinaryScalarDirectionConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                         std::shared_ptr<const BinaryElementwiseMapping> mapping, bool compactTail)
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
      return rewriter.notifyMatchFailure(op, "buffer-form binary elementwise op requires a destination");
    }
    const std::optional<BinaryOperandOrder> order =
        emissionOrder(ScalarOperand::Any, isa<ascendc::LocalTensorType>(adaptor.getLhs().getType()),
                      isa<ascendc::LocalTensorType>(adaptor.getRhs().getType()));
    if (!order) {
      return rewriter.notifyMatchFailure(op, "the selected implementation expects one tile and one scalar");
    }

    FailureOr<BinaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, *order, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }
    if constexpr (schedulerGivesTmpBuffer(Tmp)) {
      if (!adaptor.getTmpBuffer()) {
        return rewriter.notifyMatchFailure(op, "the selected implementation expects a tmp buffer operand");
      }
    }

    // `tile` 是 AFIR 的哪个操作数，直接就是方向：0 = 标量在右 = `IS_SCALAR_LATTER` 为真。
    const bool isScalarLatter = order->tile == 0U;
    Operation *targetOp = nullptr;
    if constexpr (callTakesTmpBuffer(Tmp)) {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs,
                                           adaptor.getTmpBuffer(), tiles->count, isScalarLatter);
    } else {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs, tiles->count,
                                           isScalarLatter);
    }
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BinaryElementwiseMapping> mapping_;
  bool compactTail_;
};

// 标量形态里**按 AFIR 原次序发射**的那一族：今天只有 `afir.pow`。
//
// 与上面两个 pattern 的分别就在这一件事上。`Adds` 那一族与 `Subs` 那一族发出去的调用都
// 把有形状的操作数放在第一个输入位——前者因为可交换，后者因为方向由模板实参说。而 `Pow`
// 的供给是**同名重载**：`pow.h` / `pow_reg_base.h` 各有四个 `Pow`，(t,t) / (t,s) / (s,t) /
// (s,s)，`Power` 在哪个位置上收标量由 C++ 重载决议挑。legacy 的 `PowApiCall` 因此逐个位置
// 原样打印，一次都不交换——`x ^ s` 与 `s ^ x` 靠实参位置区分，没有模板实参可以承载方向。
//
// 所以这里 `emissionOrder` 只用来**校验形态并找出哪个是 tile**（count 与块内偏移要它），
// 发射用的是 AFIR 的原次序 `{0, 1}`。两件事分开：一件是「这张图合不合这个 implementation」，
// 一件是「实参怎么排」，在 `Adds` / `Subs` 两族里恰好是同一个答案，在这里不是。
template <typename SourceOp, typename TargetOp, TmpBufferUse Tmp>
class BinaryScalarInPlaceConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  BinaryScalarInPlaceConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                       std::shared_ptr<const BinaryElementwiseMapping> mapping, bool compactTail)
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
      return rewriter.notifyMatchFailure(op, "buffer-form binary elementwise op requires a destination");
    }
    if (!emissionOrder(ScalarOperand::Any, isa<ascendc::LocalTensorType>(adaptor.getLhs().getType()),
                       isa<ascendc::LocalTensorType>(adaptor.getRhs().getType()))) {
      return rewriter.notifyMatchFailure(op, "the selected implementation expects one tile and one scalar");
    }

    // **发射次序是 AFIR 的原次序**，不是 `emissionOrder` 算出来的那一份。
    FailureOr<BinaryOperandTiles> tiles =
        materializeOperandTiles(op, adaptor, BinaryOperandOrder{0U, 1U}, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }
    if constexpr (schedulerGivesTmpBuffer(Tmp)) {
      if (!adaptor.getTmpBuffer()) {
        return rewriter.notifyMatchFailure(op, "the selected implementation expects a tmp buffer operand");
      }
    }
    Operation *targetOp = nullptr;
    if constexpr (callTakesTmpBuffer(Tmp)) {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs,
                                           adaptor.getTmpBuffer(), tiles->count);
    } else {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs, tiles->count);
    }
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BinaryElementwiseMapping> mapping_;
  bool compactTail_;
};

// 二元 + **一个由 ODS 属性来的实参** + 一块 tmp buffer。今天只有 `afir.axpy` 是这个
// 形状：发出去的调用是 `AxpyExtend(dst, x1, x2, alpha, tmp, count)`，而 alpha 在 AFIR
// 上是属性不是操作数——它不参与三段划分，也不占一块缓冲。
//
// **常量恒建 f32，不跟 tile 的元素类型走。** 包装的形参是 `const float alpha`（不是
// `const T&`），legacy 打的也是 `(float)0.800000`。这与一元那条路的
// `UnaryScalarAttrConversionPattern` **恰好相反**：`LeakyRelu` 的形参是 `const T&`，
// 给 f16 的 tile 配 f32 常量会让 T 同时推成 half 和 float。照抄那一段就会在这里错，
// 而且错得没有任何一层会报——`AxpyExtend` 的 `T` 由两个 tile 定死，alpha 是独立的
// 隐式转换，f16 常量照样编得过、只是多一次舍入。
//
// tmp buffer 这里恒收：`CalcAxpyTmpSize` 两个 dtype 分支都分一块，签名里也都有那个
// 形参。所以没有 `TmpBufferUse` 那根轴——这个形状只有一种取值。
template <typename SourceOp, typename TargetOp>
class BinaryScalarAttrConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  BinaryScalarAttrConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                    std::shared_ptr<const BinaryElementwiseMapping> mapping, bool compactTail)
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
      return rewriter.notifyMatchFailure(op, "buffer-form binary elementwise op requires a destination");
    }
    if (!isa<ascendc::LocalTensorType>(adaptor.getRhs().getType())) {
      return rewriter.notifyMatchFailure(op, "the selected implementation expects a tile right operand");
    }
    if (!adaptor.getTmpBuffer()) {
      return rewriter.notifyMatchFailure(op, "the selected API requires a tmp buffer operand");
    }

    // Axpy 两个输入都是 tile，没有标量形态（`scalarOperand` 是 `"none"`）——发射顺序
    // 恒是 `{0, 1}`。
    FailureOr<BinaryOperandTiles> tiles =
        materializeOperandTiles(op, adaptor, BinaryOperandOrder{0U, 1U}, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    Value alpha = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getF32Type(), op.getAlphaAttr());
    Operation *targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs, alpha,
                                                    adaptor.getTmpBuffer(), tiles->count);
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BinaryElementwiseMapping> mapping_;
  bool compactTail_;
};

// 比较族。与 `BinaryElementwiseConversionPattern` 的差别全在建 op 那一行，两处：
//
//   - 多一个 `CMPMODE` 实参，它由**是哪个 AFIR op** 定死（`afir.eq` → `CMPMODE::EQ`），
//     不从 IR 上读——AFIR 这一层没有这个属性，六个 op 就是六个 op。所以它是模板实参：
//     注册处写错了当场编不过，而写成 mapping 字段的话只会发出一条比错的调用。
//   - **mem 侧的 tmp buffer 排在 count 之后**（`CompareExtend(dst, s0, s1, mode, size,
//     tmp)`），与这条路上其余包装相反。次序落在 ODS 的 `arguments` 上，这里照它写。
//
// `Rhs` 与 `Tmp` 两根轴照旧：标量形态建的是 `CompareScalarExtend`（另一个 TargetOp，
// 实参位置相同），reg 侧一块 tmp 都不收但调度照分——`TmpBufferUse::Unused`。
template <typename SourceOp, typename TargetOp, ascendc::CMPMODE Mode, ScalarOperand Scalar, TmpBufferUse Tmp>
class CompareConversionPattern final : public OpConversionPattern<SourceOp> {
 public:
  using OpAdaptor = typename SourceOp::Adaptor;

  CompareConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                           std::shared_ptr<const BinaryElementwiseMapping> mapping, bool compactTail)
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
      return rewriter.notifyMatchFailure(op, "buffer-form binary elementwise op requires a destination");
    }
    // 与通用 pattern 同一条判据、同一个 `emissionOrder`。`eq` / `ne` 走 `Any`，标量在
    // 哪一边都发同一个 `CompareExtend`——`EQ` / `NE` 可交换；`lt` / `le` / `gt` / `ge`
    // 注册的是 `Rhs`，标量在左这里就不成立。
    const std::optional<BinaryOperandOrder> order =
        emissionOrder(Scalar, isa<ascendc::LocalTensorType>(adaptor.getLhs().getType()),
                      isa<ascendc::LocalTensorType>(adaptor.getRhs().getType()));
    if (!order) {
      return rewriter.notifyMatchFailure(op, "the selected implementation expects a different operand form");
    }
    if constexpr (schedulerGivesTmpBuffer(Tmp)) {
      if (!adaptor.getTmpBuffer()) {
        return rewriter.notifyMatchFailure(op, "the selected implementation expects a tmp buffer operand");
      }
    }

    FailureOr<BinaryOperandTiles> tiles = materializeOperandTiles(op, adaptor, *order, rewriter, compactTail_);
    if (failed(tiles)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR elementwise plan");
    }

    Operation *targetOp = nullptr;
    if constexpr (callTakesTmpBuffer(Tmp)) {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs, Mode, tiles->count,
                                           adaptor.getTmpBuffer());
    } else {
      targetOp = rewriter.create<TargetOp>(op.getLoc(), tiles->destination, tiles->lhs, tiles->rhs, Mode, tiles->count);
    }
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, tiles->destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BinaryElementwiseMapping> mapping_;
  bool compactTail_;
};

template <typename SourceOp, typename TargetOp, ScalarOperand Scalar, TmpBufferUse Tmp>
ImplementationRegistry::PatternRegistration<BinaryElementwiseMapping> createPatternRegistration() {
  return [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
            std::shared_ptr<const BinaryElementwiseMapping> mapping) {
    patterns.add<BinaryElementwiseConversionPattern<SourceOp, TargetOp, Scalar, Tmp>>(
        context.typeConverter, patterns.getContext(), key, std::move(mapping),
        context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) == ascend::FeatureState::Available);
  };
}

// `Tmp` 一个实参同时决定 capability 里的 tmp buffer 判据和建 op 时的实参个数，所以两者
// 不可能各说各的——这正是它是模板实参而不是 mapping 字段的原因。**但两者不再是同一个
// 答案**：`Unused` 那档 capability 要收、建 op 不传。三个取值各自把这一对说死，能写出来
// 的组合就只有这三种。
//
// `Scalar` 没有这个性质：tile 和标量都是一个 `Value`，把 `Adds` 注册成
// `ScalarOperand::None` 照样编得过。它由 `matchAndRewrite` 开头那条 match failure 兜住，
// 不是由编译器。
template <typename SourceOp, typename TargetOp, ScalarOperand Scalar = ScalarOperand::None,
          TmpBufferUse Tmp = TmpBufferUse::None>
LogicalResult registerBinary(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family,
                             uint32_t elementTypes = kArithmeticElementTypes,
                             ResultElementRule resultElementType = ResultElementRule::Same) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(family))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, family, elementTypes);
  return registry.registerImplementation<SourceOp, BinaryElementwiseMapping, BinaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BinaryElementwiseMapping{family, Scalar, resultElementType, elementTypes, schedulerGivesTmpBuffer(Tmp)},
      createPatternRegistration<SourceOp, TargetOp, Scalar, Tmp>());
}

// 比较族的注册。capability 那一侧与 `logical_and` 逐字同形（结果恒 `ui8`、两个输入
// 同 dtype），所以复用 `BinaryElementwiseCapabilityProvider`；只有 pattern 换成上面
// 那个。`Mode` 与 `TargetOp` 都是模板实参——两者必须同时改，写岔了编不过。
template <typename SourceOp, typename TargetOp, ascendc::CMPMODE Mode, ScalarOperand Scalar, TmpBufferUse Tmp>
LogicalResult registerCompare(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family,
                              uint32_t elementTypes) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(family))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, family, elementTypes);
  return registry.registerImplementation<SourceOp, BinaryElementwiseMapping, BinaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BinaryElementwiseMapping{family, Scalar, ResultElementRule::Uint8, elementTypes, schedulerGivesTmpBuffer(Tmp)},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BinaryElementwiseMapping> mapping) {
        patterns.add<CompareConversionPattern<SourceOp, TargetOp, Mode, Scalar, Tmp>>(
            context.typeConverter, patterns.getContext(), key, std::move(mapping),
            context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) ==
                ascend::FeatureState::Available);
      });
}

// 标量右操作数的一族。`Adds` / `Muls` / `Maxs` / `Mins` 两张供给表上是同一个调用、
// 都不要 tmp buffer，所以家族恒为 `Any`。
template <typename SourceOp, typename TargetOp>
LogicalResult registerScalarBinary(ImplementationRegistry &registry, llvm::StringRef id) {
  // `Any`：这四个 op 的标量格两侧都是 `2nd+swap(2nd)`，而且都可交换——标量在左发的是
  // 同一个 `Adds` / `Muls` / `Maxs` / `Mins`。
  return registerBinary<SourceOp, TargetOp, ScalarOperand::Any>(registry, id, ApiFamily::Any);
}

// 方向要发出去的那一族的注册（`sub` / `div` / `truediv`）。与 `registerScalarBinary`
// 分开是因为**这一族每个 op 每一侧都是一个独立的 implementation**：mem 侧发 `Subs` 并收
// tmp，reg 侧发 `DivExtends` 不收，两侧的调用名与实参个数都不同，`ApiFamily::Any` 说不出。
template <typename SourceOp, typename TargetOp, TmpBufferUse Tmp>
LogicalResult registerScalarBinaryDirection(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family,
                                            uint32_t elementTypes,
                                            ResultElementRule resultElementType = ResultElementRule::Same) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(family))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, family, elementTypes);
  return registry.registerImplementation<SourceOp, BinaryElementwiseMapping, BinaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BinaryElementwiseMapping{family, ScalarOperand::Any, resultElementType, elementTypes,
                               schedulerGivesTmpBuffer(Tmp)},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BinaryElementwiseMapping> mapping) {
        patterns.add<BinaryScalarDirectionConversionPattern<SourceOp, TargetOp, Tmp>>(
            context.typeConverter, patterns.getContext(), key, std::move(mapping),
            context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) ==
                ascend::FeatureState::Available);
      });
}

// 按 AFIR 原次序发射的标量形态的注册（今天只有 `pow`）。capability 与 `Any` 那一族
// 逐字相同，只有 pattern 换掉。
template <typename SourceOp, typename TargetOp, TmpBufferUse Tmp>
LogicalResult registerScalarBinaryInPlace(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family,
                                          uint32_t elementTypes) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(family))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, family, elementTypes);
  return registry.registerImplementation<SourceOp, BinaryElementwiseMapping, BinaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BinaryElementwiseMapping{family, ScalarOperand::Any, ResultElementRule::Same, elementTypes,
                               schedulerGivesTmpBuffer(Tmp)},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BinaryElementwiseMapping> mapping) {
        patterns.add<BinaryScalarInPlaceConversionPattern<SourceOp, TargetOp, Tmp>>(
            context.typeConverter, patterns.getContext(), key, std::move(mapping),
            context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) ==
                ascend::FeatureState::Available);
      });
}

// `afir.axpy`：两张供给表共用同一个扩展包装（tier D1），所以家族恒为 `Shared`、
// 只有一个 implementation。tmp buffer 恒收，见 pattern 那段。
template <typename SourceOp, typename TargetOp>
LogicalResult registerBinaryScalarAttr(ImplementationRegistry &registry, llvm::StringRef id, uint32_t elementTypes) {
  if (failed(elementwise::checkFamilyMatchesHeader<TargetOp>(ApiFamily::Shared))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(SourceOp::getOperationName(), id, ApiFamily::Shared, elementTypes);
  return registry.registerImplementation<SourceOp, BinaryElementwiseMapping, BinaryElementwiseCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BinaryElementwiseMapping{ApiFamily::Shared, ScalarOperand::None, ResultElementRule::Same, elementTypes,
                               schedulerGivesTmpBuffer(TmpBufferUse::Passed)},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BinaryElementwiseMapping> mapping) {
        patterns.add<BinaryScalarAttrConversionPattern<SourceOp, TargetOp>>(
            context.typeConverter, patterns.getContext(), key, std::move(mapping),
            context.target.getFeatureState(ascend::TargetFeature::RegBasedApiFamily) ==
                ascend::FeatureState::Available);
      });
}

}  // namespace

LogicalResult registerBinaryElementwiseImplementations(ImplementationRegistry &registry,
                                                       ascend::instruction::SelectionPolicyRegistry *) {
  // 这四个 op 的 tensor-tensor 形态在两套 API 家族上是同一个 `AscendC::` 调用，都不要
  // tmp buffer，所以各只有一个 implementation，也不查任何 feature。标量变体才有平台差异。
  if (failed(registerBinary<AddOp, ascendc::AddL2Op>(registry, "add.tensor_tensor", ApiFamily::Any)) ||
      failed(registerBinary<MulOp, ascendc::MulL2Op>(registry, "mul.tensor_tensor", ApiFamily::Any)) ||
      failed(registerBinary<MaximumOp, ascendc::MaxL2Op>(registry, "maximum.tensor_tensor", ApiFamily::Any)) ||
      failed(registerBinary<MinimumOp, ascendc::MinL2Op>(registry, "minimum.tensor_tensor", ApiFamily::Any))) {
    return failure();
  }

  // tensor-scalar 形态：右操作数是一个标量 SSA 值，建的是 `Adds` / `Muls` / `Maxs` /
  // `Mins`。同一个 AFIR op 的第二个 implementation——选哪个由操作数形态定死，两者的
  // 判据互斥（见 `getTensorTypes`），不靠优先级。
  //
  // 要走包装的是 Sub / Div（`scalar_sub.h` / `scalar_div.h`，且只有 mem-based 一侧
  // 有），它们的标量形态还没接通；tensor-tensor 形态见下面那一组。
  if (failed(registerScalarBinary<AddOp, ascendc::AddsL2Op>(registry, "add.tensor_scalar")) ||
      failed(registerScalarBinary<MulOp, ascendc::MulsL2Op>(registry, "mul.tensor_scalar")) ||
      failed(registerScalarBinary<MaximumOp, ascendc::MaxsL2Op>(registry, "maximum.tensor_scalar")) ||
      failed(registerScalarBinary<MinimumOp, ascendc::MinsL2Op>(registry, "minimum.tensor_scalar"))) {
    return failure();
  }

  // `bitwise_and`：二元一族里第一个两侧调用**不同名**的 op，形状与 `reciprocal` 那条
  // 一样——reg 侧是标准调用、不要 tmp，mem 侧走包装、要 tmp。于是两个 implementation，
  // 各自钉死一张供给表，由 `checkApiFamily` 按 target 分流。
  //
  // 包装其实一次都没用那块 tmp buffer（`bitwise_and.h` 里它只是个形参），但 legacy 照旧
  // 按 `CalcDefaultTmpSize` 给它留了一块。这里跟着留：少给的话调用的实参个数就对不上。
  //
  // 没有标量形态——legacy 两侧的 `IsNodeValid` 都明写拒绝标量输入。
  if (failed(registerBinary<BitwiseAndOp, ascendc::BitwiseAndL2Op>(registry, "bitwise_and.tensor", ApiFamily::RegBased,
                                                                   kUI8 | kI16 | kUI16 | kI32)) ||
      failed(registerBinary<BitwiseAndOp, ascendc::BitwiseAndExtendOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "bitwise_and.tensor.extend", ApiFamily::MemBased, kUI8 | kI16 | kUI16 | kI32))) {
    return failure();
  }

  // step 4。`bitwise_or` / `bitwise_xor` 与 `bitwise_and` 同一张元素类型集合，但只有
  // reg 一侧（mem 侧 legacy 一个都不注册），而且 reg 侧也走包装——legacy 的
  // `BitwiseOrAscIrCodegenImplV2` / `BitwiseXorAscIrCodegenImplV2` 发的是
  // `BitwiseOrExtend` / `BitwiseXorExtend`（`api_regbase/bitwise_or.h` /
  // `bitwise_xor.h`），不是标准调用 `AscendC::BitwiseOr` / `BitwiseXor`——两个供给表对这
  // 四位的转发是等价的，但发出去的调用字面必须与 legacy 一致。不带 tmp，没有标量形态
  // （`IsNodeValid` 拒标量）。
  if (failed(registerBinary<BitwiseOrOp, ascendc::BitwiseOrExtendRegBaseOp>(
          registry, "bitwise_or.tensor", ApiFamily::RegBased, kUI8 | kI16 | kUI16 | kI32)) ||
      failed(registerBinary<BitwiseXorOp, ascendc::BitwiseXorExtendRegBaseOp>(
          registry, "bitwise_xor.tensor", ApiFamily::RegBased, kUI8 | kI16 | kUI16 | kI32))) {
    return failure();
  }

  // `floor_div`：二元里的 `neg`——两侧的包装**同名**（`FloorDivExtend`），只有扩展头
  // 不同。差别落在另外两件事上，两件都在这四个实参里说清了：
  //
  //   - mem 侧要一块 tmp buffer（装 `Div` 的中间商再交给 `Floor`），reg 侧不要；
  //   - mem 侧的 legacy 注册只给了 f16 / f32，reg 侧连整数一起收。
  //
  // 于是 id 上只差 `_reg_base` 后缀，和 `neg.tensor.extend` 那一对同一个命名法。
  if (failed(registerBinary<FloorDivOp, ascendc::FloorDivExtendOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "floor_div.tensor.extend", ApiFamily::MemBased, kF16 | kF32)) ||
      failed(registerBinary<FloorDivOp, ascendc::FloorDivExtendRegBaseOp>(
          registry, "floor_div.tensor.extend_reg_base", ApiFamily::RegBased, kArithmeticRegElementTypes))) {
    return failure();
  }

  // `remainder`：比 `floor_div` 还整齐的一对——两侧同名、**都要 tmp buffer**，只有扩展
  // 头和收的元素类型不同。reg 侧那块 tmp 尺寸是 0，但仍然要有：它是拿来挑重载的，
  // 不是拿来算的（见 `OpVecBinary.td` 里那段）。所以两侧都是 `TmpBufferUse::Passed`。
  //
  // bf16 只有 reg 侧收，这里没有——AFIR 整个不建模 bf16。
  if (failed(registerBinary<RemainderOp, ascendc::RemainderExtendOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "remainder.tensor.extend", ApiFamily::MemBased, kF32 | kI32)) ||
      failed(registerBinary<RemainderOp, ascendc::RemainderExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "remainder.tensor.extend_reg_base", ApiFamily::RegBased, kF16 | kF32 | kI32 | kBF16))) {
    return failure();
  }
  // `sub` / `div`：两侧都是 `BinaryApiCall`，legacy 的两个 impl 类逐行同形（连
  // `IsScalarInputSupported` / `IsInplaceSupported` 都一样），所以一起注册。
  //
  // **mem 侧是这一批里第一个 `TmpBufferUse::Unused`。** 它发的是标准的 `AscendC::Sub` /
  // `AscendC::Div`，四个实参、没有 tmp 的位置；而调度按 `CalcBinaryApiTmpSize` 照样给
  // 这个节点分了一块（tensor 一支 32 字节）。那块 UB 是给标量形态的 `Subs` / `Divs` 的，
  // 这条路用不上，但收下它是必须的——bridge 已经把它挂上了 AFIR op。
  //
  // reg 侧是 fork 自己的包装（`SubExtend` / `DivExtend`，转发给同名的 `AscendC::` 调用），
  // 不要 tmp，所以是 `None`。两侧的 id 沿用 `_reg_base` 后缀那套命名。
  //
  // 元素类型：`sub` 的 mem 侧 `REG_ASC_IR` 就是算术那四个；reg 侧多收 ui8 / ui16 / bf16
  // （都建模得出来，所以都注册上），再多的 i8 / i64 / ui32 / ui64 AFIR 这一层还没有位。
  // `div` 两侧都只有 f16 / f32。
  if (failed(registerBinary<SubOp, ascendc::SubL2Op, ScalarOperand::None, TmpBufferUse::Unused>(
          registry, "sub.tensor_tensor", ApiFamily::MemBased)) ||
      failed(registerBinary<SubOp, ascendc::SubExtendRegBaseOp>(
          registry, "sub.tensor_tensor.extend_reg_base", ApiFamily::RegBased, kArithmeticRegElementTypes | kBF16)) ||
      failed(registerBinary<DivOp, ascendc::DivL2Op, ScalarOperand::None, TmpBufferUse::Unused>(
          registry, "div.tensor_tensor", ApiFamily::MemBased, kF16 | kF32)) ||
      failed(registerBinary<DivOp, ascendc::DivExtendRegBaseOp>(registry, "div.tensor_tensor.extend_reg_base",
                                                                ApiFamily::RegBased, kF16 | kF32))) {
    return failure();
  }

  // `sub` / `div` 的标量形态。**这里 tmp buffer 的取值与 tensor-tensor 那两条相反**：
  // `Subs` / `Divs` 真用那块（先把标量 `Duplicate` 成一个 block 再走 `BinaryRepeatParams`），
  // 而 tensor-tensor 发的是不带 tmp 的四参 `AscendC::Sub` —— 同一个 op 的两种形态，一个
  // `Passed` 一个 `Unused`，调度两种都分。第一次出现「同 op 两档 tmp 取值不同」。
  //
  // **reg 侧只有 `div`**：`sub_reg_base.h` 里 `SubExtends` 也在，但 legacy v2 的
  // `IsScalarInputSupported` 对 Sub 返回 false（矩阵 `reg_scalar` 列），调度产不出那种图。
  // 接了就是一个谁也走不到的 implementation。
  //
  // 元素类型跟着各自 tensor-tensor 那一条：`sub` 的 mem 侧是算术四个，`div` 两侧 f16/f32。
  if (failed(registerScalarBinaryDirection<SubOp, ascendc::SubsOp, TmpBufferUse::Passed>(
          registry, "sub.tensor_scalar", ApiFamily::MemBased, kArithmeticElementTypes)) ||
      failed(registerScalarBinaryDirection<DivOp, ascendc::DivsOp, TmpBufferUse::Passed>(
          registry, "div.tensor_scalar", ApiFamily::MemBased, kF16 | kF32)) ||
      failed(registerScalarBinaryDirection<DivOp, ascendc::DivExtendsRegBaseOp, TmpBufferUse::None>(
          registry, "div.tensor_scalar.extend_reg_base", ApiFamily::RegBased, kF16 | kF32))) {
    return failure();
  }

  // `logical_and` / `logical_or`：这条路上**第一对结果 dtype 与输入无关的 op**，也是
  // 第一对同时注册了 tensor 形态与标量形态的 op。`ResultElementRule::Uint8` 那个
  // 实参就是全部的新东西——目的地恒为 `ui8`，两个输入之间照旧要同 dtype。
  //
  // 两侧差三样——调用名、扩展头、要不要 tmp——**这是第一对三样一起错开的**。
  // `bitwise_and` 的 reg 侧发的是不带扩展头的标准调用，`floor_div` / `remainder` 两侧
  // 同名，都只差其中两样：
  //
  //   - mem 侧 `LogicalAnd(dst, s0, s1, tmp, size)`（`logical.h`），**tmp 是真用的**，
  //     所以是 `Passed`；reg 侧 `LogicalAndExtend(dst, s0, s1, size)`
  //     （`logical_reg_base.h`），`None`。
  //   - mem 侧的 apiName 是裸名 `LogicalAnd` 而不是 `LogicalAndExtend`——它照样走扩展头，
  //     所以 id 上仍然用 `_reg_base` 区分两侧，不按 apiName 取名。
  //
  // 元素类型：mem 侧 `REG_ASC_IR` 的 `T1` 就是这五个。reg 侧更宽（bf16 / i8 / i64 /
  // ui32 / ui16 / ui64 / bool），AFIR 一个都建模不出来，于是两侧写成同一个集合——
  // **这是 AFIR 的建模上限，不是 legacy 两侧真的一样。**
  constexpr uint32_t kLogicalElementTypes = kUI8 | kI16 | kI32 | kF16 | kF32;
  // reg 侧多一个 ui16。原先两侧共用上面那个集合，注释写的是「AFIR 建模上限，不是 legacy
  // 两侧真的一样」——ui16 恰恰是建模得出来的那一个，于是它被一起漏掉了。
  constexpr uint32_t kLogicalRegElementTypes = kLogicalElementTypes | kUI16 | kBF16;
  if (failed(registerBinary<LogicalAndOp, ascendc::LogicalAndExtendOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "logical_and.tensor_tensor", ApiFamily::MemBased, kLogicalElementTypes,
          ResultElementRule::Uint8)) ||
      failed(registerBinary<LogicalAndOp, ascendc::LogicalAndExtendRegBaseOp>(
          registry, "logical_and.tensor_tensor.extend_reg_base", ApiFamily::RegBased, kLogicalRegElementTypes,
          ResultElementRule::Uint8)) ||
      failed(registerBinary<LogicalOrOp, ascendc::LogicalOrExtendOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "logical_or.tensor_tensor", ApiFamily::MemBased, kLogicalElementTypes, ResultElementRule::Uint8)) ||
      failed(registerBinary<LogicalOrOp, ascendc::LogicalOrExtendRegBaseOp>(
          registry, "logical_or.tensor_tensor.extend_reg_base", ApiFamily::RegBased, kLogicalRegElementTypes,
          ResultElementRule::Uint8)) ||
      // 第 6 批。`logical_xor` 与上面两个逐字同形（四参无 tmp、结果恒 `ui8`），差别是
      // **只有 reg 侧**：legacy 的 mem 侧（`ascir_builtin_ops_v1.cpp`）不注册 `LogicalXor`,
      // 所以没有 `.tensor_tensor` 那一半，也没有标量那一半。
      //
      // dtype 用 `kLogicalElementTypes` 而不是 reg 那个更宽的集合：legacy 的
      // `REG_ASC_IR(LogicalXor)` 域里没有 ui16。
      failed(registerBinary<LogicalXorOp, ascendc::LogicalXorExtendRegBaseOp>(
          registry, "logical_xor.tensor_tensor.extend_reg_base", ApiFamily::RegBased, kLogicalRegElementTypes,
          ResultElementRule::Uint8))) {
    return failure();
  }

  // 第 6 批的三个二元，全是 reg-only。
  //
  // `fmod` / `mod` **指向同一个 AscendC op**：legacy 两个调度节点的 `GetApiName` 都是
  // `FmodExtend`、头都是 `fmod_reg_base.h`，与 `gelu`（两张供给表同一个调用）同一种情况。
  if (failed(registerBinary<FmodOp, ascendc::FmodExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "fmod.tensor.extend_reg_base", ApiFamily::RegBased, kF16 | kF32 | kBF16)) ||
      failed(registerBinary<ModOp, ascendc::FmodExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "mod.tensor.extend_reg_base", ApiFamily::RegBased, kF16 | kF32 | kBF16 | kI16 | kUI8)) ||
      // `next_after` 只收 f32：`REG_ASC_IR(NextAfter)` 的域只有 `DT_FLOAT`。
      failed(registerBinary<NextAfterOp, ascendc::NextAfterExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "next_after.tensor.extend_reg_base", ApiFamily::RegBased, kF32))) {
    return failure();
  }

  // 第 7 批的六个二元，全是 reg-only、两输入同 dtype。
  //
  // `copy_sign` / `trunc_div` 走包装但**不要 tmp**（`XxxExtend(dst, src0, src1, size)`），
  // 与 `sub` / `div` 的 reg 那一半同形；其余四个要一块 tmp，与 `fmod` 同形。
  //
  // legacy 的 `HermitePolynomialH` / `He` / `PolyGamma` 形状看着也是二元带 tmp，**但不在
  // 这一批**：它们的 `n` 是另一个类型参数（`x` 是 `T` = f32、`n` 是 `U` = i32），而
  // `areElementTypesSupported` 开头就要求两个输入元素类型逐字相同。接它们要先在那一层
  // 开出「两侧 dtype 不同」这根轴，不是在这里多写一行。
  if (failed(registerBinary<CopySignOp, ascendc::CopySignExtendRegBaseOp>(registry, "copy_sign.tensor.extend_reg_base",
                                                                          ApiFamily::RegBased, kF16 | kF32 | kBF16)) ||
      failed(registerBinary<TruncDivOp, ascendc::TruncDivExtendRegBaseOp>(registry, "trunc_div.tensor.extend_reg_base",
                                                                          ApiFamily::RegBased, kF16 | kF32 | kBF16)) ||
      failed(registerBinary<Atan2Op, ascendc::Atan2ExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "atan2.tensor.extend_reg_base", ApiFamily::RegBased, kF16 | kF32 | kBF16)) ||
      // Bessel / Igamma 那几个 legacy 的域只有 `DT_FLOAT`。
      failed(registerBinary<ZetaOp, ascendc::ZetaExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "zeta.tensor.extend_reg_base", ApiFamily::RegBased, kF32)) ||
      failed(registerBinary<IgammaOp, ascendc::IgammaExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "igamma.tensor.extend_reg_base", ApiFamily::RegBased, kF32)) ||
      failed(registerBinary<IgammacOp, ascendc::IgammacExtendRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "igammac.tensor.extend_reg_base", ApiFamily::RegBased, kF32))) {
    return failure();
  }

  // 第 8 批（二元 1 个）。`hypot` 与上面的 `zeta` / `igamma` 逐项同形——reg-only、
  // 两输入同 dtype、`TmpBufferUse::Passed`——只有供给不同：**它走标准调用**
  // `AscendC::Hypot`，所以 target op 是 `HypotL2Op`（不声明
  // `ApiExtensionOpInterface`，进 `kIsStockCall`）而不是一个 `XxxExtendRegBaseOp`。
  //
  // fork 里有一份 `api_regbase/hypot.h`（`HypotExtend`），但 legacy 一处都不加载它——
  // `HypotAscIrCodegenImplV2` 不覆写 `LoadApiHeaderFiles`。照文件名挑供给会发出一个
  // legacy 从不发的调用。
  //
  // tmp 那块的尺寸是 0（`CalcVoidTmpSizeV2`），但形参在就照收：少一个实参就调不上
  // 那个重载，与 `remainder` 的 reg 那一半同一条理由。
  if (failed(registerBinary<HypotOp, ascendc::HypotL2Op, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "hypot.tensor", ApiFamily::RegBased, kF16 | kF32 | kBF16))) {
    return failure();
  }

  // 标量形态。**三个而不是四个**：reg 侧的 `logical_or` 一个标量都不收。
  //
  // 这不是漏了。`LogicalOrAscIrCodegenImplV2` 没有覆写 `IsScalarInputSupported`，也没有
  // 覆写 `IsScalarInputSupportedIfExchangeInputs`，基类两条都返回 `false`；另外三个
  // impl 类都覆写成了 `OnlySecondInputSupportScalar`。矩阵上那一格写的就是 `no`。
  // 那边换来的是另一套东西（`MicroApiCall` / `GenerateSimtScalarExpr`），这条路上还没有
  // 对应物。`logical_reg_base.h` 里的 `LogicalOrExtends` 这个函数是有的，但**没有哪条
  // legacy 判据会让它被选中**——供给里存在不等于这条路能发它，所以不注册。
  //
  // 包装名两侧各按各的规矩派生，都不是 apiName 直接加后缀（见 `OpVecBinary.td`）：
  // mem 侧 `LogicalAndScalarExtend(dst, src, scalar, tmp, size)`，tmp 照旧 `Passed`；
  // reg 侧 `LogicalAndExtends(dst, src, scalar, size)`，`None`。
  //
  // 这一族没有走 `registerScalarBinary`：那个包装把家族写死成 `ApiFamily::Any`
  // （`Adds` / `Muls` 那几个两侧同一个调用），而这里两侧的调用不同名、tmp 需求也不同，
  // 正是它假定不成立的情形。
  if (failed(registerBinary<LogicalAndOp, ascendc::LogicalAndScalarExtendOp, ScalarOperand::Any, TmpBufferUse::Passed>(
          registry, "logical_and.tensor_scalar", ApiFamily::MemBased, kLogicalElementTypes,
          ResultElementRule::Uint8)) ||
      failed(registerBinary<LogicalAndOp, ascendc::LogicalAndExtendsRegBaseOp, ScalarOperand::Any>(
          registry, "logical_and.tensor_scalar.extend_reg_base", ApiFamily::RegBased, kLogicalRegElementTypes,
          ResultElementRule::Uint8)) ||
      failed(registerBinary<LogicalOrOp, ascendc::LogicalOrScalarExtendOp, ScalarOperand::Any, TmpBufferUse::Passed>(
          registry, "logical_or.tensor_scalar", ApiFamily::MemBased, kLogicalElementTypes, ResultElementRule::Uint8))) {
    return failure();
  }

  // `truediv`：**第一个结果 dtype 逐行不同的 op**，也是第一个 mem 侧要说出**两个**扩展头
  // 的二元 op（`scalar_div.h` + `true_div.h`，`sign` 在一元那边先兑现过前置 A）。
  //
  // 两侧差的不只是头，是**收哪几种输入**：
  //
  //   - mem 侧 `TrueDivExtend`（`true_div.h`）收 {f16, f32, i32}，`ResultElementType::
  //     SameOrF32ForI32` 把 i32→f32 那一行说出来。要 tmp buffer 且是 `Passed`——i32 那支
  //     拿它装两份 `Cast` 结果，f32 那支一次没用但形参在签名里，与 `BitwiseAndExtend` /
  //     `AxpyExtend` 同一种情况（`CalcTrueDivTmpSize` 两支都分）。
  //   - **reg 侧发的就是 `afir.div` 的 reg 实现今天已经在发的那个调用**：v2 的
  //     `TrueDivAscIrCodegenImplV2` 写的 apiName 是 `DivExtend`、头是 `div_reg_base.h`，
  //     与 `DivAscIrCodegenImplV2` 逐字相同，所以直接复用 `DivExtendRegBaseOp`，不新建
  //     def。不要 tmp（`BinaryApiCall` 的签名里没有），结果 dtype 是 `Same`。
  //
  // 所以 **i32 只有 mem 侧收**：v2 的 `kTrueDivTypePairs` 是 {f16, f32, bf16}，bf16
  // AFIR 不建模。这一行就是这个 op 存在的全部理由——两侧都写成 `Same` + {f16, f32} 的话
  // `afir.truediv` 就是 `afir.div` 的逐字别名，用例全绿而覆盖面一个没涨。
  //
  // 标量形态**只有 mem 侧**：v2 的 `IsScalarInputSupported` 对 TrueDiv 返回 false（矩阵
  // `reg_scalar = no`），reg 侧接了就是死代码。mem 侧发 `TrueDivExtends<T, U, LATTER>`，
  // 三个模板实参——`U` 即使与 `T` 相同也要写出来，否则方向那一位会落到 `U` 的位置上。
  if (failed(registerBinary<TrueDivOp, ascendc::TrueDivExtendOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "truediv.tensor_tensor.extend", ApiFamily::MemBased, kF16 | kF32 | kI32,
          ResultElementRule::SameOrF32ForI32)) ||
      failed(registerBinary<TrueDivOp, ascendc::DivExtendRegBaseOp>(registry, "truediv.tensor_tensor.extend_reg_base",
                                                                    ApiFamily::RegBased, kF16 | kF32 | kBF16)) ||
      failed(registerScalarBinaryDirection<TrueDivOp, ascendc::TrueDivExtendsOp, TmpBufferUse::Passed>(
          registry, "truediv.tensor_scalar.extend", ApiFamily::MemBased, kF16 | kF32 | kI32,
          ResultElementRule::SameOrF32ForI32))) {
    return failure();
  }

  // `pow`：与 `remainder` 同形的一对——两侧同名、都要 tmp buffer，只有扩展头不同。
  // 新的只有一件事：**两侧收的元素类型差得最远**。mem 侧 legacy 的 `REG_ASC_IR(Pow)`
  // 只有 {i32, f16, f32}；reg 侧那一版是 {i8, ui8, ui16, ui32, i16, i32, f16, f32,
  // bf16}，AFIR 建模得出来的是其中六个。于是 ODS 那一层的并集有六种，而 mem 侧只认
  // 其中一半。
  //
  // 调用名是裸名 `Pow`（不是 `PowExtend`），与 `logical_and` 的 mem 侧同一种情况：
  // 照样走扩展头，所以 id 上仍然用 `_reg_base` 区分两侧，不按 apiName 取名。
  //
  // 没有标量形态：legacy 两侧都收（mem 侧只拒全标量，reg 侧那一格是 `any`），
  // 但要 L-4 的 swap 方向，这一轮只接 tensor-tensor。
  if (failed(registerBinary<PowOp, ascendc::PowOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "pow.tensor_tensor", ApiFamily::MemBased, kF16 | kF32 | kI32)) ||
      failed(registerBinary<PowOp, ascendc::PowRegBaseOp, ScalarOperand::None, TmpBufferUse::Passed>(
          registry, "pow.tensor_tensor.extend_reg_base", ApiFamily::RegBased,
          kArithmeticElementTypes | kUI8 | kUI16 | kBF16))) {
    return failure();
  }

  // `pow` 的标量形态。**两侧都有**，而且用的是与 tensor-tensor **同一个 def**：两种形态
  // 发的是同名重载，实参位置也相同，只有那个位置上的 SSA 值从 tile 变成标量。所以这里
  // 没有新的 AscendC op，只有一个按 AFIR 原次序发射的 pattern（见
  // `BinaryScalarInPlaceConversionPattern` 那段为什么不能交换）。
  //
  // 元素类型与各自 tensor-tensor 那一条相同：legacy 的 `IsScalarInputSupported` 只管
  // 「标量在这个位置行不行」，不另收窄 dtype。
  if (failed(registerScalarBinaryInPlace<PowOp, ascendc::PowOp, TmpBufferUse::Passed>(
          registry, "pow.tensor_scalar", ApiFamily::MemBased, kF16 | kF32 | kI32)) ||
      failed(registerScalarBinaryInPlace<PowOp, ascendc::PowRegBaseOp, TmpBufferUse::Passed>(
          registry, "pow.tensor_scalar.extend_reg_base", ApiFamily::RegBased,
          kArithmeticElementTypes | kUI8 | kUI16 | kBF16))) {
    return failure();
  }

  // `axpy`：**这条路上第一个 tier D1** ——两张供给表共用同一个扩展包装
  // `AxpyExtend`（`axpy.h`，`api_regbase/` 下没有对应文件）。于是只有一个
  // implementation、不按 target 分流，id 上也没有 `_reg_base` 那一半。
  //
  // 它与上面所有 op 差的不是供给，是**发射的形状**：二元 + 一个由 ODS 属性来的实参
  // + 一块 tmp buffer。所以它走自己的 pattern（`BinaryScalarAttrConversionPattern`），
  // 不走 `registerBinary` 那条。
  //
  // 元素类型 {f16, f32}：v1 的 `T` 就是这两个。v2 多出 bf16 / u64 / i64，AFIR 这一层
  // 的 `ElementTypes` 拼不出来，所以一个 implementation 覆两侧不丢东西。
  if (failed(registerBinaryScalarAttr<AxpyOp, ascendc::AxpyExtendContiguousOp>(registry, "axpy.tensor_tensor",
                                                                               kF16 | kF32))) {
    return failure();
  }

  // 比较族（eq / ne / lt / le / gt / ge）：**六个 op 二十四个 implementation**，也是这条
  // 路上第一批「一个 AscendC def 服务六个 AFIR op」。差别只有 `CMPMODE`，所以两侧各
  // 一个 tensor-tensor 与一个 tensor-scalar 的 def，模式由模板实参给。
  //
  // 两侧差三样，与 `logical_and` 那一对同一个形状，只是每一样的取值不同：
  //
  //   - **调用名相同**（`CompareExtend` / `CompareScalarExtend`），差的是模板实参表：
  //     mem 侧一个都不打、`CMPMODE` 走实参位；reg 侧打 `<InT, 1, CMPMODE::XX>`。
  //   - **扩展头**：`compare.h` + `compare_v2.h`（要 `utils.h` 前导）对
  //     `compare_reg_base.h`。id 上照旧用 `_reg_base` 后缀分两侧。
  //   - **tmp buffer**：mem 侧 `Passed`（`CalcXxTmpSize` 按 dtype 算一块，包装真用它）;
  //     reg 侧 `Unused`——`GetCompareSizeV2` 照样分 32 字节，而 `CompareExtend` 的签名里
  //     没有那个形参。这是 `sub` / `div` 之外第二处用上第三档的地方。
  //
  // 元素类型两侧差得远：mem 侧 legacy 的 `T1` 是 {f16, f32, i32, i64}（`lt` 那一行连
  // i64 都没有），reg 侧是十二个。AFIR 建模得出来的分别是三个和六个——i64 两侧都有，
  // 但这一层的位掩码拼不出它。**六个 op 在这里是同一对集合**：mem 侧 `lt` 少的那个
  // i64 本来就不在建模范围内，所以不必逐行分。
  //
  // 标量形态两侧六个 op 全都收（`OnlySecondInputSupportScalar`）。**swap 方向只有
  // eq / ne 有，这一轮不接**——那是 L-4。
  constexpr uint32_t kCompareMemElementTypes = kF16 | kF32 | kI32;
  constexpr uint32_t kCompareRegElementTypes = kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16;
  if (failed(registerCompare<EqOp, ascendc::CompareExtendContiguousOp, ascendc::CMPMODE::EQ, ScalarOperand::None,
                             TmpBufferUse::Passed>(registry, "eq.tensor_tensor", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<EqOp, ascendc::CompareExtendContiguousRegBaseOp, ascendc::CMPMODE::EQ, ScalarOperand::None,
                             TmpBufferUse::Unused>(registry, "eq.tensor_tensor.extend_reg_base", ApiFamily::RegBased,
                                                   kCompareRegElementTypes)) ||
      failed(registerCompare<EqOp, ascendc::CompareScalarExtendContiguousOp, ascendc::CMPMODE::EQ, ScalarOperand::Any,
                             TmpBufferUse::Passed>(registry, "eq.tensor_scalar", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<EqOp, ascendc::CompareScalarExtendContiguousRegBaseOp, ascendc::CMPMODE::EQ,
                             ScalarOperand::Any, TmpBufferUse::Unused>(registry, "eq.tensor_scalar.extend_reg_base",
                                                                       ApiFamily::RegBased, kCompareRegElementTypes)) ||
      failed(registerCompare<NeOp, ascendc::CompareExtendContiguousOp, ascendc::CMPMODE::NE, ScalarOperand::None,
                             TmpBufferUse::Passed>(registry, "ne.tensor_tensor", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<NeOp, ascendc::CompareExtendContiguousRegBaseOp, ascendc::CMPMODE::NE, ScalarOperand::None,
                             TmpBufferUse::Unused>(registry, "ne.tensor_tensor.extend_reg_base", ApiFamily::RegBased,
                                                   kCompareRegElementTypes)) ||
      failed(registerCompare<NeOp, ascendc::CompareScalarExtendContiguousOp, ascendc::CMPMODE::NE, ScalarOperand::Any,
                             TmpBufferUse::Passed>(registry, "ne.tensor_scalar", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<NeOp, ascendc::CompareScalarExtendContiguousRegBaseOp, ascendc::CMPMODE::NE,
                             ScalarOperand::Any, TmpBufferUse::Unused>(registry, "ne.tensor_scalar.extend_reg_base",
                                                                       ApiFamily::RegBased, kCompareRegElementTypes)) ||
      failed(registerCompare<LtOp, ascendc::CompareExtendContiguousOp, ascendc::CMPMODE::LT, ScalarOperand::None,
                             TmpBufferUse::Passed>(registry, "lt.tensor_tensor", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<LtOp, ascendc::CompareExtendContiguousRegBaseOp, ascendc::CMPMODE::LT, ScalarOperand::None,
                             TmpBufferUse::Unused>(registry, "lt.tensor_tensor.extend_reg_base", ApiFamily::RegBased,
                                                   kCompareRegElementTypes)) ||
      failed(registerCompare<LtOp, ascendc::CompareScalarExtendContiguousOp, ascendc::CMPMODE::LT, ScalarOperand::Rhs,
                             TmpBufferUse::Passed>(registry, "lt.tensor_scalar", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<LtOp, ascendc::CompareScalarExtendContiguousRegBaseOp, ascendc::CMPMODE::LT,
                             ScalarOperand::Rhs, TmpBufferUse::Unused>(registry, "lt.tensor_scalar.extend_reg_base",
                                                                       ApiFamily::RegBased, kCompareRegElementTypes)) ||
      failed(registerCompare<LeOp, ascendc::CompareExtendContiguousOp, ascendc::CMPMODE::LE, ScalarOperand::None,
                             TmpBufferUse::Passed>(registry, "le.tensor_tensor", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<LeOp, ascendc::CompareExtendContiguousRegBaseOp, ascendc::CMPMODE::LE, ScalarOperand::None,
                             TmpBufferUse::Unused>(registry, "le.tensor_tensor.extend_reg_base", ApiFamily::RegBased,
                                                   kCompareRegElementTypes)) ||
      failed(registerCompare<LeOp, ascendc::CompareScalarExtendContiguousOp, ascendc::CMPMODE::LE, ScalarOperand::Rhs,
                             TmpBufferUse::Passed>(registry, "le.tensor_scalar", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<LeOp, ascendc::CompareScalarExtendContiguousRegBaseOp, ascendc::CMPMODE::LE,
                             ScalarOperand::Rhs, TmpBufferUse::Unused>(registry, "le.tensor_scalar.extend_reg_base",
                                                                       ApiFamily::RegBased, kCompareRegElementTypes)) ||
      failed(registerCompare<GtOp, ascendc::CompareExtendContiguousOp, ascendc::CMPMODE::GT, ScalarOperand::None,
                             TmpBufferUse::Passed>(registry, "gt.tensor_tensor", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<GtOp, ascendc::CompareExtendContiguousRegBaseOp, ascendc::CMPMODE::GT, ScalarOperand::None,
                             TmpBufferUse::Unused>(registry, "gt.tensor_tensor.extend_reg_base", ApiFamily::RegBased,
                                                   kCompareRegElementTypes)) ||
      failed(registerCompare<GtOp, ascendc::CompareScalarExtendContiguousOp, ascendc::CMPMODE::GT, ScalarOperand::Rhs,
                             TmpBufferUse::Passed>(registry, "gt.tensor_scalar", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<GtOp, ascendc::CompareScalarExtendContiguousRegBaseOp, ascendc::CMPMODE::GT,
                             ScalarOperand::Rhs, TmpBufferUse::Unused>(registry, "gt.tensor_scalar.extend_reg_base",
                                                                       ApiFamily::RegBased, kCompareRegElementTypes)) ||
      failed(registerCompare<GeOp, ascendc::CompareExtendContiguousOp, ascendc::CMPMODE::GE, ScalarOperand::None,
                             TmpBufferUse::Passed>(registry, "ge.tensor_tensor", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<GeOp, ascendc::CompareExtendContiguousRegBaseOp, ascendc::CMPMODE::GE, ScalarOperand::None,
                             TmpBufferUse::Unused>(registry, "ge.tensor_tensor.extend_reg_base", ApiFamily::RegBased,
                                                   kCompareRegElementTypes)) ||
      failed(registerCompare<GeOp, ascendc::CompareScalarExtendContiguousOp, ascendc::CMPMODE::GE, ScalarOperand::Rhs,
                             TmpBufferUse::Passed>(registry, "ge.tensor_scalar", ApiFamily::MemBased,
                                                   kCompareMemElementTypes)) ||
      failed(registerCompare<GeOp, ascendc::CompareScalarExtendContiguousRegBaseOp, ascendc::CMPMODE::GE,
                             ScalarOperand::Rhs, TmpBufferUse::Unused>(registry, "ge.tensor_scalar.extend_reg_base",
                                                                       ApiFamily::RegBased, kCompareRegElementTypes))) {
    return failure();
  }
  return success();
}

}  // namespace mlir::ascend::kernel
