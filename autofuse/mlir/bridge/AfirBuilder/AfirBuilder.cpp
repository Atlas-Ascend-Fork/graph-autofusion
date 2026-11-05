/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AfirBuilder.cpp - Build an AFIR module from neutral schedule info -*- C++ -*-===//

#include "bridge/AfirBuilder/AfirBuilder.h"
#include "bridge/AfirBuilder/WorkspaceViews.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"
#include "support/AfirElementTypes.h"
#include "support/CapabilityShortfall.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Analysis/ReducePlan.h"
#include "dialect/AFIR/Analysis/ResourceIds.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

// 32 bytes is the UB access granularity every other on-chip allocation in this
// bridge is aligned to; a tmp buffer is no different.
constexpr int64_t kTmpBufferAlignmentBytes = 32;

// Reduce 的 destination 后接变长 scratch，需单独设置 operandSegmentSizes。
// 高阶归约 API 要求至少一块 scratch。
bool mlir::afir::bridge::isReduceNodeType(llvm::StringRef type) {
  return type == "Sum" || type == "Max" || type == "Min" || type == "Mean" || type == "Prod" || type == "Any" ||
         type == "All";
}

namespace mlir {
namespace afir {
namespace bridge {

namespace {

llvm::ArrayRef<AFIRBridgeOpMapping> bridgeableOperationMappings() {
#define AFIR_BRIDGE_MAPPING(SCHEDULER_NAME, OP, FORM) {SCHEDULER_NAME, OP::getOperationName(), FORM},
  static const AFIRBridgeOpMapping mappings[] = {AFIR_BRIDGE_OPERATION_REGISTRY(AFIR_BRIDGE_MAPPING)};
#undef AFIR_BRIDGE_MAPPING
  return mappings;
}

// AscGraph 名 → AFIR 名。两边的拼写不保证一致，`LeakyRelu` 就不一致——见
// `AfirBuilder.h` 上那段。ScheduleToAfir 按这张表读 `ir_attr`，下面
// `buildBufferNodeOp` 按同一张表挂成 op 属性，两处不会再各写一个 `if`。
const FloatIrAttrMapping kFloatIrAttrMappings[] = {
    {"LeakyRelu", "negative_slope", "negative_slop"},
    {"Axpy", "alpha", "alpha"},
};

// 同一件事的 int64 那一半。多项式族八个的 `n` 两边同名（legacy 的
// `REG_ASC_IR(...).Attr<int64_t>("n")`，AFIR 的 ODS 属性也叫 `n`），但两列照旧都写出来
// ——`negative_slope` → `negative_slop` 那个先例说明「同名」不是这一族的性质。
const IntIrAttrMapping kIntIrAttrMappings[] = {
    {"ChebyshevPolynomialT", "n", "n"},        {"ChebyshevPolynomialU", "n", "n"},
    {"ChebyshevPolynomialV", "n", "n"},        {"ChebyshevPolynomialW", "n", "n"},
    {"ShiftedChebyshevPolynomialT", "n", "n"}, {"ShiftedChebyshevPolynomialU", "n", "n"},
    {"ShiftedChebyshevPolynomialV", "n", "n"}, {"ShiftedChebyshevPolynomialW", "n", "n"},
};

// 带 `indexing_maps` 的那几个节点类型，也就是**跨 GM / 片上边界**的搬运。两端的形状
// 不同（落点通常是源的一个 tile；`Nddma` 连元素数都不同），所以下标对应必须显式写出来，
// 而其余算子两端同形、不带这个属性。`Transpose` 也在内：它的两端形状不同（轴序重排），
// 下标对应同样必须显式写出来——`makeTransferIndexingMapsAttr` 从两端 axis 数组机械生成，
// perm 不需要额外信息。
//
// 一张表而不是两处 `||`：buffer 形态与 tensor 形态各有一个取用点，分开写的话加第三个
// 搬运 op 时漏掉一处的后果是那个形态下 `indexing_maps` 缺失，而 verifier 把它报成
// `InvalidIndexingMaps`——看起来像形状错，其实是这张表漏了一行。
bool isTransferNodeType(llvm::StringRef type) {
  return type == "Load" || type == "Store" || type == "Nddma" || type == "Transpose";
}

bool isFormSupported(StringRef schedulerName, BridgeOpForm form) {
  for (const AFIRBridgeOpMapping &mapping : bridgeableOperationMappings()) {
    if (mapping.scheduler_name == schedulerName) {
      return mapping.form == BridgeOpForm::Both || mapping.form == form;
    }
  }
  return false;
}

// Names the operations the registry marks as buildable in `form`, for a
// diagnostic. Built from the table rather than written out, so it cannot go
// stale the way a hand-listed message does.
std::string listSupportedOperations(BridgeOpForm form) {
  std::string names;
  for (const AFIRBridgeOpMapping &mapping : bridgeableOperationMappings()) {
    if (mapping.form != BridgeOpForm::Both && mapping.form != form) {
      continue;
    }
    if (!names.empty()) {
      names += ", ";
    }
    names += mapping.scheduler_name.str();
  }
  return names;
}

// 拼写 → MLIR 类型。**两种拼写都收**：长的是 bridge 发过来的（`DTypeStr`），短的是
// 对照表与 lit 用例里的。两张表的行由 `support/AfirElementTypes.h` 给，所以这里不会出现
// 「bridge 发得出、builder 解不出」那种一个字母的错。
//
// 收的范围比供给表宽，与 `DTypeStr` 一致：解析得出来只是第一步，收不收由
// `IsAfirModeledDType` 和各 op 的 ODS 约束说。
Type parseDtypeImpl(MLIRContext *ctx, StringRef dtype) {
  auto build = [ctx](afir::AfirElementTypeKind kind, unsigned width) -> Type {
    switch (kind) {
      case afir::AfirElementTypeKind::SignlessInteger:
        return IntegerType::get(ctx, width);
      case afir::AfirElementTypeKind::UnsignedInteger:
        return IntegerType::get(ctx, width, IntegerType::Unsigned);
      case afir::AfirElementTypeKind::Float16:
        return Float16Type::get(ctx);
      case afir::AfirElementTypeKind::Float32:
        return Float32Type::get(ctx);
      case afir::AfirElementTypeKind::BFloat16:
        return BFloat16Type::get(ctx);
    }
    return Type();
  };
  for (const afir::AfirNameableElementType &entry : afir::afirNameableElementTypes()) {
    if (dtype == entry.builderSpelling || dtype == entry.csvSpelling) {
      return build(entry.kind, entry.width);
    }
  }
  return Type();
}

RankedTensorType makeTensorType(MLIRContext *ctx, const TensorDescDTO &t) {
  Type elem = parseDtypeImpl(ctx, t.dtype);
  if (!elem) {
    return {};
  }
  return RankedTensorType::get(t.shape, elem);
}

// buffer 形态的地址空间写在值的类型上，不再只存在于 `#afir.buffer_desc` 里
// （设计文档 3.1 步 3 / D8）。空间信息在 `BufferDescDTO` 上，因此这里收的是整个
// `TensorValueDTO` 而不只是 `TensorDescDTO`——tensor 形态没有地址空间可言，
// `makeTensorType` 保持只收 `TensorDescDTO`。
MemRefType makeBufferType(MLIRContext *ctx, const TensorValueDTO &v) {
  Type elem = parseDtypeImpl(ctx, v.tensor.dtype);
  if (!elem) {
    return {};
  }
  return MemRefType::get(v.tensor.shape, elem, MemRefLayoutAttrInterface{}, afir::SpaceAttr::get(ctx, v.buffer.space));
}

FailureOr<ArrayAttr> makeExpressionArrayAttr(MLIRContext *ctx, ArrayRef<std::string> expressions) {
  SmallVector<Attribute> attrs;
  attrs.reserve(expressions.size());
  for (const std::string &expression : expressions) {
    FailureOr<afir::ExprAttr> attr = makeExprAttr(ctx, expression);
    if (failed(attr)) {
      return failure();
    }
    attrs.push_back(*attr);
  }
  return ArrayAttr::get(ctx, attrs);
}

afir::AxisType parseAxisType(StringRef s) {
  return llvm::StringSwitch<afir::AxisType>(s)
      .Case("original", afir::AxisType::Original)
      .Case("block_outer", afir::AxisType::BlockOuter)
      .Case("block_inner", afir::AxisType::BlockInner)
      .Case("tile_outer", afir::AxisType::TileOuter)
      .Case("tile_inner", afir::AxisType::TileInner)
      .Case("merged", afir::AxisType::Merged)
      .Default(afir::AxisType::Invalid);
}

FailureOr<SmallVector<afir::ExprAttr>> makeExpressionAttrs(MLIRContext *ctx, ArrayRef<std::string> expressions) {
  SmallVector<afir::ExprAttr> attrs;
  attrs.reserve(expressions.size());
  for (const auto &expression : expressions) {
    FailureOr<afir::ExprAttr> attr = makeExprAttr(ctx, expression);
    if (failed(attr)) {
      return failure();
    }
    attrs.push_back(*attr);
  }
  return attrs;
}

FailureOr<afir::TensorDescAttr> makeTensorDescAttr(MLIRContext *ctx, const TensorDescDTO &t) {
  FailureOr<SmallVector<afir::ExprAttr>> repeats = makeExpressionAttrs(ctx, t.repeats);
  FailureOr<SmallVector<afir::ExprAttr>> strides = makeExpressionAttrs(ctx, t.strides);
  FailureOr<SmallVector<afir::ExprAttr>> vectorizedStrides = makeExpressionAttrs(ctx, t.vectorized_strides);
  if (failed(repeats) || failed(strides) || failed(vectorizedStrides)) {
    return failure();
  }
  return afir::TensorDescAttr::get(ctx, t.axis, *repeats, *strides, t.vectorized_axis, *vectorizedStrides,
                                   afir::RoleAttr::get(ctx, t.role));
}

FailureOr<afir::BufferDescAttr> makeBufferDescAttr(MLIRContext *ctx, const BufferDescDTO &b) {
  FailureOr<afir::ExprAttr> sizeBytesExpr = failure();
  if (!b.sizeBytesExpr.empty()) {
    // Scheduler expressions need not be canonical; normalize before creating the attribute.
    auto parsed = afir::Expression::ParseRelaxed(b.sizeBytesExpr);
    if (parsed) {
      auto simplified = parsed.value().Simplify();
      if (simplified) {
        sizeBytesExpr = makeExprAttr(ctx, simplified.value().Serialize());
      }
    } else {
      // Let attribute creation diagnose malformed expressions.
      sizeBytesExpr = makeExprAttr(ctx, b.sizeBytesExpr);
    }
    if (failed(sizeBytesExpr)) {
      return failure();
    }
  }
  return afir::BufferDescAttr::get(ctx, afir::SpaceAttr::get(ctx, b.space), b.bufferId, b.reuseId, b.slotCount,
                                   b.queueDepth, succeeded(sizeBytesExpr) ? *sizeBytesExpr : afir::ExprAttr(),
                                   b.sizeFromResource, b.alignmentBytes, b.lifeTimeAxisId);
}

template <typename Range>
FailureOr<ArrayAttr> makeTensorDescArrayAttr(MLIRContext *ctx, const Range &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (const TensorValueDTO &value : values) {
    FailureOr<afir::TensorDescAttr> descriptor = makeTensorDescAttr(ctx, value.tensor);
    if (failed(descriptor)) {
      return failure();
    }
    attrs.push_back(*descriptor);
  }
  return ArrayAttr::get(ctx, attrs);
}

FailureOr<ArrayAttr> makeWorkspacePlanAttr(MLIRContext *ctx, ArrayRef<BufferDescDTO> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (const BufferDescDTO &value : values) {
    FailureOr<afir::BufferDescAttr> descriptor = makeBufferDescAttr(ctx, value);
    if (failed(descriptor)) {
      return failure();
    }
    attrs.push_back(*descriptor);
  }
  return ArrayAttr::get(ctx, attrs);
}

FailureOr<ArrayAttr> makeTransferIndexingMapsAttr(MLIRContext *ctx, const TensorDescDTO &source,
                                                  const TensorDescDTO &destination) {
  SmallVector<int64_t> iterationAxes;
  auto appendAxes = [&](const TensorDescDTO &tensor) {
    if (!tensor.axis.empty()) {
      for (int64_t axis : tensor.axis) {
        if (!llvm::is_contained(iterationAxes, axis)) {
          iterationAxes.push_back(axis);
        }
      }
      return;
    }
    for (int64_t axis = 0; axis < static_cast<int64_t>(tensor.shape.size()); ++axis) {
      if (!llvm::is_contained(iterationAxes, axis)) {
        iterationAxes.push_back(axis);
      }
    }
  };
  appendAxes(source);
  appendAxes(destination);

  auto makeMap = [&](const TensorDescDTO &tensor) {
    SmallVector<AffineExpr> results;
    ArrayRef<int64_t> axes = tensor.axis;
    for (int64_t index = 0; index < static_cast<int64_t>(tensor.shape.size()); ++index) {
      int64_t axis = axes.empty() || index >= static_cast<int64_t>(axes.size()) ? index : axes[index];
      auto iter = llvm::find(iterationAxes, axis);
      results.push_back(iter == iterationAxes.end()
                            ? getAffineConstantExpr(0, ctx)
                            : getAffineDimExpr(static_cast<unsigned>(iter - iterationAxes.begin()), ctx));
    }
    return AffineMap::get(iterationAxes.size(), 0, results, ctx);
  };

  SmallVector<Attribute> maps{AffineMapAttr::get(makeMap(source)), AffineMapAttr::get(makeMap(destination))};
  return ArrayAttr::get(ctx, maps);
}

// Attach the template's loop axes as an `afir.asc_graph` attribute on the func
// — carrying SchedInfo (axis order/type/size) for later tiling passes.
FailureOr<afir::AscGraphAttrGroupsAttr> makeTemplateAttr(MLIRContext *ctx, const TemplateInfo &g) {
  SmallVector<afir::AxisAttr> axes;
  for (const auto &a : g.axes) {
    FailureOr<afir::ExprAttr> size = makeExprAttr(ctx, a.size);
    if (failed(size)) {
      return failure();
    }
    afir::ExprAttr align;
    if (!a.align.empty()) {
      FailureOr<afir::ExprAttr> alignExpression = makeExprAttr(ctx, a.align);
      if (failed(alignExpression)) {
        return failure();
      }
      align = *alignExpression;
    }
    axes.push_back(afir::AxisAttr::get(ctx, a.id, StringAttr::get(ctx, a.name), parseAxisType(a.type), a.bind_block,
                                       *size, align, a.from, a.split_pair_other_id));
  }
  FailureOr<SmallVector<afir::ExprAttr>> sizeVars = makeExpressionAttrs(ctx, g.size_vars);
  if (failed(sizeVars)) {
    return failure();
  }
  return afir::AscGraphAttrGroupsAttr::get(ctx, g.tiling_key, axes, afir::AscGraphType::Compute, *sizeVars);
}

struct DynamicSizeCandidate {
  Value source;
  int64_t dimension;
  std::string expression;
};

bool proveExpressionsEqual(StringRef lhs, StringRef rhs) {
  // shape_exprs 遵循 ExprAttr 的规范形式要求。
  auto left = afir::Expression::Parse(lhs);
  auto right = afir::Expression::Parse(rhs);
  if (!left || !right) {
    return false;
  }
  auto leftSimplified = left.value().Simplify();
  auto rightSimplified = right.value().Simplify();
  if (!leftSimplified || !rightSimplified) {
    return false;
  }
  // ExpectEq(a, a) 会规范化为 True，生成的谓词需使用宽松解析。
  auto predicate = afir::Expression::ParseRelaxed("ExpectEq(" + leftSimplified.value().Serialize() + ", " +
                                                  rightSimplified.value().Serialize() + ")");
  if (!predicate) {
    return false;
  }
  auto proof = afir::detail::ExpressionAccess::Prove(predicate.value());
  return proof && proof.value() == afir::detail::TruthValue::True;
}

SmallVector<DynamicSizeCandidate> collectDynamicSizeCandidates(MemRefType resultType, ArrayRef<Value> operands,
                                                               ArrayRef<TensorValueDTO> inputInfos,
                                                               int64_t resultAxis) {
  SmallVector<DynamicSizeCandidate> candidates;
  for (auto [inputIndex, input] : llvm::enumerate(inputInfos)) {
    if (inputIndex >= operands.size()) {
      break;
    }
    // A destination buffer is allocated before its scheduled AFIR producer is
    // moved into the generated loop. Only function arguments are guaranteed to
    // remain loop-invariant here; querying a scheduled AFIR result would make
    // the candidate itself escape the loop body.
    if (!isa<BlockArgument>(operands[inputIndex])) {
      continue;
    }
    auto inputType = dyn_cast<MemRefType>(operands[inputIndex].getType());
    if (!inputType || inputType.getRank() != resultType.getRank()) {
      continue;
    }
    for (int64_t inputDim = 0; inputDim < inputType.getRank(); ++inputDim) {
      if (!input.tensor.axis.empty() && static_cast<size_t>(inputDim) >= input.tensor.axis.size()) {
        continue;
      }
      int64_t inputAxis = input.tensor.axis.empty() ? inputDim : input.tensor.axis[inputDim];
      if (inputAxis != resultAxis || input.tensor.shape_exprs.size() != static_cast<size_t>(inputType.getRank())) {
        continue;
      }
      candidates.push_back({operands[inputIndex], inputDim, input.tensor.shape_exprs[inputDim]});
    }
  }
  return candidates;
}

// 为一个节点建 Op 的过程中，整条调用链自上而下都要用到的同一组量：谁在建、建在哪、
// 建的是哪个节点、它的操作数、以及结果落在哪。此前这五六项在
// `buildNodeOp` -> `buildBufferNodeOp` -> `getBufferNodeDestination` -> `makeBufferDestination`
// -> `makeDynamicBufferDimension` 这条链上逐层原样透传，每层签名都要重列一遍。
struct NodeBuildContext {
  OpBuilder &builder;
  Location loc;
  const NodeInfo &node;
  ArrayRef<Value> operands;
  // 结果的描述。buffer 形态由调用方给出（Output 节点可能改写它）；tensor 形态由
  // `buildLogicalNodeOp` 自己从节点的 outputs 取，不看这一份。
  const TensorValueDTO &resultValue;
  // 仅 buffer 形态的 Store 用：写回的那个函数参数。其余情况为空。
  Value externalDestination;
  // 这个节点是不是建在 `afir.vector_func` 的 region 里。region 内的值是**寄存器**，
  // 没有存储，所以不挂 buffer descriptor——挂了的失败方式是资源收集那一步报
  // 「canonical buffer descriptor is missing buffer_id」（`Analysis/ResourceIds.cpp`），
  // 而那句话指不到这里。
  bool insideVectorFunc = false;

  MLIRContext *context() const {
    return builder.getContext();
  }
};

FailureOr<Value> makeDynamicBufferDimension(const NodeBuildContext &ctx, MemRefType type, int64_t resultDim) {
  OpBuilder &builder = ctx.builder;
  Location loc = ctx.loc;
  const TensorValueDTO &resultInfo = ctx.resultValue;
  if (!resultInfo.tensor.axis.empty() && resultDim >= static_cast<int64_t>(resultInfo.tensor.axis.size())) {
    emitError(loc) << "InvalidDescriptor: result axis mapping does not cover dynamic dimension " << resultDim;
    return failure();
  }
  if (resultInfo.tensor.shape_exprs.size() != static_cast<size_t>(type.getRank())) {
    emitError(loc) << "UnsupportedDynamicShape: destination shape expressions must match the result rank";
    return failure();
  }
  int64_t resultAxis = resultInfo.tensor.axis.empty() ? resultDim : resultInfo.tensor.axis[resultDim];
  const std::string &resultExpression = resultInfo.tensor.shape_exprs[resultDim];
  SmallVector<DynamicSizeCandidate> candidates =
      collectDynamicSizeCandidates(type, ctx.operands, ctx.node.inputs, resultAxis);
  for (const DynamicSizeCandidate &candidate : candidates) {
    if (proveExpressionsEqual(candidate.expression, resultExpression)) {
      return builder.create<memref::DimOp>(loc, candidate.source, candidate.dimension).getResult();
    }
  }

  // 没有同轴的输入可问时，范围就从调度器给的 shape 表达式本身算出来。
  //
  // 这一支此前返回 `arith.constant 1`：`memref.alloc` 的动态尺寸操作数在 AscendC 物化
  // 路径上没有读点（队列字节数来自 `size_bytes_expr`，成员尺寸来自 tensor descriptor），
  // 所以那个 1 不会算错任何东西——它只是一句假话。而 `memref<?x...>` 的动态维就是**这块
  // 缓冲有多大**：写 1 等于说「运行期为 1」，读 IR 的人、任何基于 memref 的分析、以及
  // ADR-11/12 要落的 `memref.view` 形态都会被它误导。
  //
  // 表达式里的自由符号是 tiling 参数（`z0Tb_size` 之类的轴尺寸，或 `s0` 这样的 shape
  // 符号），逐个取成 `afir.tiling_param`，表达式本身成为一个 `afir.symbolic_eval`；两者
  // 都落在入口块的尺寸序言里，因此支配所有 alloc。常量表达式不走这条路，直接建常量。
  int64_t literal = 0;
  if (!llvm::StringRef(resultExpression).getAsInteger(10, literal)) {
    if (literal <= 0) {
      emitError(loc) << "UnsupportedDynamicShape: destination shape expression must be positive, got "
                     << resultExpression;
      return failure();
    }
    return builder.create<arith::ConstantIndexOp>(loc, literal).getResult();
  }
  FailureOr<afir::ExprAttr> expression = afir::makeExprAttr(builder.getContext(), resultExpression);
  if (failed(expression)) {
    return failure();
  }
  Block *entry = builder.getInsertionBlock();
  if (entry == nullptr) {
    emitError(loc) << "UnsupportedDynamicShape: no insertion block for the destination extent";
    return failure();
  }
  return afir::materializeRuntimeSize(*entry, loc, *expression);
}

FailureOr<Value> makeBufferDestination(const NodeBuildContext &ctx, MemRefType type) {
  if (type.hasStaticShape()) {
    return ctx.builder.create<memref::AllocOp>(ctx.loc, type).getResult();
  }
  SmallVector<Value> dynamicSizes;
  for (auto [resultDim, size] : llvm::enumerate(type.getShape())) {
    if (size == ShapedType::kDynamic) {
      FailureOr<Value> dynamicSize = makeDynamicBufferDimension(ctx, type, static_cast<int64_t>(resultDim));
      if (failed(dynamicSize)) {
        return failure();
      }
      dynamicSizes.push_back(*dynamicSize);
    }
  }
  return ctx.builder.create<memref::AllocOp>(ctx.loc, type, dynamicSizes).getResult();
}

// The bufferizable op bases carry two optional operands -- the destination and
// the tmp buffer -- so ODS gives them AttrSizedOperandSegments, and neither the
// generic OperationState builder nor the (TypeRange, ValueRange, attrs) typed
// builder fills the segment attribute in. Both bridge paths do it here.
//
// `afir.store` is the one bufferizable operation without the trait -- its base
// has no tmp buffer, because neither Store nor Load needs a tmp buffer on either API
// family -- so the callers test for the trait rather than assuming it.
DenseI32ArrayAttr makeOperandSegmentSizes(OpBuilder &builder, size_t inputCount, bool hasDestination,
                                          bool hasTmpBuffer = false) {
  SmallVector<int32_t> segments(inputCount, 1);
  segments.push_back(hasDestination ? 1 : 0);
  segments.push_back(hasTmpBuffer ? 1 : 0);
  return builder.getDenseI32ArrayAttr(segments);
}

bool needsOperandSegmentSizes(OperationName name) {
  return name.hasTrait<OpTrait::AttrSizedOperandSegments>();
}

// The AFIR operation a scheduler node type maps to, straight out of the
// registry. Empty when the node is not in the table; the caller has already
// rejected that case, but returning empty rather than defaulting to some
// operation is the point: this used to be a four-way conditional chain whose
// else-branch silently built an `afir.add`, so every newly whitelisted node
// type was emitted as an addition until someone extended the chain.
StringRef afirOperationNameFor(StringRef schedulerName) {
  for (const AFIRBridgeOpMapping &mapping : bridgeableOperationMappings()) {
    if (mapping.scheduler_name == schedulerName) {
      return mapping.afir_operation_name;
    }
  }
  return {};
}

LogicalResult appendScheduleAttrs(OpBuilder &b, Location loc, const NodeInfo &node, const TensorValueDTO &resultValue,
                                  SmallVectorImpl<NamedAttribute> &attrs) {
  MLIRContext *ctx = b.getContext();
  // 调度位置合为一个强类型属性：轴集合与最内层循环轴的一致性由 ScheduleAttr 的
  // verifier 保证，不再分散在两个无类型属性里各自漂移。
  //
  // 此前还写过一个 `afir.vectorized_axis`，其值就是本算子局部侧 `TensorDescAttr` 的
  // `vectorized_axis` 字段原样拷贝（步 2 清掉的那批影子属性同一形态）。唯一读点
  // `TransferPlan` 已改为直接读该 descriptor，这里不再产生拷贝。
  //
  // 这里用 `getChecked` 而不是 `get`：轴集合与 loop_axis 是调度器原样交过来的外部输入，
  // 中间没有任何校验。`get` 把 verifier 放在 assert 里——断言关掉就直接构造出违反不变量
  // 的属性，打开也只是在生产 .so 里 abort。`getChecked` 把它变成一条正常的诊断。
  auto emitScheduleError = [&]() -> InFlightDiagnostic {
    return emitError(loc) << "InvalidSchedule: node '" << node.name << "': ";
  };
  // 缺省是 `no_cache`，所以空串与「没有这一项」是同一件事。拼错的取值必须报出来：
  // 静默落到 `no_cache` 的后果是每个迭代都重算——不错，但上游那次决策就无声消失了。
  afir::ExecCondition execCondition = afir::ExecCondition::NoCache;
  if (!node.exec_condition.empty()) {
    std::optional<afir::ExecCondition> parsed = afir::symbolizeExecCondition(node.exec_condition);
    if (!parsed) {
      emitScheduleError() << "unknown exec_condition '" << node.exec_condition << "'";
      return failure();
    }
    execCondition = *parsed;
  }
  if (!node.sched_axis.empty()) {
    auto sched = afir::ScheduleAttr::getChecked(emitScheduleError, ctx, ArrayRef<int64_t>(node.sched_axis),
                                                node.loop_axis, execCondition);
    if (!sched) {
      return failure();
    }
    attrs.emplace_back(StringAttr::get(ctx, afir::kSchedAttrName), sched);
  } else if (execCondition != afir::ExecCondition::NoCache) {
    // 没有 `afir.sched` 就没有地方挂它。静默丢掉的后果与上面同类，所以在这里挡住而不是
    // 让属性构造去发现——那一支根本不会执行。
    emitScheduleError() << "exec_condition '" << node.exec_condition << "' requires a scheduled axis list";
    return failure();
  }
  if (!resultValue.tensor.shape_exprs.empty()) {
    FailureOr<ArrayAttr> shapeExpressions = makeExpressionArrayAttr(ctx, resultValue.tensor.shape_exprs);
    if (failed(shapeExpressions)) {
      return failure();
    }
    attrs.emplace_back(StringAttr::get(ctx, afir::kSymbolicShapeAttrName), *shapeExpressions);
  }
  return success();
}

FailureOr<Value> getBufferNodeDestination(const NodeBuildContext &ctx, MemRefType destinationType) {
  if (ctx.node.type == "Store") {
    if (!ctx.externalDestination) {
      emitError(ctx.loc) << "InvalidReturnBoundary: buffer-form store '" << ctx.node.name
                         << "' requires an external output destination";
      return failure();
    }
    if (ctx.externalDestination.getType() != destinationType) {
      emitError(ctx.loc) << "InvalidReturnBoundary: store '" << ctx.node.name << "' destination type "
                         << ctx.externalDestination.getType() << " does not match the graph output type "
                         << destinationType;
      return failure();
    }
    return ctx.externalDestination;
  }
  return makeBufferDestination(ctx, destinationType);
}

// `destinations` 是这个节点**全部**目的地的 DTO；空表示「只有一个，就是 `resultValue`」，
// 那是一结果算子的常态。
//
// **多输出的算子必须显式传全部**：descriptor 的个数由 verifier 按算子的结果 / destination
// 个数查（`AFIRDialect.cpp` 的 `getExpectedDescriptorCount`），只挂第一个的后果是
//
//     DescriptorCountMismatch: 'afir.destination_descs' expected 2 element but found 1
//
// ——那是 verifier，**硬失败**。`afir.vector_func` 是今天唯一的多输出算子（菱形两消费者融成
// 一个 VF 就是双输出），2026-10-26 实测探针 reg 侧 5 格命中这一条。
LogicalResult appendBufferNodeAttrs(const NodeBuildContext &nodeCtx, SmallVectorImpl<NamedAttribute> &attrs,
                                    llvm::ArrayRef<TensorValueDTO> destinations = {}) {
  const NodeInfo &node = nodeCtx.node;
  const TensorValueDTO &resultValue = nodeCtx.resultValue;
  MLIRContext *ctx = nodeCtx.context();
  FailureOr<ArrayAttr> operandDescs = makeTensorDescArrayAttr(ctx, node.inputs);
  SmallVector<TensorValueDTO, 1> destinationValues;
  if (destinations.empty()) {
    destinationValues.push_back(resultValue);
  } else {
    destinationValues.assign(destinations.begin(), destinations.end());
  }
  FailureOr<ArrayAttr> destinationDescs = makeTensorDescArrayAttr(ctx, destinationValues);
  if (failed(operandDescs) || failed(destinationDescs)) {
    return failure();
  }
  // buffer descriptor 不在这里：操作数的那一份在它自己的定义点（上游 Op 的
  // destination alloc，或 kernel 参数的 arg attr），destination 的那一份由
  // `buildBufferNodeOp` 写在新建的 `memref.alloc` 上；Store 写回的是函数参数，
  // 它的那一份已经在 arg attr 上，不再复制第二份（设计文档 5.1 步 8）。
  attrs.emplace_back(StringAttr::get(ctx, afir::kOperandDescsAttrName), *operandDescs);
  if (node.type != "Store") {
    attrs.emplace_back(StringAttr::get(ctx, afir::kResultDescsAttrName), *destinationDescs);
  }
  attrs.emplace_back(StringAttr::get(ctx, afir::kDestinationDescsAttrName), *destinationDescs);
  const TensorValueDTO &scheduleValue =
      node.type == "Store" && !node.inputs.empty() ? node.inputs.front() : resultValue;
  if (failed(appendScheduleAttrs(nodeCtx.builder, nodeCtx.loc, node, scheduleValue, attrs))) {
    return failure();
  }
  if (isTransferNodeType(node.type)) {
    if (node.inputs.size() != 1) {
      emitError(nodeCtx.loc) << "InvalidOperation: " << node.type << " requires one input";
      return failure();
    }
    FailureOr<ArrayAttr> indexingMaps =
        makeTransferIndexingMapsAttr(ctx, node.inputs.front().tensor, resultValue.tensor);
    if (failed(indexingMaps)) {
      return failure();
    }
    attrs.emplace_back(StringAttr::get(ctx, afir::kIndexingMapsAttrName), *indexingMaps);
  }
  // host tiling 字段的节点级引用：`ScheduleToAfir` 那边 `GetApiTilingInfo` 有值的节点
  // 才带（`Transpose` 与 `Pad`）。挂 op 上而不是只靠 func 级
  // `afir.api_tiling_fields`，是因为同一个模板里可以有多个同类型节点，func 级分不出
  // 谁取哪个字段。
  if (node.api_tiling_field) {
    attrs.emplace_back(StringAttr::get(ctx, afir::kApiTilingFieldAttrName),
                       StringAttr::get(ctx, *node.api_tiling_field));
  }
  return success();
}

// Materializes the tmp buffer a node's API needs, as a one-dimensional
// `ui8` allocation whose extent is the canonical resource size `b{id}_size`.
//
// Bytes and elements coincide for `ui8`, which is why the dynamic extent can be
// the resource size itself rather than a placeholder: the IR then states the
// buffer's real size instead of a convenient lie. The descriptor says
// `size_from_resource`, so MemoryAllocate binds the same name when it reserves
// the TBuf -- one number, referenced twice, never recomputed here.
FailureOr<Value> makeTmpBufferAllocation(const NodeBuildContext &ctx, const TmpBufferDTO &tmpBuffer) {
  MLIRContext *context = ctx.context();
  Block *entry = ctx.builder.getInsertionBlock();
  if (entry == nullptr) {
    emitError(ctx.loc) << "InvalidBinding: no insertion block for the tmp buffer extent";
    return failure();
  }
  FailureOr<afir::ExprAttr> sizeName =
      afir::makeExprAttr(context, afir::makeCanonicalResourceSizeName(/*isQueue=*/false, tmpBuffer.id));
  if (failed(sizeName)) {
    return failure();
  }
  FailureOr<Value> extent = afir::materializeRuntimeSize(*entry, ctx.loc, *sizeName);
  if (failed(extent)) {
    return failure();
  }

  auto elementType = IntegerType::get(context, 8, IntegerType::Unsigned);
  auto type = MemRefType::get({ShapedType::kDynamic}, elementType, MemRefLayoutAttrInterface{},
                              afir::SpaceAttr::get(context, afir::Space::UB));
  auto allocation = ctx.builder.create<memref::AllocOp>(ctx.loc, type, ValueRange{*extent});
  afir::setBufferDescAttr(
      allocation.getOperation(),
      afir::BufferDescAttr::get(context, afir::SpaceAttr::get(context, afir::Space::UB), tmpBuffer.id, /*reuseId=*/-1,
                                /*slotCount=*/-1,
                                /*queueDepth=*/-1, afir::ExprAttr(),
                                /*sizeFromResource=*/true, kTmpBufferAlignmentBytes, tmpBuffer.life_time_axis_id));
  return allocation.getResult();
}

LogicalResult appendBufferNodeIrAttrs(const NodeBuildContext &ctx, SmallVectorImpl<NamedAttribute> &attrs) {
  const NodeInfo &node = ctx.node;
  // op 自己的属性（`LeakyRelu` 的斜率、`Axpy` 的 alpha）。AscGraph 上是节点的
  // `ir_attr`，AFIR 上是 op 属性、不是操作数，所以它不进 `ctx.operands`、不占缓冲、
  // 也不参与三段划分。缺了就报：一个没有斜率的 LeakyRelu、一个没有 alpha 的 Axpy
  // 都没有可用的含义。
  if (const FloatIrAttrMapping *floatAttr = findFloatIrAttr(node.type)) {
    if (!node.float_ir_attr) {
      emitError(ctx.loc) << "AfirBuilder: " << node.type << " node '" << node.name << "' carries no "
                         << floatAttr->ascgraph_attr_name;
      return failure();
    }
    attrs.emplace_back(ctx.builder.getStringAttr(floatAttr->afir_attr_name),
                       ctx.builder.getF32FloatAttr(*node.float_ir_attr));
  }
  if (const IntIrAttrMapping *intAttr = findIntIrAttr(node.type)) {
    if (!node.int_ir_attr) {
      emitError(ctx.loc) << "AfirBuilder: " << node.type << " node '" << node.name << "' carries no "
                         << intAttr->ascgraph_attr_name;
      return failure();
    }
    attrs.emplace_back(ctx.builder.getStringAttr(intAttr->afir_attr_name),
                       ctx.builder.getI64IntegerAttr(*node.int_ir_attr));
  }
  return success();
}

LogicalResult appendReduceScratchOperands(const NodeBuildContext &ctx, Value destination, OperationState &state) {
  const NodeInfo &node = ctx.node;
  if (node.tmp_buffers.empty()) {
    emitError(ctx.loc) << "InvalidOperation: reduce node '" << node.name << "' has no scheduler temporary buffer";
    return failure();
  }
  state.addOperands(destination);
  SmallVector<Attribute> scratchRoles;
  for (const TmpBufferDTO &tmp : node.tmp_buffers) {
    if (!tmp.reduce_role) {
      emitError(ctx.loc) << "InvalidBinding: reduce scratch " << tmp.id << " has no normalized role";
      return failure();
    }
    scratchRoles.push_back(ctx.builder.getStringAttr(stringifyReduceScratchRole(*tmp.reduce_role)));
    FailureOr<Value> scratch = makeTmpBufferAllocation(ctx, tmp);
    if (failed(scratch)) {
      return failure();
    }
    state.addOperands(*scratch);
  }
  state.addAttribute(kReduceScratchRolesAttr, ctx.builder.getArrayAttr(scratchRoles));
  state.addAttribute(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                     ctx.builder.getDenseI32ArrayAttr({static_cast<int32_t>(ctx.operands.size()), 1,
                                                       static_cast<int32_t>(node.tmp_buffers.size())}));
  return success();
}

LogicalResult verifyBufferTransferElementTypes(const NodeBuildContext &ctx, Value destination) {
  const NodeInfo &node = ctx.node;
  // **搬运节点的两端 dtype 不一致，这里要干净拒收，不能把球踢给 verifier。**
  //
  // `verifyTransferIndexingMaps`（`dialect/AFIR/IR/Ops/Data.cpp`）本来就会查
  // `sourceType.getElementType() != destinationType.getElementType()`，但那是**硬失败**
  // ——整条编译红掉。这条不变量在今天的几张 legacy 调度图上真会被违反：legacy 自己对某些
  // 节点做「升精度算、按原 dtype 存」（`optimize/graph_completeness/dtype_consistency.cpp`
  // 的 `GetConversionDtype`，例如 `Fma` 的 `bf16->f32`），存的时候插的 Cast 落回的是节点
  // 自己声明的 dtype，而 GM 侧的函数签名可能是另一个——两者今天没有保证一致。
  //
  // 实测过两次：`round_to_int` / `trunc_to_int`（2b）与 `afir.fma` 接 bf16，都在这一条上
  // 从「干净回落」变成「硬失败」。真正给「两端 dtype 不同但字节宽度相同」的搬运定语义
  // （capability 层拒收还是显式建模成 reinterpret）是独立一件设计活，按字节宽度悄悄放开
  // 会把硬失败换成静默算错，更坏——这里只做风险最低的那一半：**提前问一次，答不支持就
  // 回落，不让它绕到 verifier 变成红**。
  if (isTransferNodeType(node.type)) {
    auto sourceMemrefType = ctx.operands.empty() ? MemRefType() : dyn_cast<MemRefType>(ctx.operands.front().getType());
    auto destinationMemrefType = dyn_cast<MemRefType>(destination.getType());
    if (sourceMemrefType && destinationMemrefType &&
        sourceMemrefType.getElementType() != destinationMemrefType.getElementType()) {
      emitCapabilityShortfall(ctx.loc) << "UnsupportedOperation: " << node.type << " node '" << node.name
                                       << "' transfers between " << sourceMemrefType.getElementType() << " and "
                                       << destinationMemrefType.getElementType()
                                       << "; the register-form lowering does not model a cross-dtype transfer yet";
      return failure();
    }
  }
  return success();
}

LogicalResult attachBufferDestinationDescriptor(const NodeBuildContext &ctx, Value destination) {
  if (destination != ctx.externalDestination) {
    FailureOr<afir::BufferDescAttr> descriptor = makeBufferDescAttr(ctx.context(), ctx.resultValue.buffer);
    if (failed(descriptor)) {
      return failure();
    }
    afir::setBufferDescAttr(destination.getDefiningOp(), *descriptor);
  }
  return success();
}

Value buildBufferNodeOp(const NodeBuildContext &ctx, ShapedType resultTy) {
  const NodeInfo &node = ctx.node;
  auto destinationType = dyn_cast<MemRefType>(resultTy);
  if (!destinationType) {
    emitError(ctx.loc) << "AfirBuilder: buffer-form result must be a memref";
    return {};
  }
  FailureOr<Value> destination = getBufferNodeDestination(ctx, destinationType);
  if (failed(destination)) {
    return {};
  }
  if (failed(verifyBufferTransferElementTypes(ctx, *destination))) {
    return {};
  }
  if (failed(attachBufferDestinationDescriptor(ctx, *destination))) {
    return {};
  }
  SmallVector<NamedAttribute> attrs;
  if (failed(appendBufferNodeAttrs(ctx, attrs))) {
    return {};
  }
  if (failed(appendBufferNodeIrAttrs(ctx, attrs))) {
    return {};
  }
  StringRef operationName = afirOperationNameFor(node.type);
  if (operationName.empty()) {
    emitCapabilityShortfall(ctx.loc) << "UnsupportedOperation: no AFIR operation is registered for node '" << node.type
                                     << "'";
    return {};
  }
  OperationState state(ctx.loc, operationName);
  state.addOperands(ctx.operands);
  if (isReduceNodeType(node.type)) {
    if (failed(appendReduceScratchOperands(ctx, *destination, state))) {
      return {};
    }
  } else {
    state.addOperands(*destination);
  }
  // At most one: an operation needing more than one tmp buffer region would need
  // the tmp operand to become variadic, and none does today.
  if (!isReduceNodeType(node.type) && node.tmp_buffers.size() > 1) {
    emitError(ctx.loc) << "UnsupportedOperation: node '" << node.name << "' has " << node.tmp_buffers.size()
                       << " tmp buffers; buffer form supports at most one";
    return {};
  }
  bool hasTmpBuffer = false;
  if (!isReduceNodeType(node.type) && !node.tmp_buffers.empty()) {
    // 能不能带 tmp buffer 由 ODS base 决定，而 base 决定的同一件事也决定了要不要
    // `operandSegmentSizes`。调度分了 tmp buffer 却落到一个装不下它的 Op 上，说明
    // 这个 op 的 ODS 形态和调度的假设对不上——静默丢掉会变成一次越界写。
    if (!needsOperandSegmentSizes(state.name)) {
      emitError(ctx.loc) << "UnsupportedOperation: node '" << node.name << "' was given a tmp buffer but '"
                         << operationName << "' has no tmp buffer operand";
      return {};
    }
    FailureOr<Value> tmpBuffer = makeTmpBufferAllocation(ctx, node.tmp_buffers.front());
    if (failed(tmpBuffer)) {
      return {};
    }
    state.addOperands(*tmpBuffer);
    hasTmpBuffer = true;
  }
  if (node.type != "Store") {
    state.addTypes(resultTy);
  }
  if (!isReduceNodeType(node.type) && needsOperandSegmentSizes(state.name)) {
    state.addAttribute(
        OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
        makeOperandSegmentSizes(ctx.builder, ctx.operands.size(), /*hasDestination=*/true, hasTmpBuffer));
  }
  state.addAttributes(attrs);
  Operation *op = ctx.builder.create(state);
  return node.type == "Store" ? *destination : op->getResult(0);
}

Value createLogicalNodeOp(const NodeBuildContext &nodeCtx, ShapedType resultTy, ArrayRef<NamedAttribute> attrs,
                          ArrayAttr resultDescs, afir::BufferDescAttr resultBufferDesc) {
  MLIRContext *ctx = nodeCtx.context();
  StringRef type = nodeCtx.node.type;
  ArrayRef<Value> operands = nodeCtx.operands;
  auto make = [&](auto tag) -> Value {
    using OpTy = decltype(tag);
    Operation *op = nodeCtx.builder.create<OpTy>(nodeCtx.loc, TypeRange{resultTy}, ValueRange(operands), attrs);
    if (needsOperandSegmentSizes(op->getName())) {
      op->setAttr(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                  makeOperandSegmentSizes(nodeCtx.builder, operands.size(), /*hasDestination=*/false));
    }
    // Store is an optional-result operation in logical tensor form. Preserve the verifier-required descriptor shape.
    if (op->getNumResults() == 0) {
      op->setAttr(afir::kResultDescsAttrName, ArrayAttr::get(ctx, {}));
      return operands.empty() ? Value() : operands.front();
    }
    op->setAttr(afir::kResultDescsAttrName, resultDescs);
    // tensor 形态没有 destination，Op 的结果就是这块存储的定义点——**除非它在
    // `afir.vector_func` 的 region 里**：那里的值是寄存器，没有存储可描述。
    if (!nodeCtx.insideVectorFunc) {
      afir::setBufferDescAttr(op, resultBufferDesc);
    }
    return op->getResult(0);
  };
#define AFIR_BUILD_OPERATION(SCHEDULER_NAME, OP, FORM)            \
  if (type == SCHEDULER_NAME && (FORM) != BridgeOpForm::Buffer) { \
    return make(OP{});                                            \
  }
  AFIR_BRIDGE_OPERATION_REGISTRY(AFIR_BUILD_OPERATION)
#undef AFIR_BUILD_OPERATION
  emitError(nodeCtx.loc) << "AfirBuilder: unsupported node type '" << type << "'";
  return {};
}

LogicalResult appendLogicalTransferAttrs(OpBuilder &b, const NodeInfo &node, const TensorValueDTO &resultValue,
                                         SmallVectorImpl<NamedAttribute> &attrs) {
  if (!isTransferNodeType(node.type) || node.inputs.size() != 1) {
    return success();
  }
  FailureOr<ArrayAttr> indexingMaps =
      makeTransferIndexingMapsAttr(b.getContext(), node.inputs.front().tensor, resultValue.tensor);
  if (failed(indexingMaps)) {
    return failure();
  }
  attrs.emplace_back(StringAttr::get(b.getContext(), afir::kIndexingMapsAttrName), *indexingMaps);
  return success();
}

Value buildLogicalNodeOp(const NodeBuildContext &nodeCtx, ShapedType resultTy) {
  const NodeInfo &node = nodeCtx.node;
  MLIRContext *ctx = nodeCtx.context();
  // tensor 形态的结果描述取自节点自身的 outputs，而不是 `nodeCtx.resultValue`
  // （后者可能已被 Output 节点改写）。保持原有取值口径。
  const TensorValueDTO &resultValue = node.outputs.empty() ? node.output : node.outputs.front();
  SmallVector<NamedAttribute> attrs;
  FailureOr<ArrayAttr> operandDescs = makeTensorDescArrayAttr(ctx, node.inputs);
  SmallVector<TensorValueDTO, 1> resultValues{resultValue};
  FailureOr<ArrayAttr> resultDescs = makeTensorDescArrayAttr(ctx, resultValues);
  FailureOr<afir::BufferDescAttr> resultBufferDesc = makeBufferDescAttr(ctx, resultValue.buffer);
  if (failed(operandDescs) || failed(resultDescs) || failed(resultBufferDesc)) {
    return {};
  }
  attrs.emplace_back(StringAttr::get(ctx, afir::kOperandDescsAttrName), *operandDescs);

  if (failed(appendLogicalTransferAttrs(nodeCtx.builder, node, resultValue, attrs))) {
    return {};
  }

  if (failed(appendScheduleAttrs(nodeCtx.builder, nodeCtx.loc, node, resultValue, attrs))) {
    return {};
  }

  return createLogicalNodeOp(nodeCtx, resultTy, attrs, *resultDescs, *resultBufferDesc);
}

// Create one AFIR node, returning its value (null on error).
Value buildNodeOp(const NodeBuildContext &ctx, ShapedType resultTy, bool bufferForm) {
  if (bufferForm && !isFormSupported(ctx.node.type, BridgeOpForm::Buffer)) {
    emitCapabilityShortfall(ctx.loc) << "UnsupportedOperation: buffer-form bridge does not support node '"
                                     << ctx.node.type << "'; supported operations are "
                                     << listSupportedOperations(BridgeOpForm::Buffer);
    return {};
  }

  if (!ctx.node.outputs.empty() && ctx.node.outputs.size() != 1) {
    emitError(ctx.loc) << "UnsupportedOperation: node '" << ctx.node.name
                       << "' has multiple results but its AFIR operation is not multi-result capable";
    return {};
  }
  if (bufferForm) {
    return buildBufferNodeOp(ctx, resultTy);
  }
  return buildLogicalNodeOp(ctx, resultTy);
}

LogicalResult appendTensorTypes(MLIRContext *ctx, ArrayRef<IoTensor> tensors, SmallVectorImpl<Type> &types,
                                bool bufferForm) {
  for (const auto &tensor : tensors) {
    // 按值标量形参（`ScalarData`）的类型就是元素类型本身，两种形态下都一样：它不是一块
    // 存储，没有 memref 可建，也没有形状可给 tensor 类型。
    Type type = tensor.by_value ? parseDtypeImpl(ctx, tensor.info.tensor.dtype)
                : bufferForm    ? Type(makeBufferType(ctx, tensor.info))
                                : Type(makeTensorType(ctx, tensor.info.tensor));
    if (!type) {
      return failure();
    }
    types.push_back(type);
  }
  return success();
}

// 将 workspace 散写目的折成单核视图：repeat 置 1，保留目的 stride 供下降计算块偏移。
// 源、目的轴可能经过不同合轴，必须按 axis ID 匹配；源独有轴只允许长度为 1。
FailureOr<TensorValueDTO> foldBlockScatteredDestination(const NodeInfo &store, Location loc) {
  if (store.inputs.size() != 1) {
    return emitError(loc) << "InvalidOperation: Store '" << store.name << "' writing workspace requires one input",
           failure();
  }
  const TensorValueDTO &source = store.inputs.front();
  TensorValueDTO folded = store.output;
  if (source.tensor.axis.size() != source.tensor.repeats.size() ||
      folded.tensor.axis.size() != folded.tensor.repeats.size()) {
    return emitError(loc) << "InvalidDescriptor: Store '" << store.name
                          << "' has a descriptor whose axis and repeat lists differ in length",
           failure();
  }

  llvm::DenseMap<int64_t, StringRef> sourceRepeats;
  for (auto [index, axis] : llvm::enumerate(source.tensor.axis)) {
    sourceRepeats.try_emplace(axis, source.tensor.repeats[index]);
  }

  llvm::DenseSet<int64_t> destinationAxes(folded.tensor.axis.begin(), folded.tensor.axis.end());
  // 源独有轴必须退化，避免折叠时丢失有效元素。
  for (auto [index, axis] : llvm::enumerate(source.tensor.axis)) {
    if (destinationAxes.contains(axis) || proveExpressionsEqual(source.tensor.repeats[index], "1")) {
      continue;
    }
    return emitError(loc) << "InvalidDescriptor: Store '" << store.name << "' source axis " << axis << " has repeat '"
                          << source.tensor.repeats[index] << "' but does not appear in the workspace destination",
           failure();
  }

  for (auto [index, axis] : llvm::enumerate(folded.tensor.axis)) {
    const auto sourceRepeat = sourceRepeats.find(axis);
    if (sourceRepeat != sourceRepeats.end() &&
        proveExpressionsEqual(sourceRepeat->second, folded.tensor.repeats[index])) {
      continue;
    }
    // 仅允许目的独有轴或源 repeat 为 1 的轴折叠，其他长度冲突报错。
    if (sourceRepeat != sourceRepeats.end() && !proveExpressionsEqual(sourceRepeat->second, "1")) {
      return emitError(loc) << "InvalidDescriptor: Store '" << store.name << "' writes workspace with repeat '"
                            << sourceRepeat->second << "' on axis " << axis << " but the destination declares '"
                            << folded.tensor.repeats[index] << "'",
             failure();
    }
    folded.tensor.repeats[index] = "1";
    folded.tensor.shape[index] = 1;
    folded.tensor.shape_exprs[index] = "1";
  }
  return folded;
}

// 借用 Output 描述子，并持有为 workspace Store 新建的折叠描述子。
struct LogicalOverrides {
  llvm::StringMap<const TensorValueDTO *> byNode;
  std::vector<std::unique_ptr<TensorValueDTO>> owned;
};

FailureOr<LogicalOverrides> makeLogicalOverrides(const TemplateInfo &tmpl, Location loc) {
  LogicalOverrides overrides;
  for (const auto &node : tmpl.nodes) {
    if (node.type == "Output" && node.input_names.size() == 1) {
      overrides.byNode[node.input_names[0]] = &node.output;
      continue;
    }
    if (node.type != "Workspace" || node.input_names.size() != 1) {
      continue;
    }
    for (const NodeInfo &producer : tmpl.nodes) {
      if (producer.type != "Store" || !llvm::is_contained(producer.output_names, node.input_names.front())) {
        continue;
      }
      FailureOr<TensorValueDTO> folded = foldBlockScatteredDestination(producer, loc);
      if (failed(folded)) {
        return failure();
      }
      overrides.owned.push_back(std::make_unique<TensorValueDTO>(std::move(*folded)));
      overrides.byNode[producer.name] = overrides.owned.back().get();
    }
  }
  return overrides;
}

// 建一个 template 的节点序列时一路带着的状态：template 的形态、Output 节点与 workspace 汇
// 对结果描述的改写表、外部输出落点，以及名字到 SSA 值的映射。前三项只读，valueMap 边建边长。
struct TemplateBuildState {
  bool bufferForm;
  // region 内建的节点要少挂一样东西，见 `NodeBuildContext::insideVectorFunc`。
  bool insideVectorFunc = false;
  llvm::StringMap<const TensorValueDTO *> logicalOverrides;
  const llvm::StringMap<Value> &externalDestinations;
  llvm::StringMap<Value> &valueMap;
};

// 把 `Scalar` 节点的字面值解析成一个有类型的常量。
//
// legacy 是把那串文本直接拼进生成的 C++（`codegen_kernel.cpp` 的 `IsOps<Scalar>` 分支），
// 于是「2.0」「998.998f」「某个宏」全都能过，错了要到设备编译器上才报。这里不走那条
// 路：AFIR 里它必须是一个有类型的 SSA 值。
//
// 两条都要做到，而且**不能靠 APInt / APFloat 的默认行为**：
//
//   - 解析不了的写法当场报出来，不静默产出一个别的值；
//   - 能解析但**放不进目标类型**的也要报，不截断、不溢出成 Inf。
//
// 反面教材是 `StringRef::getAsInteger(unsigned, APInt&)`：它是无符号重载，返回的位宽
// 是「按字符串估出来的最小值」——"8" 给回 4 位的 `0b1000`，再 sext 到 i32 就成了 -8，
// 而负号它根本不收。走那条路的话 `.Scalar("k", "8", DT_INT32)` 会静默变成 -8，
// `.Scalar("k", "-1", DT_INT32)`（仓库里的现存写法）会被判成非法。
// `Scalar` 与常量形态的 `IndexExpr` 共用：两者到这一步都只剩「一串十进制文本 + 一个
// dtype」，解析规则与诊断没有任何差别，`what` 只用来说清是哪一类节点。
Value buildTypedConstant(OpBuilder &builder, Location loc, const NodeInfo &node, StringRef what, StringRef valueText) {
  Type elementType = parseDtypeImpl(builder.getContext(), node.output.tensor.dtype);
  if (!elementType) {
    emitCapabilityShortfall(loc) << "UnsupportedOperation: " << what << " node '" << node.name
                                 << "' has unsupported dtype '" << node.output.tensor.dtype << "'";
    return {};
  }
  llvm::StringRef text = valueText.trim();
  if (text.empty()) {
    emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' has an empty value";
    return {};
  }
  // 字面量必须以数字、正负号或小数点开头。挡的是 `inf` / `nan` / 宏名这类
  // `APFloat::convertFromString` 会收下、而 legacy 拼进 C++ 后编译不过的写法。
  if (!llvm::isDigit(text.front()) && text.front() != '-' && text.front() != '+' && text.front() != '.') {
    emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' value '" << text
                   << "' is not a numeric literal";
    return {};
  }
  text.consume_front("+");

  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    // legacy 的字面量里带 `f` 后缀是常见写法（`998.998f`），它拼进 C++ 就是合法的。
    llvm::StringRef digits = text;
    if (digits.ends_with("f") || digits.ends_with("F")) {
      digits = digits.drop_back();
    }
    // 直接按**目标**语义解析，不经过 double：先转 double 再转 f16 是两次舍入，边界值
    // 与一次到位的结果不同。
    llvm::APFloat value(floatType.getFloatSemantics());
    llvm::Expected<llvm::APFloat::opStatus> status =
        value.convertFromString(digits, llvm::APFloat::rmNearestTiesToEven);
    if (!status) {
      llvm::consumeError(status.takeError());
      emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' value '" << text
                     << "' is not a floating-point literal";
      return {};
    }
    // 溢出要报，不能悄悄变成 ±Inf——那是把一个具体的值换成了另一个值。舍入（Inexact）
    // 是正常的，浮点字面量本来就大多不能精确表示。
    if ((*status & llvm::APFloat::opOverflow) != 0) {
      emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' value '" << text
                     << "' does not fit in " << node.output.tensor.dtype;
      return {};
    }
    return builder.create<arith::ConstantOp>(loc, floatType, builder.getFloatAttr(floatType, value));
  }

  auto integerType = dyn_cast<IntegerType>(elementType);
  if (!integerType) {
    emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' has non-numeric dtype '"
                   << node.output.tensor.dtype << "'";
    return {};
  }
  // `arith.constant` 只收无符号性中立（signless）的整数。`parseDtype` 对 `uintN` 给的是
  // `ui8` 这类带符号性的类型，到 `arith.constant` 那里会以一句与 Scalar 无关的
  // verifier 报错收场——在这里说清楚是哪个节点。
  if (!integerType.isSignless()) {
    emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' dtype '"
                   << node.output.tensor.dtype << "' is not supported as a scalar operand";
    return {};
  }
  int64_t value = 0;
  if (text.getAsInteger(/*Radix=*/10, value)) {
    emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' value '" << text
                   << "' is not an integer literal";
    return {};
  }
  const unsigned width = integerType.getWidth();
  if (width < 64U && !llvm::APInt(64, static_cast<uint64_t>(value), /*isSigned=*/true).isSignedIntN(width)) {
    emitError(loc) << "UnsupportedOperation: " << what << " node '" << node.name << "' value '" << text
                   << "' does not fit in " << node.output.tensor.dtype;
    return {};
  }
  return builder.create<arith::ConstantOp>(loc, integerType, builder.getIntegerAttr(integerType, value));
}

// `IndexExpr` 的值。两条路，由表达式里有没有自由符号决定：
//
//   - **常量**（`"2"`）：直接建一个有类型的常量，与 `Scalar` 完全同路。
//   - **带自由符号**（`"(2 * s0)"`）：符号逐个取成 `afir.tiling_param`、表达式本身成为
//     一个 `afir.symbolic_eval`，两者都落在入口块的尺寸序言里——那一段支配函数体里的
//     任何消费者，且循环生成把它整体当序言看待、不进循环树。这正是 legacy 的落点：
//     `AppendConstTensorInit` 跑在 `root_loop.Generate` 之前。
//
// 第二条路算出来的是 `index`，而这个值要当元素类型的标量用，所以还要一次转换。legacy
// 那边是 `constexpr <dtype> k = <整数表达式>;`，靠 C++ 的隐式转换；这里显式写出来：
// 整数元素类型一次 `index_cast`，浮点再跟一次 `sitofp`。
Value buildIndexExprValue(OpBuilder &builder, Location loc, const NodeInfo &node, StringRef expression) {
  int64_t literal = 0;
  if (!expression.getAsInteger(10, literal)) {
    return buildTypedConstant(builder, loc, node, "IndexExpr", expression);
  }

  Type elementType = parseDtypeImpl(builder.getContext(), node.output.tensor.dtype);
  if (!elementType) {
    emitCapabilityShortfall(loc) << "UnsupportedOperation: IndexExpr node '" << node.name << "' has unsupported dtype '"
                                 << node.output.tensor.dtype << "'";
    return {};
  }
  Block *entry = builder.getInsertionBlock();
  if (entry == nullptr) {
    emitError(loc) << "AfirBuilder: IndexExpr node '" << node.name << "' has no insertion block";
    return {};
  }
  FailureOr<afir::ExprAttr> expressionAttr = afir::makeExprAttr(builder.getContext(), expression);
  if (failed(expressionAttr)) {
    return {};
  }
  FailureOr<Value> size = afir::materializeRuntimeSize(*entry, loc, *expressionAttr);
  if (failed(size)) {
    return {};
  }
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    return builder.create<arith::IndexCastOp>(loc, integerType, *size).getResult();
  }
  auto floatType = dyn_cast<FloatType>(elementType);
  if (!floatType) {
    emitError(loc) << "UnsupportedOperation: IndexExpr node '" << node.name << "' dtype '" << node.output.tensor.dtype
                   << "' is not a numeric type";
    return {};
  }
  // 先到 i64 再到浮点：`sitofp` 不收 `index`。宽度取 64 位而不是 32，免得表达式的值超出
  // i32 时静默截断——截断的后果是算出另一个数，不是报错。
  Value integer = builder.create<arith::IndexCastOp>(loc, builder.getIntegerType(64), *size).getResult();
  return builder.create<arith::SIToFPOp>(loc, floatType, integer).getResult();
}

FailureOr<bool> handleMarkerNode(OpBuilder &builder, const NodeInfo &node, TemplateBuildState &state, Location loc) {
  llvm::StringMap<Value> &valueMap = state.valueMap;
  // `Scalar` 不产生 op，只产生一个值——和 `Data` / `Output` 同类。
  if (node.type == "Scalar") {
    // 标量操作数只有 buffer 形态有。tensor 形态的 verifier（`verifyLogicalForm`）要求
    // 所有操作数都是 `TensorType`，放过去的话报出来的是「expected tensor operands and
    // result」——一句跟标量无关、也指不到这个节点的话。
    if (!state.bufferForm) {
      emitError(loc) << "UnsupportedOperation: Scalar node '" << node.name << "' is only supported in the buffer form";
      return failure();
    }
    if (!node.scalar_value) {
      emitError(loc) << "AfirBuilder: Scalar node '" << node.name << "' carries no value";
      return failure();
    }
    Value constant = buildTypedConstant(builder, loc, node, "Scalar", *node.scalar_value);
    if (!constant) {
      return failure();
    }
    valueMap[node.name] = constant;
    return true;
  }
  // `IndexExpr` 与 `Scalar` 落在同一处：都是「不产生计算、只产生一个值」的标量源，
  // legacy 把两者收进同一张 `kernel.constant_tensors`，在 kernel 顶部、所有循环之前
  // 一起物化（`AppendConstTensorInit` 跑在 `root_loop.Generate` 前面）。**它不需要归纳
  // 变量**——渲染时只把 size var 换成 TilingData 字段，轴根本不在替换表里。
  if (node.type == "IndexExpr") {
    if (!state.bufferForm) {
      emitError(loc) << "UnsupportedOperation: IndexExpr node '" << node.name
                     << "' is only supported in the buffer form";
      return failure();
    }
    if (!node.index_expr) {
      emitError(loc) << "AfirBuilder: IndexExpr node '" << node.name << "' carries no expression";
      return failure();
    }
    llvm::StringRef expression = llvm::StringRef(*node.index_expr).trim();
    Value value = buildIndexExprValue(builder, loc, node, expression);
    if (!value) {
      return failure();
    }
    valueMap[node.name] = value;
    return true;
  }
  // `Data` 与 `ScalarData` 都已经由 `mapTemplateInputs` 按参数位置放进 valueMap，这里
  // 只确认它在。两者到这一步已经没有差别：一个是 memref 块参数，一个是标量块参数，
  // 引用它们的下游节点各按类型走自己那条路。
  if (node.type == "Data" || node.type == "ScalarData") {
    if (!valueMap.count(node.name)) {
      emitError(loc) << "AfirBuilder: " << node.type << " node '" << node.name << "' is not a kernel input";
      return failure();
    }
    return true;
  }
  // workspace 的值在 `mapTemplateWorkspaces` 里已经建好——它当汇时要先于写它的 Store
  // 存在，所以不能等走到这里才建。这里只确认它在。
  if (node.type == "Workspace") {
    if (!valueMap.count(node.name)) {
      emitError(loc) << "AfirBuilder: Workspace node '" << node.name << "' has no materialized storage";
      return failure();
    }
    return true;
  }
  if (node.type != "Output") {
    return false;
  }
  if (node.input_names.size() != 1 || !valueMap.count(node.input_names[0])) {
    emitError(loc) << "AfirBuilder: Output node '" << node.name << "' needs one known input";
    return failure();
  }
  auto externalDestination = state.externalDestinations.find(node.name);
  if (externalDestination != state.externalDestinations.end()) {
    valueMap[node.name] = externalDestination->second;
  } else {
    valueMap[node.name] = valueMap[node.input_names[0]];
  }
  return true;
}

// 前向声明：region 里的节点与父图共用同一条建 op 路径。
LogicalResult buildComputeNode(OpBuilder &builder, Location loc, const NodeInfo &node, TemplateBuildState &state);

//===----------------------------------------------------------------------===//
// 二次融合：`afir.vector_func`
//===----------------------------------------------------------------------===//
//
// 这个节点**不走注册表那条通用路径**（`buildBufferNodeOp` 的 `make(OP{})`）：它要建的是一个
// 带 region 的算子，通用路径给它建出来的是一个没有 region 的空壳，而 `SingleBlockImplicit-
// Terminator` 会在 verify 时才报出来——那时已经看不出是谁建错的。
//
// region 里是 **tensor 形态**（值语义，没有 destination）。这不是写法偏好：VF 的中间结果
// 在寄存器里，一块 UB 都没分配过，而 buffer 形态的每个中间结果都要一个 `outs` 指向的存储。
// 两种形态在 bridge 这一层由 `TemplateBuildState::bufferForm` 分开，所以 region 内只要换一个
// state 就能复用 `buildComputeNode` 的全部逻辑。
LogicalResult buildVectorFuncRegion(OpBuilder &builder, Location loc, const NodeInfo &node, TemplateBuildState &state,
                                    afir::VectorFuncOp vectorFunc, ArrayRef<Value> srcs) {
  const VectorFuncBodyDTO &bodyDto = *node.vector_func_body;
  // 块参数取 src 的形状与元素类型：region 在语义上就是在**整块 tile** 上的一段数据流，
  // lane 那一维是下降的事（`GetVecLen()` 是目标属性，AFIR 这层不认识它）。
  SmallVector<Type> argumentTypes;
  SmallVector<Location> argumentLocs;
  argumentTypes.reserve(srcs.size());
  argumentLocs.reserve(srcs.size());
  for (Value src : srcs) {
    auto srcType = dyn_cast<ShapedType>(src.getType());
    if (!srcType) {
      return emitError(loc) << "AfirBuilder: VectorFunc node '" << node.name << "' src is not a shaped value",
             failure();
    }
    argumentTypes.push_back(RankedTensorType::get(srcType.getShape(), srcType.getElementType()));
    argumentLocs.push_back(loc);
  }

  OpBuilder::InsertionGuard guard(builder);
  Block *bodyBlock = builder.createBlock(&vectorFunc.getBody(), {}, argumentTypes, argumentLocs);
  builder.setInsertionPointToEnd(bodyBlock);

  // region 自己的 valueMap：它与父图**不共享**（算子是 `IsolatedFromAbove`），边界名字
  // 是唯一的入口。
  llvm::StringMap<Value> bodyValueMap;
  for (auto [index, boundaryName] : llvm::enumerate(bodyDto.boundary_input_names)) {
    bodyValueMap[boundaryName] = bodyBlock->getArgument(index);
  }
  TemplateBuildState bodyState{
      /*bufferForm=*/false, /*insideVectorFunc=*/true, {}, state.externalDestinations, bodyValueMap};
  for (const NodeInfo &bodyNode : bodyDto.nodes) {
    if (failed(buildComputeNode(builder, loc, bodyNode, bodyState))) {
      return failure();
    }
  }

  SmallVector<Value> yielded;
  yielded.reserve(bodyDto.boundary_output_names.size());
  for (auto [index, boundaryName] : llvm::enumerate(bodyDto.boundary_output_names)) {
    auto value = bodyValueMap.find(boundaryName);
    if (value == bodyValueMap.end()) {
      return emitError(loc) << "AfirBuilder: VectorFunc node '" << node.name << "' dst #" << index << " names '"
                            << boundaryName << "', which the subgraph does not produce",
             failure();
    }
    yielded.push_back(value->second);
  }
  builder.create<afir::VectorFuncYieldOp>(loc, yielded);

  return success();
}

LogicalResult buildVectorFuncNode(OpBuilder &builder, Location loc, const NodeInfo &node, TemplateBuildState &state) {
  MLIRContext *ctx = builder.getContext();
  if (!node.vector_func_body) {
    return emitError(loc) << "AfirBuilder: VectorFunc node '" << node.name << "' carries no subgraph", failure();
  }
  const VectorFuncBodyDTO &bodyDto = *node.vector_func_body;

  SmallVector<Value> srcs;
  srcs.reserve(node.input_names.size());
  for (const auto &inputName : node.input_names) {
    auto input = state.valueMap.find(inputName);
    if (input == state.valueMap.end()) {
      return emitError(loc) << "AfirBuilder: VectorFunc node '" << node.name << "' references undefined value '"
                            << inputName << "'",
             failure();
    }
    srcs.push_back(input->second);
  }
  if (srcs.size() != bodyDto.boundary_input_names.size()) {
    return emitError(loc) << "AfirBuilder: VectorFunc node '" << node.name << "' has " << srcs.size()
                          << " srcs but the subgraph names " << bodyDto.boundary_input_names.size() << " boundaries",
           failure();
  }

  // 每个输出一块 destination，与普通 buffer 形态节点同一条路（`memref.alloc` + buffer desc）。
  SmallVector<Value> dsts;
  SmallVector<Type> resultTypes;
  dsts.reserve(node.outputs.size());
  resultTypes.reserve(node.outputs.size());
  for (const TensorValueDTO &output : node.outputs) {
    MemRefType destinationType = makeBufferType(ctx, output);
    if (!destinationType) {
      return failure();
    }
    NodeBuildContext destinationCtx{builder, loc, node, srcs, output, Value()};
    FailureOr<Value> destination = makeBufferDestination(destinationCtx, destinationType);
    if (failed(destination)) {
      return failure();
    }
    // 新建的 destination（`memref.alloc`）就是这块存储的定义点，规划结果写在它上面——
    // 与 `buildBufferNodeOp` 同一条路。**不写的失败方式不在这个算子上**：下游
    // `afir.store` 的 verifier 会报「operand 0 的定义点没有 buffer_desc」，指不到这里。
    if (Operation *destinationDef = destination->getDefiningOp()) {
      FailureOr<afir::BufferDescAttr> descriptor = makeBufferDescAttr(ctx, output.buffer);
      if (failed(descriptor)) {
        return failure();
      }
      afir::setBufferDescAttr(destinationDef, *descriptor);
    }
    dsts.push_back(*destination);
    resultTypes.push_back(destinationType);
  }
  if (dsts.size() != bodyDto.boundary_output_names.size()) {
    return emitError(loc) << "AfirBuilder: VectorFunc node '" << node.name << "' has " << dsts.size()
                          << " dsts but the subgraph names " << bodyDto.boundary_output_names.size() << " boundaries",
           failure();
  }

  // 四个可选属性（`AFIR_CommonAttrs` 的 descriptor 三件套加 indexing_maps）在 ODS 生成的
  // 构造里是位置参数，所以这里显式给空——它们随后由 `appendBufferNodeAttrs` 挂上。
  auto vectorFunc = builder.create<afir::VectorFuncOp>(loc, resultTypes, srcs, dsts, ArrayAttr{}, ArrayAttr{},
                                                       ArrayAttr{}, ArrayAttr{});

  // 使用点的 descriptor 与调度属性，与普通 buffer 形态节点走同一条路
  // （`appendBufferNodeAttrs`）。**这不是装饰**：流水同步那一步按「有没有 destination
  // descs」认一个算子是不是 buffer 形态的（`AFIRPipelineSync.cpp` 的
  // `isBufferizedOperation`），认不出来就不会给这一段插 enque / deque——产物在 CPU
  // 模拟器上照样算对，到真机上才是偶发错数据。
  {
    SmallVector<NamedAttribute> attrs;
    // 调度属性那一份按第一个输出取（同一个 VF 的各输出共用一套调度），而 descriptor 要
    // **逐输出各一份**，所以 destination 列表显式传 `node.outputs`。
    NodeBuildContext attrCtx{builder, loc, node, srcs, node.outputs.front(), Value()};
    if (failed(appendBufferNodeAttrs(attrCtx, attrs, node.outputs))) {
      return failure();
    }
    for (const NamedAttribute &attr : attrs) {
      vectorFunc->setAttr(attr.getName(), attr.getValue());
    }
  }

  if (failed(buildVectorFuncRegion(builder, loc, node, state, vectorFunc, srcs))) {
    return failure();
  }

  // 父图侧：下游按输出名引用这些结果。
  for (auto [index, resultName] : llvm::enumerate(node.output_names)) {
    state.valueMap[resultName] = vectorFunc.getResult(index);
  }
  if (node.output_names.empty() && vectorFunc.getNumResults() == 1U) {
    state.valueMap[node.name] = vectorFunc.getResult(0);
  }
  return success();
}

LogicalResult buildComputeNode(OpBuilder &builder, Location loc, const NodeInfo &node, TemplateBuildState &state) {
  const bool bufferForm = state.bufferForm;
  SmallVector<Value> operands;
  for (const auto &inputName : node.input_names) {
    auto input = state.valueMap.find(inputName);
    if (input == state.valueMap.end()) {
      return emitError(loc) << "AfirBuilder: node '" << node.name << "' references undefined value '" << inputName
                            << "'",
             failure();
    }
    operands.push_back(input->second);
  }

  const TensorValueDTO *resultInfo = &node.output;
  if (auto override = state.logicalOverrides.find(node.name); override != state.logicalOverrides.end()) {
    resultInfo = override->second;
  }
  Type resultType = bufferForm ? Type(makeBufferType(builder.getContext(), *resultInfo))
                               : Type(makeTensorType(builder.getContext(), resultInfo->tensor));
  if (!resultType) {
    return failure();
  }
  Value externalDestination;
  if (bufferForm && node.type == "Store") {
    auto destination = state.externalDestinations.find(node.name);
    if (destination != state.externalDestinations.end()) {
      externalDestination = destination->second;
    }
  }
  NodeBuildContext nodeCtx{builder, loc, node, operands, *resultInfo, externalDestination, state.insideVectorFunc};
  Value result = buildNodeOp(nodeCtx, cast<ShapedType>(resultType), bufferForm);
  if (!result) {
    return failure();
  }
  state.valueMap[node.output_names.empty() ? node.name : node.output_names.front()] = result;
  return success();
}

LogicalResult buildTemplateNodes(OpBuilder &builder, Location loc, const TemplateInfo &tmpl,
                                 const llvm::StringMap<const TensorValueDTO *> &overrides,
                                 const llvm::StringMap<Value> &externalDestinations, llvm::StringMap<Value> &valueMap) {
  TemplateBuildState state{tmpl.buffer_form, /*insideVectorFunc=*/false, overrides, externalDestinations, valueMap};
  for (const auto &node : tmpl.nodes) {
    // 二次融合的节点自己带一张子图，所以不走通用的建 op 路径（理由见 `buildVectorFuncNode`）。
    if (node.type == "VectorFunc") {
      if (failed(buildVectorFuncNode(builder, loc, node, state))) {
        return failure();
      }
      continue;
    }
    FailureOr<bool> handled = handleMarkerNode(builder, node, state, loc);
    if (failed(handled)) {
      return failure();
    }
    if (*handled) {
      continue;
    }
    if (failed(buildComputeNode(builder, loc, node, state))) {
      return failure();
    }
  }
  return success();
}

LogicalResult buildReturn(OpBuilder &builder, Location loc, func::FuncOp func, ArrayRef<IoTensor> outputs,
                          const llvm::StringMap<Value> &valueMap) {
  SmallVector<Value> returns;
  for (const auto &output : outputs) {
    auto value = valueMap.find(output.name);
    if (value == valueMap.end()) {
      return emitError(loc) << "AfirBuilder: kernel output '" << output.name << "' was never produced", failure();
    }
    returns.push_back(value->second);
  }
  func::ReturnOp::create(builder, loc, ValueRange(returns));
  // 一个值一份（设计文档 5.1 步 8）。kernel 输出的载体是函数结果位——那是这个值所在
  // 存储的 ABI 事实（GM）。产出它的 Op 上那份来自调度器的节点落点描述（Store 节点的
  // 暂存缓冲，space 是 UB）说的不是同一块存储；buffer 形态本来就丢弃它（destination
  // 是 GM 块参数，没有定义 Op 可挂），tensor 形态这里做同样的事，免得同一个值挂着两份
  // 互相矛盾的 descriptor。
  for (auto [index, returned] : llvm::enumerate(returns)) {
    if (index >= func.getNumResults() || !func.getResultAttr(index, afir::kBufferDescAttrName)) {
      continue;
    }
    if (Operation *definingOp = returned.getDefiningOp()) {
      definingOp->removeAttr(afir::kBufferDescAttrName);
    }
  }
  return success();
}

LogicalResult attachTensorShapeAttrs(func::FuncOp func, const TemplateInfo &tmpl) {
  for (auto [index, input] : llvm::enumerate(tmpl.inputs)) {
    // 按值标量形参没有形状，也就没有可物化的动态维符号。
    if (!input.by_value && !input.info.tensor.shape_exprs.empty()) {
      FailureOr<ArrayAttr> expressions = makeExpressionArrayAttr(func.getContext(), input.info.tensor.shape_exprs);
      if (failed(expressions)) {
        return failure();
      }
      func.setArgAttr(index, afir::kSymbolicShapeAttrName, *expressions);
    }
  }
  for (auto [index, output] : llvm::enumerate(tmpl.outputs)) {
    if (!output.info.tensor.shape_exprs.empty()) {
      FailureOr<ArrayAttr> expressions = makeExpressionArrayAttr(func.getContext(), output.info.tensor.shape_exprs);
      if (failed(expressions)) {
        return failure();
      }
      if (tmpl.buffer_form) {
        func.setArgAttr(tmpl.inputs.size() + index, afir::kSymbolicShapeAttrName, *expressions);
      } else {
        func.setResultAttr(index, afir::kSymbolicShapeAttrName, *expressions);
      }
    }
  }
  return success();
}

// 跨 input 累积的动态维符号状态：表达式去重表（同一表达式只物化一次 SSA 值）、
// 待写入 `dim_symbols` 的条目、以及下一个符号编号。
struct DynamicShapeSymbols {
  llvm::StringMap<Value> expressionValues;
  SmallVector<Attribute> dimSymbols;
  int64_t nextSymbolId = 0;
};

LogicalResult materializeInputDynamicShapeSsa(OpBuilder &builder, func::FuncOp func, const IoTensor &input,
                                              unsigned argIndex, DynamicShapeSymbols &symbols) {
  if (input.info.tensor.shape_exprs.size() != input.info.tensor.shape.size()) {
    return func.emitError() << "input '" << input.name << "' shape expression count "
                            << input.info.tensor.shape_exprs.size() << " does not match rank "
                            << input.info.tensor.shape.size();
  }
  MLIRContext *ctx = builder.getContext();
  Value argument = func.getArgument(argIndex);
  for (auto [dimIndex, dim] : llvm::enumerate(input.info.tensor.shape)) {
    if (!ShapedType::isDynamic(dim)) {
      continue;
    }
    const std::string &expression = input.info.tensor.shape_exprs[dimIndex];
    if (expression.empty()) {
      return func.emitError() << "dynamic dimension " << dimIndex << " of input '" << input.name
                              << "' has no symbolic expression";
    }
    if (symbols.expressionValues.count(expression)) {
      continue;
    }
    Value index = builder.create<arith::ConstantIndexOp>(func.getLoc(), static_cast<int64_t>(dimIndex));
    auto dimOp = builder.create<tensor::DimOp>(func.getLoc(), argument, index);
    FailureOr<afir::ExprAttr> expressionAttr = makeExprAttr(ctx, expression);
    if (failed(expressionAttr)) {
      return failure();
    }
    dimOp->setAttr(afir::kShapeExprAttrName, *expressionAttr);
    symbols.expressionValues[expression] = dimOp.getResult();
    NamedAttrList symbol;
    symbol.append(afir::kDimSymbolIdKey, builder.getI64IntegerAttr(symbols.nextSymbolId++));
    symbol.append(afir::kDimSymbolArgKey, builder.getI64IntegerAttr(static_cast<int64_t>(argIndex)));
    symbol.append(afir::kDimSymbolDimKey, builder.getI64IntegerAttr(static_cast<int64_t>(dimIndex)));
    symbol.append(afir::kDimSymbolExprKey, *expressionAttr);
    symbols.dimSymbols.push_back(DictionaryAttr::get(ctx, symbol));
  }
  return success();
}

LogicalResult materializeDynamicShapeSsa(OpBuilder &builder, func::FuncOp func, const TemplateInfo &tmpl) {
  DynamicShapeSymbols symbols;
  for (auto [argIndex, input] : llvm::enumerate(tmpl.inputs)) {
    if (failed(materializeInputDynamicShapeSsa(builder, func, input, argIndex, symbols))) {
      return failure();
    }
  }

  if (!symbols.dimSymbols.empty()) {
    func->setAttr(afir::kDimSymbolsAttrName, ArrayAttr::get(builder.getContext(), symbols.dimSymbols));
  }
  return success();
}

ArrayAttr makeApiTilingFieldsAttr(OpBuilder &builder, ArrayRef<ApiTilingInfo> apiTilingData) {
  SmallVector<Attribute> apiAttrs;
  apiAttrs.reserve(apiTilingData.size());
  for (const ApiTilingInfo &api : apiTilingData) {
    SmallVector<Attribute> fieldAttrs;
    fieldAttrs.reserve(api.fields.size());
    for (const std::string &field : api.fields) {
      fieldAttrs.push_back(builder.getStringAttr(field));
    }

    NamedAttrList attrs;
    attrs.append(afir::kApiTilingTypeKey, builder.getStringAttr(api.type_name));
    attrs.append(afir::kApiTilingFieldKey, builder.getStringAttr(api.field_name));
    attrs.append(afir::kApiTilingFieldsKey, builder.getArrayAttr(fieldAttrs));
    apiAttrs.push_back(builder.getDictionaryAttr(attrs));
  }
  return builder.getArrayAttr(apiAttrs);
}

LogicalResult appendTemplateValueDescriptors(func::FuncOp func, OpBuilder &builder, const TemplateInfo &tmpl) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<TensorValueDTO, 6> argValues;
  argValues.reserve(tmpl.inputs.size() + (tmpl.buffer_form ? tmpl.outputs.size() : 0));
  for (const auto &input : tmpl.inputs) {
    argValues.push_back(input.info);
  }
  // 与 `argValues` 按下标对齐：按值标量形参（`ScalarData`）没有存储，因而既不挂
  // `afir.buffer_desc`，`afir.arg_descs` 里那一格也只是一份退化的视图。
  SmallVector<bool, 6> argByValue;
  argByValue.reserve(argValues.size());
  for (const auto &input : tmpl.inputs) {
    argByValue.push_back(input.by_value);
  }
  if (tmpl.buffer_form) {
    for (const auto &output : tmpl.outputs) {
      argValues.push_back(output.info);
      argByValue.push_back(false);
    }
  }
  SmallVector<TensorValueDTO, 2> resultValues;
  if (!tmpl.buffer_form) {
    resultValues.reserve(tmpl.outputs.size());
    for (const auto &output : tmpl.outputs) {
      resultValues.push_back(output.info);
    }
  }
  FailureOr<ArrayAttr> argDescs = makeTensorDescArrayAttr(ctx, argValues);
  FailureOr<ArrayAttr> resultDescs = makeTensorDescArrayAttr(ctx, resultValues);
  if (failed(argDescs) || failed(resultDescs)) {
    return failure();
  }
  // 标量形参那一格换成退化视图——无轴、无 repeats、无 strides，只留角色。AscGraph 上
  // `ScalarData` 的输出确实带着一串全 1 的 repeats，但那是「它在哪个循环里被引用」的
  // 说法，不是这个**值**的形状；照抄过来就成了一句没人读、又像真的的假话。形状留空
  // 与 `recordTmpBuffers` 给 tmp buffer 合成退化视图是同一笔。
  SmallVector<Attribute> argDescAttrs(argDescs->begin(), argDescs->end());
  for (auto [index, byValue] : llvm::enumerate(argByValue)) {
    if (byValue) {
      argDescAttrs[index] = afir::TensorDescAttr::get(ctx, /*axis=*/{}, /*repeats=*/{}, /*strides=*/{},
                                                      /*vectorized_axis=*/{}, /*vectorized_strides=*/{},
                                                      afir::RoleAttr::get(ctx, afir::Role::Input));
    }
  }
  func->setAttr(afir::kArgDescsAttrName, ArrayAttr::get(ctx, argDescAttrs));
  // buffer descriptor 是值的事实，挂在值的定义点：kernel 参数的定义点就是它的
  // arg attr，函数结果同理（设计文档 5.1 步 8）。此前它是一个与参数按下标对齐的
  // 数组属性，和 `arg_descs` 各存一份长度。
  for (auto [index, value] : llvm::enumerate(argValues)) {
    if (argByValue[index]) {
      continue;
    }
    FailureOr<afir::BufferDescAttr> descriptor = makeBufferDescAttr(ctx, value.buffer);
    if (failed(descriptor)) {
      return failure();
    }
    func.setArgAttr(index, afir::kBufferDescAttrName, *descriptor);
  }
  if (!tmpl.buffer_form) {
    func->setAttr(afir::kResultDescsAttrName, *resultDescs);
    for (auto [index, value] : llvm::enumerate(resultValues)) {
      FailureOr<afir::BufferDescAttr> descriptor = makeBufferDescAttr(ctx, value.buffer);
      if (failed(descriptor)) {
        return failure();
      }
      func.setResultAttr(index, afir::kBufferDescAttrName, *descriptor);
    }
  }
  return success();
}

LogicalResult appendTemplateMetadataAttrs(func::FuncOp func, OpBuilder &builder, const TemplateInfo &tmpl) {
  llvm::StringSet<> argumentKeys;
  for (auto [idx, input] : llvm::enumerate(tmpl.inputs)) {
    std::string key = "input:" + std::to_string(input.index);
    if (input.index < 0 || !argumentKeys.insert(key).second) {
      return func.emitError("invalid or duplicate input index");
    }
    mlir::afir::setKernelArgAttr(
        func, idx, input.by_value ? mlir::afir::KernelArgRole::ScalarInput : mlir::afir::KernelArgRole::Input,
        input.index);
  }
  for (auto [idx, output] : llvm::enumerate(tmpl.outputs)) {
    std::string key = "output:" + std::to_string(output.index);
    if (output.index < 0 || !argumentKeys.insert(key).second) {
      return func.emitError("invalid or duplicate output index");
    }
    if (tmpl.buffer_form) {
      unsigned argumentIndex = tmpl.inputs.size() + idx;
      mlir::afir::setKernelArgAttr(func, argumentIndex, mlir::afir::KernelArgRole::Output, output.index);
    } else {
      // tensor 形态下输出还是函数结果，等 `AFIRKernelBodySignature` 把它们追加成参数时
      // 再随 result attr 一起搬过去。
      func.setResultAttr(
          idx, mlir::afir::kKernelArgAttrName,
          mlir::afir::KernelArgAttr::get(builder.getContext(), mlir::afir::KernelArgRole::Output, output.index));
    }
  }
  func->setAttr(mlir::afir::kOriginalGraphNameAttrName, builder.getStringAttr(tmpl.original_name));
  // -1 是 TemplateInfo 里「未设置」的哨兵。迁移前它会原样写成 afir.stage_id = -1 /
  // afir.template_id = -1，再由每个读点各自判一次非负把它挡掉。`#afplan.variant` 表达
  // 不了「未设置」——未设置就不挂属性，下游取不到变体坐标，与此前判负失败等价。
  if (tmpl.stage_id >= 0 && tmpl.template_id >= 0) {
    func->setAttr(mlir::afplan::kVariantAttrName,
                  mlir::afplan::VariantAttr::get(builder.getContext(), tmpl.stage_id, tmpl.template_id));
  }
  func->setAttr(mlir::afir::kDoubleBufferAttrName, builder.getBoolAttr(tmpl.double_buffer));
  if (!tmpl.queue_ids.empty()) {
    func->setAttr(mlir::afir::kQueueIdsAttrName, builder.getDenseI64ArrayAttr(tmpl.queue_ids));
  }
  if (!tmpl.buffer_ids.empty()) {
    func->setAttr(mlir::afir::kBufferIdsAttrName, builder.getDenseI64ArrayAttr(tmpl.buffer_ids));
  }
  if (!tmpl.api_tiling_data.empty()) {
    func->setAttr(mlir::afir::kApiTilingFieldsAttrName, makeApiTilingFieldsAttr(builder, tmpl.api_tiling_data));
  }
  if (!tmpl.score_func.empty()) {
    func->setAttr(mlir::afplan::kScoreFuncAttrName, builder.getStringAttr(tmpl.score_func));
  }
  StringRef workspaceExpression = tmpl.workspace_size.empty() ? StringRef("0") : StringRef(tmpl.workspace_size);
  FailureOr<afir::ExprAttr> workspaceSize = makeExprAttr(builder.getContext(), workspaceExpression);
  if (failed(workspaceSize)) {
    return failure();
  }
  func->setAttr(mlir::afir::kWorkspaceSizeAttrName, *workspaceSize);
  return success();
}

LogicalResult appendTemplateDescriptors(func::FuncOp func, OpBuilder &builder, const TemplateInfo &tmpl,
                                        ArrayRef<BufferDescDTO> workspacePlan) {
  MLIRContext *ctx = builder.getContext();
  FailureOr<afir::AscGraphAttrGroupsAttr> graphAttr = makeTemplateAttr(ctx, tmpl);
  if (failed(graphAttr)) {
    return failure();
  }
  func->setAttr(afir::kAscGraphAttrName, *graphAttr);
  if (!workspacePlan.empty()) {
    FailureOr<ArrayAttr> workspacePlanAttr = makeWorkspacePlanAttr(ctx, workspacePlan);
    if (failed(workspacePlanAttr)) {
      return failure();
    }
    func->setAttr(afir::kWorkspacePlanAttrName, *workspacePlanAttr);
  }
  if (failed(appendTemplateValueDescriptors(func, builder, tmpl))) {
    return failure();
  }
  if (failed(appendTemplateMetadataAttrs(func, builder, tmpl))) {
    return failure();
  }
  if (failed(attachTensorShapeAttrs(func, tmpl))) {
    return failure();
  }
  return success();
}

void mapTemplateInputs(Block *entry, const TemplateInfo &tmpl, llvm::StringMap<Value> &valueMap) {
  for (auto [idx, input] : llvm::enumerate(tmpl.inputs)) {
    valueMap[input.name] = entry->getArgument(idx);
  }
}

void mapTemplateOutputs(Block *entry, const TemplateInfo &tmpl, llvm::StringMap<Value> &valueMap,
                        llvm::StringMap<Value> &externalDestinations) {
  if (!tmpl.buffer_form) {
    return;
  }
  for (auto [index, output] : llvm::enumerate(tmpl.outputs)) {
    Value destination = entry->getArgument(tmpl.inputs.size() + index);
    externalDestinations[output.name] = destination;
    valueMap[output.name] = destination;
    for (const auto &node : tmpl.nodes) {
      if (node.type == "Output" && node.name == output.name && node.input_names.size() == 1) {
        externalDestinations[node.input_names.front()] = destination;
      }
    }
  }
}

// workspace 作源时取自身 GM 视图，作汇时优先取 Store 的折叠视图。
// memref 描述访问视图，实际分配容量由 workspace_plan 管理。
const TensorValueDTO *workspaceValueInfo(const TemplateInfo &tmpl, const NodeInfo &node,
                                         const llvm::StringMap<const TensorValueDTO *> &overrides) {
  // 作汇时自身描述可能覆盖整块 GM，必须优先使用生产者视图。
  if (node.input_names.size() == 1) {
    for (const NodeInfo &producer : tmpl.nodes) {
      if (!llvm::is_contained(producer.output_names, node.input_names.front())) {
        continue;
      }
      auto folded = overrides.find(producer.name);
      return folded == overrides.end() ? &producer.output : folded->second;
    }
  }
  if (!node.output.tensor.shape.empty()) {
    return &node.output;
  }
  return nullptr;
}

LogicalResult mapTemplateWorkspaces(OpBuilder &builder, Location loc, const TemplateInfo &tmpl,
                                    const llvm::StringMap<const TensorValueDTO *> &overrides,
                                    llvm::StringMap<Value> &valueMap, llvm::StringMap<Value> &externalDestinations) {
  if (!tmpl.buffer_form) {
    return success();
  }
  for (const NodeInfo &node : tmpl.nodes) {
    if (node.type != "Workspace") {
      continue;
    }
    const TensorValueDTO *info = workspaceValueInfo(tmpl, node, overrides);
    if (!info) {
      return emitError(loc) << "InvalidWorkspace: Workspace node '" << node.name
                            << "' has neither its own shape nor a single producer to take it from";
    }
    auto type = dyn_cast_or_null<MemRefType>(Type(makeBufferType(builder.getContext(), *info)));
    if (!type) {
      return emitError(loc) << "InvalidWorkspace: Workspace node '" << node.name << "' has an unsupported type";
    }
    // 复用 destination 构造；没有输入值时，动态维由 shape 表达式求得。
    NodeBuildContext nodeCtx{builder, loc, node, /*operands=*/{}, *info, /*externalDestination=*/Value()};
    FailureOr<Value> value = makeBufferDestination(nodeCtx, type);
    if (failed(value)) {
      return failure();
    }
    // 保留 workspace 身份，其他字段取选中的视图；分配容量仍由 workspace_plan 给出。
    BufferDescDTO workspaceBuffer = info->buffer;
    workspaceBuffer.bufferId = node.output.buffer.bufferId;
    FailureOr<afir::BufferDescAttr> descriptor = makeBufferDescAttr(builder.getContext(), workspaceBuffer);
    if (failed(descriptor)) {
      return failure();
    }
    afir::setBufferDescAttr(value->getDefiningOp(), *descriptor);
    valueMap[node.name] = *value;
    // 当汇时，写它的那个 Store 要把它当外部 destination——与 `Output` 节点同形。
    for (const std::string &producer : node.input_names) {
      externalDestinations[producer] = *value;
    }
  }
  return success();
}

LogicalResult buildTemplateFunc(OpBuilder &moduleBuilder, const TemplateInfo &tmpl,
                                ArrayRef<BufferDescDTO> workspacePlan) {
  MLIRContext *ctx = moduleBuilder.getContext();
  SmallVector<Type> argTypes, resultTypes;
  if (failed(appendTensorTypes(ctx, tmpl.inputs, argTypes, tmpl.buffer_form))) {
    return failure();
  }
  if (tmpl.buffer_form) {
    if (failed(appendTensorTypes(ctx, tmpl.outputs, argTypes, true))) {
      return failure();
    }
  } else if (failed(appendTensorTypes(ctx, tmpl.outputs, resultTypes, false))) {
    return failure();
  }
  auto funcTy = FunctionType::get(ctx, argTypes, resultTypes);
  auto func = func::FuncOp::create(moduleBuilder, moduleBuilder.getUnknownLoc(), tmpl.name, funcTy);
  func.setPrivate();
  afir::setFunctionRole(func, afir::FunctionRole::KernelBody);
  if (failed(appendTemplateDescriptors(func, moduleBuilder, tmpl, workspacePlan))) {
    return failure();
  }

  Block *entry = func.addEntryBlock();
  OpBuilder b(entry, entry->end());
  Location loc = func.getLoc();

  if (!tmpl.buffer_form && failed(materializeDynamicShapeSsa(b, func, tmpl))) {
    return failure();
  }

  llvm::StringMap<Value> valueMap;
  llvm::StringMap<Value> externalDestinations;
  mapTemplateInputs(entry, tmpl, valueMap);
  mapTemplateOutputs(entry, tmpl, valueMap, externalDestinations);
  // 改写表两处都要用：`mapTemplateWorkspaces` 用它定 workspace 值的类型，
  // `buildTemplateNodes` 用它定 Store 的 destination 描述子。算一次，两处共用同一份。
  FailureOr<LogicalOverrides> overrides = makeLogicalOverrides(tmpl, loc);
  if (failed(overrides)) {
    return failure();
  }
  if (failed(mapTemplateWorkspaces(b, loc, tmpl, overrides->byNode, valueMap, externalDestinations))) {
    return failure();
  }
  if (failed(buildTemplateNodes(b, loc, tmpl, overrides->byNode, externalDestinations, valueMap))) {
    return failure();
  }
  if (tmpl.buffer_form) {
    func::ReturnOp::create(b, loc, ValueRange{});
    return success();
  }
  return buildReturn(b, loc, func, tmpl.outputs, valueMap);
}

FailureOr<ArrayAttr> makeVarRelationsAttr(MLIRContext *ctx, ArrayRef<VarRelationInfo> relations, OpBuilder &builder) {
  SmallVector<Attribute> attrs;
  attrs.reserve(relations.size());
  for (const auto &relation : relations) {
    NamedAttrList item;
    item.append(afir::kVarRelationDstStageKey, builder.getI64IntegerAttr(relation.dst_stage));
    item.append(afir::kVarRelationSrcStageKey, builder.getI64IntegerAttr(relation.src_stage));
    item.append(afir::kVarRelationDstVarKey, builder.getStringAttr(relation.dst_var));
    FailureOr<afir::ExprAttr> sourceExpression = makeExprAttr(ctx, relation.src_expr);
    if (failed(sourceExpression)) {
      return failure();
    }
    item.append(afir::kVarRelationSrcExprKey, *sourceExpression);
    attrs.push_back(DictionaryAttr::get(ctx, item));
  }
  return ArrayAttr::get(ctx, attrs);
}

LogicalResult setOptionalExpressionArrayAttr(Operation *op, StringRef name, ArrayRef<std::string> expressions,
                                             OpBuilder &builder) {
  if (expressions.empty()) {
    return success();
  }
  FailureOr<ArrayAttr> attr = makeExpressionArrayAttr(builder.getContext(), expressions);
  if (failed(attr)) {
    return failure();
  }
  op->setAttr(name, *attr);
  return success();
}

// 步 7 之前，未设置的 (stage, template) 坐标会以 -1 哨兵写进 IR，再由每个读点各自
// 判一次非负挡下来。`#afplan.variant` 表达不了「未设置」——未设置就不挂属性，而「没有
// 这个属性」在下游的含义是「这个 func 不是 template」：`GenTilingData` 直接不收它，
// `TilingProgramSpec::AppendTemplateSpecs` 直接跳过它。只有整个 stage 一个 template
// 都不剩时才会被兜住，部分缺失会静默通过。
//
// 因此把这条判断提到产出点：只要 plan 声明了分派表，每个 template 就必须给出坐标。
// 这比迁移前更早、诊断也更具体，而不是把哨兵值搬进 IR 再层层判负。
LogicalResult validatePlanTemplateVariants(const PlanInfo &plan, Location loc) {
  if (plan.template_counts_per_stage.empty()) {
    return success();
  }
  for (const TemplateInfo &tmpl : plan.templates) {
    if (tmpl.stage_id < 0 || tmpl.template_id < 0) {
      return emitError(loc) << "InvalidPlan: plan '" << plan.name << "' declares a dispatch table, but template '"
                            << tmpl.name << "' has no (stage, template) coordinates";
    }
  }
  return success();
}

LogicalResult appendPlanAttrs(Operation *moduleOp, OpBuilder &builder, const PlanInfo &plan) {
  // 同上：-1 是「未设置」的哨兵，`#afplan.candidate` 不表达这个状态。
  if (plan.subgraph_id >= 0 && plan.plan_id >= 0) {
    moduleOp->setAttr(mlir::afplan::kCandidateAttrName,
                      mlir::afplan::CandidateAttr::get(builder.getContext(), plan.subgraph_id, plan.plan_id));
  }
  moduleOp->setAttr(mlir::afir::kModuleKindAttrName, builder.getStringAttr(mlir::afir::kPlanModuleKind));
  // 此前还写一个 `afir.stage_count`，其值恒等于 template_counts_per_stage 的长度且全仓
  // 无读点——与步 2 清掉的那批影子属性同形，随本步删除。
  //
  // 用 `getChecked`：每个 stage 的 template 数同样是调度器原样交过来的，`get` 只在
  // assert 里校验「必须为正」，关掉断言就构造得出零 template 的 stage。
  if (!plan.template_counts_per_stage.empty()) {
    auto emitStagesError = [&]() -> InFlightDiagnostic {
      return emitError(moduleOp->getLoc()) << "InvalidPlan: plan '" << plan.name << "': ";
    };
    auto stages = mlir::afplan::StagesAttr::getChecked(emitStagesError, builder.getContext(),
                                                       ArrayRef<int64_t>(plan.template_counts_per_stage));
    if (!stages) {
      return failure();
    }
    moduleOp->setAttr(mlir::afplan::kStagesAttrName, stages);
  }
  if (!plan.frontend_shape_vars.empty()) {
    SmallVector<Attribute> vars;
    vars.reserve(plan.frontend_shape_vars.size());
    for (const std::string &var : plan.frontend_shape_vars) {
      vars.push_back(builder.getStringAttr(var));
    }
    moduleOp->setAttr(mlir::afir::kFrontendShapeVarsAttrName, builder.getArrayAttr(vars));
  }
  if (!plan.workspace_plan.empty()) {
    SmallVector<int64_t> workspaceTensorIds;
    workspaceTensorIds.reserve(plan.workspace_plan.size());
    for (const BufferDescDTO &workspace : plan.workspace_plan) {
      workspaceTensorIds.push_back(workspace.bufferId);
    }
    moduleOp->setAttr(mlir::afir::kWorkspaceTensorIdsAttrName, builder.getDenseI64ArrayAttr(workspaceTensorIds));
  }
  // 复用决策已体现在 buffer_id/reuse_id 中；此属性仅供调度指纹与 dump 使用。
  moduleOp->setAttr(mlir::afir::kIsReduceMemReuseAttrName, builder.getBoolAttr(plan.is_reduce_mem_reuse));
  moduleOp->setAttr(mlir::afir::kEnableGroupParallelAttrName, builder.getBoolAttr(plan.enable_group_parallel));
  moduleOp->setAttr(mlir::afir::kCubeTypeAttrName, builder.getI64IntegerAttr(plan.cube_type));
  if (!plan.score_func.empty()) {
    moduleOp->setAttr(mlir::afplan::kScoreFuncAttrName, builder.getStringAttr(plan.score_func));
  }
  return success();
}

LogicalResult appendPlanExpressions(Operation *moduleOp, MLIRContext *ctx, const PlanInfo &plan, OpBuilder &builder) {
  if (failed(setOptionalExpressionArrayAttr(moduleOp, mlir::afir::kOriginVarsAttrName, plan.origin_vars, builder)) ||
      failed(setOptionalExpressionArrayAttr(moduleOp, afir::kShapeGuardsAttrName, plan.shape_guards, builder)) ||
      failed(setOptionalExpressionArrayAttr(moduleOp, afir::kShapeAssertsAttrName, plan.shape_asserts, builder))) {
    return failure();
  }
  if (!plan.var_relations.empty()) {
    FailureOr<ArrayAttr> relations = makeVarRelationsAttr(ctx, plan.var_relations, builder);
    if (failed(relations)) {
      return failure();
    }
    moduleOp->setAttr(afir::kVarRelationsAttrName, *relations);
  }
  return success();
}

}  // namespace

Type parseDtype(MLIRContext *ctx, StringRef dtype) {
  return parseDtypeImpl(ctx, dtype);
}

llvm::ArrayRef<AFIRBridgeOpMapping> getAFIRBridgeableOperations() {
  return bridgeableOperationMappings();
}

bool isAFIRBufferFormOperationSupported(llvm::StringRef scheduler_name) {
  return isFormSupported(scheduler_name, BridgeOpForm::Buffer);
}

llvm::ArrayRef<FloatIrAttrMapping> getFloatIrAttrMappings() {
  return kFloatIrAttrMappings;
}

const FloatIrAttrMapping *findFloatIrAttr(llvm::StringRef schedulerName) {
  for (const FloatIrAttrMapping &mapping : kFloatIrAttrMappings) {
    if (mapping.scheduler_name == schedulerName) {
      return &mapping;
    }
  }
  return nullptr;
}

llvm::ArrayRef<IntIrAttrMapping> getIntIrAttrMappings() {
  return kIntIrAttrMappings;
}

const IntIrAttrMapping *findIntIrAttr(llvm::StringRef schedulerName) {
  for (const IntIrAttrMapping &mapping : kIntIrAttrMappings) {
    if (mapping.scheduler_name == schedulerName) {
      return &mapping;
    }
  }
  return nullptr;
}

FailureOr<afir::ExprAttr> makeExprAttr(MLIRContext *context, StringRef canonicalExpression) {
  return afir::makeExprAttr(context, canonicalExpression);
}

OwningOpRef<ModuleOp> BuildAfirModule(MLIRContext &context, const PlanInfo &sourcePlan) {
  PlanInfo plan = sourcePlan;
  if (failed(resolveWorkspaceSourceViews(plan, UnknownLoc::get(&context)))) {
    return nullptr;
  }
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<afplan::AFPlanDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();

  OpBuilder builder(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc(), plan.name);
  Operation *moduleOp = module->getOperation();
  if (failed(validatePlanTemplateVariants(plan, moduleOp->getLoc()))) {
    return nullptr;
  }
  if (failed(appendPlanAttrs(moduleOp, builder, plan)) ||
      failed(appendPlanExpressions(moduleOp, &context, plan, builder))) {
    return nullptr;
  }
  builder.setInsertionPointToStart(module->getBody());

  for (const auto &tmpl : plan.templates) {
    if (failed(buildTemplateFunc(builder, tmpl, plan.workspace_plan))) {
      return nullptr;
    }
  }

  // 全部 template 建完后追踪跨 stage 归约来源；需早于内存分配改写 SSA 关系。
  afir::annotateReductionDomains(*module);

  if (failed(verify(*module))) {
    return nullptr;
  }
  return module;
}

}  // namespace bridge
}  // namespace afir
}  // namespace mlir
