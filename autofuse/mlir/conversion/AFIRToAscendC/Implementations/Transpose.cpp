/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Transpose.cpp - AFIR transpose lowering ------------------*- C++ -*-===//
// `afir.transpose` 的 lowering，两档（覆盖矩阵 tier D2：API 名 / 扩展头 / tmp 三项全差）：
//   - reg 档 `transpose.ub_ub_reg_base`：`TransposeExtend<inner, total>(dst, src, tmp,
//     dst_dims, src_strides, dst_strides)`（`v35/ascendc/api_regbase/transpose.h:299`），
//     legacy 的 `TransposeRegApiCall` 逐字发的就是它（`v35/codegen/reg_api_call/
//     reg_transpose_api_call.cpp`）。任意 perm。`total = min(向量化轴数, 4)`，**超过 4 根**
//     向量化轴的外层轴各生成一层 `scf.for`（legacy 的 `outer_loop_axes` 同语义）。
//   - mem 档 `transpose.ub_ub_extend`：`codegen::ConfusionTranspose<T>(dst, src, tmp,
//     transposeType, tiling)`（`autofuse/ascendc/api/transpose.h:1325`），legacy 的
//     `TransposeApiCall` 逐字发的就是它（`codegen/api_call/transpose/transpose_api_call.cpp`）。
//     21 种固定 perm 折成 7 个 `AutoFuseTransposeType`，表外 perm（含 >4 维）报 unsupported
//     回落 legacy——legacy 在那里是 `TRANSPOSE_INVALID` 编译失败，报 unsupported 严格不劣于它。
// **两档的 perm 都不另存属性**：轴序差就在两端 descriptor 里。`analyzeTranspose` 是 legacy
// `ReorderInputStrideByOutputAxisOrder`（`reg_transpose_api_call.cpp:58`）的对应物：按输出
// 向量化轴序，逐轴找它在输入向量化轴序里的位置，取那一格的输入 stride；extent 取输出侧
// `repeats`（`GetTensorVectorizedRepeats`，`:30`）。
// reg 档的分维**不走 `collapseVectorizedAxes`**：合轴要求各张量的 `vectorized_axis` 是同一串
// （`VectorSegmentation.h` 的失败情形之一），而转置两端轴序天然不同——真的去合它必然失败。
// legacy 也不合（它逐轴发 `BuildTransposeLoopParams`），所以这里按输出轴序**每轴一段**手工
// 构造 `VectorPlan`，复用 `materializeVectorizedSegments`：外层段各一层 `scf.for`、上界含
// 尾块修正，两端 tile 偏移 `Σ iv × stride`（`VectorizedLoops.cpp:71-110`，与 legacy 的
// `CalcInnerOffset` 同式），最内 `total` 段带尾块修正的 extent 就是 `dst_dims`。
// Compact 布局的连续后缀按 legacy 的条件使用 ActualSize(stride)；其余 stride 保持名义布局。
// mem 档不收 >4 维（表只到 4D），也不需要 `TransferPlan`——它的全部尺寸都在 host tiling 里
// （`ConfusionTransposeTiling`，ATT 侧 `confusion_transpose.cpp` 生成），调用本身不带 dims。
// 侧表登记走 `RemovePad` 先例（布局族、有家族概念）：mem / reg 各记一行，dtype 集合记
// 可建模子集，见注册处注释。`Nddma` 不登记的理由（无家族概念）在这里不成立。
// 本文件的 stable id 与注释都不出现芯片名。
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/Transpose.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "conversion/AFIRToAscendC/VectorizedLoops.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {
constexpr unsigned kBitsPerByte = 8;
constexpr unsigned kMaxTransposeRank = 4;
constexpr unsigned kMaxCompactTransposeSegments = 3;
constexpr unsigned kTwoDimensions = 2;
constexpr unsigned kHalfElementBits = 16;

using ascend::instruction::CapabilityReason;
using ascend::instruction::CapabilitySupport;
using ascend::instruction::ImplementationKey;
using ascend::instruction::ImplementationRegistry;
using ascend::instruction::PatternBuildContext;
using elementwise::ApiFamily;

// `TransposeExtend` 一次调用能收的维数上限（模板特化面 total ≤ 4，`api_regbase/transpose.h`）。
// legacy 查的是同一个数（`kTransposeMaxDims`）：超出部分拆外层循环，不是拒绝。
constexpr size_t kTransposeMaxWindowDims = 4;

// Register the representable dtype masks for matrix checks. bf16 has its own bit;
// ui32/i64 have none. The intentional mem i32/ui32 restriction is recorded in
// ElementTypeMatrixTest; i64/bool rejection is covered by afir-transpose-dtype-boundaries.mlir.
constexpr uint32_t kTransposeMemElementTypes =
    elementwise::kF16 | elementwise::kF32 | elementwise::kI16 | elementwise::kUI16;
constexpr uint32_t kTransposeRegElementTypes = elementwise::kF16 | elementwise::kF32 | elementwise::kI16 |
                                               elementwise::kI32 | elementwise::kUI16 | elementwise::kBF16;

struct TransposeMapping {
  ApiFamily family = ApiFamily::RegBased;
  // 两侧包装都要 tmp（mem 的 TransDataTo5HD 摆位区 / reg 的 Gather 索引缓冲）。legacy 是
  // 仅有的用 `life_time_axis_id = -1` 全生命周期 TBuf 的 API；那份语义由 bridge 的
  // `CanonicalGroupKey` 分组承载（tmp 独占整块资源），这里只查「调度给没给」。
  bool needsTmpBuffer = true;
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
};

// 取一端 descriptor。转置只有一进一出，两个数组各取第一项。
TensorDescAttr getTransposeDescriptor(Operation *op, StringRef name) {
  auto descriptors = op->getAttrOfType<ArrayAttr>(name);
  if (!descriptors || descriptors.size() != 1) {
    return {};
  }
  return dyn_cast<TensorDescAttr>(descriptors[0]);
}

// 一根输出向量化轴上的三项：它自己的 extent（输出 repeats）、它在输入侧的 stride（按轴
// ID 对应到输入向量化轴序取）、它在输出侧的 stride。数组按**输出轴序**（外→内）排列，
// 这个次序就是 `TransposeExtend` 三组实参的次序。
struct TransposeAxis {
  int64_t axisId = -1;
  ExprAttr dstRepeat;
  ExprAttr srcStride;
  ExprAttr dstStride;
};

// 连续后缀和广播轴使用实际尺寸，与 legacy GetVectorizedStrideActualSizeFlag 对齐。
FailureOr<SmallVector<bool>> actualStrideFlags(TensorDescAttr descriptor) {
  SmallVector<bool> flags(descriptor.getVectorizedAxis().size(), false);
  auto one = makeExprAttr(descriptor.getContext(), "1");
  if (failed(one)) {
    return failure();
  }
  ExprAttr repeat = *one;
  ExprAttr stride = *one;
  for (size_t i = flags.size(); i > 0; --i) {
    const size_t index = i - 1;
    auto expected = multiplyExprs(descriptor.getContext(), repeat, stride);
    auto position = axisPosition(descriptor, descriptor.getVectorizedAxis()[index]);
    if (failed(expected) || failed(position)) {
      return failure();
    }
    ExprAttr current = descriptor.getVectorizedStrides()[index];
    if (provablyEqual(current, *expected)) {
      flags[index] = true;
      repeat = descriptor.getRepeats()[*position];
      stride = current;
    } else if (ProveExpressionsEqual(current.getValue().getValue(), "0")) {
      flags[index] = true;
    } else {
      break;
    }
  }
  return flags;
}

FailureOr<Value> materializeTransposeStride(Operation *op, OpBuilder &builder, ExprAttr stride, bool actual) {
  if (!actual) {
    return materializeStrideValue(op, builder, stride);
  }
  VectorSegment expression;
  expression.axisExtents.push_back(stride);
  return materializeActualSegmentExtent(op, builder, expression);
}

struct TransposeStrides {
  SmallVector<Value> source;
  SmallVector<Value> destination;
};

FailureOr<TransposeStrides> materializeTransposeStrides(Operation *op, OpBuilder &builder, ArrayRef<TransposeAxis> axes,
                                                        size_t totalDim) {
  auto input = getTransposeDescriptor(op, kOperandDescsAttrName);
  auto output = getTransposeDescriptor(op, kDestinationDescsAttrName);
  auto inputFlags = actualStrideFlags(input);
  auto outputFlags = actualStrideFlags(output);
  if (failed(inputFlags) || failed(outputFlags)) {
    return failure();
  }
  TransposeStrides strides;
  for (size_t i = axes.size() - totalDim; i < axes.size(); ++i) {
    auto position = llvm::find(input.getVectorizedAxis(), axes[i].axisId);
    const size_t sourceIndex = std::distance(input.getVectorizedAxis().begin(), position);
    auto source = materializeTransposeStride(op, builder, axes[i].srcStride, (*inputFlags)[sourceIndex]);
    auto destination = materializeTransposeStride(op, builder, axes[i].dstStride, (*outputFlags)[i]);
    if (failed(source) || failed(destination)) {
      return failure();
    }
    strides.source.push_back(*source);
    strides.destination.push_back(*destination);
  }
  return strides;
}

// perm 推导（legacy `ReorderInputStrideByOutputAxisOrder` + `GetTensorVectorizedRepeats`）。
// 收不了（数组自相矛盾、输出向量化轴在输入里找不到）一律 `failure()`，调用方翻成
// unsupported。
// capability 与 pattern 问的是同一件事，各算一遍而不是缓存——与 `RemovePad.cpp` 的
// `analyzeRemovePad` 同一个理由：缓存反而多一条「这份缓存还新鲜吗」的判据要养。
FailureOr<SmallVector<TransposeAxis>> analyzeTranspose(Operation *op) {
  TensorDescAttr input = getTransposeDescriptor(op, kOperandDescsAttrName);
  TensorDescAttr output = getTransposeDescriptor(op, kDestinationDescsAttrName);
  if (!input || !output) {
    return failure();
  }
  ArrayRef<int64_t> inputAxes = input.getVectorizedAxis();
  ArrayRef<ExprAttr> inputStrides = input.getVectorizedStrides();
  ArrayRef<int64_t> outputAxes = output.getVectorizedAxis();
  ArrayRef<ExprAttr> outputStrides = output.getVectorizedStrides();
  ArrayRef<int64_t> outputAllAxes = output.getAxis();
  ArrayRef<ExprAttr> outputRepeats = output.getRepeats();
  if (inputAxes.size() != inputStrides.size() || outputAxes.size() != outputStrides.size() || outputAxes.empty() ||
      outputAllAxes.size() != outputRepeats.size()) {
    return failure();
  }

  SmallVector<TransposeAxis> axes;
  axes.reserve(outputAxes.size());
  for (auto [index, axisId] : llvm::enumerate(outputAxes)) {
    auto outputPosition = llvm::find(outputAllAxes, axisId);
    auto inputPosition = llvm::find(inputAxes, axisId);
    if (outputPosition == outputAllAxes.end() || inputPosition == inputAxes.end()) {
      return failure();
    }
    TransposeAxis axis;
    axis.axisId = axisId;
    axis.dstRepeat = outputRepeats[std::distance(outputAllAxes.begin(), outputPosition)];
    axis.srcStride = inputStrides[std::distance(inputAxes.begin(), inputPosition)];
    axis.dstStride = outputStrides[index];
    if (!axis.dstRepeat || !axis.srcStride || !axis.dstStride) {
      return failure();
    }
    axes.push_back(std::move(axis));
  }
  return axes;
}

// 内层连续轴数：从最内向外数「输出 stride[i] == repeats[i+1] × stride[i+1]」成立的轴，
// 上限 3。legacy `GetContinuousInnerAxisNum`（`reg_transpose_api_call.cpp:41`）逐字同判
// （含 3 根封顶——4 根以上连续时把第 4 根当外层维发，dims/strides 仍完整描述布局，所以
// 封顶不产生「收不了」的形状）。
uint8_t countInnerDim(Operation *op, ArrayRef<TransposeAxis> axes) {
  MLIRContext *context = op->getContext();
  uint8_t inner = 1;
  for (int64_t index = static_cast<int64_t>(axes.size()) - 2; index >= 0; --index) {
    if (inner >= kMaxCompactTransposeSegments) {
      break;
    }
    FailureOr<ExprAttr> expected = multiplyExprs(context, axes[index + 1].dstRepeat, axes[index + 1].dstStride);
    if (failed(expected)) {
      break;
    }
    llvm::StringRef strideText = axes[index].dstStride.getValue().getValue();
    llvm::StringRef expectedText = expected->getValue().getValue();
    if (strideText != expectedText && !ProveExpressionsEqual(strideText, expectedText)) {
      break;
    }
    ++inner;
  }
  return inner;
}

// --- reg 档：合成 VectorPlan ---------------------------------------------------

// 按输出轴序每轴一段的 `VectorPlan`。**不经过 `collapseVectorizedAxes`**：合轴要求各张量
// 的 `vectorized_axis` 同一串，转置两端轴序天然不同。legacy 也不合（`BuildTransposeLoopParams`
// 逐轴发）。段 stride 那一列在输入侧取的是 **perm 重排后**的输入 stride（`analyzeTranspose`
// 已按轴 ID 对好位）；extent 一律取输出侧 repeats——一趟迭代走多远由输出说了算。
VectorPlan makeTransposeVectorPlan(const TensorDescAttr &input, const TensorDescAttr &output,
                                   ArrayRef<TransposeAxis> axes) {
  VectorPlan plan;
  plan.tensors.push_back(VectorTensor{VectorTensor::Side::Operand, 0, input});
  plan.tensors.push_back(VectorTensor{VectorTensor::Side::Destination, 0, output});
  plan.segments.reserve(axes.size());
  for (const TransposeAxis &axis : axes) {
    VectorSegment segment;
    segment.axes.push_back(axis.axisId);
    segment.axisExtents.push_back(axis.dstRepeat);
    segment.extent = axis.dstRepeat;
    segment.strides.push_back(axis.srcStride);
    segment.strides.push_back(axis.dstStride);
    plan.segments.push_back(std::move(segment));
  }
  return plan;
}

// --- mem 档：21 种 perm 折 7 个枚举 ---------------------------------------------

// perm 序列：输出第 i 根向量化轴在输入向量化轴序里的位置。legacy
// `CodeGenGetTransposeType`（`transpose_api_call.cpp:85`）的同一序列；表逐字搬移自
// `transpose_base_type.h` 的 `kPermutationTable`（只搬 `true_transpose_type`，legacy
// 也不读 `potential_axis_idx`）。2D 1 种、3D 5 种、4D 15 种，共 21 条。
std::optional<ascendc::AutoFuseTransposeType> foldMemPermutation(ArrayRef<TransposeAxis> axes, Operation *op) {
  TensorDescAttr input = getTransposeDescriptor(op, kOperandDescsAttrName);
  if (!input) {
    return std::nullopt;
  }
  ArrayRef<int64_t> inputAxes = input.getVectorizedAxis();
  SmallVector<unsigned> permutation;
  permutation.reserve(axes.size());
  for (const TransposeAxis &axis : axes) {
    auto position = llvm::find(inputAxes, axis.axisId);
    if (position == inputAxes.end()) {
      return std::nullopt;
    }
    permutation.push_back(static_cast<unsigned>(std::distance(inputAxes.begin(), position)));
  }
  using Type = ascendc::AutoFuseTransposeType;
  // 表的一行：perm 的维数与前 `size` 位下标（其余位补 0）。用聚合 struct 而不是
  // `initializer_list`——后者做不了 constexpr 表（背撑数组的存储期在常量求值里没有）。
  struct MemPermutationRow {
    unsigned size;
    std::array<unsigned, kMaxTransposeRank> digits;
    Type type;
  };
  static constexpr MemPermutationRow kTable[] = {
      {2, {1, 0, 0, 0}, Type::TRANSPOSE_ND2ND_ONLY}, {3, {0, 2, 1, 0}, Type::TRANSPOSE_ND2ND_021},
      {3, {1, 0, 2, 0}, Type::TRANSPOSE_ND2ND_102},  {3, {1, 2, 0, 0}, Type::TRANSPOSE_ND2ND_ONLY},
      {3, {2, 0, 1, 0}, Type::TRANSPOSE_ND2ND_ONLY}, {3, {2, 1, 0, 0}, Type::TRANSPOSE_ND2ND_210},
      {4, {0, 1, 3, 2}, Type::TRANSPOSE_ND2ND_021},  {4, {0, 2, 1, 3}, Type::TRANSPOSE_ND2ND_0213},
      {4, {0, 2, 3, 1}, Type::TRANSPOSE_ND2ND_021},  {4, {0, 3, 1, 2}, Type::TRANSPOSE_ND2ND_021},
      {4, {0, 3, 2, 1}, Type::TRANSPOSE_ND2ND_0321}, {4, {1, 0, 2, 3}, Type::TRANSPOSE_ND2ND_102},
      {4, {1, 2, 0, 3}, Type::TRANSPOSE_ND2ND_102},  {4, {1, 2, 3, 0}, Type::TRANSPOSE_ND2ND_ONLY},
      {4, {2, 0, 1, 3}, Type::TRANSPOSE_ND2ND_102},  {4, {2, 1, 0, 3}, Type::TRANSPOSE_ND2ND_2103},
      {4, {2, 3, 0, 1}, Type::TRANSPOSE_ND2ND_ONLY}, {4, {2, 3, 1, 0}, Type::TRANSPOSE_ND2ND_210},
      {4, {3, 0, 1, 2}, Type::TRANSPOSE_ND2ND_ONLY}, {4, {3, 1, 2, 0}, Type::TRANSPOSE_ND2ND_210},
      {4, {3, 2, 0, 1}, Type::TRANSPOSE_ND2ND_210},
  };
  for (const MemPermutationRow &row : kTable) {
    if (permutation.size() == row.size && std::equal(permutation.begin(), permutation.end(), row.digits.begin())) {
      return row.type;
    }
  }
  return std::nullopt;
}

// --- 两档共用的形态判据 ----------------------------------------------------------

bool hasTwoIndexingMaps(Operation *op) {
  auto maps = op->getAttrOfType<ArrayAttr>(kIndexingMapsAttrName);
  return maps && maps.size() == kTwoDimensions;
}

// dtype 口径 A（用户确认）：与 legacy 注册表逐字一致——mem 6 种
// （`ascir_builtin_ops_v1.cpp:688`），reg 再加 bf16（`ascir_builtin_ops_v2.cpp:1073`）。
// v2 的 bool 不收：AFIR 无 I1 通路（设计文档「不做的事 ③」）。
// **口径 A 修订（2026-10-22，用户确认）：mem 档摘除 i32/ui32。** 4 字节整型在 mem 档
// （ConfusionTranspose 公式 + v1 布局规则）下布局与公式错位，数值必错：
//   - 公式写死尾轴 16 元素对齐（`att/.../api/confusion_transpose.cpp` 的
//     `secondAxisAlign = ALIGN_UP(width, 16)`，dtype 无关）；
//   - v1 布局只对 `DT_FLOAT` 的 transpose 图提 64B（`platform/v1/alignment_strategy.cpp`
//     的 `SetAlignWidth`），其余 32B——4 字节整型折算 8 元素 ≠ 16（f16/bf16 是 2 字节，
//     32B/2B=16 恰好重合，float 系不受影响）。
// legacy 生产同错（同一调度器、同一公式；legacy ST 因手搓布局恰好掩盖，全树无 int32
// 用例）。实测证据与恢复条件见设计文档
// `2026-10-19-afir-transpose-e2e-design.md` 收尾结论「int32」两条。reg 档不受影响
// （regbase 显式传 dims/strides，布局与公式无耦合），保留口径 A。
// The type predicate also covers ui32, which has no registry-mask bit.
// Both i32/ui32 are reg-only; bf16 is registered explicitly and is also reg-only.
bool isSupportedTransposeElementType(ApiFamily family, Type type) {
  if (type.isF16() || type.isF32() || type.isSignlessInteger(kHalfElementBits) ||
      type.isUnsignedInteger(kHalfElementBits)) {
    return true;
  }
  // i32 / ui32：mem 档摘除（见上方「口径 A 修订」），仅 reg 档保留。
  if (type.isSignlessInteger(32) || type.isUnsignedInteger(32)) {
    return family == ApiFamily::RegBased;
  }
  return family == ApiFamily::RegBased && type.isBF16();
}

// 两个 capability 共用的判据：descriptor / 索引映射 / 执行单元 / 家族 / tmp / 操作数结构 /
// dtype。返回 nullopt 表示一路通过；否则是该报出去的结论。
std::optional<CapabilitySupport> checkTransposeForm(Operation *operation,
                                                    const ascend::instruction::QueryContext &context,
                                                    const TransposeMapping &mapping) {
  if (!elementwise::hasCanonicalDescriptors(operation) || !hasTwoIndexingMaps(operation)) {
    return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
  }
  // 执行单元的 feature 先于家族问（`RemovePad.cpp` 同一条序）：两者答的都是「这块目标
  // 选不选这个 implementation」，feature 不可用时的 reason 更贴近事实。
  ascend::FeatureState executionSupport = context.target.getExecutionSupport(mapping.executionSpec);
  if (executionSupport == ascend::FeatureState::Unavailable) {
    return CapabilitySupport::unsupported(CapabilityReason::TargetFeatureUnavailable);
  }
  if (executionSupport == ascend::FeatureState::Unknown) {
    return CapabilitySupport::undetermined(CapabilityReason::TargetFeatureNotModeled);
  }
  if (std::optional<CapabilitySupport> family = elementwise::checkApiFamily(mapping.family, context)) {
    return family;
  }
  if (std::optional<CapabilitySupport> tmpBuffer = elementwise::checkTmpBuffer(operation, mapping.needsTmpBuffer)) {
    return tmpBuffer;
  }
  // 操作数结构定了再读操作数：destination 缺席时按裸下标读会取到 tmpBuffer 上。
  FailureOr<OperandPartition> partition = getOperandPartition(operation);
  if (failed(partition) || partition->inputCount != 1 || partition->destinationCount != 1 ||
      partition->tmpBufferCount != 1) {
    return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
  }
  auto sourceType = dyn_cast<ascendc::LocalTensorType>(partition->inputs(operation)[0].getType());
  auto destinationType = dyn_cast<ascendc::LocalTensorType>(partition->destinations(operation)[0].getType());
  if (!sourceType || !destinationType) {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }
  // 两端同 dtype：转置的两端是两个独立操作数，ODS 的类型约束管不到这里。
  if (!isSupportedTransposeElementType(mapping.family, sourceType.getElementType()) ||
      sourceType.getElementType() != destinationType.getElementType()) {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }
  return std::nullopt;
}

// --- reg 档 capability / pattern -----------------------------------------------

class TransposeRegCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit TransposeRegCapabilityProvider(std::shared_ptr<const TransposeMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (std::optional<CapabilitySupport> form = checkTransposeForm(operation, context, *mapping_)) {
      return *form;
    }
    // 循环、tile 偏移与尾块修正都要调度给的那一套，所以只收 `Scheduled`
    // （`Nddma.cpp` 同一条）。转置的索引映射是置换、不是恒等，identity 那一档本来也落
    // 不到这里——映射缺失在 `hasTwoIndexingMaps` 就挡了。
    FailureOr<TransferPlan> plan = buildTransferPlan(operation);
    if (failed(plan) || plan->kind != TransferPlanKind::Scheduled) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (failed(analyzeTranspose(operation))) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *, const ascend::instruction::InplaceQuery &,
                                 const ascend::instruction::QueryContext &) const override {
    // `TransposeExtend` 的 src/dst 重叠（原地转置）没有核过，legacy 也不产出这种形态。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryScalarInput(Operation *, const ascend::instruction::ScalarInputQuery &,
                                     const ascend::instruction::QueryContext &) const override {
    // legacy 的 `IsNodeValid` 也拒标量输入（`v2_ascir_codegen_impl.h:3750-3754`）。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const TransposeMapping> mapping_;
};

LogicalResult emitTransposeRegCall(TransposeOp op, TransposeOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
                                   ArrayRef<TransposeAxis> axes, const VectorizedWindow &window,
                                   const ascend::ExecutionSpec &execution) {
  const size_t totalDim = window.extents.size();
  auto strides = materializeTransposeStrides(op, rewriter, axes, totalDim);
  if (failed(strides)) {
    return failure();
  }
  Operation *call = rewriter.create<ascendc::TransposeExtendRegBaseOp>(
      op.getLoc(), window.tiles[1], window.tiles[0], adaptor.getTmpBuffer(), window.extents, strides->source,
      strides->destination, rewriter.getI8IntegerAttr(static_cast<int8_t>(countInnerDim(op, axes))),
      rewriter.getI8IntegerAttr(static_cast<int8_t>(totalDim)));
  return setExecutionSpec(call, execution);
}

class TransposeRegConversionPattern final : public OpConversionPattern<TransposeOp> {
 public:
  TransposeRegConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                std::shared_ptr<const TransposeMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination() || !adaptor.getTmpBuffer()) {
      return rewriter.notifyMatchFailure(op, "buffer-form transpose requires a destination and a tmp buffer");
    }
    if (!isa<ascendc::LocalTensorType>(adaptor.getInput().getType())) {
      return rewriter.notifyMatchFailure(op, "source is not a target-dialect tile");
    }
    FailureOr<TransferPlan> transferPlan = buildTransferPlan(op);
    if (failed(transferPlan) || transferPlan->kind != TransferPlanKind::Scheduled) {
      return rewriter.notifyMatchFailure(op, "cannot build a scheduled AFIR transfer plan");
    }
    FailureOr<SmallVector<TransposeAxis>> axes = analyzeTranspose(op);
    if (failed(axes)) {
      return rewriter.notifyMatchFailure(op, "cannot derive the transposed vectorized axes");
    }
    const auto totalDim = static_cast<uint8_t>(std::min(axes->size(), kTransposeMaxWindowDims));

    // 每轴一段的合成 plan：`materializeVectorizedSegments` 拿它生成 >4 维的外层 `scf.for`
    // （上界含尾块修正）与两端偏移，并给出最内 `totalDim` 段带尾块修正的 extent（即
    // `dst_dims`）。窗口值与调用建在它留下的插入点上（最内新循环体内）。
    VectorPlan vectorPlan = makeTransposeVectorPlan(getTransposeDescriptor(op, kOperandDescsAttrName),
                                                    getTransposeDescriptor(op, kDestinationDescsAttrName), *axes);
    FailureOr<SmallVector<Value>> ordered =
        orderTilesForPlan(vectorPlan, {adaptor.getInput()}, adaptor.getDestination());
    if (failed(ordered)) {
      return rewriter.notifyMatchFailure(op, "cannot order the tiles the segmentation talks about");
    }
    FailureOr<VectorizedWindow> window = materializeVectorizedSegments(op, rewriter, vectorPlan, *transferPlan,
                                                                       *ordered, static_cast<unsigned>(totalDim));
    if (failed(window) || window->tiles.size() != vectorPlan.tensors.size() || window->extents.size() != totalDim) {
      return rewriter.notifyMatchFailure(op, "cannot materialize the vectorized segment window");
    }

    if (failed(emitTransposeRegCall(op, adaptor, rewriter, *axes, *window, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "cannot materialize transpose strides or execution spec");
    }
    // 结果是**整块**目的地 tile：窗口值定义在外层循环体里，支配不了这个 op 的使用者
    // （`afir.store` 在循环外），与 `Nddma.cpp` / `Broadcast.cpp` 同一个理由。
    rewriter.replaceOp(op, adaptor.getDestination());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const TransposeMapping> mapping_;
};

// --- mem 档 capability / pattern -----------------------------------------------

class TransposeMemCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit TransposeMemCapabilityProvider(std::shared_ptr<const TransposeMapping> mapping)
      : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *operation,
                                  const ascend::instruction::QueryContext &context) const override {
    if (!operation || !mapping_) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    if (std::optional<CapabilitySupport> form = checkTransposeForm(operation, context, *mapping_)) {
      return *form;
    }
    FailureOr<SmallVector<TransposeAxis>> axes = analyzeTranspose(operation);
    if (failed(axes)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    // 21 种表之外（含 >4 维）一律 unsupported：legacy 在这里是 `TRANSPOSE_INVALID` 编译
    // 失败（`transpose_api_call.cpp:103`），报 unsupported 让整图回落 legacy，严格不劣于它。
    if (!foldMemPermutation(*axes, operation)) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
    }
    // host tiling 字段名必须在：18 个 param 由 ATT 在 host 侧填，这一层只按字段名取整个
    // `ConfusionTransposeTiling`，没有它取不到。
    auto field = operation->getAttrOfType<StringAttr>(kApiTilingFieldAttrName);
    if (!field || !getApiTilingFieldType(operation, field.getValue())) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    return CapabilitySupport::supported();
  }

  CapabilitySupport queryInplace(Operation *, const ascend::instruction::InplaceQuery &,
                                 const ascend::instruction::QueryContext &) const override {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryScalarInput(Operation *, const ascend::instruction::ScalarInputQuery &,
                                     const ascend::instruction::QueryContext &) const override {
    // legacy 的 `IsNodeValid` 也拒标量输入（`v1_ascir_codegen_impl.h`）。
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
  }

  CapabilitySupport queryBroadcastInline(Operation *, const ascend::instruction::BroadcastInlineQuery &,
                                         const ascend::instruction::QueryContext &) const override {
    return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
  }

 private:
  std::shared_ptr<const TransposeMapping> mapping_;
};

class TransposeMemConversionPattern final : public OpConversionPattern<TransposeOp> {
 public:
  TransposeMemConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                std::shared_ptr<const TransposeMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination() || !adaptor.getTmpBuffer()) {
      return rewriter.notifyMatchFailure(op, "buffer-form transpose requires a destination and a tmp buffer");
    }
    FailureOr<SmallVector<TransposeAxis>> axes = analyzeTranspose(op);
    if (failed(axes)) {
      return rewriter.notifyMatchFailure(op, "cannot derive the transposed vectorized axes");
    }
    std::optional<ascendc::AutoFuseTransposeType> type = foldMemPermutation(*axes, op);
    if (!type) {
      return rewriter.notifyMatchFailure(op, "permutation is outside the mem-based transpose table");
    }
    std::optional<llvm::StringRef> field = op.getApiTilingField();
    if (!field) {
      return rewriter.notifyMatchFailure(op, "mem-based transpose is missing its api tiling field name");
    }
    if (!getApiTilingFieldType(op, *field)) {
      return rewriter.notifyMatchFailure(op, "TilingData is missing the transpose field '" + field->str() + "'");
    }
    auto tiling = getCanonicalTilingArgument(op->getParentOfType<func::FuncOp>());
    if (failed(tiling)) {
      return failure();
    }
    // Copy the API tiling struct locally before binding the wrapper's non-const reference.
    Value apiTilingData = createTilingSchemaMember(rewriter, op.getLoc(), *tiling, *field);

    // 本地侧偏移恒为 0（队列按 slot 轮转、TBuf 只装一个 tile），与 `Nddma.cpp` /
    // `UnaryElementwise.cpp` 同一条；legacy 的 `y[TensorVectorizedOffset(...)]` 在这些图上
    // 取的也是 0。mem 档不建循环、不发尺寸——全部尺寸都在 tiling 里。
    Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value source = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), adaptor.getInput().getType(),
                                                                   adaptor.getInput(), zeroOffset);
    Value destination = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), adaptor.getDestination().getType(),
                                                                        adaptor.getDestination(), zeroOffset);

    Operation *call = rewriter.create<ascendc::ConfusionTransposeOp>(op.getLoc(), destination, source,
                                                                     adaptor.getTmpBuffer(), *type, apiTilingData);
    if (failed(setExecutionSpec(call, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    // 尾随的 `AscendC::PipeBarrier<PIPE_ALL>()` 在 emitter 里（它是调用的一部分），不在这里。
    rewriter.replaceOp(op, adaptor.getDestination());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const TransposeMapping> mapping_;
};

}  // namespace

LogicalResult registerTransposeImplementations(ascend::instruction::ImplementationRegistry &registry,
                                               ascend::instruction::SelectionPolicyRegistry *) {
  // 注册期把「选不选中我」与「往 kernel 里内联哪份头文本」对上（`checkFamilyMatchesHeader`）。
  // reg 档的 `transpose_reg_base.h` 带 `_reg_base.h` 后缀；mem 档的两个头不带——`Any`/`Shared`
  // 之外的判据只看这个方向，中立的头（mem 的 `transpose.h`）两侧都能用。reg 档的官方头
  // （第三段列表）不参与家族判别：它是 CANN 自带的，不是哪张供给表的标记。
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::TransposeExtendRegBaseOp>(ApiFamily::RegBased)) ||
      failed(elementwise::checkFamilyMatchesHeader<ascendc::ConfusionTransposeOp>(ApiFamily::MemBased))) {
    return failure();
  }
  // 侧表登记走 `RemovePad` 先例：布局族但**有**家族概念（两张供给表各一个包装），mem / reg
  // 各记一行，`EveryElementwiseOpCoversBothApiFamilies` 的双侧覆盖断言因此管得到它。dtype
  // 集合按家族各记可建模子集（`kTransposeMemElementTypes` / `kTransposeRegElementTypes`），
  // 真实判据在 `isSupportedTransposeElementType`；ui32 在 `ElementTypes` 中无位；bf16 已显式登记。
  // legacy 声明的 i64 因 8 字节 base case 不支持而未接入，ODS 拒收行为由
  // afir-transpose-dtype-boundaries.mlir 看护，不通过无位的掩码静默宣称支持。mem 侧因摘除 i32/ui32 比 legacy
  // 窄，窄差登记在 `ElementTypeMatrixTest` 的 `kKnownNarrower`。
  // `Nddma` 不登记的理由（无家族概念、不给集合会撞 `OnlyCastPair` 的钉死名单）在这里都不成立。
  constexpr llvm::StringLiteral kRegId = "transpose.ub_ub_reg_base";
  constexpr llvm::StringLiteral kMemId = "transpose.ub_ub_extend";
  elementwise::recordRegisteredFamily(TransposeOp::getOperationName(), kMemId, ApiFamily::MemBased,
                                      kTransposeMemElementTypes);
  elementwise::recordRegisteredFamily(TransposeOp::getOperationName(), kRegId, ApiFamily::RegBased,
                                      kTransposeRegElementTypes);
  if (failed(registry.registerImplementation<TransposeOp, TransposeMapping, TransposeRegCapabilityProvider>(
          ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{kRegId.str()},
          TransposeMapping{ApiFamily::RegBased, /*needsTmpBuffer=*/true},
          [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
             std::shared_ptr<const TransposeMapping> mapping) {
            patterns.add<TransposeRegConversionPattern>(context.typeConverter, patterns.getContext(), key,
                                                        std::move(mapping));
          }))) {
    return failure();
  }
  return registry.registerImplementation<TransposeOp, TransposeMapping, TransposeMemCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{kMemId.str()},
      TransposeMapping{ApiFamily::MemBased, /*needsTmpBuffer=*/true},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const TransposeMapping> mapping) {
        patterns.add<TransposeMemConversionPattern>(context.typeConverter, patterns.getContext(), key,
                                                    std::move(mapping));
      });
}

}  // namespace mlir::ascend::kernel
