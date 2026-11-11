/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Broadcast.cpp - AFIR broadcast lowering ------------------*- C++ -*-===//
//
// `afir.broadcast` 的 lowering。**两档**，判据互斥；第二档两侧各生成各的 API：
//
//   1. `Duplicate` 一档：源在每一根向量化轴上 repeat 都是 1（源就是一个值）。legacy 的
//      `BroadcastScalar` 生成的 `Duplicate(dst, 标量, count)`。两侧各注册一个。
//   2. 分段一档：其余形态。按向量化轴的分段（`collapseVectorizedAxes`，阶段 2 合轴）
//      分解，两侧的分解方式不同，因为生成的不是同一个函数：
//        - mem 侧（`Segmented`）把最内两段交给两轴 API、其余段交给 `scf.for`；多个广播段时
//          由内而外每段一趟，趟间生成 `PipeBarrier<PIPE_V>`。
//        - reg 侧（`Extend`）把整串段落成 `BroadcastExtend<T, rank>` 的两个 shape 数组，
//          一层循环都不生成——合轴与循环分解是那个 API 自己在 tiling 里做的事。
//
// legacy 对应的四条分支在 `codegen/api_call/broadcast/broadcast_api_call.cpp`：
//
//   1. `IsBroadcastConstantTensor(x)` → `BroadcastScalar` → `Duplicate(dst, 标量, count)`；
//   2. `broadcast_num == 0`           → `DataCopy`（调度会把这种节点删掉，见该处注释）；
//   3. `broadcast_num == 1`           → 三轴 `Broadcast(dst, src, s0,s1,s2, d0,d1,d2, tmp, stride)`；
//   4. `broadcast_num == 2`           → 四轴形态。
//
// 第 2 档收的是 legacy 的 3 与 4，覆盖面严格超过它们：legacy 的四轴只认
// `(1,B,1,B)` 与 `(A,1,A,1)` 两个硬编码 pattern，其余 `ASSERT(false)`；这里是 N 个广播段
// 的通用分解。同时结构性避开 `Or`——legacy 的三轴 / 四轴在 `Z > 8` 时落
// `BroadcastMiddleDim` 的 `Or`（`broadcast.h:79` / `:88`），而这里生成的两种形态一个
// `src_m == 1`、一个 `src_z == 0`，够不着那一支。设计与探针结论见
// `docs/development/2026-09-28-afir-vectorized-axis-collapse-design.md`。
//
// 收不了的形态在 capability 里报 unsupported，含它的图整张回落 legacy——与接通之前的
// 行为逐字相同，不会静默算错。
//
// tmp buffer 判据不同侧：mem 侧调度给，reg 侧不给
// （`BroadcastAscIrCodegenImpl::CalcTmpBufSize` vs `BroadcastAscIrCodegenImplV2` 没有
// 这个覆写）。`checkTmpBuffer` 拿这个块数严格对等，所以 `Duplicate` 一档是两个
// implementation，不是一个 `Any`；分段那一档的 reg 侧同样 `needsTmpBuffer=false`，
// 而 `BroadcastExtend` 的形参表里也确实没有这块 buffer。
//
// 本文件的 stable id 与注释都不出现芯片名。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/Implementations/BroadcastImplementation.h"
#include "conversion/AFIRToAscendC/VectorizedLoops.h"
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"

#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseStockCalls.h"
#include "conversion/AFIRToAscendC/Implementations/Synchronization.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Native Duplicate types. Mem-based ui8 additionally uses the legacy duplicate.h
// wrapper; keep it separate from the stock API and the reg-based registration.
constexpr uint32_t kDuplicateElementTypes = kF16 | kF32 | kI16 | kI32 | kUI16;
constexpr unsigned kUInt8BitWidth = 8;

// 这个 op 上有两档 implementation，判据互斥（见 `analyzeSegmentedBroadcast`），
// 但前面那一长串 feature / 支持表 / dtype / tmp buffer 的检查两档逐字相同。
// 用一位区分而不是抄两份 provider：抄两份的话，第一个只在一边改的检查就会让两档说岔。
enum class BroadcastTier {
  // 源在每根向量化轴上都是 1：legacy 的 `BroadcastScalar`，生成 `Duplicate`。
  Duplicate,
  // 分段广播，mem 侧：最内两段交给两轴 API，其余段交给循环（功能需求 3 / 4）。
  Segmented,
  // 分段广播，reg 侧：整串段交给 `BroadcastExtend` 的 shape 数组，一层循环都不生成。
  Extend,
};

struct BroadcastMapping {
  ApiFamily family = ApiFamily::MemBased;
  BroadcastTier tier = BroadcastTier::Duplicate;
  uint32_t elementTypes = kDuplicateElementTypes;
  // 调度给不给 tmp buffer。给了也不传给调用——legacy 只在 int64 / uint64 时才把它
  // 接到 `Duplicate` 的参数尾巴上，而那两种 dtype 不在 `kDuplicateElementTypes` 里。
  // 这一位存在只是为了让 `checkTmpBuffer` 与调度对得上，两侧的值不同。
  bool needsTmpBuffer = false;
  ascend::ExecutionSpec executionSpec{ascend::ProcessorType::AICore, ascend::AICoreUnit::Vector,
                                      ascend::ExecutionModel::SIMD, ascend::InstructionForm::MemoryBased};
};

// `descriptor` 描述的张量，在它的每一根向量化轴上 repeat 都是 1 吗。
//
// 这是 legacy `IsBroadcastConstantTensor` 的同一条判据，读的也是同一批字段：
// 它按 `vectorized_axis_pos[i]` 去 `axis_size` 里取，descriptor 这边 `vectorized_axis`
// 存的是轴 id，位置要回 `axis` 里找——两处说的是同一个下标。
//
// 「取不到」与「不是 1」都返回 false：符号表达式（`z0t_size`）不是常量 1，
// 静态判不了就是判不了，不猜。
bool isConstantOnVectorizedAxes(afir::TensorDescAttr descriptor) {
  if (!descriptor || descriptor.getVectorizedAxis().empty()) {
    return false;
  }
  ArrayRef<int64_t> axis = descriptor.getAxis();
  ArrayRef<afir::ExprAttr> repeats = descriptor.getRepeats();
  if (axis.size() != repeats.size()) {
    return false;
  }
  for (int64_t vectorizedAxis : descriptor.getVectorizedAxis()) {
    const auto *position = llvm::find(axis, vectorizedAxis);
    if (position == axis.end()) {
      return false;
    }
    int64_t literal = 0;
    // 与 `CanonicalResources.cpp` / `CanonicalMaterialization.cpp` 判字面量同一个写法。
    if (repeats[static_cast<size_t>(position - axis.begin())].getValue().getValue().getAsInteger(10, literal) ||
        literal != 1) {
      return false;
    }
  }
  return true;
}

afir::TensorDescAttr firstDescriptor(Operation *op, llvm::StringRef name) {
  auto descriptors = op->getAttrOfType<ArrayAttr>(name);
  if (!descriptors || descriptors.empty()) {
    return {};
  }
  return dyn_cast<afir::TensorDescAttr>(descriptors[0]);
}

//===----------------------------------------------------------------------===//
// 分段广播（功能需求 3 / 4）
//===----------------------------------------------------------------------===//

// 这一档生成的是 `broadcast.h` 的包装 `Broadcast(...)`，**与 legacy 的 `BroadcastOneAxis`
// 逐字同一个调用**，分流到哪一支由包装自己按 dtype 决定：`BroadcastInt64`（i64 / ui64）、
// `BroadcastWithCast`（i8 / ui8）、`BroadcastCommon`（其余）。所以这里不按分支收窄——
// 收窄的理由只能是「legacy 在那个 dtype 上生成的不是这条路生成的调用」，而这里生成的就是同一条。
//
// 只差在 AFIR 这一层的编号上：i8 / i64 / ui64 / ui32 在 `ElementTypes` 里没有位
// （`elementTypeBit` 答 0），够不着也就谈不上收。
//
constexpr uint32_t kSegmentedElementTypes = kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16;

// `expression` 可证等于 `literal` 吗。证不了返回 false（保守方向：报不支持，不是算错）。
//
// 文本相同先短路：这里比的绝大多数是字面量对字面量，短路省掉一次 SymEngine 往返。
bool provablyLiteral(afir::ExprAttr expression, llvm::StringRef literal) {
  if (!expression) {
    return false;
  }
  llvm::StringRef text = expression.getValue().getValue();
  return text == literal || afir::ProveExpressionsEqual(text, literal);
}

// `provablyEqual` 直接用 `mlir::afir` 导出的那份（`VectorSegmentation.cpp`）：
// 文本短路 + `ProveExpressionsEqual`，空侧答 false，语义与这里原先的局部副本一致。

// 一个 `afir.broadcast` 在分段之后的形态。
struct BroadcastSegmentation {
  // `VectorPlan::tensors` 里源与目的地的下标。两侧各恰好一个，否则这个 op 不是本档的形状。
  unsigned source = 0;
  unsigned destination = 0;
  // 广播段（源在其上不前进的那些段）的下标，外→内，至少一个。
  SmallVector<size_t> broadcastSegments;
  // 交给本档自己处理的最外那一段。它之外的段由 `materializeVectorizedSegments` 变成循环。
  size_t windowStart = 0;
  // 最内那个广播段是最内两段里的哪一段：`true` 是形态 B `[normal][brc]`，`false` 是形态 A
  // `[brc][normal]`。在 `classifySegments` 里定，因为那里才是按它查过源 stride 形态的地方
  // ——下降侧再按下标算一遍的话，两处的判据就得各对一次。
  bool broadcastIsInner = false;
  // 窗口里一行到下一行的目的地步长。空表示它与行长相等（稠密布局），非空表示尾轴真有
  // 填充、窗口调用要按它传而不是按行长传。与 reg 侧 `BroadcastExtendShape::pitch` 同一个量、
  // 同一个取法（`classifyExtendSegments`），两处各算一遍。
  afir::ExprAttr pitch;
};

// `plan` 是不是本档收的形状；是就把判据用得上的那几项算出来。
//
// 收的边界（不满足一律 `failure()`，调用方翻成 unsupported、整图回落 legacy）：
//
//   - 源与目的地各恰好一个；段数 ≥ 2。单段是 `Duplicate` 一档的形状（源只有一个
//     元素），由那一档收，两档的判据因此不重叠。
//   - 至少一个广播段；广播段之间不相邻（相邻的在合轴那一步就并成一段了，没并上说明
//     stride 对不上，这里不猜）。
//   - 最内那个广播段落在最内两段之内，且那两段构成合法的两段窗口：目的地内段 stride 为 1，
//     源要么 `(0, 1)`（形态 A）要么 `(1, 0)`（形态 B）。外段的目的地 stride 是行距，
//     与内段 extent 不等即尾轴有填充——那时只收形态 A，按行距传窗口调用（见判据处）。
//
// 不在收的边界里、但本档依赖的一条：切分轴（`TileInner`）恒落在第 0 段，所以带尾块修正
// 的段 extent 只会出现在最外那一段上、只当趟数用。两段窗口里的 `dst_k` 既是一行的长度也是
// 行与行的间距（`BroadcastFirstDim` 的 `dst_inner_offset`），尾块上行长变短、行距不变——
// 切分轴真落到内侧的段上就是静默算错。这条不变量由 `getTileInnerAxis` 的 `front()` 保证，
// 那里的注释写着放宽它之前要回来补什么。
FailureOr<BroadcastSegmentation> classifySegments(const afir::VectorPlan &plan) {
  BroadcastSegmentation shape;
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
  if (count < 2) {
    return failure();
  }
  for (auto [index, segment] : llvm::enumerate(plan.segments)) {
    if (segment.strides.size() != plan.tensors.size()) {
      return failure();
    }
    if (!provablyLiteral(segment.strides[shape.source], "0")) {
      continue;
    }
    // 源与目的地都不前进的段：同一块目的地被写好几遍，是个退化形状。合轴不会造出它
    // （extent 为 1 的段已被剔除），真出现了说明 descriptor 自相矛盾，报不支持。
    if (provablyLiteral(segment.strides[shape.destination], "0")) {
      return failure();
    }
    shape.broadcastSegments.push_back(index);
  }
  if (shape.broadcastSegments.empty()) {
    return failure();
  }
  // 相邻的广播段现在收（2026-10-11）。此前这里拒掉它们，理由是「推下来是对的」与
  // 「有用例证明是对的」不是一回事；现在有用例了
  // （`test/lit/conversion/afir-broadcast-adjacent-brc.mlir`，`[brc][brc][normal]`）。
  //
  // 为什么它们会相邻：设计文档说相邻的广播段在合轴那一步就并成一段了，那句话不完全成立
  // ——目的地在两段之间有填充时（`stride[外] != stride[内] × extent[内]`）它们并不上。
  //
  // 为什么分解是对的：内侧那一趟生成两段窗口，外侧每一趟按本段在目的地上的 stride 整块
  // 复制（`materializeVectorizedSegments` 取的是 `segment.strides[destination]`，不是段长
  // 之积），所以两段之间有填充也走对地址；填充位一起复制，那些位置本来就是 don't care。
  shape.windowStart = std::min(shape.broadcastSegments.front(), count - 2);

  const afir::VectorSegment &outer = plan.segments[count - 2];
  const afir::VectorSegment &inner = plan.segments[count - 1];
  // 一行之内必须连续。这一条不放宽——它说的是「行长那一位真的是逐元素」，与行距无关。
  if (!provablyLiteral(inner.strides[shape.destination], "1")) {
    return failure();
  }
  shape.broadcastIsInner = shape.broadcastSegments.back() == count - 1;
  // 行距就是次内那一位在目的地上的 stride。**此前这里要求它等于行长（稠密），不等就整个
  // 形状拒收**，而片上 tile 的尾轴行距是被补到 32B 的（f32 → 8 个元素，f16 → 16 个），
  // 行长更短时这条等式本来就不成立：`[1,5] → [33,5]` 的目的地 descriptor 里
  // `strides` 的次内一位是 5（逻辑步长）而 `vectorized_strides` 是 8（真实行距），
  // 合轴用的是后者。于是所有尾轴不满一档的形状都被拒在这里（2026-10-12 实测：mem 侧
  // 112 张图里 33 次失败全部命中这一条，其余判据一次都没走到）。
  //
  // 改成取行距。按行距传同时满足了包装对尾轴 32B 对齐的要求——行距恒是一整档，
  // 而行长不是；这是它比「按行长传、另想办法补对齐」更可取的主要理由。
  //
  // 两种形态都收，但「换哪一边」不同（形态 B 自 2026-10-26 起收，此前这里拒掉它）。
  //
  // 形态 A：两侧的最内位是同一位，所以行距要两边一起换才满足包装的
  // `srcShape[i] == dstShape[i]`（`emitWindowCall` 里 `sourceK` 与 `destinationK` 生成的是
  // 同一个 `inner`）。
  //
  // 形态 B：源那一位恒是字面量 1（`emitWindowCall` 生成 `one_`，不是 `inner`），所以只换
  // 目的地那一边。包装走的是 `broadcast.h:304` 的 `src_k == 1 && src_m == dst_m` 一支，
  // 它对两侧 shape 的要求只落在非广播那一位（`m`）上，换 `dst_k` 不碰它；而
  // `AscendC::Broadcast<T, 2, 1>` 要的 32B 对齐天然成立——行距恒是一整档，那就是填充的
  // 定义。多写出来的那几列落进填充位，与形态 A 同一个理由：填充位 don't care
  // （`afir.store` 读这块 tile 时 UB 侧按行距、GM 侧按行长，逐行跳过它们）。
  if (!provablyEqual(outer.strides[shape.destination], inner.extent)) {
    shape.pitch = outer.strides[shape.destination];
  }
  // 形态由广播段落在窗口的哪一侧决定，不由芯片决定：
  //   形态 A `[brc][normal]` 源是 `(0, 1)`；形态 B `[normal][brc]` 源是 `(1, 0)`。
  //
  // 这一条同时把「最内那个广播段必须落在最内两段之内」钉住了，所以没有单独再查一遍：
  // 广播段更靠外的话，最内两段就都是普通段、源 stride 都不为 0，两种形态各要求其中一段
  // 的源 stride 为 0，两条都过不去。单独查一遍的版本写过，变异验证显示它一条用例都挡不住
  // ——不是用例有盲区，是那条判据本身冗余。
  if (!provablyLiteral(outer.strides[shape.source], shape.broadcastIsInner ? "1" : "0") ||
      !provablyLiteral(inner.strides[shape.source], shape.broadcastIsInner ? "0" : "1")) {
    return failure();
  }
  return shape;
}

// 把分段形态落成调用。
//
// 层次：每个广播段一趟，由内而外。最内那一趟是两段窗口的一条调用（形态 A / B）；
// 外面每一趟把内侧已经算好的一整块复制 `extent - 1` 份，趟与趟之间一条
// `PipeBarrier<PIPE_V>`。广播段之间的普通段变成 `scf.for`，嵌在对应的两趟之间——
// 内侧那一趟要在循环体里逐次做，外侧那一趟要等循环跑完、在循环外做一次。
class SegmentedBroadcastEmitter {
 public:
  SegmentedBroadcastEmitter(Operation *op, ConversionPatternRewriter &rewriter, const afir::VectorPlan &plan,
                            const BroadcastSegmentation &shape, ArrayRef<Value> extents, Value tmpBuffer,
                            const ascend::ExecutionSpec &executionSpec)
      : op_(op),
        rewriter_(rewriter),
        plan_(plan),
        shape_(shape),
        extents_(extents),
        tmpBuffer_(tmpBuffer),
        executionSpec_(executionSpec) {}

  LogicalResult emit(Value source, Value destination) {
    // 0 与 1 在这里建一次就够，而且必须在这里建：进了循环再建的常量支配不了循环之后
    // 的整块复制。这一刻的插入点在本档建的所有东西之外，所以它支配全部使用点。
    zero_ = rewriter_.create<arith::ConstantIndexOp>(loc(), 0);
    one_ = rewriter_.create<arith::ConstantIndexOp>(loc(), 1);
    return emitLevel(0, source, destination);
  }

 private:
  Location loc() const {
    return op_->getLoc();
  }

  // `segment` 那一段的元素数。`extents_` 覆盖 `[windowStart, segments.size())`。
  Value extentOf(size_t segment) const {
    return extents_[segment - shape_.windowStart];
  }

  LogicalResult call(Value destination, Value source, Value sourceM, Value sourceK, Value destinationM,
                     Value destinationK) {
    // `src_z` / `dst_z` 恒为 0。这不是省事：包装只有在 `src_k == 1 && src_z != 0` 时才走
    // `BroadcastMiddleDim`，而那是唯一会落到 `Or` 的那条路（`broadcast.h:79` / `:88`）。
    // 本档生成的两种形态一个 `src_m == 1`、一个 `src_z == 0`，都够不着它（功能需求 7）。
    Operation *call = rewriter_.create<ascendc::BroadcastDim3Op>(loc(), destination, source, sourceM, sourceK, zero_,
                                                                 destinationM, destinationK, zero_, tmpBuffer_);
    return setExecutionSpec(call, executionSpec_);
  }

  // 第 `level` 层：本层广播段之外的普通段变循环，内侧递归，最后生成本层的整块复制。
  LogicalResult emitLevel(size_t level, Value source, Value destination) {
    const size_t segment = shape_.broadcastSegments[level];
    Value innerSource = source;
    Value innerDestination = destination;
    if (level > 0) {
      // 本层循环覆盖上一个广播段与本段之间的普通段。第 0 层没有——它之外的段已经由
      // `materializeVectorizedSegments` 建成循环了。
      if (failed(buildLoops(shape_.broadcastSegments[level - 1] + 1, segment, innerSource, innerDestination))) {
        return failure();
      }
    }
    if (level + 1 == shape_.broadcastSegments.size()) {
      return emitWindowCall(innerSource, innerDestination);
    }

    // 内侧那几趟建在本层循环体里；建完要回到「本层循环体内、内层循环之后」，本层的整块
    // 复制在这里生成。`InsertPoint` 存的是 (block, iterator)，迭代器仍指着终结符，所以
    // 恢复之后落在内层循环之后、终结符之前。
    OpBuilder::InsertPoint afterInner = rewriter_.saveInsertionPoint();
    if (failed(emitLevel(level + 1, innerSource, innerDestination))) {
      return failure();
    }
    rewriter_.restoreInsertionPoint(afterInner);
    // 两趟都在 V 管线上，后一趟读的正是前一趟写的那块。必须本 pattern 自己生成：
    // `AFIRPipelineSync` 已在 conversion 之前跑完，它按 op 之间的边配对，给不出一个 op
    // 内部两趟之间的依赖。先例是现有 `Duplicate` 一档自己生成事件三件套。
    // 这一条测不出来——CPU 模拟器顺序执行，缺了不会红；lit 只能钉「生成了」。
    rewriter_.create<ascendc::PipeBarrierOp>(loc(), ascendc::Pipe::PIPE_V);
    return emitBlockCopy(segment, innerDestination);
  }

  // 最内那个广播段与它的邻段：一条两段调用。
  LogicalResult emitWindowCall(Value source, Value destination) {
    const size_t count = plan_.segments.size();
    Value outer = extentOf(count - 2);
    Value inner = extentOf(count - 1);
    //   形态 A `[brc K][normal Z]` → `Broadcast(dst, src, 1, Z, 0, K, Z, 0, tmp)` → `BroadcastFirstDim`
    //   形态 B `[normal M][brc K]` → `Broadcast(dst, src, M, 1, 0, M, K, 0, tmp)` → 原生 `Broadcast<T,2,1>`
    // 两种形态的 `dst_m` / `dst_k` 都是「外段 extent / 内段 extent」，差别只在源那两个。
    //
    // 尾轴真有填充时行长那一位生成的是行距，而且两侧一起换：包装要求
    // `srcShape[i] == dstShape[i]`，只换一边一个重载都对不上。`dst_k` 在包装里身兼行长与
    // 行距两职（`BroadcastFirstDim` 的 `dst_inner_offset`），换成行距之后行与行落在
    // `0, P, 2P, …`，正是带填充的真实布局；每行多算的那几个元素落进填充位，而填充位
    // don't care（`afir.store` 读这块 tile 时 UB 侧按行距、GM 侧按行长，逐行跳过它们）。
    //
    // 不给行距加尾块修正：尾块只让最后一趟的行长变短、不改行距，而它在这里当的就是
    // 行距。与 reg 侧「只有最外那一段带尾块修正」同一条规则。
    //
    // 两种形态都走得到这一支，而整个换掉 `inner` 对两者都对——差别在下面那行 `call`：
    // 形态 A 的 `sourceK` 也是 `inner`（两边一起换，包装要求如此），形态 B 的 `sourceK` 是
    // 字面量 `one_`（只换目的地一边）。判据那一侧的说明见 `classifySegments`。
    if (shape_.pitch) {
      FailureOr<Value> pitch = afir::materializeStrideValue(op_, rewriter_, shape_.pitch);
      if (failed(pitch)) {
        return failure();
      }
      inner = *pitch;
    }
    return call(destination, source, shape_.broadcastIsInner ? outer : one_, shape_.broadcastIsInner ? one_ : inner,
                outer, inner);
  }

  // 外层广播段的一趟：把内侧一整块复制 `extent - 1` 份。
  //
  // 偏移一整块 + repeat 减一——源区 `[0, blk)` 与写区 `[blk, extent × blk)` 正好错开，
  // 所以 `src == dst` 并不重叠。legacy 的四轴形态内部就是这么写的（`broadcast.h:537`）。
  //
  // `blk` 取的是**本段在目的地上的 stride**，不是内侧各段 extent 之积：两者在稠密布局下
  // 相等，而 stride 现成、积要在符号层再乘一遍；布局里真有填充时 stride 才是对的那个
  // （连填充一起复制，填充位本来就是 don't care）。
  LogicalResult emitBlockCopy(size_t segment, Value destination) {
    FailureOr<Value> block =
        afir::materializeStrideValue(op_, rewriter_, plan_.segments[segment].strides[shape_.destination]);
    if (failed(block)) {
      return failure();
    }
    Value repeats = rewriter_.create<arith::SubIOp>(loc(), extentOf(segment), one_);
    Value shifted = rewriter_.create<ascendc::LocalTensorSubIndexOp>(loc(), destination.getType(), destination, *block);
    return call(shifted, destination, one_, *block, repeats, *block);
  }

  // `[first, last)` 的每一段一层 `scf.for`，两个张量各偏移 `Σ iv × stride`。
  // 返回时插入点在最内层循环体内。
  LogicalResult buildLoops(size_t first, size_t last, Value &source, Value &destination) {
    SmallVector<size_t> segments;
    SmallVector<Value> inductionVars;
    for (size_t index = first; index < last; ++index) {
      auto loop = rewriter_.create<scf::ForOp>(loc(), zero_, extentOf(index), one_);
      segments.push_back(index);
      inductionVars.push_back(loop.getInductionVar());
      rewriter_.setInsertionPoint(loop.getBody()->getTerminator());
    }
    if (segments.empty()) {
      return success();
    }
    const std::pair<unsigned, Value *> tensors[] = {{shape_.source, &source}, {shape_.destination, &destination}};
    for (const auto &[tensorIndex, tile] : tensors) {
      Value offset;
      for (auto [segment, inductionVar] : llvm::zip(segments, inductionVars)) {
        FailureOr<Value> stride =
            afir::materializeStrideValue(op_, rewriter_, plan_.segments[segment].strides[tensorIndex]);
        if (failed(stride)) {
          return failure();
        }
        Value term = rewriter_.create<arith::MulIOp>(loc(), inductionVar, *stride).getResult();
        offset = offset ? rewriter_.create<arith::AddIOp>(loc(), offset, term).getResult() : term;
      }
      *tile = rewriter_.create<ascendc::LocalTensorSubIndexOp>(loc(), (*tile).getType(), *tile, offset);
    }
    return success();
  }

  Operation *op_;
  ConversionPatternRewriter &rewriter_;
  const afir::VectorPlan &plan_;
  const BroadcastSegmentation &shape_;
  ArrayRef<Value> extents_;
  Value tmpBuffer_;
  ascend::ExecutionSpec executionSpec_;
  // `emit` 建的，支配本档建的全部使用点。
  Value zero_;
  Value one_;
};

// capability 与 pattern 问的是同一件事：这个 op 是不是分段广播的形状。
// 两处各算一遍而不是缓存——与 `Duplicate` 一档的 `scalarReadEvent` 同一个理由。
FailureOr<BroadcastSegmentation> analyzeSegmentedBroadcast(Operation *op, afir::VectorPlan &plan) {
  // 这一档只收非常量源：源在每根向量化轴上都是 1 的那个形状归 `Duplicate` 一档，
  // 两档同时答「支持」会让选择变成先来后到。
  if (isConstantOnVectorizedAxes(firstDescriptor(op, afir::kOperandDescsAttrName))) {
    return failure();
  }
  FailureOr<afir::VectorPlan> collapsed = afir::collapseVectorizedAxes(op);
  if (failed(collapsed)) {
    return failure();
  }
  plan = std::move(*collapsed);
  return classifySegments(plan);
}

//===----------------------------------------------------------------------===//
// 分段广播，reg 侧（`BroadcastExtend`）
//===----------------------------------------------------------------------===//

// 这一档生成的是 reg 侧那张支持表上的 `BroadcastExtend<T, rank>`，**与 mem 侧那一档不是同一个
// 函数**：它吃整串向量化轴的 shape 数组、不吃 tmp buffer，合轴与循环分解由它自己在 tiling 里
// 做。所以这里既不建 `scf.for`，也不挑「最内两段」——形态判据换成「两个 tile 在这串段上都是
// 稠密的」。
//
// 元素类型：`broadcast_extend_impl.h:260` 的 `SupportBytes<T, 1, 2, 4, 8>` 收 b8/b16/b32/b64，
// AFIR 这一层编了号的七位全在里面——bf16 也在，它只是 b16，那条 `static_assert` 不区分
// 浮点格式。与 `kSegmentedElementTypes` 只差这一位，而那个差是真的：mem 侧那一档生成的是
// `broadcast.h` 的包装，它的分发链没有 bf16 那一支（见那个常量上面那段）。
//
// 这一位比 `ascir_op_lowering_matrix.csv` 的 `reg_dtypes` 列宽，legacy 的
// `REG_ASC_IR(Broadcast)` 两侧都没声明 bf16。按功能需求 3 的判定标准以头文件为准，证据与那道
// 门禁上的快照见 `dialect/AFIR/IR/Ops/Broadcast.td` 的 `AFIR_BroadcastElementTypes`。
constexpr uint32_t kExtendElementTypes = kF16 | kF32 | kI16 | kI32 | kUI8 | kUI16 | kBF16;

// `constRank` 的上限，`broadcast_extend_impl.h:258` 的 `static_assert` 钉的就是这个数。
// legacy 在生成调用之前也查同一个上限（`reg_broadcast_api_call.cpp:95`），只是它数的是
// 向量化轴根数，这里数的是合轴之后的段数——段数不多于轴数，所以这一条更松，不会把
// legacy 收的形状拒掉。
constexpr size_t kExtendMaxRank = 9;

// 一个 `afir.broadcast` 在 reg 侧的形态。
struct BroadcastExtendShape {
  // `VectorPlan::tensors` 里源与目的地的下标。
  unsigned source = 0;
  unsigned destination = 0;
  // 与 `plan.segments` 一一对应：本段上源不前进，`src_shape` 那一位写 1。
  SmallVector<bool> broadcast;
  // 最内那一位的 shape 改按行距传（不按段长传）时的那个行距；空值表示按段长传。
  //
  // 两侧共用这一份，不是各算一份。 `BroadcastExtendImpl` 折 shape 的时候，非广播维上
  // 要求 `srcShape[i] == dstShape[i]`（`broadcast_extend_impl.h:76`），两边各生成自己的行距
  // 就会在同一维上一个 8 一个 5、谁也不满足。所以取目的地那一份（它是被写的那块 tile,
  // 权威），源那一侧跟着生成同一个数：源 tile 的分配也是 32B 对齐的，按行距读会多读几个
  // 填充位、落到目的地的填充列上，那些位置本来就是 don't care（与 `emitBlockCopy`、
  // `materializeExtendShapeEntry` 已经在依赖的是同一条）。
  //
  // 只在行距与段长不相等（真有尾轴填充）时才非空：相等时两者是同一个数，传段长能让
  // 既有 62 份 CPU E2E 的生成文本逐字不变（段长那条路把多轴段的 extent 乘出来，行距是一个
  // 表达式，IR 不同而值相同）。为什么能改按行距传，见 `classifyDenseSide`。
  afir::ExprAttr pitch;
};

// 一侧张量在这串 shape 位上「稠密」吗。
//
// `BroadcastExtend` 把 shape 数组描述的张量当成行主序稠密的来读写，它表达不了跨度。但
// 「稠密」不等于「每一位都传段长」：最内那一位传行距（外一层的 stride）而不是行长，
// 描述出来的张量就把 32B 对齐补出来的填充列当成真数据，地址照样对得上——那些位置本来就是
// don't care。legacy 的 `GenParams` 生成的正是行距（`reg_broadcast_api_call.cpp`）。
//
// `positions` 是参与的位，外→内：目的地是全部段，源是**非广播段**（广播位传 1、不贡献跨度）。
// `shapeEntries` 是**真要生成的那几个数**，按段下标索引。**判据必须对着它们查，不是对着段长
// 查**——最内那一位可能生成的是行距，而两侧共用那一个数（见 `BroadcastExtendShape::pitch`）。
// 对着段长查会漏掉「源自己的行距与目的地的不一致」那一格：那时生成的 shape 描述的跨度不是
// 源真实的跨度，地址从第二行起就错。
//
// 两条判据，都不涉及除法：
//
//   - 最内那一位的 stride 为 1（它自己连续）；
//   - 相邻两个参与位满足 `stride[外] == shape[内] × stride[内]`。判据与合轴、与搬运偏移的
//     强度削减共用 `provablyContiguous`。
//
// 「不带填充」是其中的特例：那时行距恰好等于行长，这两条与老判据（累乘段长）等价。
bool sideIsDense(MLIRContext *context, const afir::VectorPlan &plan, ArrayRef<size_t> positions,
                 ArrayRef<afir::ExprAttr> shapeEntries, unsigned tensor) {
  if (positions.empty()) {
    return false;
  }
  if (!provablyLiteral(plan.segments[positions.back()].strides[tensor], "1")) {
    return false;
  }
  for (size_t index = positions.size(); index-- > 1;) {
    const afir::ExprAttr outerStride = plan.segments[positions[index - 1]].strides[tensor];
    const afir::ExprAttr innerStride = plan.segments[positions[index]].strides[tensor];
    if (!afir::provablyContiguous(context, outerStride, innerStride, shapeEntries[positions[index]])) {
      return false;
    }
  }
  return true;
}

// `plan` 是不是本档收的形状。
//
// 收的边界（不满足一律 `failure()`，调用方翻成 unsupported、整图回落 legacy）：
//
//   - 源与目的地各恰好一个；段数在 `[1, 9]` 内；至少一个广播段（一个都没有的不是广播，
//     调度会把那种节点删掉）；广播段上目的地必须前进（源与目的地都不前进 = 同一块写好
//     几遍，是个自相矛盾的 descriptor）。
//   - 两个 tile 在这串段上都稠密，判据与最内那一位该生成什么数一起由 `classifyDenseSide`
//     给出（源只算非广播段）。
//
// 带尾轴填充的 descriptor 现在收（2026-10-11）。此前这一档窄于 legacy：`kAligned` 下
// 次内层 stride 取 `Align(Z, 32/sizeof(T))` 而内段段长是 `Z`，老判据「`stride[外]` 等于其内
// 各段段长之积」当场不成立，整图回落。改法就是 legacy 一直在用的那一条——最内那一位传行距
// 而不是行长（见 `classifyDenseSide`）。这同时是 `RemovePad` 那条 32B 截断的前置：按行距传
// 之后 `dst_rpt_stride = 行距 × sizeof(T) / 32` 恒整除
// （`docs/development/2026-09-30-afir-remove-pad-design.md`「不做的事 ①」）。
LogicalResult identifyExtendSides(const afir::VectorPlan &plan, BroadcastExtendShape &shape) {
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

  return success();
}

FailureOr<BroadcastExtendShape> classifyExtendSegments(MLIRContext *context, const afir::VectorPlan &plan) {
  BroadcastExtendShape shape;
  if (failed(identifyExtendSides(plan, shape))) {
    return failure();
  }

  const size_t count = plan.segments.size();
  if (count == 0 || count > kExtendMaxRank) {
    return failure();
  }
  shape.broadcast.assign(count, false);
  bool anyBroadcast = false;
  for (auto [index, segment] : llvm::enumerate(plan.segments)) {
    if (segment.strides.size() != plan.tensors.size()) {
      return failure();
    }
    if (!provablyLiteral(segment.strides[shape.source], "0")) {
      continue;
    }
    if (provablyLiteral(segment.strides[shape.destination], "0")) {
      return failure();
    }
    shape.broadcast[index] = true;
    anyBroadcast = true;
  }
  if (!anyBroadcast) {
    return failure();
  }

  // 要生成的那几个数先定下来，判据对着它们查（理由见 `sideIsDense`）。
  SmallVector<afir::ExprAttr> shapeEntries;
  shapeEntries.reserve(count);
  for (const afir::VectorSegment &segment : plan.segments) {
    shapeEntries.push_back(segment.extent);
  }
  const size_t innermost = count - 1;
  if (count >= 2) {
    // 行距就是次内那一位在目的地上的 stride，不需要先验证什么就取得到。
    const afir::ExprAttr pitch = plan.segments[innermost - 1].strides[shape.destination];
    if (!provablyEqual(pitch, plan.segments[innermost].extent)) {
      // 真有尾轴填充。只收「最内那一段不是广播段」这一种形态：那时两侧的最内位是同一位，
      // 生成同一个行距才满足 `srcShape[i] == dstShape[i]`。最内那一段是广播段时，`src_shape`
      // 那一位是 1、源自己的最内参与位在更外面，两侧要对上的是另一对数——那是一个独立的
      // 形状，目前没有用例，按「宁可拒收」留给加得出用例的那次改动。
      if (shape.broadcast[innermost]) {
        return failure();
      }
      shape.pitch = pitch;
      shapeEntries[innermost] = pitch;
    }
  }

  // 两侧各查一次。目的地算全部段，源只算非广播段——广播位传 1、不贡献跨度。
  SmallVector<size_t> destinationPositions;
  SmallVector<size_t> sourcePositions;
  destinationPositions.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    destinationPositions.push_back(index);
    if (!shape.broadcast[index]) {
      sourcePositions.push_back(index);
    }
  }
  // 源全是广播段时没有非广播位：那说明源在每一根向量化轴上都不前进，也就是 `Duplicate`
  // 一档的形状，由那一档收（两档的判据因此不重叠）。
  if (sourcePositions.empty()) {
    return failure();
  }
  if (!sideIsDense(context, plan, destinationPositions, shapeEntries, shape.destination) ||
      !sideIsDense(context, plan, sourcePositions, shapeEntries, shape.source)) {
    return failure();
  }
  return shape;
}

// capability 与 pattern 问的是同一件事，各算一遍，理由同 `analyzeSegmentedBroadcast`。
FailureOr<BroadcastExtendShape> analyzeExtendBroadcast(Operation *op, afir::VectorPlan &plan) {
  // 与 mem 侧同一条分工：源在每根向量化轴上都是 1 的那个形状归 `Duplicate` 一档。
  if (isConstantOnVectorizedAxes(firstDescriptor(op, afir::kOperandDescsAttrName))) {
    return failure();
  }
  FailureOr<afir::VectorPlan> collapsed = afir::collapseVectorizedAxes(op);
  if (failed(collapsed)) {
    return failure();
  }
  plan = std::move(*collapsed);
  return classifyExtendSegments(op->getContext(), plan);
}

// 一段在 shape 数组里占的那个数。
//
// 只有最外那一段带尾块修正。 段 extent 在这一档身兼两职：最外那一段只当趟数用，
// 其余各段兼作跨度——`dst_shape[i]`（i > 0）既是那一维的长度，也是外一维相邻两份之间
// 的间距，而 tile 缓冲是按轴长分配的，最后一趟上行距并不变短。给内侧的段换上尾长会让
// `BroadcastExtend` 按一个比实际行距小的跨度走地址，算错。
//
// 这与 legacy 的 `GetOneAxisSize` 是同一条规则：切分轴不是最外那根向量化轴时取
// `tiler.Size`（轴长），是最外那根时取 `tiler.ActualSize`（带尾块修正）。按轴长生成的那一趟
// 会把填充位一起广播，那些位置本来就是 don't care。
//
// 「最外那一段」与「切分轴那一段」在这里是同一段，那是上游的不变量，本函数不查。
// `getTileInnerAxis` 取局部 descriptor `vectorized_axis` 的 `front()`，合轴保持这个次序，
// 所以切分轴恒在第 0 段。放宽那一条的人要回来给这里补一条判据：切分轴落到第 0 段之外时
// `index == 0` 这个条件就挑错段了，内侧的段会按一个比实际行距小的跨度走地址，静默算错。
FailureOr<Value> materializeExtendShapeEntry(Operation *op, OpBuilder &builder, const afir::VectorPlan &plan,
                                             const afir::TransferPlan &transferPlan, size_t index,
                                             Value tailAwareSize) {
  return afir::materializeSegmentExtent(op, builder, plan.segments[index], transferPlan.tileInnerAxis,
                                        index == 0 ? tailAwareSize : Value{});
}

class BroadcastCapabilityProvider final : public ascend::instruction::CapabilityProvider {
 public:
  explicit BroadcastCapabilityProvider(std::shared_ptr<const BroadcastMapping> mapping)
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

    return queryTensorContract(operation, *plan);
  }

  CapabilitySupport queryTensorContract(Operation *operation, const TransferPlan &plan) const {
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(operation);
    if (failed(partition) || partition->inputCount != 1 || partition->destinationCount != 1) {
      return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
    }
    Value source = partition->inputs(operation)[0];
    auto sourceType = dyn_cast<ascendc::LocalTensorType>(source.getType());
    auto destinationType = dyn_cast<ascendc::LocalTensorType>(partition->destinations(operation)[0].getType());
    if (!sourceType || !destinationType) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }
    // 广播不改 dtype——ODS 的 `verifyBroadcastShapeImpl` 已经钉住了，这里只查集合。
    if (!isSupportedElementType(mapping_->elementTypes, sourceType.getElementType()) ||
        sourceType.getElementType() != destinationType.getElementType()) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedType);
    }

    if (mapping_->tier == BroadcastTier::Duplicate) {
      // 这一档的判据：源在每一根向量化轴上都是 1。
      if (!isConstantOnVectorizedAxes(firstDescriptor(operation, afir::kOperandDescsAttrName))) {
        return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
      }
      if (!scalarReadEvent(operation, source)) {
        return CapabilitySupport::unsupported(CapabilityReason::MissingInformation);
      }
      // 这一档生成的是一次覆盖整块 tile 的填充，长度取 `materializeScheduledLocalSpan`，
      // 而那是首末元素之间的物理跨度（`1 + Σ (actual − 1) × stride`），不是元素个数。
      // 多段的 tile 上跨度大于元素数，差额正是行间填充；按跨度填会写进填充区，碰到并排
      // 摆在 UB 上的下一个张量。单段且局部 stride 为 1 时两者相等，所以必须同时检查。
      //
      // 这条判据不能照搬到 `DataMovement.cpp`。 那边走 `DataCopyPadExtend`，两侧行距
      // 是显式参数，缝隙本来就跳得过去，所以它不设这道闸——同一个形状在那边支持、在这里
      // 不支持，差别来自生成的是哪条 API，不是 tile 的几何。
      if (plan.kind == afir::TransferPlanKind::Scheduled && failed(afir::localTileIsSingleSegment(operation))) {
        return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
      }
      return CapabilitySupport::supported();
    }

    // 分段的两档。判不了的形态一律报不支持、整图回落 legacy，不猜测生成。
    afir::VectorPlan vectorPlan;
    if (mapping_->tier == BroadcastTier::Extend) {
      if (failed(analyzeExtendBroadcast(operation, vectorPlan))) {
        return CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape);
      }
      return CapabilitySupport::supported();
    }
    if (failed(analyzeSegmentedBroadcast(operation, vectorPlan))) {
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
  std::shared_ptr<const BroadcastMapping> mapping_;
};

class BroadcastDuplicateConversionPattern final : public OpConversionPattern<afir::BroadcastOp> {
 public:
  BroadcastDuplicateConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                      std::shared_ptr<const BroadcastMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(afir::BroadcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form broadcast requires a destination");
    }

    FailureOr<TransferPlan> plan = buildTransferPlan(op);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR transfer plan");
    }
    Value source = adaptor.getInput();
    Value destination = adaptor.getDestination();
    Value count;
    if (plan->kind == TransferPlanKind::Scheduled) {
      FailureOr<Value> span = materializeScheduledLocalSpan(op, rewriter, *plan);
      if (failed(span)) {
        return rewriter.notifyMatchFailure(op, "cannot materialize the scheduled local span");
      }
      count = *span;
      // 本地侧的偏移恒为 0，队列与 TBuf 都是——与 UnaryElementwise.cpp 同一条判据。
      Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      source = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), source.getType(), source, zeroOffset);
      destination =
          rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), destination.getType(), destination, zeroOffset);
    } else {
      count = rewriter.create<ascendc::LocalTensorGetSizeOp>(op.getLoc(), rewriter.getI32Type(), destination);
    }

    auto sourceType = dyn_cast<ascendc::LocalTensorType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "source is not a target-dialect tile");
    }
    auto scalar = materializeScalarRead(op, source, scalarReadEvent(op, op->getOperand(0)), rewriter);
    if (failed(scalar)) {
      return rewriter.notifyMatchFailure(op, "cannot synchronize scalar tensor read");
    }
    Operation *targetOp;
    if (sourceType.getElementType().isUnsignedInteger(kUInt8BitWidth)) {
      targetOp = rewriter.create<ascendc::DuplicateU8Op>(op.getLoc(), destination, *scalar, count);
    } else {
      targetOp = rewriter.create<ascendc::DuplicateL2Op>(op.getLoc(), destination, *scalar, count);
    }
    if (failed(setExecutionSpec(targetOp, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    rewriter.replaceOp(op, destination);
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BroadcastMapping> mapping_;
};

// 分段广播的 pattern（功能需求 3 / 4）。
//
// 三步：分段 → 外层段变循环 → 由内而外每个广播段一趟。前两步复用
// `collapseVectorizedAxes` 与 `materializeVectorizedSegments`（elementwise 用的是同一对，
// 只是 `apiWindowSize` 不同），第三步是本档自己的 `SegmentedBroadcastEmitter`。
class BroadcastSegmentedConversionPattern final : public OpConversionPattern<afir::BroadcastOp> {
 public:
  BroadcastSegmentedConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                      std::shared_ptr<const BroadcastMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(afir::BroadcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination() || !adaptor.getTmpBuffer()) {
      return rewriter.notifyMatchFailure(op, "segmented broadcast requires a destination and a tmp buffer");
    }
    FailureOr<TransferPlan> plan = buildTransferPlan(op);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR transfer plan");
    }
    afir::VectorPlan vectorPlan;
    FailureOr<BroadcastSegmentation> shape = analyzeSegmentedBroadcast(op, vectorPlan);
    if (failed(shape)) {
      return rewriter.notifyMatchFailure(op, "not a supported broadcast segmentation");
    }

    FailureOr<SmallVector<Value>> ordered =
        orderTilesForPlan(vectorPlan, {adaptor.getInput()}, adaptor.getDestination());
    if (failed(ordered)) {
      return rewriter.notifyMatchFailure(op, "cannot order the tiles the segmentation talks about");
    }
    // 窗口从最外那个广播段起：它之内的段本档要自己安排层次（循环与两趟的嵌套关系
    // 不是「所有循环建完再生成一条调用」），它之外的段是普通的外层循环。
    const auto windowSize = static_cast<unsigned>(vectorPlan.segments.size() - shape->windowStart);
    FailureOr<VectorizedWindow> window =
        materializeVectorizedSegments(op, rewriter, vectorPlan, *plan, *ordered, windowSize);
    if (failed(window) || window->tiles.size() != vectorPlan.tensors.size() || window->extents.size() != windowSize) {
      return rewriter.notifyMatchFailure(op, "cannot materialize the outer vectorized segments");
    }

    SegmentedBroadcastEmitter emitter(op, rewriter, vectorPlan, *shape, window->extents, adaptor.getTmpBuffer(),
                                      mapping_->executionSpec);
    if (failed(emitter.emit(window->tiles[shape->source], window->tiles[shape->destination]))) {
      return rewriter.notifyMatchFailure(op, "cannot emit the segmented broadcast calls");
    }
    // 结果是整块目的地 tile，不是最内那个窗口：窗口值定义在新建的循环体里，支配不了
    // 这个 op 的使用者（`afir.store` 在循环外）。
    rewriter.replaceOp(op, adaptor.getDestination());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BroadcastMapping> mapping_;
};

// reg 侧分段广播的 pattern。
//
// 一步：分段之后整串段直接落成两个 shape 数组。没有第二步——外层段不变循环、最内段
// 不挑窗口，那两件事 `BroadcastExtend` 在自己的 tiling 里做。
class BroadcastExtendConversionPattern final : public OpConversionPattern<afir::BroadcastOp> {
 public:
  BroadcastExtendConversionPattern(TypeConverter &typeConverter, MLIRContext *context, ImplementationKey key,
                                   std::shared_ptr<const BroadcastMapping> mapping)
      : OpConversionPattern(typeConverter, context), key_(std::move(key)), mapping_(std::move(mapping)) {}

  LogicalResult matchAndRewrite(afir::BroadcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = ascend::instruction::getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }
    if (!adaptor.getDestination()) {
      return rewriter.notifyMatchFailure(op, "buffer-form broadcast requires a destination");
    }
    // tmp buffer 不进调用：这一档的 `needsTmpBuffer` 是 false，调度给的块数由
    // `checkTmpBuffer` 查成 0，所以 `adaptor.getTmpBuffer()` 本来就该是空的。
    FailureOr<TransferPlan> plan = buildTransferPlan(op);
    if (failed(plan)) {
      return rewriter.notifyMatchFailure(op, "cannot build a supported AFIR transfer plan");
    }
    afir::VectorPlan vectorPlan;
    FailureOr<BroadcastExtendShape> shape = analyzeExtendBroadcast(op, vectorPlan);
    if (failed(shape)) {
      return rewriter.notifyMatchFailure(op, "not a supported broadcast segmentation");
    }

    // 尾长只有最外那一段用得上（见 `materializeExtendShapeEntry`），切分轴不落在那一段时
    // 连算都不算——算了就是一条没有使用者的 `afir.symbolic_eval` 留在函数里。
    Value tailAwareSize;
    if (plan->kind == TransferPlanKind::Scheduled &&
        llvm::is_contained(vectorPlan.segments.front().axes, plan->tileInnerAxis)) {
      FailureOr<Value> size = afir::materializeTailAwareTileSize(op, rewriter, *plan);
      if (failed(size)) {
        return rewriter.notifyMatchFailure(op, "cannot materialize the tail-aware tile size");
      }
      tailAwareSize = *size;
    }
    // 广播段至少有一个（判据里查过），所以这个 1 一定用得上，建在循环外让 IR 里的次序
    // 与数组次序无关。
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> destinationShape;
    SmallVector<Value> sourceShape;
    destinationShape.reserve(vectorPlan.segments.size());
    sourceShape.reserve(vectorPlan.segments.size());
    for (size_t index = 0; index < vectorPlan.segments.size(); ++index) {
      // 行距覆盖只在真有尾轴填充时非空。两侧都不覆盖时只物化一次、两边共用同一个值
      // ——这是既有 62 份生成文本逐字不变的那条路。
      FailureOr<Value> entry = materializeExtendShapeEntry(op, rewriter, vectorPlan, *plan, index, tailAwareSize);
      if (failed(entry)) {
        return rewriter.notifyMatchFailure(op, "cannot materialize a vectorized segment extent");
      }
      // 最内那一位带行距覆盖时两侧生成同一个值（判据保证那一段不是广播段）。
      Value entryValue = *entry;
      if (shape->pitch && index + 1 == vectorPlan.segments.size()) {
        FailureOr<Value> materialized = afir::materializeStrideValue(op, rewriter, shape->pitch);
        if (failed(materialized)) {
          return rewriter.notifyMatchFailure(op, "cannot materialize the tail row pitch");
        }
        entryValue = *materialized;
      }
      destinationShape.push_back(entryValue);
      sourceShape.push_back(shape->broadcast[index] ? one : entryValue);
    }

    // 本地侧的偏移恒为 0，队列与 TBuf 都是——与 `Duplicate` 一档同一条判据。
    Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value source = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), adaptor.getInput().getType(),
                                                                   adaptor.getInput(), zeroOffset);
    Value destination = rewriter.create<ascendc::LocalTensorSubIndexOp>(op.getLoc(), adaptor.getDestination().getType(),
                                                                        adaptor.getDestination(), zeroOffset);

    Operation *call = rewriter.create<ascendc::BroadcastExtendRegBaseOp>(op.getLoc(), destination, source,
                                                                         destinationShape, sourceShape);
    if (failed(setExecutionSpec(call, mapping_->executionSpec))) {
      return rewriter.notifyMatchFailure(op, "invalid AscendC execution spec mapping");
    }
    // 结果是整块目的地 tile，与 mem 侧那一档同一个理由：这一档虽然不建循环，但
    // `afir.store` 要的是整块，而不是偏移过的那个值。
    rewriter.replaceOp(op, adaptor.getDestination());
    return success();
  }

 private:
  ImplementationKey key_;
  std::shared_ptr<const BroadcastMapping> mapping_;
};

LogicalResult registerExtend(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family) {
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::BroadcastExtendRegBaseOp>(family))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(afir::BroadcastOp::getOperationName(), id, family, kExtendElementTypes);
  return registry.registerImplementation<afir::BroadcastOp, BroadcastMapping, BroadcastCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BroadcastMapping{family, BroadcastTier::Extend, kExtendElementTypes, /*needsTmpBuffer=*/false},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BroadcastMapping> mapping) {
        patterns.add<BroadcastExtendConversionPattern>(context.typeConverter, patterns.getContext(), key,
                                                       std::move(mapping));
      });
}

LogicalResult registerSegmented(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family) {
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::BroadcastDim3Op>(family))) {
    return failure();
  }
  elementwise::recordRegisteredFamily(afir::BroadcastOp::getOperationName(), id, family, kSegmentedElementTypes);
  return registry.registerImplementation<afir::BroadcastOp, BroadcastMapping, BroadcastCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BroadcastMapping{family, BroadcastTier::Segmented, kSegmentedElementTypes, /*needsTmpBuffer=*/true},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BroadcastMapping> mapping) {
        patterns.add<BroadcastSegmentedConversionPattern>(context.typeConverter, patterns.getContext(), key,
                                                          std::move(mapping));
      });
}

LogicalResult registerDuplicate(ImplementationRegistry &registry, llvm::StringRef id, ApiFamily family,
                                bool needsTmpBuffer) {
  if (failed(elementwise::checkFamilyMatchesHeader<ascendc::DuplicateL2Op>(family))) {
    return failure();
  }
  uint32_t elementTypes = kDuplicateElementTypes;
  if (family == ApiFamily::MemBased) {
    if (failed(elementwise::checkFamilyMatchesHeader<ascendc::DuplicateU8Op>(family))) {
      return failure();
    }
    elementTypes |= kUI8;
  }
  elementwise::recordRegisteredFamily(afir::BroadcastOp::getOperationName(), id, family, elementTypes);
  return registry.registerImplementation<afir::BroadcastOp, BroadcastMapping, BroadcastCapabilityProvider>(
      ascend::instruction::ImplementationKind::AscendC, ascend::instruction::ImplementationId{id.str()},
      BroadcastMapping{family, BroadcastTier::Duplicate, elementTypes, needsTmpBuffer},
      [](RewritePatternSet &patterns, const PatternBuildContext &context, const ImplementationKey &key,
         std::shared_ptr<const BroadcastMapping> mapping) {
        patterns.add<BroadcastDuplicateConversionPattern>(context.typeConverter, patterns.getContext(), key,
                                                          std::move(mapping));
      });
}

}  // namespace

LogicalResult registerBroadcastImplementations(ImplementationRegistry &registry,
                                               ascend::instruction::SelectionPolicyRegistry *) {
  // Native types use AscendC::Duplicate on both sides; mem-based ui8 uses the
  // legacy wrapper. Keep the existing family ids and their tmp-buffer contracts.
  if (failed(registerDuplicate(registry, "broadcast.tensor.duplicate", ApiFamily::MemBased,
                               /*needsTmpBuffer=*/true))) {
    return failure();
  }
  if (failed(registerDuplicate(registry, "broadcast.tensor.duplicate_reg_base", ApiFamily::RegBased,
                               /*needsTmpBuffer=*/false))) {
    return failure();
  }
  // 分段的两档两侧各一个，生成的是两个不同的函数：mem 侧是 `broadcast.h` 的包装，
  // reg 侧是 `BroadcastExtend<T, rank>`。两者的 tmp buffer 也不同侧（文件头）。
  if (failed(registerSegmented(registry, "broadcast.tensor.segmented", ApiFamily::MemBased))) {
    return failure();
  }
  return registerExtend(registry, "broadcast.tensor.extend_reg_base", ApiFamily::RegBased);
}

}  // namespace mlir::ascend::kernel
