/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ScheduleToAfir.cpp - ScheduledResult → AFIR (in-process) -*- C++ -*-===//

#include "common/tiling_schema/template_keys.h"
#include "bridge/ScheduleToAfir/ScheduleToAfir.h"
#include "support/AfirElementTypes.h"
#include "support/CapabilityShortfall.h"

#include <atomic>
#include <algorithm>
#include <charconv>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sys/stat.h>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "bridge/AfirBuilder/AfirBuilder.h"
#include "bridge/ScheduleToAfir/ReduceScratchAdapter.h"
#include "dialect/AFIR/Analysis/ResourceIds.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "support/CxxIdentifier.h"
#include "attribute_group/attr_group_shape_env.h"
#include "common_utils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"

namespace mlir {
namespace afir {
namespace bridge {

namespace detail {

// AFIR 的元素类型域有哪几位，权威定义在 `support/AfirElementTypes.h`。这里只剩两件 bridge
// 自己的事：`ge::DataType` 怎么对到那张表上，以及下面那张暂缓名单。
//
// 暂缓名单与支持表是两回事，改造之前它们挤在同一个 switch 里，于是立刻出事：头文件写着
// 「bf16 已经开了」，而这个函数对 `DT_BF16` 返回 `false`——两句话互相矛盾，谁都不会红。
// 分开写之后，「这一位建模了没有」由表回答，「建模了为什么还不放进来」由名单回答，而且
// 名单上每一行都必须带得出理由。

// 这张名单现在是空的。bf16 那一行 2026-10-22 撤掉，理由是量出来的，不是推断。
//
// 它原先写的是「搬运两端同 dtype 的 verifier 挡着，开闸会把 9 格干净回落变成硬失败」。
// 开闸实测（reg 侧接管表，`(bf16|bfloat16).*_codegen_v2`，28 格有目标）：
//
//     takeover 10 / hard-failure 12 / fallback 6
//
// 那条诊断确实在，12 格都是它——所以那一行拦得对，但拦的理由找错了地方：
//
//     'afir.store' op transfer source and destination must have the same element type
//
// 但首因不在搬运建模上，在 legacy 的用例样板上。 把 `cos_bf16`（硬失败）与
// `acos_bf16`（接管）的 `AutoFuseAfterOptimize` 图摆在一起，差别只有一处：
//
//   - 两张图的 Store 自己都是同 dtype 的（`Cast(f32→bf16) → Store(bf16→bf16)`，
//     `improve_precision` 的 `ProcessStoreNodes` 按预期插了那个 Cast）；
//   - `cos_bf16` 的 `Output` 节点落在缺省的 f32 上，而它与 Store 共用同一个 GM
//     `tensor_id`。于是那块 GM 被两处声明成两种 dtype，桥按 `Output` 建 GM 张量的元素
//     类型，verifier 看到的是 bf16 → f32。
//
// 源头是 `tests/framework/share_graph/src/share_graph.cc` 的建图函数漏了一行
// `y.y.dtype = af::DT_BF16;`。legacy codegen 从不读 `Output` 这一位，所以漏了也一直是绿的。
// 逐个查过 22 个 bf16 建图函数，这条判据一格不差：12 个硬失败的全缺那一行，10 个接管
// 的全有。（`add_exp_bf16` 一类也缺，但它们先被 VectorFunc 那道闸拦下，到不了 verifier。）
//
// 所以 `verifyTransfer` / `verifyTransferIndexingMaps` 是对的，不要去放宽它：那 12 张
// 图真的自相矛盾，放宽只会把硬失败换成静默按 f32 读一块 bf16 的内存。账收在用例样板那一侧
// ——那 12 个建图函数各补一行 `y.y.dtype`，再量：
//
//     takeover 18 / hard-failure 0 / fallback 10
//
// 12 格里 8 格直接翻成接管（acosh / ceil / cos / erf / log1p / pow / round / sign），另 4 格
// 变成干净回落，露出的是下一层另一个缺口：`afir.add`（add_floor / bf16_add）、
// `afir.nddma`、`afir.reciprocal` 这三个 op 在 bf16 上找不到 implementation，而
// `ascir_op_lowering_matrix.csv` 的 `reg_dtypes` 列里 legacy 三个都有 bf16。
// `ElementTypeMatrixTest` 的 `kKnownNarrower` 快照里没有它们，也就是说那道门禁本该报而没报
// ——为什么没报还没查清，别拿这段当已知结论用。
//
// 补那 12 行对 legacy 无影响，是量过的：12 个用例的第二阶段（`_e2e_v2`，编生成的 kernel 并
// 比对 expect_kernel）在 MLIR 全关（不设 `AF_ENABLE_MLIR_CODEGEN`，`ReadMlirCodegenOptions`
// 的 `enabled` 就是 false）下 12/12 过。legacy codegen 不读 `Output` 这一位，但这是实测，
// 不是据此推断。
//
// 新增一行的格式：支持表上那一位的 NAME / `ge::DataType` 的枚举名 / 一句话理由。理由必须
// 是量出来的首因——上一行就是败在「推断得很合理，但没人量过」。
#define AFIR_ELEMENT_TYPE_HELD_BACK(M)

// 这个 dtype AFIR 表达得了吗。判据与成因见头文件里的声明。
bool IsAfirModeledDType(ge::DataType dtype) {
  switch (dtype) {
// 表上有位的，缺省就是收。
#define AFIR_BRIDGE_MODELED_CASE(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH) case ge::GE:
    AFIR_ELEMENT_TYPE_TABLE(AFIR_BRIDGE_MODELED_CASE)
#undef AFIR_BRIDGE_MODELED_CASE
    break;
    default:
      return false;
  }
  // 暂缓名单上的，表上有位也先不放进来。每一行自带 `return`，不是共用后面一条——名单
  // 为空时这个 switch 必须退化成「只有 default」，而共用的那条会落在所有 case 标签之前，
  // 成为一句谁都到不了的死代码。
  switch (dtype) {
#define AFIR_BRIDGE_HELD_BACK_CASE(NAME, GE, REASON) \
  case ge::GE:                                       \
    return false;
    AFIR_ELEMENT_TYPE_HELD_BACK(AFIR_BRIDGE_HELD_BACK_CASE)
#undef AFIR_BRIDGE_HELD_BACK_CASE
    default:
      return true;
  }
}

// 名单可以是空的（今天就是），所以这里不能用 `HeldBackDType entries[]`——零长数组在 C++
// 里非法，名单清空的那一刻整个桥编不过。
llvm::ArrayRef<HeldBackDType> HeldBackDTypes() {
  static const std::vector<HeldBackDType> entries = {
#define AFIR_BRIDGE_HELD_BACK_ROW(NAME, GE, REASON) {afir::kAfirBit##NAME, REASON},
      AFIR_ELEMENT_TYPE_HELD_BACK(AFIR_BRIDGE_HELD_BACK_ROW)
#undef AFIR_BRIDGE_HELD_BACK_ROW
  };
  return entries;
}

// 桥这一层叫它什么。与 `IsAfirModeledDType` 是两个问题：这个答得出名字的比那个收的
// 多一截（i8 / i64 / ui32 / ui64），多出来的那些正是拒收诊断里要印的那个类型名。
std::optional<std::string> DTypeStr(ge::DataType dtype) {
  switch (dtype) {
#define AFIR_BRIDGE_DTYPE_NAME(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  case ge::GE:                                                                  \
    return BUILDER;
    AFIR_ELEMENT_TYPE_TABLE(AFIR_BRIDGE_DTYPE_NAME)
#undef AFIR_BRIDGE_DTYPE_NAME
#define AFIR_BRIDGE_DTYPE_NAME_ONLY(CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  case ge::GE:                                                            \
    return BUILDER;
    AFIR_ELEMENT_TYPE_NAMED_ONLY(AFIR_BRIDGE_DTYPE_NAME_ONLY)
#undef AFIR_BRIDGE_DTYPE_NAME_ONLY
    default:
      return std::nullopt;
  }
}

// axis type → AfirBuilder axis-type string.
std::optional<std::string> AxisTypeStr(af::Axis::Type type) {
  switch (type) {
    case af::Axis::Type::kAxisTypeOriginal:
      return "original";
    case af::Axis::Type::kAxisTypeBlockOuter:
      return "block_outer";
    case af::Axis::Type::kAxisTypeBlockInner:
      return "block_inner";
    case af::Axis::Type::kAxisTypeTileOuter:
      return "tile_outer";
    case af::Axis::Type::kAxisTypeTileInner:
      return "tile_inner";
    case af::Axis::Type::kAxisTypeMerged:
      return "merged";
    default:
      return std::nullopt;
  }
}

// exec_condition → AfirBuilder exec-condition string.
//
// `kConditionInvalid` 与任何将来新增的取值都落到 `nullopt`：上游那边它表示「还没定」，
// 翻成 `no_cache` 会把「没决定」说成「决定了不缓存」，两者在生成侧无法再区分。
std::optional<std::string> ExecConditionStr(af::ExecuteCondition condition) {
  switch (condition) {
    case af::ExecuteCondition::kNoCache:
      return "no_cache";
    case af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis:
      return "cache_block_split_fused_brc_axis";
    case af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis:
      return "cache_block_split_origin_brc_axis";
    default:
      return std::nullopt;
  }
}

}  // namespace detail

namespace {

// Expression (symbolic shape dim) → static int64. Non-constant dims become
// ShapedType::kDynamic (-9223372036854775808 sentinel used by MLIR); the demo
// keeps them dynamic rather than doing full symengine evaluation.
int64_t DimToStatic(const ge::Expression &e) {
  ge::Expression simplified = e.Simplify();
  int64_t v = 0;
  if (simplified.IsValid() && simplified.IsConstExpr() && simplified.GetConstValue<int64_t>(v)) {
    return v;
  }
  return INT64_MIN;  // mlir::ShapedType::kDynamic
}

std::optional<mlir::afir::Space> SpaceValue(af::Position position) {
  switch (position) {
    case af::Position::kPositionGM:
      return mlir::afir::Space::GM;
    case af::Position::kPositionVecIn:
    case af::Position::kPositionVecOut:
    case af::Position::kPositionVecCalc:
      return mlir::afir::Space::UB;
    default:
      return std::nullopt;
  }
}

// queue 还是 buffer 由 `alloc_type` 决定，不是由 position。两者正交：position 说的是
// TPosition（搬入 / 搬出 / 计算），alloc_type 说的是这块存储从队列还是从 TBuf 里取。
// 此前这里按 position 反推——「VecIn / VecOut 是队列，VecCalc 是 buffer」——两个计算
// 节点串联时就不成立了：中间张量的 position 是 VecCalc，而调度把它分进了一条队列。
std::optional<BindingDomain> DomainValue(af::AllocType allocType) {
  switch (allocType) {
    case af::AllocType::kAllocTypeQueue:
      return BindingDomain::Queue;
    case af::AllocType::kAllocTypeGlobal:
    case af::AllocType::kAllocTypeBuffer:
      return BindingDomain::Buffer;
    default:
      return std::nullopt;
  }
}

llvm::StringRef DomainName(BindingDomain domain) {
  return domain == BindingDomain::Queue ? "queue" : "buffer";
}

// 一个元素占几字节。答的范围比支持表宽，理由见
// `support/AfirElementTypes.h` 里 `AFIR_ELEMENT_TYPE_NAMED_ONLY` 的字节列那段。
std::optional<int64_t> ElementByteWidth(ge::DataType dtype) {
  switch (dtype) {
#define AFIR_BRIDGE_DTYPE_BYTES(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  case ge::GE:                                                                   \
    return BYTES;
    AFIR_ELEMENT_TYPE_TABLE(AFIR_BRIDGE_DTYPE_BYTES)
#undef AFIR_BRIDGE_DTYPE_BYTES
#define AFIR_BRIDGE_DTYPE_BYTES_ONLY(CSV, BUILDER, BYTES, GE, KIND, WIDTH) \
  case ge::GE:                                                             \
    return BYTES;
    AFIR_ELEMENT_TYPE_NAMED_ONLY(AFIR_BRIDGE_DTYPE_BYTES_ONLY)
#undef AFIR_BRIDGE_DTYPE_BYTES_ONLY
    default:
      return std::nullopt;
  }
}

// Build a tensor's shape directly from the scheduler's `repeats` — this faithfully
// carries the POST-tiling shape the scheduler produced (our ScheduleToAfir runs
// AFTER the scheduler). A load/store reshapes GM (logical rank) ↔ merged/tiled
// local, so operand and result ranks legitimately differ; the AFIR load/store
// verifier permits that. Tiling decisions ride on the func axes + per-op sched_axis
// for the M2 loop-building lowering.
std::string ExprStr(const ge::Expression &e) {
  if (!e.IsValid()) {
    return "<invalid>";
  }
  auto v = e.Simplify().Serialize();
  return v.get() == nullptr ? "<null>" : v.get();
}

mlir::LogicalResult ValidateScheduleBinding(const ScheduleBindingDTO &binding, mlir::Location loc) {
  auto checkOptionalId = [&](int64_t value, llvm::StringRef name) -> mlir::LogicalResult {
    if (value < -1) {
      mlir::emitError(loc) << "InvalidBinding: " << name << " must be -1 or non-negative";
      return mlir::failure();
    }
    return mlir::success();
  };
  if (binding.tensorId < 0) {
    mlir::emitError(loc) << "InvalidBinding: tensor_id must be non-negative";
    return mlir::failure();
  }
  if (mlir::failed(checkOptionalId(binding.queueId, "queue_id")) ||
      mlir::failed(checkOptionalId(binding.bufferId, "buffer_id")) ||
      mlir::failed(checkOptionalId(binding.reuseId, "reuse_id")) ||
      mlir::failed(checkOptionalId(binding.queueDepth, "queue_depth")) ||
      mlir::failed(checkOptionalId(binding.queueBufferCount, "queue_buffer_count"))) {
    return mlir::failure();
  }
  // `domain` 本身就是从 `alloc_type` 推出来的（见 `DomainValue`），所以这里不再拿
  // alloc_type 回头校验 domain——那是同义反复。剩下的是 queue 侧的字段完整性。
  const bool hasQueueFields = binding.queueDepth != -1 || binding.queueBufferCount != -1;
  if (binding.domain == BindingDomain::Queue) {
    if (binding.queueId == -1) {
      mlir::emitError(loc) << "InvalidBinding: queue domain requires queue_id";
      return mlir::failure();
    }
    if (binding.queueDepth < 1 || binding.queueBufferCount < 1) {
      mlir::emitError(loc) << "InvalidBinding: queue_id requires queue_depth and queue_buffer_count >= 1";
      return mlir::failure();
    }
  } else {
    if (binding.queueId != -1 || hasQueueFields) {
      mlir::emitError(loc) << "InvalidBinding: buffer domain cannot carry queue_id, queue_depth or queue_buffer_count";
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult ValidateTensorFacts(const TensorValueDTO &value, mlir::Location loc) {
  const auto &tensor = value.tensor;
  if (tensor.shape.size() != tensor.repeats.size() || tensor.shape.size() != tensor.shape_exprs.size()) {
    mlir::emitError(loc) << "InvalidDescriptor: shape, repeats and shape_exprs must have the same length";
    return mlir::failure();
  }
  if (!tensor.strides.empty() && tensor.strides.size() != tensor.shape.size()) {
    mlir::emitError(loc) << "InvalidDescriptor: strides must match tensor rank";
    return mlir::failure();
  }
  if (tensor.vectorized_axis.size() != tensor.vectorized_strides.size()) {
    mlir::emitError(loc) << "InvalidDescriptor: vectorized_axis and vectorized_strides must have the same length";
    return mlir::failure();
  }
  for (size_t index = 0; index < tensor.shape.size(); ++index) {
    if (tensor.shape[index] < 0 && tensor.shape[index] != mlir::ShapedType::kDynamic) {
      mlir::emitError(loc) << "InvalidDescriptor: shape dimension " << index << " is negative";
      return mlir::failure();
    }
    if (tensor.shape[index] == 0) {
      mlir::emitError(loc) << "InvalidDescriptor: shape dimension " << index << " must be positive";
      return mlir::failure();
    }
    if (tensor.shape_exprs[index].empty() || tensor.shape_exprs[index] == "<invalid>") {
      mlir::emitError(loc) << "InvalidDescriptor: shape expression " << index << " is invalid";
      return mlir::failure();
    }
  }
  return mlir::success();
}

}  // namespace

namespace detail {

std::optional<std::string> MakeSizeBytesExpr(ArrayRef<std::string> repeats, int64_t elementBytes) {
  if (elementBytes <= 0) {
    return std::nullopt;
  }
  int64_t constant = elementBytes;
  std::vector<std::string> symbolic;
  for (const auto &repeat : repeats) {
    if (repeat.empty()) {
      return std::nullopt;
    }
    long long value = 0;
    const auto parsed = std::from_chars(repeat.data(), repeat.data() + repeat.size(), value);
    if (parsed.ec == std::errc::result_out_of_range) {
      return std::nullopt;
    }
    if (parsed.ec == std::errc{} && parsed.ptr == repeat.data() + repeat.size()) {
      if (value <= 0) {
        return std::nullopt;
      }
      const int64_t factor = static_cast<int64_t>(value);
      if (constant > std::numeric_limits<int64_t>::max() / factor) {
        return std::nullopt;
      }
      constant *= factor;
    } else {
      symbolic.push_back(repeat);
    }
  }
  if (symbolic.empty()) {
    return std::to_string(constant);
  }
  std::string factors;
  for (const auto &factor : symbolic) {
    if (!factors.empty()) {
      factors += " * ";
    }
    factors += "(" + factor + ")";
  }
  if (constant == 1) {
    return factors;
  }
  return "(" + std::to_string(constant) + " * " + factors + ")";
}

}  // namespace detail

namespace {

std::string MaxSizeExpr(llvm::StringRef lhs, llvm::StringRef rhs) {
  if (lhs == rhs) {
    return lhs.str();
  }
  return "Max(" + lhs.str() + ", " + rhs.str() + ")";
}

std::optional<std::string> WorkspaceTensorSizeExpr(const af::AscTensor &tensor) {
  std::vector<std::string> repeats;
  repeats.reserve(tensor.attr.repeats.size());
  for (const auto &repeat : tensor.attr.repeats) {
    repeats.push_back(ExprStr(repeat));
  }
  auto elementBytes = ElementByteWidth(tensor.attr.dtype);
  if (!elementBytes) {
    return std::nullopt;
  }
  return detail::MakeSizeBytesExpr(repeats, *elementBytes);
}

mlir::LogicalResult AppendWorkspacePeerSizes(const af::AscNodePtr &node, const af::AscTensor &workspaceTensor,
                                             std::string &sizeBytesExpr, mlir::Location loc) {
  auto appendMaxSize = [&](const af::AscTensor &tensor) -> mlir::LogicalResult {
    std::optional<std::string> size = WorkspaceTensorSizeExpr(tensor);
    if (!size) {
      mlir::emitError(loc) << "InvalidWorkspace: workspace peer tensor has unsupported dtype or size";
      return mlir::failure();
    }
    sizeBytesExpr = MaxSizeExpr(sizeBytesExpr, *size);
    return mlir::success();
  };
  if (!node->inputs().empty()) {
    const af::AscTensor *inputTensor = node->inputs()[0];
    if (inputTensor == nullptr || mlir::failed(appendMaxSize(*inputTensor))) {
      return mlir::failure();
    }
  }
  for (auto &peerInput : workspaceTensor.anchor.GetPeerInDataAnchors()) {
    auto loadNode = std::dynamic_pointer_cast<af::AscNode>(peerInput->GetOwnerNode());
    if (!loadNode || loadNode->outputs().size() != 1 || loadNode->outputs()[0] == nullptr ||
        mlir::failed(appendMaxSize(*loadNode->outputs()[0]))) {
      mlir::emitError(loc) << "InvalidWorkspace: workspace consumer has no tensor output";
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::FailureOr<BufferDescDTO> ToWorkspaceDescriptor(const af::AscNodePtr &node, mlir::Location loc) {
  if (!node || node->GetType() != "Workspace" || node->outputs().size() != 1 || node->outputs()[0] == nullptr) {
    mlir::emitError(loc) << "InvalidWorkspace: workspace_nodes must contain one-output Workspace nodes";
    return mlir::failure();
  }
  const af::AscTensor &workspaceTensor = *node->outputs()[0];
  const auto &attr = workspaceTensor.attr;
  std::optional<mlir::afir::Space> space = SpaceValue(attr.mem.position);
  std::optional<int64_t> elementBytes = ElementByteWidth(attr.dtype);
  if (!space || !elementBytes || attr.mem.tensor_id < 0) {
    mlir::emitError(loc) << "InvalidWorkspace: workspace tensor has unsupported binding or dtype";
    return mlir::failure();
  }
  std::optional<std::string> sizeBytesExpr = WorkspaceTensorSizeExpr(workspaceTensor);
  if (!sizeBytesExpr) {
    mlir::emitError(loc) << "InvalidWorkspace: workspace size expression overflows int64_t";
    return mlir::failure();
  }
  if (mlir::failed(AppendWorkspacePeerSizes(node, workspaceTensor, *sizeBytesExpr, loc))) {
    return mlir::failure();
  }
  BufferDescDTO descriptor;
  descriptor.space = *space;
  descriptor.bufferId = attr.mem.tensor_id;
  descriptor.sizeBytesExpr = *sizeBytesExpr;
  descriptor.lifeTimeAxisId = -1;
  return descriptor;
}

mlir::LogicalResult AppendWorkspaceDescriptor(std::map<int64_t, BufferDescDTO> &workspaceByTensor,
                                              BufferDescDTO descriptor, mlir::Location loc) {
  const int64_t bufferId = descriptor.bufferId;
  const auto space = descriptor.space;
  const auto sizeBytesExpr = descriptor.sizeBytesExpr;
  auto [it, inserted] = workspaceByTensor.emplace(bufferId, std::move(descriptor));
  if (!inserted) {
    if (it->second.space != space) {
      mlir::emitError(loc) << "InvalidWorkspace: tensor " << bufferId << " has conflicting memory spaces";
      return mlir::failure();
    }
    it->second.sizeBytesExpr = MaxSizeExpr(it->second.sizeBytesExpr, sizeBytesExpr);
  }
  return mlir::success();
}

mlir::LogicalResult AppendWorkspacePlan(const ascir::FusedScheduledResult &fusedResult, PlanInfo &info,
                                        mlir::Location loc) {
  std::map<int64_t, BufferDescDTO> workspaceByTensor;
  for (const auto &node : fusedResult.workspace_nodes) {
    mlir::FailureOr<BufferDescDTO> descriptor = ToWorkspaceDescriptor(node, loc);
    if (mlir::failed(descriptor) ||
        mlir::failed(AppendWorkspaceDescriptor(workspaceByTensor, std::move(*descriptor), loc))) {
      return mlir::failure();
    }
  }
  info.workspace_plan.reserve(workspaceByTensor.size());
  for (auto &[tensorId, descriptor] : workspaceByTensor) {
    (void)tensorId;
    info.workspace_plan.push_back(std::move(descriptor));
  }
  return mlir::success();
}

mlir::FailureOr<TensorValueDTO> InitializeTensorValue(const af::AscTensorAttr &attr, mlir::afir::Role role,
                                                      mlir::Location loc, int64_t lifeTimeAxisId) {
  std::optional<std::string> dtype = detail::DTypeStr(attr.dtype);
  if (!dtype) {
    // 能力不足，不是坏了。 调度收的 dtype 比 AFIR 建模的多（`bool` 就是其中一个，
    // 它在这里是 12），这件事的性质是「还没建模」，legacy 收得了，应当整图干净回落。
    // 此前这里报的是硬失败：那让一批图在缺省那一档也红，而它们本可以回落。
    return mlir::afir::emitCapabilityShortfall(loc) << "UnsupportedOperation: scheduler dtype value "
                                                    << static_cast<int>(attr.dtype) << " is not modeled by AFIR";
  }
  if (!detail::IsAfirModeledDType(attr.dtype)) {
    // Bridge 叫得出名字不等于 AFIR 表达得了：bf16 这一类 `DTypeStr` 给得出规范拼写，
    // 但 ODS 的类型域收不了（`Elementwise.td` 的 `defvar` 与实现注册处那两份支持表），
    // 真正拦住它的是算子的类型约束——那是 verifier 失败，也就是硬失败。这件事同样
    // 是「还没建模」（功能需求 3 要补的就是它），所以在这里就报能力不足。
    return mlir::afir::emitCapabilityShortfall(loc)
           << "UnsupportedOperation: element type '" << *dtype << "' is not in the AFIR supply table";
  }
  std::optional<mlir::afir::Space> space = SpaceValue(attr.mem.position);
  if (!space) {
    mlir::emitError(loc) << "InvalidBinding: unsupported scheduler position " << static_cast<int>(attr.mem.position);
    return mlir::failure();
  }
  std::optional<BindingDomain> domain = DomainValue(attr.mem.alloc_type);
  if (!domain) {
    mlir::emitError(loc) << "InvalidBinding: unsupported scheduler alloc_type "
                         << static_cast<int>(attr.mem.alloc_type);
    return mlir::failure();
  }
  std::optional<int64_t> elementBytes = ElementByteWidth(attr.dtype);
  if (!elementBytes) {
    mlir::emitError(loc) << "InvalidDescriptor: cannot determine element byte width for dtype "
                         << static_cast<int>(attr.dtype);
    return mlir::failure();
  }
  TensorValueDTO value;
  value.tensor.dtype = std::move(*dtype);
  value.tensor.role = role;
  value.tensor.axis = attr.axis;
  value.schedule.tensorId = attr.mem.tensor_id;
  value.schedule.reuseId = attr.mem.reuse_id;
  value.schedule.queueId = *domain == BindingDomain::Queue ? attr.que.id : -1;
  value.schedule.queueDepth = *domain == BindingDomain::Queue ? attr.que.depth : -1;
  value.schedule.queueBufferCount = *domain == BindingDomain::Queue ? attr.que.buf_num : -1;
  value.schedule.bufferId = *domain == BindingDomain::Buffer ? attr.buf.id : -1;
  value.schedule.domain = *domain;
  value.buffer.space = *space;
  value.buffer.reuseId = attr.mem.reuse_id;
  value.buffer.bufferId = *domain == BindingDomain::Queue ? attr.que.id : attr.buf.id;
  value.buffer.lifeTimeAxisId = lifeTimeAxisId;
  if (*domain == BindingDomain::Queue) {
    value.buffer.slotCount = attr.que.buf_num;
    value.buffer.queueDepth = attr.que.depth;
  }
  if (mlir::failed(ValidateScheduleBinding(value.schedule, loc))) {
    return mlir::failure();
  }
  return value;
}

mlir::LogicalResult AppendTensorFacts(const af::AscTensorAttr &attr, TensorValueDTO &value, int64_t elementBytes,
                                      mlir::Location loc) {
  for (const auto &rep : attr.repeats) {
    value.tensor.shape.push_back(DimToStatic(rep));
    const auto expression = ExprStr(rep);
    value.tensor.repeats.push_back(expression);
    value.tensor.shape_exprs.push_back(expression);
  }
  for (const auto &stride : attr.strides) {
    value.tensor.strides.push_back(ExprStr(stride));
  }
  for (const auto &stride : attr.vectorized_strides) {
    value.tensor.vectorized_strides.push_back(ExprStr(stride));
  }
  value.tensor.vectorized_axis = attr.vectorized_axis;
  std::optional<std::string> sizeBytesExpr = detail::MakeSizeBytesExpr(value.tensor.repeats, elementBytes);
  if (!sizeBytesExpr) {
    mlir::emitError(loc) << "InvalidDescriptor: tensor byte size expression overflows int64_t";
    return mlir::failure();
  }
  // ATT computes the allocation size for internal UB resources and stores it
  // in the corresponding q<id>_size/b<id>_size tiling field.  The logical
  // tensor shape is still the correct size for external GM arguments, but
  // using it for a queue would allocate the whole tensor on every queue slot
  // instead of the scheduled tile.
  //
  // 片上资源因此只标记「大小取自本资源」，不再把字段名写进表达式：名字由消费侧按
  // `buffer_id` 派生（`afir::makeCanonicalResourceSizeName`），这里想指错也无处可写。
  if (value.schedule.domain == BindingDomain::Queue) {
    if (value.buffer.bufferId < 0) {
      mlir::emitError(loc) << "InvalidDescriptor: queue resource requires a non-negative queue id";
      return mlir::failure();
    }
    value.buffer.sizeFromResource = true;
  } else if (value.buffer.space == mlir::afir::Space::UB && value.buffer.bufferId >= 0) {
    value.buffer.sizeFromResource = true;
  } else {
    value.buffer.sizeBytesExpr = std::move(*sizeBytesExpr);
  }
  return ValidateTensorFacts(value, loc);
}

mlir::FailureOr<TensorValueDTO> ToTensorValueDTO(const af::AscTensorAttr &attr, mlir::afir::Role role,
                                                 mlir::Location loc, int64_t lifeTimeAxisId = -1) {
  mlir::FailureOr<TensorValueDTO> value = InitializeTensorValue(attr, role, loc, lifeTimeAxisId);
  if (mlir::failed(value)) {
    return mlir::failure();
  }
  const int64_t elementBytes = *ElementByteWidth(attr.dtype);
  if (mlir::failed(AppendTensorFacts(attr, *value, elementBytes, loc))) {
    return mlir::failure();
  }
  return value;
}

std::string ResultValueName(llvm::StringRef nodeName, size_t index, size_t count) {
  if (count == 1) {
    return nodeName.str();
  }
  return nodeName.str() + ".result" + std::to_string(index);
}

mlir::FailureOr<int64_t> GetIoIndex(const af::AscNodePtr &node, mlir::Location loc) {
  int64_t index = -1;
  if (!node->attr.ir_attr || node->attr.ir_attr->GetAttrValue("index", index) != af::GRAPH_SUCCESS || index < 0) {
    return mlir::emitError(loc) << "invalid IO index on node " << node->GetName();
  }
  return index;
}

mlir::FailureOr<std::vector<IoTensor>> ToIoTensors(const af::AscNodePtr &node, mlir::afir::Role role,
                                                   mlir::Location loc,
                                                   const std::map<int64_t, int64_t> &tmpBufferLifetimes) {
  std::vector<IoTensor> tensors;
  auto outputs = node->outputs();
  tensors.reserve(outputs.size());
  mlir::FailureOr<int64_t> ioIndex = GetIoIndex(node, loc);
  if (mlir::failed(ioIndex)) {
    return mlir::failure();
  }
  for (auto [index, output] : llvm::enumerate(outputs)) {
    int64_t lifeTimeAxisId = -1;
    if (output->attr.mem.position == af::Position::kPositionVecCalc &&
        output->attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
      auto lifetime = tmpBufferLifetimes.find(output->attr.buf.id);
      if (lifetime != tmpBufferLifetimes.end()) {
        lifeTimeAxisId = lifetime->second;
      }
    }
    mlir::FailureOr<TensorValueDTO> info = ToTensorValueDTO(output->attr, role, loc, lifeTimeAxisId);
    if (mlir::failed(info)) {
      return mlir::failure();
    }
    tensors.push_back({ResultValueName(node->GetName(), index, outputs.size()), std::move(*info), *ioIndex});
  }
  return tensors;
}

// `space` 必须进 key：`bufferId` 在这里是两个互不相干的 id 空间叠在一起的——UB 侧取的是
// 调度器的 `buf.id`，GM workspace 取的是张量 id（`mapTemplateWorkspaces` 就靠这个把一块
// GM 认成 workspace）。两者从 0 各数各的，撞号是常态而不是意外：一张 RCore 二相图里
// `workspace`（GM, tensor_id=2）与归约结果（UB, buf_id=2）就会落进同一个 key，被判成
// 「同一块资源上挂了两个不同视图」——一条假的 `BindingConflict`，整张图编不出来，而现场
// 指向的是一个根本不存在的冲突。
//
// 不改 `bufferId` 的取法：GM 侧那条语义是下降侧的契约（`afir.workspace_plan` 按 tensor id
// 对齐），改它要动的是另一半。分开的是同一个号在不同内存空间里不是同一块这件事。
struct ResourceKey {
  BindingDomain domain;
  mlir::afir::Space space;
  int64_t bufferId;
  int64_t reuseId;

  bool operator<(const ResourceKey &other) const {
    return std::tie(domain, space, bufferId, reuseId) <
           std::tie(other.domain, other.space, other.bufferId, other.reuseId);
  }
};

mlir::afir::CanonicalResourceFacts ToCanonicalResourceFacts(const BufferDescDTO &buffer) {
  mlir::afir::CanonicalResourceFacts facts;
  facts.space = buffer.space;
  facts.queueDepth = buffer.queueDepth;
  facts.lifeTimeAxisId = buffer.lifeTimeAxisId;
  facts.sizeBytesExpr = buffer.sizeBytesExpr;
  facts.sizeFromResource = buffer.sizeFromResource;
  return facts;
}

// 两张张量视图的差别能不能由广播解释。
//
// 广播轴本身必须是「一侧 repeat 1、stride 0」：那一侧在这根轴上不前进，片上 tile 因此
// 比整形状一侧小。非广播的外层轴也可能出现 stride 差异——广播发生在更内层时，小 tile 的
// 紧凑行距会跟着缩小。这里只放行“更内层确有广播轴”的行距差异，其余 stride 差异仍视为
// 两张真正对不上的视图。
//
// 调用侧按资源类型分流：队列上混排 broadcast tile 与 full-shape tile 是能力不足；非队列
// buffer 上二者可以按调度生命周期先后复用同一块存储。这里是资源视图的分类判据；最终地址
// 安全还由下降侧的 `sourceBroadcastsToDestination` 钉住，外层行距必须等于后续广播 extents
// 的乘积。
bool DifferenceIsExplainedByBroadcast(const TensorValueDTO &first, const TensorValueDTO &second) {
  // 轴本身对不上就不是广播能解释的了：广播只改 repeats 与 strides，不改轴的构成。
  if (first.tensor.axis != second.tensor.axis || first.tensor.vectorized_axis != second.tensor.vectorized_axis) {
    return false;
  }
  const size_t rank = first.tensor.strides.size();
  if (rank != second.tensor.strides.size() || first.tensor.repeats.size() != rank ||
      second.tensor.repeats.size() != rank) {
    return false;
  }

  auto isBroadcastSide = [](const std::string &repeat, const std::string &stride) {
    return repeat == "1" && stride == "0";
  };
  llvm::SmallVector<bool, 4> broadcastAxis(rank, false);
  bool sawBroadcastAxis = false;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (first.tensor.strides[axis] == second.tensor.strides[axis] &&
        first.tensor.repeats[axis] == second.tensor.repeats[axis]) {
      continue;
    }
    broadcastAxis[axis] = (isBroadcastSide(first.tensor.repeats[axis], first.tensor.strides[axis]) ||
                           isBroadcastSide(second.tensor.repeats[axis], second.tensor.strides[axis]));
    sawBroadcastAxis = sawBroadcastAxis || broadcastAxis[axis];
  }
  if (!sawBroadcastAxis) {
    return false;
  }

  for (size_t axis = 0; axis < rank; ++axis) {
    const bool sameStride = first.tensor.strides[axis] == second.tensor.strides[axis];
    const bool sameRepeat = first.tensor.repeats[axis] == second.tensor.repeats[axis];
    if (sameStride && sameRepeat) {
      continue;
    }
    if (broadcastAxis[axis]) {
      continue;
    }
    // 非广播轴的 stride 也可能不同：广播发生在更内层时，外层紧凑 tile 的行距会跟着
    // 缩小（例如 `[8,1,1]` 的 stride 0 是 1，而 `[8,128,1]` 的 stride 0 是 128）。
    // 行距差不是“后面有广播轴”就任意放行；full-shape 一侧的外层 stride 必须等于
    // 小 tile stride × 后续广播轴 repeat 的乘积。
    if (!sameRepeat || std::none_of(broadcastAxis.begin() + static_cast<ptrdiff_t>(axis) + 1, broadcastAxis.end(),
                                    [](bool value) { return value; })) {
      return false;
    }

    std::string firstExpected = first.tensor.strides[axis];
    std::string secondExpected = second.tensor.strides[axis];
    for (size_t broadcastAxisId = axis + 1; broadcastAxisId < rank; ++broadcastAxisId) {
      if (!broadcastAxis[broadcastAxisId]) {
        continue;
      }
      auto firstProduct = mlir::afir::MultiplyExpressions(firstExpected, second.tensor.repeats[broadcastAxisId]);
      auto secondProduct = mlir::afir::MultiplyExpressions(secondExpected, first.tensor.repeats[broadcastAxisId]);
      if (!firstProduct || !secondProduct) {
        return false;
      }
      firstExpected = firstProduct.value();
      secondExpected = secondProduct.value();
    }
    // 交叉比较：`firstExpected` 从 first 的紧凑行距推导出 second 应有的行距，
    // `secondExpected` 则相反。不能拿 expected 与自身的起点比较；广播侧的
    // repeat 为 1 时，那种写法会退化成恒真的自比较。
    if (!mlir::afir::ProveExpressionsEqual(second.tensor.strides[axis], firstExpected) &&
        !mlir::afir::ProveExpressionsEqual(first.tensor.strides[axis], secondExpected)) {
      return false;
    }
  }
  return true;
}

// 非队列资源上还有一种合法时间复用：同一逻辑 tile 的 padded / unpadded 两种向量视图。
// 典型 diamond 是 `brc -> RemovePad -> add`：broadcast 结果在 b2 上是 padded pitch
// （尾轴 5 补到 8），add 结果时间复用同一 b2，但是 unpadded pitch 5。二者的 axis、
// repeats、logical strides 完全一致，只有 vectorized_strides 不同。这是独立于 broadcast
// 的第二种合法时间复用；ATT 生成 `b<id>_size` 时按同 id 的资源需求取 max，MLIR 物化侧
// 再按各自 view.offset（这里均为 0）重解释同一块 raw storage。
bool SameLogicalTileWithDifferentVectorizedPitch(const TensorValueDTO &first, const TensorValueDTO &second) {
  if (first.tensor.axis != second.tensor.axis || first.tensor.vectorized_axis != second.tensor.vectorized_axis ||
      first.tensor.repeats != second.tensor.repeats || first.tensor.strides != second.tensor.strides) {
    return false;
  }
  return first.tensor.vectorized_strides != second.tensor.vectorized_strides;
}

mlir::LogicalResult CheckResourceCompatibility(const TensorValueDTO &first, llvm::StringRef firstOwner,
                                               const TensorValueDTO &second, llvm::StringRef owner,
                                               mlir::Location loc) {
  // descriptor 资源字段清单只在 `afir::canonicalResourceFactsAgree`：这里先查它，
  // 再查张量视图。凡是资源字段不同，都是真冲突，不能被任何视图差异“解释掉”。
  if (!mlir::afir::canonicalResourceFactsAgree(ToCanonicalResourceFacts(first.buffer),
                                               ToCanonicalResourceFacts(second.buffer))) {
    mlir::emitError(loc) << "BindingConflict: " << owner << " and " << firstOwner << " share resource domain "
                         << DomainName(first.schedule.domain) << " buffer_id=" << first.buffer.bufferId
                         << " reuse_id=" << first.buffer.reuseId << " has incompatible resource fields";
    return mlir::failure();
  }

  // Distinct queue members are packed into disjoint byte ranges by MemoryAllocate.
  // Their GM and UB views need not agree; repeated views of one tensor still do.
  if (first.schedule.domain == BindingDomain::Queue && first.schedule.tensorId != second.schedule.tensorId) {
    return mlir::success();
  }

  const bool sameTensorView =
      first.tensor.shape == second.tensor.shape && first.tensor.shape_exprs == second.tensor.shape_exprs &&
      first.tensor.repeats == second.tensor.repeats && first.tensor.axis == second.tensor.axis &&
      first.tensor.strides == second.tensor.strides && first.tensor.vectorized_axis == second.tensor.vectorized_axis &&
      (first.tensor.dtype != second.tensor.dtype ||
       first.tensor.vectorized_strides == second.tensor.vectorized_strides);
  if (sameTensorView) {
    return mlir::success();
  }

  // 以下检查处理同一 queue 成员的重复视图或非队列 buffer 的时间复用。非队列
  // `reuse_id = -1` 表示时间复用，成员偏移都是 0；资源容量由 ATT 的 `b<id>_size` 聚合，
  // 广播视图还必须能通过下降侧的外层行距判据。
  const bool isNonQueue = first.schedule.domain == BindingDomain::Buffer;
  const bool timeReusableView = DifferenceIsExplainedByBroadcast(first, second) ||
                                (isNonQueue && SameLogicalTileWithDifferentVectorizedPitch(first, second));
  if (timeReusableView) {
    if (isNonQueue) {
      return mlir::success();
    }
    mlir::afir::emitCapabilityShortfall(loc)
        << "UnsupportedOperation: " << owner << " resource domain " << DomainName(first.schedule.domain)
        << " buffer_id=" << first.buffer.bufferId << " reuse_id=" << first.buffer.reuseId
        << " carries a broadcast tile and a full-shape tile on one resource";
    return mlir::failure();
  }

  mlir::emitError(loc) << "BindingConflict: " << owner << " and " << firstOwner << " share resource domain "
                       << DomainName(first.schedule.domain) << " buffer_id=" << first.buffer.bufferId
                       << " reuse_id=" << first.buffer.reuseId << " has incompatible tensor layout";
  return mlir::failure();
}

struct TemplateResourceState {
  // 记下先占位的那个是谁：冲突诊断要同时点名两边，只说后来者等于让读的人自己去猜前一个。
  std::map<ResourceKey, std::pair<std::string, const TensorValueDTO *>> resources;
  std::map<std::pair<ResourceKey, int64_t>, std::pair<std::string, const TensorValueDTO *>> queueMembers;
};

mlir::LogicalResult ValidateTensorResource(const TensorValueDTO &value, const std::string &owner,
                                           TemplateResourceState &state, mlir::Location loc) {
  if (mlir::failed(ValidateScheduleBinding(value.schedule, loc)) || mlir::failed(ValidateTensorFacts(value, loc))) {
    return mlir::failure();
  }
  if (value.buffer.bufferId == -1) {
    return mlir::success();
  }
  ResourceKey key{value.schedule.domain, value.buffer.space, value.buffer.bufferId, value.buffer.reuseId};
  auto [it, inserted] = state.resources.emplace(key, std::make_pair(owner, &value));
  if (!inserted && mlir::failed(CheckResourceCompatibility(*it->second.second, it->second.first, value, owner, loc))) {
    return mlir::failure();
  }
  if (value.schedule.domain == BindingDomain::Queue) {
    auto [member, newMember] =
        state.queueMembers.emplace(std::make_pair(key, value.schedule.tensorId), std::make_pair(owner, &value));
    if (!newMember) {
      return CheckResourceCompatibility(*member->second.second, member->second.first, value, owner, loc);
    }
  }
  return mlir::success();
}

mlir::LogicalResult ValidateTemplateResources(const TemplateInfo &tmpl, mlir::Location loc) {
  TemplateResourceState state;
  for (const auto &input : tmpl.inputs) {
    if (mlir::failed(ValidateTensorResource(input.info, "template input '" + input.name + "'", state, loc))) {
      return mlir::failure();
    }
  }
  for (const auto &output : tmpl.outputs) {
    if (mlir::failed(ValidateTensorResource(output.info, "template output '" + output.name + "'", state, loc))) {
      return mlir::failure();
    }
  }
  for (const auto &node : tmpl.nodes) {
    for (const auto &input : node.inputs) {
      if (mlir::failed(ValidateTensorResource(input, "node '" + node.name + "' operand", state, loc))) {
        return mlir::failure();
      }
    }
    if (!node.outputs.empty()) {
      for (const auto &output : node.outputs) {
        if (mlir::failed(ValidateTensorResource(output, "node '" + node.name + "' result", state, loc))) {
          return mlir::failure();
        }
      }
    } else if (mlir::failed(ValidateTensorResource(node.output, "node '" + node.name + "' result", state, loc))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

std::string TemplateFuncName(size_t stageId, size_t templateId) {
  return "stage" + std::to_string(stageId) + "_template" + std::to_string(templateId);
}

llvm::ArrayRef<llvm::StringRef> PadApiTilingFields() {
  static constexpr llvm::StringRef fields[] = {
      "srcHeight",
      "srcWidth",
      "srcOriWidth",
      "widthWithoutLastBlock",
      "blocksPerRow",
      "heightTiling",
      "heightFractal",
      "heightFractalTail",
      "mainLoopOffset",
      "tailBlockOffset",
      "tmpBuffer1BlockNum",
      "tmpBuffer1RowNum",
      "tmpBuffer2Offset",
      "widthTiling",
      "widthFractal",
      "widthFractalTail",
      "widthFractalTailAlingned",
      "brcbTiling",
      "brcbFractal",
      "brcbFractalTail",
      "maxRepeatTimes",
      "brcbTilingRepeatTimes",
      "brcbTilingRepeatTimesTail",
      "brcbFractalTailRepeatTimes",
      "brcbFractalTailRepeatTimesTail",
      "reserved",
  };
  return fields;
}

std::optional<ApiTilingInfo> GetApiTilingInfo(llvm::StringRef nodeType, llvm::StringRef nodeName, size_t templateId) {
  // host 与 device 的字段名约定在 ATT 侧（api_tiling_gen/api/confusion_transpose.cpp 的
  // `ascgen_utils::GenValidName(node->GetName()) + "_tilingData_" + id`）。节点名可含 '/'
  // 等分隔符（如 "graph_hint/transpose"），`makeCxxIdentifier` 会把 '/' 编码成 "_x2f"，
  // 与 ATT 的单下划线替换不一致，host tiling_func 引用与 tiling struct 字段错位、host
  // 编译失败。两侧必须同名，这里跟随 ATT 的清洗规则。
  const std::string fieldName =
      ascgen_utils::GenValidName(nodeName.str() + "_tilingData_" + std::to_string(templateId));
  if (nodeType == "Transpose") {
    ApiTilingInfo info{"ConfusionTransposeTiling", fieldName, {}};
    for (int i = 0; i < 18; ++i) {
      info.fields.push_back("param" + std::to_string(i));
    }
    return info;
  }
  if (nodeType == "Pad") {
    ApiTilingInfo info{"PadTiling", fieldName, {}};
    for (llvm::StringRef field : PadApiTilingFields()) {
      info.fields.push_back(field.str());
    }
    return info;
  }
  return std::nullopt;
}

void SortAndUnique(std::vector<int64_t> &ids) {
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

template <typename GraphT>
TemplateInfo InitializeTemplateInfo(const GraphT &graph, const ascir::ScheduleGroup &group, size_t stageId,
                                    size_t templateId) {
  TemplateInfo info;
  info.name = TemplateFuncName(stageId, templateId);
  info.original_name = graph.GetName();
  info.tiling_key = common::TemplateKey(graph, templateId);
  info.stage_id = static_cast<int64_t>(stageId);
  info.template_id = static_cast<int64_t>(templateId);
  info.double_buffer = group.double_buffer;
  info.buffer_form = true;
  std::vector<af::AscNodePtr> workspaceNodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (node != nullptr && node->GetType() == "Workspace") {
      workspaceNodes.push_back(node);
    }
  }
  info.workspace_size = ExprStr(ascgen_utils::CalculateWorkspaceSize(workspaceNodes));
  if (auto iter = group.graph_name_to_score_funcs.find(graph.GetName());
      iter != group.graph_name_to_score_funcs.end()) {
    info.score_func = iter->second;
  }
  return info;
}

template <typename GraphT>
mlir::LogicalResult AppendTemplateAxes(const GraphT &graph, TemplateInfo &info, mlir::Location loc) {
  for (const auto &axis : graph.GetAllAxis()) {
    std::optional<std::string> axisType = detail::AxisTypeStr(axis->type);
    if (!axisType) {
      mlir::emitError(loc) << "unsupported scheduler axis type " << static_cast<int>(axis->type) << " on axis "
                           << axis->id;
      return mlir::failure();
    }
    info.axes.push_back({axis->id, axis->name, std::move(*axisType), ExprStr(axis->size), axis->bind_block,
                         ExprStr(axis->align), axis->from, axis->split_pair_other_id});
  }
  for (const auto &sizeVar : graph.GetAllSizeVar()) {
    info.size_vars.push_back(ExprStr(sizeVar->expr));
  }
  return mlir::success();
}

template <typename GraphT>
mlir::FailureOr<std::map<int64_t, int64_t>> CollectTmpBufferLifetimes(const GraphT &graph, mlir::Location loc) {
  std::map<int64_t, int64_t> tmpBufferLifetimes;
  for (const auto &scheduledNode : graph.GetAllNodes()) {
    for (const auto &tmpBuffer : scheduledNode->attr.tmp_buffers) {
      auto normalized = mlir::afir::bridge::translateSchedulerTmpBuffer(scheduledNode->GetType(), tmpBuffer.id,
                                                                        tmpBuffer.buf_desc.life_time_axis_id);
      if (!normalized) {
        mlir::emitError(loc) << "InvalidLifetime: scheduler tmp_buffer must have a non-negative id and a lifetime of "
                                "-1 or non-negative; "
                                "reduce scratch additionally requires a supported legacy role tag";
        return mlir::failure();
      }
      auto [it, inserted] = tmpBufferLifetimes.emplace(normalized->id, normalized->life_time_axis_id);
      if (!inserted && it->second != normalized->life_time_axis_id) {
        mlir::emitError(loc) << "InvalidLifetime: tmp_buffer " << tmpBuffer.id
                             << " has conflicting life_time_axis_id values";
        return mlir::failure();
      }
    }
  }
  return tmpBufferLifetimes;
}

// 逐个翻译一个 template 的节点时带着的上下文。前两张表按 tensor id 索引，边翻边长：
// 节点把自己的输出登记进去，后继节点从中查出自己输入的生产者名与描述符。后三项在整个
// template 内不变。
struct TemplateNodeContext {
  std::unordered_map<int64_t, std::string> producerByTensor;
  std::unordered_map<int64_t, TensorValueDTO> descriptorByTensor;
  const std::map<int64_t, int64_t> &tmpBufferLifetimes;
  size_t templateId;
  mlir::Location loc;
};

mlir::LogicalResult AppendNodeInputs(const af::AscNodePtr &node, NodeInfo &nodeInfo, const TemplateNodeContext &ctx) {
  mlir::Location loc = ctx.loc;
  for (const auto &input : node->inputs()) {
    auto producer = ctx.producerByTensor.find(input->attr.mem.tensor_id);
    if (producer == ctx.producerByTensor.end()) {
      mlir::emitError(loc) << "InvalidDescriptor: node '" << nodeInfo.name << "' input tensor "
                           << input->attr.mem.tensor_id << " has no producer";
      return mlir::failure();
    }
    nodeInfo.input_names.push_back(producer->second);
    auto descriptor = ctx.descriptorByTensor.find(input->attr.mem.tensor_id);
    if (descriptor == ctx.descriptorByTensor.end()) {
      mlir::emitError(loc) << "InvalidDescriptor: producer for tensor " << input->attr.mem.tensor_id
                           << " has no canonical descriptor";
      return mlir::failure();
    }
    nodeInfo.inputs.push_back(descriptor->second);
  }
  return mlir::success();
}

mlir::afir::Role NodeOutputRole(llvm::StringRef nodeType) {
  if (nodeType == "Data") {
    return mlir::afir::Role::Input;
  }
  if (nodeType == "Output" || nodeType == "Store") {
    return mlir::afir::Role::Output;
  }
  return mlir::afir::Role::Compute;
}

mlir::LogicalResult AppendNodeOutputs(const af::AscNodePtr &node, NodeInfo &nodeInfo, TemplateNodeContext &ctx) {
  mlir::Location loc = ctx.loc;
  auto outputs = node->outputs();
  if (outputs.empty()) {
    return mlir::success();
  }
  mlir::afir::Role role = NodeOutputRole(nodeInfo.type);
  nodeInfo.outputs.reserve(outputs.size());
  nodeInfo.output_names.reserve(outputs.size());
  for (auto [index, outputAnchor] : llvm::enumerate(outputs)) {
    int64_t lifeTimeAxisId = -1;
    if (outputAnchor->attr.mem.position == af::Position::kPositionVecCalc &&
        outputAnchor->attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
      auto lifetime = ctx.tmpBufferLifetimes.find(outputAnchor->attr.buf.id);
      if (lifetime != ctx.tmpBufferLifetimes.end()) {
        lifeTimeAxisId = lifetime->second;
      }
    }
    mlir::FailureOr<TensorValueDTO> output = ToTensorValueDTO(outputAnchor->attr, role, loc, lifeTimeAxisId);
    if (mlir::failed(output)) {
      return mlir::failure();
    }
    // GM 张量没有队列/缓冲编号，`bufferId` 因此恒为 -1。workspace 是唯一的例外：同一块
    // GM 要被两个 stage 各自认出来（phase 1 写、phase 2 读），认的依据就是 tensor id。
    // plan 级的 `ToWorkspaceDescriptor` 早已按这个规则写 `descriptor.bufferId`，这里把
    // 节点级对齐过去，两处说的才是同一件事。
    if (nodeInfo.type == "Workspace") {
      output->buffer.bufferId = outputAnchor->attr.mem.tensor_id;
    }
    nodeInfo.outputs.push_back(std::move(*output));
    nodeInfo.output_names.push_back(ResultValueName(nodeInfo.name, index, outputs.size()));
    ctx.producerByTensor[outputAnchor->attr.mem.tensor_id] = nodeInfo.output_names.back();
    ctx.descriptorByTensor[outputAnchor->attr.mem.tensor_id] = nodeInfo.outputs.back();
  }
  nodeInfo.output = nodeInfo.outputs.front();
  return mlir::success();
}

mlir::LogicalResult AppendNodeIo(const af::AscNodePtr &node, NodeInfo &nodeInfo, TemplateInfo &info,
                                 const TemplateNodeContext &ctx) {
  // `ScalarData` 与 `Data` 一样是 kernel 输入，序号空间也是同一个（legacy 的 `ParseGraph`
  // 把两者放在同一个按 `index` 排形参的循环里）。差别只在它按值传，由 `by_value` 记。
  const bool isScalarData = nodeInfo.type == "ScalarData";
  if (nodeInfo.type == "Data" || nodeInfo.type == "Output" || isScalarData) {
    mlir::afir::Role role = nodeInfo.type == "Output" ? mlir::afir::Role::Output : mlir::afir::Role::Input;
    mlir::FailureOr<std::vector<IoTensor>> tensors = ToIoTensors(node, role, ctx.loc, ctx.tmpBufferLifetimes);
    if (mlir::failed(tensors)) {
      return mlir::failure();
    }
    for (IoTensor &tensor : *tensors) {
      tensor.by_value = isScalarData;
    }
    auto &target = nodeInfo.type == "Output" ? info.outputs : info.inputs;
    target.insert(target.end(), std::make_move_iterator(tensors->begin()), std::make_move_iterator(tensors->end()));
  }
  return mlir::success();
}

// `Scalar` 节点的值在 AscGraph 上是一个字符串属性（legacy 把它整串拼进生成的 C++）。
// 这里只把那串文本原样带过 bridge，解析交给 AfirBuilder——它才知道要解析成哪个 MLIR
// 类型。带不过来就报错：一个没有值的 Scalar 节点没有可用的含义。
mlir::LogicalResult ReadScalarValue(const af::AscNodePtr &node, NodeInfo &nodeInfo, mlir::Location loc) {
  if (node->attr.ir_attr == nullptr) {
    return mlir::emitError(loc) << "UnsupportedOperation: Scalar node '" << nodeInfo.name << "' has no attributes";
  }
  std::string value;
  if (node->attr.ir_attr->GetAttrValue("value", value) != ge::GRAPH_SUCCESS) {
    return mlir::emitError(loc) << "UnsupportedOperation: Scalar node '" << nodeInfo.name << "' has no 'value'";
  }
  nodeInfo.scalar_value = std::move(value);
  return mlir::success();
}

// 节点自有的 float `ir_attr`（`LeakyRelu` 的 `negative_slope`、`Axpy` 的 `alpha`）。
// AscGraph 上是 `ir_attr`，AFIR 上是 op 属性，与 `Scalar` 的 `value` 同类——节点自己的
// 一个数，不是操作数、不占缓冲——但那边是字符串字面量、这边是 float，所以各读各的。
//
// 读哪个名字由 `getFloatIrAttrMappings()` 那张表说，AfirBuilder 按同一行挂成 AFIR 上的
// 属性名。两边的拼写不保证一致(`negative_slope` → `negative_slop`)，所以表里两个
// 名字都在。
// `IndexExpr` 节点的表达式。与 `ReadScalarValue` 同形——都只把节点自己的那个值带过
// bridge，解析交给 AfirBuilder——但这边带的是一条按 size var 写的表达式，不是字面量，
// 所以转成 AFIR 的规范串（`ExprStr`）而不是原样拷贝。
mlir::LogicalResult ReadIndexExpr(const af::AscNodePtr &node, NodeInfo &nodeInfo, mlir::Location loc) {
  if (node->attr.ir_attr == nullptr) {
    return mlir::emitError(loc) << "UnsupportedOperation: IndexExpr node '" << nodeInfo.name << "' has no attributes";
  }
  ge::Expression expr;
  if (node->attr.ir_attr->GetAttrValue("expr", expr) != ge::GRAPH_SUCCESS || !expr.IsValid()) {
    return mlir::emitError(loc) << "UnsupportedOperation: IndexExpr node '" << nodeInfo.name << "' has no 'expr'";
  }
  nodeInfo.index_expr = ExprStr(expr);
  return mlir::success();
}

mlir::LogicalResult ReadFloatIrAttr(const af::AscNodePtr &node, const mlir::afir::bridge::FloatIrAttrMapping &mapping,
                                    NodeInfo &nodeInfo, mlir::Location loc) {
  float value = 0.0F;
  if (node->attr.ir_attr == nullptr ||
      node->attr.ir_attr->GetAttrValue(std::string(mapping.ascgraph_attr_name), value) != af::GRAPH_SUCCESS) {
    return mlir::emitError(loc) << "UnsupportedOperation: " << nodeInfo.type << " node '" << nodeInfo.name
                                << "' has no '" << mapping.ascgraph_attr_name << "' attribute";
  }
  nodeInfo.float_ir_attr = value;
  return mlir::success();
}

// int64 那一半。与 `ReadFloatIrAttr` 逐项同形，只有值的类型不同——两条路的落点不同
// （一个成为 `F32FloatAttr`、一个成为 `I64IntegerAttr`，后者还要走到模板实参位上），
// 所以各读各的那张表。
mlir::LogicalResult ReadIntIrAttr(const af::AscNodePtr &node, const mlir::afir::bridge::IntIrAttrMapping &mapping,
                                  NodeInfo &nodeInfo, mlir::Location loc) {
  int64_t value = 0;
  if (node->attr.ir_attr == nullptr ||
      node->attr.ir_attr->GetAttrValue(std::string(mapping.ascgraph_attr_name), value) != af::GRAPH_SUCCESS) {
    return mlir::emitError(loc) << "UnsupportedOperation: " << nodeInfo.type << " node '" << nodeInfo.name
                                << "' has no '" << mapping.ascgraph_attr_name << "' attribute";
  }
  nodeInfo.int_ir_attr = value;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// 二次融合（`VectorFunc`）的嵌套子图
//===----------------------------------------------------------------------===//
//
// 这个节点类型不在 bridge 的算子注册表里（`AFIR_BRIDGE_OPERATION_REGISTRY`），因为它
// 不是「一个节点翻一个算子」那种形状：它翻出来的是一个带 region 的算子，region 里还有
// 一张子图。注册表里的通用路径（`buildBufferNodeOp` 的 `make(OP{})`）给它建出来的会是
// 一个没有 region 的空壳，所以两侧都走各自的专门分支：这里收集，`AfirBuilder` 构建。
constexpr const char *kVectorFuncNodeType = "VectorFunc";

// 子图内的计算节点与父图共用同一条翻译路径，所以要前向声明它。
mlir::LogicalResult AppendTemplateNode(const af::AscNodePtr &node, TemplateInfo &info, TemplateNodeContext &ctx);

// 子图的边界要在这一层解掉，不能原样翻成 AFIR 算子。
//
// 子图的边界形状是 `Data` + `Load`（入）、`Store` + `Output`（出），这两对表达的是
// UB <-> 寄存器；父图里的 `afir.load` / `afir.store` 表达的是 GM <-> UB。两者同名
// 不同义，原样翻过去会多出两趟搬运，而且两种都编得过、都算得对（只是多搬一趟），
// 没人会报错——所以这一处必须在进 MLIR 之前转换掉。
//
// 产物是两张按序号配对的边界表（见 `VectorFuncBodyDTO`）：序号取调度给的
// `vf_root_input_index` 与 `Output` 的 `index`，**不按容器顺序、不按名字猜**。
template <typename GraphT>
mlir::LogicalResult AppendVectorFuncBody(const af::AscNodePtr &vfNode, NodeInfo &vfInfo, const GraphT &graph,
                                         const TemplateNodeContext &parentCtx) {
  mlir::Location loc = parentCtx.loc;
  const std::string *subGraphName = af::AttrUtils::GetStr(vfNode->GetOpDescBarePtr(), "sub_graph_name");
  if (subGraphName == nullptr) {
    // 图自称有二次融合却没说子图在哪 —— **产生方坏了，不是能力不足**：回落会把它藏起来。
    return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc node '" << vfInfo.name
                                << "' carries no sub_graph_name attribute";
  }
  af::AscGraph subGraph("af_vector_func_body");
  if (graph.FindSubGraph(*subGraphName, subGraph) != af::SUCCESS) {
    return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc node '" << vfInfo.name << "' names subgraph '"
                                << *subGraphName << "', which is not on this graph";
  }

  // **dtype 白名单，M1 只收 f16 / f32。**
  //
  // 这一条不是「这一期懒得做」，它是在守一个不变量：VF 这道门禁此前替一批图挡着，
  // 那些图里还有别的没接的东西（bf16 / i8 / ui8 / bool 的类型域、1 维形状的 load 下降）。
  // 放它们进来之后，拦住它们的是 ODS 的类型约束与 conversion——那两种都是硬失败，
  // 而硬失败会把本来干净回落的图变成红的。实测：不加这一条，reg 侧接管表的
  // hard-failure 从 79 涨到 89。
  //
  // 判据按边界与子图节点的 dtype 来，而不是按「这个 op 支不支持」：dtype 的缺口在
  // AFIR 的支持表上（功能需求 3），与 VF 无关；VF 这一侧能做的是不要把它们放进来。
  //
  // 这道门禁今天仍然必要，而且理由要收窄。 2026-10-20 那一轮的说法是「bf16 / uint8 /
  // int16 卡的是同一个缺口，放开后都变硬失败」，依据是那条
  //
  //     'afir.store' op transfer source and destination must have the same element type
  //
  // 但那一轮量在 `6a525700`（补 12 个 bf16 用例样板漏设的 `Output` dtype）之前，所以
  // 那个数不能直接用。2026-10-26 把 bf16 / uint8 放开重量了一遍，结论是部分成立——
  // 接管表上「只差 VF」那 8 格，放开后各自落在：
  //
  //   | 用例 | 放开后 | 真实首因 |
  //   |---|---|---|
  //   | `abs_bf16` / `add_exp_bf16` / `truediv_bf16` / `relu_uint8` | **硬失败** | 上面那条 `afir.store` |
  //   | `abs_fma_bf16` | 回落 | `Fma` 这个节点类型 bridge 没接（功能需求 5） |
  //   | `abs_uint8` | 回落 | VF 边界上还有 **int16** 这一位 |
  //   | `abs_brc_add` / `load_compare_scalar_where_store` | 回落 | VF 子图里的 `Scalar` 没建模 |
  //
  // 所以：4 格确实是「搬运能不能换 dtype」，那一条修好它们才进得来，这道门禁替它们挡着
  // 硬失败；另外 2 格的卡点压根不是 dtype（一个是 op、一个是 int16），拿这道门禁当它们的
  // 原因会把活排错层。
  //
  // 订正（2026-10-28）：`abs_fma_bf16` 那一行只写对了第一因。 `Fma` 接通之后实测它不是
  // 回落而是硬失败，诊断就是上面那条 `afir.store`——legacy 调度出的签名是
  // `%output: f32`，而 store 的源经 `DtypeConsistency` 插回的 Cast 是 bf16。所以它属于上表
  // 第一行那一类（4 格变 5 格），不属于「卡点不是 dtype」那两格；接 `Fma` 这件事已因此退回，
  // 见工作总表 step 3 那一节。
  //
  // 探针那一侧的剖面与接管表不同，别混着读：同样放开之后探针 reg 侧 `ok` +17、硬失败
  // +5，而那 5 格的诊断是 `DescriptorCountMismatch: 'afir.destination_descs' expected 2
  // element but found 1`——全是 T5c（菱形两消费者）融出来的双输出 VF，bridge 只挂了一个
  // destination desc。那是 VF 这条路自己的缺陷，与搬运换 dtype 无关。
  const auto isModeledDtype = [](const std::string &dtype) {
    return dtype == "float32" || dtype == "float16" || dtype == "bfloat16" || dtype == "uint8" || dtype == "int16";
  };
  for (const TensorValueDTO &boundary : vfInfo.inputs) {
    if (!isModeledDtype(boundary.tensor.dtype)) {
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc node '" << vfInfo.name << "' reads '" << boundary.tensor.dtype
             << "', which the register-form lowering does not model yet";
    }
  }
  for (const TensorValueDTO &boundary : vfInfo.outputs) {
    if (!isModeledDtype(boundary.tensor.dtype)) {
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc node '" << vfInfo.name << "' writes '" << boundary.tensor.dtype
             << "', which the register-form lowering does not model yet";
    }
  }

  auto body = std::make_shared<VectorFuncBodyDTO>();
  body->boundary_input_names.assign(vfInfo.input_names.size(), std::string());
  body->boundary_output_names.assign(vfInfo.outputs.size(), std::string());

  TemplateInfo bodyInfo;
  // 子图内是 tensor 形态：没有 destination，也就没有存储——VF 的中间结果在寄存器里，
  // 一块 UB 都没分配过。
  bodyInfo.buffer_form = false;
  TemplateNodeContext bodyCtx{{}, {}, parentCtx.tmpBufferLifetimes, parentCtx.templateId, loc};

  // 边界靠边配对，不靠 tensor id。 分区器是成对插入 `Data` + `Load` 的
  // （`InsertDataAndLoadNode`），而那个 `Data` 的输出 attr 只设了 dtype——**没有
  // tensor id；`Load` 的输出 attr 则是从父图的生产者整份抄过来的，所以带的是父图**
  // 的 tensor id。于是「按 id 把 Data 和 Load 串起来」这条路走不通（实测 src #0 找不到
  // 边界就是这么来的），而「Data 的唯一消费者就是它那个 Load」是分区器的结构保证。
  llvm::DenseSet<const void *> boundaryLoads;
  for (const auto &subNode : subGraph.GetAllNodes()) {
    if (subNode == nullptr || subNode->GetType() != "Data") {
      continue;
    }
    int64_t parentIndex = 0;
    if (!af::AttrUtils::GetInt(subNode->GetOpDesc(), "vf_root_input_index", parentIndex)) {
      // 分区器建边后才写 `vf_root_input_index`；缺失说明这张子图没经过分区器（手工构造的
      // fixture），回退到子图自己的边界序号 `index`——legacy 的 `ParseSubGraph` 也是这么
      // 回退的，对「按序连接边界」的子图语义等价。
      mlir::FailureOr<int64_t> boundaryIndex = GetIoIndex(subNode, loc);
      if (mlir::failed(boundaryIndex)) {
        return mlir::failure();
      }
      parentIndex = *boundaryIndex;
    }
    // 外来下标作数组下标之前必须校验（编码红线 2）。越界是产生方与这里对不上，属坏了。
    if (parentIndex < 0 || static_cast<size_t>(parentIndex) >= body->boundary_input_names.size()) {
      return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc subgraph boundary '" << subNode->GetName()
                                  << "' maps to parent input " << parentIndex << ", which is out of range ("
                                  << body->boundary_input_names.size() << " srcs)";
    }

    const auto consumers = subNode->GetOutDataNodes();
    if (consumers.size() != 1U) {
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc subgraph boundary '" << subNode->GetName() << "' feeds "
             << consumers.size() << " nodes; the bridge models the single boundary load the partitioner inserts";
    }
    auto loadNode = std::dynamic_pointer_cast<af::AscNode>(consumers.at(0U));
    if (loadNode == nullptr || loadNode->GetType() != "Load") {
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc subgraph boundary '" << subNode->GetName()
             << "' is not consumed by a boundary load, which the bridge does not model";
    }
    auto loadOutputs = loadNode->outputs();
    if (loadOutputs.size() != 1U) {
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc boundary load '" << loadNode->GetName() << "' has "
             << loadOutputs.size() << " outputs";
    }
    // 这个名字就是 region 的块参数：下游算子按它引用这段数据，所以边界表记它，并且注册进
    // 子图自己的 producer 表，让下游的 `input_names` 解析得到。
    const std::string valueName = ResultValueName(loadNode->GetName(), 0, 1);
    body->boundary_input_names[static_cast<size_t>(parentIndex)] = valueName;
    const auto outputAnchor = *loadOutputs.begin();
    mlir::FailureOr<TensorValueDTO> descriptor =
        ToTensorValueDTO(outputAnchor->attr, mlir::afir::Role::Compute, loc, -1);
    if (mlir::failed(descriptor)) {
      return mlir::failure();
    }
    bodyCtx.producerByTensor[outputAnchor->attr.mem.tensor_id] = valueName;
    bodyCtx.descriptorByTensor[outputAnchor->attr.mem.tensor_id] = *descriptor;
    boundaryLoads.insert(static_cast<const void *>(loadNode.get()));
  }

  for (const auto &subNode : subGraph.GetAllNodes()) {
    if (subNode == nullptr) {
      return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc subgraph '" << *subGraphName
                                  << "' contains a null node";
    }
    const std::string subType = subNode->GetType();
    const std::string subName = subNode->GetName();

    // 边界的 `Data` 与它那个 `Load` 已经在上一趟解掉了。
    if (subType == "Data") {
      continue;
    }
    if (subType == "Load") {
      if (boundaryLoads.contains(static_cast<const void *>(subNode.get()))) {
        continue;
      }
      // 不是边界的 Load 意味着子图自己从 GM 搬东西，与「VF 只在片上算」这个前提矛盾。
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc subgraph load '" << subName
             << "' does not read a subgraph boundary, which the bridge does not model";
    }

    if (subType == "Store") {
      // 边界 Store：把它的来源直接记成「产出这条输出的那个名字」，Store 本身不翻。
      auto inputs = subNode->inputs();
      if (inputs.size() != 1U) {
        return mlir::afir::emitCapabilityShortfall(loc) << "UnsupportedOperation: VectorFunc boundary store '"
                                                        << subName << "' has " << inputs.size() << " inputs";
      }
      const int64_t sourceTensor = (*inputs.begin())->attr.mem.tensor_id;
      auto producer = bodyCtx.producerByTensor.find(sourceTensor);
      if (producer == bodyCtx.producerByTensor.end()) {
        return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc boundary store '" << subName << "' reads tensor "
                                    << sourceTensor << ", which has no producer in the subgraph";
      }
      for (const auto &outputAnchor : subNode->outputs()) {
        bodyCtx.producerByTensor[outputAnchor->attr.mem.tensor_id] = producer->second;
        auto descriptor = bodyCtx.descriptorByTensor.find(sourceTensor);
        if (descriptor != bodyCtx.descriptorByTensor.end()) {
          bodyCtx.descriptorByTensor[outputAnchor->attr.mem.tensor_id] = descriptor->second;
        }
      }
      continue;
    }

    if (subType == "Output") {
      mlir::FailureOr<int64_t> outputIndex = GetIoIndex(subNode, loc);
      if (mlir::failed(outputIndex)) {
        return mlir::failure();
      }
      if (*outputIndex < 0 || static_cast<size_t>(*outputIndex) >= body->boundary_output_names.size()) {
        return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc subgraph output '" << subName
                                    << "' maps to parent output " << *outputIndex << ", which is out of range ("
                                    << body->boundary_output_names.size() << " dsts)";
      }
      auto inputs = subNode->inputs();
      if (inputs.size() != 1U) {
        return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc subgraph output '" << subName << "' has "
                                    << inputs.size() << " inputs";
      }
      auto producer = bodyCtx.producerByTensor.find((*inputs.begin())->attr.mem.tensor_id);
      if (producer == bodyCtx.producerByTensor.end()) {
        return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc subgraph output '" << subName
                                    << "' has no producer in the subgraph";
      }
      body->boundary_output_names[static_cast<size_t>(*outputIndex)] = producer->second;
      continue;
    }

    // 子图内的缓存守卫（`exec_condition` 不是 `kNoCache`）本阶段不收。
    //
    // 这一条必须拦在这里，不能留给下游：那些节点的 `loop_axis` 在子图里是 -1，而
    // AFIR 的 `#afir.sched` 在属性构造时就查「有缓存周期就必须有 loop_axis」
    // （`AFIRDialect.cpp` 的 `ScheduleAttr::verify`）——那是硬失败，整条编译红掉，
    // 而这件事的性质是「还没建模」，应当干净回落。探针上实测到过 4 格（T5 的广播菱形）。
    if (subNode->attr.sched.exec_condition != af::ExecuteCondition::kNoCache) {
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc subgraph node '" << subName
             << "' carries a cache exec_condition, which the register-form lowering does not model yet";
    }

    // 标量链（常量 `Scalar` 与运行期 `ScalarData` 在子图内的副本）本阶段不收：它在
    // legacy 那一侧走的是值参数链，与 region 的块参数是两套传参，归 M4。
    if (subType == "Scalar" || subType == "ScalarData" || subType == "IndexExpr") {
      return mlir::afir::emitCapabilityShortfall(loc)
             << "UnsupportedOperation: VectorFunc subgraph node '" << subName << "' of type '" << subType
             << "' is not modeled yet; scalar operands inside a vector function are out of scope";
    }

    if (mlir::failed(AppendTemplateNode(subNode, bodyInfo, bodyCtx))) {
      return mlir::failure();
    }
  }

  for (auto [index, name] : llvm::enumerate(body->boundary_input_names)) {
    if (name.empty()) {
      return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc node '" << vfInfo.name << "' src #" << index
                                  << " has no boundary in the subgraph";
    }
  }
  for (auto [index, name] : llvm::enumerate(body->boundary_output_names)) {
    if (name.empty()) {
      return mlir::emitError(loc) << "InvalidDescriptor: VectorFunc node '" << vfInfo.name << "' dst #" << index
                                  << " has no boundary in the subgraph";
    }
  }

  body->nodes = std::move(bodyInfo.nodes);
  vfInfo.vector_func_body = std::move(body);
  return mlir::success();
}

mlir::LogicalResult AppendTemplateNode(const af::AscNodePtr &node, TemplateInfo &info, TemplateNodeContext &ctx) {
  mlir::Location loc = ctx.loc;
  NodeInfo nodeInfo;
  nodeInfo.name = node->GetName();
  nodeInfo.type = node->GetType();
  // `Scalar` / `ScalarData` / `IndexExpr` 与 `Data` / `Output` 同类：它们不产生 AFIR op，
  // 只产生一个值。所以不在「哪些 op 有 buffer 形态」那张表里，也不该被那张表拦住。
  // 三者的值来源各不相同——编译期字面量 / kernel 形参 / 由 size var 算出来的运行期常量
  // ——但对这道门禁是一回事。
  if (nodeInfo.type == "Scalar") {
    if (mlir::failed(ReadScalarValue(node, nodeInfo, loc))) {
      return mlir::failure();
    }
  }
  if (nodeInfo.type == "IndexExpr" && mlir::failed(ReadIndexExpr(node, nodeInfo, loc))) {
    return mlir::failure();
  }
  if (const mlir::afir::bridge::FloatIrAttrMapping *floatAttr = mlir::afir::bridge::findFloatIrAttr(nodeInfo.type);
      floatAttr != nullptr && mlir::failed(ReadFloatIrAttr(node, *floatAttr, nodeInfo, loc))) {
    return mlir::failure();
  }
  // `Workspace` 同理，只是它在一张图里出现的方向有两种：RCore phase 1 里它在 Store 之后
  // 当汇（`.x = {Store.y}`，自身描述子退化到只剩 `mem`），phase 2 里它在 Load 之前当源
  // （带完整的 axis / repeats / strides）。两边都不产生 op，只产生同一块 GM 上的一个值。
  if (const mlir::afir::bridge::IntIrAttrMapping *intAttr = mlir::afir::bridge::findIntIrAttr(nodeInfo.type);
      intAttr != nullptr && mlir::failed(ReadIntIrAttr(node, *intAttr, nodeInfo, loc))) {
    return mlir::failure();
  }
  if (info.buffer_form && nodeInfo.type != "Data" && nodeInfo.type != "Output" && nodeInfo.type != "Scalar" &&
      nodeInfo.type != "ScalarData" && nodeInfo.type != "IndexExpr" && nodeInfo.type != "Workspace" &&
      nodeInfo.type != kVectorFuncNodeType && !isAFIRBufferFormOperationSupported(nodeInfo.type)) {
    // 节点类型没建模 —— 这是能力不足（`Nddma` 就走这里），legacy 收得了，整图回落给它。
    mlir::afir::emitCapabilityShortfall(loc)
        << "UnsupportedOperation: Schedule buffer-form bridge does not support node '" << nodeInfo.name << "' of type '"
        << nodeInfo.type << "'";
    return mlir::failure();
  }
  nodeInfo.sched_axis = node->attr.sched.axis;
  nodeInfo.loop_axis = node->attr.sched.loop_axis;
  // 缓存复用周期与 `axis` / `loop_axis` 同在上游的 `SchedInfo` 里，一起读。忽略它只会
  // 慢（每个迭代重算），但照搬一个读不懂的取值会静默丢掉这次决策，所以不认识的值在
  // 这里就报出来，由调用方回落 legacy。
  if (std::optional<std::string> execCondition = detail::ExecConditionStr(node->attr.sched.exec_condition)) {
    nodeInfo.exec_condition = *execCondition;
  } else {
    return mlir::afir::emitCapabilityShortfall(loc)
           << "UnsupportedOperation: node '" << nodeInfo.name << "' has exec_condition "
           << static_cast<int32_t>(node->attr.sched.exec_condition) << ", which the Schedule bridge does not model";
  }
  // Whether a node has a tmp buffer is the scheduler's answer, not this
  // bridge's: it ran the target's own CalcTmpBufSize. Only the identity travels
  // -- the size is aggregated on the ATT side and referenced by name.
  for (const auto &tmpBuffer : node->attr.tmp_buffers) {
    auto normalized = translateSchedulerTmpBuffer(nodeInfo.type, tmpBuffer.id, tmpBuffer.buf_desc.life_time_axis_id);
    if (!normalized) {
      return mlir::emitError(loc) << "InvalidBinding: unsupported tmp buffer metadata on node " << nodeInfo.name;
    }
    nodeInfo.tmp_buffers.push_back(*normalized);
  }
  if (mlir::failed(AppendNodeInputs(node, nodeInfo, ctx)) || mlir::failed(AppendNodeOutputs(node, nodeInfo, ctx)) ||
      mlir::failed(AppendNodeIo(node, nodeInfo, info, ctx))) {
    return mlir::failure();
  }
  // 在 push_back 之前取：节点级引用（`api_tiling_field`）要随 nodeInfo 进 nodes，func 级
  // 布局声明进 api_tiling_data——两处来自同一次调用，不会说岔。
  std::optional<ApiTilingInfo> apiTiling = GetApiTilingInfo(node->GetType(), node->GetName(), ctx.templateId);
  if (apiTiling) {
    nodeInfo.api_tiling_field = apiTiling->field_name;
  }
  info.nodes.push_back(std::move(nodeInfo));
  for (const auto &output : node->outputs()) {
    if (output->attr.que.id >= 0) {
      info.queue_ids.push_back(output->attr.que.id);
    }
    if (output->attr.buf.id >= 0) {
      info.buffer_ids.push_back(output->attr.buf.id);
    }
  }
  // tmp buffer 的 id 与普通缓冲是**同一个 id 空间**（legacy 的 `AllocTmpBuffGroup` 就在
  // 这个空间里分配，且跨节点可复用），落到 TilingData 上也是同一套 `b{id}_size`。所以
  // 它并进同一张表，而不是另立一张——另立的那张（`afir.tmp_buffer_ids`）除了给
  // GenTilingData 多一条特判之外没有别的读者，两张表说同一件事就会有说岔的一天。
  for (const auto &tmpBuffer : node->attr.tmp_buffers) {
    if (tmpBuffer.id >= 0) {
      info.buffer_ids.push_back(tmpBuffer.id);
    }
  }
  if (apiTiling) {
    info.api_tiling_data.push_back(std::move(*apiTiling));
  }
  return mlir::success();
}

template <typename GraphT>
mlir::FailureOr<TemplateInfo> ToTemplateInfo(const GraphT &graph, const ascir::ScheduleGroup &group, size_t stageId,
                                             size_t templateId, mlir::Location loc) {
  TemplateInfo info = InitializeTemplateInfo(graph, group, stageId, templateId);
  if (mlir::failed(AppendTemplateAxes(graph, info, loc))) {
    return mlir::failure();
  }
  mlir::FailureOr<std::map<int64_t, int64_t>> lifetimes = CollectTmpBufferLifetimes(graph, loc);
  if (mlir::failed(lifetimes)) {
    return mlir::failure();
  }
  TemplateNodeContext nodeCtx{{}, {}, *lifetimes, templateId, loc};
  for (const auto &node : graph.GetAllNodes()) {
    if (mlir::failed(AppendTemplateNode(node, info, nodeCtx))) {
      return mlir::failure();
    }
    // 二次融合的节点还挂着一张子图。它要在父节点自己的 `NodeInfo` 收完之后再收：边界表
    // 是按父算子 srcs / dsts 的下标配对的，而那两张表正是 `AppendTemplateNode` 填出来的。
    if (node != nullptr && node->GetType() == kVectorFuncNodeType && !info.nodes.empty() &&
        mlir::failed(AppendVectorFuncBody(node, info.nodes.back(), graph, nodeCtx))) {
      return mlir::failure();
    }
  }
  if (mlir::failed(ValidateTemplateResources(info, loc))) {
    return mlir::failure();
  }
  SortAndUnique(info.queue_ids);
  SortAndUnique(info.buffer_ids);
  return info;
}

std::string ModuleName(const ascir::FusedScheduledResult &fusedResult, size_t subgraphId, size_t planId) {
  return std::string(fusedResult.fused_graph_name.GetString()) + "_subgraph" + std::to_string(subgraphId) + "_plan" +
         std::to_string(planId);
}

void AppendShapeEnvInfo(PlanInfo &info) {
  const auto *shapeEnv = af::GetCurShapeEnvContext();
  if (shapeEnv == nullptr) {
    return;
  }
  for (const auto &guard : shapeEnv->GetAllSymbolCheckInfos()) {
    info.shape_guards.push_back(ExprStr(guard.expr));
  }
  for (const auto &assertion : shapeEnv->GetAllSymbolAssertInfos()) {
    info.shape_asserts.push_back(ExprStr(assertion.expr));
  }
}

mlir::FailureOr<PlanInfo> ToPlanInfo(const ascir::FusedScheduledResult &fusedResult,
                                     const ascir::ScheduledResult &scheduledResult, size_t subgraphId, size_t planId,
                                     mlir::Location loc, llvm::ArrayRef<std::vector<size_t>> selectedTemplates) {
  PlanInfo info;
  info.name = ModuleName(fusedResult, subgraphId, planId);
  info.subgraph_id = static_cast<int64_t>(subgraphId);
  info.plan_id = static_cast<int64_t>(planId);
  info.is_reduce_mem_reuse = scheduledResult.is_reduce_mem_reuse;
  info.enable_group_parallel = scheduledResult.enable_group_parallel;
  info.cube_type = static_cast<int64_t>(scheduledResult.cube_type);
  info.score_func = scheduledResult.score_func.GetString();
  if (mlir::failed(AppendWorkspacePlan(fusedResult, info, loc))) {
    return mlir::failure();
  }
  for (const auto &originVar : fusedResult.origin_vars) {
    info.origin_vars.push_back(ExprStr(originVar));
  }
  for (const auto &frontendVar : ascgen_utils::GetFrontendShapeVars(fusedResult)) {
    if (!frontendVar.IsConstExpr()) {
      info.frontend_shape_vars.push_back(ExprStr(frontendVar));
    }
  }
  AppendShapeEnvInfo(info);

  for (const auto &[dstStage, srcStages] : scheduledResult.var_relations) {
    for (const auto &[srcStage, relations] : srcStages) {
      for (const auto &[dstVar, srcExpr] : relations) {
        info.var_relations.push_back(
            {static_cast<int64_t>(dstStage), static_cast<int64_t>(srcStage), dstVar, ExprStr(srcExpr)});
      }
    }
  }

  for (auto [stageId, group] : llvm::enumerate(scheduledResult.schedule_groups)) {
    info.template_counts_per_stage.push_back(
        static_cast<int64_t>(selectedTemplates.empty() ? group.impl_graphs.size() : selectedTemplates[stageId].size()));
    std::string keyDiagnostic;
    if (!common::ValidateTemplateKeys(group, &keyDiagnostic)) {
      mlir::emitError(loc) << "stage " << stageId << ": " << keyDiagnostic;
      return mlir::failure();
    }
    for (auto [templateId, graph] : llvm::enumerate(group.impl_graphs)) {
      if (!selectedTemplates.empty() && !llvm::is_contained(selectedTemplates[stageId], templateId)) continue;
      mlir::FailureOr<TemplateInfo> templateInfo = ToTemplateInfo(graph, group, stageId, templateId, loc);
      if (mlir::failed(templateInfo)) {
        return mlir::failure();
      }
      info.templates.push_back(std::move(*templateInfo));
    }
  }
  return info;
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> PlanToAfir(mlir::MLIRContext &context, const ascir::FusedScheduledResult &fusedResult,
                                             const ascir::ScheduledResult &scheduledResult, size_t subgraphId,
                                             size_t planId, llvm::ArrayRef<std::vector<size_t>> selectedTemplates) {
  if (!selectedTemplates.empty() && selectedTemplates.size() != scheduledResult.schedule_groups.size()) {
    mlir::emitError(mlir::UnknownLoc::get(&context)) << "Plan selection does not cover its stages";
    return nullptr;
  }
  if (scheduledResult.schedule_groups.empty()) {
    mlir::emitError(mlir::UnknownLoc::get(&context)) << "Plan has no stage";
    return nullptr;
  }
  for (auto [stageId, group] : llvm::enumerate(scheduledResult.schedule_groups)) {
    if (group.impl_graphs.empty()) {
      mlir::emitError(mlir::UnknownLoc::get(&context)) << "Plan stage " << stageId << " has no template";
      return nullptr;
    }
  }
  mlir::Location loc = mlir::UnknownLoc::get(&context);
  mlir::FailureOr<PlanInfo> info = ToPlanInfo(fusedResult, scheduledResult, subgraphId, planId, loc, selectedTemplates);
  if (mlir::failed(info)) {
    return nullptr;
  }
  return BuildAfirModule(context, *info);
}

void MaybeDumpAfirModule(const std::string &stage, mlir::ModuleOp module, const std::string &dumpRoot) {
  if (dumpRoot.empty()) {
    return;
  }
  try {
    static std::atomic<uint64_t> idx{0};
    // 模块名就是 `ModuleName(result, subgraphId, planId)`：`BuildAfirModule` 建模块时
    // 已经把它写成了符号名，这里直接读，不必再把那三项传进来重算一遍。
    llvm::StringRef name = module.getSymName().value_or("unnamed");
    std::string path = dumpRoot + "afir_" + std::to_string(idx.fetch_add(1)) + "_" + stage + "_" + name.str() + ".mlir";
    std::ofstream out(path);
    if (!out.is_open()) {
      return;
    }
    std::string text;
    llvm::raw_string_ostream os(text);
    module.print(os);
    os.flush();
    out << text << "\n";
  } catch (...) {
    // Never propagate into the codegen path.
  }
}

}  // namespace bridge
}  // namespace afir
}  // namespace mlir
