/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- ReducePlan.cpp - 一个 reduce Op 的调度推论 ---------------*- C++ -*-===//

#include "dialect/AFIR/Analysis/ReducePlan.h"

#include <optional>
#include <string>
#include <utility>

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/ReduceScratch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

using namespace mlir;

namespace mlir::afir {
namespace {

// 失败原因只回给调用方，**不发诊断**——这条契约见 `ReducePlan.h`：同一个失败在
// LoopGenerate 与能力查询两处要说不同的话。此前原因根本传不出来，于是调用方想报也
// 没得报，只能各自编一句笼统的话，真因全丢。
LogicalResult failWithReason(std::string *diagnostic, llvm::StringRef reason) {
  if (diagnostic != nullptr) {
    diagnostic->assign(reason.begin(), reason.end());
  }
  return failure();
}

// stride 是否为常量 0。bridge 对零 stride 写的就是字面量 `"0"`（调度器的
// `SizeExpr` 序列化结果），因此这里按文本判定而不去解析表达式：解析要引入 SymEngine，
// 而「是不是 0」在这条链路上是一个**已经被序列化固定**的事实。
// 判不出来时一律按「非 0」处理——漏判一维归约会让调用方拒绝下降（报错），
// 误判成归约则会静默算错值。
bool isZeroExpr(ExprAttr expr) {
  if (!expr) {
    return false;
  }
  return expr.getValue().getValue().trim() == "0";
}

// 被归约的维：destination 在这一维不前进（stride 0），而 input 前进。
// 两侧都为 0 的维是广播维，不是归约维——`CalcReduceTmpSize` 的 `r_in_ub_exp`
// 用的也是这条判据（`ascir/reg_func/reduce.cpp:66-70`）。
LogicalResult collectReducedAxes(TensorDescAttr input, TensorDescAttr destination,
                                 llvm::SmallDenseSet<int64_t, 4> &reducedAxes, std::string *diagnostic) {
  ArrayRef<int64_t> axis = destination.getAxis();
  if (axis.empty() || input.getAxis() != axis) {
    return failWithReason(diagnostic, "input and destination do not agree on one non-empty axis list");
  }
  auto inputStrides = input.getStrides();
  auto destinationStrides = destination.getStrides();
  if (inputStrides.size() != axis.size() || destinationStrides.size() != axis.size()) {
    return failWithReason(diagnostic, "stride list length does not match the axis list");
  }
  for (auto [index, axisId] : llvm::enumerate(axis)) {
    if (isZeroExpr(destinationStrides[index]) && !isZeroExpr(inputStrides[index])) {
      reducedAxes.insert(axisId);
    }
  }
  if (reducedAxes.empty()) {
    return failWithReason(diagnostic,
                          "no axis is reduced: no axis has a zero destination stride and a non-zero input stride");
  }
  return success();
}

// A scheduled loop axis may be a split or merge descendant of the original
// reduction axes. It is an accumulation axis only when every leaf in that
// provenance is reduced; a mixed A/R merged axis must not be treated as an
// accumulator because different A results would otherwise be folded together.
bool isDerivedOnlyFromReducedAxes(int64_t axisId, const llvm::SmallDenseSet<int64_t, 4> &reducedAxes,
                                  const llvm::DenseMap<int64_t, AxisAttr> &axesById,
                                  llvm::SmallDenseSet<int64_t, 8> &active) {
  if (reducedAxes.contains(axisId)) {
    return true;
  }
  auto found = axesById.find(axisId);
  if (found == axesById.end() || found->second.getFrom().empty() || !active.insert(axisId).second) {
    return false;
  }
  const bool reduced = llvm::all_of(found->second.getFrom(), [&](int64_t sourceId) {
    return isDerivedOnlyFromReducedAxes(sourceId, reducedAxes, axesById, active);
  });
  active.erase(axisId);
  return reduced;
}

// 两个入口都要的那几样东西。抽出来是为了让语义查询与完整计划**读同一份输入**——各写一遍
// 取用的后果是「同一个 op，两处对它是不是合法 reduce 的判断不一致」。
struct ReduceDescriptors {
  TensorDescAttr input;
  TensorDescAttr destination;
  ScheduleAttr sched;
};

FailureOr<ReduceDescriptors> getReduceDescriptors(Operation *op, std::string *diagnostic) {
  if (op == nullptr) {
    return failWithReason(diagnostic, "no operation");
  }
  auto destinationStyle = dyn_cast<AFIRDestinationStyleInterface>(op);
  const unsigned expectedDestinations = 1U;
  if (!destinationStyle || destinationStyle.getNumDestinations() != expectedDestinations) {
    return failWithReason(diagnostic, "operation is not destination-style with exactly one destination");
  }
  auto operandDescs = op->getAttrOfType<ArrayAttr>(kOperandDescsAttrName);
  auto destinationDescs = op->getAttrOfType<ArrayAttr>(kDestinationDescsAttrName);
  if (!operandDescs || operandDescs.empty() || !destinationDescs ||
      destinationDescs.size() != destinationStyle.getNumDestinations()) {
    return failWithReason(diagnostic, "missing operand or destination tensor descriptors");
  }
  auto input = dyn_cast<TensorDescAttr>(operandDescs[0]);
  auto destination = dyn_cast<TensorDescAttr>(destinationDescs[0]);
  if (!input || !destination) {
    return failWithReason(diagnostic, "operand or destination descriptor is not a tensor descriptor");
  }
  auto sched = op->getAttrOfType<ScheduleAttr>(kSchedAttrName);
  if (!sched || sched.getLoopAxis() < 0) {
    return failWithReason(diagnostic, "missing schedule attribute or loop axis");
  }
  return ReduceDescriptors{input, destination, sched};
}

}  // namespace

FailureOr<ReduceAccumulationInfo> analyzeReduceAccumulation(Operation *op, std::string *diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  FailureOr<ReduceDescriptors> descs = getReduceDescriptors(op, diagnostic);
  if (failed(descs)) {
    return failure();
  }

  llvm::SmallDenseSet<int64_t, 4> reducedAxes;
  if (failed(collectReducedAxes(descs->input, descs->destination, reducedAxes, diagnostic))) {
    return failure();
  }

  llvm::DenseMap<int64_t, AxisAttr> axesById;
  if (auto func = op->getParentOfType<func::FuncOp>()) {
    if (auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName)) {
      for (AxisAttr axis : graph.getAxes()) {
        axesById.try_emplace(axis.getId(), axis);
      }
    }
  }

  // 累加轴：调度路径（最外到 loop_axis）上属于归约维的轴。落在 `loop_axis` 之内的归约维
  // 由这一次 API 调用一并算完，不跨轮，因此不进这张表。
  //
  // 这里**不检查** `vectorized_axis` 的根数、不读 scratch：一个归约沿哪些轴累加，与高阶
  // API 能不能把它表示成二维、与调度器给了几块缓冲都无关。按 `sched.getAxis()` 的顺序走，
  // 不依赖哈希序。
  ReduceAccumulationInfo info;
  for (int64_t axisId : descs->sched.getAxis()) {
    llvm::SmallDenseSet<int64_t, 8> active;
    if (reducedAxes.contains(axisId) || isDerivedOnlyFromReducedAxes(axisId, reducedAxes, axesById, active)) {
      info.accumulationAxes.push_back(axisId);
    }
    if (axisId == descs->sched.getLoopAxis()) {
      break;
    }
  }
  return info;
}

FailureOr<ReducePlan> buildReducePlan(Operation *op, std::string *diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  FailureOr<ReduceDescriptors> descs = getReduceDescriptors(op, diagnostic);
  if (failed(descs)) {
    return failure();
  }
  auto destinationStyle = cast<AFIRDestinationStyleInterface>(op);
  TensorDescAttr input = descs->input;
  TensorDescAttr destination = descs->destination;

  // 语义那一半复用同一个入口，不再在这里重算一遍——两份 provenance 遍历迟早会漂。
  FailureOr<ReduceAccumulationInfo> accumulation = analyzeReduceAccumulation(op, diagnostic);
  if (failed(accumulation)) {
    return failure();
  }

  ReducePlan plan;
  plan.accumulationAxes = std::move(accumulation->accumulationAxes);

  // pattern 与 first / last：连续的若干保留轴合成 A，连续的若干归约轴合成 R。这里按
  // destination 的 vectorized stride 分类；A/R 交替超过一次就无法表示成高阶 API 的二维
  // AR/RA 布局，必须拒绝而不是把不相邻的轴相乘后静默改变局部内存顺序。
  ArrayRef<int64_t> vectorizedAxis = destination.getVectorizedAxis();
  auto vectorizedStrides = destination.getVectorizedStrides();
  if (vectorizedAxis.size() < 2 || vectorizedStrides.size() != vectorizedAxis.size() ||
      input.getVectorizedAxis() != vectorizedAxis || input.getVectorizedStrides().size() != vectorizedAxis.size()) {
    return failWithReason(
        diagnostic, "vectorized axes must be at least two and identical on both sides, with matching stride lists");
  }
  auto inputVectorizedStrides = input.getVectorizedStrides();
  auto vectorizedIsReduced = [&](size_t position) {
    return isZeroExpr(vectorizedStrides[position]) && !isZeroExpr(inputVectorizedStrides[position]);
  };
  const bool startsReduced = vectorizedIsReduced(0);
  bool crossed = false;
  unsigned reducedCount = 0;
  for (auto [position, axisId] : llvm::enumerate(vectorizedAxis)) {
    const bool reduced = vectorizedIsReduced(position);
    if (reduced != startsReduced) {
      crossed = true;
    } else if (crossed) {
      return failWithReason(
          diagnostic,
          "kept and reduced vectorized axes alternate more than once, which no two-dimensional AR/RA layout "
          "can express");
    }
    (reduced ? plan.last.axes : plan.first.axes).push_back(axisId);
    reducedCount += reduced ? 1U : 0U;
  }
  if (plan.first.axes.empty() || plan.last.axes.empty()) {
    return failWithReason(diagnostic, plan.first.axes.empty() ? "every vectorized axis is reduced, leaving no kept axis"
                                                              : "no vectorized axis is reduced");
  }
  plan.pattern = startsReduced ? ReducePattern::RA : ReducePattern::AR;
  plan.tailLast = reducedCount == 1;

  // Roles belong to operand uses, independently of their order or physical buffer ids.
  auto scratch = getReduceScratchIndices(op, destinationStyle.getNumTmpBuffers(), diagnostic);
  if (failed(scratch)) {
    return failure();
  }
  plan.apiScratchIndex = scratch->api;
  if (plan.isAccumulating()) {
    if (!scratch->roundResult) {
      return failWithReason(diagnostic, "an accumulating reduce requires a round_result scratch operand");
    }
    plan.accumulationScratchIndex = scratch->roundResult;
  }
  return plan;
}

namespace {

// 一根调度轴的 `from` 链上所有 Original 轴，收进 `collected`（SetVector，去重且保序）。
// 拆分出来的 inner / outer 指向同一个来源，所以 `z1T` 与 `z1t` 一起走只会得到一个 `z1`。
LogicalResult collectOriginalAxes(int64_t axisId, const llvm::DenseMap<int64_t, AxisAttr> &axesById,
                                  llvm::SetVector<int64_t> &collected, llvm::SmallDenseSet<int64_t, 8> &active,
                                  std::string *diagnostic) {
  auto found = axesById.find(axisId);
  if (found == axesById.end()) {
    return failWithReason(diagnostic, "a reduced axis is not declared in the asc_graph axis list");
  }
  AxisAttr axis = found->second;
  if (axis.getAxisType() == AxisType::Original) {
    if (!axis.getSize()) {
      return failWithReason(diagnostic, "an original reduction axis has no size expression");
    }
    // 字面量尺寸必须为正。判在这里而不是等物化：物化那一步
    // （`materializeExtentValue`，下界 1）在**下降**侧，而能力查询只调本函数——两处判据不
    // 同就会出现「能力说支持、下降才失败」，候选被丢掉而诊断停在「下降失败」这一层。
    // 符号尺寸判不了，归运行期的 shape 合法性约束。
    int64_t literal = 0;
    if (!axis.getSize().getValue().getValue().getAsInteger(10, literal) && literal <= 0) {
      return failWithReason(diagnostic, "an original reduction axis has a non-positive literal size");
    }
    collected.insert(axisId);
    return success();
  }
  if (axis.getFrom().empty()) {
    return failWithReason(diagnostic, "a reduced axis is neither original nor derived: its provenance is lost");
  }
  // 环只可能来自坏掉的轴图；不防会栈溢出，而不是报错。
  if (!active.insert(axisId).second) {
    return failWithReason(diagnostic, "the axis provenance chain contains a cycle");
  }
  for (int64_t sourceId : axis.getFrom()) {
    if (failed(collectOriginalAxes(sourceId, axesById, collected, active, diagnostic))) {
      return failure();
    }
  }
  active.erase(axisId);
  return success();
}

}  // namespace

FailureOr<LogicalReductionDomain> analyzeLogicalReductionDomain(Operation *op, std::string *diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  FailureOr<ReduceDescriptors> descs = getReduceDescriptors(op, diagnostic);
  if (failed(descs)) {
    return failure();
  }

  llvm::SmallDenseSet<int64_t, 4> reducedAxes;
  if (failed(collectReducedAxes(descs->input, descs->destination, reducedAxes, diagnostic))) {
    return failure();
  }

  llvm::DenseMap<int64_t, AxisAttr> axesById;
  auto func = op->getParentOfType<func::FuncOp>();
  if (func) {
    if (auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName)) {
      for (AxisAttr axis : graph.getAxes()) {
        axesById.try_emplace(axis.getId(), axis);
      }
    }
  }
  if (axesById.empty()) {
    return failWithReason(diagnostic, "the function has no asc_graph axis list");
  }

  // 按 destination 的轴序走，而不是遍历 `reducedAxes` 这个哈希集合：分母是若干个表达式的
  // 乘积，顺序进产物（`afir.symbolic_eval` 的操作数序），哈希序会让同一份输入编出不同的
  // kernel 文本，lit 就钉不住了。
  llvm::SetVector<int64_t> originals;
  llvm::SmallDenseSet<int64_t, 8> active;
  for (int64_t axisId : descs->destination.getAxis()) {
    if (!reducedAxes.contains(axisId)) {
      continue;
    }
    if (failed(collectOriginalAxes(axisId, axesById, originals, active, diagnostic))) {
      return failure();
    }
  }
  if (originals.empty()) {
    return failWithReason(diagnostic, "the reduction domain is empty after tracing provenance");
  }

  LogicalReductionDomain domain;
  for (int64_t axisId : originals) {
    domain.originalAxes.push_back(axisId);
    domain.sizes.push_back(axesById.find(axisId)->second.getSize());
  }
  return domain;
}

namespace {

bool isReduceOp(Operation *op) {
  return isa_and_nonnull<SumOp, MaxOp, MinOp, MeanOp, ProdOp, AnyOp, AllOp>(op);
}

// 这个值是不是一块 GM workspace（跨阶段/跨 template 的载体）。是的话给出它的 buffer id：
// 那是调度器的 tensor id，也是**唯一**跨 func 对得上的身份——两个阶段各自建自己的
// `memref.alloc`，彼此之间没有 SSA 边。
std::optional<int64_t> workspaceBufferId(Value value) {
  auto alloc = value.getDefiningOp<memref::AllocOp>();
  if (!alloc) {
    return std::nullopt;
  }
  auto descriptor = alloc->getAttrOfType<BufferDescAttr>(kBufferDescAttrName);
  if (!descriptor || descriptor.getSpace().getValue() != Space::GM) {
    return std::nullopt;
  }
  int64_t bufferId = descriptor.getBufferId();
  if (bufferId < 0) {
    return std::nullopt;
  }
  return bufferId;
}

// 「写进某块 workspace 的东西里有没有经过归约」。**只用于跨阶段那一侧**，见下面
// `reachesAPartialWorkspace` 的注释。
//
// `workspaceCarriesReduction` 是跨 func 的那一半：由下面的不动点算出，这里只查。memo 只在
// 同一轮不动点内有效（表一变结论就可能变），所以由调用方按轮清空。
bool touchesAReduction(Value value, const llvm::DenseMap<int64_t, bool> &workspaceCarriesReduction,
                       llvm::DenseMap<Value, bool> &memo) {
  if (auto cached = memo.find(value); cached != memo.end()) {
    return cached->second;
  }
  // 先占位成「没有」：正常 IR 的生产者锥无环，占位只为坏图下不至于栈溢出。
  memo[value] = false;
  if (std::optional<int64_t> bufferId = workspaceBufferId(value)) {
    auto found = workspaceCarriesReduction.find(*bufferId);
    bool carries = found != workspaceCarriesReduction.end() && found->second;
    memo[value] = carries;
    return carries;
  }
  Operation *producer = value.getDefiningOp();
  // 块参数即 kernel 入参。跨阶段的 partial **不**从这里进来——它走的是上面那块 GM
  // workspace；入参里与 workspace 有关的只有一个偏移标量。
  if (!producer) {
    return false;
  }
  if (isReduceOp(producer)) {
    memo[value] = true;
    return true;
  }
  bool touched = false;
  for (Value operand : producer->getOperands()) {
    if (touchesAReduction(operand, workspaceCarriesReduction, memo)) {
      touched = true;
      break;
    }
  }
  memo[value] = touched;
  return touched;
}

// 检查来源是否涉及携带归约产物的跨阶段 workspace，避免把 partial 数量当作 Mean 分母。
// 同函数内的已完成归约可继续参与 Mean，不能仅凭祖先中存在 reduce 就拒绝。
// 跨阶段暂不区分已完成结果与 partial，两者均保守地视为未证明完整归约域。
bool reachesAPartialWorkspace(Value value, const llvm::DenseMap<int64_t, bool> &workspaceCarriesReduction,
                              llvm::DenseMap<Value, bool> &memo) {
  if (auto cached = memo.find(value); cached != memo.end()) {
    return cached->second;
  }
  memo[value] = false;
  if (std::optional<int64_t> bufferId = workspaceBufferId(value)) {
    auto found = workspaceCarriesReduction.find(*bufferId);
    bool carries = found != workspaceCarriesReduction.end() && found->second;
    memo[value] = carries;
    return carries;
  }
  Operation *producer = value.getDefiningOp();
  if (!producer) {
    return false;
  }
  bool reaches = false;
  for (Value operand : producer->getOperands()) {
    if (reachesAPartialWorkspace(operand, workspaceCarriesReduction, memo)) {
      reaches = true;
      break;
    }
  }
  memo[value] = reaches;
  return reaches;
}

// 在模块内迭代传播 workspace 的归约来源，覆盖跨阶段链。
// 标记单调增长；未收敛则失败，不用不完整的结果标注归约域。
FailureOr<llvm::DenseMap<int64_t, bool>> computeWorkspaceCarriesReduction(ModuleOp module) {
  llvm::SmallDenseSet<int64_t, 4> workspaceIds;
  module->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (std::optional<int64_t> bufferId = workspaceBufferId(operand)) {
        workspaceIds.insert(*bufferId);
      }
    }
  });

  llvm::DenseMap<int64_t, bool> carries;
  bool changed = true;
  // N 个 workspace 最多发生 N 轮标记变化，之后检查收敛。
  for (size_t round = 0; round <= workspaceIds.size() + 1U; ++round) {
    if (!changed) {
      return carries;
    }
    changed = false;
    llvm::DenseMap<Value, bool> memo;
    module->walk([&](Operation *op) {
      // 本算子的 workspace 操作数与其余操作数要分开看：往一块 workspace 里写的是「其余
      // 操作数」的派生物。`afir.load %ws outs(%ub)` 反过来是读，其余操作数只有那块干净的
      // UB destination，于是这里不会给它添任何标记——读的那一侧由查表得出。
      llvm::SmallVector<int64_t, 2> written;
      bool sourceTouched = false;
      for (Value operand : op->getOperands()) {
        if (std::optional<int64_t> bufferId = workspaceBufferId(operand)) {
          written.push_back(*bufferId);
          continue;
        }
        sourceTouched = sourceTouched || touchesAReduction(operand, carries, memo);
      }
      if (!sourceTouched || written.empty()) {
        return;
      }
      for (int64_t bufferId : written) {
        bool &entry = carries[bufferId];
        if (!entry) {
          entry = true;
          changed = true;
        }
      }
    });
  }
  return failure();
}

}  // namespace

void annotateReductionDomains(ModuleOp module) {
  if (!module) {
    return;
  }
  FailureOr<llvm::DenseMap<int64_t, bool>> carries = computeWorkspaceCarriesReduction(module);
  if (failed(carries)) {
    return;
  }
  llvm::DenseMap<Value, bool> memo;
  module->walk([&](MeanOp mean) {
    if (reachesAPartialWorkspace(mean.getInput(), *carries, memo)) {
      return;
    }
    FailureOr<LogicalReductionDomain> domain = analyzeLogicalReductionDomain(mean);
    if (failed(domain)) {
      return;
    }
    mean->setAttr(kReductionDomainAttrName,
                  ReductionDomainAttr::get(module.getContext(), domain->originalAxes, domain->sizes));
  });
}

FailureOr<LogicalReductionDomain> getProvenReductionDomain(Operation *op, std::string *diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  if (!op) {
    return failWithReason(diagnostic, "no operation");
  }
  auto carried = op->getAttrOfType<ReductionDomainAttr>(kReductionDomainAttrName);
  if (!carried) {
    return failWithReason(diagnostic,
                          "the input is not proven to carry the full logical reduction domain: no "
                          "'afir.reduction_domain' was attached where the provenance was still visible");
  }
  LogicalReductionDomain domain;
  domain.originalAxes.assign(carried.getAxes().begin(), carried.getAxes().end());
  domain.sizes.assign(carried.getSizes().begin(), carried.getSizes().end());

  // 绑定校验。轴图此刻答不出来（跨阶段的合成图就答不出来，它那几根轴不在本函数的轴图里）
  // 不是错误——属性本身就是为那种情形存在的；答得出来而与属性不一致才是错误。
  FailureOr<LogicalReductionDomain> fromAxisGraph = analyzeLogicalReductionDomain(op);
  if (succeeded(fromAxisGraph) &&
      (fromAxisGraph->originalAxes != domain.originalAxes || fromAxisGraph->sizes != domain.sizes)) {
    return failWithReason(diagnostic,
                          "the carried 'afir.reduction_domain' no longer agrees with the axis graph: one of the two "
                          "was rewritten without the other");
  }
  return domain;
}

}  // namespace mlir::afir
