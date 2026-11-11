/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- LoopGenerate.cpp - Generate scheduled AFIR loops --------------------===//

#include "dialect/AFIR/Transforms/LoopGenerate.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "dialect/AFIR/Analysis/ReducePlan.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct AxisValues {
  Value axisSize;
  Value loopSize;
  Value tailSize;
};

struct LoopTreeNode;

struct LoopTreeItem {
  explicit LoopTreeItem(Operation *operation) : operation(operation) {}
  explicit LoopTreeItem(std::unique_ptr<LoopTreeNode> loop) : loop(std::move(loop)) {}

  Operation *operation = nullptr;
  std::unique_ptr<LoopTreeNode> loop;
};

struct LoopTreeNode {
  explicit LoopTreeNode(int64_t axisId = -1, LoopTreeNode *parent = nullptr) : axisId(axisId), parent(parent) {}

  int64_t axisId;
  LoopTreeNode *parent;
  SmallVector<LoopTreeItem> items;
};

class FunctionLoopGenerator {
 public:
  FunctionLoopGenerator(func::FuncOp func, afir::SymbolBindingMap bindings)
      : func(func), context(func.getContext()), builder(context), tilingSymbols(std::move(bindings)) {}

  LogicalResult run() {
    if (afir::hasGeneratedLoops(func)) {
      return success();
    }
    graph = func->getAttrOfType<afir::AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
    if (!graph) {
      return func.emitError("AFIR loop generation requires ") << mlir::afir::kAscGraphAttrName;
    }
    if (!llvm::hasSingleElement(func.getBody())) {
      return func.emitError("AFIR loop generation expects a single-block function");
    }

    Block &block = func.front();
    Operation *firstAfir = findFirstAfirOperation(block);
    if (firstAfir == nullptr) {
      return success();
    }
    builder.setInsertionPoint(firstAfir);

    if (failed(collectGraphAxes()) || failed(buildLoopTree(block))) {
      return failure();
    }
    eraseDeadScheduledDimQueries();
    if (failed(hoistLoopInvariantSetup(block, firstAfir)) || failed(generateAllAxisValues())) {
      return failure();
    }
    if (failed(validateLoopTreeValueScopes())) {
      return failure();
    }

    llvm::DenseMap<int64_t, Value> axisIndices;
    if (failed(emitTree(root, builder, axisIndices))) {
      return failure();
    }
    recordAxisBindingManifest();
    return success();
  }

  // 把这一趟打过的全部轴绑定登记到 func 上，供 MemoryAllocate 的结束校验逐条核对。
  // 顺序按 (轴 id, 角色) 排序，不按物化顺序——物化顺序随循环树形状变，而清单要能逐字节
  // 比对（lit 钉的是它的内容，不是它的生成次序）。
  void recordAxisBindingManifest() {
    if (boundAxisValues.empty()) {
      return;
    }
    SmallVector<std::pair<int64_t, int64_t>> sorted(boundAxisValues.begin(), boundAxisValues.end());
    llvm::sort(sorted);
    SmallVector<Attribute> manifest;
    manifest.reserve(sorted.size());
    for (auto [axisId, role] : sorted) {
      manifest.push_back(afir::AxisBindingAttr::get(context, axisId, static_cast<afir::AxisValueRole>(role)));
    }
    func->setAttr(afir::kAxisBindingsAttrName, ArrayAttr::get(context, manifest));
  }

  void eraseDeadScheduledDimQueries() {
    SmallVector<Operation *> deadQueries;
    llvm::DenseSet<Operation *> seen;
    for (Operation *operation : scheduledOperations) {
      for (Value result : operation->getResults()) {
        for (Operation *user : result.getUsers()) {
          auto dim = dyn_cast<memref::DimOp>(user);
          if (!dim || !dim.getResult().use_empty() || !seen.insert(user).second) {
            continue;
          }
          deadQueries.push_back(user);
        }
      }
    }
    for (Operation *operation : deadQueries) {
      operation->erase();
    }
  }

 private:
  LogicalResult collectHoistableSetup(Operation *operation, Operation *firstAfir, llvm::DenseSet<Operation *> &visiting,
                                      llvm::DenseSet<Operation *> &toHoist) {
    if (operation == nullptr || operation->getBlock() != firstAfir->getBlock() ||
        isBeforeInBlock(operation, firstAfir)) {
      return success();
    }
    if (scheduledOperations.count(operation)) {
      return operation->emitError(
          "loop-invariant setup depends on a scheduled AFIR result; values a scheduled op reads from outside the "
          "loop must be loop-invariant");
    }
    if (!visiting.insert(operation).second) {
      return success();
    }
    for (Value operand : operation->getOperands()) {
      if (Operation *definingOp = operand.getDefiningOp()) {
        if (failed(collectHoistableSetup(definingOp, firstAfir, visiting, toHoist))) {
          return failure();
        }
      }
    }
    toHoist.insert(operation);
    return success();
  }

  // 调度出去的 Op 会被搬进 `scf.for`，而循环本身建在第一个 AFIR Op 的位置上。所以
  // 它读到的每一个不参与调度的值都必须排在那个位置之前，否则生成的是「用在定义之前」
  // ——MLIR 的 dominance 校验当场拒收（`operand #N does not dominate this use`）。
  //
  // 种子是「操作数的定义方不是调度出去的 Op」，不是「操作数是 `memref.alloc`」：
  // `memref.alloc` 只是其中最常见的一类，`Scalar` / `IndexExpr` 建出来的 `arith.constant`
  // 是同样的东西。它们由 bridge 按 AscGraph 的节点次序就地建，节点排在第一个搬运之后时
  // 就会落在循环后面；legacy 把这一类统一物化在 kernel 顶部（`AppendConstTensorInit` 跑在
  // `root_loop.Generate` 之前），这里的提升就是同一件事在 MLIR 这一侧的落点。
  //
  // 调度出去的 Op 自己不在种子里——它们由循环树安排位置，跨作用域的情况由
  // `validateLoopTreeValueScopes` 判。
  LogicalResult hoistLoopInvariantSetup(Block &block, Operation *firstAfir) {
    llvm::DenseSet<Operation *> visiting;
    llvm::DenseSet<Operation *> toHoist;
    for (Operation *operation : scheduledOperations) {
      for (Value operand : operation->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (producer == nullptr || scheduledOperations.count(producer) != 0 || producer->getBlock() != &block ||
            isBeforeInBlock(producer, firstAfir)) {
          continue;
        }
        if (failed(collectHoistableSetup(producer, firstAfir, visiting, toHoist))) {
          return failure();
        }
      }
    }

    SmallVector<Operation *> blockOperations;
    blockOperations.reserve(block.getOperations().size());
    for (Operation &operation : block) {
      blockOperations.push_back(&operation);
    }
    for (Operation *operation : blockOperations) {
      if (toHoist.count(operation)) {
        operation->moveBefore(firstAfir);
      }
    }
    return success();
  }

  static bool isBeforeInBlock(Operation *operation, Operation *other) {
    for (Operation *current = operation; current != nullptr; current = current->getNextNode()) {
      if (current == other) {
        return true;
      }
    }
    return false;
  }

  // 同步族不是 payload，没有 `afir.sched`——它谈的是「这条边在哪两个 pipe 之间」，
  // 不是「这段计算落在哪个轴上」。判据写成显式的类型清单而不是「没挂 sched 就放行」：
  // 后者会让一个真忘了标注 sched 的计算 Op 静默落到循环外，而那是算错不是报错。
  static bool isSynchronizationOperation(Operation &op) {
    return isa<afir::EventRecordOp, afir::EventWaitOp, afir::PipeBarrierOp, afir::BarrierAllOp, afir::SyncAllOp>(op);
  }

  // 尺寸序言：`afir.tiling_param` 是函数里所有运行期尺寸的根（SymbolicBuilder.h），
  // `afir.symbolic_eval` 是由它们算出来的那一层（`memref.alloc(%n)` 的动态范围就取自
  // 这里）。两者既不能当插入点——那会把 `arith.muli` 排在它依赖的值之前——也不进循环树：
  // 它们没有 `afir.sched` 可言，位置由「谁都要用它」决定，不由调度决定。
  static bool isSizePrologueOperation(Operation &op) {
    return isa<afir::TilingParamOp, afir::SymbolicEvalOp>(op);
  }

  static Operation *findFirstAfirOperation(Block &block) {
    for (Operation &op : block) {
      if (afir::isAfirOperation(&op) && !isSizePrologueOperation(op)) {
        return &op;
      }
    }
    return nullptr;
  }

  LogicalResult collectGraphAxes() {
    for (afir::AxisAttr axis : graph.getAxes()) {
      if (axis.getId() < 0) {
        return func.emitError("AFIR axis id must be non-negative");
      }
      if (!graphAxesById.try_emplace(axis.getId(), axis).second) {
        return func.emitError("duplicate AFIR axis id ") << axis.getId();
      }
    }
    // `from` 是 split / merge 的来源轴引用，`AxisAttr` 自己不校验它。下游两处
    // （`generateMergedAxisSourceIndices` 与 block 偏移那支）拿它去 `graphAxesById.lookup`
    // 后直接 `getAxisType()`，查不到就是对空属性解引用；它同时还喂给
    // `AxisBindingAttr::get`，负值只会撞在 assert 上。在这里一次判掉，两处都拿到诊断。
    for (afir::AxisAttr axis : graph.getAxes()) {
      for (int64_t sourceId : axis.getFrom()) {
        if (!graphAxesById.count(sourceId)) {
          return func.emitError("AFIR axis ") << axis.getId() << " derives from unknown axis id " << sourceId;
        }
      }
    }
    return success();
  }

  FailureOr<SmallVector<int64_t>> getScheduledLoopAxes(Operation *operation) {
    auto sched = operation->getAttrOfType<afir::ScheduleAttr>(afir::kSchedAttrName);
    if (!sched || sched.getLoopAxis() < 0) {
      operation->emitError("AFIR loop generation requires afir.sched with a loop_axis");
      return failure();
    }
    // 「loop_axis 必须出现在 axis 里」是 ScheduleAttr 的不变量，但 `ScheduleAttr::get`
    // 只在 assert 里校验它——断言关掉就构造得出违反不变量的属性。这里不能把 verifier
    // 当成唯一门禁：漏判的后果不是取到错值，而是 `std::next(end())` 越界读。
    ArrayRef<int64_t> scheduledAxes = sched.getAxis();
    auto position = llvm::find(scheduledAxes, sched.getLoopAxis());
    if (position == scheduledAxes.end()) {
      operation->emitError("afir.sched loop_axis ") << sched.getLoopAxis() << " is not present in its axis list";
      return failure();
    }
    SmallVector<int64_t> path(scheduledAxes.begin(), std::next(position));
    for (int64_t id : path) {
      if (!graphAxesById.count(id)) {
        operation->emitError("afir.sched references unknown axis ") << id;
        return failure();
      }
    }
    return path;
  }

  LogicalResult buildLoopTree(Block &block) {
    SmallVector<int64_t> currentPath;
    SmallVector<LoopTreeNode *> loopNodeStack{&root};
    for (Operation &operation : block) {
      if (!afir::isAfirOperation(&operation) || isSizePrologueOperation(operation)) {
        continue;
      }
      // 同步原语留在当前层：它与邻居的相对顺序就是它的全部语义，跟着当前循环节点
      // 走既保住了这个顺序，也不会凭空把它提到循环外或压进更深一层。
      if (isSynchronizationOperation(operation)) {
        loopNodeStack.back()->items.emplace_back(&operation);
        operationPlacement[&operation] = loopNodeStack.back();
        continue;
      }
      FailureOr<SmallVector<int64_t>> operationLoopPath = getScheduledLoopAxes(&operation);
      if (failed(operationLoopPath)) {
        return failure();
      }

      size_t common = 0;
      // Find the common prefix of the current path and the operation path.
      while (common < currentPath.size() && common < operationLoopPath->size() &&
             currentPath[common] == (*operationLoopPath)[common]) {
        ++common;
      }
      currentPath.resize(common);
      loopNodeStack.resize(common + 1);
      for (size_t index = common; index < operationLoopPath->size(); ++index) {
        auto childLoopNode = std::make_unique<LoopTreeNode>((*operationLoopPath)[index], loopNodeStack.back());
        LoopTreeNode *childLoopNodePtr = childLoopNode.get();
        loopNodeStack.back()->items.emplace_back(std::move(childLoopNode));
        loopNodeStack.push_back(childLoopNodePtr);
        currentPath.push_back((*operationLoopPath)[index]);
        loopTreeAxisIds.insert((*operationLoopPath)[index]);
      }
      loopNodeStack.back()->items.emplace_back(&operation);
      operationPlacement[&operation] = loopNodeStack.back();
      scheduledOperations.insert(&operation);
    }

    // The signature pass leaves output materialization immediately after the
    // AFIR store. Move that boundary marker with its producer so tensor SSA
    // values do not escape an scf.for region.
    for (Operation &operation : block) {
      auto destination = dyn_cast<bufferization::MaterializeInDestinationOp>(operation);
      if (!destination) {
        continue;
      }
      Operation *producer = destination.getSource().getDefiningOp();
      auto placement = operationPlacement.find(producer);
      if (placement == operationPlacement.end()) {
        continue;
      }
      placement->second->items.emplace_back(&operation);
      boundaryOperations.insert(&operation);
      operationPlacement[&operation] = placement->second;
    }
    return success();
  }

  LogicalResult generateAllAxisValues() {
    zero = builder.create<arith::ConstantIndexOp>(func.getLoc(), 0);
    one = builder.create<arith::ConstantIndexOp>(func.getLoc(), 1);
    for (afir::AxisAttr axis : graph.getAxes()) {
      if (failed(generateAxisValue(axis.getId()))) {
        return failure();
      }
    }
    for (afir::AxisAttr axis : graph.getAxes()) {
      if (!isMergedLike(axis)) {
        continue;
      }
      SmallVector<Value> sourceStrides(axis.getFrom().size(), one);
      Value stride = one;
      for (size_t position = axis.getFrom().size(); position > 0; --position) {
        size_t index = position - 1;
        sourceStrides[index] = stride;
        if (index > 0) {
          AxisValues source = computedAxisValuesById.lookup(axis.getFrom()[index]);
          stride = builder.create<arith::MulIOp>(func.getLoc(), source.loopSize, stride);
        }
      }
      mergedAxisSourceStrides[axis.getId()] = std::move(sourceStrides);
    }
    for (int64_t id : loopTreeAxisIds) {
      afir::AxisAttr axis = graphAxesById.lookup(id);
      if (axis.getAxisType() == afir::AxisType::BlockOuter) {
        blockIndex = builder.create<afir::BlockIdOp>(func.getLoc(), builder.getIndexType(),
                                                     builder.getI64IntegerAttr(/*dimension=*/0));
        break;
      }
    }
    return success();
  }

  FailureOr<AxisValues> generateSplitInnerAxisValue(afir::AxisAttr axis, int64_t id) {
    if (axis.getFrom().size() != 1) {
      func.emitError("split inner axis must have exactly one source axis: ") << id;
      return failure();
    }
    FailureOr<Value> size =
        afir::materializeSymbolicExpr(builder, func.getLoc(), axis.getSize(), tilingSymbols, builder.getIndexType());
    FailureOr<AxisValues> source = generateAxisValue(axis.getFrom().front());
    if (failed(size) || failed(source)) {
      return failure();
    }
    Value tail = builder.create<arith::RemSIOp>(func.getLoc(), source->loopSize, *size);
    return AxisValues{*size, *size, tail};
  }

  FailureOr<AxisValues> generateSplitOuterAxisValue(afir::AxisAttr axis, int64_t id) {
    // BlockOuter 可以由多条源轴合并而来，这时它就是一条 Merged 轴，没有配对的 inner。
    if (axis.getAxisType() == afir::AxisType::BlockOuter && axis.getFrom().size() > 1) {
      FailureOr<Value> product = multiplySourceLoopSizes(axis);
      if (failed(product)) {
        return failure();
      }
      return AxisValues{*product, *product, zero};
    }
    if (axis.getFrom().size() != 1 || axis.getSplitPairOtherId() < 0) {
      func.emitError("split outer axis requires one source and a paired inner axis: ") << id;
      return failure();
    }
    FailureOr<AxisValues> source = generateAxisValue(axis.getFrom().front());
    FailureOr<AxisValues> inner = generateAxisValue(axis.getSplitPairOtherId());
    if (failed(source) || failed(inner)) {
      return failure();
    }
    Value axisSize = builder.create<arith::DivSIOp>(func.getLoc(), source->loopSize, inner->axisSize);
    Value hasTail = builder.create<arith::CmpIOp>(func.getLoc(), arith::CmpIPredicate::ne, inner->tailSize, zero);
    Value tailIncrement = builder.create<arith::SelectOp>(func.getLoc(), hasTail, one, zero);
    Value loopSize = builder.create<arith::AddIOp>(func.getLoc(), axisSize, tailIncrement);
    return AxisValues{axisSize, loopSize, zero};
  }

  FailureOr<AxisValues> generateAxisValueByType(afir::AxisAttr axis, int64_t id) {
    switch (axis.getAxisType()) {
      case afir::AxisType::Original: {
        FailureOr<Value> size = afir::materializeSymbolicExpr(builder, func.getLoc(), axis.getSize(), tilingSymbols,
                                                              builder.getIndexType());
        if (failed(size)) {
          return failure();
        }
        return AxisValues{*size, *size, zero};
      }
      case afir::AxisType::Merged: {
        FailureOr<Value> product = multiplySourceLoopSizes(axis);
        if (failed(product)) {
          return failure();
        }
        return AxisValues{*product, *product, zero};
      }
      case afir::AxisType::TileInner:
      case afir::AxisType::BlockInner:
        return generateSplitInnerAxisValue(axis, id);
      case afir::AxisType::TileOuter:
      case afir::AxisType::BlockOuter:
        return generateSplitOuterAxisValue(axis, id);
      case afir::AxisType::Invalid:
        func.emitError("cannot generate loop for invalid axis ") << id;
        return failure();
    }
    llvm_unreachable("unknown afir::AxisType");
  }

  // 失败时故意不把 id 从 activeCalculatingAxes 里摘掉：这一趟已经报过错，整个 pass 要
  // 失败退出，留着它不会影响后续判环。
  FailureOr<AxisValues> generateAxisValue(int64_t id) {
    auto existing = computedAxisValuesById.find(id);
    if (existing != computedAxisValuesById.end()) {
      return existing->second;
    }
    if (!graphAxesById.count(id)) {
      func.emitError("axis relation references unknown axis ") << id;
      return failure();
    }
    if (!activeCalculatingAxes.insert(id).second) {
      func.emitError("cyclic AFIR axis relation at axis ") << id;
      return failure();
    }

    FailureOr<AxisValues> result = generateAxisValueByType(graphAxesById.lookup(id), id);
    if (failed(result)) {
      return failure();
    }

    activeCalculatingAxes.erase(id);
    tagAxisValue(result->axisSize, id, afir::AxisValueRole::AxisSize);
    if (result->loopSize != result->axisSize) {
      tagAxisValue(result->loopSize, id, afir::AxisValueRole::LoopSize);
    }
    if (result->tailSize != zero) {
      tagAxisValue(result->tailSize, id, afir::AxisValueRole::TailSize);
    }
    computedAxisValuesById[id] = *result;
    return result;
  }

  FailureOr<Value> multiplySourceLoopSizes(afir::AxisAttr axis) {
    if (axis.getFrom().empty()) {
      func.emitError("merged axis has no source axes: ") << axis.getId();
      return failure();
    }
    Value product = one;
    for (int64_t sourceId : axis.getFrom()) {
      FailureOr<AxisValues> source = generateAxisValue(sourceId);
      if (failed(source)) {
        return failure();
      }
      product = builder.create<arith::MulIOp>(func.getLoc(), product, source->loopSize);
    }
    return product;
  }

  static bool isAncestor(const LoopTreeNode *ancestor, const LoopTreeNode *node) {
    for (const LoopTreeNode *current = node; current != nullptr; current = current->parent) {
      if (current == ancestor) {
        return true;
      }
    }
    return false;
  }

  // 归约结果仅可跨出其累加轴对应的循环，并将消费者重定向到 tied destination。
  // destination 在外层可见；生产者与消费者的依赖由该存储上的内存副作用表达。
  LogicalResult redirectAccumulatedOperand(OpOperand &operand, Operation *producer, LoopTreeNode *producerLoopNode,
                                           LoopTreeNode *consumerLoopNode, Operation *consumer) {
    // 仅分析累加语义；布局与 scratch 要求由后续实现检查，不能在此归为调度错误。
    std::string reason;
    FailureOr<afir::ReduceAccumulationInfo> accumulation = afir::analyzeReduceAccumulation(producer, &reason);
    if (failed(accumulation)) {
      return producer->emitError(afir::kInvalidScheduleDiagnostic)
             << ": '" << producer->getName().getStringRef()
             << "' result escapes its schedule scope but its reduction semantics cannot be analyzed: " << reason
             << "; only an accumulating reduce may escape, so either this schedule annotation is broken or the value "
                "needs tensor lifetime lowering first";
    }
    // 非累加结果不能逃出循环，否则消费者只会读到最后一轮的独立结果。
    if (!accumulation->isAccumulating()) {
      return consumer->emitError("tensor SSA value crosses sibling/outer schedule scopes: '")
             << producer->getName().getStringRef()
             << "' accumulates along no scheduled axis, so each round produces an independent result and this "
                "reference would observe only the last one; tensor lifetime lowering is required first";
    }
    auto destinationStyle = dyn_cast<afir::AFIRDestinationStyleInterface>(producer);
    if (!destinationStyle || destinationStyle.getNumDestinations() == 0) {
      return producer->emitError(afir::kInvalidScheduleDiagnostic)
             << ": accumulated result requires a tied destination";
    }

    for (LoopTreeNode *node = producerLoopNode; node != consumerLoopNode; node = node->parent) {
      if (node == nullptr) {
        // 走到根都没遇到消费者所在的节点：两者分属互不包含的作用域（兄弟循环）。
        return producer->emitError(afir::kInvalidScheduleDiagnostic)
               << ": accumulated result has consumers in different scopes";
      }
      if (!llvm::is_contained(accumulation->accumulationAxes, node->axisId)) {
        return producer->emitError(afir::kInvalidScheduleDiagnostic)
               << ": '" << producer->getName().getStringRef() << "' result escapes loop axis " << node->axisId
               << " that it does not reduce";
      }
    }
    FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(producer);
    if (failed(partition)) {
      return producer->emitError(afir::kInvalidScheduleDiagnostic) << ": invalid destination operand partition";
    }
    auto result = dyn_cast<OpResult>(operand.get());
    if (!result) {
      return failure();
    }
    for (unsigned index = 0; index < partition->destinationCount; ++index) {
      FailureOr<unsigned> tied = destinationStyle.getTiedResultIndex(index);
      if (succeeded(tied) && *tied == result.getResultNumber()) {
        operand.set(partition->destinations(producer)[index]);
        return success();
      }
    }
    return producer->emitError(afir::kInvalidScheduleDiagnostic) << ": accumulated result has no tied destination";
  }

  LogicalResult validateLoopTreeValueScopes() {
    for (Operation *operation : scheduledOperations) {
      LoopTreeNode *operationLoopNode = operationPlacement.lookup(operation);
      for (OpOperand &operand : operation->getOpOperands()) {
        Operation *producer = operand.get().getDefiningOp();
        if (!scheduledOperations.count(producer)) {
          continue;
        }
        LoopTreeNode *producerLoopNode = operationPlacement.lookup(producer);
        if (!isAncestor(producerLoopNode, operationLoopNode)) {
          if (failed(redirectAccumulatedOperand(operand, producer, producerLoopNode, operationLoopNode, operation))) {
            return failure();
          }
        }
      }
      for (Value result : operation->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (scheduledOperations.count(user) || boundaryOperations.count(user)) {
            continue;
          }
          return user->emitError("scheduled AFIR tensor result escapes its loop body");
        }
      }
    }
    return success();
  }

  // 每项生成后刷新插入点，避免继续使用已被移入子循环的后继 Op 迭代器。
  // 子循环的插入点须在 emitLoop 返回后更新。
  LogicalResult emitTree(LoopTreeNode &loop, OpBuilder &currentBuilder, llvm::DenseMap<int64_t, Value> axisIndices) {
    for (LoopTreeItem &item : loop.items) {
      if (item.operation != nullptr) {
        item.operation->moveBefore(currentBuilder.getInsertionBlock(), currentBuilder.getInsertionPoint());
        currentBuilder.setInsertionPointAfter(item.operation);
        continue;
      }
      FailureOr<Operation *> loopOperation = emitLoop(*item.loop, currentBuilder, axisIndices);
      if (failed(loopOperation)) {
        return failure();
      }
      // 块轴不生成 `scf.for`，它的子树直接用本层 builder 生成，插入点已经停在最后一项之后。
      if (*loopOperation != nullptr) {
        currentBuilder.setInsertionPointAfter(*loopOperation);
      }
    }
    return success();
  }

  // 返回本层新建的 `scf.for`；块轴不建循环，返回空。
  FailureOr<Operation *> emitLoop(LoopTreeNode &loop, OpBuilder &currentBuilder,
                                  llvm::DenseMap<int64_t, Value> axisIndices) {
    afir::AxisAttr axis = graphAxesById.lookup(loop.axisId);
    AxisValues values = computedAxisValuesById.lookup(loop.axisId);
    if (axis.getAxisType() == afir::AxisType::BlockOuter) {
      if (!blockIndex) {
        return func.emitError("missing block-axis runtime values");
      }
      Value index = currentBuilder.create<arith::RemSIOp>(func.getLoc(), blockIndex, values.loopSize);
      tagAxisValue(index, loop.axisId, afir::AxisValueRole::AxisIndex);
      axisIndices[loop.axisId] = index;
      generateMergedAxisSourceIndices(loop.axisId, index, currentBuilder, axisIndices);
      if (failed(emitTree(loop, currentBuilder, std::move(axisIndices)))) {
        return failure();
      }
      return static_cast<Operation *>(nullptr);
    }

    bool isBlockInner = axis.getAxisType() == afir::AxisType::BlockInner;
    Value upperBound = values.loopSize;
    if (isBlockInner && axisIndices.count(axis.getSplitPairOtherId())) {
      Value outerIndex = axisIndices.lookup(axis.getSplitPairOtherId());
      AxisValues outer = computedAxisValuesById.lookup(axis.getSplitPairOtherId());
      Value isFull =
          currentBuilder.create<arith::CmpIOp>(func.getLoc(), arith::CmpIPredicate::slt, outerIndex, outer.axisSize);
      upperBound = currentBuilder.create<arith::SelectOp>(func.getLoc(), isFull, values.axisSize, values.tailSize);
      tagAxisValue(upperBound, loop.axisId, afir::AxisValueRole::ActualSize);
    }

    Value blockOffset;
    if (isBlockInner && axis.getFrom().size() == 1 && axisIndices.count(axis.getSplitPairOtherId())) {
      Value outerIndex = axisIndices.lookup(axis.getSplitPairOtherId());
      blockOffset = currentBuilder.create<arith::MulIOp>(func.getLoc(), outerIndex, values.axisSize);
    }

    auto forOp = currentBuilder.create<scf::ForOp>(func.getLoc(), zero, upperBound, one);
    forOp->setAttr(afir::kAxisBindingAttrName,
                   afir::AxisBindingAttr::get(context, loop.axisId, afir::AxisValueRole::Loop));
    boundAxisValues.insert({loop.axisId, static_cast<int64_t>(afir::AxisValueRole::Loop)});
    Block &loopBody = *forOp.getBody();
    OpBuilder bodyBuilder =
        loopBody.empty() ? OpBuilder(&loopBody, loopBody.end()) : OpBuilder(&loopBody, std::prev(loopBody.end()));
    axisIndices[loop.axisId] = forOp.getInductionVar();
    if (axis.getAxisType() == afir::AxisType::TileOuter) {
      generatePairedInnerActualSize(loop.axisId, forOp.getInductionVar(), bodyBuilder);
    } else if (axis.getAxisType() == afir::AxisType::Merged) {
      generateMergedAxisSourceIndices(loop.axisId, forOp.getInductionVar(), bodyBuilder, axisIndices);
    }
    if (blockOffset) {
      Value sourceIndex = bodyBuilder.create<arith::AddIOp>(func.getLoc(), blockOffset, forOp.getInductionVar());
      int64_t sourceId = axis.getFrom().front();
      tagAxisValue(sourceIndex, sourceId, afir::AxisValueRole::AxisIndex);
      axisIndices[sourceId] = sourceIndex;
      afir::AxisAttr sourceAxis = graphAxesById.lookup(sourceId);
      if (isMergedLike(sourceAxis)) {
        generateMergedAxisSourceIndices(sourceId, sourceIndex, bodyBuilder, axisIndices);
      } else {
        generatePairedInnerActualSize(sourceId, sourceIndex, bodyBuilder);
      }
    }
    if (failed(emitTree(loop, bodyBuilder, std::move(axisIndices)))) {
      return failure();
    }
    return forOp.getOperation();
  }

  void generatePairedInnerActualSize(int64_t outerId, Value outerIndex, OpBuilder &currentBuilder) {
    afir::AxisAttr outer = graphAxesById.lookup(outerId);
    int64_t innerId = outer.getSplitPairOtherId();
    if (innerId < 0 || !graphAxesById.count(innerId)) {
      return;
    }
    afir::AxisAttr inner = graphAxesById.lookup(innerId);
    if (!isInner(inner) || inner.getSplitPairOtherId() != outerId) {
      return;
    }
    AxisValues outerValues = computedAxisValuesById.lookup(outerId);
    AxisValues innerValues = computedAxisValuesById.lookup(innerId);
    Value isFull = currentBuilder.create<arith::CmpIOp>(func.getLoc(), arith::CmpIPredicate::slt, outerIndex,
                                                        outerValues.axisSize);
    Value actual =
        currentBuilder.create<arith::SelectOp>(func.getLoc(), isFull, innerValues.axisSize, innerValues.tailSize);
    tagAxisValue(actual, innerId, afir::AxisValueRole::ActualSize);
  }

  void generateMergedAxisSourceIndices(int64_t mergedAxisId, Value mergedIndex, OpBuilder &currentBuilder,
                                       llvm::DenseMap<int64_t, Value> &axisIndices) {
    afir::AxisAttr mergedAxis = graphAxesById.lookup(mergedAxisId);
    if (!isMergedLike(mergedAxis)) {
      return;
    }
    const SmallVector<Value> &sourceStrides = mergedAxisSourceStrides.find(mergedAxisId)->second;
    for (auto [index, sourceId] : llvm::enumerate(mergedAxis.getFrom())) {
      AxisValues source = computedAxisValuesById.lookup(sourceId);
      Value quotient = mergedIndex;
      if (index + 1 < mergedAxis.getFrom().size()) {
        quotient = currentBuilder.create<arith::DivSIOp>(func.getLoc(), mergedIndex, sourceStrides[index]);
      }
      Value sourceIndex = currentBuilder.create<arith::RemSIOp>(func.getLoc(), quotient, source.loopSize);
      tagAxisValue(sourceIndex, sourceId, afir::AxisValueRole::AxisIndex);
      axisIndices[sourceId] = sourceIndex;
      afir::AxisAttr sourceAxis = graphAxesById.lookup(sourceId);
      if (isMergedLike(sourceAxis)) {
        generateMergedAxisSourceIndices(sourceId, sourceIndex, currentBuilder, axisIndices);
      } else {
        generatePairedInnerActualSize(sourceId, sourceIndex, currentBuilder);
      }
    }
  }

  // 把物化出来的标量值认回它所属的轴。这条引用在 IR 结构上不存在——轴在循环生成后
  // 已散成普通的 arith 值——所以必须显式写在定义 op 上，供 `TransferPlan` 反查。
  void tagAxisValue(Value value, int64_t axisId, afir::AxisValueRole role) {
    Operation *definition = value.getDefiningOp();
    if (definition == nullptr) {
      return;
    }
    afir::addAxisBinding(definition, afir::AxisBindingAttr::get(context, axisId, role));
    boundAxisValues.insert({axisId, static_cast<int64_t>(role)});
  }

  static bool isInner(afir::AxisAttr axis) {
    return axis.getAxisType() == afir::AxisType::BlockInner || axis.getAxisType() == afir::AxisType::TileInner;
  }

  // Merged 和多来源 BlockOuter 都须拆回源轴序号，与合并尺寸的计算保持一致。
  static bool isMergedLike(afir::AxisAttr axis) {
    if (axis.getAxisType() == afir::AxisType::Merged) {
      return true;
    }
    return axis.getAxisType() == afir::AxisType::BlockOuter && axis.getFrom().size() > 1;
  }

  func::FuncOp func;
  MLIRContext *context;
  OpBuilder builder;
  afir::AscGraphAttrGroupsAttr graph;
  llvm::DenseMap<int64_t, afir::AxisAttr> graphAxesById;
  llvm::DenseMap<int64_t, AxisValues> computedAxisValuesById;
  llvm::DenseMap<int64_t, SmallVector<Value>> mergedAxisSourceStrides;
  llvm::DenseSet<int64_t> activeCalculatingAxes;
  afir::SymbolBindingMap tilingSymbols;
  llvm::SetVector<int64_t> loopTreeAxisIds;
  LoopTreeNode root;
  llvm::DenseMap<Operation *, LoopTreeNode *> operationPlacement;
  llvm::DenseSet<Operation *> scheduledOperations;
  llvm::DenseSet<Operation *> boundaryOperations;
  // 已生成的 (轴 id, 角色)，最终写入函数的 afir.axis_bindings。
  llvm::DenseSet<std::pair<int64_t, int64_t>> boundAxisValues;
  Value zero;
  Value one;
  Value blockIndex;
};

}  // namespace

namespace mlir {
namespace afir {

LogicalResult generateAFIRLoops(func::FuncOp func, SymbolBindingMap bindings) {
  return FunctionLoopGenerator(func, std::move(bindings)).run();
}

bool hasGeneratedLoops(func::FuncOp func) {
  WalkResult found = func.walk([](Operation *op) {
    return op->hasAttr(kAxisBindingAttrName) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return found.wasInterrupted();
}

}  // namespace afir
}  // namespace mlir
