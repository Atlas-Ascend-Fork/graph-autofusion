/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRVerifyKernelComplete.cpp - Executable completeness criteria ----===//

#include "dialect/AFIR/Transforms/VerifyKernelComplete.h"

#include "dialect/AFIR/Analysis/RaceCheck.h"
#include "dialect/AFIR/IR/AFIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::afir;

namespace mlir::afir {
namespace {

constexpr llvm::StringLiteral kIncompleteDiagnostic = "Incomplete";

// C1 检查全部 memref SSA 值，包括函数参数、算子结果和嵌套 region 的块参数。
// scf.for 的 memref iter_arg 也属于 buffer，必须检查其地址空间；
// 只收集前两类会遗漏多缓冲轮转常用的循环携带值。
void collectBufferValues(func::FuncOp func, llvm::SmallVectorImpl<Value> &buffers) {
  auto collect = [&](Value value) {
    if (isa<MemRefType>(value.getType())) {
      buffers.push_back(value);
    }
  };
  for (BlockArgument argument : func.getArguments()) {
    collect(argument);
  }
  func.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      collect(result);
    }
    // 算子自己的 entry block 参数不算（那是 `func.func` 的参数，上面已经收过）；
    // 这里要的是它内嵌区域的块参数。
    if (op == func.getOperation()) {
      return;
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument argument : block.getArguments()) {
          collect(argument);
        }
      }
    }
  });
}

// 地址空间在 AFIR 层是 `#afir.space<*>`（设计文档 3.1：它是值的固有属性，写在类型上）。
// 物化之后换成目标自己的编码，本检查在两种形态下都要能用，所以两种都认——不认的只有
// 「压根没写」。目标编码一律是整数属性（AscendC 侧是 CCE-C 的地址空间号），所以这里不
// 需要认识任何一个具体目标。
bool hasTraceableAddressSpace(MemRefType type) {
  Attribute space = type.getMemorySpace();
  if (!space) {
    return false;
  }
  return isa<SpaceAttr>(space) || isa<IntegerAttr>(space);
}

LogicalResult verifyBufferTraceability(func::FuncOp func) {
  llvm::SmallVector<Value> buffers;
  collectBufferValues(func, buffers);
  LogicalResult result = success();
  for (Value buffer : buffers) {
    auto type = cast<MemRefType>(buffer.getType());
    if (!type.hasRank()) {
      result = failure();
      emitError(buffer.getLoc()) << kIncompleteDiagnostic
                                 << " C1: buffer has no rank, so its shape and strides "
                                    "cannot be traced";
      continue;
    }
    if (!hasTraceableAddressSpace(type)) {
      result = failure();
      emitError(buffer.getLoc()) << kIncompleteDiagnostic
                                 << " C1: buffer does not record an address space; AFIR "
                                    "carries it on the type as '#afir.space<*>'";
    }
  }
  // 所有权与释放点。ADR-10 选 `memref.alloc` + `memref.dealloc` 而不是 `alloca`，理由
  // 就是这一条：释放点要在 IR 里看得见。没有配对的 dealloc 时所有权无从追踪——这正是
  // 5.2 把 C1 记为「部分（所有权未 SSA 化）」的那一半。
  func.walk([&](memref::AllocOp alloc) {
    bool released =
        llvm::any_of(alloc.getMemref().getUsers(), [](Operation *user) { return isa<memref::DeallocOp>(user); });
    if (!released) {
      result = failure();
      alloc.emitError() << kIncompleteDiagnostic
                        << " C1: allocation has no 'memref.dealloc', so its release point cannot be traced";
    }
  });
  return result;
}

LogicalResult verifyAsyncCompletion(func::FuncOp func) {
  RaceChecker checker;
  llvm::SmallVector<RaceFinding> findings = checker.check(func);
  if (findings.empty()) {
    return success();
  }
  // 竞争检查器自己的诊断已经说明了是哪一对访问、什么冒险、跨几次迭代；这里只补上
  // 「这是 C2 不成立」这一层，免得读者要自己把两份文档对起来。
  func.emitError() << kIncompleteDiagnostic << " C2: " << findings.size()
                   << " access pair(s) have no representable completion condition";
  return reportRaceFindings(findings);
}

// AFIR 层允许出现的方言：AFIR 自己，加上它明确复用的那几个标准方言（约束 6）。
bool isRepresentableDialect(Dialect *dialect) {
  if (dialect == nullptr) {
    return false;
  }
  llvm::StringRef name = dialect->getNamespace();
  static constexpr llvm::StringLiteral kAllowed[] = {AFIRDialect::getDialectNamespace(),
                                                     arith::ArithDialect::getDialectNamespace(),
                                                     bufferization::BufferizationDialect::getDialectNamespace(),
                                                     BuiltinDialect::getDialectNamespace(),
                                                     cf::ControlFlowDialect::getDialectNamespace(),
                                                     func::FuncDialect::getDialectNamespace(),
                                                     index::IndexDialect::getDialectNamespace(),
                                                     math::MathDialect::getDialectNamespace(),
                                                     memref::MemRefDialect::getDialectNamespace(),
                                                     scf::SCFDialect::getDialectNamespace()};
  return llvm::is_contained(kAllowed, name);
}

LogicalResult verifyAfirOnlyRepresentation(func::FuncOp func) {
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    if (isRepresentableDialect(op->getDialect())) {
      return;
    }
    result = failure();
    op->emitError() << kIncompleteDiagnostic << " C3: '" << op->getName()
                    << "' is outside AFIR and the standard dialects it reuses, so this data or control flow is not "
                       "expressible in AFIR alone";
  });
  return result;
}

LogicalResult verifyDeterministicLowering(func::FuncOp func, ImplementationQuery implementations) {
  if (!implementations.contains) {
    return func.emitError() << kIncompleteDiagnostic << " C4: requires a frozen " << implementations.targetName
                            << " implementation registry and a selected target";
  }
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (dialect == nullptr || dialect->getNamespace() != AFIRDialect::getDialectNamespace()) {
      return;
    }
    if (implementations.contains(op)) {
      return;
    }
    result = failure();
    // 今天缺实现的失败是 `applyFullConversion` 的一句 "failed to legalize operation"，
    // 既不指明是哪条判据、也不说缺的是什么。C4 要求的正是「unsupported 编译期返回
    // 原因」，所以这条诊断必须指名道姓，并且在下降开始之前就发生。
    op->emitError() << kIncompleteDiagnostic << " C4: no " << implementations.targetName
                    << " implementation is registered for '" << op->getName()
                    << "', so lowering it to target code is not determined";
  });
  return result;
}

}  // namespace

llvm::StringRef getCompletenessCriterionName(CompletenessCriterion criterion) {
  switch (criterion) {
    case CompletenessCriterion::BufferTraceability:
      return "c1";
    case CompletenessCriterion::AsyncCompletion:
      return "c2";
    case CompletenessCriterion::AfirOnlyRepresentation:
      return "c3";
    case CompletenessCriterion::DeterministicLowering:
      return "c4";
  }
  return "unknown";
}

LogicalResult verifyCompletenessCriterion(func::FuncOp func, CompletenessCriterion criterion,
                                          ImplementationQuery implementations) {
  switch (criterion) {
    case CompletenessCriterion::BufferTraceability:
      return verifyBufferTraceability(func);
    case CompletenessCriterion::AsyncCompletion:
      return verifyAsyncCompletion(func);
    case CompletenessCriterion::AfirOnlyRepresentation:
      return verifyAfirOnlyRepresentation(func);
    case CompletenessCriterion::DeterministicLowering:
      return verifyDeterministicLowering(func, implementations);
  }
  return success();
}

}  // namespace mlir::afir
