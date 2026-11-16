/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRScalarReadSync.cpp - Record scalar read synchronization sources -===//
//
// Broadcast 的 Duplicate 实现和二元、三元 elementwise 的标量操作数需要通过 GetValue(0)
// 读取片上张量。读取前必须等待写入该存储的流水线完成此前的任务。
//
// 写入者及其流水线必须在 AFIR 计算算子尚未下降时分析：下降过程中，IR 混合了 AFIR 和目标方言，
// 已下降的写入者不再携带 AFIR trait。此时重新分析，可能得到不完整的结果；能力查询与下降
// 也可能因看到的 IR 不同而选择不同的同步事件。
//
// 本 pass 提前分析写入者的流水线，将结果保存在 afir.scalar_read_sync 属性中，供能力查询与
// 下降共用。这与 ADR-19 在 !afir.token 类型中记录流水线对的做法一致：在信息完整的阶段完成
// 分析，后续阶段直接使用结果。
//
// 执行顺序要求：
//   - 在 AFIRMemoryAllocate 之后执行，此时已生成队列和 TBuf，可以判断不同张量是否属于同一存储。
//   - 在实现选择之前执行，能力查询需要据此判断是否支持标量读取。
//   - 当前排在 AFIRExecConditionPass 之前。分析不依赖控制流位置，将算子移入 scf.if 不影响结果，
//     具体语义见 analyzeScalarWriterPipes 的注释。
//
// 标注范围：此时尚未选择实现，无法确定哪些输入最终会按标量读取。因此，对所有可能满足条件的
// 输入记录同步来源：输入是片上张量，且各向量化轴的 repeat 均可证明为 1。Broadcast 的
// isConstantOnVectorizedAxes 和 elementwise 的 isSingletonTensorInput 都蕴含这一条件，
// 因而标注范围覆盖两个调用方支持的输入形态。
//
//===----------------------------------------------------------------------===//

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "conversion/AFIRToAscendC/Implementations/Synchronization.h"

#include "dialect/AFIR/Analysis/VectorSegmentation.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::ascend::kernel {
namespace {

// 判断该输入是否可能按标量读取，标注范围见文件头说明。
bool mayBeReadAsScalar(Operation *op, unsigned inputIndex) {
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(op);
  if (failed(partition) || inputIndex >= partition->inputCount) {
    return false;
  }
  if (!isa<ascendc::LocalTensorType>(partition->inputs(op)[inputIndex].getType())) {
    return false;
  }
  auto descriptors = op->getAttrOfType<ArrayAttr>(afir::kOperandDescsAttrName);
  if (!descriptors || inputIndex >= descriptors.size()) {
    return false;
  }
  auto descriptor = dyn_cast<afir::TensorDescAttr>(descriptors[inputIndex]);
  if (!descriptor || descriptor.getVectorizedAxis().empty() ||
      descriptor.getAxis().size() != descriptor.getRepeats().size()) {
    return false;
  }
  for (int64_t vectorizedAxis : descriptor.getVectorizedAxis()) {
    FailureOr<size_t> position = afir::axisPosition(descriptor, vectorizedAxis);
    if (failed(position) || !afir::provablyOne(descriptor.getRepeats()[*position])) {
      return false;
    }
  }
  return true;
}

void recordScalarReadSync(Operation *op) {
  FailureOr<afir::OperandPartition> partition = afir::getOperandPartition(op);
  if (failed(partition) || partition->inputCount == 0) {
    return;
  }
  MLIRContext *context = op->getContext();
  SmallVector<Attribute> entries;
  bool anyRecorded = false;
  for (unsigned index = 0; index < partition->inputCount; ++index) {
    SmallVector<Attribute> pipes;
    if (mayBeReadAsScalar(op, index)) {
      FailureOr<SmallVector<afir::Pipe>> writers = analyzeScalarWriterPipes(op, index);
      if (succeeded(writers)) {
        for (afir::Pipe pipe : *writers) {
          pipes.push_back(afir::PipeAttr::get(context, pipe));
        }
      }
    }
    anyRecorded |= !pipes.empty();
    entries.push_back(ArrayAttr::get(context, pipes));
  }
  // 所有项均为空时省略属性，避免在 dump 中记录无用信息。属性缺失与对应项为空都表示
  // 没有可用的同步来源；需要标量读取的实现据此判为不支持。
  if (anyRecorded) {
    op->setAttr(afir::kScalarReadSyncAttrName, ArrayAttr::get(context, entries));
  }
}

struct AFIRScalarReadSyncPass : public PassWrapper<AFIRScalarReadSyncPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRScalarReadSyncPass)

  StringRef getArgument() const final {
    return "afir-scalar-read-sync";
  }

  StringRef getDescription() const final {
    return "Record which pipelines wrote the storage each scalar-read operand reads from";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<afir::AFIRDialect, ascendc::AscendCDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) {
        continue;
      }
      // `afir.vector_func` 的 region 里是一段在寄存器里算完的数据流，没有 tile 级的标量读。
      func.walk([](Operation *op) {
        if (afir::isAfirOperation(op) && !afir::isInsideVectorFunc(op)) {
          recordScalarReadSync(op);
        }
      });
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRScalarReadSyncPass() {
  return std::make_unique<AFIRScalarReadSyncPass>();
}

void registerAFIRScalarReadSyncPass() {
  PassRegistration<AFIRScalarReadSyncPass>();
}

}  // namespace mlir::ascend::kernel
