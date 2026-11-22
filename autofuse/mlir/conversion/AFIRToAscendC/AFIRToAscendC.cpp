/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "conversion/AFIRToAscendC/Implementations/ReducePadding.h"
#include "conversion/AFIRToAscendC/AFIRPipelineSync.h"
#include "conversion/AFIRToAscendC/AFIRSessionScopedPass.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/Capability/ImplementationSelection.h"

#include <memory>
#include <utility>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir::afir;

namespace mlir::ascend::kernel {
namespace {

// buffer 形态的判据问接口，不列 op 名：列出来就是第四处「新增 op 要改的地方」，
// 而且漏一个的表现是这道前置检查被静默跳过，不是报错。
bool requiresCanonicalDescriptors(Operation *operation) {
  auto destinationStyle = dyn_cast<AFIRDestinationStyleInterface>(operation);
  return destinationStyle && destinationStyle.isBufferizedForm();
}

bool hasCanonicalDescriptors(Operation *operation) {
  FailureOr<OperandPartition> partition = getOperandPartition(operation);
  if (failed(partition)) {
    return false;
  }
  unsigned inputCount = partition->inputCount;
  unsigned destinationCount = partition->destinationCount;
  auto hasArray = [&](StringRef name, unsigned expected) {
    auto array = operation->getAttrOfType<ArrayAttr>(name);
    return expected == 0 ? !array || array.empty() : array && array.size() == expected;
  };
  // 判据只看使用点的 tensor descriptor。值的规划结果（`afir.buffer_desc`）在值的
  // 定义点，由真正用到它的消费端各自查、各自报缺失；放进这里会让一个 Op 能不能下降
  // 取决于别的 Op 有没有标注（设计文档 5.1 步 8）。
  return hasArray(kOperandDescsAttrName, inputCount) && hasArray(kDestinationDescsAttrName, destinationCount) &&
         hasArray(kResultDescsAttrName, operation->getNumResults());
}

LogicalResult verifyCanonicalLoweringInput(Operation *root) {
  LogicalResult result = success();
  root->walk([&](Operation *operation) {
    if (failed(result) || !requiresCanonicalDescriptors(operation)) {
      return;
    }
    if (!hasCanonicalDescriptors(operation)) {
      operation->emitOpError("conversion requires canonical tensor descriptors");
      result = failure();
    }
  });
  return result;
}

struct ConvertAFIRToAscendCPass : public PassWrapper<ConvertAFIRToAscendCPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAFIRToAscendCPass)

  ConvertAFIRToAscendCPass(std::shared_ptr<const ascend::instruction::ImplementationRegistry> registry,
                           std::shared_ptr<const ascend::TargetProfile> target)
      : registry_(std::move(registry)), target_(std::move(target)) {}

  ConvertAFIRToAscendCPass(const ConvertAFIRToAscendCPass &other)
      : PassWrapper(other), registry_(other.registry_), target_(other.target_) {}

  StringRef getArgument() const final {
    return "convert-afir-to-ascendc";
  }

  StringRef getDescription() const final {
    return "Lower selected AFIR operations to AscendC";
  }

  // `index` / `math` 是符号表达式下降那一组 pattern 建的（见 runOnOperation 里为什么把它
  // 加进来）；`arith` 两边都建。
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<afir::AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, index::IndexDialect,
                    math::MathDialect, scf::SCFDialect, emitc::EmitCDialect>();
  }

  void runOnOperation() override {
    if (!registry_ || !target_ || !registry_->isFrozen()) {
      getOperation().emitError("requires a frozen Ascend implementation registry and target profile");
      signalPassFailure();
      return;
    }
    // 同步不再在这里顺手做：它产出的 AFIR 同步算子需要先经过 implementation selection，
    // 而 selection 已经跑过了。同步现在是流水线里 selection 之前的一步
    // （`createAFIRPipelineSyncPass`）。单独跑 `--convert-afir-to-ascendc` 的用例因此
    // 只处理已经同步过、或本来就不需要同步的输入。
    if (failed(verifyCanonicalLoweringInput(getOperation())) ||
        failed(ascend::instruction::validateSelectedImplementations(getOperation(), *registry_, [this](Operation *op) {
          return registry_->containsSourceOp(op->getName());
        }))) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = &getContext();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    // `!afir.token` 下降成 AscendC 的事件号。方向（pipe 对）不进转换后的类型——它在
    // AscendC 侧由 `HardEvent` 模板参数携带，由同步族的 pattern 从原类型读出来
    // （ADR-19）。宽度与 `api/ascendc` 既有用法一致（i8）。
    typeConverter.addConversion([](afir::TokenType type) -> Type { return IntegerType::get(type.getContext(), 8); });
    RewritePatternSet patterns(context);
    ascend::instruction::PatternBuildContext patternContext{typeConverter, *target_};
    ascend::instruction::populateEnabledPatterns(patterns, *registry_, patternContext);
    // 此 pass 仍需注册符号表达式下降 pattern。AFIRMemoryAllocate 虽已清理输入中的
    // symbolic_eval，但搬运 pattern 的 materializeSizeAt 会为非字面量 stride / extent
    // 新建该算子。若不注册下降，dialect conversion 会回滚整个 pattern，并在 afir.load
    // 等根算子上报告 legalize 失败，掩盖实际缺失的符号算子处理。
    afir::detail::populateSymbolicLoweringPatterns(patterns);
    populateReducePaddingPatterns(patterns);

    ConversionTarget conversionTarget(*context);
    conversionTarget.markUnknownOpDynamicallyLegal([](Operation *operation) {
      Dialect *dialect = operation->getDialect();
      if (dialect && dialect->getNamespace() == afir::AFIRDialect::getDialectNamespace()) {
        return false;
      }
      return true;
    });
    conversionTarget.addIllegalDialect<afir::AFIRDialect>();
    conversionTarget.addLegalDialect<ascendc::AscendCDialect>();
    conversionTarget.addIllegalOp<ascendc::ReduceInitOp>();
    if (failed(applyFullConversion(getOperation(), conversionTarget, std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  std::shared_ptr<const ascend::instruction::ImplementationRegistry> registry_;
  std::shared_ptr<const ascend::TargetProfile> target_;
};

struct AFIRConvertToAscendCPass : public AFIRSessionScopedPass<AFIRConvertToAscendCPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AFIRConvertToAscendCPass)

  StringRef getArgument() const final {
    return "convert-afir-to-ascendc";
  }

  StringRef getDescription() const final {
    return "Lower selected AFIR operations to AscendC";
  }

  LogicalResult populatePipeline(OpPassManager &pipeline, const AFIRToAscendCPipelineSession &session) {
    pipeline.addPass(createConvertAFIRToAscendCPass(session.implementationRegistry, session.targetProfile));
    return success();
  }

  // Mirrors the wrapped ConvertAFIRToAscendCPass's own getDependentDialects.
  void getStaticDependentDialects(DialectRegistry &registry) const {
    registry.insert<AFIRDialect, arith::ArithDialect, ascendc::AscendCDialect, index::IndexDialect, math::MathDialect,
                    scf::SCFDialect, emitc::EmitCDialect>();
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertAFIRToAscendCPass(
    std::shared_ptr<const ascend::instruction::ImplementationRegistry> registry,
    std::shared_ptr<const ascend::TargetProfile> target) {
  return std::make_unique<ConvertAFIRToAscendCPass>(std::move(registry), std::move(target));
}

std::unique_ptr<Pass> createConvertAFIRToAscendCStandalonePass() {
  return std::make_unique<AFIRConvertToAscendCPass>();
}

void registerConvertAFIRToAscendCStandalonePass() {
  PassRegistration<AFIRConvertToAscendCPass>();
}

}  // namespace mlir::ascend::kernel
