/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/CodegenOrchestration/CodegenOrchestration.h"
#include "common/autofuse_config/auto_fuse_config_parser.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "common/ge_common/debug/log.h"
#include "common_utils.h"
#include "ascir_utils.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

#include "bridge/ScheduleToAfir/ScheduleToAfir.h"
#include "support/CapabilityShortfall.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "bridge/PlanSelector/PlanSelector.h"
#include "bridge/AttSchemaAdapter/AttSchemaAdapter.h"
#include "common/tiling_schema/tiling_field_schema.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCPipeline.h"
#include "artifact/KernelSource/AggregateLaunchDriver.h"
#include "artifact/KernelSource/Emit/ApiExtensionInliner.h"
#include "ascendc_api_registry.h"
#include "artifact/TilingSource/TilingProgramSpec.h"
#include "artifact/TilingSource/TilingFuncEmitter.h"
#include "target/Ascend/TargetProfileLoader.h"
#include "target/Ascend/TargetSelection.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::bridge {

MlirCodegenOptions ReadMlirCodegenOptions() {
  MlirCodegenOptions options;
  ge::AutoFuseEnvConfigParser parser({}, {kMlirCodegenConfigOptionName.str()});
  const auto flags = parser.Parse();
  const auto config = flags.find(kMlirCodegenConfigOptionName.str());
  if (config == flags.end() || config->second == "none") {
    options.config = MlirCodegenConfig::kNone;
  } else if (config->second == "mlir") {
    options.config = MlirCodegenConfig::kMlir;
  } else if (config->second == "fallback") {
    options.config = MlirCodegenConfig::kFallback;
  } else {
    options.config = MlirCodegenConfig::kInvalid;
  }
  options.dump_root = ascir::utils::GetDumpFilePrefix();

  return options;
}

namespace {

llvm::StringRef ToString(MlirCodegenGate gate) {
  switch (gate) {
    case MlirCodegenGate::kNone:
      return "none";
    case MlirCodegenGate::kCubeFusion:
      return "cube-fusion";
    case MlirCodegenGate::kSocVersion:
      return "soc-version";
    case MlirCodegenGate::kCandidatePreparation:
      return "candidate-preparation";
    case MlirCodegenGate::kPipelineSession:
      return "pipeline-session";
    case MlirCodegenGate::kLowering:
      return "lowering";
    case MlirCodegenGate::kArtifactAssembly:
      return "artifact-assembly";
    case MlirCodegenGate::kException:
      return "exception";
  }
  return "?";
}

// 严格模式下给每一次调用打一行可机读的标签。**只在严格模式下打**，缺省那一档一个字节都不多
// 输出。
//
// 两列就够，而且两列都不能由另一列推出来：
//   `gate`      是哪一道闸门拦下的，`kNone` 表示这张图是 MLIR 发的；
//   `shortfall` 这一次下降里有没有出现过标记成「能力不足」的诊断——也就是**缺省那一档会不会
//               回落**。严格模式把结论一律压成失败，不带这一列就分不出「能力不足回落」与
//               「真坏了」，而那正是接管表三态里的两态。
//
// `shortfall` 三态而不是两态：抛异常那一支里观察者已经出了作用域（它和 `MLIRContext` 一起建
// 在 `try` 内），那时答不出来就打 `unknown`，**不许顺手填 `no`**——填了就是把「不知道」说成
// 「没有」，而接管表会照着它把一行归到硬失败那一桶。
//
// 走 stderr 而不走 dlog：legacy 的 e2e 用例把 dlog 调到 ERROR，W 级一句都看不到；而 MLIR 的
// 诊断本来就打在 stderr 上，标签与它放在一起，一次重定向就能同时拿到理由和第一条诊断。
void ReportStrictVerdict(const MlirCodegenOptions &options, const ascir::FusedScheduledResult &fused_schedule_result,
                         MlirCodegenGate gate, std::optional<bool> shortfall) {
  if (options.config != MlirCodegenConfig::kMlir) {
    return;
  }
  const char *graph_name = fused_schedule_result.fused_graph_name.GetString();
  llvm::errs() << kMlirCodegenStrictMarker << " gate=" << ToString(gate)
               << " shortfall=" << (shortfall.has_value() ? (*shortfall ? "yes" : "no") : "unknown")
               << " graph=" << (graph_name == nullptr ? "(unnamed)" : graph_name) << "\n";
}

// 把一道闸门自己的诊断字符串送到 stderr —— **只在严格模式下**，缺省那一档一个字节都不多输出
// （与 `ReportStrictVerdict` 同一条约定）。
//
// 要它是因为这些闸门的理由今天只走 `GELOGW`，而 legacy 的 e2e 用例把 dlog 调到 ERROR，W 级
// 一句都看不到：接管表里 `gate=candidate-preparation` 那一片的「首因」列因此整片是空的，
// 55 格无诊断（2026-10-26 全量表）。下降那一侧的理由本来就是 stderr 上的 MLIR 诊断，这里
// 跟着走同一条路，接管表的量具不必改一行就能把它收进同一列。
//
// 前缀用 `error: ` 是为了对齐量具里 `grep -m1 -o 'error:.*'` 那一条，与 MLIR 诊断的行首一致。
void ReportStrictDiagnostic(const MlirCodegenOptions &options, llvm::StringRef diagnostic) {
  if (options.config != MlirCodegenConfig::kMlir || diagnostic.empty()) {
    return;
  }
  llvm::errs() << "error: " << diagnostic << "\n";
}

// A static schedule has no frontend symbols to substitute. SelectStaticWinner
// accepts an empty shape map and can still retain all candidates when ATT
// costs depend on values resolved later by tiling.
mlir::bridge::ConcreteShape BuildConcreteShape(const ascir::FusedScheduledResult &fused_schedule_result) {
  (void)fused_schedule_result;
  return {};
}

mlir::FailureOr<mlir::ascend::kernel::AFIRToAscendCPipelineSession> BuildPipelineSession(mlir::MLIRContext &context,
                                                                                         const std::string &soc_version,
                                                                                         std::string &diagnostic) {
  if (soc_version.empty()) {
    diagnostic = "MLIR codegen requires CodegenOptions::soc_version to be set explicitly";
    return mlir::failure();
  }
  auto selection = mlir::ascend::TargetSelection::builtinPlatform(soc_version);
  llvm::raw_string_ostream os(diagnostic);
  auto profile = mlir::ascend::TargetProfileLoader::load(selection, os);
  if (mlir::failed(profile)) {
    // **不标记成能力不足。** 这里分不开两件事：一颗 MLIR 还没建模的真芯片（该回落），与一个
    // 根本解析不出来的 soc_version（配置坏了，该大声失败）。而 MLIR codegen 是显式开关打开
    // 的，开的人需要知道「这颗芯片这条路不覆盖」，不是静默什么都没发生。
    return mlir::failure();
  }
  auto shared_profile = std::make_shared<const mlir::ascend::TargetProfile>(std::move(*profile));
  return mlir::ascend::kernel::createAFIRToAscendCPipelineSession(context, shared_profile);
}

std::string ResolveSocVersion(const std::string &soc_version, std::string &diagnostic) {
  if (!soc_version.empty()) {
    return soc_version;
  }

  const char *runtime_soc_version = aclrtGetSocName();
  if (runtime_soc_version == nullptr || runtime_soc_version[0] == '\0') {
    diagnostic = "MLIR codegen requires soc_version; aclrtGetSocName() returned no SoC name";
    return {};
  }
  return runtime_soc_version;
}

std::string GetCandidatePassDumpDir(const std::string &dump_root, size_t subgraph_id, size_t plan_id) {
  if (dump_root.empty()) {
    return "";
  }
  // dump_root 一定以 "/" 结尾
  return dump_root + "mlir_subgraph_" + std::to_string(subgraph_id) + "_plan_" + std::to_string(plan_id);
}

void ConfigurePassDump(mlir::PassManager &pass_manager, const std::string &dump_root, size_t subgraph_id,
                       size_t plan_id) {
  const std::string candidate_dir = GetCandidatePassDumpDir(dump_root, subgraph_id, plan_id);
  if (candidate_dir.empty()) {
    return;
  }
  std::error_code ec;
  if (!std::filesystem::create_directories(candidate_dir, ec) && ec) {
    GELOGW("MLIR codegen: failed to create pass dump directory %s: %s", candidate_dir.c_str(), ec.message().c_str());
    return;
  }
  pass_manager.enableIRPrintingToFileTree([](mlir::Pass *, mlir::Operation *) { return false; },
                                          [](mlir::Pass *, mlir::Operation *) { return true; }, true, false, false,
                                          candidate_dir);
}

void NormalizePassDumpPrefixes(const std::string &dump_root, size_t subgraph_id, size_t plan_id) {
  const std::filesystem::path candidate_dir(GetCandidatePassDumpDir(dump_root, subgraph_id, plan_id));
  std::error_code ec;
  if (candidate_dir.empty() || !std::filesystem::exists(candidate_dir, ec)) {
    return;
  }

  std::vector<std::filesystem::path> files;
  for (std::filesystem::recursive_directory_iterator iter(candidate_dir, ec), end; iter != end && !ec;
       iter.increment(ec)) {
    if (iter->is_regular_file(ec) && iter->path().extension() == ".mlir") {
      files.push_back(iter->path());
    }
  }
  for (const auto &file : files) {
    const std::string name = file.filename().string();
    const size_t separator = name.find('_');
    if (separator != 1 || !std::isdigit(static_cast<unsigned char>(name[0]))) {
      continue;
    }
    const auto normalized = file.parent_path() / ("0" + name);
    std::error_code rename_ec;
    std::filesystem::rename(file, normalized, rename_ec);
    if (rename_ec) {
      GELOGW("MLIR codegen: failed to normalize pass dump filename %s: %s", file.c_str(), rename_ec.message().c_str());
    }
  }
}

// Two pieces of MLIR context that are always threaded together through the
// per-candidate lowering loop: the owning MLIRContext and the pipeline
// session built from it. Grouped so BuildCandidateAfirModules stays within
// the project's 5-parameter limit.
struct PlanLoweringSession {
  mlir::MLIRContext &context;
  const mlir::ascend::kernel::AFIRToAscendCPipelineSession &session;
};

// Outputs of PrepareMlirCandidates, grouped into one return value.
struct MlirCandidatePreparation {
  // AFIR and ATT use the same narrowed schedule and contiguous candidate IDs.
  ascir::FusedScheduledResult narrowed_result;
  std::vector<mlir::bridge::PlanSelection> selections;
  std::string op_name;
  mlir::bridge::TilingCodeBundle bundle;
};

// Outputs of BuildCandidateAfirModules: `owned` keeps each module alive; the
// corresponding entries in `views` are the non-owning views passed to the
// aggregate stage.
struct BuiltPlanModules {
  llvm::SmallVector<mlir::ModuleOp> views;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owned;
};

mlir::LogicalResult SetTargetAttr(mlir::ModuleOp module, const mlir::ascend::TargetProfile &profile) {
  auto ubSpec = profile.getMemorySpec(mlir::ascend::MemorySpace::UB);
  if (mlir::failed(ubSpec) || !ubSpec->capacityBytes) {
    GELOGW("MLIR codegen: target profile does not describe UB capacity");
    return mlir::failure();
  }
  // 值域（soc 非空、核数为正、UB 容量够放下 tiling 自身、两者都放得进 uint32）由
  // `#afir.target` 的 verifier 判。走 `getChecked` 而不是 `get`，是因为后者把 verify
  // 放在 assert 里——release 构建下越界会被带进 IR，直到 artifact 层生成 host tiling
  // 时才暴露，而 soc 根本没有读点，空串会一路无声带到 dump 里。
  auto target = mlir::afir::TargetAttr::getChecked(
      [&]() { return module->emitError("MLIR codegen: invalid target description"); }, module.getContext(),
      mlir::StringAttr::get(module.getContext(), profile.getIdentity().socVersion),
      profile.getCoreTopology().vectorCoreCount, *ubSpec->capacityBytes);
  if (!target) {
    return mlir::failure();
  }
  module->setAttr(mlir::afir::kTargetAttrName, target);
  return mlir::success();
}

// Selects the winning/candidate plans, narrows the schedule and generates ATT tiling.
mlir::FailureOr<MlirCandidatePreparation> PrepareMlirCandidates(
    const ascir::FusedScheduledResult &fused_schedule_result, MlirCallerScene scene, std::string &diagnostic,
    bool &unsupported) {
  std::vector<mlir::bridge::PlanSelection> selections;
  const bool selected =
      ascgen_utils::IsStaticSchedResult(fused_schedule_result)
          ? mlir::bridge::SelectStaticWinner(fused_schedule_result, BuildConcreteShape(fused_schedule_result), {},
                                             selections, &diagnostic, &unsupported)
          : mlir::bridge::SelectDynamicCandidates(fused_schedule_result, selections, &diagnostic, &unsupported);
  if (!selected) {
    return mlir::failure();
  }

  if (!mlir::bridge::FilterSelectionsForLowering(fused_schedule_result, selections, &diagnostic)) {
    return mlir::failure();
  }

  MlirCandidatePreparation prep;
  if (!NarrowFusedResultToSelections(fused_schedule_result, selections, prep.narrowed_result, prep.selections,
                                     &diagnostic)) {
    return mlir::failure();
  }
  prep.op_name = prep.narrowed_result.fused_graph_name.GetString();

  mlir::common::TilingFieldSchema schema;
  if (!mlir::common::BuildTilingFieldSchema(prep.narrowed_result, schema, &diagnostic)) {
    return mlir::failure();
  }

  AttOptions att_options;
  att_options.emplace("tiling_data_type_name", "AutofuseTilingData");
  const mlir::bridge::AttRequest att_request{prep.op_name, prep.narrowed_result, prep.selections,
                                             schema,       att_options,          scene == MlirCallerScene::kInductor};
  if (mlir::bridge::RunAttForCandidates(att_request, prep.bundle, &diagnostic, &unsupported) != af::SUCCESS) {
    return mlir::failure();
  }
  return prep;
}

// Lower every scheduled plan. A failure rejects the complete MLIR result.
mlir::FailureOr<BuiltPlanModules> BuildCandidateAfirModules(const PlanLoweringSession &lowering,
                                                            const ascir::FusedScheduledResult &fused_schedule_result,
                                                            const std::vector<mlir::bridge::PlanSelection> &selections,
                                                            const MlirCodegenOptions &options) {
  BuiltPlanModules built;
  built.owned.reserve(selections.size());
  for (const auto &selection : selections) {
    const auto &scheduled_result =
        fused_schedule_result.node_idx_to_scheduled_results[selection.subgraph_id][selection.plan_id];
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::afir::bridge::PlanToAfir(
        lowering.context, fused_schedule_result, scheduled_result, selection.subgraph_id, selection.plan_id);
    if (!module) {
      return mlir::failure();
    }
    if (mlir::failed(SetTargetAttr(module.get(), *lowering.session.targetProfile))) {
      // target profile 与候选无关（它来自 soc version），所以这一条不是候选的问题，
      // 丢弃它换下一个也不会好转——保持整图失败。
      return mlir::failure();
    }
    mlir::afir::bridge::MaybeDumpAfirModule("RunMlirCodegen", module.get(), options.dump_root);

    mlir::PassManager pm(&lowering.context);
    ConfigurePassDump(pm, options.dump_root, selection.subgraph_id, selection.plan_id);
    if (mlir::failed(mlir::ascend::kernel::addPlanCandidateLoweringPipeline(pm, lowering.session))) {
      // 同理：流水线**搭建**失败是会话级问题，不是这个候选的问题。
      GELOGW("MLIR codegen: AFIRToAscendC pipeline setup failed for subgraph %zu plan %zu", selection.subgraph_id,
             selection.plan_id);
      return mlir::failure();
    }
    const auto run_status = pm.run(module->getOperation());
    NormalizePassDumpPrefixes(options.dump_root, selection.subgraph_id, selection.plan_id);
    if (mlir::failed(run_status)) {
      return mlir::failure();
    }

    built.views.push_back(module.get());
    built.owned.push_back(std::move(module));
  }
  return built;
}

mlir::FailureOr<std::string> GenerateTiling(llvm::ArrayRef<mlir::ModuleOp> plan_modules,
                                            const mlir::bridge::TilingCodeBundle &bundle, MlirCallerScene scene) {
  if (scene == MlirCallerScene::kInductor) {
    auto spec = buildAggregateTilingProgramSpec(plan_modules);
    if (failed(spec)) {
      return mlir::failure();
    }
    return emitInductorTilingFunc(*spec, bundle);
  }
  return assembleTilingSources(bundle);
}

// Runs the aggregate + AutofuseLaunch driver over `plan_modules` and copies
// its artifacts, plus the per-candidate ATT tiling functions in `bundle`,
// into `result`.
bool FinalizeMlirCodegenResult(llvm::ArrayRef<mlir::ModuleOp> plan_modules, const std::string &op_name,
                               const mlir::bridge::TilingCodeBundle &bundle, MlirCallerScene scene,
                               codegen::CodegenResult &result) {
  std::string diagnostic;
  auto artifacts = mlir::ascend::kernel::runAggregateLaunchDriver(plan_modules, op_name, &diagnostic);
  if (mlir::failed(artifacts)) {
    GELOGW("MLIR codegen: aggregate launch driver failed: %s", diagnostic.c_str());
    return false;
  }
  // The kernel is compiled as one self-contained translation unit with no
  // include path of its own, so the Autofuse API extension headers the
  // operations asked for are inlined here rather than left as `#include`s.
  // Layer 1 (the ops) decided which headers; the registry below is a flat
  // name -> text table shared by the mem-based and reg-based families, whose
  // header names do not collide, so this stage needs no platform knowledge.
  const auto provide_api_extension = [](llvm::StringRef header) -> llvm::StringRef {
    return codegen::AscendCApiRegistry::GetInstance().GetFileContent(header.str());
  };
  result.kernel = mlir::ascend::kernel::inlineApiExtensionHeaders(artifacts->kernel_body, provide_api_extension) +
                  artifacts->autofuse_launch;
  auto tiling = GenerateTiling(plan_modules, bundle, scene);
  if (failed(tiling)) {
    GELOGW("MLIR codegen: failed to generate tiling sources");
    return false;
  }
  result.tiling = std::move(*tiling);
  result.tiling_data = artifacts->tiling_data;
  return true;
}

}  // namespace

MlirCodegenStatus RunMlirCodegen(const ascir::FusedScheduledResult &fused_schedule_result, MlirCallerScene scene,
                                 const std::string &soc_version, codegen::CodegenResult &result) {
  const auto options = ReadMlirCodegenOptions();
  if (options.config == MlirCodegenConfig::kInvalid) {
    GELOGE(af::FAILED, "MLIR codegen: invalid --%s value; expected none, mlir, or fallback",
           kMlirCodegenConfigOptionName.data());
    return MlirCodegenStatus::kFailed;
  }
  const auto finish = [&](MlirCodegenStatus status, llvm::StringRef reason) {
    if (options.config == MlirCodegenConfig::kNone || status == MlirCodegenStatus::kSucceeded) {
      return status;
    }
    if (options.config == MlirCodegenConfig::kFallback) {
      GELOGW("MLIR codegen: falling back to legacy after --%s=fallback (%s)", kMlirCodegenConfigOptionName.data(),
             reason.str().c_str());
      return MlirCodegenStatus::kNotSupported;
    }
    return MlirCodegenStatus::kFailed;
  };
  if (options.config == MlirCodegenConfig::kNone) {
    return finish(MlirCodegenStatus::kNotSupported, "disabled");
  }
  // **这道闸门严格模式（`kMlir`）也不收紧**：Cube 融合明确不在本条路的范围（见
  // elementwise/broadcast 收尾设计「范围」一节），收紧它只会让 legacy 的 Cube 用例整片变红，
  // 量不出任何东西。于是 `kNotSupported` 在严格模式下只剩这一个来路，「回落」与「不在范围」
  // 因此分得开，而标签里的 `gate=cube-fusion` 让接管表不必按用例名猜哪些是 Cube。
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ReportStrictVerdict(options, fused_schedule_result, MlirCodegenGate::kCubeFusion, false);
    return MlirCodegenStatus::kNotSupported;
  }
  try {
    // Context 建在最前面，为的是让**所有**失败都经过同一个观察者。此前它建在候选准备
    // 之后，于是「准备阶段的失败算哪一类」只能另立一套判据——两套判据说同一件事，
    // 迟早说岔。
    mlir::MLIRContext context;
    if (!options.dump_root.empty()) {
      // Module-scope IR printing is not safe while pass managers run in
      // parallel. Diagnostics are intentionally deterministic and opt-in.
      context.disableMultithreading();
    }
    context.getOrLoadDialect<mlir::afir::AFIRDialect>();
    context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::index::IndexDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<ascendc::AscendCDialect>();

    // 「能力不足」与「坏了」的唯一判据（support/CapabilityShortfall.h）：
    // 本次下降里出现过标记过的诊断就是前者，回落给 legacy；否则大声失败。
    //
    // **缺省是 `kFailed`**，所以新增一条没标记的失败路径只会多报一次失败，不会把一个真
    // bug 静默藏起来。
    //
    // 严格模式把这个判据短路成「一律 `kFailed`」：量「谁接管了这张图」时，回落与接管都是绿
    // 的，两者分不开。**短路的只有结论，不是观察者**——诊断照旧带标记、照旧打印，所以同一
    // 次失败在两档取值下给出的文本逐字相同，可以拿严格模式那一遍的诊断去解释缺省那一遍的
    // 回落。
    mlir::afir::CapabilityShortfallObserver observer(context);
    const auto classify = [&observer, &finish, &options, &fused_schedule_result](MlirCodegenGate gate,
                                                                                 llvm::StringRef stage) {
      ReportStrictVerdict(options, fused_schedule_result, gate, observer.sawShortfall());
      return observer.sawShortfall() ? finish(MlirCodegenStatus::kNotSupported, "capability_shortfall")
                                     : finish(MlirCodegenStatus::kFailed, stage);
    };

    std::string diagnostic;
    const auto valid_soc_version = ResolveSocVersion(soc_version, diagnostic);
    if (valid_soc_version.empty()) {
      GELOGW("MLIR codegen: failed to resolve SoC version: %s", diagnostic.c_str());
      return classify(MlirCodegenGate::kSocVersion, "soc_resolution");
    }
    // 候选准备里只有一类失败是「能力不足」：某个 stage 的每一个模板都被三条判据里的某一条
    // 拒了（`PlanSelection.h` 的 `unsupported`）。标记在这里发而不在那一层发，是因为判据只有
    // 一个——`Diagnostic` 上的 metadata——而那一层没有 MLIR context，发不出带标记的诊断。
    //
    // **剩下的失败一律不标**，尤其是 ATT 生成的 tiling_func 引用了 `TilingFieldSchema` 里没有
    // 的字段那一支（`AttSchemaAdapter.cpp:130`，reg 侧 16 格 / mem 侧 11 格）：两侧字段拼法对
    // 不上是缺陷，不是能力边界，标成能力不足就把它静默回落掉了。
    bool unsupported = false;
    auto preparation = PrepareMlirCandidates(fused_schedule_result, scene, diagnostic, unsupported);
    if (mlir::failed(preparation)) {
      GELOGW("MLIR codegen: candidate preparation failed: %s", diagnostic.c_str());
      ReportStrictDiagnostic(options, diagnostic);
      if (unsupported) {
        mlir::afir::emitCapabilityShortfall(mlir::UnknownLoc::get(&context)) << diagnostic;
      }
      return classify(MlirCodegenGate::kCandidatePreparation, "candidate_preparation");
    }

    std::string session_diagnostic;
    auto session = BuildPipelineSession(context, valid_soc_version, session_diagnostic);
    if (mlir::failed(session)) {
      GELOGW("MLIR codegen: pipeline session build failed: %s", session_diagnostic.c_str());
      return classify(MlirCodegenGate::kPipelineSession, "pipeline_session");
    }

    auto plan_modules = BuildCandidateAfirModules(PlanLoweringSession{context, *session}, preparation->narrowed_result,
                                                  preparation->selections, options);
    if (mlir::failed(plan_modules)) {
      return classify(MlirCodegenGate::kLowering, "candidate_lowering");
    }

    // 产物装配阶段**不走 classify**：这里没有任何能力判断，失败一律是坏了。而且它会先写
    // `result.kernel` 再生成 tiling，中途失败时 `result` 已经被动过——回落让 legacy 再写
    // 一遍是未定义的，宁可大声失败。
    if (!FinalizeMlirCodegenResult(plan_modules->views, preparation->op_name, preparation->bundle, scene, result)) {
      ReportStrictVerdict(options, fused_schedule_result, MlirCodegenGate::kArtifactAssembly, observer.sawShortfall());
      return finish(MlirCodegenStatus::kFailed, "artifact_generation");
    }
    // 接管了也照旧报 `shortfall`：一个候选被能力层拒掉、另一个成了，这张图仍然是 MLIR 发的，
    // 但那条诊断值得在表里看见。
    ReportStrictVerdict(options, fused_schedule_result, MlirCodegenGate::kNone, observer.sawShortfall());
    return finish(MlirCodegenStatus::kSucceeded, "completed");
  } catch (const std::exception &e) {
    GELOGW("MLIR codegen: unexpected exception: %s", e.what());
    ReportStrictVerdict(options, fused_schedule_result, MlirCodegenGate::kException, std::nullopt);
    return finish(MlirCodegenStatus::kFailed, "exception");
  } catch (...) {
    GELOGW("MLIR codegen: unexpected unknown exception");
    ReportStrictVerdict(options, fused_schedule_result, MlirCodegenGate::kException, std::nullopt);
    return finish(MlirCodegenStatus::kFailed, "exception");
  }
}

}  // namespace mlir::bridge
