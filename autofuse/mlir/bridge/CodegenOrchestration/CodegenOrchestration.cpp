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
#include "gen_tiling_impl.h"
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

// 严格模式下给每一次调用打一行可机读的标签。只在严格模式下打，缺省那一档一个字节都不多
// 输出。
//
// 两列就够，而且两列都不能由另一列推出来：
//   `gate`      是哪一道门禁拦下的，`kNone` 表示这张图是 MLIR 生成的；
//   `shortfall` 这一次下降里有没有出现过标记成「能力不足」的诊断——也就是**缺省那一档会不会
//               回落**。严格模式把结论一律压成失败，不带这一列就分不出「能力不足回落」与
//               「真坏了」，而那正是接管表三态里的两态。
//
// `shortfall` 三态而不是两态：抛异常那一支里观察者已经出了作用域（它和 `MLIRContext` 一起建
// 在 `try` 内），那时答不出来就打 `unknown`，不许顺手填 `no`——填了就是把「不知道」说成
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

// 把一道门禁自己的诊断字符串送到 stderr —— 只在严格模式下，缺省那一档一个字节都不多输出
// （与 `ReportStrictVerdict` 同一条约定）。
//
// 要它是因为这些门禁的理由今天只走 `GELOGW`，而 legacy 的 e2e 用例把 dlog 调到 ERROR，W 级
// 一句都看不到：接管表里 `gate=candidate-preparation` 那一片的「首因」列因此整片是空的，
// 55 格无诊断（2026-10-26 全量表）。下降那一侧的理由本来就是 stderr 上的 MLIR 诊断，这里
// 跟着走同一条路，接管表的测量工具不必改一行就能把它收进同一列。
//
// 前缀用 `error: ` 是为了对齐测量工具里 `grep -m1 -o 'error:.*'` 那一条，与 MLIR 诊断的行首一致。
void ReportStrictDiagnostic(const MlirCodegenOptions &options, llvm::StringRef diagnostic) {
  if (options.config != MlirCodegenConfig::kMlir || diagnostic.empty()) {
    return;
  }
  llvm::errs() << "error: " << diagnostic << "\n";
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
    // 此处不标记能力不足，因为无法区分尚未建模的真实芯片与非法 soc_version。
    // MLIR codegen 由显式开关启用，目标解析失败必须报告给调用方，不能静默回落。
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

mlir::FailureOr<common::ResourceBudget> ReadResourceBudget(const mlir::ascend::TargetProfile &profile) {
  auto ub = profile.getMemorySpec(mlir::ascend::MemorySpace::UB);
  if (mlir::failed(ub) || !ub->capacityBytes) {
    return mlir::failure();
  }
  // One target profile drives ATT, AFIR and the emitted runtime resource limit.
  return common::ResourceBudget{static_cast<uint64_t>(profile.getCoreTopology().vectorCoreCount),
                                static_cast<uint64_t>(*ub->capacityBytes)};
}

mlir::FailureOr<PlanGenerationResult> PrepareCodegenPlans(const ascir::FusedScheduledResult &schedule,
                                                          MlirCallerScene scene,
                                                          const common::ResourceBudget &resources,
                                                          std::string &diagnostic, bool &unsupported) {
  const auto mode =
      ascgen_utils::IsStaticSchedResult(schedule) ? PlanGenerationMode::Static : PlanGenerationMode::Dynamic;
  // Static schedules embed concrete dimensions, but retain all candidates.
  // The emitted host tiling program solves tile sizes and core partitioning.
  AttOptions att_options;
  // The native Inductor ABI uses modeled tiling, without the legacy PGO host protocol.
  if (scene == MlirCallerScene::kInductor) {
    att_options.emplace(att::kInternalEnableAutofusePgo, "false");
  }
  PlanGenerationResult preparation;
  if (!PreparePlanGeneration({schedule, mode, resources, scene == MlirCallerScene::kInductor, {}, att_options},
                             preparation, &diagnostic, &unsupported)) {
    return mlir::failure();
  }
  return preparation;
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
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::afir::bridge::PlanToAfir(lowering.context, fused_schedule_result, scheduled_result, selection.subgraph_id,
                                       selection.plan_id, selection.template_ids_per_stage);
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
      // 同理：流水线搭建失败是会话级问题，不是这个候选的问题。
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
                                            const mlir::bridge::TilingArtifact &tiling, MlirCallerScene scene) {
  if (scene == MlirCallerScene::kInductor || std::holds_alternative<StaticTilingData>(tiling)) {
    auto spec = buildAggregateTilingProgramSpec(plan_modules);
    if (failed(spec)) {
      return mlir::failure();
    }
    if (const auto *solved = std::get_if<StaticTilingData>(&tiling)) {
      return emitStaticTilingFunc(*spec, *solved);
    }
    return emitInductorTilingFunc(*spec, std::get<TilingCodeBundle>(tiling));
  }
  return assembleTilingSources(std::get<TilingCodeBundle>(tiling));
}

// Runs the aggregate + AutofuseLaunch driver over `plan_modules` and copies
// its artifacts, plus the per-candidate ATT tiling functions in `bundle`,
// into `result`.
bool FinalizeMlirCodegenResult(llvm::ArrayRef<mlir::ModuleOp> plan_modules, const std::string &op_name,
                               const mlir::bridge::TilingArtifact &artifact, MlirCallerScene scene,
                               codegen::CodegenResult &result) {
  codegen::CodegenResult generated;
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
  generated.kernel = mlir::ascend::kernel::inlineApiExtensionHeaders(artifacts->kernel_body, provide_api_extension) +
                     artifacts->autofuse_launch;
  auto tiling = GenerateTiling(plan_modules, artifact, scene);
  if (failed(tiling)) {
    GELOGW("MLIR codegen: failed to generate tiling sources");
    return false;
  }
  generated.tiling = std::move(*tiling);
  if (scene == MlirCallerScene::kInductor) {
    generated.tiling += "\n// AUTOFUSE_SPLIT_FILE_BEGIN: TilingDataRepr\n" + artifacts->tiling_data_repr +
                        "// AUTOFUSE_SPLIT_FILE_END: TilingDataRepr\n";
  }
  generated.tiling_data = artifacts->tiling_data;
  result = std::move(generated);
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
  // Cube 融合不在此路径范围内，严格模式 kMlir 也返回 kNotSupported。
  // 严格模式的其他失败均返回 kFailed，以区分能力范围外输入和 MLIR 接管失败。
  // gate=cube-fusion 标签供接管表分类，不依赖用例名称判断 Cube 图。
  if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
    ReportStrictVerdict(options, fused_schedule_result, MlirCodegenGate::kCubeFusion, false);
    return MlirCodegenStatus::kNotSupported;
  }
  try {
    // 在候选准备前创建 context 和观察者，使准备、下降阶段的失败使用同一分类机制。
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

    // 按 support/CapabilityShortfall.h 的诊断 metadata 分类：出现能力不足标记时允许
    // 回落 legacy，否则返回 kFailed。未标记的新失败默认视为编译错误，避免隐藏实现缺陷。
    // 严格模式统一返回 kFailed，但仍保留观察者和诊断标记；两种模式的诊断文本相同，
    // 可用严格模式的诊断解释普通模式的回落原因。
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
    std::string session_diagnostic;
    auto session = BuildPipelineSession(context, valid_soc_version, session_diagnostic);
    if (mlir::failed(session)) {
      GELOGW("MLIR codegen: pipeline session build failed: %s", session_diagnostic.c_str());
      return classify(MlirCodegenGate::kPipelineSession, "pipeline_session");
    }
    auto resources = ReadResourceBudget(*session->targetProfile);
    if (mlir::failed(resources)) {
      return classify(MlirCodegenGate::kPipelineSession, "missing_ub_capacity");
    }
    bool unsupported = false;
    auto preparation = PrepareCodegenPlans(fused_schedule_result, scene, *resources, diagnostic, unsupported);
    if (mlir::failed(preparation)) {
      GELOGW("MLIR codegen: candidate preparation failed: %s", diagnostic.c_str());
      ReportStrictDiagnostic(options, diagnostic);
      if (unsupported) {
        mlir::afir::emitCapabilityShortfall(mlir::UnknownLoc::get(&context)) << diagnostic;
      }
      return classify(MlirCodegenGate::kCandidatePreparation, "candidate_preparation");
    }

    auto plan_modules = BuildCandidateAfirModules(PlanLoweringSession{context, *session}, fused_schedule_result,
                                                  preparation->selections, options);
    if (mlir::failed(plan_modules)) {
      return classify(MlirCodegenGate::kLowering, "candidate_lowering");
    }

    // Assembly errors retain their category; the outer mode chooses failure or
    // legacy fallback. Finalization commits all artifacts together on success.
    if (!FinalizeMlirCodegenResult(plan_modules->views, fused_schedule_result.fused_graph_name.GetString(),
                                   preparation->tiling, scene, result)) {
      ReportStrictVerdict(options, fused_schedule_result, MlirCodegenGate::kArtifactAssembly, observer.sawShortfall());
      return finish(MlirCodegenStatus::kFailed, "artifact_generation");
    }
    // 接管了也照旧报 `shortfall`：一个候选被能力层拒掉、另一个成了，这张图仍然是 MLIR 生成的，
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
