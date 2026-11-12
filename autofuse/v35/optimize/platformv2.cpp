/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platformv2.h"

#include "attr_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "ascgraph_info_complete.h"
#include "un_alignment_strategy.h"
#include "pass_runner_v2.h"
#include "template_generator_v2.h"
#include "partition/vector_func_partitioner.h"
#include "task_generator/cube_schedule_case_generator.h"
#include "task_generator/schedule_task_generator.h"
#include "task_generator/concat_schedule_case_generator.h"
#include "task_generator/transpose_schedule_case_generator_v2.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "task_generator/recompute_case_generator.h"
#include "task_generator/split_schedule_case_generator.h"
#include "task_generator/indirect_load_schedule_case_generator.h"

namespace optimize {
constexpr size_t kMaxVecQueNum = 14UL;

PlatformV2::PlatformV2(bool is_default_enabled) {
  config_.max_que_num = kMaxVecQueNum;
  config_.is_support_compat_mode = true;
  config_.is_default_enabled = is_default_enabled;
}

af::Status PlatformV2::PartitionSubFunctions(af::AscGraph &impl_graph) {
  // UBFuse 上下文由 BufQueAllocator 通过内部 graph attr 传入; UBFuse 模板的 VectorFunc
  // codegen 不支持 Arange, 分区阶段关闭 Arange 的 VF 融合。
  bool disable_arange_vf = false;
  (void)af::AttrUtils::GetBool(af::AscGraphUtils::GetComputeGraph(impl_graph), kGraphAttrIsUBFuseContext,
                               disable_arange_vf);
  VectorFuncPartitioner partitioner(impl_graph, disable_arange_vf);
  GE_ASSERT_SUCCESS(partitioner.Partition(), "Failed to partition sub funcs for graph [%s].",
                    impl_graph.GetName().c_str());
  return af::SUCCESS;
}

std::unique_ptr<BaseAlignmentStrategy> PlatformV2::GetAlignmentStrategy() {
  return af::ComGraphMakeUnique<UnAlignmentStrategy>();
}

unique_ptr<BasePassRunner> PlatformV2::GetPassRunner() {
  return std::make_unique<PassRunnerV2>();
}

std::unique_ptr<BaseTemplateGenerator> PlatformV2::GetTemplateGenerator() {
  return af::ComGraphMakeUnique<TemplateGeneratorV2>();
}

std::unique_ptr<BackendSpec> PlatformV2::GetBackendSpec() const {
  constexpr uint32_t kConcatMaxInputNum = 512;
  constexpr int32_t kConcatAlgGather = 1;
  constexpr uint32_t kMaxLoadNum = 15;
  constexpr uint32_t kMaxInputNum = 14U;
  auto ret = af::ComGraphMakeUnique<BackendSpec>();
  ret->concat_max_input_num = kConcatMaxInputNum;
  ret->concat_alg = kConcatAlgGather;
  ret->gather_spec = {true, true, true, true, true};
  ret->slice_split_spec.split_lowered_to_split = true;
  ret->slice_split_spec.slice_fuse_with_end_dim_1 = true;
  ret->slice_split_spec.enable_split_flatten = true;
  ret->max_load_num = kMaxLoadNum;
  ret->max_input_nums_after_fuse = kMaxInputNum;
  // 10代表unit单元大小
  ret->max_group_num_per_compile_unit = 10;
  ret->enable_matmul_lowering_to_matmul = true;
  GELOGD(
      "platform_v2, enable_non_tail_gather = %d, enable_reduce_gather_fusion = %d, "
      "enable_gather_concat_fusion = %d, enable_gather_broadcast_fusion = %d, "
      "enable_gather_elementwise_forward_fusion = %d, max load_num = %u, max input num = %u",
      ret->gather_spec.enable_non_tail_gather, ret->gather_spec.enable_reduce_gather_fusion,
      ret->gather_spec.enable_gather_concat_fusion, ret->gather_spec.enable_gather_broadcast_fusion,
      ret->gather_spec.enable_gather_elementwise_forward_fusion, ret->max_load_num, ret->max_input_nums_after_fuse);
  ret->transpose_mode = static_cast<uint32_t>(TransposeMode::TRANSPOSE_MODE_UNNORMAL);
  // 256*1024是UB size
  ret->set_local_memory_size = 248 * 1024 - 8 * 1024 - 32 * 1024;
  ret->pgo_spec = {false};
  ret->is_default_enabled = config_.is_default_enabled;
  return ret;
}

Status PlatformV2::GenerateTasks(ascir::ImplGraph &optimize_graph, const OptimizerOptions &options,
                                 std::vector<ScheduleTask> &tasks) const {
  GE_ASSERT_SUCCESS(ConcatFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for concat");
  GE_ASSERT_SUCCESS(IndirectLoadScheduleCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for IndirectLoad");
  GE_ASSERT_SUCCESS(SplitFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for split");
  GE_ASSERT_SUCCESS(CubeFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for cube");
  GE_ASSERT_SUCCESS(TransposeFusionCaseGeneratorV2().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for Transpose");
  GE_ASSERT_SUCCESS(ReducePartitionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for Reduce");
  if (tasks.empty()) {
    GELOGI("Graph %s has no tasks after all generators, try RecomputationCaseGenerator.",
           optimize_graph.GetName().c_str());
    GE_ASSERT_SUCCESS(RecomputeCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                      "Failed to generate recomputation tasks for graph[%s].", optimize_graph.GetName().c_str());
  }
  return af::SUCCESS;
}

std::set<std::string> PlatformV2::BroadcastTypes() const {
  return {af::ascir_op::Broadcast::Type, af::ascir_op::Nddma::Type};
}

#define REGISTER_PLATFORM_V2(platform_name, suffix, is_default_enabled) \
  static PlatformRegistrar<PlatformV2> registrar_##suffix(platform_name, is_default_enabled)

REGISTER_PLATFORM_V2("3510", v2, true);
REGISTER_PLATFORM_V2("5102", V2_1, false);
REGISTER_PLATFORM_V2("9201", V2_3, false);
REGISTER_PLATFORM_V2("9202", V2_2, false);
}  // namespace optimize
