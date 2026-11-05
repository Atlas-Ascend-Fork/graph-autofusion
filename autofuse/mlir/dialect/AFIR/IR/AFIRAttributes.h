/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIRATTRIBUTES_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIRATTRIBUTES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::afir {

// Canonical AFIR attribute names. Keep these names centralized because they
// are part of the serialized IR contract and are consumed outside ODS.
inline constexpr llvm::StringLiteral kAFIRAttrPrefix = "afir.";
inline constexpr llvm::StringLiteral kOperandDescsAttrName = "afir.operand_descs";
inline constexpr llvm::StringLiteral kResultDescsAttrName = "afir.result_descs";
inline constexpr llvm::StringLiteral kArgDescsAttrName = "afir.arg_descs";
inline constexpr llvm::StringLiteral kDestinationDescsAttrName = "afir.destination_descs";
// 值的规划结果（空间 / 大小 / 对齐 / 资源身份）挂在**值的定义点**，一个值一份
// （设计文档 D2 / D3 / 3.2 / 5.1 步 8）。此前它是 4 个按下标对齐的数组属性
// （`operand_buffer_descs` / `result_buffer_descs` / `destination_buffer_descs` /
// `arg_buffer_descs`），同一个值在每个使用点各存一份，一致性靠消费端交叉校验。
//
// 现在的载体按定义点的形态分三处，都是同一个名字：
//   - `memref.alloc` 等定义 Op 上的 discardable 属性；
//   - `func.func` 的 arg attr（kernel 参数）与 result attr（tensor 形态的函数结果）；
//   - buffer 形态里 destination-style Op 的结果不单独存——它与所绑定的 destination
//     操作数是同一块存储，`getBufferDescAttr` 顺着 tie 取那一份。
inline constexpr llvm::StringLiteral kBufferDescAttrName = "afir.buffer_desc";
inline constexpr llvm::StringLiteral kWorkspacePlanAttrName = "afir.workspace_plan";
inline constexpr llvm::StringLiteral kAscGraphAttrName = "afir.asc_graph";
inline constexpr llvm::StringLiteral kIndexingMapsAttrName = "indexing_maps";
// kernel 参数在 ABI 里的身份，一个参数一份，挂在参数自己的定义点（arg attr）上
// （设计文档 3.6「Kernel 程序容器」）。它替代了四样东西：
//   `afir.input_index` / `afir.output_index`  一对互斥的整数 arg attr，两个都挂是合法 IR
//   `afir.num_inputs`                         参数个数的第二份编码（D2），见
//                                             `getCanonicalDataArgumentCount`
//   workspace「类型恰好相等」、TilingData「结构体名字是 TilingData」
//                                             消费端反推的规则，不在任何对外文档里
inline constexpr llvm::StringLiteral kKernelArgAttrName = "afir.kernel_arg";
// 轴信息的两个强类型载体（设计文档 5.1 步 3b）。它们替代了此前 5 个无类型扁平属性：
//   afir.sched_axis + afir.loop_axis      -> afir.sched          (#afir.sched)
//   afir.axis_id    + afir.axis_role      -> afir.axis_binding   (#afir.axis_binding)
//   afir.vectorized_axis                  -> 影子拷贝，直接删除，改读值自己的 TensorDescAttr
inline constexpr llvm::StringLiteral kSchedAttrName = "afir.sched";
inline constexpr llvm::StringLiteral kAxisBindingAttrName = "afir.axis_binding";
// 函数上的轴绑定清单，记录循环生成承诺的 axis_binding，只增不减。
// 属性不构成 SSA 使用；MemoryAllocate 的结束校验据此检查绑定值是否被重写删除。
inline constexpr llvm::StringLiteral kAxisBindingsAttrName = "afir.axis_bindings";
// Bridge 标注的完整逻辑归约域，下降侧消费；缺席表示未证明。
inline constexpr llvm::StringLiteral kReductionDomainAttrName = "afir.reduction_domain";
inline constexpr llvm::StringLiteral kReduceScratchRolesAttr = "afir.scratch_roles";
inline constexpr llvm::StringLiteral kReduceScratchScopeAttr = "afir.scratch_scope";
inline constexpr llvm::StringLiteral kSymbolicShapeAttrName = "afir.symbolic_shape";
inline constexpr llvm::StringLiteral kShapeExprAttrName = "afir.shape_expr";
inline constexpr llvm::StringLiteral kDimSymbolsAttrName = "afir.dim_symbols";
inline constexpr llvm::StringLiteral kDimSymbolIdKey = "id";
inline constexpr llvm::StringLiteral kDimSymbolArgKey = "arg";
inline constexpr llvm::StringLiteral kDimSymbolDimKey = "dim";
inline constexpr llvm::StringLiteral kDimSymbolExprKey = "expr";
// 此处曾有四个记录「哪个 pass 跑过」的标记属性：`afir.loop_generated` /
// `afir.memory_allocated` / `afir.ascendc_materialized` / `afir.pipeline_synchronized`
// （设计文档 D5、验收 8）。它们是同一事实的第二份编码——每个阶段的产物就摆在 IR 里
// （循环生成产出 `#afir.axis_binding`，资源物化产出 `ascendc` 方言的算子，流水同步
// 产出 enque / deque / pipe_barrier），而且 pass 执行状态既不是 payload 也不是候选
// 身份，按 2.2 不该出现在 `afir` 公开命名空间上。判据现在直接问 IR：
//   循环   -> `afir::hasGeneratedLoops`（dialect/AFIR/Transforms/LoopGenerate.h）
//   其余两个 -> conversion/AFIRToAscendC/AFIRToAscendCLoweringStage.h
inline constexpr llvm::StringLiteral kVarRelationsAttrName = "afir.var_relations";
inline constexpr llvm::StringLiteral kVarRelationDstStageKey = "dst_stage";
inline constexpr llvm::StringLiteral kVarRelationSrcStageKey = "src_stage";
inline constexpr llvm::StringLiteral kVarRelationDstVarKey = "dst_var";
inline constexpr llvm::StringLiteral kVarRelationSrcExprKey = "src_expr";
inline constexpr llvm::StringLiteral kShapeGuardsAttrName = "afir.shape_guards";
inline constexpr llvm::StringLiteral kShapeAssertsAttrName = "afir.shape_asserts";

// Plan and template metadata, including scheduling hints supplied by the bridge.
//
// 候选身份不在这里：`subgraph_id` / `plan_id` / `stage_id` / `template_id` /
// `template_counts_per_stage` / `score_func` 已按设计文档 5.1 步 7 迁到 `afplan`
// 命名空间（见 dialect/AFPlan/IR/AFPlan.h），`stage_count` 因恒等于
// `template_counts_per_stage` 的长度且全仓无读点而直接删除。判别：单候选时这条信息
// 还影响生成的 kernel 吗——是则留 `afir`（如 `module_kind`、`cube_type`），否则归
// `afplan`。
inline constexpr llvm::StringLiteral kApiTilingFieldsAttrName = "afir.api_tiling_fields";
// op 级的那一份（`afir.transpose` 等带 host tiling 的节点）：func 级
// `afir.api_tiling_fields` 声明嵌套 struct 的字段布局，本属性说这个节点取哪个字段。
// 名字不带 `afir.` 前缀——它走 ODS `arguments` 声明（`OptionalAttr<StrAttr>`），与前缀
// 属性那套 discardable 机制无关；同一模板里可以有多个同类型节点，func 级分不出来。
inline constexpr llvm::StringLiteral kApiTilingFieldAttrName = "api_tiling_field";
inline constexpr llvm::StringLiteral kApiTilingTypeKey = "type";
inline constexpr llvm::StringLiteral kApiTilingFieldKey = "field";
inline constexpr llvm::StringLiteral kApiTilingFieldsKey = "fields";
inline constexpr llvm::StringLiteral kBufferIdsAttrName = "afir.buffer_ids";
inline constexpr llvm::StringLiteral kCubeTypeAttrName = "afir.cube_type";
inline constexpr llvm::StringLiteral kDoubleBufferAttrName = "afir.double_buffer";
inline constexpr llvm::StringLiteral kEnableGroupParallelAttrName = "afir.enable_group_parallel";
inline constexpr llvm::StringLiteral kFrontendShapeVarsAttrName = "afir.frontend_shape_vars";
inline constexpr llvm::StringLiteral kIsReduceMemReuseAttrName = "afir.is_reduce_mem_reuse";
inline constexpr llvm::StringLiteral kModuleKindAttrName = "afir.module_kind";
inline constexpr llvm::StringLiteral kFunctionRoleAttrName = "afir.function_role";
// Values for kModuleKindAttrName. Centralized here (rather than in a
// conversion-only header) so both AFIR producers and AscendC consumers can
// depend on the dialect layer to read and write them consistently.
inline constexpr llvm::StringLiteral kPlanModuleKind = "plan";
inline constexpr llvm::StringLiteral kEntryModuleKind = "entry";
// module 级：这份 plan 是为哪个目标、按哪一组资源上限切出来的（soc / 核数 / UB 容量）。
// 生产者是 bridge（下降开始之前，从 `ascend::TargetProfile` 读），资源两项的消费者是
// artifact/TilingSource；`soc` 今天只写不读，它是 dump 的目标标识，理由见 AFIRAttrs.td。
inline constexpr llvm::StringLiteral kTargetAttrName = "afir.target";
inline constexpr llvm::StringLiteral kOriginVarsAttrName = "afir.origin_vars";
inline constexpr llvm::StringLiteral kOriginalGraphNameAttrName = "afir.original_graph_name";
inline constexpr llvm::StringLiteral kQueueIdsAttrName = "afir.queue_ids";
inline constexpr llvm::StringLiteral kWorkspaceSizeAttrName = "afir.workspace_size";
inline constexpr llvm::StringLiteral kWorkspaceTensorIdsAttrName = "afir.workspace_tensor_ids";

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_IR_AFIRATTRIBUTES_H
