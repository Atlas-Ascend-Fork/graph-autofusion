/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INDIRECT_LOAD_UTILS_H__
#define __INDIRECT_LOAD_UTILS_H__

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "ascir.h"
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ascgen_utils::indirect_load {
constexpr size_t kInputTensorIndex = 0UL;
constexpr size_t kIndexTensorIndex = 1UL;
constexpr char kTemplateLogicalViewAttr[] = "af.internal.indirect_load.logical_view";
constexpr char kAccessInfoAttr[] = "af.internal.indirect_load.access_info";
constexpr char kLoweringMetadataAttr[] = "af.internal.indirect_load.lowering_metadata";

enum class TemplateRole : int64_t {
  kNone,
  kSimdInputPre,
  kSimdInputPreStridedUbPath,
  kSimdIndexPre,
  kSimtInputBoundary,
  kSimtDirectGmBoundary,
  kSimtInlineTransform,
  kSimtFanoutBranch,
  kSimtOp,
  kSkInputBoundary,
  kStridedUbPath,
};

enum class Implementation : int64_t {
  // The default SIMD implementation uses MicroAPI.
  kDefault,
  kGatherApi,
};

struct TemplateBehavior {
  bool excludes_tiling_group = false;
  bool skips_main_schedule_tiling = false;
  bool skips_api_emit = false;
  bool uses_direct_gm_pipeline = false;
  bool skips_ub_lifecycle = false;
  bool skips_input_lifecycle = false;
  bool preserves_vectorized_axis = false;
};

struct TemplateAxes {
  af::AxisId outer_axis = af::kIdNone;
  af::AxisId inner_axis = af::kIdNone;
  af::AxisId input_inner_axis = af::kIdNone;
  af::AxisId index_inner_axis = af::kIdNone;
  af::AxisId tile_outer_axis = af::kIdNone;
  af::AxisId tile_inner_axis = af::kIdNone;
  std::vector<af::AxisId> vectorized_axes;
  bool synthetic_outer = false;
};

struct LogicalTensorView {
  std::vector<af::AxisId> axis_ids;
  std::vector<af::Expression> sizes;
  std::vector<af::Expression> strides;
};

enum class IndirectLoadLayoutKind : int64_t {
  kDense = 0,
  kZeroStrideCompact = 1,
  kStrided = 2,
  kUnsupported = 3,
};

enum class IndirectLoadStoreKind : int64_t {
  kDense = 0,
  kSliceStrided = 1,
  kUnsupported = 2,
};

struct IndirectLoadTensorLayout : LogicalTensorView {
  IndirectLoadLayoutKind kind = IndirectLoadLayoutKind::kUnsupported;
  std::vector<af::Expression> physical_repeats;
};

struct TemplateLogicalView {
  IndirectLoadTensorLayout input;
  IndirectLoadTensorLayout index;
  LogicalTensorView output;
  // SK 窗口复用语义：单窗口 outer 行数（含 tile 尺寸符号，如 outerTb_size）。仅当全量驻留 tmp 估算
  // 超出平台 UB 时记录（避免 SK 被 StaticUbTemplateFilter 误淘汰）。注意 Expression 默认构造为
  // 合法的空名符号，IsValid 恒真，不能作为"未设置"哨兵；未设置判定以 window_iter_axes 为空为准
  // （填充时两者原子设置），此时消费方按全量 outer 积兜底，与历史行为逐字节等价。
  af::Expression window_outer_count;
  // SK 窗口迭代轴（tile_outer 轴 ID），同时是窗口语义的"已设置"哨兵。窗口循环串行复用同一块
  // tmp，静态估算需跳过该轴及其派生外层轴；为空表示无窗口复用语义。
  std::vector<af::AxisId> window_iter_axes;
};

struct IndirectLoadAccessInfo {
  enum class Kind : int64_t { kGeneric, kEmbeddingLike };

  Kind kind = Kind::kGeneric;
  int64_t axis = -1L;
  af::Expression input_slice_bytes;
  af::Expression index_varying_extent;
  bool can_use_simt_structured = false;
  // SIMD uses a contiguous payload suffix and a zero-stride index suffix.
  // This is stricter than the generalized SIMT structured-layout condition.
  bool can_use_simd_embedding = false;
  // [SK窗口可行性] index 源头是否物理收敛（沿生产链回溯到 Load，其原始视图在
  // gather 轴 payload 维上零贡献/退化）。源头收敛时 gather 的引用按行/列成组，
  // SK 行窗口只需装载被引用分片；源头不收敛（逐元素独立引用，如
  // index=[13,10,20000] 全轴 stride 非零）时 SK 窗口必须覆盖 axis 维全部，
  // 退化为全量。kind 为 generic 但源头收敛的形态（如尾轴 gather 无 payload
  // 后缀）不受 kind 标签影响。
  bool index_source_converged = false;
};

enum class SimdFallback : int64_t {
  kRegisterGather = 0,
  kGatherApi = 1,
  kStrided = 2,
};

struct SimdLoweringMetadata {
  SimdFallback fallback = SimdFallback::kRegisterGather;
  bool try_embedding = false;
};

enum class SimtAddressPolicy : int64_t {
  kStaticPowerOfTwo = 0,
  kStaticInner = 1,
  kStructuredMagic = 2,
  kEmbedding = 3,
  kRecursive = 4,
  kStrided = 5,
};

enum class SimtOffsetWidth : int64_t {
  kUint32,
  kUint64,
};

// [SIMT 融合寄存器压力模型] 位宽感知常数与档位映射（调度期 L4 门控/线程档封顶与
// codegen 位宽漂移防护共用）。64 位基准为既有保守值（行为不变）；全成员 uint32 时
// fused body 选 32 位 OffsetType（codegen 判定），改用 32 位细化常数（fused 成员
// 上限与线程档均放宽）。取值依据、推导与档位配额表见设计文档「寄存器压力模型」。
constexpr uint64_t kSimtFusedRegisterBase64 = 24U;
constexpr uint64_t kSimtFusedRegisterPerMember64 = 18U;
constexpr uint64_t kSimtFusedRegisterPerSideLoad64 = 5U;
constexpr uint64_t kSimtFusedRegisterPerChainNode = 3U;  // 混合宽度，两档共用
constexpr uint64_t kSimtFusedRegisterBase32 = 23U;
constexpr uint64_t kSimtFusedRegisterPerMember32 = 10U;
constexpr uint64_t kSimtFusedRegisterPerSideLoad32 = 4U;
// 最低档配额：超过即 fused 发射不可行（降级 L4/V2）。
constexpr uint64_t kSimtFusedMinTierQuota = 127U;

// 位宽感知寄存器需求：all_offset_uint32 用 32 位细化常数，否则 64 位基准。
inline uint64_t SimtFusedRegisterDemand(const bool all_offset_uint32, const uint64_t member_count,
                                        const uint64_t side_load_count, const uint64_t chain_node_count) {
  if (all_offset_uint32) {
    return kSimtFusedRegisterBase32 + kSimtFusedRegisterPerMember32 * member_count +
           kSimtFusedRegisterPerSideLoad32 * side_load_count + kSimtFusedRegisterPerChainNode * chain_node_count;
  }
  return kSimtFusedRegisterBase64 + kSimtFusedRegisterPerMember64 * member_count +
         kSimtFusedRegisterPerSideLoad64 * side_load_count + kSimtFusedRegisterPerChainNode * chain_node_count;
}

// 线程档映射：需求可容纳的最高档。
inline uint32_t SimtFusedThreadTier(const uint64_t register_demand) {
  if (register_demand <= 16U) {
    return 2048U;
  }
  if (register_demand <= 32U) {
    return 1024U;
  }
  if (register_demand <= 64U) {
    return 512U;
  }
  return 256U;
}

enum class SimtLoadAddressSource : int64_t {
  kZeroOffset,
  kOutputOffset,
  kIndexOffset,
};

struct SimtPolicyMetadata {
  SimtAddressPolicy policy = SimtAddressPolicy::kRecursive;
  SimtOffsetWidth offset_width = SimtOffsetWidth::kUint64;
  uint64_t inner_span = 0U;
  uint64_t output_axis_span = 0U;
  uint64_t input_axis_stride = 0U;
  uint64_t input_axis_span = 0U;
  uint64_t input_stride_mask = 0U;
  uint64_t index_stride_mask = 0U;
  std::vector<af::Expression> runtime_params;
};

struct SimtGmTensorMetadata {
  ascir::TensorId value_tensor_id = af::kIdNone;
  ascir::TensorId gm_tensor_id = af::kIdNone;
  af::DataType dtype = af::DT_UNDEFINED;
  bool is_scalar = false;
};

struct SimtLoadMetadata {
  std::string node_name;
  SimtLoadAddressSource address_source = SimtLoadAddressSource::kOutputOffset;
  bool use_logical_offset = false;
  LogicalTensorView physical_view;
  // [行级广播 side-input] 尾轴 stride==0 且 size==1 的广播 side-input（每行读一个
  // 标量，如生产 gather+softmax 图的 load3 [8,2048,1]/strides=[2048,1,0]）：调度期
  // SIMT 边界的轴 split/merge 会把其视图改写为 rank 不匹配且尾段尺寸符号化的形态，
  // codegen 的兜底（稠密尾段取模）要求编译期常量而失效。此处保留改写前的原始
  // 视图，兜底时按原始语义重建坐标（行号定位，与 split 无关）。
  LogicalTensorView original_view;
  bool is_row_broadcast = false;
};

struct SimtOutputChainMetadata {
  std::vector<std::string> node_names;
  ascir::TensorId result_tensor_id = af::kIdNone;
  ascir::TensorId target_tensor_id = af::kIdNone;
  af::DataType dtype = af::DT_UNDEFINED;
  bool local_target = false;
  std::vector<af::Expression> target_sizes;
  std::vector<af::Expression> target_strides;
  af::Expression target_offset{af::ops::Zero};
};

struct SimtLoweringMetadata {
  bool has_post_reduce = false;
  std::vector<std::string> index_node_names;
  std::vector<std::string> output_node_names;
  std::vector<SimtLoadMetadata> index_loads;
  std::vector<SimtLoadMetadata> output_loads;
  std::vector<SimtOutputChainMetadata> output_chains;
  std::vector<SimtGmTensorMetadata> gm_tensors;
  SimtPolicyMetadata policy;
  ascir::TensorId index_result_tensor_id = af::kIdNone;
  ascir::TensorId value_tensor_id = af::kIdNone;
  ascir::TensorId output_result_tensor_id = af::kIdNone;
  af::DataType index_dtype = af::DT_UNDEFINED;
  af::DataType output_dtype = af::DT_UNDEFINED;
};

// [SIMT index 闭包预计算] SIMT body 的 Index() 逐元素求值 index 链（含软件展开的
// Ln，~25 条标量指令/元素）是 gather 形态向量管线的主要开销。当 index 链是输出
// 坐标的纯函数（无 GM Load）且值域可静态证明 fp32 精确时，可按 tile 向量化预计算
// 为 UB int32 索引表，body 的 Index() 退化为一次表读。表复用 value tensor 的存储
// （int32 视图，值 dtype ≥4B 时同宽），scratch 走 tmp buffer 协议（CalcTmpBufSize
// → b0_<id>，与 SIMD offsets buffer 同机制）。本计划由调度期（tmp 尺寸规划）与
// codegen 期（发射）共用，保证两处判定位点一致。
enum class SimtIndexPrepassNodeKind : uint8_t {
  kLeafArangeInnerRamp,    // 尾轴坐标：tile 内 ramp（每行复用 [0, inner)）
  kLeafArangeRowConstant,  // 外轴坐标：行内常量（按行 Duplicate）
  kLeafScalar,             // 编译期常量（*s 变体标量或 Duplicate 物化）
  kOpBinary,               // Add/Sub/Mul/Min/Max（含标量操作数）
  kOpUnary,                // Abs/Ln
  kOpCompare,              // Gt/Ge/Lt/Le/Eq/Ne → uint8 mask
  kOpWhere,                // Select(mask, a, b)（标量分支 Duplicate 物化）
  kCastNoop,               // 值域已证 fp32 精确：int↔float 同域透传
  kCastTruncToInt,         // float→int 截断：经 int32 槽往返两段 Cast 还原截断语义
  kPassthrough             // Store/Transpose：值透传
};

// 预计算节点的操作数描述：链内节点引用（下标）或标量字面量。
struct SimtIndexPrepassOperand {
  bool is_node = false;
  size_t node_index = 0UL;   // is_node 时：plan.nodes 下标
  std::string scalar_value;  // 标量字面量（已按值 dtype 格式化）
};

// Arange 叶的坐标参数（发射期还原坐标表达式用）。
struct SimtIndexPrepassArangeInfo {
  af::Expression view_stride;  // 逻辑输出视图上该轴的 stride
  af::Expression view_size;    // 逻辑输出视图上该轴的 size
  af::Expression base;         // Arange base 属性
  af::Expression step;         // Arange step 属性
};

struct SimtIndexPrepassNodePlan {
  SimtIndexPrepassNodeKind kind = SimtIndexPrepassNodeKind::kPassthrough;
  size_t op_code = 0UL;  // kOpBinary/kOpUnary/kOpCompare/kOpWhere 的算子分发码
  std::vector<SimtIndexPrepassOperand> operands;
  size_t slot = 0UL;  // 输出所在 4B 槽位（kOpCompare 为 mask 槽位）
  // 节点内获取即释放的临时槽：kLeafArangeInnerRamp 的 int32 ramp、kCastTruncToInt
  // 的 int32 往返、kOpWhere 的标量分支物化（Duplicate），均计入峰值。
  std::vector<size_t> temp_slots;
  SimtIndexPrepassArangeInfo arange;
};

struct SimtIndexPrepassPlan {
  bool eligible = false;
  size_t four_byte_slot_count = 0UL;            // fp32/int32 scratch 槽数峰值（含临时槽）
  size_t mask_slot_count = 0UL;                 // uint8 比较 mask 槽数
  af::Expression inner_size;                    // 逻辑输出尾轴尺寸（scratch 尺寸表达式用）
  size_t result_slot = 0UL;                     // 链结果（透传解析后）所在 4B 槽位
  std::vector<SimtIndexPrepassNodePlan> nodes;  // 拓扑序，与链节点一一对应
};

// 判定 SIMT index 链可否向量化预计算并给出发射计划。纯图结构推导（不依赖调度
// 状态）：从 IL 节点 index 输入反向收集闭包，逐节点校验算子集、Arange 叶形状
// （尾轴 ramp / 行常量）、值域区间（fp32 精确性）与 scratch 槽位需求。任一条件
// 不满足时 eligible=false，调用方回退既有逐元素路径。
af::Status BuildSimtIndexPrepassPlan(const af::AscNode &indirect_load, SimtIndexPrepassPlan &plan);

// [SIMT index 闭包预计算] 调度期 stash 的 tmp 需求：CalcTmpBufSize 所在的 impl 注册
// 库不能回调本库符号，scratch 字节数经节点 ext attr 传递；读写均为头文件内联
// （两库各自弱符号，与既有内联工具同模式）。
struct SimtIndexPrepassTmpDemand {
  bool eligible = false;
  af::Expression tmp_bytes;  // Align 后的 scratch 字节数
};
constexpr char kSimtPrepassTmpDemandAttr[] = "af.internal.indirect_load.simt_prepass_tmp_demand";

// 预计算 scratch 字节数：4B 槽与 uint8 mask 槽共用槽宽 Align(inner*4,32)（mask 槽
// 按 4B 槽宽对齐覆盖），总量 32B 对齐。调度期 stash 与 codegen 期漂移复验共用此式。
inline af::Expression CalcSimtIndexPrepassTmpBytes(const SimtIndexPrepassPlan &plan) {
  const af::Expression slot_bytes = af::sym::Align(plan.inner_size * af::Symbol(4U), 32U);
  return af::sym::Align(af::Symbol(plan.four_byte_slot_count + plan.mask_slot_count) * slot_bytes, 32U);
}

inline af::Status SetSimtIndexPrepassTmpDemand(const af::AscNodePtr &node, const SimtIndexPrepassPlan &plan) {
  const auto op_desc = node == nullptr ? nullptr : node->GetOpDesc();
  if (op_desc == nullptr || !plan.eligible) {
    return af::SUCCESS;
  }
  const SimtIndexPrepassTmpDemand demand{true, CalcSimtIndexPrepassTmpBytes(plan)};
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSimtPrepassTmpDemandAttr, demand),
                 "Set IndirectLoad SIMT prepass tmp demand failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

inline SimtIndexPrepassTmpDemand GetSimtIndexPrepassTmpDemand(const af::AscNode &node) {
  const auto op_desc = node.GetOpDesc();
  if (op_desc == nullptr) {
    return SimtIndexPrepassTmpDemand{};
  }
  return op_desc->TryGetExtAttr(kSimtPrepassTmpDemandAttr, SimtIndexPrepassTmpDemand{});
}

// [漂移防护] codegen 期重建 plan 后的 scratch 需求复验：预留尺寸以调度期 stash 的
// tmp_bytes 为准，两份图副本之间图可能被改写，重建链需要更多槽位时越界写会穿透
// 同 b0 的其他 tmp 用户——超出预留（含不可判定）一律回退逐元素路径。
inline bool SimtIndexPrepassPlanFitsTmpDemand(const SimtIndexPrepassPlan &plan,
                                              const SimtIndexPrepassTmpDemand &demand) {
  if (!plan.eligible || !demand.eligible) {
    return false;
  }
  return af::SymbolicUtils::StaticCheckLe(CalcSimtIndexPrepassTmpBytes(plan), demand.tmp_bytes) == af::TriBool::kTrue;
}

struct IndirectLoadLoweringMetadata {
  uint32_t version = 1U;
  int64_t axis = -1L;
  af::AxisId outer_axis = af::kIdNone;
  TemplateLogicalView logical_view;
  IndirectLoadAccessInfo access_info;
  SimdLoweringMetadata simd;
  SimtLoweringMetadata simt;
};

TemplateBehavior GetTemplateBehavior(const af::AscNodePtr &node);
TemplateRole GetTemplateRole(const af::AscNodePtr &node);
af::AscNodePtr GetPostReduceConsumer(const af::AscNodePtr &node);
af::AscNodePtr GetPostReduceInputProducer(const af::AscNodePtr &node);
// Returns the terminal tensor ID of a chain whose API emission is skipped.
ascir::TensorId FindSkippedChainResultTensor(const af::AscNodePtr &root);
bool ShouldSkipTpipeTensorCollection(const af::AscNodePtr &node);
af::Status InheritTemplateRoleIfIL(af::AscGraph &graph, const std::string &vf_node_name, const af::AscNodePtr &src);
af::Status SetTemplateRole(const af::AscNodePtr &node, TemplateRole role);
af::Status SetTemplateAxes(const af::AscNodePtr &node, const TemplateAxes &axes);
af::Status GetTemplateAxes(const af::AscNodePtr &node, TemplateAxes &axes);
af::Status SetTemplateLogicalView(const af::AscNodePtr &node, const TemplateLogicalView &view);
af::Status GetTemplateLogicalView(const af::AscNodePtr &node, TemplateLogicalView &view);
af::Status SetIndirectLoadAccessInfo(const af::AscNodePtr &node, const IndirectLoadAccessInfo &info);
af::Status GetIndirectLoadAccessInfo(const af::AscNodePtr &node, IndirectLoadAccessInfo &info);
// [SK窗口可行性] 现场判定 index 源头收敛性（不依赖候选流程构建的 attr，跨图副本安全）。
bool IsIndexSourceConverged(const af::AscNodePtr &indirect_load);
af::Status SetLoweringMetadata(const af::AscNodePtr &node, const IndirectLoadLoweringMetadata &metadata);
af::Status GetLoweringMetadata(const af::AscNodePtr &node, IndirectLoadLoweringMetadata &metadata);
bool HasLoweringMetadata(const af::AscNodePtr &node);
af::Status FinalizeLoweringMetadata(const af::AscNodePtr &node, bool &is_supported);
af::Status AnalyzeIndirectLoadAccess(const af::AscNodePtr &node, const TemplateLogicalView &logical_view,
                                     IndirectLoadAccessInfo &info);
af::Status SetImplementation(const af::AscNodePtr &node, Implementation implementation);
af::Status GetImplementation(const af::AscNodePtr &node, Implementation &implementation);
af::Status ClassifyIndirectLoadLayout(const LogicalTensorView &logical, IndirectLoadTensorLayout &layout,
                                      bool allow_non_overlapping_zero_stride = false);
af::Status ValidateIndirectLoadOutputLayout(const LogicalTensorView &output);
// 判断 load 的物理视图是否与输出逻辑视图覆盖同一 dense 连续区域（语义等价）：
// 1) load 视图各有效轴（size!=1 且 stride!=0）的 stride 满足后缀乘积连续性；
// 2) 所有有效轴 sizes 的乘积与输出视图 sizes 乘积符号相等。
// 满足时 load 的线性偏移与 output_index 相同，可直接使用线性偏移，无需坐标重建。
// 调度会把普通节点视图 merge/split 到模板轴空间（如 outer 被拆为
// [s3*s4*s5/Tb, Tb]），符号级全等比较会漏判这类等价视图。
bool IsDenseEquivalentView(const LogicalTensorView &load_view, const LogicalTensorView &output_view);
af::Status ClassifyIndirectLoadStoreTarget(const std::vector<af::Expression> &sizes,
                                           const std::vector<af::Expression> &strides, const af::Expression &offset,
                                           IndirectLoadStoreKind &kind);
bool ShouldApplyInputInnerVectorization(const af::AscNodePtr &node);
// 双 gather 水平融合：按 GetAllNodes 顺序收集图中全部 IndirectLoad 节点（确定性顺序）。
std::vector<af::AscNodePtr> GetAllIndirectLoadNodes(const af::AscGraph &graph);
// 多 gather 水平融合（SIMT）：组成员标记。携带该标记的 IL 走组员版 SIMT lowering 计划——
// 计划仅含本路 index 链与 input 读取，输出为写 UB 局部目标（value tensor），共享归并/输出
// 链不进入 SIMT 计划（作为普通向量代码在全部 SIMT invocation 之后执行）。
af::Status SetSimtGroupMember(const af::AscNodePtr &node);
bool IsSimtGroupMember(const af::AscNodePtr &node);
// seq-V1 融合发射标记：组的 primary IL 携带该标记时，codegen 生成单个 fused invocation
// （IndirectLoadSimtFused，逐路串行 load + 寄存器归并）；非 primary 成员跳过发射。
af::Status SetSimtFusedPrimary(const af::AscNodePtr &node, const std::vector<std::string> &member_names);
bool IsSimtFusedPrimary(const af::AscNodePtr &node);
std::vector<std::string> GetSimtFusedMemberNames(const af::AscNodePtr &node);
// 六级降级阶梯（L2/L3）：fused primary 携带的归并信息。register_merge=true 时归并树在
// FusedEval 内按拓扑序折叠为标量表达式（寄存器归并，L2）；false 时各路 value 写各自 UB
// 局部目标、归并链由向量代码执行（UB 归并，L3），merge 字段不消费。max_thread_tier 为
// 资源压力模型推得的线程档上限（模型规格见设计文档「寄存器压力模型」）。
// 防漂移约定：节点名跨图副本稳定（唯一存储介质）；视图/分类/输出链 tensor id 由 codegen
// 在当前图上重建（见 BuildSimtMergeSideLoadMetadata 与「漂移防护」注释）。
struct SimtFusedMergeInfo {
  bool register_merge = false;
  uint32_t max_thread_tier = 2048U;
  // R_est 组件计数（位宽感知门控的输入；codegen 位宽漂移防护按 64 位基准重算档位用）。
  uint32_t member_count = 0U;
  uint32_t side_load_count = 0U;
  uint32_t chain_node_count = 0U;
  std::vector<std::string> merge_node_names;   // 归并树闭包节点名（拓扑序，不含成员 IL 与 Store）
  std::vector<std::string> merge_store_names;  // Store 终端节点名（codegen 按名重解析输出链 tensor id）
};
af::Status SetSimtFusedMergeInfo(const af::AscNodePtr &node, const SimtFusedMergeInfo &info);
SimtFusedMergeInfo GetSimtFusedMergeInfo(const af::AscNodePtr &node);
// 归并树 side GM load 的判定与寻址元数据构建（调用方在各自当前图上执行——调度期门控与
// codegen 期重建各调一次，分类/视图取自当前节点，防两份图副本间的视图漂移）。
// IsSimtDirectGmLoadNode：Load 或 direct-gm Nddma（evaluator 可直读的 GM 边界节点）。
// BuildSimtMergeSideLoadMetadata：与单 gather 输出侧 load 同构——kZeroOffset/kOutputOffset
// 分类 + use_logical_offset 坐标折叠（物理 view 承载 strided/行广播/视图匹配成员 index
// load 的 ② 形态寻址——按自身物理视图从 output_index 分解，与单 gather 输出侧 load 行为
// 一致，无 N 路 index_offset 歧义）+ 行广播标记（改写前写入的节点 attr，跨漂移稳定）。
bool IsSimtDirectGmLoadNode(const af::AscNodePtr &node);
af::Status BuildSimtMergeSideLoadMetadata(const af::AscNodePtr &load, SimtLoadMetadata &metadata);
// seq-V1 融合发射模式下非 primary 成员携带该标记（发射由 fused primary 统一处理，自身跳过）；
// L4（V2 多 evaluator）不设该标记——各成员走标准单 gather invocation。
af::Status SetSimtFusedMemberEmitted(const af::AscNodePtr &node);
bool IsSimtFusedMemberEmitted(const af::AscNodePtr &node);
// SIMD input 区域节点携带的 per-IndirectLoad 输入内轴提示：调度期优先于全局 primary 模板轴，
// 用于区分双 gather 中各路 input 边界的向量化内轴归属；单 gather 场景提示值与全局值一致。
af::Status SetSimdInputInnerAxis(const af::AscNodePtr &node, af::AxisId axis);
af::AxisId GetSimdInputInnerAxis(const af::AscNodePtr &node);
af::AscNodePtr GetInputProducer(const af::AscNodePtr &node, size_t input_index);
af::AscNodePtr GetOnlyOutputConsumer(const af::AscNodePtr &node);
af::AscNodePtr FindIndirectLoadNode(const af::AscGraph &graph);
af::Status ValidateSingleIndirectLoadNode(const af::AscGraph &graph, af::AscNodePtr &node);
bool GetPrebuiltYTilingCase(const af::AscGraph &graph, af::AxisId &tile_id,
                            std::pair<af::AxisPtr, af::AxisPtr> &tiling);
}  // namespace ascgen_utils::indirect_load

#endif  // __INDIRECT_LOAD_UTILS_H__
