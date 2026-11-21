/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef V35_ATT_API_PERF_REGISTER_NDDMA_MODEL_H_
#define V35_ATT_API_PERF_REGISTER_NDDMA_MODEL_H_

#include "api_perf_register/api_perf.h"

namespace att {
/**
 * @brief NDDMA 精确性能模型的组成和计算原理。
 *
 * 当前范围：
 * - 注册 NDDMA_1D_MULTICORE_V2（effective rank=1）和 NDDMA_ND_MULTICORE_V1（effective rank=2～5）。
 * - kUBFuse Codegen 使用 {curAivM, curAlignN} 和固定 2D stride，与 raw descriptor 不等价，因此通过
 *   NodeInfo::is_cv_ub_fusion 门禁回退 legacy 模型。
 *
 * 输入和 Codegen 映射：
 * - TensorShapeInfo::repeats       -> output_dims，单核单次搬运的各维元素个数；
 * - TensorShapeInfo::gm_strides    -> input_strides，GM 元素 stride；
 * - TensorShapeInfo::strides       -> output_strides，UB 元素 stride；
 * - AscTensorAttr::vectorized_axis -> descriptor 轴序；
 * - dtype                          -> dtype_size，并选择 B8/B16/B32/B64 参数；
 * - block_dim                      -> 参与搬运的核数。
 * 三组向量表示 codegen 合轴后的 effective view，顺序与 vectorized_axis 一致，stride 单位均为元素。
 * 默认 Codegen 使用同源的 DataCopyParams，经 SetNddmaParams 左侧补 1 后生成 5 元素 API 数组；补 1 不改变
 * effective rank 和搬运语义。
 *
 * 统一公式（effective rank D=1～5）：
 *   设归一化后的维度和 stride 按内轴到外轴排列为
 *   d_j、is_j、os_j（j=0...D-1），dtype_size 为每个元素的字节数。
 *   B = dtype_size * Π(d_j)                                      // 单次搬运字节数
 *   P_is_0 = 0，P_os_0 = 0
 *   P_is_j = Σ(d_m * is_m)，P_os_j = Σ(d_m * os_m)，m=0...j-1
 *   hat_is_j = 1                         (j=0)
 *              |is_j - P_is_j| + 1      (j>0)
 *   hat_os_j = 1                         (j=0)
 *              |os_j - P_os_j| + 1      (j>0)
 *   B_j = dtype_size * Π(d_m)，m=j...D-1                    // 第 j 层剩余数据量
 *   s_j = min(hat_is_j * dtype_size, 128)
 *   g_j = min(1, hat_os_j - 1)
 *   NG_j = (a1 + a2 * B_j) * s_j
 *   NGU_j = (b1 + b2 * s_j + (b3 + b4 * s_j) * B_j) * g_j
 *   rho_j = c1 + c2 * s_j + g_j * (c3 + c4 * s_j)
 *   N_j = NG_j + NGU_j                         (block_dim<=2)
 *         (NG_j + NGU_j) * rho_j               (block_dim>2)
 *   N_base = B/T1 + H1                         (block_dim<=2)
 *             B/T2 + H2                       (block_dim>2)
 *   cycles = N_base + Σ(N_j)，j=0...D-1
 *
 * D=1 时上述求和只有一个层级，即为 1D 模型；D=2～5 时逐层累加同一组
 * 参数，不复制或展开另一套多维系数。参数表保存 T1/H1/T2/H2、a1...c4。
 * 对二维 [M,N]/[is2,is1]/[N,1] 且 is2 < is1 的转置类场景，选择
 * NDDMA_ND_MULTICORE_NG2 特化模型。
 *
 * 静态和动态 shape：
 * - 两者使用同一组参数和同一公式；静态表达式直接折叠，动态 block_dim 生成一个 TernaryOp；
 * - 动态 is/os 保留符号 Min，输出 stride 修正项按统一公式保留符号表达式；
 * - 符号化 tiling 维度仅排除静态确认的 unit 维度；对齐后的符号化 UB stride 等价性为 unknown 时保留 NG2 候选，
 *   由后续静态非法值检查和运行时 tiling 约束保证模型安全；
 * - NG2 乘子静态非正时回退 legacy，避免发出非法性能表达式；
 * - 静态非正 dim、负 input_stride、非正 output_stride、非法 dtype/schema 均记录原因并回退 legacy。
 *
 * 处理阶段：
 * 1. BuildNddmaDescriptor：从 TensorShapeInfo 构造并合并 effective descriptor，不从 legacy 标量 stride 反推；
 * 2. NormalizeNddmaDescriptor：校验 rank、向量长度、轴序和静态值；
 * 3. EvaluateNddmaModel：选择 dtype 参数、构造静态/动态 cycles，并返回 ternary_ops；
 * 4. EvaluateNddmaModel 只返回 fallback reason；TryNewNddmaModel 在 CV-Fusion、dtype/schema、descriptor 构造和
 *    模型归一化/求值失败路径统一调用 LogNddmaFallback，记录 node、raw/effective rank、candidate model 和 reason，
 *    然后继续执行原有 GetDmaPerf。
 * 模型只输出单次调用的 AIV_MTE2 cycles；全局 pipe head 仍由 PipePerfExpr 统一添加。
 *
 * raw rank 仅用于诊断日志；模型选择和公式计算只使用 effective rank。
 */
enum class NddmaFallbackReason : int32_t {
  kNone = 0,
  kNoDescriptor,
  kRankUnsupported,
  kSchemaMismatch,
  kDtypeUnsupported,
  kStrideInvalid,
  kCodegenMismatch,
  kNoRegisteredModel,
};

// schema 和 Codegen 一致性校验后的物理描述。物理描述与统计特征归一化分离，使后续 2D～5D 模型可选择
// raw 或 normalized 特征而不改变 descriptor 语义。
struct NddmaNormalizedDesc {
  std::vector<Expr> output_dims;
  std::vector<Expr> input_strides;
  std::vector<Expr> output_strides;
  std::vector<int64_t> vectorized_axis;
  size_t effective_rank{0U};
};

// 一次模型计算要么返回 cycles 和所需动态 ternary，要么返回一个稳定的 legacy fallback 原因。
struct NddmaModelResult {
  bool selected{false};
  Expr cycles;
  std::string model_name{"NDDMA_1D_MULTICORE_V2"};
  NddmaFallbackReason fallback_reason{NddmaFallbackReason::kNone};
  size_t raw_rank{0U};
  size_t effective_rank{0U};
  TernaryOpMap ternary_ops;
};

const char *NddmaFallbackReasonToString(NddmaFallbackReason reason);

NddmaFallbackReason BuildNddmaDescriptor(const TensorShapeInfo &shape_info, const std::vector<int64_t> &vectorized_axis,
                                         NddmaDescriptorInfo &descriptor, const std::vector<bool> &tile_inner = {});

NddmaFallbackReason NormalizeNddmaDescriptor(const NddmaDescriptorInfo &descriptor, NddmaNormalizedDesc &normalized);

af::Status EvaluateNddmaModel(const NddmaDescriptorInfo &descriptor, const std::string &dtype, const Expr &block_dim,
                              NddmaModelResult &result);

void LogNddmaFallback(const std::string &node_name, const std::string &dtype, const NddmaDescriptorInfo *descriptor,
                      const NddmaModelResult &result);
}  // namespace att

#endif  // V35_ATT_API_PERF_REGISTER_NDDMA_MODEL_H_
