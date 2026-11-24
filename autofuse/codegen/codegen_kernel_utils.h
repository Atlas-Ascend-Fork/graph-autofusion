/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_KERNEL_UTILS_H__
#define __CODEGEN_KERNEL_UTILS_H__

// 从超大源文件 codegen_kernel.cpp / codegen_kernel_loop.cpp 拆出的工具函数声明
// (与 LoopAxisDistance 同为全局作用域), 供两文件共用

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "codegen_kernel.h"
#include "codegen_kernel_loop.h"

// ===== CommonQueueSplit 候选规划(原 codegen_kernel_loop.cpp, 消费方 TPipe::PrepareCommonQueueSplitPlan) =====
bool IsUnconditionalCall(const codegen::ApiCall &call);
bool IsSingleQueueTensor(const codegen::TQue &queue, const codegen::Tensor &tensor);
// 定义在 codegen_kernel_loop.cpp(圈复杂度超标项, 原样未拆; 唯一调用方在本文件的GetCommonQueueSplitCandidates)
const codegen::ApiCall *GetCommonQueueSplitAnchor(const codegen::ApiCall &load, const codegen::TPipe &tpipe);
bool HasIndependentQueueLifetime(const codegen::Loop &loop, size_t begin, size_t end, const codegen::ApiCall &load,
                                 const codegen::TPipe &tpipe);
std::map<const codegen::ApiCall *, const codegen::ApiCall *> GetCommonQueueSplitCandidates(const codegen::Loop &loop,
                                                                                           const codegen::TPipe &tpipe);
void CollectCommonQueueSplitCandidates(const codegen::Loop &loop, const codegen::TPipe &tpipe,
                                       std::vector<const codegen::ApiCall *> &loads);
bool BuildCommonQueueUbGuard(const codegen::TPipe &tpipe, const af::AscGraph &graph,
                             codegen::CommonQueueSplitPlan &plan);
std::unique_ptr<codegen::TPipe> MakeCommonMmQueuePipe(const codegen::TPipe &tpipe);

// ===== TQue 规模表达式与定义发射(原 codegen_kernel.cpp, 消费方 TPipe::LocalTQueAlloc) =====
// 注: DefineConst 按右值引用消费入参, 故规模文本按值传入并 move 转交
void GenLocalTQueDefine(std::stringstream &ss, const codegen::TQue &que, std::string tensor_size_max,
                        std::string tensor_bufnum_max, bool is_ub_fuse, bool using_att_calc_qbt_size);
Status AppendMergeScopeMaxTerms(const codegen::TPipe &tpipe, const codegen::TQue &que,
                                std::stringstream &tensor_size_max, std::stringstream &tensor_bufnum_max,
                                bool &is_first);
Status AppendNotMergeTensorMaxTerms(const codegen::TPipe &tpipe, const codegen::TQue &que,
                                    std::stringstream &tensor_size_max, std::stringstream &tensor_bufnum_max,
                                    bool &is_first);
Status AppendShareGroupMaxTerms(const codegen::TPipe &tpipe, const codegen::TQue &que,
                                std::stringstream &tensor_size_max);

// ===== ApiCall 输出分配子路径(原 codegen_kernel_loop.cpp ApiCall 私有成员, 消费方 AllocOutputs/AllocQueueOutput) =====
Status AllocBufferOutput(const codegen::TPipe &tpipe, const codegen::ApiTensor &out, const codegen::Tensor &t,
                         std::stringstream &ss);
Status AllocVecCalcOutput(const codegen::TPipe &tpipe, const codegen::ApiTensor &out, const codegen::Tensor &t,
                          std::stringstream &ss);
#endif
