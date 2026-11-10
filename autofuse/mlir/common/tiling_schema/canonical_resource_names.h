/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_CANONICAL_RESOURCE_NAMES_H
#define AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_CANONICAL_RESOURCE_NAMES_H

#include <string>
#include <string_view>

// 队列和缓冲尺寸字段的规范名。
//
// q{id}_size / b{id}_size 同时用于 host TilingData 字段、kernel buffer_desc 的
// size_bytes_expr 符号以及 tiling_param 查询，三处必须一致。名称不一致可能导致
// "TilingData is missing canonical size expression field 'qN_size'"，也可能使 host
// 填写的字段未被 kernel 读取而没有诊断。
//
// bridge descriptor、GenTilingData stage schema 和供 legacy Codegen 使用的
// tiling_field_schema 共用此处命名，避免 MLIR 侧重复实现。
// ATT 侧对应逻辑位于 att/gen_model_info/parser/ascend_graph_parser.cpp 和
// common/ub_expr/asc_graph_ub_expr_builder.cpp；que.name / buf.name 非空时还支持自定义名。
// MLIR descriptor 只传递 id，不携带自定义名。目前没有 que.name 赋值路径；若要统一
// 自定义命名，需要在 ATT 迁移时同时传递名称，不能只修改此处规则。
//
// 只依赖标准库头，供 bridge 和 include 路径受限的 conversion 共用。
namespace mlir::common {

inline std::string MakeQueueSizeName(int64_t id) {
  return "q" + std::to_string(id) + "_size";
}

inline std::string MakeBufferSizeName(int64_t id) {
  return "b" + std::to_string(id) + "_size";
}

// 判断名称是否符合 q<数字>_size 或 b<数字>_size。
// 片上资源大小使用 buffer_desc 的 size_from_resource 标记，不再把规范名写进
// size_bytes_expr。BufferDescAttr verifier 会拒绝旧写法，以免 buffer_id = 2
// 却引用 q7_size 这类相互矛盾的 descriptor 通过校验。
// 这些名称保留给资源字段；shape 变量重名时 AFIRKernelBodySignature 会报告
// "canonical resource size field conflicts with a shape field"。
inline bool IsCanonicalResourceSizeName(std::string_view name) {
  constexpr std::string_view kSuffix = "_size";
  if (name.size() <= 1 + kSuffix.size() || (name.front() != 'q' && name.front() != 'b')) {
    return false;
  }
  if (name.substr(name.size() - kSuffix.size()) != kSuffix) {
    return false;
  }
  const std::string_view digits = name.substr(1, name.size() - 1 - kSuffix.size());
  for (const char ch : digits) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }
  return true;
}

}  // namespace mlir::common

#endif  // AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_CANONICAL_RESOURCE_NAMES_H
