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

// 队列 / 缓冲尺寸字段的规范名。
//
// `q{id}_size` / `b{id}_size` 既是 host 侧 TilingData 的字段名，也是 kernel 侧
// `#afir.buffer_desc` 的 `size_bytes_expr` 里那个符号名，还是 `afir.tiling_param`
// 取值用的名字。**三者必须逐字相同**：对不上的症状是
// `TilingData is missing canonical size expression field 'qN_size'`，或者更糟——host
// 填了一个 kernel 不读的字段，谁也不报错。
//
// 这条规则此前在 MLIR 侧有三份独立实现（bridge 建 descriptor、GenTilingData 建 stage
// schema、tiling_field_schema 建给 legacy Codegen 的字段表）。本头文件是 MLIR 侧唯一
// 的那一份。
//
// **另一半在 ATT**（`att/gen_model_info/parser/ascend_graph_parser.cpp` 与
// `common/ub_expr/asc_graph_ub_expr_builder.cpp`），名字的产地就在那边，且那边还多一条
// 分支：`que.name` / `buf.name` 非空时用用户指定的名字。MLIR 侧走不到那条分支——从
// IR 到这里只剩下一个 id，名字本身没跟着进来。今天没有任何地方给 `que.name` 赋值，
// 所以两边不会分叉；彻底合并需要让名字随 descriptor 进 IR，那是 ATT 迁移的工作。
// 在那之前，把 MLIR 侧的三份收成一份仍然是划算的。
//
// 头文件只依赖 `<string>`：它要同时被 bridge（有全量 autofuse 头）与 conversion
// （只有 MLIR 那一小段 include 路径）包含，多一个依赖就把后者的包含面撑开了。
namespace mlir::common {

inline std::string MakeQueueSizeName(int64_t id) {
  return "q" + std::to_string(id) + "_size";
}

inline std::string MakeBufferSizeName(int64_t id) {
  return "b" + std::to_string(id) + "_size";
}

// 这个名字是不是上面两个函数造出来的。
//
// 有了它，「片上资源的大小」就只剩一种写法：`#afir.buffer_desc` 上的
// `size_from_resource` 标记。把同一个名字塞进 `size_bytes_expr` 当表达式的旧写法由
// `BufferDescAttr` 的 verifier 拦掉——那种写法允许 `buffer_id = 2` 配 `"q7_size"`，
// descriptor 自相矛盾而无人发现。
//
// 这段命名空间因此是保留的：shape 变量不得叫 `q<数字>_size` / `b<数字>_size`，
// `AFIRKernelBodySignature` 撞名时会报 "canonical resource size field conflicts with
// a shape field"。
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
