/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_BRIDGE_ATT_SCHEMA_ADAPTER_ATT_SCHEMA_ADAPTER_H
#define AUTOFUSE_MLIR_BRIDGE_ATT_SCHEMA_ADAPTER_ATT_SCHEMA_ADAPTER_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include "bridge/AttSchemaAdapter/TilingCodeBundle.h"
#include "bridge/PlanSelector/PlanSelector.h"
#include "common/tiling_schema/tiling_field_schema.h"

namespace mlir::bridge {

// Migration-only adapter.  It calls the legacy ATT C++ generator and should
// be removed once ATT emits the MLIR-native tiling/schema contract directly.

// Generate the host tiling program through the existing ATT entry point and
// attach the candidate/schema contract consumed by the later MLIR pipeline.
// No runtime tiling result is stored here; that remains owned by
// AutofuseTiling/AutofuseLaunch.
//
// AttRequest 的引用成员只在调用点存活：一律就地构造实参，不要先存成具名变量再隔开使用。
struct AttRequest {
  const std::string &op_name;
  const ascir::FusedScheduledResult &fused_result;
  const std::vector<PlanSelection> &selections;
  const common::TilingFieldSchema &schema;
  AttOptions options;
  bool is_inductor_scene;
};

// `unsupported` 的语义与 `PlanSelection.h` 的那一位逐字相同。这里只有一条失败路径置它：
// **ATT 自己的 tiling 生成器返回 false**。判据是「ATT 拒收」与「MLIR 自己的产物和 ATT 的
// 期望不一致」之分，不是「失败发生在 ATT 里」——所以紧挨着的字段拼法不符那一支**不置**它
// （那里是 MLIR 的 `TilingFieldSchema` 发错了名字，标成能力不足就把缺陷静默回落掉了）。
af::Status RunAttForCandidates(const AttRequest &request, TilingCodeBundle &out, std::string *diagnostic = nullptr,
                               bool *unsupported = nullptr);

// Exposed only so unit tests can exercise the generated-field cross-check
// directly with hand-written source snippets, instead of requiring a real
// att::GenTilingImplAutoFuseV3 call (which needs a fully-formed schedule
// graph) to succeed end-to-end. Not part of the adapter's public API.
namespace detail {
bool QualifyApiTilingMembers(std::string &source, const common::TilingFieldSchema &schema,
                             std::string *diagnostic = nullptr);
std::vector<std::string> ExtractReferencedFieldNames(const std::string &source);
std::set<std::string> ExtractReferencedGroupMemberNames(const std::string &source);
bool ValidateGeneratedFieldReferences(const std::map<std::string, std::string> &generated,
                                      const common::TilingFieldSchema &schema, std::string *diagnostic);
}  // namespace detail

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_BRIDGE_ATT_SCHEMA_ADAPTER_ATT_SCHEMA_ADAPTER_H
