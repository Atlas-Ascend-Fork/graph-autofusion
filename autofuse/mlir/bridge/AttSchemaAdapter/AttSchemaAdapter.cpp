/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/AttSchemaAdapter/AttSchemaAdapter.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

#include "bridge/BridgeDiagnostic.h"
#include "gen_tiling_impl.h"

namespace mlir::bridge {
namespace {

// Keep this implementation isolated from the future MLIR-native ATT path.

bool ValidateSchema(const common::TilingFieldSchema &schema) {
  // Keyed on (path, name), matching tiling_field_schema.cpp's AppendField:
  // fields with the same short name (e.g. "q0_size") in different nested
  // scopes are legal and must not be rejected as duplicates here.
  std::set<std::string> seen;
  for (const auto &field : schema.fields) {
    const std::string key = field.path + common::kTilingFieldScopeKeySeparator + field.name;
    if (field.name.empty() ||
        !(std::isalpha(static_cast<unsigned char>(field.name.front())) || field.name.front() == '_') ||
        !std::all_of(field.name.begin() + 1, field.name.end(),
                     [](char value) { return std::isalnum(static_cast<unsigned char>(value)) || value == '_'; }) ||
        !seen.insert(key).second) {
      return false;
    }
  }
  return !seen.empty();
}

bool IsIdentifierChar(char value) {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

}  // namespace

namespace detail {

// ATT's generated tiling_func source accesses TilingData fields exclusively
// through the accessor macros TILING_DATA_FIELD_DEF_T generates:
// `get_<field>()`, `set_<field>(...)`, and `get_addr_<field>()`
// (att/generator/tiling_code_gen_impl.cpp uses this pattern throughout, e.g.
// "tiling_data.get_ub_size()", "tiling_data.set_tiling_key(...)"). Scan the
// generated source text for these call sites and extract the field name each
// one references, so the schema comparison below is a real cross-check of
// what ATT actually emitted, not just a self-consistency check of the schema.
std::vector<std::string> ExtractReferencedFieldNames(const std::string &source) {
  static const std::string kGet = "get_";
  static const std::string kSet = "set_";
  static const std::string kAddr = "addr_";
  std::vector<std::string> names;
  for (size_t pos = 0U; pos < source.size();) {
    size_t match_start = std::string::npos;
    size_t name_start = std::string::npos;
    const size_t get_pos = source.compare(pos, kGet.size(), kGet) == 0 ? pos : std::string::npos;
    const size_t set_pos = source.compare(pos, kSet.size(), kSet) == 0 ? pos : std::string::npos;
    if (get_pos != std::string::npos) {
      match_start = get_pos;
      name_start = get_pos + kGet.size();
      if (source.compare(name_start, kAddr.size(), kAddr) == 0) {
        name_start += kAddr.size();
      }
    } else if (set_pos != std::string::npos) {
      match_start = set_pos;
      name_start = set_pos + kSet.size();
    } else {
      ++pos;
      continue;
    }
    // Require the match to start at an identifier boundary (not itself the
    // suffix of a longer identifier like "off_set_" or "budget_").
    if (match_start > 0U && IsIdentifierChar(source[match_start - 1U])) {
      pos = match_start + 1U;
      continue;
    }
    size_t name_end = name_start;
    while (name_end < source.size() && IsIdentifierChar(source[name_end])) {
      ++name_end;
    }
    // A real accessor call is always followed by '(' (possibly after
    // whitespace); this rejects unrelated identifiers that merely start with
    // "get_"/"set_" without being one of the generated accessors.
    size_t after = name_end;
    while (after < source.size() && std::isspace(static_cast<unsigned char>(source[after])) != 0) {
      ++after;
    }
    if (name_end > name_start && after < source.size() && source[after] == '(') {
      names.push_back(source.substr(name_start, name_end - name_start));
    }
    pos = (name_end > match_start) ? name_end : match_start + 1U;
  }
  return names;
}

// Cross-checks the field names ATT's generated tiling_func source actually
// references against the schema TilingFieldSchema declares, rather than only
// validating the schema's own internal shape. A mismatch here means ATT
// derived a field (or a field spelling) that TilingFieldSchema does not know
// about, which would otherwise surface as a silent miscompile/ODR mismatch
// once both sides are wired into the same generated TilingData struct.
bool ValidateGeneratedFieldReferences(const std::map<std::string, std::string> &generated,
                                      const common::TilingFieldSchema &schema, std::string *diagnostic) {
  std::set<std::string> known_names;
  // 嵌套作用域的 `path` 就是这个 schedule group 在最外层 TilingData 里的成员名
  // （见 schedule_group_names.h），所以两半对账用的是同一张 schema。
  std::set<std::string> known_group_members;
  for (const auto &field : schema.fields) {
    known_names.insert(field.name);
    if (!field.path.empty() && field.path != "fusion") {
      known_group_members.insert(field.path);
    }
  }
  std::set<std::string> missing;
  for (const auto &[func_name, source] : generated) {
    for (const auto &name : ExtractReferencedFieldNames(source)) {
      if (known_names.find(name) == known_names.end()) {
        missing.insert(name);
      }
    }
    for (const auto &name : ExtractReferencedGroupMemberNames(source)) {
      if (known_group_members.find(name) == known_group_members.end()) {
        missing.insert(name);
      }
    }
  }
  if (!missing.empty()) {
    std::ostringstream message;
    message << "ATT generated tiling_func references field(s) not present in TilingFieldSchema:";
    for (const auto &name : missing) {
      message << " " << name;
    }
    SetDiagnostic(diagnostic, message.str());
    return false;
  }
  return true;
}

// ATT 引用的**嵌套结构体成员**名。上面那个只抓 `get_<f>()` / `set_<f>()` 这类标量访问器，
// 抓不到 `tiling_data.graph0_result0_g0_tiling_data`——多 group 时每个 schedule group 的
// 结构体是按成员名直接访问的（取引用、整体赋值），不过访问器。少了这一半对账，命名不一致
// 要到真机 host 编译期才以 `member access into incomplete type` 的形式暴露。
//
// 只认 `<标识符>_tiling_data` 这一种拼法：它就是 `ScheduleGroupIdent::GetItemPrefix()`
// 加固定后缀（`att/base/model_info.h`），也正是 schema 里这些字段的 `path`。
std::set<std::string> ExtractReferencedGroupMemberNames(const std::string &source) {
  static const std::string kSuffix = "_tiling_data";
  std::set<std::string> names;
  for (size_t pos = source.find(kSuffix); pos != std::string::npos; pos = source.find(kSuffix, pos + 1U)) {
    const size_t end = pos + kSuffix.size();
    if (end < source.size() && IsIdentifierChar(source[end])) {
      continue;
    }
    size_t begin = pos;
    while (begin > 0U && IsIdentifierChar(source[begin - 1U])) {
      --begin;
    }
    if (begin == pos) {
      continue;
    }
    // 成员访问才算，类型名与局部变量声明不算：前面必须紧跟 `.` 或 `->`。
    size_t before = begin;
    while (before > 0U && std::isspace(static_cast<unsigned char>(source[before - 1U])) != 0) {
      --before;
    }
    const bool isDot = before >= 1U && source[before - 1U] == '.';
    const bool isArrow = before >= 2U && source[before - 2U] == '-' && source[before - 1U] == '>';
    if (!isDot && !isArrow) {
      continue;
    }
    names.insert(source.substr(begin, end - begin));
  }
  return names;
}

}  // namespace detail

af::Status RunAttForCandidates(const AttRequest &request, TilingCodeBundle &out, std::string *diagnostic,
                               bool *unsupported) {
  const common::TilingFieldSchema &schema = request.schema;
  out = {};
  if (request.selections.empty()) {
    SetDiagnostic(diagnostic, "ATT adapter received no candidate selection");
    return af::FAILED;
  }
  for (const auto &selection : request.selections) {
    if (!ValidatePlanSelection(request.fused_result, selection, diagnostic)) {
      return af::FAILED;
    }
  }
  if (!ValidateSchema(schema)) {
    SetDiagnostic(diagnostic, "ATT adapter received an invalid tiling schema");
    return af::FAILED;
  }

  std::map<std::string, std::string> generated;
  // ATT 的入口按非 const 引用收 options（可能就地改写），这里照旧给它一份可改的副本——
  // 原先的按值形参也是这个语义。
  AttOptions options = request.options;
  // **ATT 自己的生成器拒收是能力边界，不是 MLIR 坏了。** 它内部还要过一遍代价模型
  // （`gen_tiling_impl.cpp` 的 `GetModelInfoMap`），而那套 perf 公式按 dtype 逐张表查，表里
  // 没有那一位就整图生成不出代价式。`SelectStaticWinner` 里那条同根因的出口已经这么标了；
  // 到得了这里的是「每个子图只有唯一候选」从而早退过代价模型的那些图（2026-10-26 探针 reg
  // 侧 2 格，末端同为 `BrcbToOneBlockPerf: Data type [bfloat16] unsatisfied`）。
  //
  // **这一位与下面的字段拼法判据刻意不同向**，理由见头文件上那段。
  if (!att::GenTilingImplAutoFuseV3(request.op_name, request.fused_result, options, generated,
                                    request.is_inductor_scene)) {
    SetDiagnostic(diagnostic, "ATT tiling generation failed");
    SetUnsupported(unsupported);
    return af::FAILED;
  }
  if (generated.empty()) {
    SetDiagnostic(diagnostic, "ATT generated an empty tiling program");
    return af::FAILED;
  }
  if (!detail::ValidateGeneratedFieldReferences(generated, schema, diagnostic)) {
    return af::FAILED;
  }

  out.tiling_functions = std::move(generated);
  out.schema_version = schema.version;
  out.schema_validated = true;
  return af::SUCCESS;
}

}  // namespace mlir::bridge
