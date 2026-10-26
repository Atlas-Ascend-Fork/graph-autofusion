/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "final_tiling_log_gen.h"
#include "common_utils.h"

#include <sstream>

namespace att::final_tiling {
namespace {

void AppendFinalTilingStructs(std::stringstream &code) {
  code << ascgen_utils::GenFinalTilingSelectionSnapshotSnippet();
  code << R"(
struct FinalTilingPipeEstimate {
  std::string name;
  double value;
  bool valid;
};

struct FinalTilingContext {
  uint32_t schema;
  std::string source;
  std::string selection_mode;
  std::string op;
  uint32_t graph;
  uint32_t result;
  uint32_t group;
  int32_t case_id;
  uint32_t tiling_key;
  int32_t score;
  std::string sub_case_tag;
  std::string template_name;
  std::string semantic_name;
  std::vector<std::string> semantic_names;
  std::vector<FinalTilingPipeEstimate> pipe_estimates;
  std::string tiling_repr;
  std::string repr_kind;
};

struct FinalTilingGroupSelection {
  uint32_t group;
  int32_t case_id;
  uint32_t tiling_key;
  int32_t score;
  std::string sub_case_tag;
  std::string template_name;
  std::string semantic_name;
};

struct TemplateSemanticInfo {
  uint32_t graph;
  uint32_t result;
  uint32_t group;
  uint32_t tiling_key;
  std::vector<std::string> semantic_names;
  std::string semantic_name;
  std::string template_name;
  std::string name_status;
};

struct TemplateSemanticRecord {
  std::string stage;
  bool has_output{false};
  size_t output_index{0UL};
  bool has_group{false};
  uint32_t group{0U};
  size_t semantic_index{0UL};
  size_t semantic_name_count{1UL};
  std::string semantic_name;
  std::string name_status;
  bool has_identity{false};
  uint32_t graph{0U};
  uint32_t result{0U};
  uint32_t tiling_key{0U};
  std::string template_name;
};

)";
}

void AppendFinalTilingSummaryStruct(std::stringstream &code, const bool null_log) {
  code << R"(
struct FinalTilingSummaryContext {
  uint32_t schema;
  std::string source;
  std::string selection_mode;
  std::string op;
  uint32_t graph;
  uint32_t result;
  std::vector<FinalTilingGroupSelection> groups;
};

inline bool ShouldEmitFinalTiling() noexcept {
)";
  code << "  return " << (null_log ? "false" : "CheckCachedLogLevel(DLOG_INFO)") << R"(;
}

struct FinalTilingObservationContext {
  bool enabled{false};
  std::vector<FinalTilingSelectionSnapshot> selections;

  void Reset(const bool observe_enabled, const size_t slot_count) noexcept {
    enabled = observe_enabled;
    selections.clear();
    try {
      if (observe_enabled) { selections.assign(slot_count, FinalTilingSelectionSnapshot{}); }
    } catch (...) {
      enabled = false;
    }
  }
};

inline thread_local FinalTilingObservationContext *g_final_tiling_observation_context = nullptr;
inline thread_local size_t g_final_tiling_observation_slot = std::numeric_limits<size_t>::max();

inline bool IsFinalTilingObservationEnabled() {
  return g_final_tiling_observation_context != nullptr && g_final_tiling_observation_context->enabled;
}

inline void DisableFinalTilingObservation() noexcept {
  if (g_final_tiling_observation_context != nullptr) { g_final_tiling_observation_context->Reset(false, 0U); }
}

inline void ResetCurrentFinalTilingSelection() {
  if (!IsFinalTilingObservationEnabled() || g_final_tiling_observation_slot >=
      g_final_tiling_observation_context->selections.size()) {
    return;
  }
  g_final_tiling_observation_context->selections[g_final_tiling_observation_slot] =
      FinalTilingSelectionSnapshot{};
}

inline void CommitCurrentFinalTilingSelection(const FinalTilingSelectionSnapshot &snapshot) {
  if (!IsFinalTilingObservationEnabled() || g_final_tiling_observation_slot >=
      g_final_tiling_observation_context->selections.size()) {
    return;
  }
  g_final_tiling_observation_context->selections[g_final_tiling_observation_slot] = snapshot;
}

inline bool GetFinalTilingSelectionSnapshot(const size_t slot, FinalTilingSelectionSnapshot &snapshot) {
  if (!IsFinalTilingObservationEnabled() || slot >= g_final_tiling_observation_context->selections.size()) {
    return false;
  }
  snapshot = g_final_tiling_observation_context->selections[slot];
  return snapshot.valid;
}

inline std::vector<FinalTilingSelectionSnapshot> GetFinalTilingSelectionSnapshots() noexcept {
  if (!IsFinalTilingObservationEnabled()) { return {}; }
  try {
    return g_final_tiling_observation_context->selections;
  } catch (...) {
    DisableFinalTilingObservation();
    return {};
  }
}

inline std::vector<FinalTilingSelectionSnapshot> CaptureFinalTilingSelectionSnapshot(
    const size_t slot, const FinalTilingSelectionSnapshot &snapshot) noexcept {
  if (!IsFinalTilingObservationEnabled() || slot >= g_final_tiling_observation_context->selections.size()) { return {}; }
  try {
    std::vector<FinalTilingSelectionSnapshot> snapshots(g_final_tiling_observation_context->selections.size());
    snapshots[slot] = snapshot;
    return snapshots;
  } catch (...) {
    DisableFinalTilingObservation();
    return {};
  }
}

inline void RestoreFinalTilingSelectionSnapshots(const std::vector<FinalTilingSelectionSnapshot> &snapshots) {
  if (!IsFinalTilingObservationEnabled()) { return; }
  auto &selections = g_final_tiling_observation_context->selections;
  std::fill(selections.begin(), selections.end(), FinalTilingSelectionSnapshot{});
  std::copy_n(snapshots.begin(), std::min(snapshots.size(), selections.size()), selections.begin());
}

class FinalTilingObservationCallGuard {
 public:
  FinalTilingObservationCallGuard(FinalTilingObservationContext &context, const bool enabled,
                                  const size_t slot_count) noexcept
      : previous_context_(g_final_tiling_observation_context), previous_slot_(g_final_tiling_observation_slot) {
    context.Reset(enabled, slot_count);
    g_final_tiling_observation_context = &context;
    g_final_tiling_observation_slot = std::numeric_limits<size_t>::max();
  }

  ~FinalTilingObservationCallGuard() {
    g_final_tiling_observation_context = previous_context_;
    g_final_tiling_observation_slot = previous_slot_;
  }

 private:
  FinalTilingObservationContext *previous_context_;
  size_t previous_slot_;
};

class FinalTilingObservationSlotGuard {
 public:
  explicit FinalTilingObservationSlotGuard(const size_t slot)
      : previous_slot_(g_final_tiling_observation_slot),
        owns_slot_(g_final_tiling_observation_context != nullptr &&
                   g_final_tiling_observation_slot == std::numeric_limits<size_t>::max()) {
    if (owns_slot_) { g_final_tiling_observation_slot = slot; }
  }

  ~FinalTilingObservationSlotGuard() {
    if (owns_slot_) { g_final_tiling_observation_slot = previous_slot_; }
  }

 private:
  size_t previous_slot_;
  bool owns_slot_;
};
)";
}

void AppendFinalTilingQuoteHelpers(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingShellQuote(const std::string &value) {
  std::string quoted;
  quoted.reserve(value.size() + 2U);
  quoted.push_back('\"');
  for (const unsigned char ch : value) {
    if (ch == '\\' || ch == '\"') {
      quoted.push_back('\\');
      quoted.push_back(static_cast<char>(ch));
    } else if (ch == '\n') {
      quoted.append("\\n");
    } else if (ch == '\r') {
      quoted.append("\\r");
    } else if (ch <= 0x1FU || ch == 0x7FU) {
      char buffer[5] = {0};
      (void)snprintf(buffer, sizeof(buffer), "%02x", static_cast<unsigned int>(ch));
      quoted.append("\\x");
      quoted.append(buffer);
    } else {
      quoted.push_back(static_cast<char>(ch));
    }
  }
  quoted.push_back('\"');
  return quoted;
}

inline std::string FinalTilingJsonQuote(const std::string &value) {
  std::string quoted;
  quoted.reserve(value.size() + 2U);
  quoted.push_back('\"');
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': quoted.append("\\\\"); break;
      case '\"': quoted.append("\\\""); break;
      case '\n': quoted.append("\\n"); break;
      case '\r': quoted.append("\\r"); break;
      case '\t': quoted.append("\\t"); break;
      default:
        if (ch <= 0x1FU || ch == 0x7FU) {
          char buffer[7] = {0};
          (void)snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned int>(ch));
          quoted.append(buffer);
        } else {
          quoted.push_back(static_cast<char>(ch));
        }
        break;
    }
  }
  quoted.push_back('\"');
  return quoted;
}
)";
}

void AppendFinalTilingChecksumHelper(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingChecksum(const std::string &input) {
  uint64_t hash = 0x6d2b79f5ULL;
  for (const unsigned char value : input) {
    hash ^= static_cast<uint64_t>(value) + 0x9e3779b9ULL + (hash << 6U) + (hash >> 2U);
    hash = (hash << 13U) | (hash >> 51U);
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}
)";
}

void AppendFinalTilingIdentityHelpers(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingStableIdPart(const std::string &value) {
  return std::to_string(value.size()) + ":" + value;
}

inline std::string FinalTilingStableId(const FinalTilingContext &context) {
  return FinalTilingStableIdPart(context.op) + "|" + std::to_string(context.graph) + "|" +
         std::to_string(context.result) + "|" + std::to_string(context.group) + "|" +
         std::to_string(context.case_id) + "|" + std::to_string(context.tiling_key) + "|" +
         FinalTilingStableIdPart(context.sub_case_tag);
}

inline std::string FinalTilingNumber(const double value) {
  return std::isfinite(value) ? std::to_string(value) : std::string("null");
}

inline std::string FinalTilingPipeJson(const FinalTilingContext &context) {
  std::string pipe = "{";
  for (size_t index = 0U; index < context.pipe_estimates.size(); ++index) {
    const auto &estimate = context.pipe_estimates[index];
    if (index != 0U) { pipe.push_back(','); }
    pipe += FinalTilingJsonQuote(estimate.name);
    pipe.push_back(':');
    pipe += (estimate.valid && std::isfinite(estimate.value)) ? std::to_string(estimate.value) : "null";
  }
  pipe.push_back('}');
  return pipe;
}

inline std::string FinalTilingCommon(const FinalTilingContext &context) {
  return std::string("schema=") + std::to_string(context.schema) + " source=" + FinalTilingShellQuote(context.source) +
      " selection_mode=" + FinalTilingShellQuote(context.selection_mode) + " operator=" +
      FinalTilingShellQuote(context.op) + " graph=" + std::to_string(context.graph) +
      " result=" + std::to_string(context.result) + " group=" + std::to_string(context.group) +
      " case_id=" + std::to_string(context.case_id) + " tiling_key=" + std::to_string(context.tiling_key) +
      " score=" + std::to_string(context.score) + " sub_case_tag=" + FinalTilingShellQuote(context.sub_case_tag) +
      " template=" + FinalTilingShellQuote(context.template_name) + " repr_kind=" +
      FinalTilingShellQuote(context.repr_kind) + " pipe_estimates=" +
      FinalTilingShellQuote(FinalTilingPipeJson(context));
}

inline std::string FinalTilingChunkPrefix(const std::string &id, const size_t seq) {
  return std::string("[ATT][FINAL_TILING_CHUNK] id=") + FinalTilingShellQuote(id) +
         " seq=" + std::to_string(seq) + " data=";
}
)";
}

void AppendFinalTilingSplitPayload(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingSummaryChunkPrefix(const std::string &id, const size_t seq) {
  return std::string("[ATT][FINAL_TILING_SUMMARY_CHUNK] id=") + FinalTilingShellQuote(id) +
         " seq=" + std::to_string(seq) + " data=";
}

inline std::vector<std::string> FinalTilingSplitPayload(const std::string &payload, const std::string &chunk_prefix) {
  constexpr size_t kLineBudget = 700U;
  std::vector<std::string> chunks;
  size_t begin = 0U;
  while (begin < payload.size()) {
    size_t end = begin;
    while (end < payload.size()) {
      size_t next = end + 1U;
      while (next < payload.size() && (static_cast<unsigned char>(payload[next]) & 0xc0U) == 0x80U) { ++next; }
      const std::string candidate = payload.substr(begin, next - begin);
      const std::string probe = chunk_prefix + FinalTilingShellQuote(candidate);
      if (probe.size() > kLineBudget && end != begin) { break; }
      end = next;
      if (probe.size() > kLineBudget) { break; }
    }
    if (end == begin) {
      end = begin + 1U;
      while (end < payload.size() && (static_cast<unsigned char>(payload[end]) & 0xc0U) == 0x80U) { ++end; }
    }
    chunks.emplace_back(payload.substr(begin, end - begin));
    begin = end;
  }
  return chunks;
}
)";
}

void AppendFinalTilingRecordFormatter(std::stringstream &code) {
  code << R"(
inline std::vector<std::string> FormatFinalTilingRecord(const FinalTilingContext &context) {
  const std::string common = FinalTilingCommon(context);
  const std::string single = std::string("[ATT][FINAL_TILING] ") + common + " tiling_repr=" +
      FinalTilingShellQuote(context.tiling_repr);
  if (single.size() <= 700U) { return {single}; }
  const std::string id = FinalTilingStableId(context);
  const std::string hash = FinalTilingChecksum(context.tiling_repr);
  const auto chunks = FinalTilingSplitPayload(context.tiling_repr, FinalTilingChunkPrefix(id, std::numeric_limits<size_t>::max()));
  std::vector<std::string> lines;
  lines.emplace_back(std::string("[ATT][FINAL_TILING_BEGIN] ") + common + " id=" + FinalTilingShellQuote(id) +
                     " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(context.tiling_repr.size()) +
                     " hash_alg=att_mix64_v1 hash=" + hash);
  for (size_t seq = 0U; seq < chunks.size(); ++seq) {
    lines.emplace_back(FinalTilingChunkPrefix(id, seq) + FinalTilingShellQuote(chunks[seq]));
  }
  lines.emplace_back(std::string("[ATT][FINAL_TILING_END] id=") + FinalTilingShellQuote(id) +
                     " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(context.tiling_repr.size()) +
                     " hash_alg=att_mix64_v1 hash=" + hash);
  return lines;
}
)";
}

void AppendFinalTilingSummaryFormatter(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingGroupsJson(const std::vector<FinalTilingGroupSelection> &groups) {
  std::string value = "{";
  for (size_t index = 0U; index < groups.size(); ++index) {
    const auto &group = groups[index];
    if (index != 0U) { value.push_back(','); }
    value += FinalTilingJsonQuote(std::to_string(group.group)) + ":{\"case_id\":" +
        std::to_string(group.case_id) + ",\"tiling_key\":" + std::to_string(group.tiling_key) +
        ",\"score\":" + std::to_string(group.score) + ",\"sub_case_tag\":" +
        FinalTilingJsonQuote(group.sub_case_tag) + ",\"template\":" + FinalTilingJsonQuote(group.template_name) +
        ",\"semantic_name\":" +
        FinalTilingJsonQuote(group.semantic_name) + "}";
  }
  value.push_back('}');
  return value;
}

inline std::vector<std::string> FormatFinalTilingSummary(const FinalTilingSummaryContext &context) {
  const std::string groups = FinalTilingGroupsJson(context.groups);
  const std::string line = std::string("[ATT][FINAL_TILING_SUMMARY] schema=") + std::to_string(context.schema) +
      " source=" + FinalTilingShellQuote(context.source) + " selection_mode=" +
      FinalTilingShellQuote(context.selection_mode) + " operator=" + FinalTilingShellQuote(context.op) +
      " graph=" + std::to_string(context.graph) + " result=" + std::to_string(context.result) +
      " groups=" + FinalTilingShellQuote(groups);
  if (line.size() <= 700U) { return {line}; }
  const std::string id = FinalTilingStableIdPart(context.op) + "|" + std::to_string(context.graph) + "|" + std::to_string(context.result);
  const std::string hash = FinalTilingChecksum(groups);
  const auto chunks = FinalTilingSplitPayload(groups, FinalTilingSummaryChunkPrefix(id, std::numeric_limits<size_t>::max()));
  std::vector<std::string> lines;
  lines.emplace_back(std::string("[ATT][FINAL_TILING_SUMMARY_BEGIN] schema=") + std::to_string(context.schema) +
      " source=" + FinalTilingShellQuote(context.source) + " selection_mode=" +
      FinalTilingShellQuote(context.selection_mode) + " operator=" + FinalTilingShellQuote(context.op) +
      " graph=" + std::to_string(context.graph) + " result=" + std::to_string(context.result) +
      " id=" + FinalTilingShellQuote(id) +
      " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(groups.size()) +
      " hash_alg=att_mix64_v1 hash=" + hash);
  for (size_t seq = 0U; seq < chunks.size(); ++seq) {
    lines.emplace_back(FinalTilingSummaryChunkPrefix(id, seq) + FinalTilingShellQuote(chunks[seq]));
  }
  lines.emplace_back(std::string("[ATT][FINAL_TILING_SUMMARY_END] id=") + FinalTilingShellQuote(id) +
      " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(groups.size()) +
      " hash_alg=att_mix64_v1 hash=" + hash);
  return lines;
}
)";
}

void AppendTemplateSemanticFormatter(std::stringstream &code) {
  code << R"(
struct TemplateSemanticFields {
  std::string family;
  std::string strategy;
  std::string role;
  std::vector<std::string> api_features;
  std::vector<std::string> transforms;
};

inline std::vector<std::string> TemplateSemanticSplit(const std::string &value, const char separator) {
  std::vector<std::string> tokens;
  size_t begin = 0UL;
  while (begin <= value.size()) {
    const size_t end = value.find(separator, begin);
    tokens.emplace_back(value.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1UL;
  }
  return tokens;
}

inline TemplateSemanticFields ParseTemplateSemanticCore(const std::string &core) {
  TemplateSemanticFields fields;
  const size_t role_pos = core.find('@');
  const size_t api_pos = core.find('#');
  const std::string base = core.substr(0UL, std::min(role_pos, api_pos));
  const size_t strategy_pos = base.find('/');
  fields.family = base.substr(0UL, strategy_pos);
  if (fields.family.empty()) { fields.family = "Unknown"; }
  if (strategy_pos != std::string::npos) {
    fields.strategy = base.substr(strategy_pos + 1UL);
  }
  if (role_pos != std::string::npos) {
    fields.role = core.substr(role_pos + 1UL, api_pos == std::string::npos
                                                ? std::string::npos : api_pos - role_pos - 1UL);
  }
  if (api_pos != std::string::npos) {
    fields.api_features = TemplateSemanticSplit(core.substr(api_pos + 1UL), '|');
  }
  return fields;
}

inline TemplateSemanticFields ParseTemplateSemanticName(const std::string &semantic_name) {
  TemplateSemanticFields fields;
  if (semantic_name.empty() || semantic_name == "Unknown") {
    fields.family = semantic_name.empty() ? "Unknown" : semantic_name;
    return fields;
  }
  const size_t transform_pos = semantic_name.find('+');
  const std::string core = transform_pos == std::string::npos
                                ? semantic_name
                                : semantic_name.substr(0UL, transform_pos);
  fields = ParseTemplateSemanticCore(core);
  if (transform_pos != std::string::npos) {
    fields.transforms = TemplateSemanticSplit(semantic_name.substr(transform_pos + 1UL), '+');
  }
  return fields;
}

inline std::string TemplateSemanticPrefix(const TemplateSemanticRecord &record) {
  std::string value = "[ATT][TEMPLATE] stage=" + record.stage;
  if (record.has_output) { value += " output[" + std::to_string(record.output_index) + "]"; }
  if (record.has_group) { value += " group=" + std::to_string(record.group); }
  value += " semantic_index=" + std::to_string(record.semantic_index);
  value += " semantic_name_count=" + std::to_string(record.semantic_name_count);
  return value;
}

inline std::string TemplateSemanticCoreLine(const TemplateSemanticRecord &record) {
  const std::string semantic_name = record.semantic_name.empty() ? "Unknown" : record.semantic_name;
  std::string line = TemplateSemanticPrefix(record) + " semantic_name=" + FinalTilingShellQuote(semantic_name);
  if (!record.name_status.empty()) { line += " name_status=" + record.name_status; }
  return line;
}

inline std::string TemplateSemanticFieldSummary(const std::string &field) {
  const size_t separator = field.find('=');
  const std::string key = separator == std::string::npos ? "field" : field.substr(0UL, separator);
  return key + "_hash=" + FinalTilingChecksum(field) + " " + key + "_len=" + std::to_string(field.size());
}

inline std::string TemplateSemanticLineSummary(const std::string &line) {
  return "line_hash=" + FinalTilingChecksum(line) + " line_len=" + std::to_string(line.size());
}

inline void BoundTemplateSemanticLines(std::vector<std::string> &lines, const std::string &prefix) {
  constexpr size_t kTemplateSemanticLineBudget = 700U;
  for (auto &line : lines) {
    if (line.size() > kTemplateSemanticLineBudget) {
      line = prefix + " " + TemplateSemanticLineSummary(line);
    }
  }
}

inline std::vector<std::string> TemplateSemanticDetailLines(const std::string &prefix,
                                                            const std::vector<std::string> &fields) {
  constexpr size_t kTemplateSemanticLineBudget = 700U;
  const std::string &prefix_line = prefix;
  std::string current = prefix_line;
  std::vector<std::string> lines;
  for (const auto &field : fields) {
    const std::string item = " " + field;
    if (prefix_line.size() + item.size() > kTemplateSemanticLineBudget) {
      if (current != prefix_line) {
        lines.emplace_back(current);
        current = prefix_line;
      }
      lines.emplace_back(prefix_line + " " + TemplateSemanticFieldSummary(field));
      continue;
    }
    if (current != prefix_line && current.size() + item.size() > kTemplateSemanticLineBudget) {
      lines.emplace_back(current);
      current = prefix_line;
    }
    current += item;
  }
  if (current != prefix_line) { lines.emplace_back(current); }
  return lines;
}

inline std::vector<std::string> TemplateSemanticFieldLines(const std::string &prefix,
                                                           const TemplateSemanticFields &fields) {
  std::string api_features;
  for (size_t index = 0UL; index < fields.api_features.size(); ++index) {
    if (index != 0UL) { api_features.push_back('|'); }
    api_features += fields.api_features[index];
  }
  std::string transforms;
  for (size_t index = 0UL; index < fields.transforms.size(); ++index) {
    if (index != 0UL) { transforms.push_back('+'); }
    transforms += fields.transforms[index];
  }
  const std::vector<std::string> values = {
      "semantic_schema=1",
      "family=" + FinalTilingShellQuote(fields.family),
      "strategy=" + FinalTilingShellQuote(fields.strategy),
      "role=" + FinalTilingShellQuote(fields.role),
      "api_features=" + FinalTilingShellQuote(api_features),
      "transforms=" + FinalTilingShellQuote(transforms)};
  return TemplateSemanticDetailLines(prefix + " fields", values);
}

inline std::vector<std::string> FormatTemplateSemanticRecord(const TemplateSemanticRecord &record) {
  constexpr size_t kTemplateSemanticLineBudget = 700U;
  const TemplateSemanticFields semantic_fields = ParseTemplateSemanticName(record.semantic_name);
  const std::string core = TemplateSemanticCoreLine(record);
  const std::string prefix = TemplateSemanticPrefix(record);
  std::vector<std::string> lines{core};
  const auto field_lines = TemplateSemanticFieldLines(prefix, semantic_fields);
  lines.insert(lines.end(), field_lines.begin(), field_lines.end());
  if (!record.has_identity) {
    BoundTemplateSemanticLines(lines, prefix);
    return lines;
  }
  const std::string detail = prefix + " detail graph=" + std::to_string(record.graph) + " result=" +
      std::to_string(record.result) + " tiling_key=" + std::to_string(record.tiling_key) + " template=" +
      FinalTilingShellQuote(record.template_name);
  if (detail.size() <= kTemplateSemanticLineBudget) {
    lines.emplace_back(detail);
    BoundTemplateSemanticLines(lines, prefix);
    return lines;
  }
  const std::vector<std::string> fields = {
      "graph=" + std::to_string(record.graph),
      "result=" + std::to_string(record.result),
      "tiling_key=" + std::to_string(record.tiling_key),
      "template=" + FinalTilingShellQuote(record.template_name)};
  const auto detail_lines = TemplateSemanticDetailLines(prefix + " detail", fields);
  lines.insert(lines.end(), detail_lines.begin(), detail_lines.end());
  BoundTemplateSemanticLines(lines, prefix);
  return lines;
}
)";
}

void AppendFinalTilingEmitters(std::stringstream &code) {
  code << R"(
inline void EmitFinalTilingLines(const std::vector<std::string> &lines) noexcept {
  try {
    for (const auto &line : lines) { ATT_FINAL_TILING_LOGI(OP_NAME, "%s", line.c_str()); }
  } catch (...) {
    // Observability must not change the tiling result.
  }
}

inline void EmitFinalTilingSummary(const FinalTilingSummaryContext &context) noexcept {
  try {
    EmitFinalTilingLines(FormatFinalTilingSummary(context));
  } catch (...) {
    // Observability must not change the tiling result.
  }
}

inline void EmitTemplateSemanticRecord(const TemplateSemanticRecord &record) noexcept {
  try {
    EmitFinalTilingLines(FormatTemplateSemanticRecord(record));
  } catch (...) {
    // Observability must not change the tiling result.
  }
}

inline void AppendTemplateSemanticName(std::vector<std::string> &names, const std::string &name) {
  if (!name.empty() && std::find(names.begin(), names.end(), name) == names.end()) {
    names.push_back(name);
  }
}

inline std::vector<std::string> GetFinalTilingSemanticNames(const FinalTilingContext &context) {
  std::vector<std::string> names;
  for (const auto &name : context.semantic_names) {
    AppendTemplateSemanticName(names, name);
  }
  AppendTemplateSemanticName(names, context.semantic_name);
  if (names.empty()) {
    names.emplace_back("");
  }
  return names;
}

inline void EmitTemplateSemanticRecord(const FinalTilingContext &context, const bool ambiguous = false) noexcept {
  try {
    const auto semantic_names = GetFinalTilingSemanticNames(context);
    const bool unavailable = semantic_names.size() == 1UL && semantic_names[0].empty();
    for (size_t semantic_index = 0UL; semantic_index < semantic_names.size(); ++semantic_index) {
      TemplateSemanticRecord record;
      record.stage = "att_return";
      record.has_group = true;
      record.group = context.group;
      record.semantic_index = semantic_index;
      record.semantic_name_count = semantic_names.size();
      record.semantic_name = semantic_names[semantic_index];
      record.name_status = unavailable ? "unavailable" : (ambiguous ? "ambiguous" : "");
      record.has_identity = true;
      record.graph = context.graph;
      record.result = context.result;
      record.tiling_key = context.tiling_key;
      record.template_name = context.template_name;
      EmitTemplateSemanticRecord(record);
    }
  } catch (...) {
    // Observability must not change the tiling result.
  }
}

inline void EmitFinalTilingRecord(const FinalTilingContext &context) noexcept {
  try {
    EmitFinalTilingLines(FormatFinalTilingRecord(context));
  } catch (...) {
    // Formatting and allocation failures are also observational only.
  }
}
)";
}

}  // namespace

std::string GenFinalTilingLogHelpers(const bool null_log) {
  std::stringstream code;
  code << "#include <algorithm>" << std::endl;
  code << "#include <cmath>" << std::endl;
  code << "#include <cstddef>" << std::endl;
  code << "#include <cstdint>" << std::endl;
  code << "#include <cstdio>" << std::endl;
  code << "#include <iomanip>" << std::endl;
  code << "#include <limits>" << std::endl;
  code << "#include <sstream>" << std::endl;
  code << "#include <string>" << std::endl;
  code << "#include <vector>" << std::endl;
  AppendFinalTilingStructs(code);
  AppendFinalTilingSummaryStruct(code, null_log);
  AppendFinalTilingQuoteHelpers(code);
  AppendFinalTilingChecksumHelper(code);
  AppendFinalTilingIdentityHelpers(code);
  AppendFinalTilingSplitPayload(code);
  AppendFinalTilingRecordFormatter(code);
  AppendFinalTilingSummaryFormatter(code);
  AppendTemplateSemanticFormatter(code);
  AppendFinalTilingEmitters(code);
  return code.str();
}

}  // namespace att::final_tiling
