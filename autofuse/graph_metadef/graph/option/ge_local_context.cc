/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_local_context.h"

#include <utility>
#include "nlohmann/json.hpp"
#include "framework/common/debug/ge_log.h"
#include "ge_common_af/ge_api_types_af.h"

namespace af {
using Json = nlohmann::json;

namespace {
int32_t GetTimeoutValue(const std::string &timeout_option) {
  std::string timeout_str = "-1";
  (void)GetThreadLocalContext().GetOption(timeout_option, timeout_str);
  int timeout_value;
  try {
    timeout_value = std::stoi(timeout_str);
  } catch (...) {
    timeout_value = -1;
    GELOGW("option %s's value %s is invalid, expect an integer value", timeout_option.c_str(), timeout_str.c_str());
  }
  return timeout_value;
}
}  // namespace

GEThreadLocalContext &GetThreadLocalContext() {
  static thread_local GEThreadLocalContext thread_context;
  return thread_context;
}

graphStatus GEThreadLocalContext::GetOption(const std::string &key, std::string &option) {
  if (optimization_option_.GetValue(key, option) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  const std::map<std::string, std::string>::const_iterator graph_iter = graph_options_.find(key);
  if (graph_iter != graph_options_.end()) {
    option = graph_iter->second;
    return GRAPH_SUCCESS;
  }
  const std::map<std::string, std::string>::const_iterator session_iter = session_options_.find(key);
  if (session_iter != session_options_.end()) {
    option = session_iter->second;
    return GRAPH_SUCCESS;
  }
  const std::map<std::string, std::string>::const_iterator global_iter = global_options_.find(key);
  if (global_iter != global_options_.end()) {
    option = global_iter->second;
    return GRAPH_SUCCESS;
  }
  return GRAPH_PARAM_INVALID;
}

void GEThreadLocalContext::SetGlobalOption(std::map<std::string, std::string> options_map) {
  global_options_.clear();
  global_options_ = std::move(options_map);

  SetStreamSyncTimeout(GetTimeoutValue("stream_sync_timeout"));
  SetEventSyncTimeout(GetTimeoutValue("event_sync_timeout"));

  std::string option_name_map = "";
  if (option_name_map_.empty() &&
      (GetThreadLocalContext().GetOption(::af::OPTION_NAME_MAP, option_name_map) == GRAPH_SUCCESS)) {
    (void)SetOptionNameMap(option_name_map);
  }
}

void GEThreadLocalContext::SetSessionOption(std::map<std::string, std::string> options_map) {
  session_options_.clear();
  session_options_ = std::move(options_map);
}

void GEThreadLocalContext::SetGraphOption(std::map<std::string, std::string> options_map) {
  graph_options_.clear();
  graph_options_ = std::move(options_map);
}

graphStatus GEThreadLocalContext::SetOptionNameMap(const std::string &option_name_map_json) {
  if (!option_name_map_.empty()) {
    GELOGD("option name map has set, don't need reset");
    return ge::GRAPH_SUCCESS;
  }
  Json option_json;
  try {
    option_json = Json::parse(option_name_map_json);
  } catch (nlohmann::json::parse_error &) {
    GELOGE(ge::GRAPH_FAILED, "Parse JsonStr to Json failed, JsonStr: %s", option_name_map_json.c_str());
    return ge::GRAPH_FAILED;
  }
  for (auto iter : option_json.items()) {
    if (iter.key().empty()) {
      GELOGE(ge::GRAPH_FAILED, "Check option_name_map failed, key is null");
      return ge::GRAPH_FAILED;
    }
    if (static_cast<std::string>(iter.value()).empty()) {
      GELOGE(ge::GRAPH_FAILED, "Check option_name_map failed, value is null");
      return ge::GRAPH_FAILED;
    }
    option_name_map_.insert({iter.key(), static_cast<std::string>(iter.value())});
  }
  return ge::GRAPH_SUCCESS;
}

const std::string &GEThreadLocalContext::GetReadableName(const std::string &key) {
  auto iter = option_name_map_.find(key);
  if (iter != option_name_map_.end()) {
    GELOGD("Option %s's readable name is show name: %s", key.c_str(), iter->second.c_str());
    return iter->second;
  }
  GELOGD("Option %s's readable name is GE IR option: %s", key.c_str(), key.c_str());
  return key;
}

std::map<std::string, std::string> GEThreadLocalContext::GetAllGraphOptions() const {
  return graph_options_;
}
std::map<std::string, std::string> GEThreadLocalContext::GetAllSessionOptions() const {
  return session_options_;
}
std::map<std::string, std::string> GEThreadLocalContext::GetAllGlobalOptions() const {
  return global_options_;
}

std::map<std::string, std::string> GEThreadLocalContext::GetAllOptions() const {
  std::map<std::string, std::string> options_all;
  options_all.insert(graph_options_.cbegin(), graph_options_.cend());
  options_all.insert(session_options_.cbegin(), session_options_.cend());
  options_all.insert(global_options_.cbegin(), global_options_.cend());
  return options_all;
}

void GEThreadLocalContext::SetStreamSyncTimeout(const int32_t timeout) {
  stream_sync_timeout_ = timeout;
}

void GEThreadLocalContext::SetEventSyncTimeout(const int32_t timeout) {
  event_sync_timeout_ = timeout;
}

int32_t GEThreadLocalContext::StreamSyncTimeout() const {
  return stream_sync_timeout_;
}

int32_t GEThreadLocalContext::EventSyncTimeout() const {
  return event_sync_timeout_;
}

OptimizationOption &GEThreadLocalContext::GetOo() {
  return optimization_option_;
}
}  // namespace af
