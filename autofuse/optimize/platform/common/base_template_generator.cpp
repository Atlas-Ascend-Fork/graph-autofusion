/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_template_generator.h"

#include <map>
#include <string>

#include "optimize/graph_pass/pass_utils.h"

namespace optimize {
namespace {

constexpr char kNddmaSuffix[] = "_nddma";
constexpr char kLoadToNddmaSuffix[] = "_load_to_nddma";

bool HasSuffix(const std::string &value, const std::string &suffix) {
  return value.size() > suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool IsSameNddmaTemplate(const af::AscGraph &lhs, const af::AscGraph &rhs) {
  std::vector<af::AscNodePtr> lhs_nodes;
  for (const auto &node : lhs.GetAllNodes()) {
    lhs_nodes.push_back(node);
  }
  std::vector<af::AscNodePtr> rhs_nodes;
  for (const auto &node : rhs.GetAllNodes()) {
    rhs_nodes.push_back(node);
  }
  if (lhs_nodes.size() != rhs_nodes.size()) {
    return false;
  }
  for (size_t i = 0UL; i < lhs_nodes.size(); ++i) {
    if (lhs_nodes[i] == nullptr || rhs_nodes[i] == nullptr) {
      return false;
    }
    if (lhs_nodes[i]->GetType() != rhs_nodes[i]->GetType()) {
      return false;
    }
    const auto lhs_outputs = lhs_nodes[i]->outputs();
    const auto rhs_outputs = rhs_nodes[i]->outputs();
    if (lhs_outputs.size() != rhs_outputs.size()) {
      return false;
    }
    if (!lhs_outputs.empty() &&
        !PassUtils::IsExprVectorEqual(lhs_outputs[0]->attr.repeats, rhs_outputs[0]->attr.repeats)) {
      return false;
    }
  }
  GELOGD("Nddma templates are the same, lhs [%s], rhs [%s], node count %zu.", lhs.GetName().c_str(),
         rhs.GetName().c_str(), lhs_nodes.size());
  return true;
}

void DedupNddmaTemplates(const std::vector<autoschedule::AutoScheduleOutput> &generated_cases,
                         std::unordered_set<std::string> &drop_case_names) {
  GELOGD("Start to dedup nddma templates, generated cases: %zu.", generated_cases.size());
  std::map<std::string, const af::AscGraph *> nddma_templates;
  std::map<std::string, const af::AscGraph *> load_to_nddma_templates;
  for (const auto &generated_case : generated_cases) {
    const std::string &name = generated_case.scheduled_graph.GetName();
    if (HasSuffix(name, kLoadToNddmaSuffix)) {
      const std::string key = name.substr(0UL, name.size() - (sizeof(kLoadToNddmaSuffix) - 1UL));
      load_to_nddma_templates.emplace(key, &generated_case.scheduled_graph);
    } else if (HasSuffix(name, kNddmaSuffix)) {
      const std::string key = name.substr(0UL, name.size() - (sizeof(kNddmaSuffix) - 1UL));
      nddma_templates.emplace(key, &generated_case.scheduled_graph);
    }
  }
  GELOGD("Collected nddma templates: %zu, load_to_nddma templates: %zu.", nddma_templates.size(),
         load_to_nddma_templates.size());
  for (const auto &entry : load_to_nddma_templates) {
    const auto nddma_it = nddma_templates.find(entry.first);
    if (nddma_it == nddma_templates.end()) {
      continue;
    }
    if (!IsSameNddmaTemplate(*entry.second, *nddma_it->second)) {
      continue;
    }
    GELOGD("Drop duplicated nddma template [%s], same as load_to_nddma template [%s], key [%s].",
           nddma_it->second->GetName().c_str(), entry.second->GetName().c_str(), entry.first.c_str());
    drop_case_names.emplace(nddma_it->second->GetName());
  }
}

}  // namespace

std::vector<autoschedule::AutoScheduleOutput> BaseTemplateGenerator::GetBasedCasesByGenMode(
    const GenerationMode mode, const std::vector<autoschedule::AutoScheduleOutput> &tiling_cases,
    const std::vector<autoschedule::AutoScheduleOutput> &generated_cases) {
  if (mode == GenerationMode::kBaseCase) {
    return tiling_cases;
  }
  if (mode == GenerationMode::kAppendCase) {
    std::vector<autoschedule::AutoScheduleOutput> merged;
    merged.reserve(tiling_cases.size() + generated_cases.size());  // 预分配内存，避免多次扩容
    merged.insert(merged.end(), tiling_cases.begin(), tiling_cases.end());
    merged.insert(merged.end(), generated_cases.begin(), generated_cases.end());
    return merged;
  }
  GELOGW("Unknown generation mode: %u", mode);
  return {};
}

af::Status BaseTemplateGenerator::Generate(BaseTemplate &strategy, const af::AscGraph &origin_graph,
                                           const std::vector<autoschedule::AutoScheduleOutput> &based_cases,
                                           std::vector<autoschedule::AutoScheduleOutput> &generated_cases,
                                           std::unordered_set<std::string> &drop_case_names) {
  for (const auto &based_case : based_cases) {
    autoschedule::AutoScheduleOutput generated_output(strategy.GenName(based_case.scheduled_graph.GetName()).c_str());
    GE_ASSERT_TRUE(generated_output.scheduled_graph.CopyFrom(based_case.scheduled_graph));
    generated_output.var_relations_ = based_case.var_relations_;

    if (strategy.Generate(origin_graph, based_case.scheduled_graph, generated_output.scheduled_graph) != af::SUCCESS) {
      GELOGD("Generate template failed, %s.", generated_output.scheduled_graph.GetName().c_str());
      continue;
    }

    // 生成打分函数
    const auto score_func = strategy.GetScoreFunc(origin_graph, generated_output.scheduled_graph);
    if (!score_func.empty()) {
      generated_output.score_func = score_func;
    }

    GELOGD("Generate template success, %s.", generated_output.scheduled_graph.GetName().c_str());
    generated_cases.push_back(generated_output);

    if (strategy.NeedDropBasedCase(origin_graph, based_case.scheduled_graph, generated_output.scheduled_graph)) {
      GELOGD("New template is better than original general template, drop it, %s.",
             based_case.scheduled_graph.GetName().c_str());
      drop_case_names.emplace(based_case.scheduled_graph.GetName());
    }
  }
  return af::SUCCESS;
}

af::Status BaseTemplateGenerator::GenerateTemplates(const af::AscGraph &origin_graph,
                                                    std::vector<autoschedule::AutoScheduleOutput> &tiling_cases) {
  if (strategies_.empty()) {
    GELOGD("No template strategies found.");
    return af::SUCCESS;
  }
  std::vector<autoschedule::AutoScheduleOutput> generated_cases;
  std::unordered_set<std::string> drop_case_names;

  for (const auto &strategy : strategies_) {
    GE_CHECK_NOTNULL(strategy);
    const auto &based_cases = GetBasedCasesByGenMode(strategy->GetGenerationMode(), tiling_cases, generated_cases);
    GE_ASSERT_SUCCESS(Generate(*strategy, origin_graph, based_cases, generated_cases, drop_case_names));
  }

  if (generated_cases.empty() && drop_case_names.empty()) {
    return af::SUCCESS;
  }

  DedupNddmaTemplates(generated_cases, drop_case_names);

  // 拼接生成的新模板
  tiling_cases.insert(tiling_cases.end(), generated_cases.begin(), generated_cases.end());

  // 把需要删除的模板删掉
  if (!drop_case_names.empty()) {
    std::vector<autoschedule::AutoScheduleOutput> reserved_cases;
    reserved_cases.reserve(tiling_cases.size());
    for (const auto &c : tiling_cases) {
      if (drop_case_names.count(c.scheduled_graph.GetName()) == 0UL) {
        reserved_cases.push_back(c);
      }
    }
    tiling_cases.swap(reserved_cases);
  }
  return af::SUCCESS;
}

}  // namespace optimize
