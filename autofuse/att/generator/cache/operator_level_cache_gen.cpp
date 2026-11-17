/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "operator_level_cache_gen.h"
#include <set>
#include <utility>
#include "common/code_printer.h"
#include "generator/preprocess/args_manager.h"
#include "util/base_types_printer.h"

namespace att {
namespace cache {
namespace {
std::vector<std::pair<std::string, std::string>> GetVarAccessors(const TilingModelInfo &tiling_model_info) {
  std::vector<std::pair<std::string, std::string>> var_accessors;
  std::set<std::string> visited_var_names;
  std::set<std::string> all_groups_prefix;
  for (const auto &model_info : tiling_model_info) {
    all_groups_prefix.insert(model_info.schedule_group_ident.GetGroupPrefix());
  }
  for (const auto &model_info : tiling_model_info) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto input_vars = args_manager.GetInputVars();
    bool is_unique_group = (all_groups_prefix.size() == 1);
    std::string group_prefix =
        is_unique_group ? "" : (model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data.");

    for (const auto &var : input_vars) {
      std::string var_name = Str(var);
      if (visited_var_names.find(var_name) == visited_var_names.end()) {
        visited_var_names.insert(var_name);
        std::string accessor = "tiling_data." + group_prefix + "get_" + var_name + "()";
        var_accessors.emplace_back(var_name, accessor);
      }
    }
  }
  return var_accessors;
}

std::string GenShapeKeyToStringCode(const std::string &key_name) {
  std::stringstream ss;
  ss << "[&" << key_name << "]()->std::string {\n";
  ss << "  std::string out;\n";
  ss << "  for (size_t i = 0; i < " << key_name << ".size(); ++i) {\n";
  ss << "    if (i != 0) {\n";
  ss << "      out.append(\",\");\n";
  ss << "    }\n";
  ss << "    out.append(std::to_string(" << key_name << "[i]));\n";
  ss << "  }\n";
  ss << "  return out;\n";
  ss << "}.operator()().c_str()";
  return ss.str();
}

std::string GenOperatorCacheLog(const std::string &message, const std::string &key_name) {
  if (key_name.empty()) {
    return "    OP_LOGD(OP_NAME, \"[Operator Cache] " + message + " key=[]\");";
  }
  return "    OP_LOGD(OP_NAME, \"[Operator Cache] " + message + " key=[%s]\", " + GenShapeKeyToStringCode(key_name) +
         ");";
}

std::string GenOperatorCacheAgingLog(const std::string &cache_context, const std::string &key_name) {
  const std::string min_count = cache_context + "::GetLastAgedMinCount()";
  if (key_name.empty()) {
    return "    OP_LOGD(OP_NAME, \"[Operator Cache] CACHE CLEARED AND SAVE SUCCESS: min_count=%lu, key=[]\", " +
           min_count + ");";
  }
  return "    OP_LOGD(OP_NAME, \"[Operator Cache] CACHE CLEARED AND SAVE SUCCESS: min_count=%lu, key=[%s]\", " +
         min_count + ", " + GenShapeKeyToStringCode(key_name) + ");";
}

void GenCacheQueryCode(ge::CodePrinter &code_printer, const std::string &cache_context, const std::string &key_name,
                       bool return_on_hit, const std::string &query_guard) {
  if (!query_guard.empty()) {
    code_printer.AddLine("  if (" + query_guard + ") {");
  }
  code_printer.AddLine("  auto *cached_tiling_data = " + cache_context + "::FindOperatorCache(operator_cache_key);");
  code_printer.AddLine("  if (cached_tiling_data != nullptr) {");
  code_printer.AddLine("    memcpy(&tiling_data, cached_tiling_data, sizeof(tiling_data));");
  code_printer.AddLine(GenOperatorCacheLog("HIT!", key_name));
  if (return_on_hit) {
    code_printer.AddLine("    return true;");
  } else {
    code_printer.AddLine("    cache_hit = true;");
  }
  code_printer.AddLine("  } else {");
  code_printer.AddLine(GenOperatorCacheLog("MISS!", key_name));
  code_printer.AddLine("  }");
  if (!query_guard.empty()) {
    code_printer.AddLine("  }");
  }
  code_printer.AddLine("");
}

void GenCacheSaveCode(ge::CodePrinter &code_printer, const std::string &cache_context, const std::string &key_name) {
  code_printer.AddLine("  const auto cache_save_result = " + cache_context + "::SaveOperatorCache(operator_cache_key" +
                       ", tiling_data);");
  code_printer.AddLine("  if (cache_save_result == OperatorCacheSaveResult::kSaved) {");
  code_printer.AddLine(GenOperatorCacheLog("SAVE SUCCESS:", key_name));
  code_printer.AddLine("  } else if (cache_save_result == OperatorCacheSaveResult::kClearedAndSaved) {");
  code_printer.AddLine(GenOperatorCacheAgingLog(cache_context, key_name));
  code_printer.AddLine("  } else if (cache_save_result == OperatorCacheSaveResult::kFailed) {");
  code_printer.AddLine(GenOperatorCacheLog("SAVE FAILED:", key_name));
  code_printer.AddLine("  }");
}
}  // namespace

af::Status OperatorLevelCacheGen::GenFixedSizeHashMapDef(ge::CodePrinter &code_printer) {
  // 生成FixedSizeHashMap模板类定义
  std::string hashmap_code = GenHashMapTemplate();
  code_printer.AddLine(hashmap_code);
  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenTilingCacheContext(ge::CodePrinter &code_printer) {
  // 生成TilingCacheContext类
  std::string context_class = GenContextClass();
  code_printer.AddLine(context_class);
  code_printer.AddLine("");
  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenOperatorCacheTypes(ge::CodePrinter &code_printer) {
  // Keep the key layout compact for static-shape operators where
  // std::array<uint32_t, 0> is implementation-defined.
  code_printer.AddLine("#pragma pack(push, 1)");
  code_printer.AddLine("struct OperatorCacheKey {");
  code_printer.AddLine("  std::array<int64_t, kInputShapeSize> input_shapes;");
  code_printer.AddLine("  uint32_t request_block_dim;");
  code_printer.AddLine("  uint32_t request_ub_size;");
  code_printer.AddLine("  bool operator==(const OperatorCacheKey &other) const {");
  code_printer.AddLine(
      "    return input_shapes == other.input_shapes && request_block_dim == other.request_block_dim &&");
  code_printer.AddLine("           request_ub_size == other.request_ub_size;");
  code_printer.AddLine("  }");
  code_printer.AddLine("};");
  code_printer.AddLine("#pragma pack(pop)");
  code_printer.AddLine("");
  // 第一级：算子级缓存（使用显式OperatorCacheKey）
  code_printer.AddLine("template <typename TilingData>");
  code_printer.AddLine(
      "using OperatorLevelCache = FixedSizeHashMap<kInputShapeSize, kOperatorCacheCapacity, TilingData, "
      "OperatorCacheKey>;");
  code_printer.AddLine("");

  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenOperatorCacheFunctions(ge::CodePrinter &code_printer,
                                                            const std::string &tiling_data_type_name) {
  // 生成算子级缓存函数（使用R"()"格式以提高性能）
  std::string find_func = R"(
bool FindOperatorCache(const OperatorCacheKey& key, )" +
                          tiling_data_type_name + R"(& tiling_data, OperatorLevelCache<)" + tiling_data_type_name +
                          R"(>& cache) {
  const auto* result = cache.Find(key);
  if (result != nullptr) {
    tiling_data = *result;
    return true;
  }
  return false;
}
)";

  std::string save_func = R"(
bool SaveOperatorCache(const OperatorCacheKey& key, const )" +
                          tiling_data_type_name + R"(& tiling_data, OperatorLevelCache<)" + tiling_data_type_name +
                          R"(>& cache) {
  return cache.Insert(key, tiling_data);
}
)";

  code_printer.AddLine(find_func);
  code_printer.AddLine(save_func);

  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenSaveCacheCalls(ge::CodePrinter &code_printer,
                                                    const TilingModelInfo &tiling_model_info,
                                                    const TilingCodeGenConfig &config) {
  if (!config.cache_enabled_at_compile_time) {
    return af::SUCCESS;
  }
  const auto var_accessors = GetVarAccessors(tiling_model_info);
  if (var_accessors.empty()) {
    // 静态Shape场景：使用空key进行缓存
    GELOGI("Static shape detected, using empty key for operator level cache, model[%s].",
           tiling_model_info[0].graph_name.c_str());
    code_printer.AddLine("  // 静态Shape场景：input_shapes 使用全零，但资源请求仍参与缓存 key");
    GenCacheSaveCode(code_printer, "TilingCacheContext<" + config.tiling_data_type_name + ">", "");
    return af::SUCCESS;
  }
  GenCacheSaveCode(code_printer, "TilingCacheContext<" + config.tiling_data_type_name + ">", "input_shapes");
  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenInitAndQueryCacheCode(ge::CodePrinter &code_printer,
                                                           const TilingModelInfo &tiling_model_info,
                                                           const TilingCodeGenConfig &config, bool return_on_hit,
                                                           const std::string &query_guard) {
  if (!config.cache_enabled_at_compile_time) {
    return af::SUCCESS;
  }
  const std::string cache_context = "TilingCacheContext<" + config.tiling_data_type_name + ">";
  const auto var_accessors = GetVarAccessors(tiling_model_info);
  if (var_accessors.empty()) {
    // 静态Shape场景：使用空key进行缓存查询
    GELOGI("Static shape detected, using empty key for operator level cache query, model[%s].",
           tiling_model_info[0].graph_name.c_str());
    code_printer.AddLine("  // 静态Shape场景：算子级缓存查询（全零 input_shapes）");
    code_printer.AddLine("  std::array<int64_t, kInputShapeSize> input_shapes = {};");
    code_printer.AddLine("  const uint32_t request_block_dim = tiling_data.get_block_dim();");
    code_printer.AddLine("  const uint32_t request_ub_size = tiling_data.get_ub_size();");
    code_printer.AddLine(
        "  const OperatorCacheKey operator_cache_key{input_shapes, request_block_dim, request_ub_size};");
    GenCacheQueryCode(code_printer, cache_context, "", return_on_hit, query_guard);
    return af::SUCCESS;
  }

  code_printer.AddLine("  // 第一级：算子级缓存查询，收集所有原始轴");
  std::string array_init = "  std::array<int64_t, kInputShapeSize> input_shapes = {";
  for (size_t i = 0; i < var_accessors.size(); ++i) {
    if (i > 0) {
      array_init += ", ";
    }
    array_init += var_accessors[i].second;
  }
  array_init += "};";
  code_printer.AddLine(array_init);

  code_printer.AddLine("  const uint32_t request_block_dim = tiling_data.get_block_dim();");
  code_printer.AddLine("  const uint32_t request_ub_size = tiling_data.get_ub_size();");
  code_printer.AddLine(
      "  const OperatorCacheKey operator_cache_key{input_shapes, request_block_dim, request_ub_size};");

  GenCacheQueryCode(code_printer, cache_context, "input_shapes", return_on_hit, query_guard);

  return af::SUCCESS;
}

std::string OperatorLevelCacheGen::GenContextClass() {
  std::stringstream ss;

  ss << R"(
enum class OperatorCacheSaveResult {
  kSaved,
  kClearedAndSaved,
  kFailed,
};

/**
 * @brief Tiling缓存上下文类
 * 线程级别的缓存上下文，使用thread_local存储，无需线程ID
 */
template <typename TilingData>
class TilingCacheContext {
)" << GenContextClassStructure()
     << GenContextClassPublicMethods() << GenContextCacheOperations() << GenContextHashFunction() << R"(
};
)";

  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextClassStructure() {
  std::stringstream ss;

  ss << R"(
private:
  // 第一级：算子级缓存（thread_local，使用unique_ptr避免栈溢出）
  // 注意：使用kInputShapeSize大小的key，以支持不同数量的输入变量
  inline static thread_local std::unique_ptr<OperatorLevelCache<TilingData>> operator_cache_;
  inline static thread_local bool initialized_ = false;
  // 访问计数（用于LRU老化）
  inline static thread_local std::array<uint64_t, kOperatorCacheCapacity> access_counts_;
  inline static thread_local uint64_t last_aged_min_count_ = 0;
)";

  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextClassPublicMethods() {
  std::stringstream ss;

  ss << R"(
public:

  // 获取算子级缓存实例
  static OperatorLevelCache<TilingData>& GetOperatorCache() {
    if (!initialized_) {
      initialized_ = true;
      operator_cache_ = std::make_unique<OperatorLevelCache<TilingData>>();
      // 初始化访问计数
      for (size_t i = 0; i < kOperatorCacheCapacity; ++i) {
        access_counts_[i] = 0;
      }
      last_aged_min_count_ = 0;
    }
    return *operator_cache_;
  }

  // 清除算子级缓存
  static void ClearOperatorCache() {
    operator_cache_.reset();
    initialized_ = false;
  }

  static uint64_t GetLastAgedMinCount() {
    return last_aged_min_count_;
  }
)";

  return ss.str();
}

std::string OperatorLevelCacheGen::GenFindOperatorCacheImpl() {
  std::stringstream ss;
  ss << R"(
  // 查询算子级缓存（更新访问计数）
  static )"
     << "TilingData" << R"(* FindOperatorCache(const OperatorCacheKey& key) {
    )"
     << "TilingData"
     << R"(* result = GetOperatorCache().Find(key);
    if (result != nullptr) {
      // 更新访问计数
      size_t hash = Hash(key);
      size_t index = hash % kOperatorCacheCapacity;
      access_counts_[index]++;
    }
    return result;
  }
)";
  return ss.str();
}

std::string OperatorLevelCacheGen::GenSaveOperatorCacheImpl() {
  std::stringstream ss;
  ss << R"(
  // 插入算子级缓存（带LRU老化）
  static OperatorCacheSaveResult SaveOperatorCache(const OperatorCacheKey& key,
                                                   const )"
     << "TilingData"
     << R"(& tiling_data) {
    auto& cache = GetOperatorCache();

    // 1. 尝试直接插入
    if (cache.Insert(key, tiling_data)) {
      return OperatorCacheSaveResult::kSaved;
    }

    // 2. 缓存满，执行LRU老化
    if (cache.Size() >= kOperatorCacheCapacity * kLoadFactorThreshold) {
      uint64_t min_count = access_counts_[0];
      for (size_t i = 1; i < kOperatorCacheCapacity; ++i) {
        if (access_counts_[i] < min_count) {
          min_count = access_counts_[i];
        }
      }
      last_aged_min_count_ = min_count;

      // 清空缓存后重新插入
      cache.Clear();
      for (size_t i = 0; i < kOperatorCacheCapacity; ++i) {
        access_counts_[i] = 0;
      }

      // 重新插入
      return cache.Insert(key, tiling_data) ? OperatorCacheSaveResult::kClearedAndSaved
                                                  : OperatorCacheSaveResult::kFailed;
    }

    return OperatorCacheSaveResult::kFailed;
  }
)";
  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextCacheOperations() {
  std::stringstream ss;
  ss << GenFindOperatorCacheImpl();
  ss << "\n";
  ss << GenSaveOperatorCacheImpl();
  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextHashFunction() {
  std::stringstream ss;

  ss << R"(
private:
  // Hash函数
  static size_t Hash(const OperatorCacheKey& key) {
    size_t hash = 0;
    for (const auto& value : key.input_shapes) {
      constexpr uint32_t kHashPrime = 0x9e3779b9;  // 黄金比例的整数表示，用于hash混合
      hash ^= value + kHashPrime + (hash << 6) + (hash >> 2);
    }
    constexpr uint32_t kHashPrime = 0x9e3779b9;
    hash ^= key.request_block_dim + kHashPrime + (hash << 6) + (hash >> 2);
    hash ^= key.request_ub_size + kHashPrime + (hash << 6) + (hash >> 2);
    return hash;
  }
)";

  return ss.str();
}
}  // namespace cache
}  // namespace att
