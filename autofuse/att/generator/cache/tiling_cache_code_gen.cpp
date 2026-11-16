/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_cache_code_gen.h"
#include "common/code_printer.h"

namespace att {
namespace cache {
void TilingCacheCodeGen::GenConstantDefs(ge::CodePrinter &code_printer, size_t input_vars_size) {
  // 直接定义常量值，避免optiling命名空间访问问题
  code_printer.AddLine("// ATT缓存相关常量");
  code_printer.AddLine("constexpr size_t kInputShapeSize = " + std::to_string(input_vars_size) + ";");
  code_printer.AddLine("constexpr size_t kOperatorCacheCapacity = 128;  // 算子级缓存容量");
  code_printer.AddLine("constexpr double kLoadFactorThreshold = 0.8;   // 负载因子阈值");
  code_printer.AddLine("");
}

std::string TilingCacheCodeGen::GenHashMapTemplate() {
  std::stringstream ss;

  ss << "template <size_t KEY_SIZE, size_t CAPACITY, typename VALUE_TYPE, "
        "typename KEY_TYPE = std::array<int64_t, KEY_SIZE>>\n";
  ss << "class FixedSizeHashMap {\n";
  ss << GenHashMapClassStructure();
  ss << GenHashMapConstructor();
  ss << GenHashMapPublicMethods();
  ss << "};\n";

  return ss.str();
}

std::string TilingCacheCodeGen::GenHashMapClassStructure() {
  std::stringstream ss;

  ss << "private:\n";
  ss << "  using Key = KEY_TYPE;\n";
  ss << "  using Value = VALUE_TYPE;\n";
  ss << "\n";
  ss << "  enum BucketState { kEmpty, kOccupied, kDeleted };\n";
  ss << "  struct Bucket {\n";
  ss << "    Key key;\n";
  ss << "    Value value;\n";
  ss << "    BucketState state;\n";
  ss << "  };\n";
  ss << "\n";
  ss << "  std::array<Bucket, CAPACITY> buckets;\n";
  ss << "  size_t size_ = 0;\n";
  ss << "\n";
  ss << "  // Hash - 大驼峰命名\n";
  ss << GenHashFunction();
  ss << "\n";
  ss << "  // FindIndex - 大驼峰命名\n";
  ss << GenFindIndexFunction();
  ss << "\n";

  return ss.str();
}

std::string TilingCacheCodeGen::GenHashMapConstructor() {
  std::stringstream ss;

  ss << "public:\n";
  ss << "  FixedSizeHashMap() : size_(0) {\n";
  ss << "    for (size_t i = 0; i < CAPACITY; ++i) {\n";
  ss << "      buckets[i].state = kEmpty;\n";
  ss << "    }\n";
  ss << "  }\n";
  ss << "\n";

  return ss.str();
}

std::string TilingCacheCodeGen::GenFindMethod() {
  std::stringstream ss;
  ss << "  // Find - 大驼峰命名\n";
  ss << "  Value* Find(const Key &key) {\n";
  ss << "    size_t index = FindIndex(key);\n";
  ss << "    if (index < CAPACITY && buckets[index].state == kOccupied) {\n";
  ss << "      return &buckets[index].value;\n";
  ss << "    }\n";
  ss << "    return nullptr;\n";
  ss << "  }\n";
  ss << "\n";
  ss << "  const Value* Find(const Key &key) const {\n";
  ss << "    return const_cast<FixedSizeHashMap*>(this)->Find(key);\n";
  ss << "  }\n";
  return ss.str();
}

std::string TilingCacheCodeGen::GenInsertMethod() {
  std::stringstream ss;
  ss << "  // Insert - 大驼峰命名\n";
  ss << "  bool Insert(const Key &key, const Value &value) {\n";
  ss << "    if (size_ >= CAPACITY * kLoadFactorThreshold) {\n";
  ss << "      return false;  // 80%容量阈值\n";
  ss << "    }\n";
  ss << "    size_t index = FindIndex(key);\n";
  ss << "    if (index >= CAPACITY) {\n";
  ss << "      size_t hash = Hash(key) % CAPACITY;\n";
  ss << "      for (size_t i = 0; i < CAPACITY; ++i) {\n";
  ss << "        index = (hash + i) % CAPACITY;\n";
  ss << "        if (buckets[index].state == kEmpty) {\n";
  ss << "          buckets[index].key = key;\n";
  ss << "          buckets[index].value = value;\n";
  ss << "          buckets[index].state = kOccupied;\n";
  ss << "          size_++;\n";
  ss << "          return true;\n";
  ss << "        }\n";
  ss << "      }\n";
  ss << "      return false;\n";
  ss << "    }\n";
  ss << "    buckets[index].value = value;\n";
  ss << "    return true;\n";
  ss << "  }\n";
  return ss.str();
}

std::string TilingCacheCodeGen::GenEraseMethod() {
  std::stringstream ss;
  ss << "  // Erase - 大驼峰命名\n";
  ss << "  bool Erase(const Key &key) {\n";
  ss << "    size_t index = FindIndex(key);\n";
  ss << "    if (index < CAPACITY && buckets[index].state == kOccupied) {\n";
  ss << "      buckets[index].state = kDeleted;\n";
  ss << "      size_--;\n";
  ss << "      return true;\n";
  ss << "    }\n";
  ss << "    return false;\n";
  ss << "  }\n";
  return ss.str();
}

std::string TilingCacheCodeGen::GenClearAndSizeMethods() {
  std::stringstream ss;
  ss << "  // Clear - 大驼峰命名\n";
  ss << "  void Clear() {\n";
  ss << "    for (auto& bucket : buckets) {\n";
  ss << "      bucket.state = kEmpty;\n";
  ss << "    }\n";
  ss << "    size_ = 0;\n";
  ss << "  }\n";
  ss << "\n";
  ss << "  size_t Size() const { return size_; }\n";
  ss << "  bool Empty() const { return size_ == 0; }\n";
  return ss.str();
}

std::string TilingCacheCodeGen::GenHashMapPublicMethods() {
  std::stringstream ss;
  ss << GenFindMethod();
  ss << "\n";
  ss << GenInsertMethod();
  ss << "\n";
  ss << GenEraseMethod();
  ss << "\n";
  ss << GenClearAndSizeMethods();
  return ss.str();
}

std::string TilingCacheCodeGen::GenHashFunction() {
  std::stringstream ss;

  ss << "  size_t Hash(const Key &key) const {\n";
  ss << "    size_t hash = 0;\n";
  ss << "    constexpr uint32_t kHashPrime = 0x9e3779b9;  // 黄金比例的整数表示，用于hash混合\n";
  ss << "    if constexpr (std::is_same_v<Key, std::array<int64_t, KEY_SIZE>>) {\n";
  ss << "      for (const auto& value : key) {\n";
  ss << "        hash ^= value + kHashPrime + (hash << 6) + (hash >> 2);\n";
  ss << "      }\n";
  ss << "    } else {\n";
  ss << "      for (const auto& value : key.input_shapes) {\n";
  ss << "        hash ^= value + kHashPrime + (hash << 6) + (hash >> 2);\n";
  ss << "      }\n";
  ss << "      hash ^= key.request_block_dim + kHashPrime + (hash << 6) + (hash >> 2);\n";
  ss << "      hash ^= key.request_ub_size + kHashPrime + (hash << 6) + (hash >> 2);\n";
  ss << "    }\n";
  ss << "    return hash;\n";
  ss << "  }\n";

  return ss.str();
}

std::string TilingCacheCodeGen::GenFindIndexFunction() {
  std::stringstream ss;

  ss << "  size_t FindIndex(const Key &key) const {\n";
  ss << "    size_t hash = Hash(key) % CAPACITY;\n";
  ss << "    size_t start = hash;\n";
  ss << "    do {\n";
  ss << "      if (buckets[hash].state == kEmpty) {\n";
  ss << "        return CAPACITY;\n";
  ss << "      } else if (buckets[hash].state == kOccupied && buckets[hash].key == key) {\n";
  ss << "        return hash;\n";
  ss << "      }\n";
  ss << "      hash = (hash + 1) % CAPACITY;\n";
  ss << "    } while (hash != start);\n";
  ss << "    return CAPACITY;\n";
  ss << "  }\n";

  return ss.str();
}
}  // namespace cache
}  // namespace att
