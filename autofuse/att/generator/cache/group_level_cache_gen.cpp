/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "group_level_cache_gen.h"
#include "common/code_printer.h"

namespace att {
namespace cache {
af::Status GroupLevelCacheGen::GenFixedSizeHashMapDef(ge::CodePrinter &code_printer) {
  // 生成FixedSizeHashMap模板类定义
  std::string hashmap_code = GenHashMapTemplate();
  code_printer.AddLine(hashmap_code);
  return af::SUCCESS;
}

af::Status GroupLevelCacheGen::GenGroupCacheTypes(ge::CodePrinter &code_printer, size_t cache_capacity) {
  // 注意：常量定义(kInputShapeSize等)已在GenCacheHashMapDef中统一生成
  // 这里只生成GroupLevelCache类型定义

  // 第二级：Group间缓存（使用TilingDataCopy）
  code_printer.AddLine("using GroupLevelCache = FixedSizeHashMap<kInputShapeSize, " + std::to_string(cache_capacity) +
                       ", TilingDataCopy>;");
  code_printer.AddLine("");

  return af::SUCCESS;
}

af::Status GroupLevelCacheGen::GenGroupCacheFunctions(ge::CodePrinter &code_printer,
                                                      const std::string &tiling_data_type_name) {
  // 生成Group级缓存查找和保存函数
  // 改为参数传递方式，不使用全局静态变量
  // 生成缓存函数
  std::string cache_decl = R"(
// 第二级：Group间缓存（通过参数传递）
static inline bool FindGroupCache(const std::array<int64_t, kInputShapeSize> &key,
                                   )" +
                           tiling_data_type_name + R"(& tiling_data,
                                   GroupLevelCache &group_level_cache) {
  auto *result = group_level_cache.Find(key);
  if (result != nullptr) {
    OP_LOGI(OP_NAME, "[Group Cache] HIT! key=[%s]", [&key]()->std::string {
      std::string out;
      for (auto axis : key) {
        out.append(std::to_string(axis));
      }
      return out;
    }.operator()().c_str());
    GetScheduleGroupTilingData(*result, tiling_data);
    return true;
  }
  OP_LOGI(OP_NAME, "[Group Cache] MISS! key=[%s]", [&key]()->std::string {
    std::string out;
    for (auto axis : key) {
      out.append(std::to_string(axis));
    }
    return out;
  }.operator()().c_str());
  return false;
}

// 保存到Group级缓存
static inline bool SaveGroupCache(const std::array<int64_t, kInputShapeSize>& key,
                                   const TilingDataCopy& data,
                                   GroupLevelCache &group_level_cache) {
  bool success = group_level_cache.Insert(key, data);
  OP_LOGI(OP_NAME, "[Group Cache] SAVE %s: key=[%s], tiling_key=%u",
         success ? "SUCCESS" : "FAILED", [&key]()->std::string {
           std::string out;
           for (auto axis : key) {
             out.append(std::to_string(axis));
           }
           return out;
         }.operator()().c_str(), static_cast<unsigned>(data.tiling_key));
  return success;
}

)";
  code_printer.AddLine(cache_decl);

  return af::SUCCESS;
}
}  // namespace cache
}  // namespace att
