/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_TILING_FIELD_SCHEMA_H
#define AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_TILING_FIELD_SCHEMA_H

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/schedule_result.h"

namespace mlir::common {

// Bump this value when the legacy ATT/Codegen field contract changes.  The
// schema version is part of the artifact/cache compatibility boundary.
inline constexpr char kTilingFieldSchemaVersion[] = "autofuse.tiling.schema.v2";

// Joins a TilingField's (path, name) into the composite key used to dedup
// fields within one scope -- see TilingField::path below. Not a valid
// identifier character, so it cannot collide with either component.
inline constexpr char kTilingFieldScopeKeySeparator = '\x1f';

enum class TilingFieldType { kUInt32, kUInt64, kOpaqueStruct };

struct TilingField {
  std::string name;
  // Scope this field was generated in, e.g. "fusion" for fusion-level common
  // fields, "" for the flat single-(subgraph,plan,stage) case, or
  // "graph<subgraph>_result<plan>_g<stage>_tiling_data" for the nested
  // multi-group case — which is also that group's member name in the outermost
  // TilingData, see schedule_group_names.h.  This is not just diagnostic metadata: it is the
  // dedup scope key (see BuildTilingFieldSchema) and the struct-nesting key
  // used by EmitTilingDataStructCpp, mirroring legacy ProcessSingleGroup
  // (one flat scope) vs. ProcessMultiGroup (one isolated scope per
  // (subgraph, plan, stage)).
  std::string path;
  TilingFieldType type{TilingFieldType::kUInt32};
  // Populated only for kOpaqueStruct fields that come from an API-tiling
  // node (e.g. Transpose/Pad) whose device-side type is already known via
  // ascgen_utils::GetApiTilingTypeName.  Empty means the opaque field's
  // concrete type is not yet known and is emitted as a comment placeholder.
  std::string device_type_name;
};

struct TilingFieldSchema {
  std::vector<TilingField> fields;
  std::string version{kTilingFieldSchemaVersion};
};

// Transitional builder for the non-MLIR schema consumed by legacy Codegen and
// ATT.  It intentionally depends on the legacy schedule/AscGraph model.  When
// ATT is migrated, replace this input boundary with an MLIR/DTO boundary
// rather than carrying this dependency into the final MLIR common layer.
// The input is kept at FusedScheduledResult level because workspace fields and
// graph-level tiling keys are fusion-level data, not properties of one impl graph.
bool BuildTilingFieldSchema(const ascir::FusedScheduledResult &fused_result, TilingFieldSchema &schema,
                            std::string *diagnostic = nullptr);

// Transitional overload for legacy Codegen/ATT call sites that already
// iterate ScheduledResult objects.  This overload is expected to be replaced
// together with the legacy ATT schedule representation.
bool BuildTilingFieldSchema(const ascir::ScheduledResult &scheduled_result, size_t subgraph_id, size_t plan_id,
                            TilingFieldSchema &schema, std::string *diagnostic = nullptr);

// Emit the stable primitive portion of the existing TilingData declaration.
// Opaque fields are retained in the schema but are emitted as comments (or,
// when device_type_name is known, as TILING_DATA_FIELD_DEF_T_STRUCT
// references) until the corresponding API adapter is fully migrated.
std::string EmitTilingDataStructCpp(const TilingFieldSchema &schema,
                                    std::string_view struct_name = "AutofuseTilingData");

std::optional<std::string> LookupFieldName(const TilingFieldSchema &schema, std::string_view semantic_key);

}  // namespace mlir::common

#endif  // AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_TILING_FIELD_SCHEMA_H
