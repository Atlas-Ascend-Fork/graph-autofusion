/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_TILINGSCHEMA_H
#define AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_TILINGSCHEMA_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir::ascend::kernel {

// Checked, read-only view of an EmitAsc tiling structure. The MLIRContext owns
// its storage; field order and integer widths are preserved.
class TilingSchema {
 public:
  TilingSchema() = default;

  // Return an empty schema for a non-struct or malformed field layout.
  static TilingSchema get(Type type);
  static TilingSchema get(MLIRContext *context, StringAttr name, ArrayAttr types, ArrayAttr names);

  explicit operator bool() const {
    return static_cast<bool>(type_);
  }
  operator Type() const {
    return type_;
  }
  Type getType() const {
    return type_;
  }
  MLIRContext *getContext() const {
    return type_.getContext();
  }
  StringAttr getNameAttr() const;
  ArrayAttr getTypesAttr() const;
  ArrayAttr getNamesAttr() const;

  // Look up a field's type by name. Assumes the schema was already validated
  // by get(): unlike getTilingSchemaField(Type, ...), this does not re-walk
  // the layout, so it is safe to call per field in a loop over one schema.
  // Returns an empty type if the field is not present.
  Type getField(StringRef field) const;

 private:
  explicit TilingSchema(Type type) : type_(type) {}
  Type type_;
};

TilingSchema getTilingSchema(Type type);
// Return an empty type for invalid schemas or missing fields.
Type getTilingSchemaField(Type type, StringRef field);

// Materialize a field access through the validated schema view. This is the
// single AscendC-side entry point for reading TilingData members.
Value createTilingSchemaMember(OpBuilder &builder, Location loc, Value value, StringRef field);
// Same, but for a schema already validated by the caller (e.g. inside a loop
// over several fields of the same struct) so the layout is not re-walked.
Value createTilingSchemaMember(OpBuilder &builder, Location loc, const TilingSchema &schema, Value value,
                               StringRef field);

}  // namespace mlir::ascend::kernel

#endif  // AUTOFUSE_MLIR_CONVERSION_AFIRTOASCENDC_TILINGSCHEMA_H
