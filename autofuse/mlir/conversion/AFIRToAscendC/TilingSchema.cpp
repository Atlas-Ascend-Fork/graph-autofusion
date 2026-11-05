/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"

namespace mlir::ascend::kernel {
namespace {

bool isValidLayout(MLIRContext *context, StringAttr name, ArrayAttr types, ArrayAttr names,
                   llvm::DenseSet<Type> &visited) {
  if (!context || !name || name.getValue().empty() || !types || !names || types.size() != names.size() ||
      name.getContext() != context || types.getContext() != context || names.getContext() != context) {
    return false;
  }
  llvm::StringSet<> fields;
  for (auto [fieldName, fieldType] : llvm::zip(names, types)) {
    auto stringName = dyn_cast<StringAttr>(fieldName);
    auto typeAttr = dyn_cast<TypeAttr>(fieldType);
    if (!stringName || stringName.getValue().empty() || !typeAttr || stringName.getContext() != context ||
        typeAttr.getValue().getContext() != context || !fields.insert(stringName.getValue()).second) {
      return false;
    }
    auto nested = dyn_cast<emitasc::PyStructType>(typeAttr.getValue());
    if (nested && visited.insert(nested).second &&
        !isValidLayout(context, nested.getNameAttr(), nested.getTypesAttr(), nested.getNamesAttr(), visited)) {
      return false;
    }
  }
  return true;
}

}  // namespace

TilingSchema TilingSchema::get(Type type) {
  auto structure = dyn_cast_or_null<emitasc::PyStructType>(type);
  if (!structure) {
    return {};
  }
  llvm::DenseSet<Type> visited;
  visited.insert(type);
  if (!isValidLayout(type.getContext(), structure.getNameAttr(), structure.getTypesAttr(), structure.getNamesAttr(),
                     visited)) {
    return {};
  }
  return TilingSchema(type);
}

TilingSchema TilingSchema::get(MLIRContext *context, StringAttr name, ArrayAttr types, ArrayAttr names) {
  llvm::DenseSet<Type> visited;
  if (!isValidLayout(context, name, types, names, visited)) {
    return {};
  }
  return TilingSchema(emitasc::PyStructType::get(context, name, types, names));
}

StringAttr TilingSchema::getNameAttr() const {
  return cast<emitasc::PyStructType>(type_).getNameAttr();
}
ArrayAttr TilingSchema::getTypesAttr() const {
  return cast<emitasc::PyStructType>(type_).getTypesAttr();
}
ArrayAttr TilingSchema::getNamesAttr() const {
  return cast<emitasc::PyStructType>(type_).getNamesAttr();
}

Type TilingSchema::getField(StringRef field) const {
  if (!*this) {
    return {};
  }
  for (auto [name, fieldType] : llvm::zip(getNamesAttr(), getTypesAttr())) {
    if (cast<StringAttr>(name).getValue() == field) {
      return cast<TypeAttr>(fieldType).getValue();
    }
  }
  return {};
}

TilingSchema getTilingSchema(Type type) {
  return TilingSchema::get(type);
}

Type getTilingSchemaField(Type type, StringRef field) {
  return getTilingSchema(type).getField(field);
}

Type getApiTilingFieldType(Operation *op, StringRef field) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return {};
  }
  auto tiling = getCanonicalTilingArgument(func);
  return succeeded(tiling) ? getTilingSchemaField(tiling->getType(), field) : Type();
}

Value createTilingSchemaMember(OpBuilder &builder, Location loc, Value value, StringRef field) {
  return createTilingSchemaMember(builder, loc, getTilingSchema(value.getType()), value, field);
}

Value createTilingSchemaMember(OpBuilder &builder, Location loc, const TilingSchema &schema, Value value,
                               StringRef field) {
  return builder.create<emitasc::MemberOp>(loc, schema.getField(field), value, field);
}

}  // namespace mlir::ascend::kernel
