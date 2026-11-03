/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/Emit/TilingDataHeaderEmitter.h"

#include "final_tiling_selection_snapshot.h"
#include "support/CxxIdentifier.h"

#include <string>

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {
// ATT 侧的 TilingData 标量字段一律是无符号定宽整数，三处发射（字段声明、字段访问器、
// root 转发访问器）必须写出同一个拼法。
std::string scalarTypeName(mlir::IntegerType integer) {
  return ("uint" + llvm::Twine(integer.getWidth()) + "_t").str();
}

struct NestedScalarOwner {
  mlir::ascend::kernel::TilingSchema type;
  std::string path;
};

void collectNestedScalarOwners(mlir::ascend::kernel::TilingSchema structure, const std::string &path,
                               llvm::SmallVectorImpl<NestedScalarOwner> &owners) {
  for (auto [name, fieldType] : llvm::zip(structure.getNamesAttr(), structure.getTypesAttr())) {
    auto fieldName = cast<StringAttr>(name);
    auto typeAttr = cast<TypeAttr>(fieldType);
    auto nested = mlir::ascend::kernel::getTilingSchema(typeAttr.getValue());
    if (!nested) {
      continue;
    }
    const std::string nestedPath = path.empty() ? fieldName.getValue().str() : path + "." + fieldName.getValue().str();
    if (llvm::any_of(nested.getTypesAttr(), [](Attribute attr) {
          auto type = cast<TypeAttr>(attr);
          return isa<IntegerType>(type.getValue());
        })) {
      owners.push_back({nested, nestedPath});
    } else {
      collectNestedScalarOwners(nested, nestedPath, owners);
    }
  }
}

LogicalResult emitRootForwardingAccessors(mlir::ascend::kernel::TilingSchema structure, llvm::raw_ostream &output) {
  if (structure.getNameAttr().getValue() != "AutofuseTilingData") {
    return success();
  }
  llvm::SmallVector<NestedScalarOwner> owners;
  collectNestedScalarOwners(structure, {}, owners);
  if (owners.size() != 1) {
    return success();
  }
  const NestedScalarOwner &owner = owners.front();
  llvm::StringSet<> rootFields;
  for (Attribute name : structure.getNamesAttr()) {
    auto field = cast<StringAttr>(name);
    rootFields.insert(field.getValue());
  }
  for (auto [name, fieldType] : llvm::zip(owner.type.getNamesAttr(), owner.type.getTypesAttr())) {
    auto fieldName = cast<StringAttr>(name);
    auto typeAttr = cast<TypeAttr>(fieldType);
    auto integer = dyn_cast<IntegerType>(typeAttr.getValue());
    if (!integer || rootFields.contains(fieldName.getValue())) {
      continue;
    }
    const llvm::StringRef field = fieldName.getValue();
    std::string typeName = scalarTypeName(integer);
    output << "    inline void set_" << field << "(" << typeName << " value) { " << owner.path << ".set_" << field
           << "(value); }\n"
           << "    inline " << typeName << " get_" << field << "() const { return " << owner.path << ".get_" << field
           << "(); }\n"
           << "    inline " << typeName << " *get_addr_" << field << "() { return " << owner.path << ".get_addr_"
           << field << "(); }\n";
  }
  return success();
}

// Legacy ATT sources use root accessors for the single-plan/single-stage case.
// Forward them to the canonical nested MLIR TilingData field until ATT consumes
// schema paths directly.
// TODO: Remove these forwarding accessors after ATT emits schema-qualified paths.
// 结构体的字段签名，用来判同名结构体是否同形。
std::string makeLayoutSignature(mlir::ascend::kernel::TilingSchema structure) {
  std::string layout;
  llvm::raw_string_ostream layoutStream(layout);
  for (auto [name, fieldType] : llvm::zip(structure.getNamesAttr(), structure.getTypesAttr())) {
    layoutStream << cast<StringAttr>(name).getValue() << ":" << cast<TypeAttr>(fieldType).getValue() << ";";
  }
  layoutStream.flush();
  return layout;
}

// 字段直接进 output；访问器先写进 accessorStream，由调用方统一放到结构体末尾。
LogicalResult emitStructFields(mlir::ascend::kernel::TilingSchema structure, llvm::raw_ostream &output,
                               llvm::raw_ostream &accessorStream) {
  for (auto [name, fieldType] : llvm::zip(structure.getNamesAttr(), structure.getTypesAttr())) {
    auto fieldName = cast<StringAttr>(name);
    auto typeAttr = cast<TypeAttr>(fieldType);
    if (!mlir::afir::isValidCxxIdentifier(fieldName.getValue())) {
      return failure();
    }
    Type type = typeAttr.getValue();
    llvm::StringRef field = fieldName.getValue();
    if (auto nested = mlir::ascend::kernel::getTilingSchema(type)) {
      // Matches legacy TILING_DATA_FIELD_DEF_T_STRUCT, which does not
      // generate accessors for nested-struct fields.
      output << "    " << nested.getNameAttr().getValue() << " " << field << ";\n";
      continue;
    }
    auto integer = dyn_cast<IntegerType>(type);
    if (!integer) {
      return failure();
    }
    // ATT-generated tiling_func code (see AttSchemaAdapter's
    // ExtractReferencedFieldNames) calls get_<field>()/set_<field>(...)/
    // get_addr_<field>() on every scalar field, following the legacy
    // TILING_DATA_FIELD_DEF_T macro expansion. Emit the same accessors so
    // that generated code compiles against this MLIR-emitted header.
    std::string typeName = scalarTypeName(integer);
    output << "    " << typeName << " " << field << ";\n";
    accessorStream << "    inline void set_" << field << "(" << typeName << " value) { " << field << " = value; }\n"
                   << "    inline " << typeName << " get_" << field << "() const { return " << field << "; }\n"
                   << "    inline " << typeName << " *get_addr_" << field << "() { return &" << field << "; }\n";
  }
  return success();
}

LogicalResult emitType(Type type, llvm::StringMap<std::string> &layouts, llvm::raw_ostream &output) {
  auto structure = mlir::ascend::kernel::getTilingSchema(type);
  if (!structure) {
    return success();
  }
  auto name = structure.getNameAttr().getValue();
  if (!mlir::afir::isValidCxxIdentifier(name)) {
    return failure();
  }
  std::string layout = makeLayoutSignature(structure);
  auto existingLayout = layouts.find(name);
  if (existingLayout != layouts.end()) {
    // 同名结构体必须同形；已发射过的不再发射第二遍。
    return success(existingLayout->second == layout);
  }
  layouts.try_emplace(name, std::move(layout));
  for (Attribute fieldType : structure.getTypesAttr()) {
    if (failed(emitType(cast<TypeAttr>(fieldType).getValue(), layouts, output))) {
      return failure();
    }
  }

  // 字段与访问器分两段发射：先铺完所有字段，再把访问器get/set统一放到结构体末尾。
  output << "struct " << name << " {\n";
  std::string accessors;
  llvm::raw_string_ostream accessorStream(accessors);
  if (failed(emitStructFields(structure, output, accessorStream)) ||
      failed(emitRootForwardingAccessors(structure, accessorStream))) {
    return failure();
  }
  accessorStream.flush();
  // 全是嵌套结构体字段时一个访问器都没有，那就不要留下一行空行。
  if (!accessors.empty()) {
    output << "\n" << accessors;
  }
  output << "};\n\n";
  if (name == "AutofuseTilingData") {
    output << "struct AutofuseTilingDataPerf {\n"
           << "    AutofuseTilingData tiling_data;\n"
           << "    double best_perf;\n"
           << "    std::vector<FinalTilingSelectionSnapshot> final_tiling_selections;\n"
           << "};\n\n";
  }
  return success();
}
}  // namespace

LogicalResult mlir::ascend::kernel::emitTilingDataHeader(ArrayRef<mlir::ascend::kernel::TilingSchema> schemas,
                                                         llvm::raw_ostream &output) {
  output << "#pragma once\n#include <cstdint>\n#include <vector>\n\n"
         << ascgen_utils::GenFinalTilingSelectionSnapshotSnippet() << "\n";
  llvm::StringMap<std::string> layouts;
  for (mlir::ascend::kernel::TilingSchema schema : schemas) {
    if (!schema || failed(emitType(schema, layouts, output))) {
      return failure();
    }
  }
  return success();
}
