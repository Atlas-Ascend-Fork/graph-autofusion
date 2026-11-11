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

#include <set>
#include <string>

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir::ascend::kernel;

using namespace mlir;
using namespace mlir::afir;

namespace {
// ATT 侧的 TilingData 标量字段一律是无符号定宽整数，三处生成（字段声明、字段访问器、
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
      // generate accessors for nested-struct fields. 单 stage wrapper 额外的
      // 转发访问器见 emitGroupForwardingAccessors。
      output << "    " << nested.getNameAttr().getValue() << " " << field << ";\n";
      continue;
    }
    auto integer = dyn_cast<IntegerType>(type);
    if (!integer) {
      return failure();
    }
    // Emit get_<field>(), set_<field>() and get_addr_<field>() for each scalar,
    // matching legacy TILING_DATA_FIELD_DEF_T. ATT-generated tiling functions call
    // these accessors (see AttSchemaAdapter's ExtractReferencedFieldNames).
    std::string typeName = scalarTypeName(integer);
    output << "    " << typeName << " " << field << ";\n";
    accessorStream << "    inline void set_" << field << "(" << typeName << " value) { " << field << " = value; }\n"
                   << "    inline " << typeName << " get_" << field << "() const { return " << field << "; }\n"
                   << "    inline " << typeName << " *get_addr_" << field << "() { return &" << field << "; }\n";
  }
  return success();
}

// 单 stage 的 wrapper（ATT 侧的「组结构体」，如 `AscGraph0ScheduleResult0G0TilingData`）：
// ATT 多 result 模式在组结构体上**直接**调 `tiling_data.get_block_dim()` / `set_z0t_size(...)`
// （solver、schedule_group_tail），legacy 那一层对应的是逐组扁平结构体（自带全套访问器）。
// MLIR 把组建成 wrapper{g0_tiling_data}，所以要为 stage 的每个标量字段转发 set/get/get_addr。
// 只处理「恰好一个嵌套字段」的 wrapper：多 stage 时同名字段会在转发名上撞车，那档今天
// 没有 ATT 侧对应（见 `GenTilingData` 的命名条件），不转发。根（AutofuseTilingData）也
// 不走这里——单 plan 的根级转发由 `emitRootForwardingAccessors` 负责，两边同时开会重名。
LogicalResult emitGroupForwardingAccessors(mlir::ascend::kernel::TilingSchema structure,
                                           llvm::raw_ostream &accessorStream) {
  if (structure.getNameAttr().getValue() == "AutofuseTilingData" || structure.getNamesAttr().size() != 1UL) {
    return success();
  }
  auto wrapperFieldType = cast<TypeAttr>(structure.getTypesAttr()[0]).getValue();
  auto stage = mlir::ascend::kernel::getTilingSchema(wrapperFieldType);
  if (!stage) {
    return success();
  }
  llvm::StringRef member = cast<StringAttr>(structure.getNamesAttr()[0]).getValue();
  for (auto [name, fieldType] : llvm::zip(stage.getNamesAttr(), stage.getTypesAttr())) {
    auto integer = dyn_cast<IntegerType>(cast<TypeAttr>(fieldType).getValue());
    if (!integer) {
      continue;
    }
    llvm::StringRef field = cast<StringAttr>(name).getValue();
    std::string typeName = scalarTypeName(integer);
    accessorStream << "    inline void set_" << field << "(" << typeName << " value) { " << member << ".set_" << field
                   << "(value); }\n"
                   << "    inline " << typeName << " get_" << field << "() { return " << member << ".get_" << field
                   << "(); }\n"
                   << "    inline " << typeName << " *get_addr_" << field << "() { return " << member << ".get_addr_"
                   << field << "(); }\n";
  }
  return success();
}

// 这些类型名在生成的 TilingData 头文件里**不定义**：C++ 定义来自 CANN 官方头
// （`kernel_tiling.h` 里 `namespace AscendC::tiling` 的定义 + 顶层的
// `using AscendC::tiling::PadTiling;` 转发），本文件只引用——成员声明
// （`struct Stage { PadTiling pad_tiling_0; };`）不需要布局可见，include 打头就够。
// 若这里再发一份全局定义，与官方头进同一个 TU 就是重定义冲突。legacy 的生成头
// （`codegen_tiling_data.cpp:38` 的 macros_and_includes）同样只 include 不定义。
struct ExternalTilingType {
  llvm::StringRef typeName;
  llvm::StringRef providerInclude;
};
constexpr ExternalTilingType kExternalTilingTypes[] = {
    {"ConfusionTransposeTiling", "kernel_tiling/kernel_tiling.h"},
    {"PadTiling", "kernel_tiling/kernel_tiling.h"},
};

// 表上的名字返回提供定义的官方头；不在表上返回空。
llvm::StringRef externalProviderInclude(llvm::StringRef typeName) {
  for (const ExternalTilingType &entry : kExternalTilingTypes) {
    if (entry.typeName == typeName) {
      return entry.providerInclude;
    }
  }
  return {};
}

LogicalResult emitType(Type type, llvm::StringMap<std::string> &layouts, std::set<llvm::StringRef> &providerIncludes,
                       llvm::raw_ostream &output) {
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
    // 同名结构体必须同形；已生成过的不再生成第二遍。
    return success(existingLayout->second == layout);
  }
  layouts.try_emplace(name, std::move(layout));
  if (llvm::StringRef provider = externalProviderInclude(name); !provider.empty()) {
    // 外部类型：记下 include、跳过定义；同名同形的查重照走（上面已记账）。
    providerIncludes.insert(provider);
    return success();
  }
  for (Attribute fieldType : structure.getTypesAttr()) {
    if (failed(emitType(cast<TypeAttr>(fieldType).getValue(), layouts, providerIncludes, output))) {
      return failure();
    }
  }

  // 字段与访问器分两段生成：先铺完所有字段，再把访问器get/set统一放到结构体末尾。
  output << "struct " << name << " {\n";
  std::string accessors;
  llvm::raw_string_ostream accessorStream(accessors);
  if (failed(emitStructFields(structure, output, accessorStream)) ||
      failed(emitGroupForwardingAccessors(structure, accessorStream)) ||
      failed(emitRootForwardingAccessors(structure, accessorStream))) {
    return failure();
  }
  accessorStream.flush();
  // 全为嵌套结构体时没有访问器，不额外输出空行。
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
  // 先把结构体正文攒进缓冲：外部类型的 include 集要到走完 schema 才知道，而它必须
  // 打在文件头。std::set 给出确定的发射顺序。
  std::string body;
  llvm::raw_string_ostream bodyStream(body);
  llvm::StringMap<std::string> layouts;
  std::set<llvm::StringRef> providerIncludes;
  for (mlir::ascend::kernel::TilingSchema schema : schemas) {
    if (!schema || failed(emitType(schema, layouts, providerIncludes, bodyStream))) {
      return failure();
    }
  }
  bodyStream.flush();
  output << "#pragma once\n#include <cstdint>\n#include <vector>\n\n"
         << ascgen_utils::GenFinalTilingSelectionSnapshotSnippet() << "\n";
  for (llvm::StringRef provider : providerIncludes) {
    output << "#include \"" << provider << "\"\n";
  }
  output << "\n" << body;
  return success();
}

namespace {
LogicalResult emitReprFields(TilingSchema schema, llvm::StringRef access, unsigned depth, llvm::raw_ostream &output) {
  const std::string indent(depth * 2U, ' ');
  bool first = true;
  for (auto [nameAttr, typeAttr] : llvm::zip(schema.getNamesAttr(), schema.getTypesAttr())) {
    auto name = cast<StringAttr>(nameAttr).getValue();
    if (!mlir::afir::isValidCxxIdentifier(name)) {
      return failure();
    }
    Type type = cast<TypeAttr>(typeAttr).getValue();
    const std::string member = access.str() + name.str();
    output << "  repr << \"" << (first ? "" : ",") << "\\n" << indent << "." << name << " = ";
    if (auto nested = getTilingSchema(type)) {
      output << "{\";\n";
      if (failed(emitReprFields(nested, member + ".", depth + 1U, output))) {
        return failure();
      }
      output << "  repr << \"\\n" << indent << "}\";\n";
    } else {
      if (!isa<IntegerType>(type)) {
        return failure();
      }
      // Unary plus prints byte-sized integers as numbers, not characters.
      output << "\" << +" << member << ";\n";
    }
    first = false;
  }
  return success();
}
}  // namespace

LogicalResult mlir::ascend::kernel::emitTilingDataRepr(TilingSchema schema, llvm::raw_ostream &output) {
  if (!schema || !mlir::afir::isValidCxxIdentifier(schema.getNameAttr().getValue())) {
    return failure();
  }
  output << "#include <sstream>\n#include <string>\n#include \"autofuse_tiling_data.h\"\n"
         << "#if defined(__clang__)\n#pragma GCC diagnostic push\n"
         << "#pragma GCC diagnostic ignored \"-Wreturn-type-c-linkage\"\n#endif\n"
         << "extern \"C\" std::string GetTilingDataRepr(const " << schema.getNameAttr().getValue()
         << " *tiling_data) {\n"
         << "  if (tiling_data == nullptr) { return std::string(); }\n"
         << "  std::stringstream repr;\n  repr << \"" << schema.getNameAttr().getValue() << "{\";\n";
  if (failed(emitReprFields(schema, "tiling_data->", 1U, output))) {
    return failure();
  }
  output << "  repr << \"\\n}\";\n  return repr.str();\n}\n"
         << "#if defined(__clang__)\n#pragma GCC diagnostic pop\n#endif\n";
  return success();
}
