/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ir/ir_data_type_symbol_store.h"
#include "framework/common/debug/ge_log.h"
#include "common/ge_common/string_util.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
namespace af {
namespace {
graphStatus UpdateOpOuputListDtype(const OpDescPtr &op, const size_t &start, const size_t &end,
                                   const std::vector<DataType> &dtypes) {
  GE_ASSERT((end - start) == dtypes.size(), "Size mismatch when update %s output [%zu, %zu) with %zu dtypes",
            op->GetName().c_str(), start, end, dtypes.size());
  for (size_t i = start; i < end; i++) {
    auto desc = op->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(desc);
    GELOGI("Update op %s output %s:%zu with dtype %s", op->GetType().c_str(), op->GetName().c_str(), i,
           TypeUtils::DataTypeToSerialString(dtypes[i - start]).c_str());
    desc->SetDataType(dtypes[i - start]);
  }
  return GRAPH_SUCCESS;
}

graphStatus UpdateOpOuputDtype(const OpDescPtr &op, const size_t &start, const size_t &end, const DataType &dtype) {
  for (size_t i = start; i < end; i++) {
    auto desc = op->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(desc);
    GELOGI("Update op %s output %s:%zu with dtype %s", op->GetType().c_str(), op->GetName().c_str(), i,
           TypeUtils::DataTypeToSerialString(dtype).c_str());
    desc->SetDataType(dtype);
  }
  return GRAPH_SUCCESS;
}

const char *ToString(const IrOutputType &type) {
  if (type == kIrOutputRequired) {
    return "Required";
  }
  if (type == kIrOutputDynamic) {
    return "Dynamic";
  }
  return "Unknown";
}

std::string RemoveQuotes(const std::string &str) {
  std::string result = str;
  if (!result.empty() && result.front() == '\"') {
    result.erase(result.begin());
  }

  if (!result.empty() && result.back() == '\"') {
    result.pop_back();
  }
  return result;
}
}  // namespace

bool IRDataTypeSymbolStore::IsSupportSymbolicInferDtype() const {
  return !named_syms_.empty();
}
graphStatus IRDataTypeSymbolStore::InferDtype(const OpDescPtr &op) const {
  GE_ASSERT_NOTNULL(op);
  GELOGD("Start infer output dtype for op %s by syms", op->GetName().c_str());
  std::map<SymDtype *, TypeOrTypes> cached;  // 缓存每个Sym的求值结果，避免重复求值
  for (auto &item : named_syms_) {
    auto *sym = item.second;
    GE_WARN_ASSERT_GRAPH_SUCCESS(sym->Eval(*op, cached[sym]), "Failed eval sym %s of op %s", sym->DebugString().c_str(),
                                 op->GetType().c_str());
    GELOGD("Succeed eval and checking sym %s", sym->DebugString().c_str());
  }

  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
  GE_WARN_ASSERT_GRAPH_SUCCESS(af::GetIrOutputDescRange(op, ir_output_2_range));
  GE_WARN_ASSERT(ir_output_2_range.size() == op->GetIrOutputs().size(), "Failed to get output instance info of %s %s",
                 op->GetName().c_str(), op->GetType().c_str());

  GE_WARN_ASSERT(output_syms_.size() == op->GetIrOutputs().size(), "Op %s %s has %zu ir outputs, but %zu output syms",
                 op->GetName().c_str(), op->GetType().c_str(), op->GetIrOutputs().size(), output_syms_.size());
  // 对全部输出表达式进行求值，并更新到op上
  for (size_t i = 0U; i < output_syms_.size(); i++) {
    auto *sym = output_syms_[i];
    GE_ASSERT_NOTNULL(sym);
    if (sym->IsLegacy()) {
      GELOGW("Trying to infer legacy output %s(%s) of %s(%s)", sym->Id().c_str(), sym->DebugString().c_str(),
             op->GetName().c_str(), op->GetType().c_str());
      return GRAPH_FAILED;
    }

    TypeOrTypes type_or_types;
    auto cached_iter = cached.find(sym);
    if (cached_iter != cached.end()) {
      type_or_types = cached_iter->second;
    } else {
      GE_WARN_ASSERT_GRAPH_SUCCESS(sym->Eval(*op, type_or_types));
    }

    auto &output_range = ir_output_2_range[i];
    size_t start = output_range.first;
    size_t end = output_range.first + output_range.second;

    if (type_or_types.IsListType()) {  // ListType表示输出为动态输出，并且每个输出的类型可以不同
      GE_WARN_ASSERT(output_name_and_types_[i].second == kIrOutputDynamic,
                     "Op %s %s output %s bind to list-type sym %s", op->GetType().c_str(),
                     ToString(output_name_and_types_[i].second), output_name_and_types_[i].first.c_str(),
                     sym->Id().c_str());
      GE_WARN_ASSERT_GRAPH_SUCCESS(UpdateOpOuputListDtype(op, start, end, type_or_types.UnsafeGetTypes()));
    } else {
      GE_WARN_ASSERT_GRAPH_SUCCESS(UpdateOpOuputDtype(op, start, end, type_or_types.UnsafeGetType()));
    }
  }

  return GRAPH_SUCCESS;
}

// 创建一个命名Sym，用于符号出现早于IR输入的情况
SymDtype *IRDataTypeSymbolStore::GetOrCreateSymbol(const std::string &origin_sym_id) {
  std::string sym_id = RemoveQuotes(origin_sym_id);
  for (auto &sym : syms_) {
    if (sym->Id() == sym_id) {
      return sym.get();
    }
  }
  auto sym = MakeShared<SymDtype>(sym_id);
  GE_ASSERT_NOTNULL(sym, "Failed create symbol %s", sym_id.c_str());
  syms_.emplace_back(sym);
  return syms_.back().get();
}

SymDtype *IRDataTypeSymbolStore::SetInputSymbol(const std::string &ir_input, IrInputType input_type,
                                                const std::string &sym_id) {
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  sym->BindIrInput(ir_input, input_type, num_ir_inputs++);
  return sym;
}

// 调用DATATYPE声明时，绑定Sym的取值范围
SymDtype *IRDataTypeSymbolStore::DeclareSymbol(const std::string &sym_id, const TensorType &types) {
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  (void)named_syms_.emplace(sym_id, sym);
  sym->BindAllowedDtypes(types);
  return sym;
}

SymDtype *IRDataTypeSymbolStore::DeclareSymbol(const std::string &sym_id, const ListTensorType &types) {
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  (void)named_syms_.emplace(sym_id, sym);
  sym->BindAllowedDtypes(types);
  return sym;
}

SymDtype *IRDataTypeSymbolStore::DeclareSymbol(const std::string &sym_id, const Promote &types) {
  std::vector<SymDtype *> syms;
  for (auto &id : types.Syms()) {
    GE_ASSERT(id != sym_id, "Trying to promote symbol %s with itself", sym_id.c_str());
    auto *sym = GetOrCreateSymbol(id);
    GE_ASSERT_NOTNULL(sym);
    syms.emplace_back(sym);
  }

  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  (void)named_syms_.emplace(sym_id, sym);
  sym->BindExpression(MakeShared<PromotionSymDtypeExpression>(syms));
  return sym;
}

// 创建输出的Symbol表达式，用于支持类型推导
SymDtype *IRDataTypeSymbolStore::SetOutputSymbol(const std::string &ir_output, IrOutputType output_type,
                                                 const std::string &sym_id) {
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  output_syms_.emplace_back(sym);
  output_name_and_types_.emplace_back(ir_output, output_type);
  return sym;
}

graphStatus IRDataTypeSymbolStore::GetPromoteIrInputList(std::vector<std::vector<size_t>> &promote_index_list) {
  for (const auto &named_sym : named_syms_) {
    GE_ASSERT_NOTNULL(named_sym.second);
    if (named_sym.second->Type() == ExpressionType::kPromote) {
      auto ir_input_indexes = named_sym.second->GetIrInputIndexes();
      promote_index_list.push_back(ir_input_indexes);
    }
  }
  return ge::GRAPH_SUCCESS;
}

SymDtype *IRDataTypeSymbolStore::DeclareSymbol(const string &sym_id, const OrderedTensorTypeList &types) {
  GELOGD("Bind symbol %s with ordered list-dtypes %s", sym_id.c_str(), types.ToString().c_str());
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  GE_ASSERT_TRUE(named_syms_.emplace(sym_id, sym).second, "Symbol %s has been declared", sym_id.c_str());
  sym->BindAllowedOrderedDtypes(types);
  return sym;
}
bool IRDataTypeSymbolStore::IsSupportOrderedSymbolicInferDtype() const {
  if (syms_.empty()) {
    return false;
  }

  size_t base_size = 0;
  bool found = false;

  for (const auto &sym_dtype : syms_) {
    if (sym_dtype != nullptr && sym_dtype->IsOrderedList()) {
      size_t current_size = sym_dtype->GetOrderedTensorTypeList().GetOrderedDtypes().size();
      if (current_size > 0) {
        base_size = current_size;
        found = true;
        break;
      }
    }
  }

  if (!found) {
    return false;
  }

  return std::all_of(syms_.begin(), syms_.end(), [&base_size](const std::shared_ptr<SymDtype> &sym_dtype) {
    if (sym_dtype == nullptr) {
      return false;
    }
    if (!sym_dtype->IsOrderedList()) {
      return false;
    }
    size_t current_size = sym_dtype->GetOrderedTensorTypeList().GetOrderedDtypes().size();
    return current_size == base_size;
  });
}

}  // namespace af
