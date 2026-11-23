/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_ATTR_GROUP_ATTR_GROUP_SHAPE_ENV_H
#define INC_GRAPH_ATTR_GROUP_ATTR_GROUP_SHAPE_ENV_H

#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <set>
#include "attribute_group/af_attr_group_base.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "common/checker.h"
#include "graph/detail/attributes_holder.h"

namespace af {

ShapeEnvAttr *GetCurShapeEnvContext();
void SetCurShapeEnvContext(ShapeEnvAttr *shape_env);

class Source {
 public:
  virtual ~Source() = default;
  // 目的是用于codegen GetAllSym时生成符号来源
  virtual std::string GetSourceStr() const = 0;
  // 目的是用于codegen infershape\tiling时生成符号索引
  virtual std::string GetGlobalIndexStr() const;

  virtual size_t GetGlobalIndex() const {
    return global_index_;
  }
  void SetGlobalIndex(size_t global_index) {
    global_index_ = global_index;
  };
  // todoo: 兼容上库，待删除
  virtual int32_t GetInputDataIdx() const {
    return std::numeric_limits<int32_t>::max();
  };
  virtual size_t GetDimIdx() const {
    return std::numeric_limits<size_t>::max();
  }

 private:
  size_t global_index_{std::numeric_limits<size_t>::max()};
};
using SourcePtr = std::shared_ptr<Source>;

struct ShapeEnvExpressionHash {
  uint64_t operator()(const Expression &e) const {
    return e.Hash();
  }
};
struct ShapeEnvExpressionKeyEq {
  bool operator()(const Expression &lhs, const Expression &rhs) const {
    return lhs == rhs;
  }
};
struct SymbolCheckInfo {
  Expression expr;
  std::string file;
  int64_t line{};
  std::string dfx_info;
  explicit SymbolCheckInfo(const Expression &in_expr, const std::string &in_file = "", const int64_t in_line = -1,
                           const std::string &dfx = "")
      : expr(in_expr), file(in_file), line(in_line), dfx_info(dfx) {}
  SymbolCheckInfo() = default;
  bool operator==(const SymbolCheckInfo &other) const {
    return this->expr == other.expr;
  }
};

struct SymbolCheckInfoKeyLess {
  bool operator()(const SymbolCheckInfo &a, const SymbolCheckInfo &b) const {
    // 只比较expr， file与line暂不比较
    return a.expr.Compare(b.expr) < 0;
  }
};

// 配置符号的生成方式
// dynamic：不管hint值是否相等，均生成新符号
// duck：当hint值相同时，则不生成新符号，使用之前生成过的符号
// static：根据hint值生成符号，同时添加一个Assert（sym == hint）的guard
enum class DynamicMode { kDynamic = 0, kDuck = 1, kStatic = 2, kEnd = 3 };

struct ShapeEnvSetting {
  bool specialize_zero_one{false};
  DynamicMode dynamic_mode{DynamicMode::kDynamic};
  ShapeEnvSetting() = default;
  ShapeEnvSetting(const bool in_specialize_zero_one, const DynamicMode &in_dynamic_mode)
      : specialize_zero_one(in_specialize_zero_one), dynamic_mode(in_dynamic_mode) {}
};

struct Replacement {
  Expression replace_expr;
  int32_t rank;
  bool has_replace;
  Replacement(const Expression &a, const int32_t in_rank, bool in_has_replace = false)
      : replace_expr(a), rank(in_rank), has_replace(in_has_replace) {}
  Replacement() : rank(0), has_replace(false) {}
  bool operator<=(const Replacement &other);
};

class ShapeEnvAttr : public AttrGroupsBase {
 public:
  ShapeEnvAttr() = default;
  ~ShapeEnvAttr() override = default;
  explicit ShapeEnvAttr(const ShapeEnvSetting &shape_env_setting) : shape_env_setting_(shape_env_setting) {}

  ShapeEnvAttr(const ShapeEnvAttr &other);
  ShapeEnvAttr &operator=(const ShapeEnvAttr &other);
  graphStatus Serialize(proto::AttrGroupDef &attr_group_def) override;
  graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) override;
  // 只支持int32，uint32, int64, uint64
  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, Symbol>::type CreateSymbol(T hint, const SourcePtr &source) {
    const std::lock_guard<std::mutex> lk(mutex_);
    auto hint_int64 = static_cast<int64_t>(hint);
    GE_ASSERT_TRUE((shape_env_setting_.dynamic_mode >= DynamicMode::kDynamic) &&
                       (shape_env_setting_.dynamic_mode < DynamicMode::kEnd),
                   "Invalid dynamic mode: %d, should be in [%d, %d), create symbol failed",
                   static_cast<int32_t>(shape_env_setting_.dynamic_mode), static_cast<int32_t>(DynamicMode::kDynamic),
                   static_cast<int32_t>(DynamicMode::kEnd));
    if (shape_env_setting_.specialize_zero_one && ((hint_int64 == 0) || (hint_int64 == 1))) {
      GELOGI("Create symbol %d in specialize_zero_one mode, source: %s", hint_int64, source->GetSourceStr().c_str());
      return Symbol(hint_int64);
    }
    if (shape_env_setting_.dynamic_mode != DynamicMode::kDynamic) {
      // 非动态模式，hint值相同使用同一个符号
      const auto &iter = value_to_symbol_.find(hint_int64);
      if (iter != value_to_symbol_.end()) {
        GE_ASSERT_TRUE(!iter->second.empty());
        return Symbol(iter->second.front().Str().get());
      }
    }
    auto global_index = unique_sym_id_++;
    GE_ASSERT_TRUE(global_index < std::numeric_limits<uint64_t>::max(),
                   "unique_sym_id_ is " PRIu64 ". will reach the maximum value of uint64.", unique_sym_id_);
    source->SetGlobalIndex(global_index);
    const std::string sym_name = "s" + std::to_string(global_index);
    auto sym = Symbol(sym_name.c_str());
    symbol_to_source_.emplace(sym, source);
    symbol_to_value_.emplace(sym, hint_int64);
    const auto iter = value_to_symbol_.find(hint_int64);
    if (iter != value_to_symbol_.end()) {
      iter->second.emplace_back(sym);
    } else {
      std::vector<Expression> syms = {sym};
      value_to_symbol_.emplace(hint_int64, syms);
    }
    // 静态场景需要增加一个s == hint的Assert信息
    if (shape_env_setting_.dynamic_mode == DynamicMode::kStatic) {
      ASSERT_SYMBOL_EQ(sym, Symbol(hint_int64));
    }
    return sym;
  }
  std::vector<std::pair<Expression, SourcePtr>> GetAllSym2Src();

  Expression Simplify(const Expression &expr);
  void SimplifySymbolCheckInfo();
  Expression EvaluateExpr(const Expression &expr);
  graphStatus AppendReplacement(const Expression &target, const Expression &replacement);
  graphStatus AppendSymbolAssertInfo(const Expression &expr, const std::string &file = "", const int64_t line = 0L);
  graphStatus AppendSymbolCheckInfo(const Expression &expr, const std::string &file = "", const int64_t line = 0L);
  const std::vector<SymbolCheckInfo> GetAllSymbolCheckInfos() const;
  const std::vector<SymbolCheckInfo> GetAllSymbolAssertInfos() const;
  bool HasSymbolCheckInfo(const Expression &e) const;
  bool HasSymbolAssertInfo(const Expression &e) const;
  TriBool HasSymbolInfo(const Expression &expr) const;
  std::unique_ptr<AttrGroupsBase> CloneAf() override;
  void SetGuardDfxContextInfo(const std::string &guard_dfx_info) const;
  void ClearGuardDfxContextInfo() const;
  void ClearSymbolValueInfo() {
    symbol_to_value_.clear();
    value_to_symbol_.clear();
  }

 private:
  void SimplifySymbolCheckInfo(std::set<SymbolCheckInfo, SymbolCheckInfoKeyLess> &symbol_check_infos) const;
  void AppendInitReplacement(const Expression &expr);
  Expression FindReplacements(const Expression &expr);
  graphStatus MergeReplacement(const Expression &expr1, const Expression &expr2);
  graphStatus FindRootExpr(const Expression &expr, Expression &root_expr) const;
  graphStatus SerializeSymbolCheckInfos(proto::ShapeEnvAttrGroupsDef *shape_env_group);
  graphStatus MergePath();
  graphStatus SerializeSymbolInfo(proto::ShapeEnvAttrGroupsDef *shape_env_group);
  graphStatus DeserializeSymbolInfo(const proto::ShapeEnvAttrGroupsDef &shape_env_group);
  graphStatus DeserializeSymbolCheckInfos(const proto::ShapeEnvAttrGroupsDef &shape_env_group);
  std::string GetGuardDfxContextInfo() const;
  bool CheckReplacementCycle(const Expression &expr1, const Expression &expr2) const;
  using UMapExprReplacement =
      std::unordered_map<Expression, Replacement, ShapeEnvExpressionHash, ShapeEnvExpressionKeyEq>;
  using UMapExprInt = std::unordered_map<Expression, int64_t, ShapeEnvExpressionHash, ShapeEnvExpressionKeyEq>;
  using UMapExprSource = std::unordered_map<Expression, SourcePtr, ShapeEnvExpressionHash, ShapeEnvExpressionKeyEq>;
  UMapExprReplacement replacements_;
  UMapExprInt symbol_to_value_;
  UMapExprSource symbol_to_source_;
  std::map<int64_t, std::vector<Expression>> value_to_symbol_;
  std::set<SymbolCheckInfo, SymbolCheckInfoKeyLess> symbol_check_infos_;
  std::set<SymbolCheckInfo, SymbolCheckInfoKeyLess> symbol_assert_infos_;
  ShapeEnvSetting shape_env_setting_;
  size_t unique_sym_id_{0U};
  std::mutex mutex_;
  thread_local static std::string guard_dfx_info_;
};

}  // namespace af

#endif  // INC_GRAPH_ATTR_GROUP_ATTR_GROUP_SHAPE_ENV_H
