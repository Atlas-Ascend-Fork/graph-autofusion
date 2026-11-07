/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expression_impl.h"

#include <type_traits>
#include <queue>
#include <typeinfo>
#include <symengine/assumptions.h>
#include <symengine/functions.h>
#include <symengine/simplify.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
#include <symengine/visitor.h>
#include <symengine/logic.h>
#include <symengine/parser.h>

#include "expr_print_manager.h"
#include "const_values.h"
#include "expr_parser.h"
#include "common/checker.h"

namespace af {
namespace {
constexpr const char_t *kInvalidName = "INVALID_NAME";

inline std::string sym_str(const SymEngineExprPtr &expr) {
  GE_ASSERT_TRUE(!expr.is_null());
  return expr->__str__();
}

bool IsNegative(const SymEngineExprPtr &expr) {
  GE_ASSERT_TRUE(!expr.is_null());
  GELOGD("Negative check, %s type is %s", sym_str(expr).c_str(), type_code_name(expr->get_type_code()).c_str());
  if (SymEngine::is_a_Number(*expr.get())) {
    const auto *s = SymEngine::down_cast<const SymEngine::Number *>(expr.get());
    return s->is_negative();
  }

  if (SymEngine::is_a<SymEngine::Mul>(*expr.get())) {
    const auto *s = SymEngine::down_cast<const SymEngine::Mul *>(expr.get());
    return s->get_coef()->is_negative();
  }
  return false;
}

SymEngine::integer_class ExtractCoeff(const SymEngineExprPtr &expr) {
  GE_ASSERT_TRUE(!expr.is_null());
  GELOGD("expr [%s] type is %s", sym_str(expr).c_str(), type_code_name(expr->get_type_code()).c_str());
  if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
    return SymEngine::rcp_static_cast<const SymEngine::Integer>(expr)->as_integer_class();
  } else if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
    for (const auto &arg : expr->get_args()) {
      GELOGD("Mul expr [%s] type is %s", sym_str(arg).c_str(), type_code_name(arg->get_type_code()).c_str());
      if (SymEngine::is_a<SymEngine::Integer>(*arg)) {
        return SymEngine::rcp_static_cast<const SymEngine::Integer>(arg)->as_integer_class();
      }
    }
  }
  return SymEngine::integer_class(1);
}

SymEngine::integer_class ComputeGCD(const std::vector<SymEngine::integer_class> &coeff_vec) {
  GE_ASSERT_TRUE(!coeff_vec.empty());
  SymEngine::integer_class gcd = coeff_vec[0];
  for (size_t i = 1; i < coeff_vec.size(); ++i) {
    SymEngine::mp_gcd(gcd, gcd, coeff_vec[i]);
  }
  return gcd;
}

SymEngine::integer_class ComputeExprGCD(const SymEngineExprPtr &expr) {
  GE_ASSERT_TRUE(SymEngine::is_a<SymEngine::Add>(*expr));
  std::vector<SymEngine::integer_class> coeff_vec;
  for (const auto &arg : expr->get_args()) {
    auto coeff = ExtractCoeff(arg);
    coeff_vec.push_back(coeff);
    GELOGD("expr [%s], coeff = %lu", sym_str(arg).c_str(), SymEngine::mp_get_ui(coeff));
  }
  return ComputeGCD(coeff_vec);
}

SymEngine::map_basic_basic CreateMulDict(const SymEngine::vec_basic &args,
                                         SymEngine::RCP<const SymEngine::Number> &coeff) {
  SymEngine::map_basic_basic dict;
  for (const auto &arg : args) {
    GELOGD("arg [%s] type %s", sym_str(arg).c_str(), type_code_name(arg->get_type_code()).c_str());
    auto it = dict.find(arg);
    if (it != dict.end()) {
      dict[arg] = add(it->second, SymEngine::one);
    } else {
      if (IsNegative(arg)) {
        coeff = SymEngine::minus_one;
      }
      if (SymEngine::is_a<SymEngine::Integer>(*arg)) {
        auto in = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg);
        if (in->is_one() || in->is_minus_one()) {
          continue;
        }
      }
      dict[arg] = SymEngine::one;
    }
  }
  return dict;
}

std::vector<SymEngineExprPtr> TrDivision(const SymEngineExprPtr &lhs, const SymEngineExprPtr &rhs) {
  std::vector<SymEngineExprPtr> vec;
  if (SymEngine::is_a<SymEngine::Mul>(*lhs) && SymEngine::is_a<SymEngine::Mul>(*rhs)) {
    SymEngineExprPtr ratio = SymEngine::div(lhs, rhs);
    if (SymEngine::is_a<SymEngine::Symbol>(*ratio) || SymEngine::eq(*ratio, *SymEngine::one)) {
      vec.push_back(ratio);
      vec.push_back(SymEngine::one);
      return vec;
    }

    ratio = SymEngine::div(rhs, lhs);
    if (SymEngine::is_a<SymEngine::Symbol>(*ratio) || SymEngine::eq(*ratio, *SymEngine::one)) {
      vec.push_back(SymEngine::one);
      vec.push_back(ratio);
      return vec;
    }
  }
  vec.push_back(lhs);
  vec.push_back(rhs);
  return vec;
}

SymEngineExprPtr DivByFactor(const SymEngineExprPtr &x, const SymEngine::integer_class &factor) {
  GE_ASSERT_TRUE(!x.is_null());
  GELOGD("DivByFactor [%s], factor = %lu", sym_str(x).c_str(), SymEngine::mp_get_ui(factor));
  if (SymEngine::is_a<SymEngine::Integer>(*x)) {
    SymEngine::integer_class x_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(x)->as_integer_class();
    SymEngine::integer_class result;
    SymEngine::mp_divexact(result, x_val, factor);
    return SymEngine::integer(result);
  } else if (SymEngine::is_a<SymEngine::Mul>(*x)) {
    const auto &args = x->get_args();
    GE_ASSERT_TRUE(!args.empty());
    SymEngine::integer_class total_coeff(1);
    SymEngine::vec_basic remaining_args;
    for (const auto &arg : args) {
      if (SymEngine::is_a<SymEngine::Integer>(*arg)) {
        total_coeff *= SymEngine::rcp_static_cast<const SymEngine::Integer>(arg)->as_integer_class();
      } else {
        remaining_args.push_back(arg);
      }
    }

    SymEngine::integer_class new_coeff;
    SymEngine::mp_divexact(new_coeff, total_coeff, factor);
    SymEngine::vec_basic new_args;
    GELOGD("total_coeff=%lu, factor=%lu, new_coeff=%lu", SymEngine::mp_get_ui(total_coeff),
           SymEngine::mp_get_ui(factor), SymEngine::mp_get_ui(new_coeff));
    if (new_coeff != 1) {
      new_args.push_back(SymEngine::integer(new_coeff));
    }
    new_args.insert(new_args.end(), remaining_args.begin(), remaining_args.end());
    SymEngine::RCP<const SymEngine::Number> coeff = SymEngine::one;
    auto dict = CreateMulDict(new_args, coeff);
    return SymEngine::Mul::from_dict(coeff, std::move(dict));
  } else {
    GELOGW("unsupported type %s", type_code_name(x->get_type_code()).c_str());
    return x;
  }
}
}  // namespace

ExpressionImplPtr ExpressionImpl::CreateExpressionImpl(const std::string &name) {
  return ComGraphMakeUnique<ExpressionImpl>(name);
}

ExpressionImpl::~ExpressionImpl() {}

std::string ExpressionImpl::Str(const StrType type) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  if (type == StrType::kStrCpp) {
    if (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)) {
      const auto &x = SymEngine::down_cast<const SymEngine::Rational &>(*sym_expr_);
      auto dens = x.get_den();
      auto nums = x.get_num();
      GE_ASSERT_TRUE(!dens.is_null());
      GE_ASSERT_TRUE(!nums.is_null());
      return "Rational(" + nums->__str__() + " , " + dens->__str__() + ")";
    }
  }
  if (((GetExprType() == ExprType::kExprOperation) || (GetExprType() == ExprType::kExprOperationBoolean)) &&
      (GetOpType() != OperationType::kOpNone)) {
    auto printer = ExprManager::GetInstance().GetPrinter(GetOpType());
    GE_ASSERT_NOTNULL(printer);
    return printer(sym_expr_->get_args(), type);
  }
  return sym_expr_->__str__();
}

ExpressionImplPtr ExpressionImpl::Parse(const std::string &expr_str) {
  Scanner scanner(expr_str);
  ExprParser expr_parser(scanner);
  auto ret = expr_parser.ParserExpression();
  GE_WARN_ASSERT(ret != nullptr, "Parse expression %s abnormal.", expr_str.c_str());
  return ret;
}

namespace {
// 完整解析输入串：解析失败或输入串存在未消费的残留 token（如 "a(s0)" 被截断为 "a"）均视为失败
ExpressionImplPtr ParseWholeExpression(const std::string &expr_str) {
  Scanner scanner(expr_str);
  ExprParser expr_parser(scanner);
  auto ret = expr_parser.ParserExpression();
  if (ret == nullptr || !expr_parser.IsAtEnd()) {
    return nullptr;
  }
  return ret;
}
}  // namespace

ExpressionImplPtr ExpressionImpl::Deserialize(const std::string &expr_str) {
  auto ret = ParseWholeExpression(expr_str);
  if (ret == nullptr) {
    GELOGE(FAILED, "Parse expression str %s failed or incomplete.", expr_str.c_str());
    return nullptr;
  }
  const std::string reserialized = ret->Str();
  if (reserialized == expr_str) {
    return ret;
  }
  // SymEngine 表达式树是规范形：语义相同的表达式解析后得到相同的树
  // （Add 合并同类项、交换律参数按规范化顺序排列）。因此将打印结果再次解析
  // 并比较两棵树，即可区分"仅打印形式不同"与"语义不等价"。
  // 背景：SymEngine 规范打印会重排负系数 Add 项、交换交换律算子参数，合法的
  // 序列化串（如 guard）曾因字节不等被拒，返回 nullptr 后上层 guard 集合
  // 比较器解引用空指针断言崩溃。
  const auto reparsed = ParseWholeExpression(reserialized);
  if (reparsed != nullptr && ret->Compare(*reparsed) == 0) {
    GELOGW("Parse expression str %s abnormal, reserialized to %s, they are semantically equal, accepted.",
           expr_str.c_str(), reserialized.c_str());
    return ret;
  }
  GELOGE(FAILED, "Parse expression str %s abnormal, reserialized to %s, they are not semantically equal.",
         expr_str.c_str(), reserialized.c_str());
  return nullptr;
}

ExpressionImplPtr ExpressionImpl::Replace(const std::map<ExpressionImpl *, ExpressionImpl *> &replace_vars) const {
  SymEngine::map_basic_basic sym_replace_vars;
  for (const auto &sym_expr_impl_ptr_item : replace_vars) {
    GE_ASSERT_NOTNULL(sym_expr_impl_ptr_item.first);
    GE_ASSERT_NOTNULL(sym_expr_impl_ptr_item.second);
    GE_ASSERT_TRUE(!sym_expr_impl_ptr_item.first->sym_expr_.is_null());
    GE_ASSERT_TRUE(!sym_expr_impl_ptr_item.second->sym_expr_.is_null());
    sym_replace_vars.emplace(sym_expr_impl_ptr_item.first->sym_expr_, sym_expr_impl_ptr_item.second->sym_expr_);
  }
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  SymEngineExprPtr replaced_expr = sym_expr_->xreplace(sym_replace_vars);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(replaced_expr);
}

ExpressionImplPtr ExpressionImpl::Subs(const std::map<ExpressionImpl *, ExpressionImpl *> &subs_vars) const {
  SymEngine::map_basic_basic sym_replace_vars;
  for (const auto &sym_expr_impl_ptr_item : subs_vars) {
    GE_ASSERT_NOTNULL(sym_expr_impl_ptr_item.first);
    GE_ASSERT_NOTNULL(sym_expr_impl_ptr_item.second);
    GE_ASSERT_TRUE(!sym_expr_impl_ptr_item.first->sym_expr_.is_null());
    GE_ASSERT_TRUE(!sym_expr_impl_ptr_item.second->sym_expr_.is_null());
    sym_replace_vars.emplace(sym_expr_impl_ptr_item.first->sym_expr_, sym_expr_impl_ptr_item.second->sym_expr_);
  }
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  SymEngineExprPtr subs_expr = sym_expr_->subs(sym_replace_vars);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(subs_expr);
}

void ExpressionImpl::AsNumerDenom(ExpressionImplPtr &numer, ExpressionImplPtr &denom) const {
  SymEngineExprPtr symengine_numer;
  SymEngineExprPtr symengine_denom;
  as_numer_denom(sym_expr_, outArg(symengine_numer), outArg(symengine_denom));
  numer = CreateExpressionImpl<const SymEngineExprPtr &>(symengine_numer);
  denom = CreateExpressionImpl<const SymEngineExprPtr &>(symengine_denom);
}

ExpressionImplPtr ExpressionImpl::Simplify() const {
  SymEngineExprPtr expanded_expr = SymEngine::expand(sym_expr_);
  SymEngineExprPtr simplified_expr = SymEngine::simplify(expanded_expr);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(simplified_expr);
}

bool ExpressionImpl::ContainVar(const ExpressionImpl *e) const {
  GE_ASSERT_NOTNULL(e);
  GE_ASSERT_TRUE(!e->sym_expr_.is_null());
  if (!(e->sym_expr_->get_args().empty())) {
    return false;
  }
  for (const auto &arg : FreeSymbols()) {
    GE_ASSERT_NOTNULL(arg);
    GE_ASSERT_NOTNULL(e);
    GE_ASSERT_TRUE(!arg->sym_expr_.is_null());
    GE_ASSERT_TRUE(!e->sym_expr_.is_null());
    if (SymEngine::eq(*arg->sym_expr_, *(e->sym_expr_))) {
      return true;
    }
  }
  return false;
}

std::vector<ExpressionImplPtr> ExpressionImpl::FreeSymbols() const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  auto free_symbols = SymEngine::free_symbols(*sym_expr_);
  std::vector<ExpressionImplPtr> ret;
  for (const auto &sym_arg : free_symbols) {
    auto expr = ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_arg);
    ret.emplace_back(std::move(expr));
  }
  return ret;
}

bool ExpressionImpl::operator==(const ExpressionImpl &e) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  GE_ASSERT_TRUE(!e.sym_expr_.is_null());
  return SymEngine::eq(*sym_expr_, *e.sym_expr_);
}

ExpressionImplPtr ExpressionImpl::CanonicalizeBoolExpr() {
  GE_ASSERT_NOTNULL(sym_expr_.get());
  GELOGI("EXPR(before) [%s]", sym_str(sym_expr_).c_str());
  OperationType type = GetOpType();
  std::unordered_map<OperationType, RelationalFunc> op_func_map = {
      {OperationType::kOpEq, [](const auto &a, const auto &b) { return SymEngine::Eq(a, b); }},
      {OperationType::kOpNe, [](const auto &a, const auto &b) { return SymEngine::Ne(a, b); }},
      {OperationType::kOpLt, [](const auto &a, const auto &b) { return SymEngine::Lt(a, b); }},
      {OperationType::kOpLe, [](const auto &a, const auto &b) { return SymEngine::Le(a, b); }}};

  if (op_func_map.find(type) == op_func_map.end()) {
    GELOGI("EXPR(after) [%s]", sym_str(sym_expr_).c_str());
    return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr_);
  }

  SymEngine::vec_basic args = sym_expr_.get()->get_args();
  GE_ASSERT_TRUE(args.size() == kSizeTwo);
  SymEngineExprPtr rhs = SymEngine::sub(args[1], args[0]);
  SymEngineExprPtr lhs = SymEngine::zero;
  GELOGI("step1 rhs = [%s], lhs = [%s]", sym_str(rhs).c_str(), sym_str(lhs).c_str());

  rhs = SymEngine::expand(rhs);
  // 有理系数通分：分母为正整数且非 1 时用分子替换差值（分母恒正，丢弃不改变
  // 差值符号，Eq/Ne/Lt/Le 语义均保持）；使 (1/2)*s0 == 680 与 s0 == 1360 归一到
  // 同一规范形，跨形态 guard 匹配不依赖查询侧与记录侧表达式字面同构
  SymEngineExprPtr numer;
  SymEngineExprPtr denom;
  as_numer_denom(rhs, outArg(numer), outArg(denom));
  if (SymEngine::is_a<SymEngine::Integer>(*denom)) {
    const auto &denom_int = SymEngine::down_cast<const SymEngine::Integer &>(*denom);
    if (denom_int.is_positive() && !(denom_int.is_one())) {
      rhs = numer;
    }
  }
  if (SymEngine::is_a<SymEngine::Add>(*rhs)) {
    SymEngine::integer_class gcd = ComputeExprGCD(rhs);
    GELOGD("[%s], gcd = %lu", sym_str(rhs).c_str(), SymEngine::mp_get_ui(gcd));
    if (SymEngine::mp_get_ui(gcd) > 1) {
      SymEngine::vec_basic div_gcd_vec;
      for (const auto &arg : rhs->get_args()) {
        auto expr_div = DivByFactor(arg, gcd);
        div_gcd_vec.push_back(expr_div);
        GELOGD("DivByFactor [%s] -> [%s]", sym_str(arg).c_str(), sym_str(expr_div).c_str());
      }
      rhs = SymEngine::add(div_gcd_vec);
    }
  }
  GELOGI("step2 rhs = [%s], lhs = [%s]", sym_str(rhs).c_str(), sym_str(lhs).c_str());

  if (SymEngine::is_a<SymEngine::Add>(*rhs)) {
    SymEngine::vec_basic pos_vec;
    SymEngine::vec_basic neg_vec;
    for (const auto &arg : rhs->get_args()) {
      if (IsNegative(arg)) {
        GELOGD("negative push [%s]", sym_str(arg).c_str());
        neg_vec.push_back(SymEngine::sub(SymEngine::zero, arg));
      } else {
        pos_vec.push_back(arg);
      }
    }
    rhs = SymEngine::add(pos_vec);
    lhs = SymEngine::add(neg_vec);
  }
  if (IsNegative(rhs) && SymEngine::is_number_and_zero(*lhs)) {
    lhs = SymEngine::sub(SymEngine::zero, rhs);
    rhs = SymEngine::zero;
  }
  GELOGI("step3 rhs = [%s], lhs = [%s]", sym_str(rhs).c_str(), sym_str(lhs).c_str());

  auto vec = TrDivision(lhs, rhs);
  GELOGI("step4 rhs = [%s], lhs = [%s]", sym_str(vec[1]).c_str(), sym_str(vec[0]).c_str());

  auto expr_new = op_func_map[type](vec[0], vec[1]);
  GELOGI("EXPR(after) [%s] (canonicalized)", sym_str(expr_new).c_str());
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(expr_new);
}

std::vector<ExpressionImplPtr> ExpressionImpl::GetArgs() {
  std::vector<ExpressionImplPtr> args;
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  for (auto &arg : sym_expr_.get()->get_args()) {
    args.emplace_back(ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(arg));
  }
  return args;
}

double ExpressionImpl::GetIntegerResult(const SymEngine::Integer &integer_expr) const {
  if (integer_expr.is_zero()) {
    return 0;
  } else if (integer_expr.is_positive()) {
    return static_cast<double>(integer_expr.as_uint());
  }
  return static_cast<double>(integer_expr.as_int());
}

bool ExpressionImpl::GetResult(double &result) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  if (SymEngine::is_a<SymEngine::Integer>(*sym_expr_)) {
    const auto &integer_expr = SymEngine::down_cast<const SymEngine::Integer &>(*sym_expr_);
    result = GetIntegerResult(integer_expr);
    return true;
  }
  if (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)) {
    const auto &rational_expr = SymEngine::down_cast<const SymEngine::Rational &>(*sym_expr_);
    result = GetIntegerResult(*(rational_expr.get_num())) / GetIntegerResult(*(rational_expr.get_den()));
    return true;
  }
  if (SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) {
    const auto &real_double_expr = SymEngine::down_cast<const SymEngine::RealDouble &>(*sym_expr_);
    result = real_double_expr.as_double();
    return true;
  }
  return false;
}

uint64_t ExpressionImpl::Hash() const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  return sym_expr_->hash();
}

int64_t ExpressionImpl::Compare(const ExpressionImpl &e) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  GE_ASSERT_TRUE(!e.sym_expr_.is_null());
  return sym_expr_->__cmp__(*e.sym_expr_);
}

bool ExpressionImpl::IsVariableExpr() const {
  return GetExprType() == ExprType::kExprVariable;
}

bool ExpressionImpl::IsBooleanExpr() const {
  return (GetExprType() == ExprType::kExprOperationBoolean) || (GetExprType() == ExprType::kExprConstantBoolean);
}

bool ExpressionImpl::GetConstValue(uint32_t &value) const {
  uint64_t result = 0UL;
  GE_ASSERT_TRUE(GetConstValue(result));
  value = static_cast<uint32_t>(result);
  return true;
}

bool ExpressionImpl::GetConstValue(uint64_t &value) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  // 无符号整数类型
  GE_ASSERT_TRUE(SymEngine::is_a<SymEngine::Integer>(*sym_expr_),
                 "Cannot get const uint value from a expression: %s not Integer.", Str().c_str());
  const auto &integer_expr = SymEngine::down_cast<const SymEngine::Integer &>(*sym_expr_);
  value = integer_expr.as_uint();
  return true;
}

bool ExpressionImpl::GetConstValue(int32_t &value) const {
  int64_t result = 0L;
  GE_ASSERT_TRUE(GetConstValue(result));
  value = static_cast<int32_t>(result);
  return true;
}

bool ExpressionImpl::GetConstValue(int64_t &value) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  // 整数类型
  GE_ASSERT_TRUE(SymEngine::is_a<SymEngine::Integer>(*sym_expr_),
                 "Cannot get const int value from a expression: %s not Integer.", Str().c_str());
  const auto &integer_expr = SymEngine::down_cast<const SymEngine::Integer &>(*sym_expr_);
  value = integer_expr.as_int();
  return true;
}

bool ExpressionImpl::GetConstValue(bool &value) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  // bool类型
  GE_ASSERT_TRUE(SymEngine::is_a<SymEngine::BooleanAtom>(*sym_expr_),
                 "Cannot get const bool value from a expression: %s not BooleanAtom.", Str().c_str());
  const auto &bool_expr = SymEngine::down_cast<const SymEngine::BooleanAtom &>(*sym_expr_);
  value = bool_expr.get_val();
  return true;
}

bool ExpressionImpl::GetConstValue(float &value) const {
  double result = 0L;
  GE_ASSERT_TRUE(GetConstValue(result));
  value = static_cast<float>(result);
  return true;
}

bool ExpressionImpl::GetConstValue(double &value) const {
  GE_ASSERT_TRUE(!sym_expr_.is_null());
  GE_ASSERT_TRUE(
      (SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) || (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)),
      "Cannot get const float value from a expression: %s not RealDouble or Rational.", Str().c_str());
  if (SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) {
    const auto &real_double_expr = SymEngine::down_cast<const SymEngine::RealDouble &>(*sym_expr_);
    value = real_double_expr.as_double();
  } else {
    // 分数
    const auto &rational_expr = SymEngine::down_cast<const SymEngine::Rational &>(*sym_expr_);
    value = GetIntegerResult(*(rational_expr.get_num())) / GetIntegerResult(*(rational_expr.get_den()));
  }
  return true;
}

OperationType ExpressionImpl::GetOpType() const {
  if (sym_expr_.is_null()) {
    GELOGE(ge::PARAM_INVALID, "sym_expr_ is null.");
    return OperationType::kOpNone;
  }
  if (SymEngine::is_a<SymEngine::Add>(*sym_expr_)) {
    return OperationType::kOpAdd;
  }
  if (SymEngine::is_a<SymEngine::Mul>(*sym_expr_)) {
    return OperationType::kOpMul;
  }
  if (SymEngine::is_a<SymEngine::Max>(*sym_expr_)) {
    return OperationType::kOpMax;
  }
  if (SymEngine::is_a<SymEngine::Min>(*sym_expr_)) {
    return OperationType::kOpMin;
  }
  if (SymEngine::is_a<SymEngine::Pow>(*sym_expr_)) {
    return OperationType::kOpPow;
  }
  if (SymEngine::is_a<SymEngine::Mod>(*sym_expr_)) {
    return OperationType::kOpMod;
  }
  if (SymEngine::is_a<SymEngine::Log>(*sym_expr_)) {
    return OperationType::kOpLog;
  }
  if (SymEngine::is_a<SymEngine::Abs>(*sym_expr_)) {
    return OperationType::kOpAbs;
  }
  if (SymEngine::is_a<SymEngine::Ceiling>(*sym_expr_)) {
    return OperationType::kOpCeil;
  }
  if (SymEngine::is_a<SymEngine::Floor>(*sym_expr_)) {
    return OperationType::kOpFloor;
  }
  if (SymEngine::is_a<SymEngine::Equality>(*sym_expr_)) {
    return OperationType::kOpEq;
  }
  if (SymEngine::is_a<SymEngine::Unequality>(*sym_expr_)) {
    return OperationType::kOpNe;
  }
  if (SymEngine::is_a<SymEngine::LessThan>(*sym_expr_)) {
    return OperationType::kOpLe;
  }
  if (SymEngine::is_a<SymEngine::StrictLessThan>(*sym_expr_)) {
    return OperationType::kOpLt;
  }
  if (SymEngine::is_a<SymEngine::And>(*sym_expr_)) {
    return OperationType::kOpLogicalAnd;
  }
  if (SymEngine::is_a<SymEngine::Or>(*sym_expr_)) {
    return OperationType::kOpLogicalOr;
  }
  return OperationType::kOpNone;
}

std::string ExpressionImpl::GetName() const {
  if (IsConstExpr() || GetExprType() == ExprType::kExprVariable) {
    if (name_.empty()) {
      static std::atomic<size_t> unique_id(0);
      // 此处不应该使用Str()拼接，比如对于1.0,会生成Const_1.0_1，如果codegen采用此名字定义c++变量会编译报错
      name_ = "Const_" + std::to_string(unique_id.fetch_add(1));
    }
    return name_;
  } else {
    return kInvalidName;
  }
}

ExprType ExpressionImpl::GetExprType() const {
  if (sym_expr_.is_null()) {
    GELOGE(ge::PARAM_INVALID, "sym_expr_ is null.");
    return ExprType::kExprNone;
  }
  if (SymEngine::is_a_Number(*sym_expr_)) {
    if (SymEngine::is_a<SymEngine::Integer>(*sym_expr_)) {
      return ExprType::kExprConstantInteger;
    } else if (SymEngine::is_a<SymEngine::RealDouble>(*sym_expr_)) {
      return ExprType::kExprConstantRealDouble;
    } else if (SymEngine::is_a<SymEngine::Rational>(*sym_expr_)) {
      return ExprType::kExprConstantRation;
    } else {
      GELOGE(ge::PARAM_INVALID, "Unsupported type for expression %s", sym_expr_->__str__().c_str());
      return ExprType::kExprNone;
    }
  }
  if (SymEngine::is_a<SymEngine::BooleanAtom>(*sym_expr_)) {
    return ExprType::kExprConstantBoolean;
  }
  if (SymEngine::is_a<SymEngine::Symbol>(*sym_expr_)) {
    return ExprType::kExprVariable;
  }
  if (SymEngine::is_a_Boolean(*sym_expr_)) {
    return ExprType::kExprOperationBoolean;
  }
  return ExprType::kExprOperation;
}

bool ExpressionImpl::IsConstExpr() const {
  return GetExprType() < ExprType::kExprVariable;
}

ExpressionImpl::ExpressionImpl(int32_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(int64_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(uint32_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(uint64_t const_value, const std::string &name)
    : sym_expr_(SymEngine::integer(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(double const_value, const std::string &name)
    : sym_expr_(SymEngine::real_double(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(bool const_value, const std::string &name)
    : sym_expr_(SymEngine::boolean(const_value)), name_(name) {}

ExpressionImpl::ExpressionImpl(const std::string &name) : sym_expr_(SymEngine::symbol(name)), name_(name) {}

ExpressionImpl::ExpressionImpl(const SymEngineExprPtr &sym_expr, const std::string &name)
    : sym_expr_(sym_expr), name_(name) {}

ExpressionImplPtr Add(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::add(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Sub(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::sub(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Mul(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::mul(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Div(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::div(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Max(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::max({a->sym_expr_, b->sym_expr_});
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Min(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::min({a->sym_expr_, b->sym_expr_});
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Abs(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::abs(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Pow(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::pow(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Exp(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::pow(SymEngine::E, a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Mod(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::mod(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Log(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::log(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Log(const ExpressionImplPtr &arg, const ExpressionImplPtr &base) {
  GE_ASSERT_NOTNULL(arg);
  GE_ASSERT_NOTNULL(base);
  GE_ASSERT_TRUE(!arg->sym_expr_.is_null());
  GE_ASSERT_TRUE(!base->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::log(arg->sym_expr_, base->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Coeff(const ExpressionImplPtr &b, const ExpressionImplPtr &x, const ExpressionImplPtr &n) {
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_NOTNULL(x);
  GE_ASSERT_NOTNULL(n);
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  GE_ASSERT_TRUE(!x->sym_expr_.is_null());
  GE_ASSERT_TRUE(!n->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::coeff(*b->sym_expr_.get(), *x->sym_expr_.get(), *n->sym_expr_.get());
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Ceiling(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::ceiling(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Floor(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::floor(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Rational(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  if (SymEngine::is_a<SymEngine::Integer>(*a->sym_expr_) && SymEngine::is_a<SymEngine::Integer>(*b->sym_expr_)) {
    const auto &integer_expr_a = SymEngine::down_cast<const SymEngine::Integer &>(*a->sym_expr_);
    const auto &integer_expr_b = SymEngine::down_cast<const SymEngine::Integer &>(*b->sym_expr_);
    SymEngineExprPtr sym_expr = SymEngine::Rational::from_two_ints(integer_expr_a, integer_expr_b);
    auto impl = ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
    return impl;
  } else {
    GELOGE(ge::PARAM_INVALID, "unsupported rational expr");
    return nullptr;
  }
}

std::ostream &operator<<(std::ostream &os, const ExpressionImpl &e) {
  os << e.Str();
  return os;
}

ExpressionImplPtr Eq(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::Eq(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Ne(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::Ne(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr Lt(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::Lt(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}
ExpressionImplPtr Le(const ExpressionImplPtr &a, const ExpressionImplPtr &b) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_NOTNULL(b);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  GE_ASSERT_TRUE(!b->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::Le(a->sym_expr_, b->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}
ExpressionImplPtr Not(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  if (!SymEngine::is_a_Boolean(*a->sym_expr_)) {
    GELOGE(ge::PARAM_INVALID, "Logic operator Not only can handle Boolean expression:%s", a->Str().c_str());
    return nullptr;
  }
  SymEngineExprPtr sym_expr =
      SymEngine::logical_not(SymEngine::rcp_static_cast<const SymEngine::Boolean>(a->sym_expr_));
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}
ExpressionImplPtr Neg(const ExpressionImplPtr &a) {
  GE_ASSERT_NOTNULL(a);
  GE_ASSERT_TRUE(!a->sym_expr_.is_null());
  SymEngineExprPtr sym_expr = SymEngine::neg(a->sym_expr_);
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(sym_expr);
}

ExpressionImplPtr LogicalAnd(const std::vector<ExpressionImplPtr> &s) {
  SymEngine::set_boolean set;
  for (const auto &e : s) {
    GE_ASSERT_NOTNULL(e);
    GE_ASSERT_TRUE(!e->sym_expr_.is_null());
    GE_ASSERT_TRUE(SymEngine::is_a_Boolean(*e->sym_expr_), "Logic operator And only can handle Boolean expression: %s",
                   e->Str().c_str());
    set.insert(SymEngine::rcp_static_cast<const SymEngine::Boolean>(e->sym_expr_));
  }
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(SymEngine::logical_and(set));
}

ExpressionImplPtr LogicalOr(const std::vector<ExpressionImplPtr> &s) {
  SymEngine::set_boolean set;
  for (const auto &e : s) {
    GE_ASSERT_NOTNULL(e);
    GE_ASSERT_TRUE(!e->sym_expr_.is_null());
    GE_ASSERT_TRUE(SymEngine::is_a_Boolean(*e->sym_expr_), "Logic operator Or only can handle Boolean expression: %s",
                   e->Str().c_str());
    set.insert(SymEngine::rcp_static_cast<const SymEngine::Boolean>(e->sym_expr_));
  }
  return ExpressionImpl::CreateExpressionImpl<const SymEngineExprPtr &>(SymEngine::logical_or(set));
}
}  // namespace af
