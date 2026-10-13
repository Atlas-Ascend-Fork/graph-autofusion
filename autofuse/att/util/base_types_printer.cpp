/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "base_types_printer.h"

namespace af {
void to_json(nlohmann::json &j, const Expression &arg) {
  std::string expr;
  if (arg.IsValid()) {
    auto str_ptr = arg.Str();
    if (str_ptr) {
      expr = str_ptr.get();
    }
  }
  j = nlohmann::json{
      {expr},
  };
}
}  // namespace af
namespace att {
namespace {
void AddDenominatorSymbol(const Expr &symbol, const ExprExprMap &container_expr,
                          std::set<std::string> &denominator_symbols) {
  auto iter = container_expr.find(symbol);
  if (iter == container_expr.end()) {
    denominator_symbols.insert(Str(symbol));
    return;
  }
  for (const auto &arg : iter->second.FreeSymbols()) {
    if (arg.GetExprType() == af::ExprType::kExprVariable) {
      denominator_symbols.insert(Str(arg));
    }
  }
}

void CollectDenominatorSymbols(const Expr &expr, const ExprExprMap &container_expr,
                               std::set<std::string> &denominator_symbols) {
  if (!expr.IsValid() || expr.IsConstExpr()) {
    return;
  }
  Expr numer;
  Expr denom;
  expr.AsNumerDenom(numer, denom);
  if (denom.IsValid() && !denom.IsConstExpr()) {
    for (const auto &symbol : denom.FreeSymbols()) {
      if (symbol.GetExprType() == af::ExprType::kExprVariable) {
        AddDenominatorSymbol(symbol, container_expr, denominator_symbols);
      }
    }
  }
  Expr expr_copy = expr;
  for (const auto &arg : expr_copy.GetArgs()) {
    CollectDenominatorSymbols(arg, container_expr, denominator_symbols);
  }
}

std::string GenDenominatorGuard(const std::set<std::string> &denominator_symbols) {
  std::string guard_code;
  for (const auto &symbol : denominator_symbols) {
    guard_code += "    if (" + symbol + " <= 0) {\n";
    guard_code += "      OP_LOGW(OP_NAME, \"Invalid workspace denominator " + symbol + "=%lf.\", " + symbol + ");\n";
    guard_code += "      return false;\n";
    guard_code += "    }\n";
  }
  return guard_code;
}
}  // namespace

const std::string AddAnotationBlock(std::string strs, std::string indent) {
  std::string str = indent + "/*\n" + strs + indent + "*/\n";
  return str;
}
const std::string AddAnotationLine(std::string strs, std::string indent) {
  std::string str = indent + "// " + strs;
  return str;
}

void ScanContainer(const Expr &container, const Expr &container_expr, std::set<std::string> &arg_names,
                   ExprExprMap &param_map) {
  for (const auto &arg : container_expr.FreeSymbols()) {
    arg_names.insert(Str(arg));
  }
  param_map[container] = container_expr;
}

void AnalysisArg(const Expr &arg, const ExprExprMap &container_expr, std::set<std::string> &arg_names,
                 ExprExprMap &param_map) {
  if (arg.GetExprType() == af::ExprType::kExprVariable) {
    auto iter1 = container_expr.find(arg);
    if (iter1 != container_expr.end()) {
      ScanContainer(arg, iter1->second, arg_names, param_map);
    } else {
      arg_names.insert(Str(arg));
    }
  }
}

const std::string GenWorkspaceRelatedVars(const std::map<int64_t, Expr> &workspace_size_map,
                                          const ExprExprMap &container_expr) {
  std::string ret;
  std::set<std::string> arg_names;
  std::set<std::string> denominator_symbols;
  ExprExprMap params_map;
  for (const auto &workspace_size_func : workspace_size_map) {
    GELOGD("The workspace_size func is [%s].", Str(workspace_size_func.second).c_str());
    CollectDenominatorSymbols(workspace_size_func.second, container_expr, denominator_symbols);
    for (const auto &arg : workspace_size_func.second.FreeSymbols()) {
      AnalysisArg(arg, container_expr, arg_names, params_map);
    }
  }
  for (const auto &arg_name : arg_names) {
    ret += "    double " + arg_name + " = tiling_data.get_" + arg_name + "();\n";
  }
  ret += GenDenominatorGuard(denominator_symbols);
  for (const auto &workspace_size : workspace_size_map) {
    auto tensor_id = std::to_string(workspace_size.first);
    ret += "\n    auto it" + tensor_id + " = workspace_map.find(" + tensor_id + ");\n";
    ret += "    if (it" + tensor_id + " != workspace_map.end()) {\n";
    ret += "        it" + tensor_id + "->second = Max(it" + tensor_id + "->second, static_cast<uint64_t>(" +
           Str(workspace_size.second) + "));\n";
    ret += "    } else {\n";
    ret += "        workspace_map[" + tensor_id + "] = static_cast<uint64_t>(" + Str(workspace_size.second) + ");\n";
    ret += "    }";
  }
  return ret;
}

const std::string GenRelatedVars(const std::vector<Expr> &funcs, const ExprExprMap &container_expr,
                                 const std::map<Expr, std::vector<Expr>, ExprCmp> &args) {
  std::string ret;
  std::set<std::string> arg_names;
  ExprExprMap params_map;
  for (const auto &func : funcs) {
    GELOGD("The func is [%s].", Str(func).c_str());
    for (const auto &arg : func.FreeSymbols()) {
      GELOGD("Analysis arg [%s].", Str(arg).c_str());
      auto iter = args.find(arg);
      if (iter != args.end()) {
        for (const auto &item : iter->second) {
          GELOGD("Insert arg [%s].", Str(item).c_str());
          arg_names.insert(Str(item));
        }
      } else {
        AnalysisArg(arg, container_expr, arg_names, params_map);
      }
    }
  }
  for (const auto &arg_name : arg_names) {
    ret += "    double " + arg_name + " = tiling_data.get_" + arg_name + "();\n";
  }
  for (const auto &pair : params_map) {
    ret += "    double " + Str(pair.first) + " = " + Str(pair.second) + ";\n";
  }
  return ret;
}

const std::string GenBufRelatedVars(const Expr &func, const ExprExprMap &container_expr) {
  std::string ret;
  std::set<std::string> arg_names;
  ExprExprMap params_map;
  for (const auto &arg : func.FreeSymbols()) {
    AnalysisArg(arg, container_expr, arg_names, params_map);
  }
  for (const auto &arg_name : arg_names) {
    ret += "    double " + arg_name + " = tiling_data.get_" + arg_name + "();\n";
  }

  std::map<std::string, ASTNode> ast_expr_map;
  Optimizer ast_optimizer;
  for (const auto &pair : params_map) {
    Parser parser(Str(pair.second));
    ASTPtr ast = parser.Parse();
    ast_optimizer.Optimize(ast);
    ast_expr_map.emplace(Str(pair.first), *ast.get());
  }
  std::string tmp_vars = ast_optimizer.GenerateCode();
  ret += tmp_vars;
  for (const auto &pair : ast_expr_map) {
    auto &ast = pair.second;
    std::string return_expr = ast_optimizer.RebuildExpr(ast, 1);
    ret += "    double " + pair.first + " = " + return_expr + ";\n";
  }
  Parser parser(Str(func));
  ASTPtr ast = parser.Parse();
  ast_optimizer.Optimize(ast);
  std::string func_tmp_vars = ast_optimizer.GenerateCode();
  ret += func_tmp_vars;
  std::string func_return_expr = ast_optimizer.RebuildExpr(*ast.get(), 1);
  ret += "    return " + func_return_expr + ";\n";
  return ret;
}

std::string BaseTypeUtils::DumpHardware(const HardwareDef hardware) {
  const auto &hardware_name_iter = kHardwareNameMap.find(hardware);
  if (hardware_name_iter == kHardwareNameMap.cend()) {
    return kHardwareNameMap.at(HardwareDef::HARDWAREERR);
  }
  return hardware_name_iter->second;
}

std::string BaseTypeUtils::DtypeToStr(ge::DataType dtype) {
  const std::map<ge::DataType, const ge::char_t *> kTypeName = {
      {ge::DT_FLOAT, "float32"}, {ge::DT_FLOAT16, "float16"}, {ge::DT_BF16, "bfloat16"}, {ge::DT_INT8, "int8"},
      {ge::DT_UINT8, "uint8"},   {ge::DT_INT16, "int16"},     {ge::DT_UINT16, "uint16"}, {ge::DT_INT32, "int32"},
      {ge::DT_UINT32, "uint32"}, {ge::DT_INT64, "int64"},     {ge::DT_UINT64, "uint64"}, {ge::DT_DOUBLE, "double"},
      {ge::DT_BOOL, "bool"}};
  const auto &type_name_iter = kTypeName.find(dtype);
  if (type_name_iter == kTypeName.end()) {
    return "unknown";
  }
  return type_name_iter->second;
}

}  // namespace att
