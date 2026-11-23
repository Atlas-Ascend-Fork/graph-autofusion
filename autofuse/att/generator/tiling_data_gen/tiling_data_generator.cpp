/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_data_generator.h"
#include <regex>
#include <unordered_set>
#include <set>
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "code_printer.h"
#include "generator_utils/tilingdata_gen_utils.h"
#include "generator/preprocess/ast_optimizer.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace att {
namespace {
constexpr int32_t kNotAlignedSize = 1;  // if align is 1, do not need align
constexpr char kLoopNumSuffix[] = "_loop_num";
constexpr char kTailSizeSuffix[] = "_tail_size";
// (kTilingDataPrefix + tiling_data + kTilingDataSuffix) equal to tiling_data in real code.
// for example, tail_size in real code is tiling_data.get_tail_size()
constexpr char kTilingDataPrefix[] = "tiling_data.get_";
constexpr char kTilingDataSuffix[] = "()";
constexpr char kGetSizeSuffix[] = "_size";
constexpr char kBlockDim[] = "block_dim";
constexpr char kGmSize[] = "gm_size";
const std::unordered_map<TilingDataType, std::string> kTilingDataTypeToAnnotation = {
    {TilingDataType::AXIS_ALIGNED_SIZE,
     "  // 参数：{轴}_aligned_size\n"
     "  // 含义：本轴对齐后的大小\n"
     "  // 约束：仅Ascend IR表达了对齐时，才会生成该参数\n"
     "  // 计算公式：{轴}_aligned_size = ({轴}_size - 1) / ALIGN_SIZE * ALIGN_SIZE + ALIGN_SIZE"},
    {TilingDataType::AXIS_TAIL_SIZE,
     "  // 参数：{轴}_tail_size\n"
     "  // 含义：本轴的最后一次循环，需要循环多少次最内轴元素\n"
     "  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)\n"
     "  // 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)"},
    {TilingDataType::AXIS_LOOP_NUM,
     "  // 参数：{轴}_loop_num\n"
     "  // 含义：本轴的父轴需要循环多少次本轴\n"
     "  // 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)\n"
     "  // 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)\n"
     "  //                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size"},
    {TilingDataType::SPLIT_OUTER_AXIS_TAIL_TAIL_SIZE,
     "  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_tail_size\n"
     "  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次最内轴元素\n"
     "  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；\n"
     "  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);\n"
     "  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_tail_size = ({轴的父轴}_tail_size % {轴}_size) == 0 ? {轴}_size : "
     "\n"
     "  //         ({轴的父轴}_tail_size % {轴}_size)"},
    {TilingDataType::SPLIT_OUTER_AXIS_TAIL_LOOP_NUM,
     "  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num\n"
     "  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴\n"
     "  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；\n"
     "  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);\n"
     "  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)\n"
     "  //                                                  = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size"},
    {TilingDataType::SPLIT_OUTER_AXIS_TAIL_LOOP_NUM,
     "  // 参数：{轴的父轴}_tail_{切分类型}_{轴}_loop_num\n"
     "  // 含义：本轴的父轴是按照{切分类型}切分后的尾块，父轴的尾块需要循环多少次本轴\n"
     "  // 约束：1.仅父轴按照{切分类型}切分时生成该参数；\n"
     "  //      2.当前仅支持父轴对Block切分的场景(比如先切Block，后切Tile);\n"
     "  // 计算公式：{轴的父轴}_tail_{切分类型}_{轴}_loop_num = Ceil({轴的父轴}_tail_size / {轴}_size)\n"
     "  //                                               = (({轴的父轴}_tail_size + {轴}_size) - 1) / {轴}_size"},
    {TilingDataType::USED_BLOCK_DIM,
     "  // 参数：block_dim\n"
     "  // 含义：实际使用的aiv core核数\n"
     "  // 约束：所有场景均会生成该参数\n"
     "  // 计算公式：block_dim = 所有按照Block切分轴的loop_num的乘积（若未对Block切分，则为1）"},
    {TilingDataType::BUFFER_SIZE,
     "  // 含义：Tbuf或者Queue使用的内存大小\n"
     "  // 约束：定义了Tbuf或者Queue则会生成该参数\n"
     "  // 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存"},
    {TilingDataType::TENSOR_SIZE, "  // 含义：Tensor所占用内存大小"}};
using IsGenEnableFunc = std::function<bool(const ExtraInfoConfig &)>;
const std::unordered_map<TilingDataGenType, IsGenEnableFunc> kIsGenEnableFuncMap = {
    {TilingDataGenType::AXES_TILING_DATA_GEN,
     [](const ExtraInfoConfig &config) -> bool { return config.do_axes_calc; }},
    {TilingDataGenType::GENERAL_TILING_DATA_GEN,
     [](const ExtraInfoConfig &config) -> bool {
       (void)config;
       return true;
     }},
    {TilingDataGenType::ALL_TILING_DATA_GEN,
     [](const ExtraInfoConfig &config) -> bool {
       (void)config;
       return true;
     }},
    {TilingDataGenType::MEMORY_TILING_DATA_GEN, [](const ExtraInfoConfig &config) -> bool {
       (void)config;
       return true;
     }}};

bool IsGenEnable(const TilingDataGenType tiling_data_gen_type) {
  const auto iter = kIsGenEnableFuncMap.find(tiling_data_gen_type);
  return (iter != kIsGenEnableFuncMap.cend()) && iter->second;
}

bool IsVariable(const std::string &var_or_expr) {
  // 检查匹配是否为变量，即它不是数字且不是函数的一部分
  if (std::all_of(var_or_expr.begin(), var_or_expr.end(), ::isdigit)) {
    return false;  // 是数字则返回false
  }
  // if match is expr, such as (B * N), return false
  std::regex pattern("^[a-zA-Z0-9_]+$");
  return std::regex_match(var_or_expr, pattern);
}

inline std::string GetTilingDataVar(const std::string &var_or_expr) {
  if (IsVariable(var_or_expr)) {
    return kTilingDataPrefix + var_or_expr + kTilingDataSuffix;
  }
  return var_or_expr;
}

std::string StrCalcTail(const std::string &a, const std::string &b) {
  std::stringstream tail_calc_expr;
  tail_calc_expr << "(" << GetTilingDataVar(a) << " % " << GetTilingDataVar(b) << ") == 0 ? " << GetTilingDataVar(b)
                 << " : (" << GetTilingDataVar(a) << " % " << GetTilingDataVar(b) << ")";
  return tail_calc_expr.str();
}

std::string StrCeilDiv(const std::string &a, const std::string &b) {
  // ((a + b) - 1) / b
  std::stringstream ceil_div_expr;
  const auto &tiling_data_b = GetTilingDataVar(b);
  ceil_div_expr << "((" << GetTilingDataVar(a) << " + " << tiling_data_b << ")"
                << " - 1) / " << tiling_data_b;
  return ceil_div_expr.str();
}

std::string StrAlign(const std::string &var, const std::string &align) {
  // (var - 1) * align / align + align
  std::stringstream aligned_expr;
  aligned_expr << "(" << GetTilingDataVar(var) << " - 1) / " << align << " * " << align << " + " << align;
  return aligned_expr.str();
}

af::Status GetOneParentAxesTilingDataSize(const AttAxis *parent_size, std::string &result) {
  std::unordered_set<std::string> parent_contain_vars;
  GELOGD("Parent size expr: %s", parent_size->size->symbol_expr.Str().get());
  for (const auto &primary_arg : parent_size->size->symbol_expr.FreeSymbols()) {
    if (primary_arg.GetExprType() == af::ExprType::kExprVariable) {
      parent_contain_vars.insert(Str(primary_arg));
    }
  }
  std::vector<std::pair<Expr, Expr>> var_replacement;
  for (auto &var : parent_contain_vars) {
    GELOGD("contains var: %s", var.c_str());
    var_replacement.emplace_back(
        std::make_pair(CreateExpr(var.c_str()), CreateExpr((kTilingDataPrefix + var + kTilingDataSuffix).c_str())));
  }
  result += Str(parent_size->size->symbol_expr.Replace(var_replacement));
  return af::SUCCESS;
}

std::string EnsurePrecedence(const std::string &expr) {
  return "(" + expr + ")";
}

std::string GetAxisAlignedSizeName(const std::string &axis_name) {
  constexpr char KAlignedSizeSuffix[] = "_aligned_size";
  return axis_name + std::string(KAlignedSizeSuffix);
}

std::string GetAxisLoopNumName(const std::string &axis_name) {
  return axis_name + std::string(kLoopNumSuffix);
}

std::string GetAxisTailSizeName(const std::string &axis_name) {
  return axis_name + std::string(kTailSizeSuffix);
}

// 轴对应的{外轴}Tail{切分类型}{轴}
std::string GetSplitTailPrefix(const std::string &from_axis_name, const AttAxisPtr &axis) {
  std::string tail_prefix(from_axis_name);
  tail_prefix.append("_tail");
  tail_prefix.append(axis->bind_multicore ? "_block_" : "_tile_");
  tail_prefix.append(axis->name);
  return tail_prefix;
}

bool IsAxis1DependAxis2(const AttAxis *axis1, const AttAxis *axis2) {
  std::unordered_set<AttAxisPtr> visited;
  for (const auto parent_axis : axis1->from_axis) {
    if (parent_axis->name == axis2->name) {
      return true;
    }
    if (IsAxis1DependAxis2(parent_axis, axis2)) {
      return true;
    }
  }
  return false;
}

void SetTilingData(const TilingDataType tiling_data_type, const std::string &tiling_data_name,
                   const std::string &tiling_data_expr, TilingDataMap &tiling_data_map) {
  tiling_data_map[tiling_data_name] = std::make_pair(tiling_data_type, tiling_data_expr);
  GELOGD("Add tiling data:(type:%d, name:%s, expr:%s)", static_cast<int32_t>(tiling_data_type),
         tiling_data_name.c_str(), tiling_data_expr.c_str());
}
}  // namespace

af::Status AxesTilingDataGen::AddAxesAlignedSize() {
  for (const auto &axis : model_info_.arg_list) {
    // note: all axis aligned is set by user, get_{axis} must be valid
    const auto &axis_size_var = std::dynamic_pointer_cast<SymVarInfo>(axis->size);
    if ((axis->axis_pos == AxisPosition::ORIGIN) && (axis_size_var != nullptr) &&
        (!axis_size_var->align.IsConstExpr() ||
         (axis_size_var->align.IsConstExpr() &&
          (af::SymbolicUtils::StaticCheckGt(axis_size_var->align, af::Symbol(kNotAlignedSize)) ==
           af::TriBool::kTrue)))) {
      std::string axis_size_name;
      // axes_aligned_size = (axes_size - 1) / ALIGN_SIZE * ALIGN_SIZE + ALIGN_SIZE, origin axis use SizeVar name
      const auto align_expr = StrAlign(Str(axis->size->symbol_expr), Str(axis_size_var->align));
      GE_ASSERT_SUCCESS(SetAxisArgExpr(axis->name, {TilingDataType::AXIS_ALIGNED_SIZE,
                                                    GetAxisAlignedSizeName(axis->name), align_expr}),
                        "Set aligned size of axis[%s] failed.", axis->name.c_str());
    }
  }
  return af::SUCCESS;
}

af::Status AxesTilingDataGen::AddAxesTailSizeAndLoopNum() {
  // axis_name->axis_base_size
  std::map<std::string, uint32_t> axes_names_to_base_sizes;
  for (const auto &axis : model_info_.arg_list) {
    // Only split can make tail size and loop num, INNER axis size provide split base size
    if (axis->axis_pos != AxisPosition::INNER) {
      continue;
    }
    GE_ASSERT_TRUE(axis->from_axis.size() == 1UL, "axis[%s] is an inner axis and should only have one from_axis.",
                   axis->name.c_str());
    // 轴对应的BaseSize
    const auto axis_base_size = GetArgExpr(axis->name);
    // 轴的TailSize = 父轴的BaseSize % 轴的BaseSize
    std::string parent_base_size;
    GE_ASSERT_SUCCESS(GetOneParentAxesTilingDataSize(axis->from_axis[0], parent_base_size),
                      "Get parent size failed of axis[%s]", axis->name.c_str());
    const auto axis_tail_size = StrCalcTail(parent_base_size, Str(axis_base_size));
    GE_ASSERT_SUCCESS(
        SetAxisArgExpr(axis->name, {TilingDataType::AXIS_TAIL_SIZE, GetAxisTailSizeName(axis->name), axis_tail_size}),
        "Set axis[%s] tail size failed", axis->name.c_str());
    // 轴的LoopNum = Ceil(父轴的BaseSize / 轴的BaseSize) = (父轴的BaseSize + 轴的BaseSize - 1) / 轴的BaseSize
    const auto axis_loop_num = StrCeilDiv(parent_base_size, Str(axis_base_size));
    GE_ASSERT_SUCCESS(
        SetAxisArgExpr(axis->name, {TilingDataType::AXIS_LOOP_NUM, GetAxisLoopNumName(axis->name), axis_loop_num}),
        "Set axis[%s] loop num failed", axis->name.c_str());
    GELOGD("Add axis[%s] args: tail_size[%s], loop_num[%s], parent_base_size[%s], axis_base_size[%s]",
           axis->name.c_str(), axis_tail_size.c_str(), axis_loop_num.c_str(), parent_base_size.c_str(),
           axis_base_size.Str().get());
  }
  return af::SUCCESS;
}

af::Status AxesTilingDataGen::AddSplitOuterAxisTailArgs() {
  for (const auto &axis : model_info_.arg_list) {
    if (axis->axis_pos != AxisPosition::INNER) {
      continue;
    }
    GE_ASSERT_TRUE(axis->from_axis.size() == 1UL, "axis[%s] is an inner axis and should only have one from_axis.",
                   axis->name.c_str());
    // INNER axis should only has one parent axis
    std::string parents_size;
    GE_ASSERT_SUCCESS(GetOneParentAxesTilingDataSize(axis->from_axis[0], parents_size),
                      "Get parent size failed of axis[%s]", axis->name.c_str());
    GELOGD("INNER axis[%s] parent axis size[%zu] expr[%s] bind_multicore[%d] parent_axis->axis_type[%d]",
           axis->name.c_str(), axis->from_axis.size(), parents_size.c_str(), axis->bind_multicore,
           static_cast<int32_t>(axis->axis_pos));
    //  s1
    //   |
    //  |  |
    // s1B s1b(INNER)
    //      |
    //     |  |
    //   s1bT s1bt(INNER, match)
    // 过滤条件：父轴若存在tail_size，则可能会切出尾块，会生成对应参数
    for (const auto &parent_axis : axis->from_axis) {
      const auto parent_axis_tail_size = GetAxisTilingData(parent_axis->name, TilingDataType::AXIS_TAIL_SIZE);
      if (!parent_axis_tail_size.first.empty()) {
        // 轴对应的{外轴}Tail{切分类型}{轴}TailSize = {外轴}TailSize % {轴}BaseSize
        const auto axis_base_size = GetArgExpr(axis->name);
        const auto &axis_base_size_str = Str(axis_base_size);
        const auto tail_part_tail_size = StrCalcTail(parent_axis_tail_size.first, axis_base_size_str);
        GE_ASSERT_SUCCESS(
            SetAxisArgExpr(axis->name,
                           {TilingDataType::SPLIT_OUTER_AXIS_TAIL_TAIL_SIZE,
                            GetSplitTailPrefix(parent_axis->name, axis).append(kTailSizeSuffix), tail_part_tail_size}),
            "Set split outer axis tail size failed, axis[%s]", axis->name.c_str());
        // 轴对应的{外轴}Tail{切分类型}{轴}LoopNum = StrCeilDiv({外轴}TailSize / {轴}BaseSize)
        const auto tail_part_loop_num = StrCeilDiv(parent_axis_tail_size.first, axis_base_size_str);
        GE_ASSERT_SUCCESS(
            SetAxisArgExpr(axis->name,
                           {TilingDataType::SPLIT_OUTER_AXIS_TAIL_LOOP_NUM,
                            GetSplitTailPrefix(parent_axis->name, axis).append(kLoopNumSuffix), tail_part_loop_num}),
            "Set split outer axis tail loop num failed, axis[%s]", axis->name.c_str());
        GELOGD("Add parent axis[%s] tail block axis[%s] args: loop_num[%s], tail_size[%s]", parent_axis->name.c_str(),
               axis->name.c_str(), tail_part_loop_num.c_str(), tail_part_tail_size.c_str());
      }
    }
  }
  return af::SUCCESS;
}

std::vector<AxisTilingData> AxesTilingDataGen::GetAxisTilingData(const std::string &axis_name) const {
  std::vector<AxisTilingData> axis_tiling_data_res;
  const auto &axes_tiling_data_iter = axes_tiling_data_map_.find(axis_name);
  const bool found = (axes_tiling_data_iter != axes_tiling_data_map_.cend());
  GELOGD("Get axis[%s] tiling data size[%zu]", axis_name.c_str(), found ? axes_tiling_data_iter->second.size() : 0U);
  if (found) {
    return axes_tiling_data_iter->second;
  }
  // cannot find tiling of axis
  return axis_tiling_data_res;
}

af::Status AxesTilingDataGen::SetAxisArgExpr(const std::string &axis_name, const AxisTilingData &axis_tiling_data) {
  axes_tiling_data_map_[axis_name].emplace_back(axis_tiling_data);
  GELOGD("Add tiling data axis[%s], type:%d, name:%s, expr:%s", axis_name.c_str(),
         static_cast<int32_t>(axis_tiling_data.arg_type), axis_tiling_data.arg_name.c_str(),
         axis_tiling_data.arg_expr.c_str());
  return af::SUCCESS;
}

std::pair<std::string, std::string> AxesTilingDataGen::GetAxisTilingData(const std::string &axis_name,
                                                                         const TilingDataType arg_type) const {
  const auto axis_all_tiling_data = GetAxisTilingData(axis_name);
  for (const auto &axis_tiling_data : axis_all_tiling_data) {
    if (arg_type == axis_tiling_data.arg_type) {
      return std::make_pair(axis_tiling_data.arg_name, axis_tiling_data.arg_expr);
    }
  }
  // cannot found
  GELOGD("Cannot find axis tiling data by axis_name[%s], arg_type[%d]", axis_name.c_str(),
         static_cast<int32_t>(arg_type));
  return std::pair<std::string, std::string>();
}

std::vector<std::pair<std::string, std::string>> AxesTilingDataGen::GetTilingDataWithAnnotation() const {
  std::vector<std::pair<std::string, std::string>> tiling_data_names;
  for (const auto &axis_name : ordered_axes_names_) {
    const auto axis_tiling_datas = GetAxisTilingData(axis_name);
    for (const auto &axis_tiling_data : axis_tiling_datas) {
      std::string arg_annotation;
      const auto &annotation_iter = kTilingDataTypeToAnnotation.find(axis_tiling_data.arg_type);
      if (annotation_iter != kTilingDataTypeToAnnotation.cend()) {
        arg_annotation = annotation_iter->second;
      }
      tiling_data_names.emplace_back(axis_tiling_data.arg_name, arg_annotation);
    }
  }
  return tiling_data_names;
}

std::vector<std::pair<std::string, std::string>> AxesTilingDataGen::GetAxesTilingDataWithExpr() const {
  std::vector<std::pair<std::string, std::string>> tiling_datas_exprs;
  std::vector<AxisTilingData> all_axis_tiling_datas;
  for (const auto &ordered_axes_name : ordered_axes_names_) {
    const auto axis_tiling_datas = GetAxisTilingData(ordered_axes_name);
    all_axis_tiling_datas.insert(all_axis_tiling_datas.cend(), axis_tiling_datas.begin(), axis_tiling_datas.end());
  }
  // make sure loop_num/tail_size is before tail_block args
  std::sort(all_axis_tiling_datas.begin(), all_axis_tiling_datas.end(),
            [](const AxisTilingData &a, const AxisTilingData &b) -> bool {
              return static_cast<int32_t>(a.arg_type) < static_cast<int32_t>(b.arg_type);
            });
  tiling_datas_exprs.reserve(all_axis_tiling_datas.size());
  for (const auto &axis_tiling_data : all_axis_tiling_datas) {
    tiling_datas_exprs.emplace_back(std::make_pair(axis_tiling_data.arg_name, axis_tiling_data.arg_expr));
    GELOGD("Get axes tiling data(arg_name[%s], arg_expr[%s])", axis_tiling_data.arg_name.c_str(),
           axis_tiling_data.arg_expr.c_str());
  }
  return tiling_datas_exprs;
}

std::vector<std::string> AxesTilingDataGen::GetTilingFuncImpl(const std::string &tiling_type) const {
  // gen axes tiling func impl
  std::vector<std::string> impl_codes;
  impl_codes.emplace_back(("  void UpdateAxesTilingData(" + tiling_type + "& tiling_data) {"));
  // example: tiling_data.set_{{tiling_data_name}}(tiling_data_expr)
  for (const auto &tiling_code : GetAxesTilingDataWithExpr()) {
    impl_codes.emplace_back(("    tiling_data.set_") + (tiling_code.first) + ("(") + (tiling_code.second) + (");"));
  }
  impl_codes.emplace_back("  }\n");
  return impl_codes;
}

void AxesTilingDataGen::MakeSureParentAxesFirst() {
  std::vector<AttAxisPtr> ordered_axes;
  ordered_axes.reserve(model_info_.arg_list.size());
  for (const auto &axis : model_info_.arg_list) {
    ordered_axes.emplace_back(axis);
  }
  std::sort(ordered_axes.begin(), ordered_axes.end(), [](const AttAxisPtr &axis1, const AttAxisPtr &axis2) -> bool {
    return IsAxis1DependAxis2(axis1.get(), axis2.get());
  });
  std::stringstream ss;
  for (const auto &ordered_axis : ordered_axes) {
    ordered_axes_names_.emplace_back(ordered_axis->name);
    ss << ordered_axis->name << " ";
  }
  GELOGD("Gen ordered(parent first order) axes[%s]", ss.str().c_str());
}

af::Status AxesTilingDataGen::Init() {
  // 添加轴size对齐表达式
  GE_ASSERT_SUCCESS(AddAxesAlignedSize(), "Add aligned sizes expr failed.");
  // 添加轴的tail_size和loop_num表达式
  GE_ASSERT_SUCCESS(AddAxesTailSizeAndLoopNum(), "Add axis tail sizes and loop num expr failed.");
  // 添加尾块的tail_size和尾块的loop_num表达式
  GE_ASSERT_SUCCESS(AddSplitOuterAxisTailArgs(), "Add axis split tail part tail size and loop num expr failed.");
  // make sure parent axes code is before child axes
  MakeSureParentAxesFirst();
  is_initialized_ = true;
  return af::SUCCESS;
}

std::string AxesTilingDataGen::GetTilingFuncInvoke() const {
  return "\t\tUpdateAxesTilingData(tiling_data);\n";
}

Expr AxesTilingDataGen::GetArgExpr(const std::string &axis_name) const {
  for (const auto &axis : model_info_.arg_list) {
    if (axis->name == axis_name) {
      if (!(axis->size == nullptr)) {
        return axis->size->symbol_expr;
      }
    }
  }
  Expr res;
  return res;
}

af::Status BlockDimTilingDataGen::Init() {
  GE_ASSERT_SUCCESS(AddUsedCoreNum());
  return af::SUCCESS;
}

af::Status BlockDimTilingDataGen::AddUsedCoreNum() {
  std::string used_core_num_str;
  constexpr char KNonBlockSplitCoreNum[] = "1";
  for (const auto &axis : model_info_.arg_list) {
    if (axis->bind_multicore && (axis->axis_pos == AxisPosition::INNER)) {
      GE_ASSERT_NOTNULL(axes_tiling_data_gen_);
      GE_ASSERT_TRUE(axes_tiling_data_gen_->IsInitialized(), "Axes tiling data gen has not been init.");
      const auto axis_loop_num = axes_tiling_data_gen_->GetAxisTilingData(axis->name, TilingDataType::AXIS_LOOP_NUM);
      if (!used_core_num_str.empty()) {
        used_core_num_str.append(" * ");
      }
      used_core_num_str.append(axis_loop_num.second.empty() ? KNonBlockSplitCoreNum
                                                            : EnsurePrecedence(axis_loop_num.second));
    }
  }
  if (used_core_num_str.empty()) {
    used_core_num_str = KNonBlockSplitCoreNum;
  }
  SetTilingData(TilingDataType::USED_BLOCK_DIM, kBlockDim, used_core_num_str, tiling_data_map_);
  return af::SUCCESS;
}

std::vector<std::string> BlockDimTilingDataGen::GetTilingFuncImpl(const std::string &tiling_type) const {
  std::vector<std::string> impl_codes;
  impl_codes.emplace_back(("  void UpdateGeneralTilingData(" + tiling_type + "& tiling_data) {"));
  // example: tiling_data.set_{{tiling_data_name}}(tiling_data_expr)
  for (const auto &tiling_code : tiling_data_map_) {
    impl_codes.emplace_back("    tiling_data.set_" + (tiling_code.first) + ("(") + (tiling_code.second.second) +
                            (");"));
  }
  impl_codes.emplace_back("  }\n");
  return impl_codes;
}

std::string BlockDimTilingDataGen::GetTilingFuncInvoke() const {
  return "\t\tUpdateGeneralTilingData(tiling_data);\n";
}

std::string MemoryTilingDataGen::GenFuncImpl(const std::pair<std::string, Expr> &var_name_to_expr,
                                             const std::string &tiling_type, const ExprExprMap &container_expr) const {
  std::vector<Expr> used_vars;
  std::set<std::string> define_vars;
  std::string func_impl;
  const auto &var_name = var_name_to_expr.first;
  const auto &var_expr = var_name_to_expr.second;
  GELOGD("var_name = [%s], var_expr = [%s]", var_name.c_str(), var_expr.Str().get());
  func_impl += "  void Set" + var_name + "(" + tiling_type + " &tiling_data) {\n";
  std::vector<std::pair<Expr, Expr>> replace_map;
  for (const auto &pair : container_expr) {
    replace_map.emplace_back(std::make_pair(pair.first, pair.second));
  }
  Expr used_expr = var_expr.Replace(replace_map);
  for (const auto &vars : used_expr.FreeSymbols()) {
    used_vars.emplace_back(vars);
  }
  for (const auto &origin_var : used_vars) {
    if (!IsValid(used_expr)) {
      GELOGW("Container[%s] expr doesn't exist.", var_name.c_str());
      continue;
    }
    if ((origin_var.IsConstExpr()) || (define_vars.find(Str(origin_var)) != define_vars.cend())) {
      continue;
    }
    func_impl += "    const auto " + Str(origin_var) + " = tiling_data.get_" + Str(origin_var) + "();\n";
    define_vars.insert(Str(origin_var));
  }

  Optimizer ast_optimizer;
  Parser parser(Str(used_expr));
  ASTPtr ast = parser.Parse();
  GE_ASSERT_NOTNULL(ast, "Parse expr failed: %s", Str(used_expr).c_str());
  ast_optimizer.Optimize(ast);
  std::string tmp_var = ast_optimizer.RebuildExpr(*ast.get(), 1);

  func_impl += ast_optimizer.GenerateCode();
  // 防止溢出导致编译错误，由于内存类tiling data类型均为uint32_t，这里增加强转uint32_t
  func_impl += "    tiling_data.set_" + var_name + "(static_cast<uint32_t>(" + tmp_var + "));\n";
  func_impl += "  }\n";
  return func_impl;
}

std::string MemoryTilingDataGen::GenFuncInvoke(const std::string &var_name) const {
  return std::string("    Set") + var_name + "(tiling_data);\n";
}

af::Status MemoryTilingDataGen::Init() {
  // OptionParam
  for (const auto &container : model_info_.container_exprs) {
    SetTilingData(TilingDataType::BUFFER_SIZE, container.first, Str(container.second), tiling_data_map_);
  }
  std::vector<std::string> tensors_name;
  for (const auto &tensor : model_info_.tensor_exprs) {
    SetTilingData(TilingDataType::TENSOR_SIZE, tensor.first, Str(tensor.second), tiling_data_map_);
  }
  return af::SUCCESS;
}

std::vector<std::string> MemoryTilingDataGen::GetTilingFuncImpl(const std::string &tiling_type) const {
  std::vector<std::string> func_impls;
  for (const auto &container : model_info_.container_exprs) {
    func_impls.emplace_back(GenFuncImpl(container, tiling_type, model_info_.variable_expr_map));
  }
  for (const auto &container : model_info_.tensor_exprs) {
    func_impls.emplace_back(GenFuncImpl(container, tiling_type, model_info_.variable_expr_map));
  }
  return func_impls;
}

std::string MemoryTilingDataGen::GetTilingFuncInvoke() const {
  std::string func_invoke;
  for (const auto &container : model_info_.container_exprs) {
    func_invoke += GenFuncInvoke(container.first);
  }
  for (const auto &container : model_info_.tensor_exprs) {
    func_invoke += GenFuncInvoke(container.first);
  }
  return func_invoke;
}

std::vector<std::string> TilingDataGenerator::GetTilingFuncImpl(const uint32_t tiling_key,
                                                                const TilingDataGenType tiling_data_gen_type) const {
  // gen axes tiling func impl
  std::vector<std::string> impl_codes;
  std::unordered_set<std::string> uniq_codes;
  for (const auto &tiling_data_gen : GetTilingDataGens(tiling_key)) {
    if ((tiling_data_gen_type == TilingDataGenType::ALL_TILING_DATA_GEN) ||
        ((tiling_data_gen->GetTilingDataGenType() == tiling_data_gen_type) && (IsGenEnable(tiling_data_gen_type)))) {
      const auto &impl_code = tiling_data_gen->GetTilingFuncImpl(extra_info_config_.tiling_data_type_name);
      std::copy_if(impl_code.begin(), impl_code.end(), std::back_inserter(impl_codes),
                   [&uniq_codes](const std::string &code) { return uniq_codes.insert(code).second; });
    }
  }
  return impl_codes;
}

std::string TilingDataGenerator::GetTilingFuncInvoke(const uint32_t tiling_key,
                                                     const TilingDataGenType tiling_data_gen_type) const {
  std::string invoke_code;
  for (const auto &tiling_data_gen : GetTilingDataGens(tiling_key)) {
    if ((tiling_data_gen_type == TilingDataGenType::ALL_TILING_DATA_GEN) ||
        ((tiling_data_gen->GetTilingDataGenType() == tiling_data_gen_type) && (IsGenEnable(tiling_data_gen_type)))) {
      invoke_code.append(tiling_data_gen->GetTilingFuncInvoke());
    }
  }
  return invoke_code;
}

af::Status TilingDataGenerator::Init() {
  // generate axes tiling data, such as loop num, tail size, tail block args.
  if (inited_) {
    GELOGI("Tiling data has been inited.");
    return af::SUCCESS;
  }
  for (const auto &model_info : model_info_list_) {
    GE_ASSERT_SUCCESS(GenTilingData(model_info), "Generate tiling data failed, tiling_key[%u].",
                      model_info.tiling_case_id);
  }
  inited_ = true;
  return af::SUCCESS;
}

af::Status TilingDataGenerator::GenTilingData(const ModelInfo &model_info) {
  const auto tiling_key = model_info.tiling_case_id;
  // Init tiling data gen for axes
  auto axes_tiling_data_gen = af::MakeShared<AxesTilingDataGen>(model_info);
  GE_ASSERT_NOTNULL(axes_tiling_data_gen, "Init AxesTilingDataGen failed, tiling_key[%u].", tiling_key);
  GE_ASSERT_SUCCESS(axes_tiling_data_gen->Init());
  graphs_tiling_data_gens_[tiling_key].emplace_back(axes_tiling_data_gen);

  // Init tiling data gen for BlockDimTilingDataGen
  auto block_tiling_data_gen = af::MakeShared<BlockDimTilingDataGen>(axes_tiling_data_gen, model_info);
  GE_ASSERT_NOTNULL(block_tiling_data_gen, "Init BlockTilingDataGen failed, tiling_key[%u].", tiling_key);
  GE_ASSERT_SUCCESS(block_tiling_data_gen->Init());
  graphs_tiling_data_gens_[tiling_key].emplace_back(block_tiling_data_gen);

  // Init tiling data gen for MemoryTilingDataGen
  auto memory_tiling_data_gen = af::MakeShared<MemoryTilingDataGen>(model_info);
  GE_ASSERT_NOTNULL(memory_tiling_data_gen, "Init MemoryTilingDataGen failed, tiling_key[%u].", tiling_key);
  GE_ASSERT_SUCCESS(memory_tiling_data_gen->Init());
  graphs_tiling_data_gens_[tiling_key].emplace_back(memory_tiling_data_gen);
  return af::SUCCESS;
}

std::vector<std::pair<std::string, std::string>> TilingDataGenerator::GetTilingDataWithAnnotation(
    const uint32_t tiling_key, const TilingDataGenType tiling_data_gen_type) const {
  std::vector<std::pair<std::string, std::string>> res;
  const auto &iter = graphs_tiling_data_gens_.find(tiling_key);
  if (iter == graphs_tiling_data_gens_.cend()) {
    return res;
  }
  for (const auto &tiling_data_gen : iter->second) {
    GELOGD("GetTilingDataWithAnnotation tiling_data_gen_type[%d] gen_type[%d]",
           static_cast<int32_t>(tiling_data_gen_type), static_cast<int32_t>(tiling_data_gen->GetTilingDataGenType()));
    if (((tiling_data_gen_type == TilingDataGenType::ALL_TILING_DATA_GEN) ||
         (tiling_data_gen->GetTilingDataGenType() == tiling_data_gen_type)) &&
        (IsGenEnable(tiling_data_gen_type))) {
      const auto &data_with_annotation = tiling_data_gen->GetTilingDataWithAnnotation();
      res.insert(res.end(), data_with_annotation.cbegin(), data_with_annotation.cend());
      break;
    }
  }
  return res;
}

std::vector<std::pair<std::string, std::string>> TilingDataGenerator::GetTilingDataWithAnnotation(
    const TilingDataGenType tiling_data_gen_type) const {
  std::vector<std::pair<std::string, std::string>> res;
  for (const auto &iter : graphs_tiling_data_gens_) {
    for (const auto &tiling_data_gen : iter.second) {
      GELOGD("GetTilingDataWithAnnotation tiling_data_gen_type[%d] gen_type[%d]",
             static_cast<int32_t>(tiling_data_gen_type), static_cast<int32_t>(tiling_data_gen->GetTilingDataGenType()));
      if ((tiling_data_gen->GetTilingDataGenType() == tiling_data_gen_type) && (IsGenEnable(tiling_data_gen_type))) {
        const auto &data_with_annotation = tiling_data_gen->GetTilingDataWithAnnotation();
        res.insert(res.end(), data_with_annotation.cbegin(), data_with_annotation.cend());
      }
    }
  }
  return res;
}

std::vector<TilingDataGenPtr> TilingDataGenerator::GetTilingDataGens(const uint32_t tiling_key) const {
  const auto &iter = graphs_tiling_data_gens_.find(tiling_key);
  if (iter != graphs_tiling_data_gens_.cend()) {
    return iter->second;
  }
  return std::vector<TilingDataGenPtr>{};
}

std::vector<std::pair<std::string, std::string>> TilingDataGenBase::GetTilingDataWithAnnotation() const {
  std::vector<std::pair<std::string, std::string>> tiling_data_names;
  for (const auto &tiling_data : tiling_data_map_) {
    std::string arg_annotation;
    const auto &annotation_iter = kTilingDataTypeToAnnotation.find(tiling_data.second.first);
    if (annotation_iter != kTilingDataTypeToAnnotation.cend()) {
      arg_annotation = annotation_iter->second;
    }
    tiling_data_names.emplace_back(tiling_data.first, arg_annotation);
  }
  return tiling_data_names;
}
}  // namespace att
