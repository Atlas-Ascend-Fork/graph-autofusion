/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_CODE_GEN_PREPROCESS_ARGS_MANAGER_H_
#define ATT_CODE_GEN_PREPROCESS_ARGS_MANAGER_H_
#include <map>
#include <vector>

#include "base/model_info.h"
#include "generator/preprocess/var_info.h"

namespace att {
class ArgsManager {
 public:
  explicit ArgsManager(const ModelInfo &model_info) : model_info_(model_info) {}
  ~ArgsManager() = default;
  uint32_t GetTilingCaseId() const;
  /**
   * @brief 解析model_info,可选择是否做变量替换
   * @return bool 如果轴的size等信息非空，且在设置变量替换前提下，变量替换成功，那么返回true
   */
  bool Process(bool do_var_replace = true);
  /**
   * @brief 做变量替换。要求需要先调用process，否则会返回false。
   */
  bool DoVarsReplace();
  /**
   * @brief 设置变量的取值。调用该接口，待搜索变量中会剔除vars中的变量
   * @return bool 返回设置变量的值是否成功
   */
  bool SetSolvedVars(const std::vector<Expr> &vars);
  /**
   * @brief 获取待搜索变量，带搜索变量不包含常量以及运行时输入参数
   * @return 输出，所有待运行时搜索变量
   */
  std::vector<Expr> GetSearchableVars() const;
  /**
   * @brief 按照给定的硬件参数获取相关的运行时待搜索参数。
   * @return 输出，硬件相关待搜索参数
   */
  std::vector<Expr> GetSearchableVars(const HardwareDef scope) const;
  /**
   * @brief 获取变量替换前后的对应关系，例如baseM要求128对齐，替换后变成baseM=aligned_baseM * 128
   * 该接口会返回aligned_baseM -- baseM的映射
   * @return key为替换后的符号，value为替换前的原始符号。如果不做变量替换，这里返回空
   */
  ExprExprMap GetVarsRelations() const;
  /**
   * @brief 获取变量替换前后的对应关系，例如baseM要求128对齐，替换后变成baseM=aligned_baseM * 128
   * 该接口会返回baseM -- aligned_baseM*128的映射
   * @return key为原始符号，value为替换表达式。如果不做变量替换，这里返回空
   */
  ExprExprMap GetExprRelations() const;
  /**
   * @brief 获取运行时输入参数
   * @return 运行时输入
   */
  std::vector<Expr> GetInputVars() const;
  /**
   * @brief 获取运行时输入参数
   * @return 运行时输入的有效值范围
   */
  std::vector<std::pair<Expr, std::pair<int64_t, int64_t>>> GetInputVarsRange() const;
  /**
   * @brief 获取常量变量
   * @return 所有常量变量, key:变量， value：变量的取值
   */
  ExprUintMap GetConstVars() const;
  /**
   * @brief 获取已求解变量，不包含const以及input变量
   */
  std::vector<Expr> GetSolvedVars() const;
  /**
   * @brief 获取给定带搜参数相关的硬件
   * @param var 输入， 带搜参数
   * @return vector 相关硬件
   */
  std::vector<HardwareDef> GetRelatedHardware(const Expr &var) const;
  /**
   * @brief 获取所有硬件约束表达式
   */
  std::map<HardwareDef, Expr> GetTotalHardwareCons(bool do_container_replace = false) const;
  /**
   * @brief 获取所有切分轴之间的不等式约束表达式
   */
  std::vector<Expr> GetTotalCutCons() const;
  /**
   * @brief 获取给定带搜参数的相关硬件
   * @param var 输入，带搜索参数
   * @return 与var相关的硬件
   */
  Expr GetUsedHardwareInfo(const HardwareDef scope) const;
  /**
   * @brief 获取目标表达式，min(max(pipe))
   * @return 返回目标表达式
   */
  std::map<PipeType, Expr> GetObjectFunc() const;
  /**
   * @brief 获取待搜参数的祖先轴的符号表达式
   * @param var 输入，带搜索参数
   * @return std::vector<Expr> 祖先轴的size
   */
  std::vector<Expr> GetAncestor(const Expr &var) const;
  /**
   * @brief 获取待搜参数的祖先轴的名字
   * @param var 输入，带搜索参数
   * @return std::vector<Expr> 祖先轴的名字
   */
  std::vector<std::string> GetAncestorNames(const Expr &var) const;
  /**
   * @brief 获取给定变量的最大值表达式
   * @param var 输入，待搜索参数
   * @return 输出，最大值表达式
   */
  Expr GetMaxValue(const Expr &var) const;
  /**
   * @brief 获取给定变量的最小值表达式
   * @param var 输入，待搜索参数
   * @return 输出，最小值表达式
   */
  Expr GetMinValue(const Expr &var) const;
  /**
   * @brief 获取给定待搜参数的默认初始值表达式，通常切分轴的给最内轴大小初始值为1，最内轴初始值为原始值大小
   * @param var 输入，带搜索参数
   * @return 输出，初始值表达式
   */
  Expr GetDefaultInitValue(const Expr &var) const;
  /**
   * @brief 获取var的对齐值。
   */
  Expr GetVarAlignValue(const Expr &var) const;
  /**
   * @brief 获取var的建议对齐值。
   */
  uint32_t GetVarPromptAlignValue(const Expr &var) const;

  /**
   * @brief 获取类型占用大小
   */
  uint32_t GetDataTypeSizeVar(const Expr &var) const;
  /**
   * @brief 获取是否是concat外轴
   */
  bool IsConcatOuterDim(const Expr &var) const;
  /**
   * @brief 获取是否是concat内轴
   */
  bool IsConcatInnerDim(const Expr &var) const;
  /**
   * @brief 获取var的父亲轴，原始轴的父亲轴为空,如果某根轴来源于多根轴合并而来，这里返回多个父亲节点。
   */
  std::vector<Expr> GetParentVars(const Expr &var) const;
  /**
   * @brief 获取处于node最内轴的size表达式
   */
  std::vector<Expr> GetNodeInnerestDimSizes() const;
  /**
   * @brief 获取tensor表达式
   */
  const ExprExprMap &GetContainerMap() const;
  /**
   * @brief 获取tensor名称
   */
  const std::map<Expr, std::string, ExprCmp> &GetContainerNames() const;
  /**
   * @brief 获取执行次数表达式替换表
   */
  std::vector<std::pair<Expr, Expr>> GetTernaryOpReplaceVars() const;
  /**
   * @brief 获取执行次数表达式参数集
   */
  std::map<Expr, std::vector<Expr>, ExprCmp> GetTernaryOpRelatedVars() const;
  /**
   * @brief 获取 TernaryOp 映射（用于获取描述信息）
   */
  const std::map<Expr, TernaryOp, ExprCmp> &GetTernaryOps() const;
  /**
   * @brief 获取性能公式语义化拆解信息
   */
  const std::vector<PerfBreakdownGroup> &GetPerfBreakdowns() const;
  /**
   * @brief 获取多核头开销
   */
  Expr GetHeadCost() const;
  /**
   * @brief 获取轴优先级信息
   */
  ExprUintMap GetAxesPriority() const;
  ExprUintMap GetAxesOrder() const;
  /**
   * @brief 获取Model Info
   */
  const ModelInfo &GetModelInfo() const;

 private:
  /**
   * @brief 对model_info中的信息进行解析，提取出可替换的变量，并对model_info中的表达式进行变量替换
   * @param replaced_vars 参数输出 返回替换后的符号变量与替换前的符号变量的映射
   * @param replacements 参数输出 保存替换前符号变量与替换后符号表达式的映射
   * @return bool 如果变量替换成功，则返回true，否则返回false
   */
  bool ReplaceVars(ExprExprMap &replaced_vars, ExprExprMap &replacements, ExprExprMap &new_expr_replacements);
  /**
   * @brief 将类的成员变量置空
   */
  void Reset();
  /**
   * @brief 设置原始变量的替换信息，并创建替换后变量的信息
   * @param replaced_vars 参数输出 返回替换后的符号变量与替换前的符号变量的映射
   * @param replacements 参数输出 保存替换前符号变量与替换后符号表达式的映射
   */
  bool UpdateVarInfos(const ExprExprMap &replaced_vars, const ExprExprMap &replacement,
                      const ExprExprMap &new_expr_replacements);
  /**
   * @brief 设置原始model info中的信息，例如变量的最大值初始值等
   */
  void SetOrigExprs();
  bool SetNewVarInfoAttrs(const Expr &old_var, const ExprExprMap &replacement, const ExprExprMap ori_to_new_vars_map,
                          const ExprExprMap local_new_expr_replacements, VarInfo &new_var_info);
  Expr GetNewExprMaxValueReplaced(const Expr &ori_expr, const Expr &max_value);
  Expr GetNewExprInitValueReplaced(const Expr &new_var);
  static VarInfo &SetSizeInfo(VarInfo &info, const SymVarInfoPtr &var_info, const AttAxis *arg_axis);
  static VarInfo &SetInitSize(VarInfo &info, const bool is_last);
  static VarInfo GetNaiveVarInfo(const AttAxis *arg_axis);
  static ExprInfoMap GetOrigVarInfos(const ModelInfo &model_info);
  static void ReplaceNewExpr(ExprExprMap &new_expr_replacements);
  bool replacement_done_{false};
  ExprInfoMap vars_infos_;
  std::map<HardwareDef, Expr> hardware_cons_;       // 硬件相关约束
  std::vector<Expr> cut_leq_cons_;                  // 变量替换以及花间之后的不等式
  std::vector<std::pair<Expr, Expr>> cut_eq_cons_;  // 变量替换以及化简之后的等式约束
  std::map<PipeType, Expr> objs_;
  const ModelInfo &model_info_;
  std::vector<Expr> solved_vars_;
  ExprExprMap ori_var_init_values_;
  ExprExprMap ori_var_max_values_;
  ExprExprMap ori_var_align_values_;
  ExprExprMap replaced_var_init_values_;
  std::map<Expr, TernaryOp, ExprCmp> ternary_op_;
  std::vector<PerfBreakdownGroup> perf_breakdowns_;
};
bool GetNewVarsInExpr(const Expr &expr, const ExprExprMap &new_expr_replacements, std::vector<Expr> &expr_args);
bool SplitVars(const AttAxisPtr &arg_axis, ExprInfoMap &var_infos);
}  // namespace att
#endif  // ATT_CODE_GEN_PREPROCESS_ARGS_MANAGER_H_
