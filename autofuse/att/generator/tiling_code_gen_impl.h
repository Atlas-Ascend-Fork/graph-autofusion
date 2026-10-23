/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_CODE_GEN_IMPL_H_
#define ATT_TILING_CODE_GEN_IMPL_H_

#include <string>
#include <set>
#include <memory>
#include <map>
#include <tuple>
#include <utility>
#include "code_printer.h"
#include "common/tiling_source_dependencies.h"
#include "base/model_info.h"
#include "generator_config.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "extra_info_gen/extra_info_generator.h"
#include "util/duration.h"
#include "gen_model_info/api_tiling_gen/gen_api_tiling.h"
#include "cache/operator_level_cache_gen.h"
#include "cache/group_level_cache_gen.h"

namespace att {
struct GenTilingTailImplExtParams {
  std::unordered_map<std::string, std::string> cache_reuse_info = {};
  VarRelations var_relations = {};
  EnableGroupParallels enable_group_parallels = {};
  TensorIdSet workspace_tensor_id_set = {};

  GenTilingTailImplExtParams() = default;
  GenTilingTailImplExtParams(std::unordered_map<std::string, std::string> cache_reuse_info_,
                             VarRelations var_relations_, EnableGroupParallels enable_group_parallels_,
                             TensorIdSet workspace_tensor_id_set_)
      : cache_reuse_info(std::move(cache_reuse_info_)),
        var_relations(std::move(var_relations_)),
        enable_group_parallels(std::move(enable_group_parallels_)),
        workspace_tensor_id_set(std::move(workspace_tensor_id_set_)) {}
};

class TilingCodeGenImpl {
  // asc_graph_id->impl_graph_id->schedule_result_id->(score_func_name, score_func_impl)
  using AscGraphNamepspaceMap = std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>;
  using FusedGraphNamespaceMap = std::map<size_t, AscGraphNamepspaceMap>;

 public:
  void SetFinalTilingObservationMetadata(const size_t slot, const size_t slot_count,
                                         const std::map<std::string, int32_t> &sub_case_ids);
  TilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                    const TilingModelInfo &tiling_model_info, const ScoreFuncs &score_funcs, const bool is_uniq_group);
  virtual ~TilingCodeGenImpl() = default;

  af::Status GenTilingHead(std::map<std::string, std::string> &tiling_res,
                           const EnableGroupParallels &enable_group_parallels = {});
  af::Status GenTilingTail(std::map<std::string, std::string> &tiling_res, GenTilingTailImplExtParams ext_params = {});
  af::Status GenTiling(std::map<std::string, std::string> &tiling_res,
                       std::unordered_map<std::string, std::string> cache_reuse_info = {}, uint32_t cache_capacity = 0,
                       const EnableGroupParallels &enable_group_parallels = {});
  af::Status FinishGeneratedHeaders(std::map<std::string, std::string> &tiling_res);
  static af::Status FinishGeneratedHeaders(
      const std::map<autofuse::GeneratedHeaderId, autofuse::GeneratedCode> &generated_headers,
      const std::string &tiling_data_type_name, bool is_autofuse, std::map<std::string, std::string> &tiling_res);
  const std::map<autofuse::GeneratedHeaderId, autofuse::GeneratedCode> &GetGeneratedHeaders() const {
    return atomic_headers_;
  }

  // 设置每个ScheduleResult的Group个数
  void SetScheduleResultGroupNums(const std::map<std::pair<size_t, size_t>, size_t> &group_nums) {
    schedule_result_group_nums_ = group_nums;
  }

  // 获取当前ScheduleResult的Group个数，用于并发性能优化
  uint32_t GetGroupNumForCurrentScheduleResult(const std::pair<size_t, size_t> &schedule_result_key) const;

 protected:
  void AddAtomicHeaderLine(autofuse::GeneratedHeaderId header_id, const std::string &line);
  void AppendAtomicHeaderBody(autofuse::GeneratedHeaderId header_id, const std::string &body);
  void ResetTranslationUnit();
  void RequireTranslationUnitSystemHeader(const std::string &header);
  void RequireVarRelationSystemHeaders();
  void RequireTranslationUnitExternalHeader(const std::string &header);
  void RequireTranslationUnitGeneratedHeader(autofuse::GeneratedHeaderId header_id);
  void RequireTranslationUnitTilingDataHeader();
  void AddApiTilingPreambles();
  af::Status FinishTranslationUnit(const std::string &key, std::map<std::string, std::string> &tiling_res);
  // 用于判断求解器是否有效
  af::Status CheckImplPtr(const std::string &indent);
  af::Status GetReuseVarNames(std::map<std::string, std::string> &var_names_to_reuse_var_name);
  // 用于构造一个用于复制的结构体
  af::Status GenStructCopyDef();
  af::Status CollectStructCopyVars(std::set<std::string> &tiling_data_vars);
  // 用于构造一个用于缓存复用的哈希表
  af::Status GenCacheHashMapDef();

  // 用于生成duration相关的代码段
  af::Status GenDurationBeginCode(const TilingFuncDurationType type, const std::string &indent);
  af::Status GenDurationEndCode(const TilingFuncDurationType type, const std::string &indent);

  // schedule group相关
  af::Status ObtainInnerParams(std::map<std::string, std::set<std::string>> &hardware_map,
                               FusedGraphNamespaceMap &namespace_map);
  // 生成sche group的tiling函数初始化部分
  af::Status GenGetTilingForAllInitLines(bool pgo = false);
  af::Status GenGetResultSummary(const size_t asc_graph_id);
  // 生成sche group的tiling函数，不支持工具场景
  af::Status GenGetTilingForScheduleResult();
  af::Status GenGetTilingForAllSchedulesResults(const uint32_t asc_graph_id,
                                                const AscGraphNamepspaceMap &asc_graph_map);
  af::Status GenFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map);
  af::Status GenFusedScheduleResultsGetTilingCoreDefine(const FusedGraphNamespaceMap &namespace_map);
  af::Status GenFusedScheduleResultsCoreSelection(const FusedGraphNamespaceMap &namespace_map);
  af::Status GenFusedScheduleResultsGetTilingObserveWrapper(const FusedGraphNamespaceMap &namespace_map);
  void GenFinalTilingSummaryForResult(size_t asc_graph_id, size_t result_id,
                                      const std::map<size_t, std::pair<std::string, std::string>> &groups);
  af::Status GenEnableGroupParallelFunctions(const FusedGraphNamespaceMap &namespace_map);
  af::Status GenEnableGroupParallelInvoke(size_t asc_graph_id, const AscGraphNamepspaceMap &asc_graph_namespace_map);
  af::Status GenEnableGroupParallelPgoInvoke(const std::string &tiling_name, bool is_pointer, const std::string &indent,
                                             std::string &invoke_code);
  af::Status GenPGOFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map);
  af::Status GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map);
  af::Status GenPGOByCoreNumSearchTilingKeyCollectTilingData(FusedGraphNamespaceMap namespace_map);
  void GenGetScoreFuncs(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map);
  af::Status GenPGOGetTilingForAll();
  void GenGetScoreFuncsCalling(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map);
  // 生成sche group的cache初始化部分
  void GenCacheInit();
  void GenSetHardwareCodes(const std::string &group_prefix, const std::set<std::string> &hardware_names);
  af::Status GenScheduleGroupDoTiling(std::string &check_cond, const std::string &hardware_param,
                                      const std::string &schedule_result_prefix);
  void GenGetScheduleResultTail(const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  void GenUpdateWorkspace(const size_t asc_graph_id, const size_t impl_graph_id);
  void GenWorkspaceOffsetHelpers();
  void GenWorkspaceOffsetReset(const std::string &tiling_data_name);
  void GenWorkspaceOffsetFinalize(const std::string &tiling_data_name);
  // 生成DoGroupTiling公共函数，支持首次Tiling和二次Tiling
  af::Status GenDoGroupTilingFunction(const size_t asc_graph_id, const size_t impl_graph_id,
                                      const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  // 辅助函数：生成perf计算和更新部分
  af::Status GenGetScheduleResultPerfAndTail(const size_t asc_graph_id, const size_t impl_graph_id,
                                             const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  af::Status GenGetScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id,
                                  const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                  const std::map<std::string, std::set<std::string>> &hardware_map);
  af::Status GenUpdatePerf(const size_t asc_graph_id, const size_t impl_graph_id,
                           const std::vector<std::string> &groups_perf,
                           const std::vector<std::pair<size_t, size_t>> &serialized_edges,
                           const std::vector<std::string> &groups_block_num,
                           const std::vector<std::string> &assign_max_block_num);
  void GenGetMaxScoreIndex(const AscGraphNamepspaceMap &namespace_map);

  void GenScheduleResultGetTilingCalling(const std::string &index, const std::string &ident = "");
  af::Status GenGetAllSchedulesResults(const AscGraphNamepspaceMap &namespace_map);
  void GenPGOUpdateTilingInfo(const size_t asc_graph_id, const size_t impl_graph_id,
                              const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  void GenFillOtherGroupsGetTiling(const size_t asc_graph_id, const size_t impl_graph_id,
                                   const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                   const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
                                   const std::map<std::string, std::set<std::string>> &hardware_map);
  af::Status GenPGOGetScheduleResultPerGroup(const size_t asc_graph_id, const size_t impl_graph_id,
                                             const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                             const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
                                             const std::map<std::string, std::set<std::string>> &hardware_map);
  af::Status GenPGOScheduleGroupSearchEntry(const size_t asc_graph_id, const size_t impl_graph_id,
                                            const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                            const std::map<std::string, std::set<std::string>> &hardware_map,
                                            const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
                                            const std::string &result_name);
  af::Status GenPGOGetScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id,
                                     const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                     const std::map<std::string, std::set<std::string>> &hardware_map);
  void GenPGOGetAllSchedulesResults(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map);
  void GenPGOByCoreNumGetScheduleResult(
      const size_t asc_graph_id, const size_t impl_graph_id,
      const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
      const std::map<std::string, std::set<std::string>> &hardware_map,
      const std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> &var_relation);
  std::string GenLaunchLikeInputOutputDef(bool is_define = true);
  std::string GenPgoTensorArgsDef(bool is_define = true) const;
  void GenPGOMultiGroupBlockDimList(const FusedGraphNamespaceMap &namespace_map, std::string &block_dim_list_arg);
  af::Status GenCastReuseTilingDataCode(const ReuseScheduleGroupInfo &reuse_info, const ReuseScheduleGroupInfo &info);
  bool IsScheduleResultEnableParallel(const size_t asc_graph_id, const size_t impl_graph_id) const;
  bool GenUpdateCurPerfAndBlockByGroupIfNeeded(const size_t asc_graph_id,
                                               const AscGraphNamepspaceMap &asc_graph_map) const;
  // -----------------------小shape优化相关---------------------------
  bool HitSmallShapePattern(ArgsManager &args_manager) const;
  // 生成GetTiling的PGO接口函数
  af::Status GenGetTilingWithCaseId(bool is_tail = false);

  // 辅助函数：获取GetTiling函数的参数定义字符串
  std::string GetGetTilingParamDefines(bool use_cache, bool use_workspace, std::string &cache_define_head,
                                       std::string &cache_define_func, std::string &cache_used) const;
  // 辅助函数：生成GetTiling函数签名
  void GenGetTilingFunctionSignature(const std::string &workspace_define, const std::string &cache_define_func,
                                     const std::string &cache_define_head);
  // 辅助函数：生成GetTiling函数体（GetTilingKey调用和返回逻辑）
  af::Status GenGetTilingFunctionBody(bool use_cache, bool is_tail, const std::string &cache_used);
  void GenSingleGroupFinalTilingObserveBegin();
  void GenSingleGroupFinalTilingObserveReset();
  void GenSingleGroupFinalTilingObserveEmit(bool need_operator_cache);
  af::Status GenSingleGroupGetTilingCoreAlias();
  // 辅助函数：生成GetTilingKey调用逻辑
  af::Status GenGetTilingKeyCall(const std::string &cache_used);
  // 辅助函数：生成duration代码（begin或end）
  af::Status GenDurationCode(bool is_begin);
  // 辅助函数：生成operator level cache保存逻辑
  af::Status GenOperatorCacheSaveCode(bool need_operator_cache);
  // 校验 force_tiling_case 配置
  af::Status ValidateForceTilingCase(const std::map<string, int32_t> &group_tiling_case_ids,
                                     int32_t min_tiling_case_size) const;
  af::Status ValidateSingleModeForceTilingCase(int32_t min_tiling_case_size) const;
  af::Status ValidateGroupModeForceTilingCase(const std::map<string, int32_t> &group_tiling_case_ids) const;

  // 生成硬件信息的日志语句
  af::Status GenHardwareSummary(const ModelInfo &model_info);
  // 生成硬件信息的判断语句
  af::Status GenHardwareJudge(const ModelInfo &model_info);
  // 生成输入信息的日志语句
  af::Status GenInputSummary(const ModelInfo &model_info);
  // 生成score函数
  af::Status GenCalcScore(const ModelInfo &model_info);
  // 生成score计算相关变量
  void GenCalcScoreVarsDefine();
  // 生成所有tiling case的score计算逻辑
  af::Status GenAllSameScoreTilingCases(std::map<std::string, std::vector<const ModelInfo *>> &same_args_name_to_graphs,
                                        const std::vector<std::string> &ordered_assemble_args_name);
  // 生成ub/corenum相关的tiling的上限值
  void InitTilingUpperBound(const std::vector<Expr> &hardware_args, const ArgsManager &args_manager,
                            const HardwareDef &hardware_def, std::map<std::string, bool> &visited);
  af::Status GenSmallShapeTiling(const ModelInfo &model_info);
  // 生成求解器的基类
  virtual af::Status GenSolverBaseClass() = 0;
  // 生成由solver_pass_gen构造的求解器子类
  virtual af::Status GenSolverTiling(const ModelInfo &model_info) = 0;
  // 调用求解器的函数
  virtual af::Status GenDoTilingCommon(const ModelInfo &model_info, const std::pair<std::string, std::string> &codes);
  virtual af::Status GenDoTiling(const ModelInfo &model_info) = 0;
  // 获取tiling data拷贝
  virtual af::Status GenGetTilingDataFromCopy();
  // 缓存复用
  virtual af::Status GenFindCacheAndSaveCache();
  // 更新最优模板
  virtual af::Status GenUpdateBetterTiling();
  // 根据目标表达式和ub占用率选择更好的模板
  virtual af::Status GenSelectBetterTilingBasedOnObjAndUbRatio();
  // 寻找最优模板，enable_group_parallel_optimize 表示是否启用 Group 并发性能优化，group_num 表示当前 ScheduleResult
  // 包含的 Group 数量
  virtual af::Status GenFindPerfBetterTilingbyCaseId(bool enable_group_parallel_optimize = false,
                                                     bool add_core_num_param = false, uint32_t group_num = 1);
  virtual af::Status GenSearchAllTilingbyCaseId();
  // 多模板情况下算法的模板选择逻辑
  virtual af::Status GenGetTilingKey();
  af::Status GenFinalTilingEmitter();
  void GenFinalTilingSemanticFallback();
  af::Status GenFinalTilingContextQuery();
  af::Status GenFinalTilingByCaseEmitter();
  af::Status CollectFinalTilingReprFields(std::set<std::string> &fields);
  af::Status GenFinalTilingReprFunction();
  af::Status GenFinalTilingRepresentation();
  virtual af::Status GenPGOSearchTilingKey();
  void GenPGOSearchTilingKeyUniqGroupBatch();
  virtual af::Status ValidateSingleResultAndGroup(bool need_free_memory = false);
  // 根据caseid生成选择逻辑
  virtual af::Status GenGetTilingbyCaseId();
  virtual af::Status GenPGODefaultTiling();
  virtual af::Status GenPGOTilingCase(const ModelInfo &model_info);
  virtual af::Status GenPGOGetTilingbyCaseId();
  virtual af::Status GenerateInputParamsAndTiling();
  virtual af::Status GenPGOByCoreNumSearchTilingKeySingleGroup();
  virtual af::Status GenPGOByCoreNumSearchTilingKey();
  virtual af::Status GenPGOByCoreNumTilingForAll();
  void GenPGOByCoreNumDoTiling(const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
                               const uint32_t group_index, const size_t asc_graph_id, const size_t impl_graph_id);
  void GenPGOByCoreNumFunctionHead(size_t impl_graph_id);
  bool TryGenPGOByCoreNumReuseTiling(size_t asc_graph_id, size_t impl_graph_id, size_t group_id, uint32_t group_index);
  void GenPGOByCoreNumGetAllSchedulesResults(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map);
  // 该函数用于构造tiling data的set与get内容
  virtual af::Status GenExtraParamCode(const ModelInfo &model_info, std::string &pass_code);
  virtual af::Status GenGetSetTilingImpl(const ModelInfo &model_info);
  // 在tiling data头文件中生成外部函数的定义
  virtual af::Status GenExternFuncDef();
  // 生成宏函数与include信息
  virtual af::Status GenMacroInclude();
  void GenPgoHeaderCodesTail();
  void GenPgoCallbackDefs(ge::CodePrinter &pgo_header);
  void GenPgoConfigDefs(ge::CodePrinter &pgo_header);
  // 生成工具函数
  virtual af::Status GenToolFuncs();
  // 生成tilingimpl的基类public函数
  virtual af::Status GenTilingImplPublicFunc();
  // 生成基类中GetTilingData/SetTilingData/SetWorkspaceSize的虚函数默认实现
  af::Status GenVirtualDataTransferFuncs();
  // 生成求解器子类
  virtual af::Status GenTilingCaseImpl(const ModelInfo &model_info);
  void GenInductorExecutePGOSolver(const ModelInfo &model_info);
  // 生成预处理函数
  virtual af::Status GenPreTiling(const ModelInfo &model_info);
  // 生成高阶api的tiling
  virtual af::Status GenDoApiTiling(const ModelInfo &model_info);
  // 提供tiling data的额外性能评估函数
  virtual af::Status GenExtraEvalFunc(const ModelInfo &model_info);
  // 生成基于基本tiling参数计算其他参数的逻辑，如外轴大小等
  virtual af::Status GenExtraTilingData(const ModelInfo &model_info);
  // 生成tiling评估打印
  virtual af::Status GenExtraSummaryInfo(const ModelInfo &model_info, const ArgsManager &args_manager,
                                         std::string &case_info_str);
  // 生成不同pipe的obj
  virtual af::Status GenPipeTypeObj(const ModelInfo &model_info);
  // 该函数用于生成memory tiling的相关参数
  virtual af::Status GenMemoryParamCode(const ModelInfo &model_info);
  virtual af::Status GenExtraTilingFuncImpl(const ModelInfo &model_info);
  virtual af::Status GenExtraTilingFuncInvoke(const ModelInfo &model_info);
  // 生成硬件占用的评估函数
  virtual af::Status GenHardwareCons(const ModelInfo &model_info);
  // 生成目标函数
  virtual af::Status GenGetObj(const ModelInfo &model_info);

  // 辅助函数：生成ArrangeBlockOffsets函数声明
  void GenArrangeBlockOffsetsDeclarations(const FusedGraphNamespaceMap &namespace_map);
  // 辅助函数：生成DoGroupTiling函数的调用GetTiling部分
  void GenDoGroupTilingGetTilingCalls(const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  // 辅助函数：生成DoGroupTiling函数的失败处理部分
  void GenDoGroupTilingFailureHandler(const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  // 辅助函数：生成Group并行场景的首次Tiling部分
  void GenGroupParallelFirstTiling(const size_t impl_graph_id);
  // 辅助函数：生成Group并行场景的二次Tiling部分
  void GenGroupParallelSecondTiling(const size_t impl_graph_id,
                                    const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  // 辅助函数：生成Group并行首次Tiling后的声明和计算
  void GenGroupParallelFirstTilingDecls(const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  // 辅助函数：生成单Group场景的tiling处理
  af::Status GenSingleGroupScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id,
                                          const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                          const std::map<std::string, std::set<std::string>> &hardware_map);
  // 辅助函数：生成perf更新代码（消除GenUpdatePerf和GenGetScheduleResultPerfAndTail的重复）
  static std::string GenPerfUpdateCode(const std::vector<std::string> &groups_perf,
                                       const std::vector<std::string> &groups_block_num, const std::string &indent);
  std::vector<std::pair<size_t, size_t>> GetSerializedGroupEdges(
      const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
      const std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> &var_relation,
      bool enable_group_parallel) const;
  std::vector<std::pair<size_t, size_t>> GetWorkspaceSharedGroupEdges(
      const size_t asc_graph_id, const size_t impl_graph_id,
      const std::map<size_t, std::pair<std::string, std::string>> &graph_info) const;
  std::string GenSerializedEdgePenalty(const std::vector<std::string> &groups_perf,
                                       const std::vector<std::pair<size_t, size_t>> &serialized_edges,
                                       const std::string &indent) const;
  // 辅助函数：生成best perf更新代码
  void GenBestPerfUpdateCode(const size_t asc_graph_id, const size_t impl_graph_id,
                             const std::vector<std::string> &assign_max_block_num, const std::string &indent,
                             const std::string &objective_log = {});

  // 为所有group生成cache line冲突检测helper函数
  af::Status GenConflictGroupHelpers(const size_t asc_graph_id, const size_t impl_graph_id,
                                     const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  // 为单个group生成cache line冲突检测helper函数
  af::Status GenConflictGroupHelper(const ModelInfo &model_info, const std::string &group_item_prefix);
  // 生成冲突helper的调用代码
  std::string GenConflictGroupInvoke(const size_t asc_graph_id, const size_t impl_graph_id, size_t group_id,
                                     const std::string &group_item_prefix) const;
  // 生成冲突表达式上下文代码（返回{code, ok}）
  std::pair<std::string, bool> GenConflictExprContextCode(const ModelInfo &model_info, const ge::Expression &expr,
                                                          std::set<std::string> &declared_symbols) const;
  // 生成混合perf聚合更新代码（含冲突标志）
  static std::string GenMixedPerfUpdateCode(const std::vector<std::string> &groups_perf,
                                            const std::vector<std::string> &groups_block_num,
                                            const std::vector<std::string> &groups_conflict_flags,
                                            const std::string &indent);

  ge::CodePrinter tiling_data_;
  ge::CodePrinter tiling_func_;
  ge::CodePrinter tiling_head_;
  std::map<autofuse::GeneratedHeaderId, autofuse::GeneratedCode> atomic_headers_;
  autofuse::GeneratedCode translation_unit_;
  std::string op_name_;
  TilingCodeGenConfig config_;
  ExtraInfoConfig extra_info_config_;
  TilingDataGenerator tiling_data_manager_;
  ExtraInfoGenerator extra_info_generator_;
  const TilingModelInfo &tiling_model_info_;
  bool is_uniq_group_{true};  // 表示是否是唯一的ScheduleGroup，大部分场景不会切分成多个ScheduleGroup，所以默认为true
  bool hardware_has_ub_{
      false};  // 表示model_info的hardware_cons中是否包含UB，如果包含的话在选择模板时同时考虑目标表达式和UB占用率
  // schedule result打分函数, 当前支持ScheduleResult的选择，考虑未来支持其他级别的打分选择
  ScoreFuncs score_funcs_;
  std::unordered_map<std::string, std::string> cache_reuse_info_{};
  VarRelations var_relations_{};
  EnableGroupParallels enable_group_parallels_{};
  TensorIdSet workspace_tensor_id_set_{};
  uint32_t cache_capacity_{0};
  bool with_reuse_info_{false};
  std::string arrange_code_;
  size_t final_tiling_observation_slot_{0UL};
  size_t final_tiling_observation_slot_count_{1UL};
  std::map<std::string, int32_t> final_tiling_sub_case_ids_{};
  // 存储每个ScheduleResult的Group个数：(asc_graph_id, impl_graph_id) -> group_num
  std::map<std::pair<size_t, size_t>, size_t> schedule_result_group_nums_;

  // 缓存代码生成器
  std::unique_ptr<cache::OperatorLevelCacheGen> operator_level_cache_gen_;
  std::unique_ptr<cache::GroupLevelCacheGen> group_level_cache_gen_;

 private:
  // 判断CacheLineConfig是否为需要参与冲突检测的DMA方向
  static bool IsConflictCacheLineConfig(const CacheLineConfig &cfg);

  af::Status GenExpressionMacro();
  // 用于获取不同硬件信息的获取代码
  af::Status GetRelatedHardware(std::map<std::string, std::string> &hardware_info);

  // 用于生成duration相关的代码段
  af::Status GenDurationCommonCode();
  af::Status GenDurationPrintCode(const std::string &indent);
  af::Status GenDurationClearCode(const std::string &indent);

  // 辅助函数：生成FindPerfBetterTilingbyCaseId的子部分
  static std::string GenPerformanceAdjustmentCode(bool enable_group_parallel_optimize, bool add_core_num_param,
                                                  uint32_t group_num, bool is_uniq_group);
  static std::string GenLogOutputCodeWithUb(const bool is_uniq_group);
  af::Status GenFindPerfBetterTilingbyCaseIdWithUb(bool enable_group_parallel_optimize, bool add_core_num_param,
                                                   uint32_t group_num, bool is_uniq_group);
  af::Status GenFindPerfBetterTilingbyCaseIdWithoutUb(bool enable_group_parallel_optimize, bool add_core_num_param,
                                                      uint32_t group_num);

  // -----------------------以下函数生成tilingdata------------------------
  af::Status GenProtectedVars();
  af::Status GenBaseTilingData(std::map<std::string, std::string> &type_name_to_definition);
  af::Status GenHeaderCodesHead();
  af::Status GenHeaderCodesTail();
  af::Status GenHeaderCodesBody();
  af::Status GenHeaderCodesSummaryBody();
  af::Status GenHeaderInclude();
  af::Status GenHeaderVarsDef();
  af::Status GenScheduleGroupTilingHead();
  af::Status GenScheduleGroupTilingTail();

  // -----------------------以下函数生成tilingimpl的基类------------------------
  af::Status GenGetTiling();
  af::Status GenTilingImplBaseClass();

  // 生成公共框架代码，不同类型的求解器可以自行构建基类及求解器信息
  af::Status GenCommonFrameWork();
  // 生成公共框架结构体定义，不同类型求解均需要使用
  af::Status GenCommonStruct();

  // 生成性能评估函数
  af::Status GenEvalFunc(const ModelInfo &model_info);
  // 生成TilingSummary函数
  af::Status GenTilingSummary(const ModelInfo &model_info);
  // 生成后处理函数
  af::Status GenPostTiling(const ModelInfo &model_info);

  // 由tilingkey获取对应的tilingimpl的指针
  af::Status GenImplPtr();
  af::Status GenTemplateSemanticInfos();
  // 由tilingkey获取对应的性能公式
  af::Status GenGetPerf();
  // 由tilingkey获取对应的日志信息
  af::Status GenGetSummary();
  af::Status GenReuseGroupTilingWrapperGetTiling(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  af::Status GenReuseGroupTilingWrapperGetPerf(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  void GenReuseGroupTilingWrapperFinalObservation(const std::string &reuse_prefix);
  void GenReuseGroupSnapshotMapping();
  af::Status GenReuseGroupTilingWrapperFinalEmitter(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  af::Status GenReuseGroupTilingWrapperGetSummary(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  af::Status GenReuseGroupTilingWrapperGetTemplateSemanticInfos(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  af::Status GenReuseGroupTilingWrapperTemplateIdentity(const std::string &object_name, bool emit_status);
  void GenReuseGroupSelectedTemplateIdentity(const std::string &object_name);
  af::Status GenReuseGroupTilingWrapperSelectionIdentity();
  af::Status GenReuseGroupTilingWrapper(std::map<std::string, std::string> &tiling_res);
  af::Status GenPGOReuseGroupTilingWrapper();
  af::Status GenTilingKeyFunc();
  void GenTilingHeadMultiGroup();
  void InitTilingHeadCodeGeneration();
  void GenPgoSearchConfigDef();
  af::Status InitTilingGeneration(const std::unordered_map<std::string, std::string> &cache_reuse_info,
                                  uint32_t cache_capacity);
  void GenGroupNamespaceHead(const ScheduleGroupIdent &cur_ident);
  void RequireReuseGroupTranslationUnitHeaders();
  void RequireGroupTranslationUnitHeaders();
  af::Status FinishGroupTiling(const ScheduleGroupIdent &cur_ident, std::map<std::string, std::string> &tiling_res);

  // 辅助函数：从所有model info中收集输入变量名并返回数量
  size_t CollectInputVarsSize() const;

  // -----------------------生成固定的入口函数---------------------------
  af::Status GenGetTilingImpl();
  af::Status GenIsStaticShape();
  af::Status GenTilingFuncCallEntrance();
  af::Status GenGeneralTiling(const ModelInfo &model_info);

  af::Status GenVariableAnnotation(const ArgsManager &args_manager);

  // 辅助函数：生成Group级缓存查询代码
  af::Status GenGroupCacheLookupCode();
  // 辅助函数：生成模板迭代排序逻辑
  af::Status GenTemplateIterationLogic();

  af::Status GenOpLog(const std::string &indent, const std::string &log);
  af::Status GenOpLog(const std::string &indent, const std::string &uniq_log, const std::string &sched_log);
};

using TilingCodeGenImplPtr = std::shared_ptr<TilingCodeGenImpl>;
}  // namespace att
#endif
