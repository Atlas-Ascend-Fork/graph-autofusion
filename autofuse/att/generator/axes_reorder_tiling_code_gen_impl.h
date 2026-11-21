/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_AXES_REORDER_TILING_CODE_GEN_IMPL_H_
#define ATT_AXES_REORDER_TILING_CODE_GEN_IMPL_H_
#include <string>
#include "tiling_code_gen_impl.h"
#include "solver_pass_manager.h"
namespace att {
class AxesReorderTilingCodeGenImpl : public TilingCodeGenImpl {
 public:
  explicit AxesReorderTilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                                        const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                        const bool is_uniq_group)
      : TilingCodeGenImpl(op_name, config, model_infos, score_funcs, is_uniq_group) {}
  ~AxesReorderTilingCodeGenImpl() override = default;

 protected:
  af::Status GenSolverBaseClass() override;
  af::Status GenTilingImplPublicFunc() override;
  af::Status GenSolverTiling(const ModelInfo &model_info) override;
  af::Status GenDoTiling(const ModelInfo &model_info) override;
  af::Status GenToolFuncs() override;
  af::Status GenHardwareCons(const ModelInfo &model_info) override;
  af::Status GenPipeTypeObj(const ModelInfo &model_info) override;
  af::Status GenGetObj(const ModelInfo &model_info) override;
  af::Status GenExtraSummaryInfo(const ModelInfo &model_info, const ArgsManager &args_manager,
                                 std::string &case_info_str) override;

 private:
  // 辅助函数：配置SolverPassManager的公共参数
  void ConfigureSolverPassManagerCommon(SolverPassManager &solver_pass_manager, const ModelInfo &model_info);
};
}  // namespace att
#endif
