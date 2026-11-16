/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_TILING_DATA_H__
#define __CODEGEN_TILING_DATA_H__

#include <string>

#include "ascir.h"
#include "schedule_result.h"
namespace codegen {

class TilingData {
 public:
  explicit TilingData(const std::string &kernel, const std::string &name_class = "TilingData");
  std::string Generate(const ascir::FusedScheduledResult &fused_schedule_result, bool is_inductor = false);
  std::string GenCVConstTilingData(const std::string &tiling_data_struct_name, bool is_inductor_scene);
  std::string GenCVConstReplace(const std::string &tiling_data_struct_name);
  std::string GenerateConst(const ascir::FusedScheduledResult &fused_schedule_result, bool is_inductor_scene = true);
  static void GetTqueAndTbufId(const af::AscGraph &graph, std::set<int64_t> &q_ids, std::set<int64_t> &b_ids);
  static void GetTmpBufName(const af::AscGraph &graph, std::set<int64_t> &b_ids);
  static af::Status GetApiTilingDataName(const ascir::NodeView &node, std::vector<std::string> &api_tiling_data_names);

 protected:
  std::string class_name;
  std::string kernel_name;

  static std::string transposeApiTilingData;
  static std::string macros_and_includes;

  static std::string pgo_perf_struct;

  std::string ClassBegin(const std::string &begin_kernel_name, const std::string &begin_class_name) const;
  std::string DataFieldDefine(ascir::SizeVar &size) const;
  std::string DataFieldConstDefine(ascir::SizeVar &size);
  std::string StructDataFiledDefine(const std::string &type_name, const std::string &filed_name) const;
  std::string ClassEnd() const;
  std::string ClassRegister();
  void ProcessSingleGroup(const ascir::ScheduleGroup &schedule_group, std::stringstream &ss);
  void ProcessMultiGroup(uint64_t pos, const int graph_id, const std::vector<ascir::ScheduleGroup> &schedule_groups,
                         std::stringstream &ss1, std::stringstream &ss2);
  void AddApiTilingData(const af::AscGraph &graph, std::stringstream &ss, uint32_t tiling_case_id);

 private:
  std::string ConstApiTilingDataFiledDefine(std::string &type_name, std::string &field_name,
                                            const ascir::NodeView &node);
  void ConstTilingDataFieldPopBack();

  std::string GenGenTilingDataFieldConstDefFunc() const;
  std::string GenGenTilingDataFieldConstValueFunc() const;
  af::Status ProcessCubeFusionResult(ascir::FusedScheduledResult &schedule_result);

  std::string GenTingDataField(std::string field_name) const;
  std::string GetNameOfGenTilingDataFieldConstDefFunc(const std::string field_name) const;
  std::string GetNameOfGenTilingDataFieldConstDefFuncSimple(const std::string field_name) const;
  std::string GetNameOfGenTilingDataFieldConstValueFuncSimple(const std::string field_name) const;

  std::string GenStringReplaceFunc() const;
  std::string GenConstGenResultReplace() const;

  std::string GenTilingDataFieldConstDefFunc(std::string &dtype, std::string &filed_name);
  std::string GetCommonTilingField(bool is_group, const ascir::FusedScheduledResult &fused_schedule_result);
  std::string GenConstCommonTilingField(const std::vector<std::string> &tiling_key_fileds,
                                        const std::vector<ascir::TensorId> &workspace_tensor_id);

  std::string TqueOrTbufDataFieldDefine(int64_t index, const std::string &que_or_buf) const;
  std::string TqueOrTbufDataFieldConstDefine(int64_t index, const std::string &que_or_buf);

  std::string TmpBufDataFieldDefine(const std::string &tmp_tbuf_name) const;
  std::string TmpBufDataFieldConstDefine(const std::string &tmp_tbuf_name);

  std::string DataFieldConstDefine(const std::string &buf_name);

  void GenTqueTbufTmpBufFunc(const std::set<int64_t> &q_ids, const std::set<int64_t> &b_ids, std::stringstream &ss);

  std::vector<std::string> const_tiling_data_field;
  std::stringstream pre_var_ss;
  std::stringstream pre_func_ss;
  std::vector<std::string> field_var_defs_;

  bool const_mode_ = false;  // const模式下生成字段都为const的，且有初值
};
};  // namespace codegen

#endif
