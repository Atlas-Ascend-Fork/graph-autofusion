/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_CODEGEN_H__
#define __AUTOFUSE_CODEGEN_H__

#include "ascir.h"
#include "codegen_tiling.h"
#include "ascgen_log.h"
#include "schedule_result.h"
std::string RemoveSubDirInclude(const std::string &kernel_str);
namespace codegen {
struct CodegenResult {
  std::string proto = "";
  std::string tiling_data = "";
  std::string tiling = "";
  std::string kernel = "";
  std::string infer_shape = "";
};

struct CodegenOptions {
  std::string tiling_lib_path;
  std::string tiling_lib_codegen_symbol;
  bool using_att_calc_qbt_size = true;
  std::string soc_version;
};

class Codegen {
 public:
  explicit Codegen(const CodegenOptions &options);
  Status Generate(const ::ascir::FusedScheduledResult &fused_schedule_result, CodegenResult &result) const;
  Status Generate(const std::map<std::string, std::string> &shape_info,
                  const ::ascir::FusedScheduledResult &fused_schedule_result, CodegenResult &result) const;
  // launch_decouple：rtLaunch 分派入口形态，由 pyautofuse codegen_bin 入口传入 true；
  // 缺省 false 生成物与 legacy 完全一致。
  Status GenerateForInductor(const ::ascir::FusedScheduledResult &fused_schedule_result, CodegenResult &result,
                             bool launch_decouple = false) const;

  std::string GenerateTilingData(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                 bool is_inductor = false) const;

  // Host-only generation is supported. Cache execution counts are conservative
  // without final records from GenerateKernel on this same scheduled graph.
  // GenerateKernel resets and republishes those records for a new generation.
  Status GenerateTiling(const ::ascir::FusedScheduledResult &fused_schedule_result,
                        const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
                        const std::string &core_num,
                        std::map<std::string, std::string> &tiling_file_name_to_content) const;
  std::map<std::string, std::string> GenerateTiling(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                    const std::map<std::string, std::string> &shape_info,
                                                    const std::string &pgo_dir, const std::string &core_num) const;
  Status GenerateTilingForInductor(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                   std::map<std::string, std::string> &tiling_file_name_to_content,
                                   bool launch_decouple = false) const;
  std::map<std::string, std::string> GenerateTilingForInductor(
      const ::ascir::FusedScheduledResult &fused_schedule_result, bool launch_decouple = false) const;

  Status GenerateKernel(const ::ascir::FusedScheduledResult &fused_schedule_result, std::string &result,
                        bool is_inductor = false, bool launch_decouple = false) const;
  // export_name 参数化导出符号名（默认 GetKernelBin；CV rtLaunch 双 bin 为
  // GetKernelBinAic/GetKernelBinAiv）。
  std::string GenGetKernelAndJson(const std::string &kernel_path, const std::string &json_path,
                                  const std::string &export_name = "GetKernelBin") const;
  std::string GenerateInferShape(const std::vector<std::vector<std::string>> &symbol_shape_str,
                                 const std::map<std::string, std::string> &shape_info) const;

  std::string GeneratorPgo(const ::ascir::FusedScheduledResult &fused_schedule_result,
                           const std::string &pgo_dir) const;

 private:
  TilingLib tiling_lib_;
  bool using_att_calc_qbt_size_;
  std::string soc_version_;
};
}  // namespace codegen

#endif
