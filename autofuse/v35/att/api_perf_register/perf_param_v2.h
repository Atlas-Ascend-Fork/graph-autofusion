/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_UTIL_PARAMS_V2_H_
#define ATT_UTIL_PARAMS_V2_H_

#include <string>
#include "base/base_types.h"
#include "api_perf_register/perf_param.h"

namespace att {
namespace dma_calibration_v2 {
struct Range {
  int32_t lower;
  int32_t upper;
};

struct PayloadDomain {
  Range cores;
  Range bytes;
};

// Width, gap and pitch are in FP32 elements. Payload is in bytes.
inline constexpr int32_t kFloat32Bytes = 4;
inline constexpr int32_t kAlignedRowElements = 32 / kFloat32Bytes;
inline constexpr PayloadDomain kLoadSmallPayload{{8, 48}, {512, 8192}};
inline constexpr PayloadDomain kStoreSmallPayload{{8, 56}, {1024, 8192}};
inline constexpr PayloadDomain kLoadAlignedDensePayload{{16, 56}, {1024, 4096}};
inline constexpr PayloadDomain kLoadAlignedDenseBulkPayload{{16, 56}, {16384, 65536}};
inline constexpr Range kLoadDenseBulkRows{512, 2048};
inline constexpr Range kLoadDenseBulkWidth{8, 32};
inline constexpr Range kLoadCompactRows{2, 18};
inline constexpr Range kLoadCompactWidth{32, 256};
inline constexpr Range kLoadCompactGap{64, 4096};
inline constexpr Range kLoadWidePitchRows{256, 512};
inline constexpr Range kLoadWidePitch{4096, 16384};
inline constexpr int32_t kLoadWidePitchWidth = 32;
inline constexpr int32_t kLoadWidePitchAlignment = 4096;
inline constexpr int32_t kLoadWideGapPitch = 131072;
inline constexpr int32_t kLoadWideGapRowsCenter = 24;
inline constexpr int32_t kLoadWideGapRowsDistance = 8;
inline constexpr int32_t kLoadWideGapWidthCenter = 576;
inline constexpr int32_t kLoadWideGapWidthRadius = 192;
inline constexpr int32_t kLoadWideGapCoresCenter = 48;
inline constexpr int32_t kLoadWideGapCoresRadius = 8;
inline constexpr Range kLoadSingleRowWidth{512, 30680};
inline constexpr Range kLoadSingleRowCores{24, 56};
inline constexpr Range kLoadLoopOuter{24, 56};
inline constexpr Range kLoadLoopRows{2, 6};
inline constexpr Range kLoadLoopWidth{32, 64};
inline constexpr Range kLoadNarrowLoopRows{2, 8};
inline constexpr Range kLoadNarrowLoopWidth{16, 16};
inline constexpr Range kLoadLoopCores{16, 56};
}  // namespace dma_calibration_v2

class PerfParamTableV2 : public PerfParamTable {
 public:
  PerfParamTableV2();
  ~PerfParamTableV2() override = default;
  const std::map<std::string, std::vector<VfInstructPerf>> &GetVfInstructPerfTable() const;
  [[nodiscard]] const std::vector<VfInstructPerf> &GetVfInstructPerfTable(
      [[maybe_unused]] const std::string &vf_instruct_type) const override;
  const std::map<std::string, std::vector<VfInstructDtypeMappingPerf>> &GetVfInstructDtypeMappingPerfTable() const;
  [[nodiscard]] const std::vector<VfInstructDtypeMappingPerf> &GetVfInstructDtypeMappingPerfTable(
      [[maybe_unused]] const std::string &vf_instruct_type) const override;
  [[nodiscard]] Expr GetVectorFunctionHeadCost() const override;
  [[nodiscard]] std::string GetApiRegisterVerName() const override;
  [[nodiscard]] Expr GetOpHeadCost() const override;
  [[nodiscard]] uint32_t GetMicroApiLen() const override {
    return arch_param::kV2ArchHardwareConfig.vector_len_size;
  }
  [[nodiscard]] const std::string *GetAscendCApiPerfTable() const override;
  [[nodiscard]] PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) const override;
  [[nodiscard]] static Expr GetMTE2PipeHead(const std::vector<NodeInfo> &node_infos,
                                            std::map<Expr, TernaryOp, ExprCmp> &ternary_ops);

 private:
  std::map<std::string, std::vector<VfInstructPerf>> vf_instruct_type_2_api_perf_;
  std::map<std::string, std::vector<VfInstructDtypeMappingPerf>> vf_instruct_type_2_dtype_mapping_api_perf_;
  std::map<PipeType, PipeHeadPerfFunc> pipes_head_perf_;
};

class TilingScheduleConfigTableV2 : public TilingScheduleConfigTable {
 public:
  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override {
    return false;
  }
  [[nodiscard]] bool IsEnableCacheLineCheck() const override {
    return true;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return {true, af::Symbol(0.01), af::Symbol(1.0)};
  }
  [[nodiscard]] double GetUbThresholdPerfValEffect() const override {
    constexpr double kDefaultUbThresholdPerfValEffect = 0.05;
    return kDefaultUbThresholdPerfValEffect;
  }
  [[nodiscard]] double GetPerfEffectVal() const override {
    constexpr double kDefaultPerfEffectVal = 2000.0;
    return kDefaultPerfEffectVal;
  }
  [[nodiscard]] bool PreferObjectiveForLowUbSingleBlock() const override {
    return true;
  }
  [[nodiscard]] double GetCrossUbThresholdMaxRelativeCostGap() const override {
    constexpr double kMaxRelativeCostGap = 0.10;
    return kMaxRelativeCostGap;
  }
  [[nodiscard]] TilingScheduleConfig GetModelTilingScheduleConfig() const override {
    TilingScheduleConfig config;
    config.trade_off_config = GetTradeOffConfig();
    config.cache_line_size = GetCacheLineSize();
    config.vector_len_size = GetVectorLenSize();
    return config;
  }
  [[nodiscard]] uint32_t GetCacheLineSize() const override {
    return arch_param::kV2ArchHardwareConfig.cache_line_size;
  }
  [[nodiscard]] uint32_t GetVectorLenSize() const override {
    return arch_param::kV2ArchHardwareConfig.vector_len_size;
  }
  // 新增：V2形态使能Reduce分核惩罚
  [[nodiscard]] bool IsCoreNumThresholdPenaltyEnable() const override {
    return true;
  }
};
// ArgReduce has expensive index tracking and small outputs. Compare compute costs rather
// than excluding useful core counts through output cacheline/UB occupancy heuristics.
class ArgReduceTilingScheduleConfigTableV2 : public TilingScheduleConfigTableV2 {
 public:
  [[nodiscard]] bool IsEnableCacheLineCheck() const override {
    return false;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return TradeOffConfig{false};
  }
  [[nodiscard]] double GetPerfEffectVal() const override {
    return 0.0;
  }
  [[nodiscard]] TilingScheduleConfigPriority GetConfigPriority() const override {
    return TilingScheduleConfigPriority::kHeavyOpPriority;
  }
};
extern const std::string kParamV2Info;
inline const std::string kNddma = "Nddma";
}  // namespace att
#endif  // ATT_UTIL_PARAMS_V2_H_
