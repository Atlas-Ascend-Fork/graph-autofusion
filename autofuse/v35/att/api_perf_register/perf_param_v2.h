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
constexpr double kManyShortCompactStrideBase = 0.0032923310947161463;
constexpr double kManyShortCompactStridePerCore = 0.006521231195284009 / 56.0;

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
inline constexpr PayloadDomain kStoreAlignedMediumPayload{{8, 40}, {12288, 32768}};
inline constexpr PayloadDomain kStoreAlignedConcurrentPayload{{42, 56}, {8448, 20480}};
inline constexpr PayloadDomain kLoadAlignedDensePayload{{16, 56}, {1024, 4096}};
inline constexpr PayloadDomain kLoadAlignedDenseBulkPayload{{16, 56}, {16384, 65536}};
inline constexpr Range kLoadDenseBulkRows{512, 2048};
inline constexpr Range kLoadDenseBulkWidth{8, 32};
// FP32 ordinary Pad Load with aligned wide rows and no GM/UB gap.
inline constexpr PayloadDomain kLoadAlignedDenseWidePayload{{6, 56}, {49152, 122880}};
inline constexpr Range kLoadDenseWideRows{5, 24};
inline constexpr Range kLoadDenseWideWidth{1024, 4096};
// FP32 aligned Normal/Compact Load with a dense UB and a wide GM row pitch.
inline constexpr Range kLoadAlignedLargeRows{104, 184};
inline constexpr Range kLoadAlignedLargeWidth{112, 168};
inline constexpr Range kLoadAlignedLargePitch{28672, 36864};
inline constexpr Range kLoadAlignedLargeCores{32, 56};
inline constexpr int32_t kLoadAlignedLargeBurstElements = 128 / kFloat32Bytes;
inline constexpr Range kLoadCompactRows{2, 18};
inline constexpr Range kLoadCompactWidth{32, 256};
inline constexpr Range kLoadCompactGap{64, 4096};
// Aligned rows with a moderate GM pitch and a dense UB destination.
inline constexpr Range kLoadCompactMediumRows{32, 80};
inline constexpr Range kLoadCompactMediumWidth{32, 128};
inline constexpr Range kLoadCompactMediumPitch{256, 768};
inline constexpr Range kLoadCompactMediumCores{18, 56};
inline constexpr Range kLoadCompactMediumBytes{4096, 27648};
// A single Compact DMA with a full 128-byte FP32 row and a packed UB.
inline constexpr Range kLoadCompactWidePitch{3072, 6144};
inline constexpr Range kLoadCompactWidePitchRows{48, 164};
inline constexpr Range kLoadCompactWidePitchCores{38, 56};
inline constexpr int32_t kLoadCompactWidePitchWidth = 128 / kFloat32Bytes;
// Single-element rows of an ordinary FP32 Compact copy.
inline constexpr Range kLoadCompactScalarPitch{32, 64};
inline constexpr Range kLoadCompactScalarRows{32, 128};
inline constexpr Range kLoadCompactScalarCores{2, 6};
// FP32 physical GM row pitch in elements, covered by independent Compact Load samples.
inline constexpr Range kLoadMergedCompactPitch{1966080, 2228224};
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
inline constexpr Range kLoadFlattenedSingleRowWidth{2112, 16384};
inline constexpr Range kLoadFlattenedSingleRowCores{42, 56};
inline constexpr Range kLoadSingleRowLowCoreWidth{1024, 8192};
inline constexpr Range kLoadSingleRowLowCores{4, 23};
inline constexpr Range kLoadLoopOuter{24, 56};
inline constexpr Range kLoadLoopRows{2, 6};
inline constexpr Range kLoadLoopWidth{32, 64};
inline constexpr Range kLoadNarrowLoopRows{2, 8};
inline constexpr Range kLoadNarrowLoopWidth{16, 16};
inline constexpr Range kLoadLoopCores{16, 56};
// Aligned wide-row, single hardware loop. GM pitch is in FP32 elements.
inline constexpr Range kLoadWideLoopOuter{3, 8};
inline constexpr Range kLoadWideLoopRows{9, 20};
inline constexpr Range kLoadWideLoopWidth{96, 160};
inline constexpr Range kLoadWideLoopPitch{768, 1280};
inline constexpr Range kLoadWideLoopCores{24, 56};
// Normal LoopMode with tightly packed 128-byte rows and a longer hardware loop.
inline constexpr Range kLoadLongLoopOuter{16, 24};
inline constexpr Range kLoadLongLoopRows{9, 20};
inline constexpr Range kLoadLongLoopWidth{32, 32};
inline constexpr Range kLoadLongLoopPitch{512, 832};
inline constexpr Range kLoadLongLoopCores{24, 40};
// Normal FP32 LoopMode with aligned rows and a GM pitch larger than the copied span.
inline constexpr Range kLoadSparseLoopOuter{2, 8};
inline constexpr Range kLoadSparseLoopRows{8, 16};
inline constexpr Range kLoadSparseLoopWidth{32, 128};
inline constexpr Range kLoadSparseLoopPitch{2048, 6144};
inline constexpr Range kLoadSparseLoopCores{9, 22};
// FP16 Compact LoopMode Store geometry in elements.
inline constexpr uint32_t kCompactStoreElementBytes = 2;
inline constexpr int32_t kCompactStoreUbAlignment = 16;
inline constexpr int32_t kCompactStoreGmRowPitch = 6;
inline constexpr int32_t kCompactStoreGmLoopPitch = 24;
inline constexpr int32_t kCompactStoreOuter = 5;
inline constexpr Range kCompactStoreRows{1, 4};
inline constexpr Range kCompactStoreWidth{1, 4};
inline constexpr Range kCompactStoreCores{1, 12};
// Validation envelope for applying the existing Store stride model to a folded
// FP16 long-loop descriptor. Values are in iterations and elements.
inline constexpr Range kFoldedStoreLongLoopOuter{208, 331};
inline constexpr Range kFoldedStoreLongLoopGmRowPitch{336, 384};
}  // namespace dma_calibration_v2

class PerfParamTableV2 : public PerfParamTable {
 public:
  PerfParamTableV2();
  ~PerfParamTableV2() override = default;
  const std::map<std::string, std::vector<VfInstructPerf>> &GetVfInstructPerfTable() const;
  [[nodiscard]] const std::vector<VfInstructPerf> &GetVfInstructPerfTable(
      [[maybe_unused]] const std::string &vf_instruct_type) const override;
  [[nodiscard]] const std::vector<VfInstructPerf> &GetVectorFunctionInstructPerfTable(
      const std::string &vf_instruct_type) const override;
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
  [[nodiscard]] bool EnableBalancedCoreProbe() const override {
    return true;
  }

  [[nodiscard]] double GetUbPreferenceMaxRelativeCostGap() const override {
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
