/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__
#define __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__

#include "ascendc_ir.h"

namespace af {
namespace ascir {

std::vector<std::unique_ptr<TmpBufDesc>> CalcLog2TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcModTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcReduceTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcErfTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcGeluTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcTanhTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcGatherTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcPowTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcExp2TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcIgammaTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcIgammacTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcLgammaTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcVoidTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcPolygammaTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcBucketizeTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> GetCompareSizeV2([[maybe_unused]] const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSinTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAsinTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAsinhTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAtanTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAtanhTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcCosTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAcosTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcCoshTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcDigammaTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcErfcTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAcoshTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcAtan2TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcCeilTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSinhTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcTanTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcTruncTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcXorTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcTransposeTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcModifiedBesselI0TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcModifiedBesselI1TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcModifiedBesselK0TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcModifiedBesselK1TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcScaledModifiedBesselK0TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcScaledModifiedBesselK1TmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcIsInfTmpSize(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcSoftmaxTmpSizeV2(const AscNode &node);
std::vector<std::unique_ptr<TmpBufDesc>> CalcMaskedFillTmpSize(const AscNode &node);
}  // namespace ascir
}  // namespace af
#endif  // __ASCIR_REG_FUNC_DEFAULT_REG_FUNC_V2_H__
