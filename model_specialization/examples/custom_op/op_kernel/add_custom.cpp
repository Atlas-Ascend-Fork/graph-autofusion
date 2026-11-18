/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "add_custom_tiling.h"

class KernelAdd {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
    length_ = length;
    uint32_t offset = AscendC::GetBlockIdx() * length;
    x_.SetGlobalBuffer((__gm__ float *)x + offset, length);
    y_.SetGlobalBuffer((__gm__ float *)y + offset, length);
    z_.SetGlobalBuffer((__gm__ float *)z + offset, length);
    pipe_.InitBuffer(xQueue_, 1, length * sizeof(float));
    pipe_.InitBuffer(yQueue_, 1, length * sizeof(float));
    pipe_.InitBuffer(zQueue_, 1, length * sizeof(float));
  }
  __aicore__ inline void Process() {
    auto x = xQueue_.AllocTensor<float>();
    auto y = yQueue_.AllocTensor<float>();
    AscendC::DataCopy(x, x_, length_);
    AscendC::DataCopy(y, y_, length_);
    xQueue_.EnQue(x);
    yQueue_.EnQue(y);
    x = xQueue_.DeQue<float>();
    y = yQueue_.DeQue<float>();
    auto z = zQueue_.AllocTensor<float>();
    AscendC::Add(z, x, y, length_);
    zQueue_.EnQue(z);
    xQueue_.FreeTensor(x);
    yQueue_.FreeTensor(y);
    z = zQueue_.DeQue<float>();
    AscendC::DataCopy(z_, z, length_);
    zQueue_.FreeTensor(z);
  }

 private:
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_, yQueue_;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> zQueue_;
  AscendC::GlobalTensor<float> x_, y_, z_;
  uint32_t length_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
  AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(AddCustomTilingData);
  GET_TILING_DATA(data, tiling);
  // This sample accepts only float[1024], split evenly across eight blocks.
  if (data.totalLength != 1024 || data.tileNum != 8 || AscendC::GetBlockIdx() >= 8) {
    return;
  }
  KernelAdd op;
  op.Init(x, y, z, data.totalLength / data.tileNum);
  op.Process();
  AscendC::PipeBarrier<PIPE_ALL>();
}
