/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_DATACOPY_H__
#define __ASCENDC_API_DATACOPY_H__

template <typename T>
inline __aicore__ void DataCopyPadExtend(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,
                                         uint32_t dst_stride) {
  AscendC::DataCopyExtParams param;
  if (unlikely(src_stride == 0 && dst_stride == 0)) {
    param.blockCount = 1;
    param.blockLen = block_count * block_len * sizeof(T);
    param.srcStride = 0;
    param.dstStride = 0;
  } else {
    uint32_t align_num = ONE_BLK_SIZE / sizeof(T);
    param.blockCount = block_count;
    param.blockLen = block_len * sizeof(T);
    param.srcStride = src_stride * sizeof(T);
    param.dstStride = dst_stride / align_num;
  }

  AscendC::DataCopyPadExtParams<T> pad_params = {true, 0, 0, 0};
  AscendC::DataCopyPad(dst, src, param, pad_params);
}

template <typename T>
inline __aicore__ void DataCopyPadExtend(const AscendC::GlobalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,
                                         uint32_t dst_stride) {
  uint32_t align_num = ONE_BLK_SIZE / sizeof(T);
  AscendC::DataCopyExtParams param;
  param.blockCount = block_count;
  param.blockLen = block_len * sizeof(T);
  param.srcStride = src_stride / align_num;
  param.dstStride = dst_stride * sizeof(T);

  AscendC::DataCopyPad(dst, src, param);
}

template <typename T>
inline __aicore__ void DataCopyExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                      const uint32_t size) {
  AscendC::DataCopy(dst, src, size);
}

#endif  // __ASCENDC_API_DATACOPY_H__
