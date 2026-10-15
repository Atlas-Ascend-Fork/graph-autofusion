/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Source Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SK_FLAG_DIAG_H_
#define SK_FLAG_DIAG_H_

#include "kernel_operator.h"

inline __aicore__ void send_all_flag_aic_aiv(void) {
  if ASCEND_IS_AIC {
    for (int i = 0; i <= 10; i++) {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(i);
      }
    }
  }
}

inline __aicore__ void wait_all_flag_aic_aiv(void) {
  if ASCEND_IS_AIV {
    for (int i = 0; i <= 10; i++) {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreWaitFlag(i);
      }
    }
  }
}

inline __aicore__ void send_all_flag_aiv_aic(void) {
  if ASCEND_IS_AIV {
    for (int i = 0; i <= 10; i++) {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(i);
      }
    }
  }
}

inline __aicore__ void wait_all_flag_aiv_aic(void) {
  if ASCEND_IS_AIC {
    for (int i = 0; i <= 10; i++) {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreWaitFlag(i);
      }
    }
  }
}

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
constexpr int INTRA_FLAG_ID_OFFSET = 16;

template <pipe_t aic_pipe, pipe_t aiv_pipe>
inline __aicore__ void test_intra_core_sync_flag_pair(int flag_id) {
  // AIC -> AIV0/AIV1: AIC flag_id and flag_id + 16 map to the same
  // flag_id on AIV0 and AIV1 respectively.
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIC {
    for (int j = 0; j < 15; j++) {
      AscendC::CrossCoreSetFlag<0x4, aic_pipe>(flag_id);
      AscendC::CrossCoreSetFlag<0x4, aic_pipe>(flag_id + INTRA_FLAG_ID_OFFSET);
    }
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    for (int j = 0; j < 15; j++) {
      AscendC::CrossCoreWaitFlag<0x4, aiv_pipe>(flag_id);
    }
  }

  // AIV0/AIV1 -> AIC: each AIV sets flag_id, which maps to the AIC
  // flag_id and flag_id + 16 wait slots respectively.
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    for (int j = 0; j < 15; j++) {
      AscendC::CrossCoreSetFlag<0x4, aiv_pipe>(flag_id);
    }
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIC {
    for (int j = 0; j < 15; j++) {
      AscendC::CrossCoreWaitFlag<0x4, aic_pipe>(flag_id);
      AscendC::CrossCoreWaitFlag<0x4, aic_pipe>(flag_id + INTRA_FLAG_ID_OFFSET);
    }
  }
}

inline __aicore__ void test_intra_core_sync_flag(int flag_id) {
  // The eight pairs cover every valid AIC and AIV pipe once as both a
  // producer and consumer without enumerating redundant pipe pairs.
  test_intra_core_sync_flag_pair<PIPE_S, PIPE_S>(flag_id);
  test_intra_core_sync_flag_pair<PIPE_M, PIPE_S>(flag_id);
  test_intra_core_sync_flag_pair<PIPE_MTE1, PIPE_S>(flag_id);
  test_intra_core_sync_flag_pair<PIPE_MTE2, PIPE_S>(flag_id);
  test_intra_core_sync_flag_pair<PIPE_FIX, PIPE_S>(flag_id);
  test_intra_core_sync_flag_pair<PIPE_FIX, PIPE_MTE2>(flag_id);
  test_intra_core_sync_flag_pair<PIPE_FIX, PIPE_MTE3>(flag_id);
  test_intra_core_sync_flag_pair<PIPE_FIX, PIPE_V>(flag_id);
}

inline __aicore__ void test_intra_core_sync_flags(int flag_id = -1) {
  if (flag_id >= 0) {
    if (flag_id <= 10) {
      test_intra_core_sync_flag(flag_id);
    }
    return;
  }

  for (int i = 0; i <= 10; i++) {
    test_intra_core_sync_flag(i);
  }
}
#endif

inline __aicore__ void test_cross_core_sync_flags(int flag_id = -1) {
  if (flag_id >= 0 && flag_id <= 15) {
    // only diag one single event id
    // AIC -> AIV diag
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIC {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flag_id);
      }
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreWaitFlag(flag_id);
      }
    }

    // AIV -> AIC diag
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag_id);
      }
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIC {
      for (int j = 0; j < 15; j++) {
        AscendC::CrossCoreWaitFlag(flag_id);
      }
    }
  } else {
    // AIC -> AIV
    AscendC::SyncAll<false>();
    send_all_flag_aic_aiv();
    AscendC::SyncAll<false>();
    wait_all_flag_aic_aiv();

    // AIV -> AIC
    AscendC::SyncAll<false>();
    send_all_flag_aiv_aic();
    AscendC::SyncAll<false>();
    wait_all_flag_aiv_aic();
  }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  test_intra_core_sync_flags(flag_id);
#endif
}

#endif  // SK_FLAG_DIAG_H_
