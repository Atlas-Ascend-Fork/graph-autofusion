/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "tikicpulib.h"
#include "kernel_operator.h"
using namespace AscendC;

#include "utils.h"
#include "test_api_utils.h"
#include "duplicate.h"
#include "broadcast.h"

TEST(TestApiBroadcast, Test_a1_to_ab) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (half *)AscendC::GmAlloc(sizeof(half) * a * 1);
  auto *y = (half *)AscendC::GmAlloc(sizeof(half) * a * b);
  half expect[a][b];

  for (int i = 0; i < a; i++) {
    x[i] = (double)i;
    for (int j = 0; j < b; j++) {
      expect[i][j] = (double)i;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, half *x, half *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * 1);
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (double)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1c_to_abc) {
  // 构造测试输入和预期结果
  int a = 10, b = 10, c = 16;
  auto *x = (half *)AscendC::GmAlloc(sizeof(half) * a * 1 * c);
  auto *y = (half *)AscendC::GmAlloc(sizeof(half) * a * b * c);
  half expect[a][b][c];

  for (int i = 0; i < a; i++) {
    for (int k = 0; k < c; ++k) {
      x[i * c + k] = (double)(i * 1000 + k);
    }
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; ++k) {
        expect[i][j][k] = (double)(i * 1000 + k);
      }
    }
  }
  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, half *x, half *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * 1 * c);
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b * c);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; ++k) {
        auto diff = (double)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }
  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1c_to_abc_withcopy) {
  // 构造测试输入和预期结果
  int a = 10, b = 10, c = 16;
  auto *x = (half *)AscendC::GmAlloc(sizeof(half) * a * 1 * c);
  auto *y = (half *)AscendC::GmAlloc(sizeof(half) * a * b * c);
  half expect[a][b][c];

  for (int i = 0; i < a; i++) {
    for (int k = 0; k < c; ++k) {
      x[i * c + k] = (double)(i * 1000 + k);
    }
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; ++k) {
        expect[i][j][k] = (double)(i * 1000 + k);
      }
    }
  }
  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, half *x, half *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * 1 * c);
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b * c);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; ++k) {
        auto diff = (double)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }
  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1_to_ab_uint8) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * 1);
  auto *y = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * b);
  uint8_t expect[a][b];

  for (uint32_t i = 0; i < a; i++) {
    x[i] = i;
    for (uint32_t j = 0; j < b; j++) {
      expect[i][j] = i;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, uint8_t *x, uint8_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint8_t) * a * 1);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint8_t>();
    auto l_y = ybuf.Get<uint8_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1b_to_ab_uint8) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * 1 * b);
  auto *y = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * b);
  uint8_t expect[a][b];

  for (uint32_t j = 0; j < b; j++) {
    x[j] = j;
    for (uint32_t i = 0; i < a; i++) {
      expect[i][j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, uint8_t *x, uint8_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint8_t) * 1 * b);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint8_t>();
    auto l_y = ybuf.Get<uint8_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, b, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1bc_to_abc_uint8) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 8;
  auto *x = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * 1 * b * c);
  auto *y = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * b * c);
  uint8_t expect[a][b][c];

  for (uint32_t j = 0; j < b; j++) {
    for (uint32_t k = 0; k < c; k++) {
      x[j * c + k] = j * c + k;
      for (uint32_t i = 0; i < a; i++) {
        expect[i][j][k] = j * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, uint8_t *x, uint8_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint8_t) * 1 * b * c);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint8_t>();
    auto l_y = ybuf.Get<uint8_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, 1, b, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1c_to_abc_uint8) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 32;
  auto *x = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * 1 * c);
  auto *y = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * b * c);
  uint8_t expect[a][b][c];

  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t k = 0; k < c; k++) {
      x[i * c + k] = i * c + k;
      for (uint32_t j = 0; j < b; j++) {
        expect[i][j][k] = i * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, uint8_t *x, uint8_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint8_t) * a * 1 * c);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint8_t>();
    auto l_y = ybuf.Get<uint8_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_11_to_ab_uint8) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * 1 * 1);
  auto *y = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * b);
  uint8_t expect[a][b];

  x[0] = 3;
  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t j = 0; j < b; j++) {
      expect[i][j] = x[0];
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, uint8_t *x, uint8_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint8_t) * 1 * 1);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint8_t>();
    auto l_y = ybuf.Get<uint8_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, 1);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, 1, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1b_to_ab_int64) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * 1 * b);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a * b);
  int64_t expect[a][b];

  for (uint32_t j = 0; j < b; j++) {
    x[j] = j;
    for (uint32_t i = 0; i < a; i++) {
      expect[i][j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int64_t *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int64_t) * 1 * b);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int64_t>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, b, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (int64_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// TEST(TestApiBroadcast, Test_1b_to_ab_int64_jd) {
//   // 构造测试输入和预期结果
//   int a = 3, b = 600;
//   auto *x = (int64_t*)AscendC::GmAlloc(sizeof(int64_t) * 1 * b);
//   auto *y = (int64_t*)AscendC::GmAlloc(sizeof(int64_t) * a * b);
//   int64_t expect[a][b];

//   for (uint32_t j = 0; j < b; j++) {
//     x[j] = j;
//     for (uint32_t i = 0; i < a; i++) {
//       expect[i][j] = j;
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, int64_t *x, int64_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(int64_t) * 1 * b);
//     tpipe.InitBuffer(ybuf, sizeof(int64_t) * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<int64_t>();
//     auto l_y = ybuf.Get<int64_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, b);
//     GmToUb(l_y, y, a * b);

//     Broadcast(l_y, l_x, 1, b, 0, a, b, 0, l_tmp);

//     UbToGm(y, l_y, a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       auto diff = (int64_t)(y[i*b + j] - expect[i][j]);
//       if (diff < -1e-5 || diff > 1e-5) {
//         diff_count++;
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

TEST(TestApiBroadcast, Test_1bc_to_abc_int64) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 8;
  auto *x = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * 1 * b * c);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a * b * c);
  int64_t expect[a][b][c];

  for (uint32_t j = 0; j < b; j++) {
    for (uint32_t k = 0; k < c; k++) {
      x[j * c + k] = j * c + k;
      for (uint32_t i = 0; i < a; i++) {
        expect[i][j][k] = j * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, int64_t *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int64_t) * 1 * b * c);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int64_t>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, 1, b, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (int64_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1_to_ab_int64) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a * 1);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a * b);
  int64_t expect[a][b];

  for (uint32_t i = 0; i < a; i++) {
    x[i] = i;
    for (uint32_t j = 0; j < b; j++) {
      expect[i][j] = i;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int64_t *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int64_t) * a * 1);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int64_t>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (int64_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// TEST(TestApiBroadcast, Test_a1c_to_abc_int64) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32, c = 32;
//   auto *x = (int64_t*)AscendC::GmAlloc(sizeof(int64_t) * a * 1 * c);
//   auto *y = (int64_t*)AscendC::GmAlloc(sizeof(int64_t) * a * b * c);
//   int64_t expect[a][b][c];

//   for (uint32_t i = 0; i < a; i++) {
//     for (uint32_t k = 0; k < c; k++) {
//       x[i * c + k] = i * c + k;
//       for (uint32_t j = 0; j < b; j++) {
//         expect[i][j][k] = i * c + k;
//       }
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, int c, int64_t *x, int64_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(int64_t) * a * 1 * c);
//     tpipe.InitBuffer(ybuf, sizeof(int64_t) * a * b * c);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<int64_t>();
//     auto l_y = ybuf.Get<int64_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, a * c);
//     GmToUb(l_y, y, a * b * c);

//     Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

//     UbToGm(y, l_y, a * b * c);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       for (int k = 0; k < c; k++) {
//         auto diff = (int64_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
//         if (diff < -1e-5 || diff > 1e-5) {
//           diff_count++;
//         }
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

TEST(TestApiBroadcast, Test_11_to_ab_int64) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * 1 * 1);
  auto *y = (int64_t *)AscendC::GmAlloc(sizeof(int64_t) * a * b);
  int64_t expect[a][b];

  x[0] = 3;
  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t j = 0; j < b; j++) {
      expect[i][j] = x[0];
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int64_t *x, int64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int64_t) * 1 * 1);
    tpipe.InitBuffer(ybuf, sizeof(int64_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int64_t>();
    auto l_y = ybuf.Get<int64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, 1);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, 1, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (int64_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1bc_to_abc_uint64) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 8;
  auto *x = (uint64_t *)AscendC::GmAlloc(sizeof(uint64_t) * 1 * b * c);
  auto *y = (uint64_t *)AscendC::GmAlloc(sizeof(uint64_t) * a * b * c);
  uint64_t expect[a][b][c];

  for (uint32_t j = 0; j < b; j++) {
    for (uint32_t k = 0; k < c; k++) {
      x[j * c + k] = j * c + k;
      for (uint32_t i = 0; i < a; i++) {
        expect[i][j][k] = j * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, uint64_t *x, uint64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint64_t) * 1 * b * c);
    tpipe.InitBuffer(ybuf, sizeof(uint64_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint64_t>();
    auto l_y = ybuf.Get<uint64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, 1, b, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint64_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1_to_ab_uint64) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (uint64_t *)AscendC::GmAlloc(sizeof(uint64_t) * a * 1);
  auto *y = (uint64_t *)AscendC::GmAlloc(sizeof(uint64_t) * a * b);
  uint64_t expect[a][b];

  for (uint32_t i = 0; i < a; i++) {
    x[i] = i;
    for (uint32_t j = 0; j < b; j++) {
      expect[i][j] = i;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, uint64_t *x, uint64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint64_t) * a * 1);
    tpipe.InitBuffer(ybuf, sizeof(uint64_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint64_t>();
    auto l_y = ybuf.Get<uint64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint64_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// TEST(TestApiBroadcast, Test_a1c_to_abc_uint64) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32, c = 32;
//   auto *x = (uint64_t*)AscendC::GmAlloc(sizeof(uint64_t) * a * 1 * c);
//   auto *y = (uint64_t*)AscendC::GmAlloc(sizeof(uint64_t) * a * b * c);
//   uint64_t expect[a][b][c];

//   for (uint32_t i = 0; i < a; i++) {
//     for (uint32_t k = 0; k < c; k++) {
//       x[i * c + k] = i * c + k;
//       for (uint32_t j = 0; j < b; j++) {
//         expect[i][j][k] = i * c + k;
//       }
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, int c, uint64_t *x, uint64_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(uint64_t) * a * 1 * c);
//     tpipe.InitBuffer(ybuf, sizeof(uint64_t) * a * b * c);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<uint64_t>();
//     auto l_y = ybuf.Get<uint64_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, a * c);
//     GmToUb(l_y, y, a * b * c);

//     Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

//     UbToGm(y, l_y, a * b * c);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       for (int k = 0; k < c; k++) {
//         auto diff = (uint64_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
//         if (diff < -1e-5 || diff > 1e-5) {
//           diff_count++;
//         }
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

TEST(TestApiBroadcast, Test_11_to_ab_uint64) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (uint64_t *)AscendC::GmAlloc(sizeof(uint64_t) * 1 * 1);
  auto *y = (uint64_t *)AscendC::GmAlloc(sizeof(uint64_t) * a * b);
  uint64_t expect[a][b];

  x[0] = 3;
  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t j = 0; j < b; j++) {
      expect[i][j] = x[0];
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, uint64_t *x, uint64_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint64_t) * 1 * 1);
    tpipe.InitBuffer(ybuf, sizeof(uint64_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint64_t>();
    auto l_y = ybuf.Get<uint64_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, 1);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, 1, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint64_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1_to_ab_stride) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (half *)AscendC::GmAlloc(sizeof(half) * a * 8);
  auto *y = (half *)AscendC::GmAlloc(sizeof(half) * a * b);
  half expect[a][b];

  for (int i = 0; i < a; i++) {
    x[i * 8] = (double)i;
    for (int z = 1; z < 8; z++) {
      x[i * 8 + z] = (double)z;
    }
    for (int j = 0; j < b; j++) {
      expect[i][j] = (double)i;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, half *x, half *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * 8);
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * 8);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp, 8);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (double)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1_to_ab_uint8_stride) {
  // 构造测试输入和预期结果，源按 last_dim_stride 跨步存放，行内其余位置填哨兵值
  int a = 16, b = 32, stride = 2;
  const uint8_t sentinel = 0xFF;
  auto *x = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * stride);
  auto *y = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * b);
  uint8_t expect[a][b];

  for (int i = 0; i < a; i++) {
    for (int z = 0; z < stride; z++) {
      x[i * stride + z] = (z == 0) ? (uint8_t)i : sentinel;
    }
    for (int j = 0; j < b; j++) {
      expect[i][j] = (uint8_t)i;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int stride, uint8_t *x, uint8_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint8_t) * a * stride);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * a * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<uint8_t>();
    auto l_y = ybuf.Get<uint8_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * stride);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp, stride);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, stride, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      if (y[i * b + j] != expect[i][j]) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1_to_ab_uint8_stride_tail) {
  // 内轴非块对齐，覆盖多行与尾部场景
  int a = 48, b = 20, stride = 3;
  const uint8_t sentinel = 0xFF;
  auto *x = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * stride);
  auto *y = (uint8_t *)AscendC::GmAlloc(sizeof(uint8_t) * a * b);
  uint8_t expect[a][b];

  for (int i = 0; i < a; i++) {
    for (int z = 0; z < stride; z++) {
      x[i * stride + z] = (z == 0) ? (uint8_t)i : sentinel;
    }
    for (int j = 0; j < b; j++) {
      expect[i][j] = (uint8_t)i;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int stride, uint8_t *x, uint8_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint8_t) * a * stride);
    tpipe.InitBuffer(ybuf, sizeof(uint8_t) * a * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<uint8_t>();
    auto l_y = ybuf.Get<uint8_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * stride);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp, stride);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, stride, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      if (y[i * b + j] != expect[i][j]) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1_to_ab_int8_stride) {
  int a = 8, b = 17, stride = 3;
  const int8_t sentinel = 99;
  auto *x = (int8_t *)AscendC::GmAlloc(sizeof(int8_t) * a * stride);
  auto *y = (int8_t *)AscendC::GmAlloc(sizeof(int8_t) * a * b);
  int8_t expect[a][b];

  for (int i = 0; i < a; i++) {
    for (int z = 0; z < stride; z++) {
      x[i * stride + z] = (z == 0) ? static_cast<int8_t>(-i - 1) : sentinel;
    }
    for (int j = 0; j < b; j++) {
      expect[i][j] = static_cast<int8_t>(-i - 1);
    }
  }

  auto kernel = [](int a, int b, int stride, int8_t *x, int8_t *y) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int8_t) * a * stride);
    tpipe.InitBuffer(ybuf, sizeof(int8_t) * a * b);
    tpipe.InitBuffer(tmp, 16 * 1024);

    auto l_x = xbuf.Get<int8_t>();
    auto l_y = ybuf.Get<int8_t>();
    auto l_tmp = tmp.Get<uint8_t>();
    GmToUb(l_x, x, a * stride);
    GmToUb(l_y, y, a * b);
    Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp, stride);
    UbToGm(y, l_y, a * b);
  };

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, stride, x, y);

  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      if (y[i * b + j] != expect[i][j]) {
        diff_count++;
      }
    }
  }
  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

template <typename T>
static void CheckByteBroadcastStride(int rows, int columns, int stride, bool bool_values) {
  constexpr int guard = ONE_BLK_SIZE;
  const int input_size = KernelUtils::BlkAlign<T>(guard + rows * stride + guard);
  const int output_size = KernelUtils::BlkAlign<T>(guard + rows * columns + guard);
  // Match the byte-broadcast workspace contract in ascir/reg_func/broadcast.cpp.
  const int tmp_size =
      (KernelUtils::BlkAlign<half>(rows) + KernelUtils::BlkAlign<half>(rows * columns)) * sizeof(half) + 128 * 1024;
  auto *x = static_cast<T *>(AscendC::GmAlloc(input_size * sizeof(T)));
  auto *y = static_cast<T *>(AscendC::GmAlloc(output_size * sizeof(T)));
  const T values[] = {static_cast<T>(0), static_cast<T>(1), static_cast<T>(127), static_cast<T>(128),
                      static_cast<T>(255)};
  for (int i = 0; i < input_size; i++) {
    x[i] = static_cast<T>(37);
  }
  for (int row = 0; row < rows; row++) {
    x[guard + row * stride] = bool_values ? static_cast<T>(row % 2) : values[row % 5];
  }
  for (int i = 0; i < output_size; i++) {
    y[i] = static_cast<T>(73);
  }

  auto kernel = [](T *input, T *output, int input_size, int output_size, int tmp_size, int rows, int columns,
                   int stride) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> source, destination, temporary;
    pipe.InitBuffer(source, input_size * sizeof(T));
    pipe.InitBuffer(destination, output_size * sizeof(T));
    pipe.InitBuffer(temporary, tmp_size);
    auto src = source.Get<T>();
    auto dst = destination.Get<T>();
    auto tmp = temporary.Get<uint8_t>();
    GmToUb(src, input, input_size);
    GmToUb(dst, output, output_size);
    Broadcast(dst[ONE_BLK_SIZE], src[ONE_BLK_SIZE], rows, 1, 0, rows, columns, 0, tmp, stride);
    UbToGm(output, dst, output_size);
  };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, x, y, input_size, output_size, tmp_size, rows, columns, stride);
  size_t differences = 0;
  for (int i = 0; i < output_size; i++) {
    const T expected =
        i >= guard && i < guard + rows * columns ? x[guard + ((i - guard) / columns) * stride] : static_cast<T>(73);
    differences += y[i] != expected;
  }
  EXPECT_EQ(differences, 0U) << "rows=" << rows << ", columns=" << columns << ", stride=" << stride;
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, ByteStridePreservesValuesAndGuards) {
  for (int stride : {1, 2, 32}) {
    CheckByteBroadcastStride<uint8_t>(33, 32, stride, false);
    CheckByteBroadcastStride<int8_t>(33, 32, stride, false);
  }
}

TEST(TestApiBroadcast, BoolStridePaddedWidth) {
  CheckByteBroadcastStride<uint8_t>(5, 1024, 32, true);
}

// TEST(TestApiBroadcast, Test_a1_to_ab_int32) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32;
//   auto *x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * a * 1);
//   auto *y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * a * b);
//   int32_t expect[a][b];

//   for (uint32_t i = 0; i < a; i++) {
//     x[i] = i;
//     for (uint32_t j = 0; j < b; j++) {
//       expect[i][j] = i;
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, int32_t *x, int32_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * 1);
//     tpipe.InitBuffer(ybuf, sizeof(int32_t) * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<int32_t>();
//     auto l_y = ybuf.Get<int32_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, a);
//     GmToUb(l_y, y, a * b);

//     Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

//     UbToGm(y, l_y, a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       auto diff = (uint32_t)(y[i*b + j] - expect[i][j]);
//       if (diff < -1e-5 || diff > 1e-5) {
//         diff_count++;
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

TEST(TestApiBroadcast, Test_1b_to_ab_int32) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * 1 * b);
  auto *y = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b);
  int32_t expect[a][b];

  for (uint32_t j = 0; j < b; j++) {
    x[j] = j;
    for (uint32_t i = 0; i < a; i++) {
      expect[i][j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int32_t *x, int32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * 1 * b);
    tpipe.InitBuffer(ybuf, sizeof(int32_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, b, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1bc_to_abc_int32) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 8;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * 1 * b * c);
  auto *y = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b * c);
  int32_t expect[a][b][c];

  for (uint32_t j = 0; j < b; j++) {
    for (uint32_t k = 0; k < c; k++) {
      x[j * c + k] = j * c + k;
      for (uint32_t i = 0; i < a; i++) {
        expect[i][j][k] = j * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, int32_t *x, int32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * 1 * b * c);
    tpipe.InitBuffer(ybuf, sizeof(int32_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, 1, b, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1c_to_abc_int32) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 32;
  auto *x = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * 1 * c);
  auto *y = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * a * b * c);
  int32_t expect[a][b][c];

  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t k = 0; k < c; k++) {
      x[i * c + k] = i * c + k;
      for (uint32_t j = 0; j < b; j++) {
        expect[i][j][k] = i * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, int32_t *x, int32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int32_t) * a * 1 * c);
    tpipe.InitBuffer(ybuf, sizeof(int32_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int32_t>();
    auto l_y = ybuf.Get<int32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// TEST(TestApiBroadcast, Test_11_to_ab_int32) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32;
//   auto *x = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * 1 * 1);
//   auto *y = (int32_t*)AscendC::GmAlloc(sizeof(int32_t) * a * b);
//   int32_t expect[a][b];

//   x[0] = 3;
//   for (uint32_t i = 0; i < a; i++) {
//     for (uint32_t j = 0; j < b; j++) {
//       expect[i][j] = x[0];
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, int32_t *x, int32_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(int32_t) * 1 * 1);
//     tpipe.InitBuffer(ybuf, sizeof(int32_t) * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<int32_t>();
//     auto l_y = ybuf.Get<int32_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, 1);
//     GmToUb(l_y, y, a * b);

//     Broadcast(l_y, l_x, 1, 1, 0, a, b, 0, l_tmp);

//     UbToGm(y, l_y, a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
//       if (diff < -1e-5 || diff > 1e-5) {
//         diff_count++;
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

// TEST(TestApiBroadcast, Test_a1_to_ab_uint32) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32;
//   auto *x = (uint32_t*)AscendC::GmAlloc(sizeof(uint32_t) * a * 1);
//   auto *y = (uint32_t*)AscendC::GmAlloc(sizeof(uint32_t) * a * b);
//   uint32_t expect[a][b];

//   for (uint32_t i = 0; i < a; i++) {
//     x[i] = i;
//     for (uint32_t j = 0; j < b; j++) {
//       expect[i][j] = i;
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, uint32_t *x, uint32_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(uint32_t) * a * 1);
//     tpipe.InitBuffer(ybuf, sizeof(uint32_t) * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<uint32_t>();
//     auto l_y = ybuf.Get<uint32_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, a);
//     GmToUb(l_y, y, a * b);

//     Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

//     UbToGm(y, l_y, a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       auto diff = (uint32_t)(y[i*b + j] - expect[i][j]);
//       if (diff < -1e-5 || diff > 1e-5) {
//         diff_count++;
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

TEST(TestApiBroadcast, Test_1b_to_ab_uint32) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (uint32_t *)AscendC::GmAlloc(sizeof(uint32_t) * 1 * b);
  auto *y = (uint32_t *)AscendC::GmAlloc(sizeof(uint32_t) * a * b);
  uint32_t expect[a][b];

  for (uint32_t j = 0; j < b; j++) {
    x[j] = j;
    for (uint32_t i = 0; i < a; i++) {
      expect[i][j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, uint32_t *x, uint32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint32_t) * 1 * b);
    tpipe.InitBuffer(ybuf, sizeof(uint32_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint32_t>();
    auto l_y = ybuf.Get<uint32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, b, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1bc_to_abc_uint32) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 8;
  auto *x = (uint32_t *)AscendC::GmAlloc(sizeof(uint32_t) * 1 * b * c);
  auto *y = (uint32_t *)AscendC::GmAlloc(sizeof(uint32_t) * a * b * c);
  uint32_t expect[a][b][c];

  for (uint32_t j = 0; j < b; j++) {
    for (uint32_t k = 0; k < c; k++) {
      x[j * c + k] = j * c + k;
      for (uint32_t i = 0; i < a; i++) {
        expect[i][j][k] = j * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, uint32_t *x, uint32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint32_t) * 1 * b * c);
    tpipe.InitBuffer(ybuf, sizeof(uint32_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint32_t>();
    auto l_y = ybuf.Get<uint32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, 1, b, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1c_to_abc_uint32) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 32;
  auto *x = (uint32_t *)AscendC::GmAlloc(sizeof(uint32_t) * a * 1 * c);
  auto *y = (uint32_t *)AscendC::GmAlloc(sizeof(uint32_t) * a * b * c);
  uint32_t expect[a][b][c];

  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t k = 0; k < c; k++) {
      x[i * c + k] = i * c + k;
      for (uint32_t j = 0; j < b; j++) {
        expect[i][j][k] = i * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, uint32_t *x, uint32_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint32_t) * a * 1 * c);
    tpipe.InitBuffer(ybuf, sizeof(uint32_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint32_t>();
    auto l_y = ybuf.Get<uint32_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// TEST(TestApiBroadcast, Test_a1_to_ab_int16) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32;
//   auto *x = (int16_t*)AscendC::GmAlloc(sizeof(int16_t) * a * 1);
//   auto *y = (int16_t*)AscendC::GmAlloc(sizeof(int16_t) * a * b);
//   int16_t expect[a][b];

//   for (uint32_t i = 0; i < a; i++) {
//     x[i] = i;
//     for (uint32_t j = 0; j < b; j++) {
//       expect[i][j] = i;
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, int16_t *x, int16_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(int16_t) * a * 1);
//     tpipe.InitBuffer(ybuf, sizeof(int16_t) * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<int16_t>();
//     auto l_y = ybuf.Get<int16_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, a);
//     GmToUb(l_y, y, a * b);

//     Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

//     UbToGm(y, l_y, a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       auto diff = (uint32_t)(y[i*b + j] - expect[i][j]);
//       if (diff < -1e-5 || diff > 1e-5) {
//         diff_count++;
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

TEST(TestApiBroadcast, Test_1b_to_ab_int16) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (int16_t *)AscendC::GmAlloc(sizeof(int16_t) * 1 * b);
  auto *y = (int16_t *)AscendC::GmAlloc(sizeof(int16_t) * a * b);
  int16_t expect[a][b];

  for (uint32_t j = 0; j < b; j++) {
    x[j] = j;
    for (uint32_t i = 0; i < a; i++) {
      expect[i][j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int16_t *x, int16_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int16_t) * 1 * b);
    tpipe.InitBuffer(ybuf, sizeof(int16_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int16_t>();
    auto l_y = ybuf.Get<int16_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, b, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1bc_to_abc_int16) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 8;
  auto *x = (int16_t *)AscendC::GmAlloc(sizeof(int16_t) * 1 * b * c);
  auto *y = (int16_t *)AscendC::GmAlloc(sizeof(int16_t) * a * b * c);
  int16_t expect[a][b][c];

  for (uint32_t j = 0; j < b; j++) {
    for (uint32_t k = 0; k < c; k++) {
      x[j * c + k] = j * c + k;
      for (uint32_t i = 0; i < a; i++) {
        expect[i][j][k] = j * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, int16_t *x, int16_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int16_t) * 1 * b * c);
    tpipe.InitBuffer(ybuf, sizeof(int16_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int16_t>();
    auto l_y = ybuf.Get<int16_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, 1, b, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1c_to_abc_int16) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 32;
  auto *x = (int16_t *)AscendC::GmAlloc(sizeof(int16_t) * a * 1 * c);
  auto *y = (int16_t *)AscendC::GmAlloc(sizeof(int16_t) * a * b * c);
  int16_t expect[a][b][c];

  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t k = 0; k < c; k++) {
      x[i * c + k] = i * c + k;
      for (uint32_t j = 0; j < b; j++) {
        expect[i][j][k] = i * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, int16_t *x, int16_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(int16_t) * a * 1 * c);
    tpipe.InitBuffer(ybuf, sizeof(int16_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<int16_t>();
    auto l_y = ybuf.Get<int16_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// TEST(TestApiBroadcast, Test_11_to_ab_int16) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32;
//   auto *x = (int16_t*)AscendC::GmAlloc(sizeof(int16_t) * 1 * 1);
//   auto *y = (int16_t*)AscendC::GmAlloc(sizeof(int16_t) * a * b);
//   int16_t expect[a][b];

//   x[0] = 3;
//   for (uint32_t i = 0; i < a; i++) {
//     for (uint32_t j = 0; j < b; j++) {
//       expect[i][j] = x[0];
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, int16_t *x, int16_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(int16_t) * 1 * 1);
//     tpipe.InitBuffer(ybuf, sizeof(int16_t) * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<int16_t>();
//     auto l_y = ybuf.Get<int16_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, 1);
//     GmToUb(l_y, y, a * b);

//     Broadcast(l_y, l_x, 1, 1, 0, a, b, 0, l_tmp);

//     UbToGm(y, l_y, a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
//       if (diff < -1e-5 || diff > 1e-5) {
//         diff_count++;
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

// TEST(TestApiBroadcast, Test_a1_to_ab_uint16) {
//   // 构造测试输入和预期结果
//   int a = 16, b = 32;
//   auto *x = (uint16_t*)AscendC::GmAlloc(sizeof(uint16_t) * a * 1);
//   auto *y = (uint16_t*)AscendC::GmAlloc(sizeof(uint16_t) * a * b);
//   uint16_t expect[a][b];

//   for (uint32_t i = 0; i < a; i++) {
//     x[i] = i;
//     for (uint32_t j = 0; j < b; j++) {
//       expect[i][j] = i;
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, uint16_t *x, uint16_t *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(uint16_t) * a * 1);
//     tpipe.InitBuffer(ybuf, sizeof(uint16_t) * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<uint16_t>();
//     auto l_y = ybuf.Get<uint16_t>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, a);
//     GmToUb(l_y, y, a * b);

//     Broadcast(l_y, l_x, a, 1, 0, a, b, 0, l_tmp);

//     UbToGm(y, l_y, a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       auto diff = (uint32_t)(y[i*b + j] - expect[i][j]);
//       if (diff < -1e-5 || diff > 1e-5) {
//         diff_count++;
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }

TEST(TestApiBroadcast, Test_1b_to_ab_uint16) {
  // 构造测试输入和预期结果
  int a = 16, b = 32;
  auto *x = (uint16_t *)AscendC::GmAlloc(sizeof(uint16_t) * 1 * b);
  auto *y = (uint16_t *)AscendC::GmAlloc(sizeof(uint16_t) * a * b);
  uint16_t expect[a][b];

  for (uint32_t j = 0; j < b; j++) {
    x[j] = j;
    for (uint32_t i = 0; i < a; i++) {
      expect[i][j] = j;
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, uint16_t *x, uint16_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint16_t) * 1 * b);
    tpipe.InitBuffer(ybuf, sizeof(uint16_t) * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint16_t>();
    auto l_y = ybuf.Get<uint16_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b);
    GmToUb(l_y, y, a * b);

    Broadcast(l_y, l_x, 1, b, 0, a, b, 0, l_tmp);

    UbToGm(y, l_y, a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      auto diff = (uint32_t)(y[i * b + j] - expect[i][j]);
      if (diff < -1e-5 || diff > 1e-5) {
        diff_count++;
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1bc_to_abc_uint16) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 8;
  auto *x = (uint16_t *)AscendC::GmAlloc(sizeof(uint16_t) * 1 * b * c);
  auto *y = (uint16_t *)AscendC::GmAlloc(sizeof(uint16_t) * a * b * c);
  uint16_t expect[a][b][c];

  for (uint32_t j = 0; j < b; j++) {
    for (uint32_t k = 0; k < c; k++) {
      x[j * c + k] = j * c + k;
      for (uint32_t i = 0; i < a; i++) {
        expect[i][j][k] = j * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, uint16_t *x, uint16_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint16_t) * 1 * b * c);
    tpipe.InitBuffer(ybuf, sizeof(uint16_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint16_t>();
    auto l_y = ybuf.Get<uint16_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, b * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, 1, b, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1c_to_abc_uint16) {
  // 构造测试输入和预期结果
  int a = 16, b = 32, c = 32;
  auto *x = (uint16_t *)AscendC::GmAlloc(sizeof(uint16_t) * a * 1 * c);
  auto *y = (uint16_t *)AscendC::GmAlloc(sizeof(uint16_t) * a * b * c);
  uint16_t expect[a][b][c];

  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t k = 0; k < c; k++) {
      x[i * c + k] = i * c + k;
      for (uint32_t j = 0; j < b; j++) {
        expect[i][j][k] = i * c + k;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, int c, uint16_t *x, uint16_t *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(uint16_t) * a * 1 * c);
    tpipe.InitBuffer(ybuf, sizeof(uint16_t) * a * b * c);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<uint16_t>();
    auto l_y = ybuf.Get<uint16_t>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * c);
    GmToUb(l_y, y, a * b * c);

    Broadcast(l_y, l_x, a, 1, c, a, b, c, l_tmp);

    UbToGm(y, l_y, a * b * c);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, c, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < c; k++) {
        auto diff = (uint32_t)(y[i * b * c + j * c + k] - expect[i][j][k]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_a1a1_to_abab_half) {
  // 构造测试输入和预期结果
  int a = 2, b = 8;
  auto *x = (half *)AscendC::GmAlloc(sizeof(half) * a * 1 * a * 1);
  auto *y = (half *)AscendC::GmAlloc(sizeof(half) * a * b * a * b);
  half expect[a][b][a][b];

  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t k = 0; k < a; k++) {
      x[i * a + k] = i * a + k;
    }
  }

  for (uint32_t i = 0; i < a; i++) {
    for (uint32_t k = 0; k < b; k++) {
      for (uint32_t j = 0; j < a; j++) {
        for (uint32_t z = 0; z < b; z++) {
          expect[i][k][j][z] = i * a + j;
        }
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, half *x, half *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * a * 1 * a * 1);
    tpipe.InitBuffer(ybuf, sizeof(half) * a * b * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, a * 1 * a * 1);
    GmToUb(l_y, y, a * b * a * b);

    Broadcast(l_y, l_x, a, 1, a, 1, a, b, a, b, l_tmp);

    UbToGm(y, l_y, a * b * a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < a; k++) {
        for (int z = 0; z < b; z++) {
          auto diff = (uint32_t)(y[i * b * a * b + j * a * b + k * b + z] - expect[i][j][k][z]);
          if (diff < -1e-5 || diff > 1e-5) {
            diff_count++;
          }
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

TEST(TestApiBroadcast, Test_1a1_to_bab_half) {
  // 构造测试输入和预期结果
  int a = 2, b = 8;
  auto *x = (half *)AscendC::GmAlloc(sizeof(half) * 1 * a * 1);
  auto *y = (half *)AscendC::GmAlloc(sizeof(half) * b * a * b);
  half expect[b][a][b];

  for (uint32_t k = 0; k < a; k++) {
    x[k] = k;
  }

  for (uint32_t k = 0; k < b; k++) {
    for (uint32_t j = 0; j < a; j++) {
      for (uint32_t z = 0; z < b; z++) {
        expect[k][j][z] = j;
      }
    }
  }

  // 构造Api调用函数
  auto kernel = [](int a, int b, half *x, half *y) {
    // 1. 分配内存
    TPipe tpipe;
    TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
    tpipe.InitBuffer(xbuf, sizeof(half) * 1 * a * 1);
    tpipe.InitBuffer(ybuf, sizeof(half) * b * a * b);
    tpipe.InitBuffer(tmp, 8 * 1024);

    auto l_x = xbuf.Get<half>();
    auto l_y = ybuf.Get<half>();
    auto l_tmp = tmp.Get<uint8_t>();

    GmToUb(l_x, x, 1 * a * 1);
    GmToUb(l_y, y, b * a * b);

    Broadcast(l_y, l_x, 1, 1, a, 1, 1, b, a, b, l_tmp);

    UbToGm(y, l_y, b * a * b);
  };

  // 调用kernel
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1, a, b, x, y);

  // 验证结果
  int diff_count = 0;
  for (int j = 0; j < b; j++) {
    for (int k = 0; k < a; k++) {
      for (int z = 0; z < b; z++) {
        auto diff = (uint32_t)(y[j * a * b + k * b + z] - expect[j][k][z]);
        if (diff < -1e-5 || diff > 1e-5) {
          diff_count++;
        }
      }
    }
  }

  EXPECT_EQ(diff_count, 0);
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}

// TEST(TestApiBroadcast, Test_1b1b_to_abab_half) {
//   // 构造测试输入和预期结果
//   int a = 2, b = 32;
//   auto *x = (half*)AscendC::GmAlloc(sizeof(half) * 1 * b * 1 * b);
//   auto *y = (half*)AscendC::GmAlloc(sizeof(half) * a * b * a * b);
//   half expect[a][b][a][b];

//   for (uint32_t i = 0; i < b; i++) {
//     for (uint32_t k = 0; k < b; k++) {
//       x[i * b + k] = i * b + k;
//     }
//   }

//   for (uint32_t i = 0; i < a; i++) {
//     for (uint32_t k = 0; k < b; k++) {
//       for (uint32_t j = 0; j < a; j++) {
//         for (uint32_t z = 0; z < b; z++) {
//           expect[i][k][j][z] = k * b + z;
//         }
//       }
//     }
//   }

//   // 构造Api调用函数
//   auto kernel = [](int a, int b, half *x, half *y) {
//     // 1. 分配内存
//     TPipe tpipe;
//     TBuf<TPosition::VECCALC> xbuf, ybuf, tmp;
//     tpipe.InitBuffer(xbuf, sizeof(half) * 1 * b * 1 * b);
//     tpipe.InitBuffer(ybuf, sizeof(half) * a * b * a * b);
//     tpipe.InitBuffer(tmp, 8 * 1024);

//     auto l_x = xbuf.Get<half>();
//     auto l_y = ybuf.Get<half>();
//     auto l_tmp = tmp.Get<uint8_t>();

//     GmToUb(l_x, x, 1 * b * 1 * b);
//     GmToUb(l_y, y, a * b * a * b);

//     Broadcast(l_y, l_x, 1, b, 1, b, a, b, a, b, l_tmp);

//     UbToGm(y, l_y, a * b * a * b);
//   };

//   // 调用kernel
//   AscendC::SetKernelMode(KernelMode::AIV_MODE);
//   ICPU_RUN_KF(kernel, 1, a, b, x, y);

//   // 验证结果
//   int diff_count = 0;
//   for (int i = 0; i < a; i++) {
//     for (int j = 0; j < b; j++) {
//       for (int k = 0; k < a; k++) {
//         for (int z = 0; z < b; z++) {
//           auto diff = (uint32_t)(y[i * b * a * b + j * a * b + k * b + z] - expect[i][j][k][z]);
//           if (diff < -1e-5 || diff > 1e-5) {
//             diff_count++;
//           }
//         }
//       }
//     }
//   }

//   EXPECT_EQ(diff_count, 0);
// }
