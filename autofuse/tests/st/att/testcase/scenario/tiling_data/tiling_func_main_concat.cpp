/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "Concat_tiling_data.h"
#include "autofuse_tiling_func_common.h"
using namespace ::optiling;

void PrintResult(graph_normalTilingData &tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_tiling_key();
  std::cout << "get_tiling_key" << " = " << tiling_key << std::endl;
  if (tiling_key == 1101) {
    std::cout << "get_nbo_size" << " = " << tilingData.get_nbo_size() << std::endl;
    std::cout << "get_nio_size" << " = " << tilingData.get_nio_size() << std::endl;
    std::cout << "get_block_dim" << " = " << tilingData.get_block_dim() << std::endl;
    std::cout << "get_ub_size" << " = " << tilingData.get_ub_size() << std::endl;
    std::cout << "get_A" << " = " << tilingData.get_A() << std::endl;
    std::cout << "get_R" << " = " << tilingData.get_R() << std::endl;
    std::cout << "get_nio_tail_size" << " = " << tilingData.get_nio_tail_size() << std::endl;
    std::cout << "get_nio_loop_num" << " = " << tilingData.get_nio_loop_num() << std::endl;
    std::cout << "get_nbo_tail_block_nio_tail_size" << " = " << tilingData.get_nbo_tail_tile_nio_tail_size()
              << std::endl;
    std::cout << "get_nbo_tail_block_nio_loop_num" << " = " << tilingData.get_nbo_tail_tile_nio_loop_num() << std::endl;
    std::cout << "get_nbo_tail_size" << " = " << tilingData.get_nbo_tail_size() << std::endl;
    std::cout << "get_nbo_loop_num" << " = " << tilingData.get_nbo_loop_num() << std::endl;
  }

  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_normalTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_A(1536);
  tilingData.set_R(128);
  tilingData.set_BL(8);

  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(1000);
  tilingData.set_R(256);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(112);
  tilingData.set_R(512);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(1000);
  tilingData.set_R(544);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  tilingData.set_block_dim(64);
  tilingData.set_A(1000);
  tilingData.set_R(511);
  if (GetTiling(tilingData, -1, nullptr)) {
    PrintResult(tilingData);
  } else {
    std::cout << "addlayernorm tiling func execute failed." << std::endl;
    return -1;
  }

  return 0;
}
