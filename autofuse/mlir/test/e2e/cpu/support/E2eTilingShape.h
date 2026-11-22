/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_TILING_SHAPE_H_
#define AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_TILING_SHAPE_H_

#include <cstdint>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

namespace mlir::afir {
// A scheduled transfer addresses its tile as `(blockIdx * block_inner + round) * tile`,
// with a `select` picking the short length on the last tile. Each of those terms is
// only exercised under a particular tiling shape, and ATT decides that shape at run
// time from the shape, the core count and the UB budget. So a case can stop covering
// the term it was written for -- a shape change, a new ATT heuristic, a different UB
// budget -- and still pass every `EXPECT_FLOAT_EQ`, because the terms it no longer
// reaches are simply zero.
//
// TilingShape turns "this case covers the multi-core path" from a comment into
// something the case asserts about itself, and prints the observed shape either way.
struct TilingShape {
  // Cores the kernel is launched with; the `blockIdx` term is dead unless > 1.
  uint32_t block_dim = 0;
  // `<axis>Tb_size`: tiles per core, i.e. trips of the generated loop. The
  // induction-variable term is dead unless > 1.
  uint32_t block_inner = 0;
  // `<axis>t_size`: extent of the vectorized axis, i.e. elements per tile.
  uint32_t tile = 0;
  // Elements along the axis that `tile` and `block_inner` split.
  uint32_t element_count = 0;

  uint32_t TileCount() const {
    return tile == 0U ? 0U : (element_count + tile - 1U) / tile;
  }
  uint32_t TailElements() const {
    return tile == 0U ? 0U : element_count % tile;
  }
  bool HasTail() const {
    return TailElements() != 0U;
  }
};

// Prints the observed shape next to the coverage each term gets. Always call it:
// on a failure it is the first thing that explains which address term was live.
inline void ReportTilingShape(const std::string &case_name, const TilingShape &shape) {
  std::cout << "[  TILING  ] " << case_name << ": element_count=" << shape.element_count << " tile=" << shape.tile
            << " block_inner=" << shape.block_inner << " block_dim=" << shape.block_dim
            << " | tile_count=" << shape.TileCount() << " tail=" << shape.TailElements()
            << " | covers blockIdx=" << (shape.block_dim > 1U ? "yes" : "NO")
            << " round=" << (shape.block_inner > 1U ? "yes" : "NO")
            << " tail_select=" << (shape.HasTail() ? "yes" : "NO") << std::endl;
}

// The three predicates below are the executable form of "this case is here to
// cover X". Use the one the case was written for; a case that covers nothing
// interesting should say so rather than stay silent.
inline testing::AssertionResult CoversMultipleBlocks(const TilingShape &shape) {
  if (shape.block_dim > 1U) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure() << "block_dim=" << shape.block_dim
                                     << ", so the blockIdx term of the transfer offset is always 0 and this case no"
                                        " longer covers the multi-core path it was written for";
}

inline testing::AssertionResult CoversMultipleRounds(const TilingShape &shape) {
  if (shape.block_inner > 1U) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure() << "block_inner=" << shape.block_inner
                                     << ", so the tile loop runs at most once per core and this case no longer covers"
                                        " the induction-variable term of the transfer offset";
}

inline testing::AssertionResult CoversTailTile(const TilingShape &shape) {
  if (shape.HasTail()) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure() << "element_count=" << shape.element_count
                                     << " is a multiple of tile=" << shape.tile
                                     << ", so the tail `select` always picks the full length and this case no longer"
                                        " covers the short-tile path";
}
}  // namespace mlir::afir

// `add_mlir_e2e_cpu_case` passes the generated getter names down as macros, because
// several registrations share one test source while the generated field names differ
// per registration. Renaming a scheduled axis therefore breaks the build here rather
// than quietly dropping the case's coverage claim.
#if defined(AF_MLIR_E2E_CPU_TILE_ACCESSOR) && defined(AF_MLIR_E2E_CPU_BLOCK_INNER_ACCESSOR)
#define AF_E2E_TILING_SHAPE(tiling_data, block_dim, element_count)                                                    \
  ::mlir::afir::TilingShape {                                                                                         \
    (block_dim), (tiling_data).AF_MLIR_E2E_CPU_BLOCK_INNER_ACCESSOR(), (tiling_data).AF_MLIR_E2E_CPU_TILE_ACCESSOR(), \
        (element_count)                                                                                               \
  }
#else
// 没声明访问器的注册：切分形状观测不到，于是 `tile` 与 `block_inner` 留 0。这不是
// 「取默认值」，是照实说不知道——`TileCount` / `TailElements` 在 `tile == 0` 上本来就答 0，
// 打印出来的 round 与 tail_select 因此是 `NO`，不会把未知冒充成已覆盖。COVERS 这时也提不了：
// `AddCpuCase.cmake` 要求声明 COVERS 必须同时给 TILE_ACCESSOR，所以三条断言一条都不会读它们。
#define AF_E2E_TILING_SHAPE(tiling_data, block_dim, element_count) \
  ::mlir::afir::TilingShape {                                      \
    (block_dim), 0U, 0U, (element_count)                           \
  }
#endif

// One check per term the registration's COVERS list claims; a no-op for the terms it
// does not claim. A case registered with COVERS `none` asserts nothing here, which is
// the honest outcome: its offset is identically zero and it is testing something else.
#ifdef AF_MLIR_E2E_CPU_COVERS_MULTI_BLOCK
#define AF_E2E_CHECK_MULTI_BLOCK_(shape) EXPECT_TRUE(::mlir::afir::CoversMultipleBlocks(shape))
#else
#define AF_E2E_CHECK_MULTI_BLOCK_(shape) ((void)0)
#endif

#ifdef AF_MLIR_E2E_CPU_COVERS_MULTI_ROUND
#define AF_E2E_CHECK_MULTI_ROUND_(shape) EXPECT_TRUE(::mlir::afir::CoversMultipleRounds(shape))
#else
#define AF_E2E_CHECK_MULTI_ROUND_(shape) ((void)0)
#endif

#ifdef AF_MLIR_E2E_CPU_COVERS_TAIL
#define AF_E2E_CHECK_TAIL_(shape) EXPECT_TRUE(::mlir::afir::CoversTailTile(shape))
#else
#define AF_E2E_CHECK_TAIL_(shape) ((void)0)
#endif

#define AF_E2E_ASSERT_TILING_COVERAGE(shape) \
  do {                                       \
    AF_E2E_CHECK_MULTI_BLOCK_(shape);        \
    AF_E2E_CHECK_MULTI_ROUND_(shape);        \
    AF_E2E_CHECK_TAIL_(shape);               \
  } while (false)

#endif  // AUTOFUSE_MLIR_TEST_E2E_CPU_SUPPORT_E2E_TILING_SHAPE_H_
