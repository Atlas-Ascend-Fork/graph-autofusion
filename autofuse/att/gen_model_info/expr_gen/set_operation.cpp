/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "set_operation.h"
#include <unordered_map>
#include "common/checker.h"

namespace att {
std::vector<DimRange> SetOperation::Diff(DimRange &range1, DimRange &range2) {
  std::vector<DimRange> new_dim_ranges;
  if ((range1.lower_bound == range2.lower_bound) && (range1.upper_bound == range2.upper_bound)) {
    return new_dim_ranges;
  }
  Expr ori_lower_bound = range1.lower_bound;
  Expr ori_upper_bound = range1.upper_bound;
  Expr lower_bound = range2.lower_bound;
  Expr upper_bound = range2.upper_bound;

  DimRange lower_range;
  DimRange upper_range;
  lower_range.lower_bound = ori_lower_bound;
  lower_range.upper_bound = lower_bound;
  upper_range.lower_bound = upper_bound;
  upper_range.upper_bound = ori_upper_bound;

  if (lower_range.lower_bound == lower_range.upper_bound) {
    new_dim_ranges.emplace_back(upper_range);
    return new_dim_ranges;
  }
  if (upper_range.lower_bound == upper_range.upper_bound) {
    new_dim_ranges.emplace_back(lower_range);
    return new_dim_ranges;
  }
  new_dim_ranges.emplace_back(lower_range);
  new_dim_ranges.emplace_back(upper_range);
  return new_dim_ranges;
}

TensorRange SetOperation::Diff(TensorRange &range1, TensorRange &range2) {
  TensorRange new_range;
  if (range1.size() != range2.size()) {
    GELOGD("Diff left size is not the same as right size.");
    return new_range;
  }
  for (auto coordinate1 : range1) {
    for (auto coordinate2 : range2) {
      std::vector<std::unordered_map<uint32_t, DimRange>> range_map;
      for (size_t i = 0U; i < coordinate1.size(); i++) {
        std::unordered_map<uint32_t, DimRange> map;
        auto dim_range1 = coordinate1[i];
        auto dim_range2 = coordinate2[i];
        auto diff_dim_range = Diff(dim_range1, dim_range2);
        diff_dim_range.emplace_back(dim_range2);
        for (size_t j = 0U; j < diff_dim_range.size(); j++) {
          map[j] = diff_dim_range[j];
        }
        range_map.emplace_back(map);
      }
      std::vector<std::vector<uint32_t>> seq;
      for (auto &map : range_map) {
        std::vector<uint32_t> num;
        for (auto &iter : map) {
          num.push_back(iter.first);
        }
        seq.push_back(num);
      }
      std::vector<std::vector<uint32_t>> res;
      std::vector<uint32_t> tmp;
      ProductImplement(seq, res, 0U, tmp);

      for (size_t k = 0U; k < res.size(); k++) {
        Coordinates coordinate;
        std::vector<uint32_t> comb = res[k];
        for (size_t idx = 0U; idx < coordinate1.size(); idx++) {
          uint32_t seq_idx = comb[idx];
          auto dim_range = range_map[idx][seq_idx];
          coordinate.emplace_back(dim_range);
        }
        if (coordinate2 != coordinate) {
          new_range.emplace_back(coordinate);
        }
      }
    }
  }
  return new_range;
}

DimRange SetOperation::Intersection(DimRange &range1, DimRange &range2) {
  DimRange new_dim_range;
  Expr lower_bound = af::sym::Max(range1.lower_bound, range2.lower_bound);
  Expr upper_bound = af::sym::Min(range1.upper_bound, range2.upper_bound);

  new_dim_range.lower_bound = lower_bound;
  new_dim_range.upper_bound = upper_bound;
  return new_dim_range;
}

TensorRange SetOperation::Intersection(TensorRange &range1, TensorRange &range2) {
  TensorRange new_range;
  for (auto &coordinate1 : range1) {
    for (auto &coordinate2 : range2) {
      Coordinates new_coordinates;
      for (size_t i = 0U; i < coordinate1.size(); i++) {
        auto dim_range1 = coordinate1[i];
        auto dim_range2 = coordinate2[i];
        auto intersect_dim_range = Intersection(dim_range1, dim_range2);
        new_coordinates.emplace_back(intersect_dim_range);
      }
      new_range.emplace_back(new_coordinates);
    }
  }
  return new_range;
}

void SetOperation::ProductImplement(std::vector<std::vector<uint32_t>> &seq, std::vector<std::vector<uint32_t>> &res,
                                    uint32_t layer, std::vector<uint32_t> &tmp) {
  if (seq.size() != 0U) {
    if (layer < seq.size() - 1U) {
      for (size_t i = 0U; i < seq[layer].size(); i++) {
        std::vector<uint32_t> new_tmp;
        new_tmp.assign(tmp.begin(), tmp.end());
        new_tmp.push_back(seq[layer][i]);
        ProductImplement(seq, res, layer + 1U, new_tmp);
      }
    } else if (layer == seq.size() - 1U) {
      for (size_t j = 0U; j < seq[layer].size(); j++) {
        tmp.push_back(seq[layer][j]);
        res.push_back(tmp);
        tmp.pop_back();
      }
    }
  }
}

Expr SetOperation::SetComputation(TensorRange &range) {
  Expr total_num = CreateExpr(0U);
  for (auto &coordinate : range) {
    Expr num = CreateExpr(1U);
    for (auto &dim_range : coordinate) {
      num = af::sym::Mul(num, af::sym::Max(CreateExpr(0U), af::sym::Sub(dim_range.upper_bound, dim_range.lower_bound)));
    }
    total_num = af::sym::Add(total_num, num);
  }
  return total_num;
}
}  // namespace att
