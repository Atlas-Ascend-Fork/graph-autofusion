/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nddma_model.h"

#include <limits>
#include <unordered_set>
#include "base/att_const_values.h"
#include "common_utils.h"
#include "framework/common/debug/ge_log.h"

namespace att {
namespace {
constexpr size_t kMinNddmaRank = 1U;
constexpr size_t kMaxNddmaRank = 5U;
constexpr uint64_t kLowCoreUpperBound = 2U;
constexpr uint64_t kInputStrideUpperBound = 128U;
constexpr uint64_t kCalibratedNg2InnerBytesThreshold = 128U;

struct Nddma1DParams {
  double t1;
  double h1;
  double t2;
  double h2;
  double a1, a2, b1, b2, b3, b4, c1, c2, c3, c4;
};

struct NddmaNg2Params {
  double g10;
  double g11_m;
  double g00;
  double g01_m;
};

struct NddmaNg2Config {
  size_t n_pos;
  size_t m_pos;
};

const Nddma1DParams *GetNddma1DParams(uint64_t dtype_size) {
  static constexpr Nddma1DParams kB8 = {11.7626,     194.421,      6.05735,     373.274,      2.2901165,
                                        0.014561351, -69.410121,   -2.0861213,  6.9111296,    -0.014567499,
                                        2.2823982,   0.0054381207, -0.29869463, -0.0054311976};
  static constexpr Nddma1DParams kB16 = {25.7579,      204.604,     13.259,     399.909,     1.9720487,
                                         0.0073653238, -65.754578,  -1.7626747, 3.4565828,   -0.0073719789,
                                         1.6501023,    0.016002982, 0.33029872, -0.015981633};
  static constexpr Nddma1DParams kB32 = {57.2624,      235.137,      29.4096,     453.859,    1.2140303,
                                         0.0037227686, 31.174096,    -0.83730451, 0.21970194, -0.0032801415,
                                         1.9774169,    0.0056400644, -0.61376163, 0.021569482};
  static constexpr Nddma1DParams kB64 = {57.2346,      243.205,    29.3906,    468.971,     1.7169033,
                                         0.0015857085, 63.566231,  -1.4367614, 0.090304855, -0.0012246069,
                                         0.76071525,   0.02630908, 0.89945693, -0.015756802};
  switch (dtype_size) {
    case 1U:
      return &kB8;
    case 2U:
      return &kB16;
    case 4U:
      return &kB32;
    case 8U:
      return &kB64;
    default:
      return nullptr;
  }
}

const NddmaNg2Params *GetNddmaNg2Params(uint64_t dtype_size) {
  static constexpr NddmaNg2Params kB8 = {-0.117205844044, 0.00527570594895, 2.43399203072, 0.0358600644964};
  static constexpr NddmaNg2Params kB16 = {-0.0808448415704, 0.00809434315612, 2.09290858265, 0.0538904957722};
  static constexpr NddmaNg2Params kB32 = {-0.148267095023, 0.0350078015331, 2.65419909697, 0.0320943041806};
  static constexpr NddmaNg2Params kB64 = {0.183728548508, 0.0297912599903, 0.138100895716, 0.364174713053};
  switch (dtype_size) {
    case 1U:
      return &kB8;
    case 2U:
      return &kB16;
    case 4U:
      return &kB32;
    case 8U:
      return &kB64;
    default:
      return nullptr;
  }
}

// Direct1D whole-API refit. Keep the existing bilinear arithmetic and the
// legacy higher-rank/NG2 residual tables separate: a fitted direct1D residual
// is not an independently calibrated higher-rank building block.
const Nddma1DParams *GetCalibrated1DParams(uint64_t dtype_size) {
  static constexpr Nddma1DParams kSize1 = {19.804176287674611,
                                           383.18507637562016,
                                           7.3622873632996821,
                                           410.08277965993307,
                                           0,
                                           0.01638919820216082,
                                           0,
                                           0,
                                           6.3500085339606187,
                                           -0.015902012318163614,
                                           2.1593534546872641,
                                           0.0016113600504118414,
                                           -0.076526252749587992,
                                           -0.002081040746331446};
  static constexpr Nddma1DParams kSize2 = {110.99512994171366,
                                           372.03538613769922,
                                           20.880243660515681,
                                           392.11553775116982,
                                           0,
                                           0.008480690419577443,
                                           0,
                                           0,
                                           2.8654879909598114,
                                           -0.0067546979518560354,
                                           2.1985688324677297,
                                           -0.00050001547789896381,
                                           -0.024133022925236149,
                                           -0.0010298012577624018};
  // The constrained 4/8-byte physical 1D fits reach zero inverse bandwidth in the low-core branch.
  // A finite t1 of 1e30 makes bytes/t1 negligible while retaining startup and stride costs.
  static constexpr Nddma1DParams kSize4 = {1e+30,
                                           399.79451701846955,
                                           111.30186040884911,
                                           380.20779936092111,
                                           0,
                                           0.0041808859558160605,
                                           -0.0032385006575487434,
                                           0.00080962516438718584,
                                           0.16995157378087231,
                                           -0.00022404971274912235,
                                           2.2671010691747822,
                                           0.00045939527932410191,
                                           -0.37127013690741806,
                                           0.000866140142348487};
  static constexpr Nddma1DParams kSize8 = {1e+30,
                                           419.59765761088846,
                                           89.728783044552372,
                                           411.62148274570325,
                                           0,
                                           0.001964136704570765,
                                           -0.039251990033263379,
                                           0.0049064987541579223,
                                           0.072867233427661618,
                                           -0.00046470078288990109,
                                           2.1762692322415575,
                                           0.0035470893458878919,
                                           -0.12573210312713756,
                                           -0.0012697028815567149};
  switch (dtype_size) {
    case 1U:
      return &kSize1;
    case 2U:
      return &kSize2;
    case 4U:
      return &kSize4;
    case 8U:
      return &kSize8;
    default:
      return nullptr;
  }
}

bool BuildPhysical1DView(const NddmaNormalizedDesc &normalized, NddmaNormalizedDesc &physical) {
  const size_t rank = normalized.output_dims.size();
  size_t first = rank;
  Expr elements = af::sym::kSymbolOne;
  for (size_t pos = 0U; pos < rank; ++pos) {
    if (af::SymbolicUtils::StaticCheckEq(normalized.output_dims[pos], af::sym::kSymbolOne) == af::TriBool::kTrue) {
      continue;
    }
    if (first == rank) {
      first = pos;
    } else if (af::SymbolicUtils::StaticCheckEq(normalized.input_strides[pos],
                                                normalized.input_strides[first] * elements) != af::TriBool::kTrue ||
               af::SymbolicUtils::StaticCheckEq(normalized.output_strides[pos],
                                                normalized.output_strides[first] * elements) != af::TriBool::kTrue) {
      return false;
    }
    elements = elements * normalized.output_dims[pos];
  }
  if (first == rank ||
      af::SymbolicUtils::StaticCheckEq(normalized.input_strides[first], af::sym::kSymbolZero) == af::TriBool::kTrue) {
    return false;
  }
  // This is a cost-only view. Preserve the actual descriptor/rank for logging.
  // Unit axes and jointly contiguous axes do not create extra DMA payload.
  physical = NddmaNormalizedDesc{};
  physical.effective_rank = 1U;
  physical.output_dims = {elements};
  physical.input_strides = {normalized.input_strides[first]};
  physical.output_strides = {normalized.output_strides[first]};
  physical.vectorized_axis = {normalized.vectorized_axis[first]};
  return true;
}

struct CalibratedNg2Params {
  double small_inner[4];
  double large_inner[4];
};

const CalibratedNg2Params *GetCalibratedNg2Params(uint64_t dtype_size) {
  static constexpr CalibratedNg2Params kSize1 = {
      {454.62985744360179, 2883.0433674319024, 2539.5240826150948, 0},
      {155.86615533000824, 16992.537142363326, 502.18408077094404, 1567.2702352776323}};
  static constexpr CalibratedNg2Params kSize2 = {
      {454.57389906919423, 1139.3068621999473, 1324.1407381801332, 208.02520354552965},
      {171.48824711051748, 8983.8345837638135, 28.059865391365431, 35.194618896097147}};
  static constexpr CalibratedNg2Params kSize4 = {
      {477.67750975920245, 415.45180464192788, 567.57933302664685, 258.11080050891985},
      {194.04230121257388, 4640.0150678912751, 189.36469434493978, 359.00573918803889}};
  switch (dtype_size) {
    case 1U:
      return &kSize1;
    case 2U:
      return &kSize2;
    case 4U:
      return &kSize4;
    default:
      return nullptr;
  }
}

bool HasValidVectorLengths(const NddmaDescriptorInfo &descriptor) {
  const size_t rank = descriptor.output_dims.size();
  return rank == descriptor.input_strides.size() && rank == descriptor.output_strides.size();
}

bool HasValidAxisOrder(const NddmaDescriptorInfo &descriptor) {
  if (descriptor.vectorized_axis.size() != descriptor.output_dims.size()) {
    return false;
  }
  std::unordered_set<int64_t> unique_axes;
  for (const auto axis : descriptor.vectorized_axis) {
    if (!unique_axes.emplace(axis).second) {
      return false;
    }
  }
  return true;
}

bool IsStaticNonPositive(const Expr &expr) {
  return expr.IsConstExpr() && af::SymbolicUtils::StaticCheckLe(expr, af::sym::kSymbolZero) == af::TriBool::kTrue;
}

bool IsStaticNegative(const Expr &expr) {
  return expr.IsConstExpr() && af::SymbolicUtils::StaticCheckLt(expr, af::sym::kSymbolZero) == af::TriBool::kTrue;
}

bool IsStaticallyEqualOrUnknown(const Expr &lhs, const Expr &rhs) {
  return af::SymbolicUtils::StaticCheckEq(lhs, rhs) != af::TriBool::kFalse;
}

bool HasInvalidStaticValue(const NddmaNormalizedDesc &descriptor) {
  for (const auto &dim : descriptor.output_dims) {
    if (!IsValid(dim) || IsStaticNonPositive(dim)) {
      return true;
    }
  }
  for (const auto &stride : descriptor.input_strides) {
    if (!IsValid(stride) || IsStaticNegative(stride)) {
      return true;
    }
  }
  for (const auto &stride : descriptor.output_strides) {
    if (!IsValid(stride) || IsStaticNonPositive(stride)) {
      return true;
    }
  }
  return false;
}

bool GetDtypeSize(const std::string &dtype, uint64_t &dtype_size) {
  const auto iter = kDataTypeSizeMap.find(dtype);
  return iter != kDataTypeSizeMap.end() && iter->second.GetConstValue(dtype_size);
}

bool HasStaticByteCountOverflow(const Expr &dim, uint64_t dtype_size) {
  if (dtype_size == 0U) {
    return false;
  }
  int64_t dim_value = 0;
  if (!dim.IsConstExpr() || !dim.GetConstValue(dim_value) || dim_value <= 0) {
    return false;
  }
  return static_cast<uint64_t>(dim_value) > std::numeric_limits<uint64_t>::max() / dtype_size;
}

NddmaFallbackReason ValidateNddmaDescriptorInput(const TensorShapeInfo &shape_info,
                                                 const std::vector<int64_t> &vectorized_axis,
                                                 const std::vector<bool> &tile_inner) {
  if (shape_info.repeats.empty()) {
    return NddmaFallbackReason::kNoDescriptor;
  }
  if (shape_info.repeats.size() != shape_info.gm_strides.size() ||
      shape_info.repeats.size() != shape_info.strides.size()) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  if (vectorized_axis.empty() || vectorized_axis.size() != shape_info.repeats.size()) {
    return NddmaFallbackReason::kCodegenMismatch;
  }
  if (!tile_inner.empty() && tile_inner.size() != shape_info.repeats.size()) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  return NddmaFallbackReason::kNone;
}

void BuildNddmaEffectiveView(const TensorShapeInfo &shape_info, const std::vector<bool> &tile_inner,
                             std::vector<Expr> &effective_repeats, std::vector<Expr> &effective_gm_strides,
                             std::vector<Expr> &effective_ub_strides, std::vector<int64_t> &effective_axes) {
  const auto &origin_repeats =
      shape_info.origin_repeats.size() == shape_info.repeats.size() ? shape_info.origin_repeats : shape_info.repeats;
  const size_t rank = origin_repeats.size();
  Expr prev_repeat = CreateExpr(1);
  Expr prev_actual_repeat = CreateExpr(1);
  Expr prev_gm_stride = CreateExpr(1);
  Expr prev_ub_stride = CreateExpr(1);
  bool has_non_zero_axis = false;
  for (size_t pos = rank; pos-- > 0U;) {
    const bool ignore_zero_axis =
        ascgen_utils::ShouldIgnoreDataCopyZeroAxis(has_non_zero_axis, pos, shape_info.strides);
    const bool zero_axis =
        af::SymbolicUtils::StaticCheckEq(shape_info.gm_strides[pos], CreateExpr(0)) == af::TriBool::kTrue &&
        af::SymbolicUtils::StaticCheckEq(shape_info.strides[pos], CreateExpr(0)) == af::TriBool::kTrue &&
        ignore_zero_axis;
    if (zero_axis) {
      continue;
    }
    has_non_zero_axis = true;
    const Expr cur_gm_stride = prev_gm_stride * prev_repeat;
    const Expr cur_ub_stride = prev_ub_stride * prev_repeat;
    const bool tile_boundary = !tile_inner.empty() && pos + 1U < rank && tile_inner[pos + 1U];
    if (!ascgen_utils::IsDataCopyAxisContinuous(cur_gm_stride, cur_ub_stride, shape_info.gm_strides[pos],
                                                shape_info.strides[pos]) ||
        effective_repeats.empty() || tile_boundary) {
      effective_repeats.emplace_back(shape_info.repeats[pos]);
      effective_gm_strides.emplace_back(shape_info.gm_strides[pos]);
      effective_ub_strides.emplace_back(shape_info.strides[pos]);
      effective_axes.emplace_back(static_cast<int64_t>(pos));
      prev_gm_stride = shape_info.gm_strides[pos];
      prev_ub_stride = shape_info.strides[pos];
      prev_repeat = origin_repeats[pos];
      prev_actual_repeat = shape_info.repeats[pos];
      continue;
    }
    const Expr product = origin_repeats[pos] * prev_repeat;
    effective_repeats.back() = shape_info.repeats[pos] * prev_actual_repeat;
    prev_repeat = product;
    prev_actual_repeat = shape_info.repeats[pos] * prev_actual_repeat;
  }
  std::reverse(effective_repeats.begin(), effective_repeats.end());
  std::reverse(effective_gm_strides.begin(), effective_gm_strides.end());
  std::reverse(effective_ub_strides.begin(), effective_ub_strides.end());
  std::reverse(effective_axes.begin(), effective_axes.end());
}

NddmaFallbackReason MapNddmaEffectiveAxes(std::vector<int64_t> &effective_axes,
                                          const std::vector<int64_t> &vectorized_axis) {
  for (auto &axis : effective_axes) {
    if (axis < 0 || static_cast<size_t>(axis) >= vectorized_axis.size()) {
      return NddmaFallbackReason::kCodegenMismatch;
    }
    axis = vectorized_axis[static_cast<size_t>(axis)];
  }
  return NddmaFallbackReason::kNone;
}

Expr AbsExpr(const Expr &value) {
  return af::sym::Max(value, af::sym::kSymbolZero - value);
}

Expr BuildNddma1DResidual(const Nddma1DParams &p, const Expr &bytes, const Expr &input_stride,
                          const Expr &output_stride, uint64_t dtype_size, bool high_core) {
  const Expr s = af::sym::Min(CreateExpr(kInputStrideUpperBound), input_stride * CreateExpr(dtype_size));
  const Expr g = af::sym::Min(af::sym::kSymbolOne, output_stride - af::sym::kSymbolOne);
  const Expr ng = (CreateExpr(p.a1) + CreateExpr(p.a2) * bytes) * s;
  const Expr ngu = (CreateExpr(p.b1) + CreateExpr(p.b2) * s + (CreateExpr(p.b3) + CreateExpr(p.b4) * s) * bytes) * g;
  const Expr rho = CreateExpr(p.c1) + CreateExpr(p.c2) * s + g * (CreateExpr(p.c3) + CreateExpr(p.c4) * s);
  return (ng + ngu) * (high_core ? rho : CreateExpr(1));
}

af::Status SelectCoreBranch(const Expr &block_dim, const Expr &low_core, const Expr &high_core,
                            NddmaModelResult &result) {
  if (af::SymbolicUtils::StaticCheckLe(block_dim, CreateExpr(kLowCoreUpperBound)) == af::TriBool::kTrue) {
    result.cycles = low_core;
    return af::SUCCESS;
  }
  if (af::SymbolicUtils::StaticCheckGt(block_dim, CreateExpr(kLowCoreUpperBound)) == af::TriBool::kTrue) {
    result.cycles = high_core;
    return af::SUCCESS;
  }
  auto low_case = std::make_shared<IfCase>(low_core);
  auto high_case = std::make_shared<IfCase>(high_core);
  GE_ASSERT_NOTNULL(low_case);
  GE_ASSERT_NOTNULL(high_case);
  GetPerfVar("nddma_1d_multicore", result.cycles, result.ternary_ops);
  TernaryOp ternary(CondType::K_LE, block_dim, CreateExpr(kLowCoreUpperBound), std::move(low_case),
                    std::move(high_case));
  ternary.SetVariable(result.cycles);
  result.ternary_ops[result.cycles] = std::move(ternary);
  return af::SUCCESS;
}

NddmaFallbackReason BuildNddmaCoreCycles(const NddmaNormalizedDesc &normalized, const Nddma1DParams &params,
                                         uint64_t dtype_size, Expr &low_core, Expr &high_core) {
  Expr total_elements = CreateExpr(1);
  for (const auto &dim : normalized.output_dims) {
    total_elements = total_elements * dim;
  }
  if (HasStaticByteCountOverflow(total_elements, dtype_size)) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  const Expr total_bytes = total_elements * CreateExpr(dtype_size);
  low_core = total_bytes / CreateExpr(params.t1) + CreateExpr(params.h1);
  high_core = total_bytes / CreateExpr(params.t2) + CreateExpr(params.h2);
  Expr input_prefix = af::sym::kSymbolZero;
  Expr output_prefix = af::sym::kSymbolZero;
  Expr has_inner_axis = af::sym::kSymbolZero;
  for (size_t i = 0; i < normalized.effective_rank; ++i) {
    // Valid dimensions are positive integers. A unit axis performs no stride advance,
    // including when a symbolic tiling dimension evaluates to one at runtime.
    const Expr active = af::sym::Min(af::sym::kSymbolOne, normalized.output_dims[i] - af::sym::kSymbolOne);
    const Expr input_stride =
        (af::sym::kSymbolOne - has_inner_axis) * normalized.input_strides[i] +
        has_inner_axis * (AbsExpr(normalized.input_strides[i] - input_prefix) + af::sym::kSymbolOne);
    const Expr output_stride =
        (af::sym::kSymbolOne - has_inner_axis) * normalized.output_strides[i] +
        has_inner_axis * (AbsExpr(normalized.output_strides[i] - output_prefix) + af::sym::kSymbolOne);
    Expr suffix_elements = CreateExpr(1);
    for (size_t suffix = i; suffix < normalized.effective_rank; ++suffix) {
      suffix_elements = suffix_elements * normalized.output_dims[suffix];
    }
    const Expr bytes = suffix_elements * CreateExpr(dtype_size);
    low_core = low_core + active * BuildNddma1DResidual(params, bytes, input_stride, output_stride, dtype_size, false);
    high_core = high_core + active * BuildNddma1DResidual(params, bytes, input_stride, output_stride, dtype_size, true);
    input_prefix = input_prefix + active * normalized.output_dims[i] * normalized.input_strides[i];
    output_prefix = output_prefix + active * normalized.output_dims[i] * normalized.output_strides[i];
    has_inner_axis = af::sym::Min(af::sym::kSymbolOne, has_inner_axis + active);
  }
  return NddmaFallbackReason::kNone;
}

bool IsNg2Eligible(const NddmaNormalizedDesc &normalized, NddmaNg2Config &config, bool require_packed = true) {
  std::vector<size_t> non_unit_positions;
  for (size_t pos = 0U; pos < normalized.output_dims.size(); ++pos) {
    // A symbolic tiling dimension cannot always be proven greater than one. Treat only statically unit dimensions
    // as excluded; codegen guarantees active tiling dimensions are represented by this path.
    if (af::SymbolicUtils::StaticCheckLe(normalized.output_dims[pos], af::sym::kSymbolOne) != af::TriBool::kTrue) {
      non_unit_positions.emplace_back(pos);
    }
  }
  if (non_unit_positions.size() != 2U) {
    return false;
  }
  config.n_pos = non_unit_positions[0U];
  config.m_pos = non_unit_positions[1U];
  const auto &dims = normalized.output_dims;
  const auto &input_strides = normalized.input_strides;
  const auto &output_strides = normalized.output_strides;
  const auto input_order = af::SymbolicUtils::StaticCheckLt(input_strides[config.m_pos], input_strides[config.n_pos]);
  const auto output_unit = af::SymbolicUtils::StaticCheckEq(output_strides[config.n_pos], af::sym::kSymbolOne);
  const auto output_dim_eq = af::SymbolicUtils::StaticCheckEq(output_strides[config.m_pos], dims[config.n_pos]);
  GELOGD("[ATT NDDMA] NG2 eligibility: n_pos=%zu, m_pos=%zu, input_order=%d, output_unit=%d, output_dim_eq=%d",
         config.n_pos, config.m_pos, static_cast<int>(input_order), static_cast<int>(output_unit),
         static_cast<int>(output_dim_eq));
  return input_order == af::TriBool::kTrue && output_unit == af::TriBool::kTrue &&
         (!require_packed || IsStaticallyEqualOrUnknown(output_strides[config.m_pos], dims[config.n_pos]));
}

NddmaFallbackReason BuildNg2Cycles(const NddmaNormalizedDesc &normalized, const Nddma1DParams &params,
                                   const NddmaNg2Params &ng2_params, uint64_t dtype_size, Expr &low_core,
                                   Expr &high_core, const NddmaNg2Config &config, bool generate_high_core = true) {
  const Expr m = normalized.output_dims[config.m_pos];
  const Expr n = normalized.output_dims[config.n_pos];
  const Expr total_bytes = m * n * CreateExpr(dtype_size);
  if (HasStaticByteCountOverflow(m * n, dtype_size)) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  const Expr ng2 =
      (CreateExpr(ng2_params.g10) + CreateExpr(ng2_params.g11_m) * m) * normalized.input_strides[config.m_pos] +
      CreateExpr(ng2_params.g00) + CreateExpr(ng2_params.g01_m) * m;
  if (IsStaticNonPositive(ng2)) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  const Expr b1 = n * CreateExpr(dtype_size);
  const Expr low_ng1 =
      BuildNddma1DResidual(params, b1, normalized.input_strides[config.n_pos], CreateExpr(1), dtype_size, false);
  low_core = total_bytes / CreateExpr(params.t1) + CreateExpr(params.h1) + low_ng1 * ng2;
  if (generate_high_core) {
    const Expr high_ng1 =
        BuildNddma1DResidual(params, b1, normalized.input_strides[config.n_pos], CreateExpr(1), dtype_size, true);
    high_core = total_bytes / CreateExpr(params.t2) + CreateExpr(params.h2) + high_ng1 * ng2;
  }
  return NddmaFallbackReason::kNone;
}

bool CanUseCalibratedNg2(const NddmaNormalizedDesc &normalized, const NddmaNg2Config &config, const Expr &block_dim,
                         bool require_packed = true) {
  // The collected API is source-unit-stride and packed UB. Preserve the
  // existing model for padded/uncertain layouts and the unmeasured low cores.
  return af::SymbolicUtils::StaticCheckLe(block_dim, CreateExpr(kLowCoreUpperBound)) != af::TriBool::kTrue &&
         af::SymbolicUtils::StaticCheckEq(normalized.input_strides[config.m_pos], af::sym::kSymbolOne) ==
             af::TriBool::kTrue &&
         (!require_packed ||
          af::SymbolicUtils::StaticCheckEq(normalized.output_strides[config.m_pos],
                                           normalized.output_dims[config.n_pos]) == af::TriBool::kTrue) &&
         af::SymbolicUtils::StaticCheckLt(normalized.input_strides[config.n_pos],
                                          normalized.output_dims[config.m_pos]) != af::TriBool::kTrue;
}

af::Status BuildCalibratedNg2High(const NddmaNormalizedDesc &normalized, const NddmaNg2Config &config,
                                  const CalibratedNg2Params &params, uint64_t dtype_size, const Expr &block_dim,
                                  Expr &high_core, NddmaModelResult &result) {
  const Expr m = normalized.output_dims[config.m_pos];
  const Expr n = normalized.output_dims[config.n_pos];
  const Expr bytes = m * n * CreateExpr(dtype_size);
  const Expr shared = af::sym::Max(block_dim - CreateExpr(32), af::sym::kSymbolZero) / CreateExpr(24);
  const Expr b_small = bytes / CreateExpr(16384);
  const Expr b_large = bytes / CreateExpr(65536);
  const Expr gap =
      af::sym::Min(af::sym::kSymbolOne, af::sym::Max(normalized.input_strides[config.n_pos] - m, af::sym::kSymbolZero));
  const auto &p = params.small_inner;
  const auto &q = params.large_inner;
  const Expr small =
      CreateExpr(p[0]) + CreateExpr(p[1]) * b_small +
      CreateExpr(p[2]) * af::sym::Max(bytes - CreateExpr(16384), af::sym::kSymbolZero) / CreateExpr(16384) +
      CreateExpr(p[3]) * shared * b_small;
  const Expr large = CreateExpr(q[0]) + CreateExpr(q[1]) * b_large + CreateExpr(q[2]) * shared * b_large +
                     CreateExpr(q[3]) * shared * b_large * gap;
  const Expr inner_bytes = n * CreateExpr(dtype_size);
  if (af::SymbolicUtils::StaticCheckLt(inner_bytes, CreateExpr(kCalibratedNg2InnerBytesThreshold)) ==
      af::TriBool::kTrue) {
    high_core = small;
  } else if (af::SymbolicUtils::StaticCheckGe(inner_bytes, CreateExpr(kCalibratedNg2InnerBytesThreshold)) ==
             af::TriBool::kTrue) {
    high_core = large;
  } else {
    GetPerfVar("nddma_ng2_inner", high_core, result.ternary_ops);
    TernaryOp ternary(CondType::K_LT, inner_bytes, CreateExpr(kCalibratedNg2InnerBytesThreshold),
                      std::make_shared<IfCase>(small), std::make_shared<IfCase>(large));
    ternary.SetVariable(high_core);
    result.ternary_ops[high_core] = std::move(ternary);
  }
  return af::SUCCESS;
}

}  // namespace

const char *NddmaFallbackReasonToString(NddmaFallbackReason reason) {
  static constexpr const char *kReasons[] = {
      "none",           "no_descriptor",    "rank_unsupported",   "schema_mismatch", "dtype_unsupported",
      "stride_invalid", "codegen_mismatch", "no_registered_model"};
  const auto index = static_cast<size_t>(reason);
  return index < (sizeof(kReasons) / sizeof(kReasons[0])) ? kReasons[index] : "unknown";
}

NddmaFallbackReason BuildNddmaDescriptor(const TensorShapeInfo &shape_info, const std::vector<int64_t> &vectorized_axis,
                                         NddmaDescriptorInfo &descriptor, const std::vector<bool> &tile_inner) {
  descriptor = NddmaDescriptorInfo{};
  const auto input_reason = ValidateNddmaDescriptorInput(shape_info, vectorized_axis, tile_inner);
  if (input_reason != NddmaFallbackReason::kNone) {
    return input_reason;
  }
  std::vector<Expr> effective_repeats;
  std::vector<Expr> effective_gm_strides;
  std::vector<Expr> effective_ub_strides;
  std::vector<int64_t> effective_axes;
  BuildNddmaEffectiveView(shape_info, tile_inner, effective_repeats, effective_gm_strides, effective_ub_strides,
                          effective_axes);
  const auto axis_reason = MapNddmaEffectiveAxes(effective_axes, vectorized_axis);
  if (axis_reason != NddmaFallbackReason::kNone) {
    return axis_reason;
  }
  descriptor.output_dims = std::move(effective_repeats);
  descriptor.input_strides = std::move(effective_gm_strides);
  descriptor.output_strides = std::move(effective_ub_strides);
  descriptor.vectorized_axis = std::move(effective_axes);
  GELOGD(
      "[ATT NDDMA] effective view: raw_rank=%zu, effective_rank=%zu, repeats=[%s], gm_strides=[%s], ub_strides=[%s], "
      "axes=[%s]",
      shape_info.repeats.size(), descriptor.output_dims.size(), GetVecString(descriptor.output_dims).c_str(),
      GetVecString(descriptor.input_strides).c_str(), GetVecString(descriptor.output_strides).c_str(),
      ascgen_utils::VectorToStr(descriptor.vectorized_axis).c_str());
  return NddmaFallbackReason::kNone;
}

NddmaFallbackReason NormalizeNddmaDescriptor(const NddmaDescriptorInfo &descriptor, NddmaNormalizedDesc &normalized) {
  normalized = NddmaNormalizedDesc{};
  normalized.effective_rank = descriptor.output_dims.size();
  if (normalized.effective_rank < kMinNddmaRank || normalized.effective_rank > kMaxNddmaRank) {
    return NddmaFallbackReason::kRankUnsupported;
  }
  if (!HasValidVectorLengths(descriptor)) {
    return NddmaFallbackReason::kSchemaMismatch;
  }
  if (!HasValidAxisOrder(descriptor)) {
    return NddmaFallbackReason::kCodegenMismatch;
  }
  normalized.output_dims.assign(descriptor.output_dims.rbegin(), descriptor.output_dims.rend());
  normalized.input_strides.assign(descriptor.input_strides.rbegin(), descriptor.input_strides.rend());
  normalized.output_strides.assign(descriptor.output_strides.rbegin(), descriptor.output_strides.rend());
  normalized.vectorized_axis.assign(descriptor.vectorized_axis.rbegin(), descriptor.vectorized_axis.rend());
  return HasInvalidStaticValue(normalized) ? NddmaFallbackReason::kStrideInvalid : NddmaFallbackReason::kNone;
}

af::Status EvaluateNddmaModel(const NddmaDescriptorInfo &descriptor, const std::string &dtype, const Expr &block_dim,
                              NddmaModelResult &result) {
  result = NddmaModelResult{};
  NddmaNormalizedDesc normalized;
  result.fallback_reason = NormalizeNddmaDescriptor(descriptor, normalized);
  result.effective_rank = normalized.effective_rank;
  if (result.fallback_reason != NddmaFallbackReason::kNone) {
    return af::SUCCESS;
  }
  if (normalized.effective_rank < kMinNddmaRank || normalized.effective_rank > kMaxNddmaRank) {
    result.fallback_reason = NddmaFallbackReason::kNoRegisteredModel;
    return af::SUCCESS;
  }
  uint64_t dtype_size = 0U;
  if (!GetDtypeSize(dtype, dtype_size)) {
    result.fallback_reason = NddmaFallbackReason::kDtypeUnsupported;
    return af::SUCCESS;
  }
  NddmaNormalizedDesc physical_1d;
  const bool direct_1d = BuildPhysical1DView(normalized, physical_1d);
  const auto *params = direct_1d ? GetCalibrated1DParams(dtype_size) : GetNddma1DParams(dtype_size);
  if (params == nullptr || IsStaticNonPositive(block_dim)) {
    result.fallback_reason =
        params == nullptr ? NddmaFallbackReason::kDtypeUnsupported : NddmaFallbackReason::kSchemaMismatch;
    return af::SUCCESS;
  }
  Expr low_core;
  Expr high_core;
  NddmaNg2Config ng2_config{};
  const auto *ng2_params = IsNg2Eligible(normalized, ng2_config) ? GetNddmaNg2Params(dtype_size) : nullptr;
  if (ng2_params != nullptr) {
    const auto *calibrated = GetCalibratedNg2Params(dtype_size);
    const bool use_calibrated = calibrated != nullptr && CanUseCalibratedNg2(normalized, ng2_config, block_dim);
    result.fallback_reason =
        BuildNg2Cycles(normalized, *params, *ng2_params, dtype_size, low_core, high_core, ng2_config, !use_calibrated);
    if (use_calibrated && result.fallback_reason == NddmaFallbackReason::kNone) {
      GE_ASSERT_SUCCESS(
          BuildCalibratedNg2High(normalized, ng2_config, *calibrated, dtype_size, block_dim, high_core, result));
    }
    result.model_name = use_calibrated ? "NDDMA_ND_MULTICORE_NG2_V2" : "NDDMA_ND_MULTICORE_NG2";
  } else {
    result.fallback_reason =
        BuildNddmaCoreCycles(direct_1d ? physical_1d : normalized, *params, dtype_size, low_core, high_core);
  }
  if (result.fallback_reason != NddmaFallbackReason::kNone) {
    return af::SUCCESS;
  }
  // A padded symbolic row pitch can equal the actual inner dimension at runtime
  // (e.g. 8*ceil(n/8) when n=8). Keep the original padded-layout estimate and
  // select the calibrated packed API cost only when that equality holds.
  NddmaNg2Config packed_config{};
  const auto *packed_params = GetCalibratedNg2Params(dtype_size);
  bool guarded_packed_ng2 = false;
  if (packed_params != nullptr && IsNg2Eligible(normalized, packed_config, false) &&
      CanUseCalibratedNg2(normalized, packed_config, block_dim, false)) {
    const Expr &pitch = normalized.output_strides[packed_config.m_pos];
    const Expr &inner = normalized.output_dims[packed_config.n_pos];
    if ((!pitch.IsConstExpr() || !inner.IsConstExpr()) &&
        af::SymbolicUtils::StaticCheckEq(pitch, inner) != af::TriBool::kTrue) {
      Expr packed_high;
      GE_ASSERT_SUCCESS(BuildCalibratedNg2High(normalized, packed_config, *packed_params, dtype_size, block_dim,
                                               packed_high, result));
      Expr guarded_high;
      GetPerfVar("nddma_ng2_packed", guarded_high, result.ternary_ops);
      TernaryOp packed_guard(CondType::K_EQ, pitch, inner, std::make_shared<IfCase>(packed_high),
                             std::make_shared<IfCase>(high_core));
      packed_guard.SetVariable(guarded_high);
      result.ternary_ops[guarded_high] = std::move(packed_guard);
      high_core = guarded_high;
      guarded_packed_ng2 = true;
      result.model_name = "NDDMA_ND_MULTICORE_NG2_V2";
    }
  }
  GE_ASSERT_SUCCESS(SelectCoreBranch(block_dim, low_core, high_core, result));
  // Fitted residual terms can become non-positive outside their measured
  // domain (for example, tiny B with a large effective stride).  Do not emit
  // an invalid performance expression; use the existing legacy path instead.
  if (IsStaticNonPositive(result.cycles)) {
    result.selected = false;
    result.fallback_reason = NddmaFallbackReason::kSchemaMismatch;
    return af::SUCCESS;
  }
  result.selected = true;
  if (ng2_params == nullptr && !guarded_packed_ng2) {
    result.model_name =
        (direct_1d || normalized.effective_rank == 1U) ? "NDDMA_1D_MULTICORE_V2" : "NDDMA_ND_MULTICORE_V1";
  }
  result.fallback_reason = NddmaFallbackReason::kNone;
  return af::SUCCESS;
}

void LogNddmaFallback(const std::string &node_name, const std::string &dtype, const NddmaDescriptorInfo *descriptor,
                      const NddmaModelResult &result) {
  GELOGW(
      "[ATT NDDMA] fallback: node=%s, raw_rank=%zu, effective_rank=%zu, dtype=%s, candidate_model=%s, "
      "fallback_reason=%s",
      node_name.c_str(), result.raw_rank, result.effective_rank, dtype.c_str(), result.model_name.c_str(),
      NddmaFallbackReasonToString(result.fallback_reason));
  if (descriptor != nullptr) {
    GELOGD("[ATT NDDMA] fallback detail: output_dims=[%s], input_strides=[%s], output_strides=[%s]",
           GetVecString(descriptor->output_dims).c_str(), GetVecString(descriptor->input_strides).c_str(),
           GetVecString(descriptor->output_strides).c_str());
  }
}
}  // namespace att
