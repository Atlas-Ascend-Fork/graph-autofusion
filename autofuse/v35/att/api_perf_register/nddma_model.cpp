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
#include "perf_param_v2.h"

namespace att {
namespace {
constexpr size_t kMinNddmaRank = 1U;
constexpr size_t kMaxNddmaRank = 5U;
constexpr uint64_t kLowCoreUpperBound = 2U;
constexpr uint64_t kInputStrideUpperBound = 128U;
constexpr uint64_t kCalibratedNg2InnerBytesThreshold = 128U;
constexpr double kSparseRank3ResidualScale = 0.017212331646550515;
constexpr double kSparseRank4PackedResidualScale = 0.016809401000806315;
constexpr double kSparseRank4PaddedResidualScale = 0.030215324807753154;

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

// Report the expression family before tiling has supplied values for symbolic guards.
std::string RuntimeModelDispatch(const std::string &matched, const std::string &otherwise) {
  return "NDDMA_RUNTIME_DISPATCH[" + matched + "|" + otherwise + "]";
}

af::Status SelectCoreBranch(const Expr &block_dim, const Expr &low_core, const Expr &high_core,
                            const std::string &low_model, const std::string &high_model, NddmaModelResult &result) {
  if (af::SymbolicUtils::StaticCheckLe(block_dim, CreateExpr(kLowCoreUpperBound)) == af::TriBool::kTrue) {
    result.cycles = low_core;
    result.model_name = low_model;
    return af::SUCCESS;
  }
  if (af::SymbolicUtils::StaticCheckGt(block_dim, CreateExpr(kLowCoreUpperBound)) == af::TriBool::kTrue) {
    result.cycles = high_core;
    result.model_name = high_model;
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
  result.model_name = low_model == high_model ? low_model : RuntimeModelDispatch(low_model, high_model);
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

Expr Packed3DHighCoreCycles(const Expr &payload_elements, uint64_t dtype_size) {
  constexpr double kStartupCycles = 177.39897232376856;
  constexpr double kCyclesPerKiB = 76.27710129248759;
  constexpr uint64_t kBytesPerKiB = 1024U;
  return CreateExpr(kStartupCycles) + CreateExpr(kCyclesPerKiB * dtype_size / kBytesPerKiB) * payload_elements;
}

bool ApplyPacked3DHighCoreCost(const NddmaNormalizedDesc &desc, uint64_t dtype_size, const Expr &block_dim,
                               Expr &high_core, NddmaModelResult &result) {
  constexpr uint64_t kMaxFittedCoreCount = 56U;
  const auto upper = CreateExpr(kMaxFittedCoreCount);
  if (desc.effective_rank != 3U || dtype_size != sizeof(uint32_t) ||
      af::SymbolicUtils::StaticCheckLe(block_dim, CreateExpr(kLowCoreUpperBound)) == af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckGt(block_dim, upper) == af::TriBool::kTrue) {
    return false;
  }
  const auto &inner = desc.output_dims[0];
  const auto &middle = desc.output_dims[1];
  const auto &outer = desc.output_dims[2];
  const auto plane = inner * middle;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  if (af::SymbolicUtils::StaticCheckGt(inner, one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckGt(middle, one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckLe(outer, one) == af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[0], middle) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[1], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[2], plane) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[1], inner) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[2], plane) != af::TriBool::kTrue) {
    return false;
  }

  // Whole-API fit for a packed batch of transposed planes, measured with
  // fresh GM addresses and normal L2 policy. Payload is in KiB.
  const auto fitted = Packed3DHighCoreCycles(outer * plane, dtype_size);
  if (af::SymbolicUtils::StaticCheckGt(outer, one) == af::TriBool::kTrue &&
      af::SymbolicUtils::StaticCheckLe(block_dim, upper) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  // Tiling can reduce the outer axis to one or exceed the fitted core
  // count. In both cases keep the original descriptor-based estimate.
  const auto active_outer = af::sym::Min(one, af::sym::Max(af::sym::kSymbolZero, outer - one));
  const auto supported_cores = af::sym::Min(one, af::sym::Max(af::sym::kSymbolZero, upper + one - block_dim));
  Expr guarded;
  GetPerfVar("nddma_packed3d", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, active_outer * supported_cores, one, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

// Calibrate the legacy ND residual independently of its bandwidth/startup terms.
// The bounded sparse plane was validated with both normal and disabled L2 hints.
bool BuildSparseRank4Inputs(const NddmaNormalizedDesc &desc, uint64_t dtype_size, uint64_t pitch, Expr &outside,
                            Expr &payload, Expr &residual_scale) {
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &dims = desc.output_dims;
  const auto &width = dims.back();
  const auto &one = af::sym::kSymbolOne;
  const auto &zero = af::sym::kSymbolZero;
  const auto lanes = CreateExpr(2U);
  const auto equals = [](const Expr &a, const Expr &b) {
    return af::SymbolicUtils::StaticCheckEq(a, b) == af::TriBool::kTrue;
  };
  uint64_t source_width = 0U;
  if (pitch != 65536U || !gm[2].GetConstValue(source_width) || (source_width != 512U && source_width != 1024U) ||
      !equals(width, gm[2])) {
    return false;
  }
  const auto &tile1 = dims[2];
  const auto &tile2 = dims[1];
  const auto aligned_pitch = CreateExpr(8U) * af::sym::Ceiling(tile2 / CreateExpr(4U));
  // Source runs beyond 8 KiB exhibit a separate measured cost increase.
  // Keep those runs and unmeasured outer extents on the legacy formula.
  outside = outside + af::sym::Max(zero, one - tile1) + af::sym::Max(zero, tile1 - CreateExpr(3U)) +
            af::sym::Max(zero, lanes - tile2) +
            af::sym::Min(af::sym::Max(zero, tile2 - CreateExpr(4U)), AbsExpr(tile2 - CreateExpr(8U))) +
            af::sym::Max(zero, tile1 * tile2 - CreateExpr(12U)) +
            af::sym::Max(zero, width * tile1 - CreateExpr(2048U)) + AbsExpr(ub[2] - aligned_pitch) +
            AbsExpr(ub[3] - aligned_pitch * tile1);
  payload = width * tile1 * tile2 * lanes * CreateExpr(dtype_size);
  const auto padded = af::sym::Min(one, AbsExpr(ub[2] - lanes * tile2));
  residual_scale = CreateExpr(kSparseRank4PackedResidualScale) +
                   padded * CreateExpr(kSparseRank4PaddedResidualScale - kSparseRank4PackedResidualScale);
  return true;
}

void ApplySparseFitCost(const Expr &payload, const Expr &residual_scale, const Nddma1DParams &params,
                        const Expr &outside, Expr &high_core, NddmaModelResult &result) {
  const auto &zero = af::sym::kSymbolZero;
  const auto base = payload / CreateExpr(params.t2) + CreateExpr(params.h2);
  const auto fitted = base + residual_scale * (high_core - base);
  if (af::SymbolicUtils::StaticCheckEq(outside, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return;
  }
  Expr guarded;
  GetPerfVar("nddma_sparse_nd", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, outside, zero, std::make_shared<IfCase>(fitted), std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
}

bool ApplySparseNdResidualScale(const NddmaNormalizedDesc &desc, uint64_t dtype_size, const Nddma1DParams &params,
                                const Expr &block_dim, Expr &high_core, NddmaModelResult &result) {
  const size_t rank = desc.effective_rank;
  if ((rank != 3U && rank != 4U) || dtype_size != sizeof(uint32_t)) {
    return false;
  }
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &dims = desc.output_dims;
  const auto &width = dims.back();
  const auto &one = af::sym::kSymbolOne;
  const auto &zero = af::sym::kSymbolZero;
  const auto rows = CreateExpr(32U);
  const auto lanes = CreateExpr(2U);
  const auto equals = [](const Expr &a, const Expr &b) {
    return af::SymbolicUtils::StaticCheckEq(a, b) == af::TriBool::kTrue;
  };
  uint64_t pitch = 0U;
  if (!gm[1].GetConstValue(pitch) || !equals(gm.back(), one) || !equals(gm[0], gm[1] * rows) ||
      !equals(dims[0], lanes) || !equals(ub[0], one) || !equals(ub[1], lanes)) {
    return false;
  }
  // The four-axis calibration is selected by its measured descriptor layout;
  // the high-core branch already carries the symbolic block_dim dependence.
  // Keep the original bounded domain for the three-axis fit until it is refit.
  Expr outside = rank == 3U
                     ? af::sym::Max(zero, CreateExpr(40U) - block_dim) + af::sym::Max(zero, block_dim - CreateExpr(56U))
                     : CreateExpr(0U);
  Expr payload;
  Expr residual_scale;
  if (rank == 3U) {
    if ((pitch != 65536U && pitch != 131072U) || !equals(dims[1], rows) || !equals(ub[2], rows * lanes)) {
      return false;
    }
    outside = outside + af::sym::Max(zero, CreateExpr(192U) - width) + af::sym::Max(zero, width - CreateExpr(256U));
    payload = width * rows * lanes * CreateExpr(dtype_size);
    residual_scale = CreateExpr(kSparseRank3ResidualScale);
  } else {
    if (!BuildSparseRank4Inputs(desc, dtype_size, pitch, outside, payload, residual_scale)) {
      return false;
    }
  }
  if (af::SymbolicUtils::StaticCheckGt(outside, zero) == af::TriBool::kTrue) {
    return false;
  }
  ApplySparseFitCost(payload, residual_scale, params, outside, high_core, result);
  return true;
}

Expr OutsideRange(const Expr &value, int32_t lower, int32_t upper) {
  return af::sym::Max(af::sym::kSymbolZero, af::sym::Max(CreateExpr(lower) - value, value - CreateExpr(upper)));
}

bool ApplyFp32UnitBroadcastRowCost(const NddmaNormalizedDesc &desc, const Expr &block_dim, Expr &high_core,
                                   NddmaModelResult &result) {
  const auto &one = af::sym::kSymbolOne;
  const auto &zero = af::sym::kSymbolZero;
  if (desc.effective_rank != 2U || af::SymbolicUtils::StaticCheckEq(desc.input_strides[0], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(desc.input_strides[1], zero) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(desc.output_strides[0], one) != af::TriBool::kTrue) {
    return false;
  }
  constexpr int32_t kMinWidth = 1024;
  constexpr int32_t kMaxWidth = 8192;
  constexpr int32_t kMinCores = 4;
  constexpr int32_t kMaxCores = 24;
  constexpr int32_t kRowAlignment = 32 / sizeof(float);
  constexpr double kStartup = 282.35013548341914;
  constexpr double kBytesPerCycle = 29.240946869971374;
  const auto &width = desc.output_dims[0];
  const Expr violation = AbsExpr(desc.output_dims[1] - one) + AbsExpr(desc.output_strides[1] - width) +
                         AbsExpr(af::sym::Mod(width, CreateExpr(kRowAlignment))) +
                         OutsideRange(width, kMinWidth, kMaxWidth) + OutsideRange(block_dim, kMinCores, kMaxCores);
  const auto supported = af::SymbolicUtils::StaticCheckEq(violation, zero);
  if (supported == af::TriBool::kFalse) {
    return false;
  }
  const Expr fitted = CreateExpr(kStartup) + width * CreateExpr(sizeof(float)) / CreateExpr(kBytesPerCycle);
  if (supported == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_unit_broadcast_row", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, fitted, high_core);
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

bool ScaleNdResidual(const Expr &violation, const Expr &payload, const Nddma1DParams &params, double scale,
                     const char *perf_var, Expr &high_core, NddmaModelResult &result) {
  const auto &zero = af::sym::kSymbolZero;
  const auto supported = af::SymbolicUtils::StaticCheckEq(violation, zero);
  if (supported == af::TriBool::kFalse) {
    return false;
  }
  const Expr baseline = payload / CreateExpr(params.t2) + CreateExpr(params.h2);
  const Expr fitted = baseline + CreateExpr(scale) * (high_core - baseline);
  if (supported == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar(perf_var, guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

// Consecutive FP32 GM rows with a wide, padded UB row. Preserve the ND payload
// and startup terms; long-call production-grid samples calibrate only its residual.
bool ApplyFp32WidePaddedRowScale(const NddmaNormalizedDesc &desc, const Nddma1DParams &params, const Expr &block_dim,
                                 Expr &high_core, NddmaModelResult &result) {
  if (desc.effective_rank != 2U && desc.effective_rank != 3U) {
    return false;
  }
  const auto &dims = desc.output_dims;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  if (af::SymbolicUtils::StaticCheckEq(gm[0], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue) {
    return false;
  }
  const auto &width = dims[0];
  const auto &rows = dims.back();
  const auto alignment = CreateExpr(8);
  const auto &zero = af::sym::kSymbolZero;
  Expr violation = zero;
  for (const auto &check :
       {OutsideRange(width, 967, 1039), OutsideRange(rows, 18, 30), OutsideRange(block_dim, 48, 56),
        AbsExpr(gm.back() - width), AbsExpr(ub.back() - alignment * af::sym::Ceiling(af::sym::Div(width, alignment))),
        OutsideRange(ub.back() - width, 1, 7)}) {
    if (af::SymbolicUtils::StaticCheckGt(check, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + check;
  }
  if (desc.effective_rank == 3U) {
    violation = violation + AbsExpr(dims[1] - one);
  }
  constexpr double kResidualScale = 1.9724183277398053;
  return ScaleNdResidual(violation, rows * width * CreateExpr(sizeof(uint32_t)), params, kResidualScale,
                         "nddma_wide_padded_fp32", high_core, result);
}

// Short FP16 rows sampled at a non-unit GM inner stride and padded in UB.
bool ApplyFp16ShortInnerStrideScale(const NddmaNormalizedDesc &desc, const Nddma1DParams &params, Expr &high_core,
                                    NddmaModelResult &result) {
  if (desc.effective_rank != 2U) {
    return false;
  }
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  uint64_t outer_stride = 0;
  uint64_t inner_stride = 0;
  if (!gm[1].GetConstValue(outer_stride) || outer_stride < 2048U || outer_stride > 3456U ||
      !gm[0].GetConstValue(inner_stride) || inner_stride < 33U || inner_stride > 49U ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue) {
    return false;
  }
  const auto &rows = desc.output_dims[1];
  const auto &width = desc.output_dims[0];
  const auto &zero = af::sym::kSymbolZero;
  const auto alignment = CreateExpr(16);
  const auto violation = OutsideRange(rows, 2, 4) + OutsideRange(width, 28, 46) + OutsideRange(ub[1] - width, 1, 15) +
                         AbsExpr(ub[1] - alignment * af::sym::Ceiling(af::sym::Div(width, alignment))) +
                         af::sym::Max(zero, width * gm[0] - gm[1]);
  constexpr double kResidualScale = 0.37866923612622494;
  return ScaleNdResidual(violation, rows * width * CreateExpr(sizeof(uint16_t)), params, kResidualScale,
                         "nddma_short_inner_stride_fp16", high_core, result);
}

// Refit only the existing ND residual for FP16 with consecutive GM
// planes written as packed or padded UB planes. A fourth axis may be physically unit.
bool ApplySparseFp16NdResidualScale(const NddmaNormalizedDesc &desc, const Nddma1DParams &params, const Expr &block_dim,
                                    Expr &high_core, NddmaModelResult &result) {
  constexpr uint64_t kMinGmRowPitch = 2960U;
  constexpr uint64_t kMaxGmRowPitch = 3544U;
  const auto rank = desc.effective_rank;
  if (rank != 3U && rank != 4U) {
    return false;
  }
  const auto &dims = desc.output_dims;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  const auto rows = CreateExpr(4);
  const auto alignment = CreateExpr(16);
  uint64_t gm_pitch = 0;
  if (!gm[1].GetConstValue(gm_pitch) || gm_pitch < kMinGmRowPitch || gm_pitch > kMaxGmRowPitch ||
      af::SymbolicUtils::StaticCheckEq(gm.back(), one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[0], rows * gm[1]) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue) {
    return false;
  }
  const auto &width = dims[0];
  const auto &planes = dims.back();
  const auto packed_pitch = rows * width;
  const auto padded_pitch = alignment * af::sym::Ceiling(af::sym::Div(packed_pitch, alignment));
  const auto padded_violation = OutsideRange(planes, 8, 36) + AbsExpr(ub.back() - padded_pitch);
  const auto packed_violation = OutsideRange(planes, 20, 40) + AbsExpr(ub.back() - packed_pitch);
  Expr violation = OutsideRange(width, 104, 128) + AbsExpr(dims[1] - rows) + OutsideRange(block_dim, 48, 56) +
                   AbsExpr(ub[1] - width) + af::sym::Min(padded_violation, packed_violation);
  if (rank == 4U) {
    violation = violation + AbsExpr(dims[2] - one);
  }
  const Expr payload = width * rows * planes * CreateExpr(sizeof(uint16_t));
  constexpr double kResidualScale = 0.040936234851127434;
  return ScaleNdResidual(violation, payload, params, kResidualScale, "nddma_sparse_fp16", high_core, result);
}

// A narrow strided source row and a separate, much wider GM plane stride.
bool ApplyFp16TwoLevelSparseScale(const NddmaNormalizedDesc &desc, const Nddma1DParams &params, const Expr &block_dim,
                                  Expr &high_core, NddmaModelResult &result) {
  constexpr uint64_t kMinGmRowPitch = 3168U;
  constexpr uint64_t kMaxGmRowPitch = 3520U;
  constexpr uint64_t kMinGroupsPerPlane = 100U;
  constexpr uint64_t kMaxGroupsPerPlane = 120U;
  constexpr int32_t kMaxOuterRepeat = 7;
  const auto rank = desc.effective_rank;
  if (rank != 3U && rank != 4U) {
    return false;
  }
  const auto &dims = desc.output_dims;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  uint64_t row_pitch = 0;
  uint64_t plane_pitch = 0;
  if (!gm[0].GetConstValue(row_pitch) || row_pitch < kMinGmRowPitch || row_pitch > kMaxGmRowPitch ||
      !gm[2].GetConstValue(plane_pitch) || af::SymbolicUtils::StaticCheckEq(gm[1], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue) {
    return false;
  }
  constexpr uint64_t kRows = 4;
  const auto group_pitch = kRows * row_pitch;
  if (plane_pitch % group_pitch != 0 || plane_pitch / group_pitch < kMinGroupsPerPlane ||
      plane_pitch / group_pitch > kMaxGroupsPerPlane) {
    return false;
  }
  const auto rows = CreateExpr(kRows);
  const auto plane = rows * dims[1];
  const auto batch = plane * dims[2];
  Expr violation = AbsExpr(dims[0] - rows) + OutsideRange(dims[1], 24, 42) + OutsideRange(dims[2], 20, 40) +
                   OutsideRange(block_dim, 48, 56) + AbsExpr(ub[1] - rows) + AbsExpr(ub[2] - plane);
  Expr payload = batch * CreateExpr(sizeof(uint16_t));
  if (rank == 4U) {
    violation =
        violation + OutsideRange(dims[3], 1, kMaxOuterRepeat) + AbsExpr(gm[3] - dims[1]) + AbsExpr(ub[3] - batch);
    payload = payload * dims[3];
  }
  constexpr double kResidualScale = 0.007910991623176598;
  return ScaleNdResidual(violation, payload, params, kResidualScale, "nddma_two_level_sparse_fp16", high_core, result);
}

// The inner four strided lanes cover the same GM period as each outer row.
bool ApplyFp16CommensurateStrideScale(const NddmaNormalizedDesc &desc, const Nddma1DParams &params,
                                      const Expr &block_dim, Expr &high_core, NddmaModelResult &result) {
  const auto rank = desc.effective_rank;
  if (rank != 3U && rank != 4U) {
    return false;
  }
  const auto &dims = desc.output_dims;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  const auto lanes = CreateExpr(4);
  uint64_t stride = 0;
  if (!gm[0].GetConstValue(stride) || stride < 3072U || stride > 3584U ||
      af::SymbolicUtils::StaticCheckEq(gm[1], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[2], lanes * gm[0]) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[1], lanes) != af::TriBool::kTrue) {
    return false;
  }
  const auto plane = lanes * dims[1];
  Expr violation = AbsExpr(dims[0] - lanes) + OutsideRange(dims[1], 32, 40) + OutsideRange(dims[2], 88, 128) +
                   AbsExpr(ub[2] - plane) + OutsideRange(block_dim, 48, 56);
  if (rank == 4U) {
    violation = violation + AbsExpr(dims[3] - one);
  }
  constexpr double kResidualScale = 0.011888285459372117;
  return ScaleNdResidual(violation, plane * dims[2] * CreateExpr(sizeof(uint16_t)), params, kResidualScale,
                         "nddma_commensurate_stride_fp16", high_core, result);
}

bool ApplyFp32SparseWidePlaneScale(const NddmaNormalizedDesc &desc, const Nddma1DParams &params, const Expr &block_dim,
                                   Expr &high_core, NddmaModelResult &result) {
  constexpr int32_t kGmPlanePitch = 36864;
  if (desc.effective_rank != 3U) {
    return false;
  }
  const auto &dims = desc.output_dims;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  const auto lanes = CreateExpr(16);
  const auto width = CreateExpr(48);
  if (af::SymbolicUtils::StaticCheckEq(gm[0], lanes) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[1], CreateExpr(kGmPlanePitch)) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[2], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[1], width) != af::TriBool::kTrue) {
    return false;
  }
  const auto plane = dims[1] * width;
  const auto violation = AbsExpr(dims[0] - width) + OutsideRange(dims[1], 16, 32) + AbsExpr(dims[2] - lanes) +
                         OutsideRange(block_dim, 48, 56) + AbsExpr(ub[2] - plane);
  constexpr double kResidualScale = 0.0760036317579465;
  const auto payload = plane * lanes * CreateExpr(sizeof(float));
  return ScaleNdResidual(violation, payload, params, kResidualScale, "nddma_sparse_wide_plane_fp32", high_core, result);
}

bool ApplyFp32PackedShortRowResidualScale(const NddmaNormalizedDesc &desc, const Nddma1DParams &params,
                                          const Expr &block_dim, Expr &high_core, NddmaModelResult &result) {
  constexpr uint64_t kMinGmRowPitch = 120U;
  constexpr uint64_t kMaxGmRowPitch = 168U;
  if (desc.effective_rank != 3U) {
    return false;
  }
  const auto &dims = desc.output_dims;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &one = af::sym::kSymbolOne;
  const auto lanes = CreateExpr(3);
  const auto alignment = CreateExpr(8);
  uint64_t gm_pitch = 0;
  if (!gm[1].GetConstValue(gm_pitch) || gm_pitch < kMinGmRowPitch || gm_pitch > kMaxGmRowPitch ||
      af::SymbolicUtils::StaticCheckEq(gm[0], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(gm[2], lanes) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[0], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(ub[1], lanes) != af::TriBool::kTrue) {
    return false;
  }
  const auto ub_pitch = alignment * af::sym::Ceiling(af::sym::Div(lanes * dims[1], alignment));
  const auto overlap = af::sym::Max(af::sym::kSymbolZero, dims[2] * lanes - gm[1]);
  const auto violation = AbsExpr(dims[0] - lanes) + OutsideRange(dims[1], 32, 56) + OutsideRange(dims[2], 32, 56) +
                         OutsideRange(block_dim, 44, 56) + AbsExpr(ub[2] - ub_pitch) + overlap;
  constexpr double kResidualScale = 0.10462327934265224;
  const auto payload = lanes * dims[1] * dims[2] * CreateExpr(sizeof(float));
  return ScaleNdResidual(violation, payload, params, kResidualScale, "nddma_packed_short_fp32", high_core, result);
}

// A symbolic unit axis can hide a packed physical rank-3 transfer. The
// source may be contiguous along either outer axis; UB stays densely packed.
bool ApplyFp32UnitAxisPacked3DCost(const NddmaNormalizedDesc &desc, const Expr &block_dim, Expr &high_core,
                                   NddmaModelResult &result) {
  if (desc.effective_rank != 4U) {
    return false;
  }
  const auto &dims = desc.output_dims;
  const auto &gm = desc.input_strides;
  const auto &ub = desc.output_strides;
  const auto &zero = af::sym::kSymbolZero;
  const auto &one = af::sym::kSymbolOne;
  const auto lanes = CreateExpr(16);
  const auto width = CreateExpr(48);
  const auto equal = [](const Expr &lhs, const Expr &rhs) {
    return af::SymbolicUtils::StaticCheckEq(lhs, rhs) == af::TriBool::kTrue;
  };
  if (!equal(gm[0], lanes) || !equal(ub[0], one)) {
    return false;
  }
  const auto plane = width * lanes;
  const bool contiguous_middle = equal(gm[2], one) && equal(gm[3], plane) && equal(dims[2], lanes);
  const bool contiguous_outer = equal(gm[3], one) && equal(gm[2], plane) && equal(dims[3], lanes);
  if (!contiguous_middle && !contiguous_outer) {
    return false;
  }
  const auto &rows = contiguous_middle ? dims[3] : dims[2];
  const auto violation = AbsExpr(dims[0] - width) + AbsExpr(dims[1] - one) + OutsideRange(rows, 16, 28) +
                         OutsideRange(block_dim, 48, 56) + AbsExpr(ub[2] - width) + AbsExpr(ub[3] - width * dims[2]);
  const auto supported = af::SymbolicUtils::StaticCheckEq(violation, zero);
  if (supported == af::TriBool::kFalse) {
    return false;
  }
  auto fitted = Packed3DHighCoreCycles(plane * rows, sizeof(float));
  if (supported == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_packed3d_unit_axis", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
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

Expr GetNg2Multiplier(const NddmaNormalizedDesc &normalized, const NddmaNg2Params &params,
                      const NddmaNg2Config &config) {
  const auto &m = normalized.output_dims[config.m_pos];
  return (CreateExpr(params.g10) + CreateExpr(params.g11_m) * m) * normalized.input_strides[config.m_pos] +
         CreateExpr(params.g00) + CreateExpr(params.g01_m) * m;
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
  const Expr ng2 = GetNg2Multiplier(normalized, ng2_params, config);
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

// Preserve the small-inner NG2 formula. Only its bandwidth terms are
// recalibrated for wide GM strides with a packed FP16 UB descriptor.
bool ApplyWideFp16Ng2High(const NddmaNormalizedDesc &normalized, const NddmaNg2Config &config,
                          const CalibratedNg2Params &params, const Expr &block_dim, Expr &high_core,
                          NddmaModelResult &result) {
  const auto &m = normalized.output_dims[config.m_pos];
  const auto &n = normalized.output_dims[config.n_pos];
  const auto &gm_pitch = normalized.input_strides[config.n_pos];
  const auto &ub_pitch = normalized.output_strides[config.m_pos];
  const auto &zero = af::sym::kSymbolZero;
  constexpr int64_t kMinPitch = 1282048;
  constexpr int64_t kMaxPitch = 1648640;
  if (af::SymbolicUtils::StaticCheckGe(gm_pitch, CreateExpr(kMinPitch)) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckLe(gm_pitch, CreateExpr(kMaxPitch)) != af::TriBool::kTrue) {
    return false;
  }
  const auto outside = [&](const Expr &value, int32_t lower, int32_t upper) {
    return af::sym::Max(zero, af::sym::Max(CreateExpr(lower) - value, value - CreateExpr(upper)));
  };
  Expr violation = zero;
  for (const auto &check : {outside(m, 1, 752), outside(n, 16, 32), outside(block_dim, 32, 56),
                            AbsExpr(af::sym::Mod(n, CreateExpr(16))), AbsExpr(ub_pitch - n)}) {
    if (af::SymbolicUtils::StaticCheckGt(check, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + check;
  }
  constexpr double kPayloadCost = 1751.89015325;
  constexpr double kSharedCost = 753.57769812;
  const auto payload = m * n * CreateExpr(sizeof(uint16_t)) / CreateExpr(16384);
  const auto shared = af::sym::Max(block_dim - CreateExpr(32), zero) / CreateExpr(24);
  const auto &small = params.small_inner;
  const auto fitted =
      high_core + payload * (CreateExpr(kPayloadCost - small[1]) + shared * CreateExpr(kSharedCost - small[3]));
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_ng2_wide_fp16", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

bool ApplyFp16MediumPitchNg2High(const NddmaNormalizedDesc &desc, const NddmaNg2Config &config, Expr &high_core,
                                 NddmaModelResult &result) {
  const auto &m = desc.output_dims[config.m_pos];
  const auto &n = desc.output_dims[config.n_pos];
  const auto &ub_pitch = desc.output_strides[config.m_pos];
  const auto &zero = af::sym::kSymbolZero;
  uint64_t gm_pitch = 0;
  if (!desc.input_strides[config.n_pos].GetConstValue(gm_pitch) || gm_pitch < 12032U || gm_pitch > 15104U) {
    return false;
  }
  const auto violation = OutsideRange(m, 144, 208) + OutsideRange(n, 106, 128) + AbsExpr(ub_pitch - n);
  const auto supported = af::SymbolicUtils::StaticCheckEq(violation, zero);
  if (supported == af::TriBool::kFalse) {
    return false;
  }
  // Keep the large-inner packed NG2 startup and contention terms. Replace
  // only the measured payload slope for this range of GM pitches.
  constexpr double kPayloadDelta = 3918.5085980813437;
  const auto fitted = high_core + CreateExpr(kPayloadDelta * sizeof(uint16_t) / 65536.0) * m * n;
  if (supported == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_ng2_medium_pitch_fp16", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

bool ApplySmallInt64Ng2High(const NddmaNormalizedDesc &normalized, const NddmaNg2Config &config, const Expr &block_dim,
                            Expr &high_core, NddmaModelResult &result) {
  const auto &m = normalized.output_dims[config.m_pos];
  const auto &n = normalized.output_dims[config.n_pos];
  const auto &pitch = normalized.output_strides[config.m_pos];
  const auto &zero = af::sym::kSymbolZero;
  const auto outside = [&](const Expr &value, int32_t lower, int32_t upper) {
    return af::sym::Max(zero, af::sym::Max(CreateExpr(lower) - value, value - CreateExpr(upper)));
  };
  Expr violation = zero;
  for (const auto &part : {outside(m, 24, 64), outside(n, 2, 8), outside(block_dim, 16, 56),
                           AbsExpr(normalized.input_strides[config.m_pos] - af::sym::kSymbolOne),
                           AbsExpr(normalized.input_strides[config.n_pos] - m), outside(pitch - n, 0, 1)}) {
    if (af::SymbolicUtils::StaticCheckGt(part, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + part;
  }
  // Small INT64 transposes are startup dominated. Independent fresh-address holdout
  // validates this constant only within the measured shape, pitch and core bounds.
  const auto fitted = CreateExpr(506.0);
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_ng2_small_int64", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

bool ApplyPaddedNg2High(const NddmaNormalizedDesc &normalized, const NddmaNg2Config &config, const Expr &block_dim,
                        Expr &high_core, NddmaModelResult &result, bool sparse_source = false) {
  const auto &m = normalized.output_dims[config.m_pos];
  const auto &n = normalized.output_dims[config.n_pos];
  const auto &step = normalized.input_strides[config.n_pos];
  const auto &pitch = normalized.output_strides[config.m_pos];
  const auto &zero = af::sym::kSymbolZero;
  const auto &one = af::sym::kSymbolOne;
  const auto alignment = CreateExpr(8);  // FP32 elements in one 32-byte UB block.
  // Both formulas use the same padded UB layout. The sparse source has a
  // much larger GM pitch than its contiguous read width.
  const auto pitch_domain = sparse_source ? OutsideRange(step, 768, 1280) : OutsideRange(step, 12, 32);
  Expr violation = zero;
  for (const auto &part :
       {pitch_domain, OutsideRange(m, sparse_source ? 80 : 12, sparse_source ? 132 : 32),
        OutsideRange(n, sparse_source ? 88 : 193, sparse_source ? 136 : 519),
        OutsideRange(block_dim, sparse_source ? 32 : 16, 56), sparse_source ? CreateExpr(0) : AbsExpr(step - m),
        AbsExpr(pitch - alignment * af::sym::Ceiling(af::sym::Div(n, alignment))),
        af::sym::Max(zero, n + one - pitch)}) {
    if (af::SymbolicUtils::StaticCheckGt(part, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + part;
  }

  const double startup = sparse_source ? 186.219 : 162.5294017115518;
  const double bandwidth_cost = sparse_source ? 4993.870 : 6594.471271606094;
  const double contention_cost = sparse_source ? 653.375 : 565.9335005446032;
  const auto payload = m * n / CreateExpr(16384);  // FP32 payload in units of 64 KiB.
  const auto shared = af::sym::Max(block_dim - CreateExpr(32), zero) / CreateExpr(24);
  const auto fitted =
      CreateExpr(startup) + payload * (CreateExpr(bandwidth_cost) + CreateExpr(contention_cost) * shared);
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar(sparse_source ? "nddma_ng2_sparse_padded" : "nddma_ng2_padded", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

// A short transposed row can read few contiguous GM bursts despite the large
// pitch between those bursts. Keep the existing head plus payload cost form.
bool ApplyShortPaddedNg2High(const NddmaNormalizedDesc &desc, const NddmaNg2Config &config, const Expr &block_dim,
                             Expr &high_core, NddmaModelResult &result) {
  const auto &m = desc.output_dims[config.m_pos];
  const auto &n = desc.output_dims[config.n_pos];
  const auto &step = desc.input_strides[config.n_pos];
  const auto &pitch = desc.output_strides[config.m_pos];
  const auto &zero = af::sym::kSymbolZero;
  constexpr int32_t kFp32UbBlockElements = 8;
  constexpr int32_t kFp32GmBurstElements = 32;
  const auto alignment = CreateExpr(kFp32UbBlockElements);
  Expr violation = zero;
  for (const auto &part :
       {OutsideRange(m, 64, 160), OutsideRange(n, 9, 12), OutsideRange(step, 2048, 8192),
        OutsideRange(block_dim, 16, 56), AbsExpr(af::sym::Mod(step, CreateExpr(kFp32GmBurstElements))),
        AbsExpr(pitch - alignment * af::sym::Ceiling(af::sym::Div(n, alignment)))}) {
    if (af::SymbolicUtils::StaticCheckGt(part, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + part;
  }
  constexpr double kStartupCycles = 513.736889;
  constexpr double kCyclesPerElement = 0.1240756744;
  const auto fitted = CreateExpr(kStartupCycles) + m * n * CreateExpr(kCyclesPerElement);
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_ng2_short_padded_fp32", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, fitted, high_core);
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

struct DensePaddedNddmaCalibration {
  int32_t min_rows;
  int32_t max_rows;
  int32_t min_width;
  int32_t max_width;
  int32_t min_gm_pitch;
  int32_t max_gm_pitch;
  int32_t gm_alignment;
  int32_t max_gm_gap;
  int32_t min_active_cores;
  int32_t max_active_cores;
  bool requires_ub_padding;
  double startup;
  double cycles_per_element;
  const char *perf_var;
  bool broadcasts_inner{false};
};

constexpr DensePaddedNddmaCalibration kSmallDensePaddedTranspose = {
    16, 40, 17, 32, 24, 48, 8, 32, 4, 16, false, 499.88124, 0.0629557012, "nddma_small_dense_padded_transpose"};
constexpr DensePaddedNddmaCalibration kDensePaddedTranspose = {32,
                                                               128,
                                                               25,
                                                               53,
                                                               32,
                                                               128,
                                                               32,
                                                               32,
                                                               32,
                                                               56,
                                                               true,
                                                               347.52579736,
                                                               3731.65830216 / 16384.0,
                                                               "nddma_dense_padded_transpose"};
// Use the existing startup and payload form for rows confined to a 32-element padded span.
constexpr DensePaddedNddmaCalibration kNarrowDensePaddedTranspose = {64,
                                                                     128,
                                                                     25,
                                                                     31,
                                                                     64,
                                                                     128,
                                                                     32,
                                                                     32,
                                                                     28,
                                                                     54,
                                                                     true,
                                                                     323.07841553006364,
                                                                     0.1979937390396284,
                                                                     "nddma_narrow_dense_padded_transpose"};
constexpr DensePaddedNddmaCalibration kPaddedInnerBroadcast = {
    64, 192, 18, 31, 0, 0, 1, 0, 28, 56, false, 265.32598703, 0.1179375330, "nddma_padded_inner_broadcast", true};

bool ApplyDensePaddedNddma(const NddmaNormalizedDesc &desc, const Expr &block_dim, Expr &high_core,
                           NddmaModelResult &result, const DensePaddedNddmaCalibration &calibration) {
  constexpr size_t kInner = 0;
  constexpr size_t kRows = 1;
  constexpr size_t kOuter = 2;
  if (desc.effective_rank != 2U && desc.effective_rank != 3U) {
    return false;
  }
  const auto &one = af::sym::kSymbolOne;
  const auto &zero = af::sym::kSymbolZero;
  const auto &m = desc.output_dims[kRows];
  const auto &n = desc.output_dims[kInner];
  const auto &step = desc.input_strides[kInner];
  const auto &pitch = desc.output_strides[kRows];
  constexpr int32_t kFp32UbBlockElements = 8;
  const auto alignment = CreateExpr(kFp32UbBlockElements);
  const Expr outer = desc.effective_rank == 3U ? desc.output_dims[kOuter] : one;
  // A zero GM stride repeats each source value across the inner UB row.
  const Expr gm_gap =
      calibration.broadcasts_inner
          ? Expr(zero)
          : af::sym::Max(zero, m - step) + af::sym::Max(zero, step - m - CreateExpr(calibration.max_gm_gap));
  Expr violation = zero;
  for (const auto &part : {AbsExpr(outer - one), OutsideRange(m, calibration.min_rows, calibration.max_rows),
                           OutsideRange(n, calibration.min_width, calibration.max_width),
                           OutsideRange(step, calibration.min_gm_pitch, calibration.max_gm_pitch), gm_gap,
                           AbsExpr(af::sym::Mod(step, CreateExpr(calibration.gm_alignment))),
                           OutsideRange(block_dim, calibration.min_active_cores, calibration.max_active_cores),
                           AbsExpr(desc.input_strides[kRows] - one), AbsExpr(desc.output_strides[kInner] - one),
                           AbsExpr(pitch - alignment * af::sym::Ceiling(af::sym::Div(n, alignment))),
                           calibration.requires_ub_padding ? af::sym::Max(zero, n + one - pitch) : Expr(zero)}) {
    if (af::SymbolicUtils::StaticCheckGt(part, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + part;
  }
  const auto fitted = CreateExpr(calibration.startup) + m * n * CreateExpr(calibration.cycles_per_element);
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar(calibration.perf_var, guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, fitted, high_core);
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

// The outer axis is contiguous in GM. The two inner axes have a regular GM stride and padded UB rows.
bool ApplyFp32ShortStrided3DCost(const NddmaNormalizedDesc &desc, const Expr &block_dim, Expr &high_core,
                                 NddmaModelResult &result) {
  constexpr size_t kInner = 0U;
  constexpr size_t kMiddle = 1U;
  constexpr size_t kOuter = 2U;
  const auto &one = af::sym::kSymbolOne;
  if (desc.effective_rank != 3U ||
      af::SymbolicUtils::StaticCheckEq(desc.input_strides[kOuter], one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(desc.output_strides[kInner], one) != af::TriBool::kTrue) {
    return false;
  }
  const auto &width = desc.output_dims[kInner];
  const auto &rows = desc.output_dims[kMiddle];
  const auto &planes = desc.output_dims[kOuter];
  const auto &pitch = desc.input_strides[kInner];
  const auto alignment = CreateExpr(dma_calibration_v2::kAlignedRowElements);
  const auto ub_rows = alignment * af::sym::Ceiling(af::sym::Div(width, alignment));
  const auto &zero = af::sym::kSymbolZero;
  Expr violation = zero;
  for (const auto &part :
       {OutsideRange(width, 9, 13), OutsideRange(rows, 9, 13), OutsideRange(planes, 2, 8), OutsideRange(pitch, 32, 96),
        AbsExpr(af::sym::Mod(pitch, alignment)), OutsideRange(block_dim, 8, 48),
        AbsExpr(desc.input_strides[kMiddle] - width * pitch), AbsExpr(desc.output_strides[kMiddle] - ub_rows),
        AbsExpr(desc.output_strides[kOuter] - rows * ub_rows)}) {
    if (af::SymbolicUtils::StaticCheckGt(part, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + part;
  }
  constexpr double kStartupCycles = 608.2696492508591;
  constexpr double kCyclesPerElement = 0.23230562028891266;
  const auto fitted = CreateExpr(kStartupCycles) + CreateExpr(kCyclesPerElement) * width * rows * planes;
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_short_strided3d_fp32", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, fitted, high_core);
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

std::vector<Expr> GetNarrowSourceDomain(const Expr &m, const Expr &n, const Expr &step, const Expr &block_dim,
                                        bool include_wider_source, bool wide_pitch) {
  auto source_width = OutsideRange(m, 1, 8);
  if (include_wider_source) {
    // The same request-cost model was verified for a wider source over this
    // narrower range of long axes and GM pitches.
    const auto wider_domain = OutsideRange(m, 9, 11) + OutsideRange(n, 2608, 2880) + OutsideRange(step, 324, 384);
    source_width = af::sym::Min(source_width, wider_domain);
  }
  return wide_pitch
             ? std::vector<Expr>{OutsideRange(m, 4, 10), OutsideRange(n, 1280, 2688), OutsideRange(step, 768, 1280),
                                 OutsideRange(block_dim, 32, 56), AbsExpr(af::sym::Mod(n, CreateExpr(8)))}
             : std::vector<Expr>{source_width, OutsideRange(n, 512, 16384), OutsideRange(step, 30, 512)};
}

// Long strided reads with a narrow contiguous GM window are request-bound.
// Fit the existing startup + payload + shared-core form using the number of
// strided positions; UB payload alone hides the cost when the GM window is tiny.
bool ApplyNarrowSourceHigh(const NddmaNormalizedDesc &normalized, const NddmaNormalizedDesc *physical,
                           const Expr &block_dim, Expr &high_core, NddmaModelResult &result, bool wide_pitch = false) {
  NddmaNg2Config config{};
  if (physical == nullptr && !IsNg2Eligible(normalized, config, false)) {
    return false;
  }
  if (physical != nullptr &&
      (wide_pitch ||
       af::SymbolicUtils::StaticCheckEq(physical->output_strides[0], af::sym::kSymbolOne) != af::TriBool::kTrue)) {
    return false;
  }
  const Expr m = physical == nullptr ? normalized.output_dims[config.m_pos] : af::sym::kSymbolOne;
  const Expr n = physical == nullptr ? normalized.output_dims[config.n_pos] : physical->output_dims[0];
  const Expr step = physical == nullptr ? normalized.input_strides[config.n_pos] : physical->input_strides[0];
  const auto &zero = af::sym::kSymbolZero;
  Expr violation = zero;
  auto checks = GetNarrowSourceDomain(m, n, step, block_dim, physical == nullptr, wide_pitch);
  if (physical == nullptr) {
    checks.emplace_back(AbsExpr(normalized.input_strides[config.m_pos] - af::sym::kSymbolOne));
    checks.emplace_back(AbsExpr(normalized.output_strides[config.m_pos] - n));
  } else {
    checks.emplace_back(AbsExpr(physical->output_strides[0] - af::sym::kSymbolOne));
  }
  for (const auto &check : checks) {
    if (af::SymbolicUtils::StaticCheckGt(check, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + check;
  }
  const Expr shared = af::sym::Max(block_dim - CreateExpr(32), zero) / CreateExpr(24);
  // Use the fresh NORMAL policy for narrow reads, where neighbouring
  // blocks can share cache lines even when each launch reads new data.
  const Expr fitted =
      wide_pitch ? CreateExpr(422.798) + n / CreateExpr(2048) * (CreateExpr(6219.263) + CreateExpr(2394.068) * shared)
                 : CreateExpr(267.664177) + n * (CreateExpr(4.98188927) + CreateExpr(4.53423011) * shared);
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar(wide_pitch ? "nddma_narrow_source_wide_pitch" : "nddma_narrow_source", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
}

// Full-payload writeback competes with the next packed transpose read. This
// effective MTE2 cost is calibrated separately from the isolated NDDMA API.
bool ApplyDirectStoreContention(const NddmaNormalizedDesc &normalized, const Expr &block_dim, Expr &high_core,
                                NddmaModelResult &result) {
  if (normalized.effective_rank != 2U) {
    return false;
  }
  const auto &n = normalized.output_dims[0];
  const auto &m = normalized.output_dims[1];
  const auto &zero = af::sym::kSymbolZero;
  const Expr bytes = CreateExpr(sizeof(float)) * m * n;
  const auto outside = [&](const Expr &value, int32_t lower, int32_t upper) {
    return af::sym::Max(zero, af::sym::Max(CreateExpr(lower) - value, value - CreateExpr(upper)));
  };
  Expr violation = zero;
  for (const auto &part :
       {outside(m, 2, 8), outside(n, 512, 16384), outside(bytes, 8192, 131072),
        AbsExpr(normalized.input_strides[1] - af::sym::kSymbolOne), AbsExpr(normalized.input_strides[0] - m),
        AbsExpr(normalized.output_strides[0] - af::sym::kSymbolOne), AbsExpr(normalized.output_strides[1] - n),
        AbsExpr(af::sym::Mod(n, CreateExpr(8)))}) {
    if (af::SymbolicUtils::StaticCheckGt(part, zero) == af::TriBool::kTrue) {
      return false;
    }
    violation = violation + part;
  }
  // Train-only hinge fit; block_dim is the candidate's active parallelism,
  // never the device's total core count. See modeling/mte2/nddma direct-store evidence.
  const auto fitted =
      high_core + CreateExpr(0.0038940645164791633) * af::sym::Max(bytes * block_dim - CreateExpr(4456448), zero);
  if (af::SymbolicUtils::StaticCheckEq(violation, zero) == af::TriBool::kTrue) {
    high_core = fitted;
    return true;
  }
  Expr guarded;
  GetPerfVar("nddma_direct_store", guarded, result.ternary_ops);
  TernaryOp guard(CondType::K_EQ, violation, zero, std::make_shared<IfCase>(fitted),
                  std::make_shared<IfCase>(high_core));
  guard.SetVariable(guarded);
  result.ternary_ops[guarded] = std::move(guard);
  high_core = guarded;
  return true;
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
                              NddmaModelResult &result, bool direct_store) {
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
  // NG2 is a fitted multiplier, not a signed correction. Use the existing ND
  // model outside its positive domain, including symbolic tiles at runtime.
  if (ng2_params != nullptr && IsStaticNonPositive(GetNg2Multiplier(normalized, *ng2_params, ng2_config))) {
    ng2_params = nullptr;
  }
  std::string low_model = ng2_params != nullptr                            ? "NDDMA_ND_MULTICORE_NG2"
                          : (direct_1d || normalized.effective_rank == 1U) ? "NDDMA_1D_MULTICORE_V2"
                                                                           : "NDDMA_ND_MULTICORE_V1";
  std::string high_model = low_model;
  if (ng2_params != nullptr) {
    const auto *calibrated = GetCalibratedNg2Params(dtype_size);
    const bool use_calibrated = calibrated != nullptr && CanUseCalibratedNg2(normalized, ng2_config, block_dim);
    result.fallback_reason =
        BuildNg2Cycles(normalized, *params, *ng2_params, dtype_size, low_core, high_core, ng2_config, !use_calibrated);
    if (use_calibrated && result.fallback_reason == NddmaFallbackReason::kNone) {
      GE_ASSERT_SUCCESS(
          BuildCalibratedNg2High(normalized, ng2_config, *calibrated, dtype_size, block_dim, high_core, result));
      high_model = "NDDMA_ND_MULTICORE_NG2_V2";
    }
    const Expr multiplier = GetNg2Multiplier(normalized, *ng2_params, ng2_config);
    if (!multiplier.IsConstExpr() && result.fallback_reason == NddmaFallbackReason::kNone) {
      Expr nd_low;
      Expr nd_high;
      const auto nd_fallback_reason = BuildNddmaCoreCycles(normalized, *params, dtype_size, nd_low, nd_high);
      const auto guard_domain = [&](Expr &cycles, const Expr &fallback, const char *name) {
        Expr guarded;
        GetPerfVar(name, guarded, result.ternary_ops);
        TernaryOp guard(CondType::K_LT, af::sym::kSymbolZero, multiplier, std::make_shared<IfCase>(cycles),
                        std::make_shared<IfCase>(fallback));
        guard.SetVariable(guarded);
        result.ternary_ops[guarded] = std::move(guard);
        cycles = guarded;
      };
      if (nd_fallback_reason == NddmaFallbackReason::kNone) {
        if (af::SymbolicUtils::StaticCheckGt(block_dim, CreateExpr(kLowCoreUpperBound)) != af::TriBool::kTrue) {
          guard_domain(low_core, nd_low, "nddma_ng2_low_domain");
          low_model = RuntimeModelDispatch(low_model, "NDDMA_ND_MULTICORE_V1");
        }
        if (!use_calibrated &&
            af::SymbolicUtils::StaticCheckLe(block_dim, CreateExpr(kLowCoreUpperBound)) != af::TriBool::kTrue) {
          guard_domain(high_core, nd_high, "nddma_ng2_high_domain");
          high_model = RuntimeModelDispatch(high_model, "NDDMA_ND_MULTICORE_V1");
        }
      }
    }
  } else {
    result.fallback_reason =
        BuildNddmaCoreCycles(direct_1d ? physical_1d : normalized, *params, dtype_size, low_core, high_core);
  }
  if (result.fallback_reason != NddmaFallbackReason::kNone) {
    return af::SUCCESS;
  }
  const auto track_fit = [&](const char *name, const auto &apply_fit) {
    const auto guard_count = result.ternary_ops.size();
    if (!apply_fit()) {
      return false;
    }
    high_model = guard_count == result.ternary_ops.size() ? std::string(name) : RuntimeModelDispatch(name, high_model);
    return true;
  };
  if (ng2_params == nullptr && !direct_1d) {
    if (dtype == "float16") {
      (void)track_fit("NDDMA_ND_MULTICORE_SHORT_INNER_STRIDE_FP16_V1",
                      [&] { return ApplyFp16ShortInnerStrideScale(normalized, *params, high_core, result); });
      if (!track_fit("NDDMA_ND_MULTICORE_SPARSE_FP16_V2", [&] {
            return ApplySparseFp16NdResidualScale(normalized, *params, block_dim, high_core, result);
          })) {
        (void)track_fit("NDDMA_ND_MULTICORE_TWO_LEVEL_SPARSE_FP16_V1", [&] {
          return ApplyFp16TwoLevelSparseScale(normalized, *params, block_dim, high_core, result);
        });
        (void)track_fit("NDDMA_ND_MULTICORE_FP16_COMMENSURATE_STRIDES_V1", [&] {
          return ApplyFp16CommensurateStrideScale(normalized, *params, block_dim, high_core, result);
        });
      }
    }
    if (dtype == "float32") {
      if (!track_fit("NDDMA_ND_MULTICORE_WIDE_PADDED_ROW_FP32_V1",
                     [&] { return ApplyFp32WidePaddedRowScale(normalized, *params, block_dim, high_core, result); }) &&
          !track_fit("NDDMA_ND_MULTICORE_SPARSE_WIDE_PLANE_FP32_V1", [&] {
            return ApplyFp32SparseWidePlaneScale(normalized, *params, block_dim, high_core, result);
          })) {
        (void)track_fit("NDDMA_ND_MULTICORE_PACKED_SHORT_FP32_V1", [&] {
          return ApplyFp32PackedShortRowResidualScale(normalized, *params, block_dim, high_core, result);
        });
      }
    }
    if (!track_fit("NDDMA_ND_MULTICORE_PACKED3D_V1", [&] {
          return ApplyPacked3DHighCoreCost(normalized, dtype_size, block_dim, high_core, result) ||
                 (dtype == "float32" && ApplyFp32UnitAxisPacked3DCost(normalized, block_dim, high_core, result));
        })) {
      const char *sparse_model =
          normalized.effective_rank == 3U ? "NDDMA_ND_MULTICORE_SPARSE3D_V2" : "NDDMA_ND_MULTICORE_SPARSE4D_V2";
      (void)track_fit(sparse_model, [&] {
        return ApplySparseNdResidualScale(normalized, dtype_size, *params, block_dim, high_core, result);
      });
    }
  }
  if (dtype == "float32" && !direct_1d && ng2_params == nullptr) {
    (void)track_fit("NDDMA_ND_MULTICORE_SHORT_STRIDED3D_FP32_V1",
                    [&] { return ApplyFp32ShortStrided3DCost(normalized, block_dim, high_core, result); });
  }
  // A padded symbolic row pitch can equal the actual inner dimension at runtime
  // (e.g. 8*ceil(n/8) when n=8). Select the packed calibration only on equality.
  NddmaNg2Config packed_config{};
  const auto *packed_params = GetCalibratedNg2Params(dtype_size);
  if (packed_params != nullptr && IsNg2Eligible(normalized, packed_config, false) &&
      CanUseCalibratedNg2(normalized, packed_config, block_dim, false)) {
    const Expr &pitch = normalized.output_strides[packed_config.m_pos];
    const Expr &inner = normalized.output_dims[packed_config.n_pos];
    if (dtype == "float32" && normalized.effective_rank == 2U &&
        af::SymbolicUtils::StaticCheckEq(pitch, inner) != af::TriBool::kTrue) {
      (void)track_fit("NDDMA_ND_MULTICORE_NG2_PADDED_V1",
                      [&] { return ApplyPaddedNg2High(normalized, packed_config, block_dim, high_core, result); });
      (void)track_fit("NDDMA_ND_MULTICORE_NG2_SPARSE_PADDED_FP32_V1", [&] {
        return ApplyPaddedNg2High(normalized, packed_config, block_dim, high_core, result, true);
      });
      (void)track_fit("NDDMA_ND_MULTICORE_NG2_SHORT_PADDED_FP32_V1",
                      [&] { return ApplyShortPaddedNg2High(normalized, packed_config, block_dim, high_core, result); });
    }
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
      high_model = RuntimeModelDispatch("NDDMA_ND_MULTICORE_NG2_V2", high_model);
    }
    if (dtype == "float16" && normalized.effective_rank == 2U) {
      (void)track_fit("NDDMA_ND_MULTICORE_NG2_MEDIUM_PITCH_FP16_V1",
                      [&] { return ApplyFp16MediumPitchNg2High(normalized, packed_config, high_core, result); });
      (void)track_fit("NDDMA_ND_MULTICORE_NG2_WIDE_FP16_V1", [&] {
        return ApplyWideFp16Ng2High(normalized, packed_config, *packed_params, block_dim, high_core, result);
      });
    }
  }
  NddmaNg2Config small_config{};
  if (dtype == "int64" && normalized.effective_rank == 2U && IsNg2Eligible(normalized, small_config, false)) {
    (void)track_fit("NDDMA_ND_MULTICORE_NG2_SMALL_INT64_V1",
                    [&] { return ApplySmallInt64Ng2High(normalized, small_config, block_dim, high_core, result); });
  }
  if (dtype == "float32") {
    (void)track_fit("NDDMA_ND_MULTICORE_NARROW_SOURCE_V1", [&] {
      return ApplyNarrowSourceHigh(normalized, direct_1d ? &physical_1d : nullptr, block_dim, high_core, result);
    });
    if (!direct_1d) {
      (void)track_fit("NDDMA_ND_MULTICORE_NARROW_SOURCE_WIDE_PITCH_FP32_V1",
                      [&] { return ApplyNarrowSourceHigh(normalized, nullptr, block_dim, high_core, result, true); });
    }
  }
  if (dtype == "float32") {
    (void)track_fit("NDDMA_ND_UNIT_BROADCAST_ROW_FP32_V1",
                    [&] { return ApplyFp32UnitBroadcastRowCost(normalized, block_dim, high_core, result); });
    (void)track_fit("NDDMA_ND_MULTICORE_SMALL_DENSE_PADDED_FP32_V1", [&] {
      return ApplyDensePaddedNddma(normalized, block_dim, high_core, result, kSmallDensePaddedTranspose);
    });
    (void)track_fit("NDDMA_ND_MULTICORE_DENSE_PADDED_FP32_V1", [&] {
      return ApplyDensePaddedNddma(normalized, block_dim, high_core, result, kDensePaddedTranspose);
    });
    (void)track_fit("NDDMA_ND_MULTICORE_NARROW_DENSE_PADDED_FP32_V1", [&] {
      return ApplyDensePaddedNddma(normalized, block_dim, high_core, result, kNarrowDensePaddedTranspose);
    });
    (void)track_fit("NDDMA_ND_PADDED_INNER_BROADCAST_FP32_V1", [&] {
      return ApplyDensePaddedNddma(normalized, block_dim, high_core, result, kPaddedInnerBroadcast);
    });
  }
  if (direct_store && dtype == "float32") {
    (void)track_fit("NDDMA_ND_MULTICORE_DIRECT_STORE_V1",
                    [&] { return ApplyDirectStoreContention(normalized, block_dim, high_core, result); });
  }
  GE_ASSERT_SUCCESS(SelectCoreBranch(block_dim, low_core, high_core, low_model, high_model, result));
  // Fitted residual terms can become non-positive outside their measured
  // domain (for example, tiny B with a large effective stride).  Do not emit
  // an invalid performance expression; use the existing legacy path instead.
  if (IsStaticNonPositive(result.cycles)) {
    result.selected = false;
    result.fallback_reason = NddmaFallbackReason::kSchemaMismatch;
    return af::SUCCESS;
  }
  result.selected = true;
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
