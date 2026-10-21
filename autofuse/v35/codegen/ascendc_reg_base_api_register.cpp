/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_api_registry.h"

namespace codegen {
namespace {
class Register {
 public:
  Register();
};

static std::string StripBroadcastSelfIncludes(const std::string &src) {
  std::string result;
  size_t pos = 0;
  while (pos < src.size()) {
    size_t end = src.find('\n', pos);
    if (end == std::string::npos) end = src.size();
    std::string line = src.substr(pos, end - pos);
    if (line.find("#include \"broadcast") == std::string::npos) {
      result += line + "\n";
    }
    pos = end + 1;
  }
  return result;
}

static std::string StripReduceSelfIncludes(const std::string &src) {
  std::string result;
  size_t pos = 0;
  while (pos < src.size()) {
    size_t end = src.find('\n', pos);
    if (end == std::string::npos) end = src.size();
    std::string line = src.substr(pos, end - pos);
    if (line.find("#include \"reduce/") == std::string::npos) {
      result += line + "\n";
    }
    pos = end + 1;
  }
  return result;
}

Register::Register() {
  const std::string kAscendcCastRegStr = {
#include "cast_reg_base.h"
  };
  const std::string kAscendcCompareRegStr = {
#include "compare_reg_base.h"
  };
  const std::string kAscendcBitwiseAndRegStr = {
#include "bitwise_and_reg_base.h"
  };
  const std::string kAscendcBitwiseOrRegStr = {
#include "bitwise_or_reg_base.h"
  };
  const std::string kAscendcBitwiseXorRegStr = {
#include "bitwise_xor_reg_base.h"
  };
  const std::string kAscendcBitwiseNotRegStr = {
#include "bitwise_not_reg_base.h"
  };
  const std::string kAscendcConcatRegBaseStr = {
#include "concat_reg_base.h"
  };
  const std::string kAscendcDatacopyRegBaseStr = {
#include "datacopy_reg_base.h"
  };
  const std::string kAscendcDatacopyNddmaRegBaseStr = {
#include "datacopy_nddma_reg_base.h"
  };
  const std::string kAscendcReduce_initRegBase = {
#include "reduce_init_reg_base.h"
  };
  const std::string kAscendcReduceExtendRegBase = StripReduceSelfIncludes(std::string{
#include "reduce/reduce_common_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_common_util_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_common_ar_reuse_align_less_than_vl_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_common_ar_reuse_align_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_common_ra_reuse_align_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_common_ar_ra_reuse_unalign_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_sum_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_mean_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_max_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_min_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_prod_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_any_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_all_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/mean_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/sum_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce/reduce_xor_sum_3510_impl_reg_base.h"
                                                  }) +
                                                  StripReduceSelfIncludes(std::string{
#include "reduce_extend_reg_base.h"
                                                  });
  const std::string kAscendcFloorDivRegBaseStr = {
#include "floor_div_reg_base.h"
  };
  const std::string kAscendcSignRegBaseStr = {
#include "sign_reg_base.h"
  };
  const std::string kAscendcWhereRegBaseStr = {
#include "where_reg_base.h"
  };
  const std::string kAscendcGatherRegBaseStr = {
#include "gather_reg_base.h"
  };
  const std::string kAscendcUtilsRegBaseStr = {
#include "utils_reg_base.h"
  };
  const std::string kAscendcBroadcastRegStr = StripBroadcastSelfIncludes(std::string{
#include "broadcast_extend_impl_reg_base.h"
                                              }) +
                                              StripBroadcastSelfIncludes(std::string{
#include "broadcast_reg_base.h"
                                              });
  const std::string kAscendcLogicalNotStr = {
#include "logical_not_reg_base.h"
  };
  const std::string kAscendcClipByValueRegStr = {
#include "clipbyvalue_reg_base.h"
  };
  const std::string kAscendcLogicalRegBaseStr = {
#include "logical_reg_base.h"
  };
  const std::string kAscendcLogicalXorRegBaseStr = {
#include "logical_xor_reg_base.h"
  };
  const std::string kAscendcPowRegBaseStr = {
#include "pow_reg_base.h"
  };
  const std::string kAscendcExp2RegBaseStr = {
#include "exp2_reg_base.h"
  };
  const std::string kAscendcLog1pRegBaseStr = {
#include "log1p_reg_base.h"
  };
  const std::string kAscendcErfRegBaseStr = {
#include "erf_reg_base.h"
  };
  const std::string kAscendcTanhRegBaseStr = {
#include "tanh_reg_base.h"
  };
  const std::string kAscendcSubRegBaseStr = {
#include "sub_reg_base.h"
  };
  const std::string kAscendcDivRegBaseStr = {
#include "div_reg_base.h"
  };
  const std::string kAscendcSplitRegBaseStr = {
#include "split_reg_base.h"
  };
  const std::string kAscendcAtan2RegBaseStr = {
#include "atan2_reg_base.h"
  };
  const std::string kAscendcCopySignRegBaseStr = {
#include "copy_sign_reg_base.h"
  };
  const std::string kAscendcErfcxRegBaseStr = {
#include "erfcx_reg_base.h"
  };
  const std::string kAscendcExpm1RegBaseStr = {
#include "expm1_reg_base.h"
  };
  const std::string kAscendcFmodRegBaseStr = {
#include "fmod_reg_base.h"
  };
  const std::string kAscendcTruncDivRegBaseStr = {
#include "trunc_div_reg_base.h"
  };
  const std::string kAscendcRemainderRegBaseStr = {
#include "remainder_reg_base.h"
  };
  const std::string kAscendcSoftmaxAfRegBaseStr = {
#include "softmax_af_reg_base.h"
  };
  const std::string kAscendcNegRegBaseStr = {
#include "neg_reg_base.h"
  };
  const std::string kAscendcSquareRegBaseStr = {
#include "square_reg_base.h"
  };
  const std::string kAscendcTransposeRegBaseStr = {
#include "transpose_reg_base.h"
  };
  const std::string kAscendcTrigonometricFunctionUtilsRegBaseStr = {
#include "trigonometric_function_utils_reg_base.h"
  };
  const std::string kAscendcModifiedBesselUtilsRegBaseStr = {
#include "modified_bessel_utils_reg_base.h"
  };
  const std::string kAscendcModifiedBesselI0RegBaseStr = {
#include "modified_bessel_i0_reg_base.h"
  };
  const std::string kAscendcModifiedBesselI1RegBaseStr = {
#include "modified_bessel_i1_reg_base.h"
  };
  const std::string kAscendcModifiedBesselK0RegBaseStr = {
#include "modified_bessel_k0_reg_base.h"
  };
  const std::string kAscendcModifiedBesselK1RegBaseStr = {
#include "modified_bessel_k1_reg_base.h"
  };
  const std::string kAscendcLaguerrePolynomialLRegBaseStr = {
#include "laguerre_polynomial_l_reg_base.h"
  };
  const std::string kAscendcLegendrePolynomialPRegBaseStr = {
#include "legendre_polynomial_p_reg_base.h"
  };
  const std::string kAscendcAiryAiRegBaseStr = {
#include "airy_ai_reg_base.h"
  };
  const std::string kAscendcErfinvRegBaseStr = {
#include "erfinv_reg_base.h"
  };
  const std::string kAscendcBesselJUtilsRegBaseStr = {
#include "bessel_j_utils_reg_base.h"
  };
  const std::string kAscendcI0RegBaseStr = {
#include "i0_reg_base.h"
  };
  const std::string kAscendcI0eRegBaseStr = {
#include "i0e_reg_base.h"
  };
  const std::string kAscendcI1eRegBaseStr = {
#include "i1e_reg_base.h"
  };
  const std::string kAscendcBesselJ0RegBaseStr = {
#include "bessel_j0_reg_base.h"
  };
  const std::string kAscendcBesselJ1RegBaseStr = {
#include "bessel_j1_reg_base.h"
  };
  const std::string kAscendcBesselY0RegBaseStr = {
#include "bessel_y0_reg_base.h"
  };
  const std::string kAscendcBesselY1RegBaseStr = {
#include "bessel_y1_reg_base.h"
  };
  const std::string kAscendcScaledModifiedBesselK0RegBaseStr = {
#include "scaled_modified_bessel_k0_reg_base.h"
  };
  const std::string kAscendcScaledModifiedBesselK1RegBaseStr = {
#include "scaled_modified_bessel_k1_reg_base.h"
  };
  const std::string kAscendcSphericalBesselJ0RegBaseStr = {
#include "spherical_bessel_j0_reg_base.h"
  };
  const std::string kAscendcIgammaRegBaseStr = {
#include "igamma_reg_base.h"
  };
  const std::string kAscendcIgammacRegBaseStr = {
#include "igammac_reg_base.h"
  };
  const std::string kAscendcIgammacHelperSeriesRegBaseStr = {
#include "igammac_helper/series_reg_base.h"
  };
  const std::string kAscendcIgammacHelperContinuedFractionRegBaseStr = {
#include "igammac_helper/continued_fraction_reg_base.h"
  };
  const std::string kAscendcIgammacHelperAsymptoticSeriesRegBaseStr = {
#include "igammac_helper/asymptotic_series_reg_base.h"
  };
  const std::string kAscendcIgammacHelperSeriesComplementRegBaseStr = {
#include "igammac_helper/series_complement_reg_base.h"
  };
  const std::string kAscendcZetaRegBaseStr = {
#include "zeta_reg_base.h"
  };
  const std::string kAscendcNdtrRegBaseStr = {
#include "ndtr_reg_base.h"
  };
  const std::string kAscendcNdtriRegBaseStr = {
#include "ndtri_reg_base.h"
  };
  const std::string kAscendcLogNdtrRegBaseStr = {
#include "log_ndtr_reg_base.h"
  };
  const std::string kAscendcNextAfterRegBaseStr = {
#include "next_after_reg_base.h"
  };
  const std::string kAscendcPolyGammaRegBaseStr = {
#include "polygamma_reg_base.h"
  };
  const std::string kAscendcRandomRegBaseStr = {
#include "random_reg_base.h"
  };
  const std::string kAscendcSignBitRegBaseStr = {
#include "signbit_reg_base.h"
  };
  const std::string kAscendcFrexpRegBaseStr = {
#include "frexp_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialUtilsRegBaseStr = {
#include "shifted_chebyshev_polynomial_utils_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialTRegBaseStr = {
#include "shifted_chebyshev_polynomial_t_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialURegBaseStr = {
#include "shifted_chebyshev_polynomial_u_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialVRegBaseStr = {
#include "shifted_chebyshev_polynomial_v_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialWRegBaseStr = {
#include "shifted_chebyshev_polynomial_w_reg_base.h"
  };
  const std::string kAscendcChebyshevPolynomialUtilsRegBaseStr = {
#include "chebyshev_polynomial_utils_reg_base.h"
  };
  const std::string kAscendcChebyshevPolynomialTRegBaseStr = {
#include "chebyshev_polynomial_t_reg_base.h"
  };
  const std::string kAscendcChebyshevPolynomialURegBaseStr = {
#include "chebyshev_polynomial_u_reg_base.h"
  };
  const std::string kAscendcChebyshevPolynomialVRegBaseStr = {
#include "chebyshev_polynomial_v_reg_base.h"
  };
  const std::string kAscendcChebyshevPolynomialWRegBaseStr = {
#include "chebyshev_polynomial_w_reg_base.h"
  };
  const std::string kAscendcHermitePolynomialHRegBaseStr = {
#include "hermite_polynomial_h_reg_base.h"
  };
  const std::string kAscendcHermitePolynomialHeRegBaseStr = {
#include "hermite_polynomial_he_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSimtRegBaseStr = {
#include "indirect_load_simt_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSimdPolicyRegBaseStr = {
#include "indirect_load_simd_policy_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSimdRegBaseStr = {
#include "indirect_load_simd_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSkRegBaseStr = {
#include "indirect_load_sk_reg_base.h"
  };
  const std::string kAscendcSinRegBaseStr = {
#include "sin_reg_base.h"
  };
  const std::string kAscendcCosRegBaseStr = {
#include "cos_reg_base.h"
  };
  const std::string kAscendcAtanRegBaseStr = {
#include "atan_reg_base.h"
  };
  const std::string kAscendcAsinRegBaseStr = {
#include "asin_reg_base.h"
  };
  const std::string kAscendcAcosRegBaseStr = {
#include "acos_reg_base.h"
  };
  const std::string kAscendcTanRegBaseStr = {
#include "tan_reg_base.h"
  };
  const std::string kAscendcAtanhRegBaseStr = {
#include "atanh_reg_base.h"
  };
  const std::string kAscendcSinhRegBaseStr = {
#include "sinh_reg_base.h"
  };
  const std::string kAscendcAsinhRegBaseStr = {
#include "asinh_reg_base.h"
  };
  const std::string kAscendcRsqrtRegBaseStr = {
#include "rsqrt_reg_base.h"
  };
  const std::string kAscendcReciprocalRegBaseStr = {
#include "reciprocal_reg_base.h"
  };
  const std::string kAscendcCeilRegBaseStr = {
#include "ceil_reg_base.h"
  };
  const std::string kAscendcAcoshRegBaseStr = {
#include "acosh_reg_base.h"
  };
  const std::string kAscendcCoshRegBaseStr = {
#include "cosh_reg_base.h"
  };
  std::unordered_map<std::string, std::string> api_to_file{
      {"cast_reg_base.h", kAscendcCastRegStr},
      {"compare_reg_base.h", kAscendcCompareRegStr},
      {"bitwise_and_reg_base.h", kAscendcBitwiseAndRegStr},
      {"bitwise_or_reg_base.h", kAscendcBitwiseOrRegStr},
      {"bitwise_xor_reg_base.h", kAscendcBitwiseXorRegStr},
      {"bitwise_not_reg_base.h", kAscendcBitwiseNotRegStr},
      {"concat_reg_base.h", kAscendcConcatRegBaseStr},
      {"datacopy_reg_base.h", kAscendcDatacopyRegBaseStr},
      {"datacopy_nddma_reg_base.h", kAscendcDatacopyNddmaRegBaseStr},
      {"pow_reg_base.h", kAscendcPowRegBaseStr},
      {"exp2_reg_base.h", kAscendcExp2RegBaseStr},
      {"log1p_reg_base.h", kAscendcLog1pRegBaseStr},
      {"erf_reg_base.h", kAscendcErfRegBaseStr},
      {"tanh_reg_base.h", kAscendcTanhRegBaseStr},
      {"reduce_init_reg_base.h", kAscendcReduce_initRegBase},
      {"reduce_extend_reg_base.h", kAscendcReduceExtendRegBase},
      {"floor_div_reg_base.h", kAscendcFloorDivRegBaseStr},
      {"sign_reg_base.h", kAscendcSignRegBaseStr},
      {"where_reg_base.h", kAscendcWhereRegBaseStr},
      {"gather_reg_base.h", kAscendcGatherRegBaseStr},
      {"broadcast_reg_base.h", kAscendcBroadcastRegStr},
      {"utils_reg_base.h", kAscendcUtilsRegBaseStr},
      {"logical_not_reg_base.h", kAscendcLogicalNotStr},
      {"clipbyvalue_reg_base.h", kAscendcClipByValueRegStr},
      {"logical_reg_base.h", kAscendcLogicalRegBaseStr},
      {"logical_xor_reg_base.h", kAscendcLogicalXorRegBaseStr},
      {"split_reg_base.h", kAscendcSplitRegBaseStr},
      {"sub_reg_base.h", kAscendcSubRegBaseStr},
      {"div_reg_base.h", kAscendcDivRegBaseStr},
      {"atan2_reg_base.h", kAscendcAtan2RegBaseStr},
      {"copy_sign_reg_base.h", kAscendcCopySignRegBaseStr},
      {"erfcx_reg_base.h", kAscendcErfcxRegBaseStr},
      {"expm1_reg_base.h", kAscendcExpm1RegBaseStr},
      {"fmod_reg_base.h", kAscendcFmodRegBaseStr},
      {"trunc_div_reg_base.h", kAscendcTruncDivRegBaseStr},
      {"remainder_reg_base.h", kAscendcRemainderRegBaseStr},
      {"softmax_af_reg_base.h", kAscendcSoftmaxAfRegBaseStr},
      {"neg_reg_base.h", kAscendcNegRegBaseStr},
      {"square_reg_base.h", kAscendcSquareRegBaseStr},
      {"transpose_reg_base.h", kAscendcTransposeRegBaseStr},
      {"trigonometric_function_utils_reg_base.h", kAscendcTrigonometricFunctionUtilsRegBaseStr},
      {"modified_bessel_utils_reg_base.h", kAscendcModifiedBesselUtilsRegBaseStr},
      {"modified_bessel_i0_reg_base.h", kAscendcModifiedBesselI0RegBaseStr},
      {"modified_bessel_i1_reg_base.h", kAscendcModifiedBesselI1RegBaseStr},
      {"modified_bessel_k0_reg_base.h", kAscendcModifiedBesselK0RegBaseStr},
      {"modified_bessel_k1_reg_base.h", kAscendcModifiedBesselK1RegBaseStr},
      {"laguerre_polynomial_l_reg_base.h", kAscendcLaguerrePolynomialLRegBaseStr},
      {"legendre_polynomial_p_reg_base.h", kAscendcLegendrePolynomialPRegBaseStr},
      {"airy_ai_reg_base.h", kAscendcAiryAiRegBaseStr},
      {"erfinv_reg_base.h", kAscendcErfinvRegBaseStr},
      {"i0_reg_base.h", kAscendcI0RegBaseStr},
      {"i0e_reg_base.h", kAscendcI0eRegBaseStr},
      {"i1e_reg_base.h", kAscendcI1eRegBaseStr},
      {"bessel_j_utils_reg_base.h", kAscendcBesselJUtilsRegBaseStr},
      {"bessel_j0_reg_base.h", kAscendcBesselJ0RegBaseStr},
      {"bessel_j1_reg_base.h", kAscendcBesselJ1RegBaseStr},
      {"bessel_y0_reg_base.h", kAscendcBesselY0RegBaseStr},
      {"bessel_y1_reg_base.h", kAscendcBesselY1RegBaseStr},
      {"scaled_modified_bessel_k0_reg_base.h", kAscendcScaledModifiedBesselK0RegBaseStr},
      {"scaled_modified_bessel_k1_reg_base.h", kAscendcScaledModifiedBesselK1RegBaseStr},
      {"spherical_bessel_j0_reg_base.h", kAscendcSphericalBesselJ0RegBaseStr},
      {"igamma_reg_base.h", kAscendcIgammaRegBaseStr},
      {"igammac_reg_base.h", kAscendcIgammacRegBaseStr},
      {"igammac_helper/series_reg_base.h", kAscendcIgammacHelperSeriesRegBaseStr},
      {"igammac_helper/continued_fraction_reg_base.h", kAscendcIgammacHelperContinuedFractionRegBaseStr},
      {"igammac_helper/asymptotic_series_reg_base.h", kAscendcIgammacHelperAsymptoticSeriesRegBaseStr},
      {"igammac_helper/series_complement_reg_base.h", kAscendcIgammacHelperSeriesComplementRegBaseStr},
      {"zeta_reg_base.h", kAscendcZetaRegBaseStr},
      {"ndtr_reg_base.h", kAscendcNdtrRegBaseStr},
      {"ndtri_reg_base.h", kAscendcNdtriRegBaseStr},
      {"log_ndtr_reg_base.h", kAscendcLogNdtrRegBaseStr},
      {"next_after_reg_base.h", kAscendcNextAfterRegBaseStr},
      {"polygamma_reg_base.h", kAscendcPolyGammaRegBaseStr},
      {"random_reg_base.h", kAscendcRandomRegBaseStr},
      {"signbit_reg_base.h", kAscendcSignBitRegBaseStr},
      {"frexp_reg_base.h", kAscendcFrexpRegBaseStr},
      {"shifted_chebyshev_polynomial_utils_reg_base.h", kAscendcShiftedChebyshevPolynomialUtilsRegBaseStr},
      {"shifted_chebyshev_polynomial_t_reg_base.h", kAscendcShiftedChebyshevPolynomialTRegBaseStr},
      {"shifted_chebyshev_polynomial_u_reg_base.h", kAscendcShiftedChebyshevPolynomialURegBaseStr},
      {"shifted_chebyshev_polynomial_v_reg_base.h", kAscendcShiftedChebyshevPolynomialVRegBaseStr},
      {"shifted_chebyshev_polynomial_w_reg_base.h", kAscendcShiftedChebyshevPolynomialWRegBaseStr},
      {"chebyshev_polynomial_utils_reg_base.h", kAscendcChebyshevPolynomialUtilsRegBaseStr},
      {"chebyshev_polynomial_t_reg_base.h", kAscendcChebyshevPolynomialTRegBaseStr},
      {"chebyshev_polynomial_u_reg_base.h", kAscendcChebyshevPolynomialURegBaseStr},
      {"chebyshev_polynomial_v_reg_base.h", kAscendcChebyshevPolynomialVRegBaseStr},
      {"chebyshev_polynomial_w_reg_base.h", kAscendcChebyshevPolynomialWRegBaseStr},
      {"hermite_polynomial_h_reg_base.h", kAscendcHermitePolynomialHRegBaseStr},
      {"hermite_polynomial_he_reg_base.h", kAscendcHermitePolynomialHeRegBaseStr},
      {"indirect_load_simt_reg_base.h", kAscendcIndirectLoadSimtRegBaseStr},
      {"indirect_load_simd_policy_reg_base.h", kAscendcIndirectLoadSimdPolicyRegBaseStr},
      {"indirect_load_simd_reg_base.h", kAscendcIndirectLoadSimdRegBaseStr},
      {"indirect_load_sk_reg_base.h", kAscendcIndirectLoadSkRegBaseStr},
      {"sin_reg_base.h", kAscendcSinRegBaseStr},
      {"cos_reg_base.h", kAscendcCosRegBaseStr},
      {"atan_reg_base.h", kAscendcAtanRegBaseStr},
      {"asin_reg_base.h", kAscendcAsinRegBaseStr},
      {"acos_reg_base.h", kAscendcAcosRegBaseStr},
      {"tan_reg_base.h", kAscendcTanRegBaseStr},
      {"atanh_reg_base.h", kAscendcAtanhRegBaseStr},
      {"sinh_reg_base.h", kAscendcSinhRegBaseStr},
      {"asinh_reg_base.h", kAscendcAsinhRegBaseStr},
      {"rsqrt_reg_base.h", kAscendcRsqrtRegBaseStr},
      {"reciprocal_reg_base.h", kAscendcReciprocalRegBaseStr},
      {"ceil_reg_base.h", kAscendcCeilRegBaseStr},
      {"acosh_reg_base.h", kAscendcAcoshRegBaseStr},
      {"cosh_reg_base.h", kAscendcCoshRegBaseStr},
  };

  AscendCApiRegistry::GetInstance().RegisterApi(api_to_file);
}

Register __attribute__((unused)) reg_base_api_register;
}  // namespace
}  // namespace codegen
