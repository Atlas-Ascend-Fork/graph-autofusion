# -----------------------------------------------------------------------------------------------------------
#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software: you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(test_regbase_ascendc OBJECT
    ./test_sub.cpp
    ./test_log1p.cpp
    ./test_erfcx.cpp
    ./test_floor_div.cpp
    ./test_cast.cpp
    ./test_compare.cpp
    ./test_concat.cpp
    ./test_gather.cpp
    ./test_logical_and.cpp
    ./test_logical_not.cpp
    ./test_logical_or.cpp
    ./test_max.cpp
    ./test_min.cpp
    ./test_where.cpp
    ./test_reduce_init.cpp
    ./test_reduce_extend.cpp
    ./test_add.cpp
    ./test_airy_ai.cpp
    ./test_bitwise_and.cpp
    ./test_bitwise_not.cpp
    ./test_bitwise_or.cpp
    ./test_bitwise_xor.cpp
    ./test_rshift.cpp
    ./test_sin.cpp
    ./test_pow.cpp
    ./test_reciprocal.cpp
    ./test_reciprocal_extend.cpp
    ./test_relu.cpp
    ./test_rsqrt.cpp
    ./test_round.cpp
    ./test_sinh.cpp
    ./test_tanh.cpp
    ./test_silu.cpp
    ./test_tan.cpp
    ./test_square.cpp
    ./test_signbit.cpp
    ./test_round_to_int.cpp
    ./test_trunc_div.cpp
    ./test_trunc_to_int.cpp
    ./test_xor.cpp
    ./test_remainder.cpp
    ./test_modified_bessel_i0.cpp
    ./test_modified_bessel_i1.cpp
    ./test_modified_bessel_k0.cpp
    ./test_modified_bessel_k1.cpp
    ./test_scaled_modified_bessel_k0.cpp
    ./test_scaled_modified_bessel_k1.cpp
    ./test_i0e.cpp
    ./test_i1e.cpp
    ./test_erfinv.cpp
    ./test_laguerre_polynomial_l.cpp
    ./test_legendre_polynomial_p.cpp
    ./test_igamma.cpp
    ./test_igammac.cpp
    ./test_polygamma.cpp
    ./test_frexp.cpp
    ./test_transpose.cpp
    ./test_bessel_j0.cpp
    ./test_bessel_j1.cpp
    ./test_bessel_y0.cpp
    ./test_bessel_y1.cpp
    ./test_bucketize.cpp
    ./test_shifted_chebyshev_polynomial_t.cpp
    ./test_shifted_chebyshev_polynomial_u.cpp
    ./test_shifted_chebyshev_polynomial_v.cpp
    ./test_shifted_chebyshev_polynomial_w.cpp
    ./test_chebyshev_polynomial_t.cpp
    ./test_chebyshev_polynomial_u.cpp
    ./test_chebyshev_polynomial_v.cpp
    ./test_chebyshev_polynomial_w.cpp
    ./test_zeta.cpp
    ./test_spherical_bessel_j0.cpp
    ./test_ndtri.cpp
    ./test_ndtr.cpp
    ./test_log_ndtr.cpp
    ./test_nextafter.cpp
    ./test_i0.cpp
    ./test_acosh.cpp
    ./test_cosh.cpp
    ./test_ceil.cpp
    ./test_hermite_polynomial_h.cpp
    ./test_hermite_polynomial_he.cpp
    ./test_hypot.cpp
    ./test_softmax.cpp
    )
target_link_libraries(test_regbase_ascendc
    ascendc_api_regbase_extend
    tikicpulib_ascend950pr_9599
    Boost::boost
    GTest::gtest)

target_include_directories(test_regbase_ascendc PUBLIC
    ${CODE_ROOT_DIR}/ascendc
    ${CODE_ROOT_DIR}/v35/ascendc
    ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include
)
