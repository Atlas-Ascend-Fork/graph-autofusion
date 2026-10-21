# -----------------------------------------------------------------------------------------------------------
#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software: you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(ascendc_api_regbase_extend_src
        concat.h
        datacopy.h
        broadcast.h
        broadcast_extend_impl.h
        datacopy_nddma.h
        gather.h
        clipbyvalue.h
        utils.h
        logical_not.h
        logical_xor.h
        cast.h
        logical.h
        compare.h
        bitwise_and.h
        bitwise_or.h
        bitwise_xor.h
        bitwise_not.h
        floor_div.h
        pow.h
        polygamma.h
        random.h
        split.h
        where.h
        erf.h
        erfinv.h
        exp2.h
        log1p.h
        tanh.h
        reciprocal.h
        sub.h
        div.h
        sign.h
        signbit.h
        reduce_init.h
        reduce_extend.h
        reduce/reduce_common.h
        reduce/reduce_common_util_3510_impl.h
        reduce/reduce_common_ar_reuse_align_less_than_vl_3510_impl.h
        reduce/reduce_common_ar_reuse_align_3510_impl.h
        reduce/reduce_common_ra_reuse_align_3510_impl.h
        reduce/reduce_common_ar_ra_reuse_unalign_3510_impl.h
        reduce/reduce_sum_3510_impl.h
        reduce/reduce_mean_3510_impl.h
        reduce/reduce_max_3510_impl.h
        reduce/reduce_min_3510_impl.h
        reduce/reduce_prod_3510_impl.h
        reduce/reduce_any_3510_impl.h
        reduce/reduce_all_3510_impl.h
        reduce/mean_3510_impl.h
        reduce/sum_3510_impl.h
        reduce/reduce_xor_sum_3510_impl.h
        atan2.h
        airy_ai.h
        copy_sign.h
        erfcx.h
        expm1.h
        fmod.h
        neg.h
        square.h
        softmax_af.h
        trunc_div.h
        remainder.h
        bessel_j_utils.h
        bessel_j0.h
        bessel_j1.h
        spherical_bessel_j0.h
        bessel_y0.h
        bessel_y1.h
        bucketize.h
        modified_bessel_i0.h
        i0e.h
        modified_bessel_i1.h
        i1e.h
        modified_bessel_k0.h
        modified_bessel_k1.h
        modified_bessel_utils.h
        scaled_modified_bessel_k0.h
        scaled_modified_bessel_k1.h
        laguerre_polynomial_l.h
        legendre_polynomial_p.h
        igamma.h
        igammac.h
        trigonometric_function_utils.h
        transpose.h
        frexp.h
        zeta.h
        ndtri.h
        shifted_chebyshev_polynomial_utils.h
        shifted_chebyshev_polynomial_t.h
        shifted_chebyshev_polynomial_u.h
        shifted_chebyshev_polynomial_v.h
        shifted_chebyshev_polynomial_w.h
        ndtr.h
        log_ndtr.h
        next_after.h
        i0.h
        hermite_polynomial_h.h
        hermite_polynomial_he.h
        igammac_helper/series.h
        igammac_helper/continued_fraction.h
        igammac_helper/asymptotic_series.h
        igammac_helper/series_complement.h
        hypot.h
        indirect_load_simd_policy.h
        indirect_load_simd.h
        indirect_load_sk.h
        indirect_load_simt.h
        chebyshev_polynomial_utils.h
        chebyshev_polynomial_t.h
        chebyshev_polynomial_u.h
        chebyshev_polynomial_v.h
        chebyshev_polynomial_w.h
        sin.h
        cos.h
        atan.h
        sinh.h
        cosh.h
        asinh.h
        acosh.h
        ceil.h
        asin.h
        acos.h
        tan.h
        atanh.h
        silu.h
        rsqrt.h
)

set(generated_regbase_files)

foreach(header ${ascendc_api_regbase_extend_src})
    string(REGEX REPLACE "\\.h$" "_reg_base.h" str_header ${header})
    get_filename_component(output_subdir ${str_header} DIRECTORY)
    if(output_subdir)
        set(make_dir_cmd COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${output_subdir})
    else()
        set(make_dir_cmd)
    endif()
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            ${make_dir_cmd}
            COMMAND cat ${header}
            | sed "'1i\\R\"===('"
            | sed "'$$a\\)===\"'"
            > ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
            DEPENDS ${header}
    )
    list(APPEND generated_regbase_files ${CMAKE_CURRENT_BINARY_DIR}/${str_header})
endforeach()

add_custom_target(generated_regbase_files ALL DEPENDS ${generated_regbase_files})

add_library(ascendc_api_regbase_extend INTERFACE)
target_include_directories(ascendc_api_regbase_extend INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(ascendc_api_regbase_extend generated_regbase_files)
