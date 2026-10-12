function(mark_indirect_load_codegen_and_e2e test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${ARGN})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE ${ARGN})
endfunction()

function(mark_indirect_load_red test_name)
    set_property(TEST ${test_name}_codegen_v2 ${test_name}_e2e_v2 APPEND PROPERTY LABELS red)
endfunction()

function(mark_indirect_load_codegen test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${ARGN})
endfunction()

function(add_indirect_load_e2e_case test_name rank axis input_pre_type use_exp2 tiling_key selected_template expected_templates x0 x1 x2 x3 index0 index1 index2 index3)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        ${ARGN}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            indirect_load_store_test_kernel.cpp
            indirect_load_store_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)

    set(case_definitions
        IL_CASE_STORE
        IL_RANK=${rank}
        IL_AXIS=${axis}
        IL_INPUT_PRE_TYPE=${input_pre_type}
        IL_USE_EXP2=${use_exp2}
        IL_TILING_KEY=${tiling_key}
        IL_SELECTED_TEMPLATE=${selected_template}
        IL_EXPECTED_TEMPLATES=${expected_templates}
        IL_X_S0=${x0}
        IL_X_S1=${x1}
        IL_X_S2=${x2}
        IL_X_S3=${x3}
        IL_INDEX_S0=${index0}
        IL_INDEX_S1=${index1}
        IL_INDEX_S2=${index2}
        IL_INDEX_S3=${index3})
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

function(mark_indirect_load_static_shape test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_STATIC_SHAPE)
endfunction()

function(mark_indirect_load_sk test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_EXPECT_SK)
endfunction()

function(mark_indirect_load_wide_types test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_DATA_FLOAT IL_INDEX_INT64)
endfunction()

function(mark_indirect_load_input_outer_stride test_name stride)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_INPUT_OUTER_STRIDE=${stride})
endfunction()
function(mark_indirect_load_large_simt test_name)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_EXPECT_SIMT_MULTI_ROUND)
endfunction()

function(mark_indirect_load_mixed_index_pre test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_MIXED_INDEX_PRE)
endfunction()

function(mark_indirect_load_output_post test_name output_post_type expect_simt)
    mark_indirect_load_codegen_and_e2e(${test_name}
        IL_OUTPUT_POST_TYPE=${output_post_type}
        IL_EXPECT_SIMT=${expect_simt})
endfunction()

function(mark_indirect_load_simt_selected test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_EXPECT_SIMT_SELECTED)
endfunction()

function(mark_indirect_load_simt_codegen test_name policy offset_bits)
    mark_indirect_load_codegen(${test_name}
        IL_EXPECT_SIMT_POLICY=${policy}
        IL_EXPECT_SIMT_OFFSET_BITS=${offset_bits})
endfunction()

# Registers an IndirectLoad backend E2E case whose kernel sources are
# ${kernel_stem}_kernel.cpp / ${kernel_stem}_tiling.cpp.
function(add_indirect_load_graph_case test_name kernel_stem tiling_key)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            ${kernel_stem}_kernel.cpp
            ${kernel_stem}_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    mark_indirect_load_codegen_and_e2e(${test_name} ${ARGN})
endfunction()

# Sweep registration: aggregates several IndirectLoad variants into ONE ST case.
#
# Every variant still gets its own codegen/e2e binary because the generated
# kernel/tiling units export fixed symbols (`AutofuseTiling`, the tiling-data
# struct, ...) and bake the value dtype into the kernel, so linking variants into
# a single binary is not possible without mass symbol renaming.  Instead the
# variants are built as private targets (NO_TEST) and one driver test runs them
# sequentially, keeping the case count at one per sweep axis.
#
# Usage:
#   add_indirect_load_sweep_variant(<sweep> <variant> <tiling_key> <defs...>)
#   finalize_indirect_load_sweep(<sweep>)
function(add_indirect_load_sweep_variant sweep_name variant_name tiling_key)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${variant_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${variant_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            ${variant_name}_kernel.cpp
            ${variant_name}_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp
        NO_TEST)
    mark_indirect_load_codegen_and_e2e(${variant_name} ${ARGN})
    set(${sweep_name}_variants "${${sweep_name}_variants}" "${variant_name}" PARENT_SCOPE)
endfunction()

function(finalize_indirect_load_sweep sweep_name)
    set(variants ${${sweep_name}_variants})
    if(NOT variants)
        message(FATAL_ERROR "sweep '${sweep_name}' has no variant")
    endif()
    set(codegen_deps)
    set(e2e_deps)
    set(codegen_content "set(SWEEP_NAME \"${sweep_name}_codegen\")\n")
    string(APPEND codegen_content "set(SWEEP_REPORT_DIR \"${CMAKE_INSTALL_PREFIX}/report/v35/st\")\n")
    string(APPEND codegen_content "set(SWEEP_LD_LIBRARY_PATH \"${BACKEND_E2E_RUNTIME_LD_LIBRARY_PATH}\")\n")
    string(APPEND codegen_content "set(SWEEP_EXES\n")
    set(e2e_content "set(SWEEP_NAME \"${sweep_name}_e2e\")\n")
    string(APPEND e2e_content "set(SWEEP_REPORT_DIR \"${CMAKE_INSTALL_PREFIX}/report/v35/st\")\n")
    string(APPEND e2e_content "set(SWEEP_EXES\n")
    foreach(variant IN LISTS variants)
        list(APPEND codegen_deps ${variant}_codegen_v2)
        list(APPEND e2e_deps ${variant}_e2e_v2)
        string(APPEND codegen_content "  \"$<TARGET_FILE:${variant}_codegen_v2>\"\n")
        string(APPEND e2e_content "  \"$<TARGET_FILE:${variant}_e2e_v2>\"\n")
    endforeach()
    string(APPEND codegen_content ")\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/run_indirect_load_sweep.cmake\")\n")
    string(APPEND e2e_content ")\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/run_indirect_load_sweep.cmake\")\n")
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${sweep_name}_codegen_runner.cmake CONTENT "${codegen_content}")
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${sweep_name}_e2e_runner.cmake CONTENT "${e2e_content}")

    # Custom targets so `make <sweep>_codegen_v2/_e2e_v2` builds all variants.
    add_custom_target(${sweep_name}_codegen_v2 DEPENDS ${codegen_deps})
    add_custom_target(${sweep_name}_e2e_v2 DEPENDS ${e2e_deps})

    add_test(NAME ${sweep_name}_codegen_v2
             COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/${sweep_name}_codegen_runner.cmake)
    set_tests_properties(${sweep_name}_codegen_v2 PROPERTIES LABELS "st;build_backend_test1;${sweep_name}_codegen_v2")
    add_test(NAME ${sweep_name}_e2e_v2
             COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/${sweep_name}_e2e_runner.cmake)
    set_tests_properties(${sweep_name}_e2e_v2 PROPERTIES LABELS "st;build_backend_test2;${sweep_name}_e2e_v2")
endfunction()
function(mark_indirect_load_implementation test_name implementation)
    mark_indirect_load_codegen(${test_name}
        IL_SELECTED_IMPLEMENTATION=${implementation})
endfunction()

function(add_indirect_load_mixed_e2e_case test_name rank axis is_simt selected_template expected_templates static_shape x0 x1 x2 x3 index0 index1 index2 index3)
    if(is_simt)
        set(tiling_key 2)
    else()
        set(tiling_key 0)
    endif()
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            indirect_load_mixed_elementwise_test_kernel.cpp
            indirect_load_mixed_elementwise_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_CASE_STORE
        IL_CASE_MIXED
        IL_MIXED_ELEMENTWISE
        IL_EXPECT_SIMT=${is_simt}
        IL_RANK=${rank}
        IL_AXIS=${axis}
        IL_INPUT_PRE_TYPE=0
        IL_USE_EXP2=1
        IL_TILING_KEY=${tiling_key}
        IL_SELECTED_TEMPLATE=${selected_template}
        IL_EXPECTED_TEMPLATES=${expected_templates}
        IL_X_S0=${x0}
        IL_X_S1=${x1}
        IL_X_S2=${x2}
        IL_X_S3=${x3}
        IL_INDEX_S0=${index0}
        IL_INDEX_S1=${index1}
        IL_INDEX_S2=${index2}
        IL_INDEX_S3=${index3})
    if(static_shape)
        list(APPEND case_definitions IL_STATIC_SHAPE)
    endif()
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

function(mark_indirect_load_simt_elementwise_coverage test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_SIMT_ELEMENTWISE_COVERAGE)
endfunction()

add_indirect_load_e2e_case(indirect_load_rank2_axis1_simd 2 1 1 1 0 0 3 5 8 1 1 4 6 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_simd)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int64_gather 3 1 0 0 1 0 7 16 31 10 1 16 31 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int64_gather)
mark_indirect_load_implementation(indirect_load_rank3_axis1_float_int64_gather 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_float_int64_gather
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_torch_gather_frontend 3 1 0 0 3 2 7 54 4 8 1 54 4 8 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_torch_gather_frontend)
mark_indirect_load_sk(indirect_load_rank3_axis1_torch_gather_frontend)
mark_indirect_load_wide_types(indirect_load_rank3_axis1_torch_gather_frontend)
mark_indirect_load_output_post(indirect_load_rank3_axis1_torch_gather_frontend 2 1)
mark_indirect_load_input_outer_stride(indirect_load_rank3_axis1_torch_gather_frontend 64)


add_indirect_load_e2e_case(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd 4 1 3 0 0 0 1 8 13 3 8 8 17 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd)
mark_indirect_load_mixed_index_pre(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd)
mark_indirect_load_output_post(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd 11 0)
mark_indirect_load_codegen_and_e2e(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd
    IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_SIMD)
target_include_directories(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd_e2e_v2 PRIVATE
    ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)

add_indirect_load_mixed_e2e_case(indirect_load_mixed_rank4_axis2_simd 4 2 0 0 1 FALSE 3 4 16 7 2 3 17 5)

function(add_indirect_load_broadcast_test test_name template input_element_count index_element_count broadcast_axes_mask
         output_relu clear_broadcast_source_view)
    if(ARGC GREATER 7)
        set(input_broadcast ${ARGV7})
    else()
        set(input_broadcast 1)
    endif()
    if(ARGC GREATER 8)
        set(index_broadcast ${ARGV8})
    else()
        set(index_broadcast 1)
    endif()
    set(expect_simt 0)
    set(expect_sk 0)
    set(tiling_options)
    if(template STREQUAL "simt")
        set(expect_simt 1)
        set(tiling_options TILING_KEY 1)
    elseif(template STREQUAL "sk")
        set(expect_sk 1)
        set(tiling_options TILING_KEY 2)
    elseif(NOT template STREQUAL "simd")
        message(FATAL_ERROR "Unsupported IndirectLoad template: ${template}")
    endif()

    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        ${tiling_options}
        KERNEL_SRC
            indirect_load_broadcast_test_kernel.cpp
            indirect_load_broadcast_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_CASE_BROADCAST
        IL_INPUT_BROADCAST=${input_broadcast}
        IL_INDEX_BROADCAST=${index_broadcast}
        IL_HAS_INPUT_ELEMENT=${input_element_count}
        IL_HAS_INDEX_ELEMENT=${index_element_count}
        IL_HAS_OUTPUT_RELU=${output_relu}
        IL_BROADCAST_AXES_MASK=${broadcast_axes_mask}
        IL_CLEAR_BROADCAST_SOURCE_VIEW=${clear_broadcast_source_view}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk})
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

# Direct Broadcast covers IndirectLoad axis 2, its inner neighbor, and a degenerate source crossing the axis boundary.
add_indirect_load_broadcast_test(indirect_load_broadcast_cross_boundary_simt_fallback_test simt 0 0 14 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_cross_boundary_simt_fallback_test
                                   IL_DEGENERATE_BROADCAST=1 IL_OUTPUT_S0=10 IL_OUTPUT_S1=10 IL_OUTPUT_S2=20 IL_OUTPUT_S3=20)
add_indirect_load_broadcast_test(indirect_load_broadcast_axis_simd_test simd 1 1 4 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_inner_adjacent_simd_test simd 0 0 8 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_inner_adjacent_simd_test
                                   IL_INPUT_ABS_BEFORE_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_continuous_simd_test simd 0 0 12 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_continuous_simd_test IL_CONTINUOUS_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_continuous_index_simt_test simt 0 0 12 0 0 0 1)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_continuous_index_simt_test
                                   IL_CONTINUOUS_INDEX_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_cross_boundary_simt_test simt 0 0 10 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_axis_simt_test simt 1 1 4 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_reduce_simt_fallback_test simt 0 0 2 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_reduce_simt_fallback_test IL_BROADCAST_POST_REDUCE=1)

# A scalar Broadcast after an Element cannot use the physical-view inline path. SIMD keeps it as a regular op;
# SIMT emits it inside the fused scalar body.
add_indirect_load_broadcast_test(indirect_load_broadcast_retained_simd_test simd 0 0 10 0 0 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_retained_simd_test IL_RETAIN_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_retained_simt_test simt 0 0 10 0 0 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_retained_simt_test IL_RETAIN_BROADCAST=1)

# Retain the two-input Transpose baseline. The other existing targets below compose input Transpose,
# Broadcast, a plain index, and an output-side broadcast Load without increasing the target count.
add_indirect_load_broadcast_test(indirect_load_both_transpose_simd_test simd 0 0 0 0 0 0 0)
mark_indirect_load_static_shape(indirect_load_both_transpose_simd_test)
mark_indirect_load_codegen_and_e2e(indirect_load_both_transpose_simd_test
                                   IL_BOTH_TRANSPOSE=1 IL_OUTPUT_S0=2 IL_OUTPUT_S1=2 IL_OUTPUT_S2=1 IL_OUTPUT_S3=4)

# Exercise both SIMD tiling cases explicitly. Mixed cases transpose across the gather axis, broadcast
# the singleton suffix, and add a column-only GM bias; non-square shapes and non-symmetric data expose
# both permutation mistakes and output-linear offsets incorrectly used for the broadcast side input.
add_indirect_load_broadcast_test(indirect_load_both_transpose_simd_fallback_test simd 0 0 8 0 0 1 0)
mark_indirect_load_static_shape(indirect_load_both_transpose_simd_fallback_test)
mark_indirect_load_codegen_and_e2e(indirect_load_both_transpose_simd_fallback_test
                                   IL_INPUT_TRANSPOSE=1 IL_OUTPUT_BROADCAST_ADD=1 IL_FORCE_TILING_CASE=0
                                   IL_OUTPUT_S0=2 IL_OUTPUT_S1=3 IL_OUTPUT_S2=4 IL_OUTPUT_S3=32)
mark_indirect_load_red(indirect_load_both_transpose_simd_fallback_test)

add_indirect_load_broadcast_test(indirect_load_both_transpose_simd_nddma_test simd 0 0 8 0 0 1 0)
mark_indirect_load_static_shape(indirect_load_both_transpose_simd_nddma_test)
mark_indirect_load_codegen_and_e2e(indirect_load_both_transpose_simd_nddma_test
                                   IL_INPUT_TRANSPOSE=1 IL_OUTPUT_BROADCAST_ADD=1 IL_FORCE_TILING_CASE=1
                                   IL_OUTPUT_S0=2 IL_OUTPUT_S1=3 IL_OUTPUT_S2=4 IL_OUTPUT_S3=32)

add_indirect_load_broadcast_test(indirect_load_both_transpose_simt_test simt 0 0 8 0 0 1 0)
mark_indirect_load_static_shape(indirect_load_both_transpose_simt_test)
mark_indirect_load_codegen_and_e2e(indirect_load_both_transpose_simt_test
                                   IL_INPUT_TRANSPOSE=1 IL_OUTPUT_BROADCAST_ADD=1
                                   IL_OUTPUT_S0=2 IL_OUTPUT_S1=3 IL_OUTPUT_S2=4 IL_OUTPUT_S3=32)

# Keep one SK Broadcast regression outside the SIMD/SIMT matrix.
add_indirect_load_broadcast_test(indirect_load_broadcast_elements_sk_test sk 2 2 3 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_index_physical_view_simt_test simt 2 2 3 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_index_physical_view_simt_test IL_AIC_REPRO=1)

# Regression: a unary elementwise op after an index Broadcast must retain the source physical view in SIMT.
add_indirect_load_broadcast_test(indirect_load_broadcast_index_abs_simt_test simt 0 1 2 0 0 0 1)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_index_abs_simt_test IL_INDEX_ABS_DENSE_VIEW=1)

# Regression: a three-input Where chain must remain in the SIMT index region before IndirectLoad.
set(indirect_load_broadcast_index_where_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_broadcast_index_where_simt_test)
file(MAKE_DIRECTORY ${indirect_load_broadcast_index_where_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_broadcast_index_where_simt_test
    WORKDIR ${indirect_load_broadcast_index_where_simt_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_broadcast_where_test_kernel.cpp
        indirect_load_broadcast_where_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_index_where_simt_test IL_CASE_BROADCAST_WHERE)

# Regression: Where index branches use different physical views.  The row-broadcast branch
# must not provide the index offset for the dense branch.
set(indirect_load_broadcast_index_mixed_view_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_broadcast_index_mixed_view_simt_test)
file(MAKE_DIRECTORY ${indirect_load_broadcast_index_mixed_view_simt_test_workdir})

# Strict regression for the user-provided GraphHint: Where + Broadcast + IndirectLoad
# followed by ReduceSum, including the non-overlapping zero-stride/physical-gap table view.
set(indirect_load_graph_hint_reduce_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_graph_hint_reduce_simt_test)
file(MAKE_DIRECTORY ${indirect_load_graph_hint_reduce_simt_test_workdir})

# Exact 30x3x23 GraphHint reproduction for the SIMD IndirectLoad gather path.
set(indirect_load_graph_hint_simd_repro_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_graph_hint_simd_repro)
file(MAKE_DIRECTORY ${indirect_load_graph_hint_simd_repro_workdir})
do_backend_e2e_st_test(indirect_load_graph_hint_simd_repro
    WORKDIR ${indirect_load_graph_hint_simd_repro_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 0
    KERNEL_SRC
        indirect_load_graph_hint_simd_repro_kernel.cpp
        indirect_load_graph_hint_simd_repro_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_graph_hint_simd_repro
                                   IL_CASE_BROADCAST_WHERE IL_GRAPH_HINT_SIMD_REPRO)

# Exact reproduction of the user-provided embedding slice GraphHint.
add_indirect_load_graph_case(indirect_load_graph_hint_embedding_slice
    indirect_load_graph_hint_embedding_slice 0
    IL_CASE_BROADCAST_WHERE IL_GRAPH_HINT_EMBEDDING_SLICE)

# Minimal SIMT reproducer: a partially zero-stride mask view is enough to trigger the bad evaluator offset.
add_indirect_load_graph_case(indirect_load_user_masked_embedding_minimal
    indirect_load_user_masked_embedding_minimal 1
    IL_CASE_BROADCAST_WHERE IL_USER_MASKED_EMBEDDING_MINIMAL)

# Full three-input reproduction of the user-provided embedding + ge + mul + sum GraphHint.

# Run the exact user graph through the normal optimizer/codegen flow without
# forcing a template.  This guards the Embedding-like candidate selection.
add_indirect_load_graph_case(indirect_load_user_masked_embedding_sum_full_auto
    indirect_load_user_masked_embedding_sum_full 1
    IL_CASE_BROADCAST_WHERE IL_USER_MASKED_EMBEDDING_SUM_FULL IL_USER_MASKED_EMBEDDING_AUTO_SELECT)

# Exact user position-bias GraphHint: relative-position bucket arithmetic,
# transposed [8,32,1] table view, axis-1 IndirectLoad, side-input add and no reduction.
# [disabled] 分支行为差异待修：IL+bias+Exp+Sum在分支上未融合Reduce（缺ReduceSum），develop正常
# add_indirect_load_graph_case(indirect_load_user_position_bias_exp_sum
#     indirect_load_user_position_bias_exp_sum 1
#     IL_CASE_BROADCAST_WHERE IL_USER_POSITION_BIAS_EXP_SUM)

add_indirect_load_graph_case(indirect_load_user_position_bias
    indirect_load_user_position_bias 1
    IL_CASE_BROADCAST_WHERE IL_USER_POSITION_BIAS)

# User graph: position-bias buckets, axis-1 IndirectLoad over the transposed [8,32,1]
# table, two broadcast side inputs, Exp and the final Sum over the lookup axis.

# User graph: embedding + Sum over the lookup axis.
add_indirect_load_graph_case(indirect_load_user_embedding_sum indirect_load_user_embedding_sum 1
    IL_CASE_BROADCAST_WHERE IL_USER_EMBEDDING_SUM)

# IndirectLoad sweep：把值 dtype / 强制模板 / 动态 shape 三个关注点各自聚合成 1 个 ST 用例，
# 避免 dtype × 场景 × 模板 的笛卡尔积；每个 variant 仍是独立的 codegen/e2e 二进制。

# 值 dtype 轴：embedding 场景遍历 int64/uint64（8 字节，SIMD+SIMT 选择）与
# int8/uint8/bool（1 字节，仅 SIMT），SK 均排除。
add_indirect_load_sweep_variant(indirect_load_user_dtype_sweep indirect_load_user_dtype_sweep_int64 0
    IL_CASE_BROADCAST_WHERE IL_USER_DTYPE_INT64)
add_indirect_load_sweep_variant(indirect_load_user_dtype_sweep indirect_load_user_dtype_sweep_uint64 0
    IL_CASE_BROADCAST_WHERE IL_USER_DTYPE_UINT64)
add_indirect_load_sweep_variant(indirect_load_user_dtype_sweep indirect_load_user_dtype_sweep_int8 0
    IL_CASE_BROADCAST_WHERE IL_USER_DTYPE_INT8)
add_indirect_load_sweep_variant(indirect_load_user_dtype_sweep indirect_load_user_dtype_sweep_uint8 0
    IL_CASE_BROADCAST_WHERE IL_USER_DTYPE_UINT8)
add_indirect_load_sweep_variant(indirect_load_user_dtype_sweep indirect_load_user_dtype_sweep_bool 0
    IL_CASE_BROADCAST_WHERE IL_USER_DTYPE_BOOL)
finalize_indirect_load_sweep(indirect_load_user_dtype_sweep)

# 强制模板轴：dense int64 gather 遍历 auto / SIMD(kRegisterGather,kGatherApi) / SIMT。
add_indirect_load_sweep_variant(indirect_load_user_int64_dense_sweep indirect_load_user_int64_dense_sweep_auto 0
    IL_CASE_BROADCAST_WHERE IL_USER_INT64_GATHER_DENSE)
add_indirect_load_sweep_variant(indirect_load_user_int64_dense_sweep indirect_load_user_int64_dense_sweep_simd 0
    IL_CASE_BROADCAST_WHERE IL_USER_INT64_GATHER_DENSE IL_USER_INT64_GATHER_DENSE_SIMD)
add_indirect_load_sweep_variant(indirect_load_user_int64_dense_sweep indirect_load_user_int64_dense_sweep_simt 0
    IL_CASE_BROADCAST_WHERE IL_USER_INT64_GATHER_DENSE IL_USER_INT64_GATHER_DENSE_SIMT)
finalize_indirect_load_sweep(indirect_load_user_int64_dense_sweep)

# 动态 shape 轴：两个用户 GraphHint（gather_sum_transpose / abs_embedding_sum）。
add_indirect_load_sweep_variant(indirect_load_user_dynamic_shape_sweep
    indirect_load_user_dynamic_shape_sweep_gather_sum_transpose 0
    IL_CASE_BROADCAST_WHERE IL_USER_GATHER_SUM_TRANSPOSE)
add_indirect_load_sweep_variant(indirect_load_user_dynamic_shape_sweep
    indirect_load_user_dynamic_shape_sweep_abs_embedding_sum 0
    IL_CASE_BROADCAST_WHERE IL_USER_ABS_EMBEDDING_SUM)
finalize_indirect_load_sweep(indirect_load_user_dynamic_shape_sweep)

# Rank-2 embedding view followed by Reduce on the payload dimension.  Keep the
# original rank-3 lookup-reduction case above unchanged as a regression guard.
add_indirect_load_graph_case(indirect_load_user_embedding_sum_rank2
    indirect_load_user_embedding_sum_rank2 1
    IL_CASE_BROADCAST_WHERE IL_USER_EMBEDDING_SUM IL_USER_EMBEDDING_SUM_RANK2)

# Same GraphHint and runtime buffers as the user embedding+Sum graph, but force the SIMD candidate
# to exercise the vector-gather path used by the board kernel in the simulator.
add_indirect_load_graph_case(indirect_load_user_embedding_sum_simd indirect_load_user_embedding_sum_simd 0
    IL_CASE_BROADCAST_WHERE IL_USER_EMBEDDING_SUM IL_USER_EMBEDDING_SUM_SIMD)

# User graph: embedding followed by scalar Mul.

# User graph: bf16 embedding with mean/rsqrt normalization and two outputs.
add_indirect_load_graph_case(indirect_load_user_layernorm indirect_load_user_layernorm 1
    IL_CASE_BROADCAST_WHERE IL_USER_LAYERNORM)

# User graph: gather(+bias+bmm) followed by a tail-axis softmax pattern. The fused
# candidate must hit the Softmax dedicated path (SoftmaxAR) instead of Max/Sum composites.
# Gated off by default: the fused codegen chain still trips IsTailBroadcastNode
# (Broadcast in/out vectorized-axis count mismatch) on this branch; enable once the
# gather+softmax dedicated path is completed.
option(ENABLE_IL_USER_SOFTMAX "Build the gather+softmax dedicated-path regression case" OFF)
if(ENABLE_IL_USER_SOFTMAX)
    add_indirect_load_graph_case(indirect_load_user_softmax indirect_load_user_softmax 1
        IL_CASE_BROADCAST_WHERE IL_USER_SOFTMAX)
endif()

# Same LayerNorm split topology with a small shape; force the SIMD candidate to
# determine whether the branch itself, rather than UB pressure, is the blocker.

# User graph: one embedding result fans out to Exp and Abs, then merges by Add.
# Run both IL SIMD and IL SIMT templates to compare branch handling.
add_indirect_load_graph_case(indirect_load_user_embedding_exp_abs_add_simt
    indirect_load_user_embedding_exp_abs_add_simt 1
    IL_USER_EMBEDDING_EXP_ABS_ADD)

# Fan-out matrix: split at IndirectLoad or at a post-IndirectLoad elementwise node,
# with either two ordinary Stores or one ordinary Store plus one Reduce Store.
function(add_indirect_load_user_fanout_case test_name)
    add_indirect_load_graph_case(${test_name} ${test_name} 1 IL_USER_FANOUT ${ARGN})
endfunction()

add_indirect_load_user_fanout_case(indirect_load_user_fanout_direct_reduce_simt
                                   IL_USER_FANOUT_REDUCE)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_post_reduce_simt
                                   IL_USER_FANOUT_POST IL_USER_FANOUT_REDUCE)

add_indirect_load_user_fanout_case(indirect_load_user_fanout_direct_stores_simd
                                   IL_USER_FANOUT_SIMD)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_direct_stores_simt)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_post_stores_simd
                                   IL_USER_FANOUT_SIMD IL_USER_FANOUT_POST)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_post_stores_simt
                                   IL_USER_FANOUT_POST)

# Regression: two GM output chains, with one chain reading a broadcast GM side input.
add_indirect_load_graph_case(indirect_load_user_fanout_side_input_simt
    indirect_load_user_fanout_side_input 1 IL_USER_FANOUT_SIDE_INPUT)

# Minimal red regression: a post-IndirectLoad fanout Add consumes an independent Load side input.
add_indirect_load_graph_case(indirect_load_user_side_input_fanout_simd
    indirect_load_user_side_input_fanout_simd 0
    IL_USER_SIDE_INPUT_FANOUT IL_USER_SIDE_INPUT_FANOUT_SIMD)
mark_indirect_load_red(indirect_load_user_side_input_fanout_simd)

add_indirect_load_graph_case(indirect_load_user_side_input_fanout_simt
    indirect_load_user_side_input_fanout_simt 1
    IL_USER_SIDE_INPUT_FANOUT)
mark_indirect_load_red(indirect_load_user_side_input_fanout_simt)

# User graph: two IndirectLoads over two tensors followed by Add (dual-IL graph).
# Keep the case sources for future capability work, but do not register the unsupported
# topology in the default build/test collection.
option(ENABLE_IL_DUAL_IL "Build the dual-IndirectLoad regression case" OFF)
if(ENABLE_IL_DUAL_IL)
    add_indirect_load_graph_case(indirect_load_user_add_gather indirect_load_user_add_gather 1
        IL_CASE_BROADCAST_WHERE IL_DUAL_IL_GATHER)
endif()

# Regression: SIMT post-reduce Output() evaluator uses full output_index instead of address.index_offset
# to access index-tensor GM, causing OOB reads when output_index exceeds index tensor size.
set(indirect_load_embedding_reduce_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_embedding_reduce_simt_test)
file(MAKE_DIRECTORY ${indirect_load_embedding_reduce_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_embedding_reduce_simt_test
    WORKDIR ${indirect_load_embedding_reduce_simt_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_embedding_reduce_simt_test_kernel.cpp
        indirect_load_embedding_reduce_simt_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_embedding_reduce_simt_test IL_CASE_BROADCAST_WHERE IL_EMBEDDING_REDUCE)

# Regression: a dual-input Add before IndirectLoad with a post-fused ReduceSum must keep a valid SIMT kernel.
set(indirect_load_add_il_reduce_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_add_il_reduce_test)
file(MAKE_DIRECTORY ${indirect_load_add_il_reduce_test_workdir})

# Same-view tensor fan-in without Broadcast: the binary operation is coordinate-preserving.
add_indirect_load_broadcast_test(indirect_load_index_binary_same_view_simd_test simd 0 1 0 0 0 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_index_binary_same_view_simd_test
                                   IL_INDEX_BINARY_SAME_VIEW=1 IL_BINARY_ELEMENT_KIND=3)

add_indirect_load_broadcast_test(indirect_load_complex_broadcast_simd_test simd 0 0 3 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_complex_broadcast_simd_test
                                   IL_COMPLEX_BROADCAST=1 IL_BINARY_ELEMENT_KIND=3)

add_indirect_load_broadcast_test(indirect_load_complex_broadcast_simt_test simt 0 0 3 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_complex_broadcast_simt_test IL_COMPLEX_BROADCAST=1 IL_COMPLEX_SIMT=1)

# Input Broadcast whose source is a multi-input binary element chain: the SIMD candidate keeps the direct Broadcast and
# candidate consumes the computed source through its physical view (source multi-input scenario).
# Note: SIMT rejects multi-input input sources by design (ValidateSimtTemplateRegion requires a Load
# boundary), covered by UT SimdRegionMetadataAndSimtRejectsMultiInputRegion.
add_indirect_load_broadcast_test(indirect_load_complex_input_broadcast_simd_test simd 0 0 12 0 0 1 0)
mark_indirect_load_codegen_and_e2e(indirect_load_complex_input_broadcast_simd_test
                                   IL_COMPLEX_INPUT_BROADCAST=1 IL_BINARY_ELEMENT_KIND=3)

function(add_indirect_load_stride_zero_test test_name template input_zero_stride_mask index_zero_stride_mask
         input_element_count index_element_count)
    set(expect_simt 0)
    set(expect_sk 0)
    set(tiling_options)
    if(template STREQUAL "simt")
        set(expect_simt 1)
        set(tiling_options TILING_KEY 1)
    elseif(template STREQUAL "sk")
        set(expect_sk 1)
        set(tiling_options TILING_KEY 2)
    elseif(NOT template STREQUAL "simd")
        message(FATAL_ERROR "Unsupported IndirectLoad template: ${template}")
    endif()

    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        ${tiling_options}
        KERNEL_SRC
            indirect_load_stride_zero_test_kernel.cpp
            indirect_load_stride_zero_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_CASE_STRIDE_ZERO
        IL_INPUT_ZERO_STRIDE_MASK=${input_zero_stride_mask}
        IL_INDEX_ZERO_STRIDE_MASK=${index_zero_stride_mask}
        IL_HAS_INPUT_ELEMENT=${input_element_count}
        IL_HAS_INDEX_ELEMENT=${index_element_count}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk})
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

# Mask bit d denotes that logical axis d has stride 0. The Element case covers both input and index zero-stride paths.
# Direct zero-stride classification is covered by the layout and schedule UT.

function(add_indirect_load_torch_gather_strided_test test_name tiling_key expect_simt expect_sk
         input_stride0 input_stride1 input_stride2 index_stride0 index_stride1 index_stride2)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            indirect_load_torch_gather_strided_test_kernel.cpp
            indirect_load_torch_gather_strided_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_CASE_TORCH_STRIDED
        IL_EXPECT_TILING_KEY=${tiling_key}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk}
        IL_INPUT_STRIDE0=${input_stride0}
        IL_INPUT_STRIDE1=${input_stride1}
        IL_INPUT_STRIDE2=${input_stride2}
        IL_INDEX_STRIDE0=${index_stride0}
        IL_INDEX_STRIDE1=${index_stride1}
        IL_INDEX_STRIDE2=${index_stride2})
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

# Cover non-contiguous layouts without a gap-kind/template Cartesian product: the inner-gap case also has an
# outer gap, while the SIMT outer-gap case covers the pure outer-gap policy. Keep one SK runtime representative.
add_indirect_load_torch_gather_strided_test(indirect_load_rank3_axis1_input_index_gap_simd_test
    0 0 0 384 10 1 192 10 1)
add_indirect_load_torch_gather_strided_test(indirect_load_rank3_axis1_input_index_gap_simt_test
    0 1 0 384 10 1 192 10 1)
add_indirect_load_torch_gather_strided_test(indirect_load_rank3_axis1_input_index_outer_gap_simt_test
    0 1 0 192 5 1 128 5 1)
set(indirect_load_index_select_defs IL_INDEX_SELECT_CASE=1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_input_index_gap_simt_test
                                   ${indirect_load_index_select_defs})


set(indirect_load_embedding_test_workdir ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_embedding_test)
file(MAKE_DIRECTORY ${indirect_load_embedding_test_workdir})
do_backend_e2e_st_test(indirect_load_embedding_test
    WORKDIR ${indirect_load_embedding_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 0
    KERNEL_SRC
        indirect_load_embedding_test_kernel.cpp
        indirect_load_embedding_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_embedding_test IL_CASE_EMBEDDING IL_EMBEDDING_SIZE=13)

# Non-block-aligned payload regression: exercise the SIMD Embedding head/body/tail split.
# The SIMT Embedding policy's non-power-of-two division path is covered by the auto-selected
# IL_CASE_EMBEDDING case above, so only the SIMD-forced variant needs its own registration.
add_indirect_load_graph_case(indirect_load_embedding_tail_simd
    indirect_load_embedding_tail_simd 0
    IL_CASE_EMBEDDING IL_EMBEDDING_SIZE=13 IL_EMBEDDING_DIRECT=1)

set(indirect_load_broadcast_index_mixed_view_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_broadcast_index_mixed_view_simt_test)
file(MAKE_DIRECTORY ${indirect_load_broadcast_index_mixed_view_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_broadcast_index_mixed_view_simt_test
    WORKDIR ${indirect_load_broadcast_index_mixed_view_simt_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_broadcast_where_test_kernel.cpp
        indirect_load_broadcast_where_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_index_mixed_view_simt_test
                                   IL_CASE_BROADCAST_WHERE IL_INDEX_MIXED_VIEW)

set(indirect_load_graph_hint_reduce_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_graph_hint_reduce_simt_test)
file(MAKE_DIRECTORY ${indirect_load_graph_hint_reduce_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_graph_hint_reduce_simt_test
    WORKDIR ${indirect_load_graph_hint_reduce_simt_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_graph_hint_reduce_simt_test_kernel.cpp
        indirect_load_graph_hint_reduce_simt_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_graph_hint_reduce_simt_test
                                   IL_CASE_BROADCAST_WHERE IL_GRAPH_HINT_REDUCE)

set(indirect_load_add_il_reduce_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_add_il_reduce_test)
file(MAKE_DIRECTORY ${indirect_load_add_il_reduce_test_workdir})
do_backend_e2e_st_test(indirect_load_add_il_reduce_test
    WORKDIR ${indirect_load_add_il_reduce_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_add_il_reduce_test_kernel.cpp
        indirect_load_add_il_reduce_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_add_il_reduce_test IL_CASE_BROADCAST_WHERE IL_ADD_IL_REDUCE)
