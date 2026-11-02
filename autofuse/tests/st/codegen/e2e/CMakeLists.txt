set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ASCEND_ROOT}/tools/tikicpulib/lib)
find_package(tikicpulib QUIET)

# 要修复的目标列表
set(TIKICPULIB_TARGETS
        tikicpulib_ascend910B1
        tikicpulib_ascend950pr_9599
        # 添加其他可能出错的 target
)

foreach(tgt IN LISTS TIKICPULIB_TARGETS)
    if(TARGET ${tgt})
        # 修复 INTERFACE_INCLUDE_DIRECTORIES
        get_target_property(include_dirs ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(NOT include_dirs)
            set(include_dirs "")
        endif()

        set(valid_includes "")
        foreach(dir IN LISTS include_dirs)
            # 跳过明显错误的嵌套路径（可选）
            if(dir MATCHES "tools/tikicpulib/tools/tikicpulib")
                message(VERBOSE "Skipping malformed include dir: ${dir}")
                continue()
            endif()

            # 只保留真实存在的目录
            if(IS_DIRECTORY "${dir}")
                list(APPEND valid_includes "${dir}")
            else()
                message(VERBOSE "Skipping non-existent include dir: ${dir}")
            endif()
        endforeach()

        set_target_properties(${tgt}
                PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${valid_includes}"
        )

        # 同样修复 INTERFACE_LINK_DIRECTORIES（如有需要）
        get_target_property(link_dirs ${tgt} INTERFACE_LINK_DIRECTORIES)
        if(link_dirs)
            set(valid_links "")
            foreach(dir IN LISTS link_dirs)
                if(IS_DIRECTORY "${dir}")
                    list(APPEND valid_links "${dir}")
                endif()
            endforeach()
            set_target_properties(${tgt}
                    PROPERTIES INTERFACE_LINK_DIRECTORIES "${valid_links}"
            )
        endif()
    endif()
endforeach()
include(e2e.cmake)

add_library(e2e_com SHARED
    ./e2e_load_abs_store.cpp
    ./e2e_load_cast_store.cpp
    ./e2e_load_leakyrelu_store.cpp
    ./e2e_load_transpose_store.cpp
    ./e2e_load_ub_scalar_add_store.cpp
    ./e2e_load_ub_scalar_div_store.cpp
    ./e2e_load_sub_store.cpp
    ./e2e_discrete_load.cpp
    # ./e2e_softmax.cpp
    ./e2e_broadcast.cpp
    ./e2e_rmax.cpp
    ./e2e_rargmax.cpp
    ./e2e_rmean.cpp
    ./e2e_constant_load_gt_store.cpp
    ./e2e_multi_vectorize_axis.cpp
    ./e2e_rmin.cpp
    ./e2e_load_concat_store.cpp
    ./e2e_rprod.cpp
    ./e2e_broadcast_force_merge.cpp
    ./e2e_constant_load_le_store.cpp
    # ./e2e_load_zeros_like_store.cpp
    ./e2e_load_compare_ub_scalar_eq_store.cpp
    ./e2e_load_compare_tensor_eq_store.cpp
    ./e2e_load_compare_tensor_ne_store.cpp
    ./e2e_load_compare_tensor_gt_store.cpp
    ./e2e_clip_by_value_float.cpp
    ./e2e_pow_int32.cpp
    ./e2e_load_removepad_store.cpp
    ./e2e_load_ub2ub_abs_store.cpp
    ./e2e_load_isfinite_store.cpp
    ./e2e_load_max_min_store.cpp
    ./e2e_rsum.cpp
    ./e2e_load_reciprocal_store.cpp
    ./e2e_load_bitwiseand_store.cpp
    ./e2e_load_logicalor_ub_scalar_store.cpp
    ./e2e_load_where_store.cpp
    ./e2e_load_where_x2x3_is_ubscalar_store.cpp
    ./e2e_load_strided_slice_store.cpp
    ./e2e_gather_abs_store.cpp
    ./e2e_load_erf_store.cpp
    ./e2e_load_scalar_sub_store.cpp
    ./e2e_load_gather_abs_store.cpp
    ./e2e_load_logicalnot_store.cpp
    )
target_link_libraries(e2e_com PUBLIC
    -Wl,--no-as-needed
    ascir
    optimize
    codegen
    aihac_ir
    aihac_ir_register
    graph_af
    )
target_include_directories(e2e_com PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

# add_subdirectory(tail_process_z0_split_t_b)
# add_subdirectory(load_cast_store_expect_code)
add_subdirectory(load_leakyrelu_store_expect_code)
add_subdirectory(load_transpose_store_expect_code)
add_subdirectory(load_ub2ub_abs_store_expect_code)
add_subdirectory(load_ub_scalar_add_store_expect_code)
add_subdirectory(load_ub_scalar_gt_store_expect_code)
# add_subdirectory(load_ub_scalar_div_store_expect_code)
# add_subdirectory(load_rprod_store)
# add_subdirectory(clip_by_value_float)
# add_subdirectory(pow_int32)
add_subdirectory(load_isfinite_store)
# add_subdirectory(load_compare_half_tensor_eq_store)
# add_subdirectory(load_compare_ub_scalar_tensor_eq_store)
# add_subdirectory(load_compare_int64_tensor_eq_store)
# add_subdirectory(load_compare_int64_tensor_ne_store)
# add_subdirectory(load_compare_int64_tensor_gt_store)
add_subdirectory(discrete_store)
add_subdirectory(discrete_load)
# add_subdirectory(discrete_store_merge_axis)
add_subdirectory(store_scalar)
add_subdirectory(broadcast_force_merge)
# add_subdirectory(load_rmax_store)
# add_subdirectory(load_rargmax_store)
# add_subdirectory(load_rmean_store)
# add_subdirectory(load_removepad_store)
# add_subdirectory(load_rmin_store)
# add_subdirectory(load_broadcast_axis_s1_store_expect_code)
# add_subdirectory(load_multi_axis_store)
# add_subdirectory(load_zeros_like_store_expect_code)
add_subdirectory(load_abs_store_expect_code)
add_subdirectory(load_sub_store_expect_code)
add_subdirectory(broadcast_merge_axis)
# add_subdirectory(concat_last_dim)
add_subdirectory(concat_same_tail_dim)
add_subdirectory(concat_small_tail_dim)
add_subdirectory(concat_inter_dim)
add_subdirectory(concat_3d_last_dim)
# add_subdirectory(concat_mult_inputs)
add_subdirectory(dynamic_inputs_and_outputs)
add_subdirectory(load_max_min_store)
# add_subdirectory(load_rsum_store)
add_subdirectory(load_rsum_block_store)
add_subdirectory(constant_load_gt_store_expect_code)
add_subdirectory(constant_load_le_store_expect_code)
add_subdirectory(load_reciprocal_store)
add_subdirectory(load_bitwiseand_store)
# add_subdirectory(load_logicalor_ub_scalar_store)
add_subdirectory(load_where_store_expect_code)
add_subdirectory(load_where_x2x3_is_ubscalar_store_expect_code)
add_subdirectory(load_where_x2x3_is_ubscalar_throwfor_store_expect_code)
add_subdirectory(load_strided_slice_store)
# add_subdirectory(load_rsum_ra_store)
# add_subdirectory(gather_abs_one_axis)
# add_subdirectory(schedule_multi_group)
add_subdirectory(load_store_expect_code)
add_subdirectory(schedule_multi_group_ws_reuse_output)
add_subdirectory(broadcast_multi_axes)
add_subdirectory(load_erf_store)
add_subdirectory(load_scalar_sub_store)
add_subdirectory(load_gather_split_bt_t_abs_store)
add_subdirectory(load_gather_split_t_bt_abs_store)
add_subdirectory(load_gather_split_b_t_abs_store)
add_subdirectory(load_gather_split_bt_abs_store)
add_subdirectory(store_empty_tensor)
add_subdirectory(load_rsum_invalid_axis_store)
add_subdirectory(load_gather_first_axis_split_b_t_abs_store)
add_subdirectory(load_logicalnot_store)
