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

include_directories(${CODE_ROOT_DIR}/tests/framework/share_graph/include)
include_directories(${CODE_ROOT_DIR}/tests/st/backend_e2e)
include(backend_e2e.cmake)

set(INDUCTOR_HOST_HELPER_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(INDUCTOR_ASCEND_LIBRARY_DIRS "")
if(ASCEND_INSTALL_PATH)
    set(INDUCTOR_ASCEND_LIBRARY_DIRS
        ${ASCEND_INSTALL_PATH}/lib64
        ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
        ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib
        ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/${CMAKE_SYSTEM_PROCESSOR}
        ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}
        ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib/device
        ${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/device/lib64
        ${ASCEND_INSTALL_PATH}/devlib
        ${ASCEND_INSTALL_PATH}/devlib/${CMAKE_SYSTEM_PROCESSOR}
        ${ASCEND_INSTALL_PATH}/devlib/linux/${CMAKE_SYSTEM_PROCESSOR}
        ${ASCEND_INSTALL_PATH}/runtime/lib64/stub
        ${ASCEND_INSTALL_PATH}/runtime/lib64/stub/linux/${CMAKE_SYSTEM_PROCESSOR}
        ${ASCEND_INSTALL_PATH}/toolkit/devlib
    )
endif()
set(INDUCTOR_ASCEND_VALID_LIBRARY_DIRS "")
foreach(dir IN LISTS INDUCTOR_ASCEND_LIBRARY_DIRS)
    if(IS_DIRECTORY "${dir}")
        list(APPEND INDUCTOR_ASCEND_VALID_LIBRARY_DIRS "${dir}")
    endif()
endforeach()
list(REMOVE_DUPLICATES INDUCTOR_ASCEND_VALID_LIBRARY_DIRS)
set(INDUCTOR_ASCEND_LIBRARY_PATH "")
if(INDUCTOR_ASCEND_VALID_LIBRARY_DIRS)
    list(JOIN INDUCTOR_ASCEND_VALID_LIBRARY_DIRS ":" INDUCTOR_ASCEND_LIBRARY_PATH)
endif()

add_library(inductor_host_helper_common STATIC ${INDUCTOR_HOST_HELPER_COMMON_DIR}/inductor_host_helper_common.cpp)
target_include_directories(inductor_host_helper_common PUBLIC
    ${INDUCTOR_HOST_HELPER_COMMON_DIR}
)
target_link_libraries(inductor_host_helper_common PUBLIC
    dl
    json
)
target_compile_options(inductor_host_helper_common PUBLIC
    -U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1
)

function(add_inductor_ascend_runtime_path target)
    if(INDUCTOR_ASCEND_LIBRARY_PATH)
        set_target_properties(${target} PROPERTIES BUILD_RPATH "${INDUCTOR_ASCEND_LIBRARY_PATH}")
        target_link_options(${target} PRIVATE -Wl,--disable-new-dtags)
    endif()
endfunction()

function(add_inductor_host_abi1_helper helper_target codegen_outputs workdir)
    set(options)
    set(one_value_args DYNAMIC_ARG_COUNT)
    set(multi_value_args EXTRA_INCLUDE_DIRS)
    cmake_parse_arguments(HOST_HELPER "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
    if(HOST_HELPER_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unexpected add_inductor_host_abi1_helper args: ${HOST_HELPER_UNPARSED_ARGUMENTS}")
    endif()

    set(dynamic_shape_arg_count 0)
    if(DEFINED HOST_HELPER_DYNAMIC_ARG_COUNT AND NOT "${HOST_HELPER_DYNAMIC_ARG_COUNT}" STREQUAL "")
        set(dynamic_shape_arg_count ${HOST_HELPER_DYNAMIC_ARG_COUNT})
    endif()
    add_executable(${helper_target} ${INDUCTOR_HOST_HELPER_COMMON_DIR}/inductor_host_abi1_helper.cpp)
    add_dependencies(${helper_target} ${codegen_outputs})
    target_include_directories(${helper_target} PRIVATE ${workdir} ${HOST_HELPER_EXTRA_INCLUDE_DIRS})
    target_compile_definitions(${helper_target} PRIVATE
        INDUCTOR_HOST_HELPER_DYNAMIC_ARG_COUNT=${dynamic_shape_arg_count}
    )
    target_link_libraries(${helper_target} PRIVATE inductor_host_helper_common)
    add_inductor_ascend_runtime_path(${helper_target})
endfunction()

function(add_inductor_split_compile_test_runtime test_target)
    add_inductor_ascend_runtime_path(${test_target})
endfunction()

add_subdirectory(add_abs_test)
add_subdirectory(sub_abs_test)
add_subdirectory(scalar_float_inf_test)
add_subdirectory(scalar_div_inf_test)
add_subdirectory(add_gelu_test)
add_subdirectory(compare_test)
add_subdirectory(compare_x2_tensor_test)
add_subdirectory(compare_x2_tensor_int32_test)
add_subdirectory(compare_x2_tensor_int64_eq_test)
add_subdirectory(compare_x2_tensor_int64_gt_test)
add_subdirectory(add_neg_test)
add_subdirectory(load_to_store_and_abs_test)
add_subdirectory(load_unalign_pad_test)
add_subdirectory(scalar_cast_add_test)
# add_subdirectory(concat_test)
add_subdirectory(concat_all_aligned_test)
add_subdirectory(concat_to_stores_test)
add_subdirectory(brc_inline_test)
add_subdirectory(load_where_store_test)
add_subdirectory(isinf_maskedfill_test)
add_subdirectory(load_where_x2_x3_is_ubscalar_store_test)
add_subdirectory(load_where_x2_is_ubscalar_store_test)
add_subdirectory(load_where_x3_is_ubscalar_store_test)
add_subdirectory(load_logical_not_store_test)
add_subdirectory(truediv_abs_test)
add_subdirectory(add_rsqrt_test)
# add_subdirectory(sub_transpose_abs_test)
add_subdirectory(axpy_abs_test)
add_subdirectory(axpy_abs_half_test)
add_subdirectory(load_pow_all_input_is_scalar_store_test)
add_subdirectory(pgo_add_abs_test)
# add_subdirectory(pgo_sub_transpose_abs_test)
add_subdirectory(matmul_elemwise_test)
add_subdirectory(conv2d_elemwise_test)
# add_subdirectory(argmax_test)
add_subdirectory(inductor_topn_test)
add_subdirectory(inductor_matmul_elemwise_test)
add_subdirectory(inductor_tail_brc_tail_reduce_test)
add_subdirectory(inductor_topn_concat_test)
add_subdirectory(pgo_add_abs_inductor_test)
add_subdirectory(pgo_add_abs_inductor_concat_test)
add_subdirectory(workspace_uint64_multi_reduce_test)
