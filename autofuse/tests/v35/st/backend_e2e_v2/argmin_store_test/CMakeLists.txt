# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

function(add_argmin_backend_case name)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    backend_e2e_st_test(${name}
        CODEGEN argmin_backend_generator.cpp
        KERNEL_SRC
            ${name}/argmin_kernel.cpp
            ${name}/argmin_tiling.cpp
            ${name}/autofuse_tiling_data.h
        TEST_SRC test_e2e_argmin_kernel.cpp)
    target_include_directories(${name}_e2e_v2 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    foreach(definition IN LISTS ARGN)
        target_compile_definitions(${name}_codegen_v2 PRIVATE ${definition})
        target_compile_definitions(${name}_e2e_v2 PRIVATE ${definition})
    endforeach()
endfunction()

add_argmin_backend_case(argmin_store_test)
add_argmin_backend_case(argmin_rcore_i32_test ARGMIN_RCORE ARGMIN_I32)
add_argmin_backend_case(argmin_allload_test ARGMIN_ALLLOAD)
add_argmin_backend_case(argmin_value_test ARGMIN_WITH_VALUE)
add_argmin_backend_case(argmin_value_i32_test ARGMIN_WITH_VALUE ARGMIN_I32)
add_argmin_backend_case(argmin_value_rcore_i64_test ARGMIN_WITH_VALUE ARGMIN_RCORE ARGMIN_I64)
add_argmin_backend_case(argmin_value_ra_i64_test ARGMIN_WITH_VALUE ARGMIN_RA ARGMIN_I64)
add_argmin_backend_case(argmin_value_ra_rcore_test ARGMIN_WITH_VALUE ARGMIN_RA ARGMIN_RCORE)
