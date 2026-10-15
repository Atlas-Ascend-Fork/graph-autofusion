# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
add_library(intf_llt_options INTERFACE)

target_compile_definitions(intf_llt_options INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(intf_llt_options INTERFACE
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage -fprofile-update=atomic>
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan -fsanitize=leak -static-libtsan>
)

target_link_options(intf_llt_options INTERFACE
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
    $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -static-libasan -fsanitize=undefined  -static-libubsan -fsanitize=leak -static-libtsan>
)

target_link_libraries(intf_llt_options INTERFACE
    -lpthread
    $<$<BOOL:${ENABLE_GCOV}>:-lgcov>
)

if(ENABLE_CPP_UTEST)
    add_library(intf_llt_ut INTERFACE)
    target_compile_definitions(intf_llt_ut INTERFACE CFG_BUILD_DEBUG)
    target_compile_options(intf_llt_ut INTERFACE -g -w -fPIC -pipe -Werror -Wno-error=deprecated-declarations)
    target_link_libraries(intf_llt_ut INTERFACE intf_llt_options GTest::gtest mockcpp)
endif()
