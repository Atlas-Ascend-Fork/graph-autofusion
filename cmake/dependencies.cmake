# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH $ENV{ASCEND_HOME_PATH}/)

set(CMAKE_MODULE_PATH
  ${CANN_CMAKE_DIR}/modules
  ${CMAKE_MODULE_PATH}
)
message(STATUS "CMAKE_MODULE_PATH:${CMAKE_MODULE_PATH}")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type(default Release)"
    FORCE)
endif ()

find_cann_package(runtime MODULE REQUIRED)
find_cann_package(unified_dlog MODULE REQUIRED)
find_cann_package(securec MODULE REQUIRED)
find_cann_package(acl_rt MODULE REQUIRED)
find_cann_package(acl_rtc MODULE REQUIRED)
find_cann_package(ascendcl MODULE REQUIRED)
find_cann_package(ascend_dump MODULE REQUIRED)
find_cann_package(ascendc_runtime MODULE REQUIRED)
find_cann_package(error_manager MODULE REQUIRED)
find_cann_package(msprof MODULE REQUIRED)
find_cann_package(platform MODULE REQUIRED)
find_cann_package(asc_devkit MODULE REQUIRED)
add_cann_third_party(json)

include(${CMAKE_CURRENT_LIST_DIR}/llvm21/Bootstrap.cmake)
autofuse_prepare_llvm21()
