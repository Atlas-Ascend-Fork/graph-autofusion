# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software and can be redistributed and used under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should have received a copy of the License along with this program. If not, see
# <https://www.cann-3rd.obs.cn-north-4.myhuaweicloud.com/cann-open-software-license-agreement-v2-0.txt>.
# ----------------------------------------------------------------------------------------------------------

function(autofuse_prepare_llvm21)
    if(NOT ENABLE_AUTOFUSE_MLIR)
        return()
    endif()

    if(LLVM_BUILD_DIR OR DEFINED ENV{LLVM_BUILD_DIR}
            OR MLIR_DIR OR DEFINED ENV{MLIR_DIR}
            OR LLVM_DIR OR DEFINED ENV{LLVM_DIR})
        return()
    endif()

    set(bootstrap_dir "${CMAKE_BINARY_DIR}/llvm21-bootstrap")
    set(llvm21_project_url
        "https://raw.gitcode.com/cann-src-third-party/llvm/blobs/fb9fee0934ac633b3d2951438ddeee37bab53478/llvm-project-llvmorg-21.1.8.src.tar.xz")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            -S "${CMAKE_SOURCE_DIR}/cmake/llvm21"
            -B "${bootstrap_dir}"
            -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}
            -DLLVM21_BUILD_JOBS=12
            -DLLVM21_ENABLE_PYTHON_BINDINGS=OFF
            -DLLVM21_PROJECT_URL=${llvm21_project_url}
        RESULT_VARIABLE configure_result)
    if(configure_result)
        message(FATAL_ERROR "Autofuse LLVM 21 bootstrap configure failed")
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            --build "${bootstrap_dir}"
            --target third_party_llvm21
            --parallel 12
        RESULT_VARIABLE build_result)
    if(build_result)
        message(FATAL_ERROR "Autofuse LLVM 21 bootstrap build failed")
    endif()

    file(READ "${bootstrap_dir}/llvm21-install-path.txt" install_path)
    string(STRIP "${install_path}" install_path)
    set(LLVM_BUILD_DIR "${install_path}" CACHE PATH "Autofuse LLVM 21 install path" FORCE)
endfunction()
