# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 量具，不是测试：**不接任何 check-* 目标**，产物是打到 stdout 的一张表。
#
# 链接方式照 CPU E2E 的生成器（test/e2e/cpu/AddCpuCase.cmake 里的 ${target}_generator）：
# 只链 MlirE2eCpuGraphBuilder（→ aihac_codegen + metadef）与 runtime stub，MLIR / AFIR 全部
# 从 libaihac_codegen.so 里走。**不要像 UT 那样再静态链 AFIRToAscendC 一份**——一个进程里
# 两份同名代码，调度一进 ScheduleUtils::ClearAllSizeVar 就段错误。
add_executable(af-capability-probe
    ${MLIR_CODE_ROOT_DIR}/test/probe/CapabilityBoundaryProbe.cpp
)

target_include_directories(af-capability-probe PRIVATE
    ${MLIR_CODE_ROOT_DIR}
    ${CODE_ROOT_DIR}/optimize
    ${CODE_ROOT_DIR}/codegen
)

# 跑的时候 LD_LIBRARY_PATH 要与 CPU E2E 用的那一份一致（含 CANN 的 devlib 与 metadef 的几个
# build 目录）；少了它调度会在 ClearAllSizeVar 里段错误。见源文件顶部那段。
target_link_libraries(af-capability-probe PRIVATE
    MlirE2eCpuGraphBuilder
    -Wl,--no-as-needed ${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET} -Wl,--as-needed
)
