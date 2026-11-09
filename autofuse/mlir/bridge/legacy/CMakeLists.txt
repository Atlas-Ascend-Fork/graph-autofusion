# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Legacy schedule and ATT/Codegen adapters used during the migration to MLIR.
#
# Although these adapters consume legacy Autofuse/ATT types, their callers
# belong to the MLIR path. The root project adds them only when MLIR is enabled.
add_library(AutofuseMlirScheduleLegacyAdapter STATIC
  ${MLIR_CODE_ROOT_DIR}/common/fingerprint/schedule_fingerprint.cpp
  ${MLIR_CODE_ROOT_DIR}/common/tiling_schema/tiling_field_schema.cpp
)

target_include_directories(AutofuseMlirScheduleLegacyAdapter PUBLIC
  ${CODE_ROOT_DIR}
  ${CODE_ROOT_DIR}/inc
  ${CODE_ROOT_DIR}/inc/graph_metadef
  ${CODE_ROOT_DIR}/inc/graph_metadef/graph
  ${CODE_ROOT_DIR}/common
  ${MLIR_CODE_ROOT_DIR}
)

target_link_libraries(AutofuseMlirScheduleLegacyAdapter PUBLIC
  AutofuseMlirScheduleCommon
  ascend_protobuf
  metadef_headers
  unified_dlog
  intf_pub
)

if(TARGET metadef_protos_af)
  add_dependencies(AutofuseMlirScheduleLegacyAdapter metadef_protos_af)
endif()

if(ENABLE_AUTOFUSE_MLIR)
  find_package(LLVM REQUIRED CONFIG)
endif()

add_library(AutofuseMlirScheduleBridge STATIC
  ${MLIR_CODE_ROOT_DIR}/bridge/AttSchemaAdapter/AttSchemaAdapter.cpp
  ${MLIR_CODE_ROOT_DIR}/bridge/PlanSelector/PlanSelection.cpp
  ${MLIR_CODE_ROOT_DIR}/bridge/PlanSelector/PlanSelector.cpp
)

target_include_directories(AutofuseMlirScheduleBridge PUBLIC
  ${CODE_ROOT_DIR}
  ${CODE_ROOT_DIR}/inc
  ${CODE_ROOT_DIR}/inc/graph_metadef
  ${CODE_ROOT_DIR}/inc/graph_metadef/graph
  ${CODE_ROOT_DIR}/common
  ${CODE_ROOT_DIR}/att
  ${CODE_ROOT_DIR}/att/base
  ${CODE_ROOT_DIR}/att/gen_model_info
  ${MLIR_CODE_ROOT_DIR}
)

target_include_directories(AutofuseMlirScheduleBridge SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS})

target_link_libraries(AutofuseMlirScheduleBridge PUBLIC
  AutofuseMlirScheduleLegacyAdapter
)

target_link_libraries(aihac_codegen PRIVATE
  AutofuseMlirScheduleBridge
)
