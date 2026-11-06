# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Exercise the production library, not the legacy UT pyautofuse target, which
# links separate codegen/ATT test libraries. Reuse the real binding sources.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
Python3_add_library(AutofuseMlirPythonBinding MODULE
  ${CODE_ROOT_DIR}/compiler/py_module/pyascir.cpp
  ${CODE_ROOT_DIR}/compiler/py_module/pyascir_types.cpp
  ${CODE_ROOT_DIR}/compiler/py_module/pyascir_common_utils.cpp
  ${CODE_ROOT_DIR}/compiler/py_module/pyautofuse.cpp
)
set_target_properties(AutofuseMlirPythonBinding PROPERTIES OUTPUT_NAME pyautofuse)
target_include_directories(AutofuseMlirPythonBinding PRIVATE
  ${CODE_ROOT_DIR}
  ${CODE_ROOT_DIR}/optimize
  ${CODE_ROOT_DIR}/codegen
)
target_compile_options(AutofuseMlirPythonBinding PRIVATE
  -Wno-error=variadic-macros -Wno-missing-field-initializers -Wno-cast-function-type
)
target_compile_definitions(AutofuseMlirPythonBinding PRIVATE
  ENABLE_AUTOFUSE_MLIR=1 google=ascend_private
)
target_link_libraries(AutofuseMlirPythonBinding PRIVATE
  af_intf_pub aihac_codegen aihac_ir ascend_protobuf json graph_af graph_base_af
  unified_dlog metadef_headers ascir_ops_headers atrace_headers atrace_share c_sec_headers
)
