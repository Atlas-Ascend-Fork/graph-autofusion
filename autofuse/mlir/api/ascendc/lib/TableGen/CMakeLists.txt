# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(TARGET_NAME ascir-tblgen)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  TableGen
)

add_tablegen(${TARGET_NAME} AscIR
  PARTIAL_SOURCES_INTENDED
  DESTINATION "${ASCIR_BINARY_DIR}"
  EXPORT AscIR
  GenAPITypedefs.cpp
  GenAPITypes.cpp
  GenPybindDefs.cpp
  GenPybindDefsTypes.cpp
  GenOpEmitDecls.cpp
  GenOpEmitDefs.cpp
  main.cpp
  PrintDecls.cpp
  Utils.cpp
  )

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Tablegenning")
mlir_check_all_link_libraries(${TARGET_NAME})
