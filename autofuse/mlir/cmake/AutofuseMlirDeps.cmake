# Load MLIR/LLVM packages and CMake modules for the MLIR subtree.
include_guard(GLOBAL)

if(NOT ENABLE_AUTOFUSE_MLIR)
    return()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/AutofuseMlirPackagePaths.cmake)

find_package(MLIR REQUIRED CONFIG)
if(LLVM_DIR)
    find_package(LLVM REQUIRED CONFIG)
endif()

if(DEFINED MLIR_CMAKE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
endif()
if(DEFINED LLVM_CMAKE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
endif()

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# add_mlir_doc() 把生成的 .md 复制到 ${MLIR_BINARY_DIR}/docs/。MLIRConfig.cmake 不定义
# MLIR_BINARY_DIR（它只在 LLVM 树内构建时存在），留空会让复制目标退化成绝对路径
# /docs/... 而失败。MLIR_BINARY_DIR 在 AddMLIR.cmake 中仅被 add_mlir_doc 使用，
# 因此在此指向本项目的 MLIR 构建目录是安全的，且让全部方言 doc target 产物落在
# <build>/autofuse/mlir/docs/Dialects/ 下。
set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
