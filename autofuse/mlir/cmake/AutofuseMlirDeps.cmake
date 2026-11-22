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

# add_mlir_doc() 将文档复制到 ${MLIR_BINARY_DIR}/docs/，但安装的 MLIRConfig.cmake
# 不定义 MLIR_BINARY_DIR；留空会误写到绝对路径 /docs/。
# 此变量在 AddMLIR.cmake 中仅供文档复制使用，因此设为本项目 MLIR 构建目录，
# 使生成文档位于 <build>/autofuse/mlir/docs/Dialects/。
set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
