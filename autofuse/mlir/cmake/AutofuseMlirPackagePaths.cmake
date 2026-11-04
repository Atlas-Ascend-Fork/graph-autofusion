# Resolve package paths before any LLVM-dependent bridge target is configured.
include_guard(GLOBAL)

if(NOT ENABLE_AUTOFUSE_MLIR)
    return()
endif()

# These are user inputs. Import environment values as normal variables rather
# than cache entries so a reconfigure can observe a changed toolchain.
# Explicit -D values remain higher-priority cache inputs.
if(NOT DEFINED LLVM_BUILD_DIR AND DEFINED ENV{LLVM_BUILD_DIR})
    set(LLVM_BUILD_DIR "$ENV{LLVM_BUILD_DIR}")
endif()
if(NOT DEFINED MLIR_DIR AND DEFINED ENV{MLIR_DIR})
    set(MLIR_DIR "$ENV{MLIR_DIR}")
endif()
if(NOT DEFINED LLVM_DIR AND DEFINED ENV{LLVM_DIR})
    set(LLVM_DIR "$ENV{LLVM_DIR}")
endif()

get_filename_component(_af_mlir_root "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)

# Prefer the build root when supplied. This prevents stale MLIR_DIR/LLVM_DIR
# cache entries from overriding the current explicit LLVM_BUILD_DIR.
if(LLVM_BUILD_DIR)
    set(_af_llvm_root "${LLVM_BUILD_DIR}")
    set(_af_mlir_cmake_dir "${_af_llvm_root}/lib/cmake/mlir")
elseif(MLIR_DIR)
    set(_af_mlir_cmake_dir "${MLIR_DIR}")
    get_filename_component(_af_llvm_root "${_af_mlir_cmake_dir}" DIRECTORY)
    get_filename_component(_af_llvm_root "${_af_llvm_root}" DIRECTORY)
    get_filename_component(_af_llvm_root "${_af_llvm_root}" DIRECTORY)
elseif(LLVM_DIR)
    get_filename_component(_af_llvm_root "${LLVM_DIR}" DIRECTORY)
    get_filename_component(_af_llvm_root "${_af_llvm_root}" DIRECTORY)
    get_filename_component(_af_llvm_root "${_af_llvm_root}" DIRECTORY)
    set(_af_mlir_cmake_dir "${_af_llvm_root}/lib/cmake/mlir")
else()
    set(_af_llvm_root "${_af_mlir_root}/.deps/llvm")
    set(_af_mlir_cmake_dir "${_af_llvm_root}/lib/cmake/mlir")
endif()

if(MLIR_DIR AND NOT LLVM_BUILD_DIR)
    set(_af_mlir_cmake_dir "${MLIR_DIR}")
endif()
if(LLVM_DIR AND NOT LLVM_BUILD_DIR)
    set(_af_llvm_cmake_dir "${LLVM_DIR}")
else()
    set(_af_llvm_cmake_dir "${_af_llvm_root}/lib/cmake/llvm")
endif()
if(NOT EXISTS "${_af_mlir_cmake_dir}/MLIRConfig.cmake")
    message(FATAL_ERROR
        "ENABLE_AUTOFUSE_MLIR=ON requires MLIRConfig.cmake. "
        "Use the Autofuse MLIR development container or install LLVM/MLIR, then set "
        "-DLLVM_BUILD_DIR=/path/to/llvm-build. "
        "Checked: ${_af_mlir_cmake_dir}")
endif()
if(NOT EXISTS "${_af_llvm_cmake_dir}/LLVMConfig.cmake")
    message(FATAL_ERROR "LLVMConfig.cmake not found in ${_af_llvm_cmake_dir}")
endif()

# Derived paths stay out of CMakeCache.txt. Subdirectories inherit these normal
# variables, while user-provided cache inputs remain available for reconfigure.
set(MLIR_DIR "${_af_mlir_cmake_dir}")
set(LLVM_DIR "${_af_llvm_cmake_dir}")
list(APPEND CMAKE_PREFIX_PATH "${_af_llvm_root}")
