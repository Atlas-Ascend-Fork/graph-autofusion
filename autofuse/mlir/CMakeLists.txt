cmake_minimum_required(VERSION 3.16)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AutofuseMlirDeps.cmake)

if(NOT ENABLE_AUTOFUSE_MLIR)
    return()
endif()

message(STATUS "Enable Autofuse MLIR migration infrastructure")

# Enable CTest registration for standalone MLIR builds; the main autofuse test
# entry enables it only when RUN_TEST is set.
enable_testing()

set(AF_ASCENDC_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/api/ascendc" CACHE PATH "AscendC API dialect/target source directory")
if(NOT EXISTS "${AF_ASCENDC_API_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "AscendC API sources are missing under ${AF_ASCENDC_API_DIR}")
endif()

# CMake versions before 3.23 do not track nested .td includes in the TableGen
# fallback. Keep the dependency list at the MLIR layer so every TableGen
# subdirectory is covered.
set(_AF_MLIR_SAVED_LLVM_TARGET_DEPENDS "${LLVM_TARGET_DEPENDS}")
file(GLOB_RECURSE AF_MLIR_TD_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.td")
list(APPEND LLVM_TARGET_DEPENDS ${AF_MLIR_TD_FILES})

set(AF_ASCENDC_API_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/api/ascendc")
add_subdirectory(${AF_ASCENDC_API_DIR} ${AF_ASCENDC_API_BINARY_DIR})

# Single owner of the AscendC API include pair (source + tablegen output).
# Consumers link this instead of repeating both paths.
add_library(ascendc_api_headers INTERFACE)
target_include_directories(ascendc_api_headers INTERFACE
    ${AF_ASCENDC_API_DIR}/include
    ${AF_ASCENDC_API_BINARY_DIR}/include
)

if(ENABLE_AUTOFUSE_MLIR_TESTS AND NOT TARGET GTest::gtest)
    add_cann_third_party(gtest)
endif()

add_subdirectory(target/Ascend)
add_subdirectory(dialect/AFIR)
add_subdirectory(dialect/AFPlan)
add_subdirectory(conversion)
add_subdirectory(artifact)
add_subdirectory(bridge)
add_subdirectory(tools/af-afir-gen)
add_subdirectory(tools/af-opt)
add_subdirectory(test)

set(LLVM_TARGET_DEPENDS "${_AF_MLIR_SAVED_LLVM_TARGET_DEPENDS}")
unset(_AF_MLIR_SAVED_LLVM_TARGET_DEPENDS)
unset(AF_MLIR_TD_FILES)
