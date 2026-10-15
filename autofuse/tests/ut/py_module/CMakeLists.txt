find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

file(GLOB_RECURSE CODEGEN_TEST_SRCS CONFIGURE_DEPENDS "*.cpp")
add_library(pyautofuse_runtime_error_tiling SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/py_module/pyautofuse_runtime_error_tiling.cpp)
set_target_properties(pyautofuse_runtime_error_tiling PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../..)
add_library(test_py_module OBJECT
        ${CODEGEN_TEST_SRCS}
)
add_dependencies(test_py_module pyautofuse_runtime_error_tiling)
target_link_options(test_py_module PRIVATE
        -Wl,--no-as-needed
)
target_link_libraries(test_py_module PRIVATE
    atrace
    pyautofuse
    graph_base_af
    graph_af
    json
    Python3::Python
    GTest::gtest)
