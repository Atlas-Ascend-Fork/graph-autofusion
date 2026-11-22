backend_e2e_st_test(transpose_int8_test
    CODEGEN transpose_int8_backend_generate.cpp
    KERNEL_SRC
        transpose_int8_test_kernel.cpp
        transpose_int8_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_transpose_int8_kernel.cpp)

set_target_properties(transpose_int8_test_codegen_v2 PROPERTIES
    BUILD_RPATH
        "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression;${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir;${CMAKE_BINARY_DIR}/autofuse/ascir/meta")
target_link_options(transpose_int8_test_codegen_v2 PRIVATE -Wl,--disable-new-dtags)
