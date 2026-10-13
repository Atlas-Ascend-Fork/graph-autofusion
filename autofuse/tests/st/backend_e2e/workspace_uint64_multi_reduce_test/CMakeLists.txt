backend_e2e_st_test(workspace_uint64_multi_reduce_test
    CODEGEN workspace_uint64_multi_reduce_backend_generate.cpp
    KERNEL_SRC
        workspace_uint64_multi_reduce_kernel.cpp
        workspace_uint64_multi_reduce_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_workspace_uint64_multi_reduce.cpp)

target_include_directories(workspace_uint64_multi_reduce_test_codegen PRIVATE
    ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph)
target_link_libraries(workspace_uint64_multi_reduce_test_codegen easy_asc_graph)
