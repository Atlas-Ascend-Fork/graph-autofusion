backend_e2e_st_test(workspace_uint64_multi_reduce_test
    CODEGEN workspace_uint64_multi_reduce_backend_generate.cpp
    KERNEL_SRC
        workspace_uint64_multi_reduce_kernel.cpp
        workspace_uint64_multi_reduce_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_workspace_uint64_multi_reduce.cpp)

set(host_tiling_workdir ${CMAKE_CURRENT_BINARY_DIR}/host_tiling)
file(MAKE_DIRECTORY ${host_tiling_workdir})
do_backend_e2e_st_test(workspace_uint64_tiling_test
    HOST_ONLY
    WORKDIR ${host_tiling_workdir}
    CODEGEN workspace_uint64_multi_reduce_backend_generate.cpp
    KERNEL_SRC
        workspace_uint64_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_workspace_uint64_tiling.cpp)
target_compile_definitions(workspace_uint64_tiling_test_codegen PRIVATE WORKSPACE_TILING_ONLY=1)
set_tests_properties(workspace_uint64_tiling_test_codegen PROPERTIES WORKING_DIRECTORY ${host_tiling_workdir})

foreach(codegen_target workspace_uint64_multi_reduce_test_codegen workspace_uint64_tiling_test_codegen)
    target_include_directories(${codegen_target} PRIVATE ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph)
    target_link_libraries(${codegen_target} easy_asc_graph)
endforeach()
