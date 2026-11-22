# HOST_ONLY（tiling-only）e2e用例：codegen阶段以att_accuracy_level=0生成分段缓存tiling代码，
# e2e阶段host侧两次调用AutofuseTiling（同分段不同shape），验证分段缓存MISS→SAVE→HIT复用链路
set(segment_cache_tiling_workdir ${CMAKE_CURRENT_BINARY_DIR}/host_tiling)
file(MAKE_DIRECTORY ${segment_cache_tiling_workdir})
do_backend_e2e_st_test(segment_cache_tiling_test
    HOST_ONLY
    WORKDIR ${segment_cache_tiling_workdir}
    CODEGEN segment_cache_backend_generate.cpp
    KERNEL_SRC
        segment_cache_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_segment_cache_tiling.cpp)

target_include_directories(segment_cache_tiling_test_codegen PRIVATE ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph)
target_link_libraries(segment_cache_tiling_test_codegen easy_asc_graph)
