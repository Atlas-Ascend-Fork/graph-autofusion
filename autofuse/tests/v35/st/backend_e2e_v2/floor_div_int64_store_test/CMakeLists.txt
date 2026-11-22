backend_e2e_st_test(floor_div_int64_store_test
    CODEGEN floor_div_int64_store_backend_generator.cpp
    KERNEL_SRC
        floor_div_int64_store_test_kernel.cpp
        floor_div_int64_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_floor_div_int64_store_kernel.cpp)
