backend_e2e_st_test(cast_uint8_to_int64_test
    CODEGEN cast_uint8_to_int64_backend_generate.cpp
    KERNEL_SRC
        cast_uint8_to_int64_test_kernel.cpp
        cast_uint8_to_int64_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cast_uint8_to_int64_expect_kernel.cpp)
