backend_e2e_st_test(cast_float_uint64_test
    CODEGEN cast_float_uint64_backend_generate.cpp
    KERNEL_SRC
        cast_float_uint64_test_kernel.cpp
        cast_float_uint64_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_cast_float_uint64_expect_kernel.cpp)
