backend_e2e_st_test(remainder_fp16_test
    CODEGEN remainder_fp16_backend_generate.cpp
    KERNEL_SRC
        remainder_fp16_test_kernel.cpp
        remainder_fp16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_remainder_fp16_expect_kernel.cpp)
