backend_e2e_st_test(remainder_f32_test
    CODEGEN remainder_f32_backend_generate.cpp
    KERNEL_SRC
        remainder_f32_test_kernel.cpp
        remainder_f32_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_remainder_f32_expect_kernel.cpp)
