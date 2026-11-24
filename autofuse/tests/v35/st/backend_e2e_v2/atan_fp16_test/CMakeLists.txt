backend_e2e_st_test(atan_fp16_test
    CODEGEN atan_fp16_backend_generate.cpp
    KERNEL_SRC
        atan_fp16_test_kernel.cpp
        atan_fp16_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_atan_fp16_expect_kernel.cpp)
