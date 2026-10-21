backend_e2e_st_test(bitwise_bool_store_test
    CODEGEN bitwise_bool_store_backend_generator.cpp
    KERNEL_SRC
        bitwise_bool_store_test_kernel.cpp
        bitwise_bool_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_bitwise_bool_store_kernel.cpp)
