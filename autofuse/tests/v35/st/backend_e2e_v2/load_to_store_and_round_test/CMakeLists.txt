backend_e2e_st_test(load_to_store_and_round_test
    CODEGEN load_to_store_and_round_backend_generate.cpp
    KERNEL_SRC
        load_to_store_and_round_test_kernel.cpp
        load_to_store_and_round_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_to_store_and_round_expect_kernel.cpp)
