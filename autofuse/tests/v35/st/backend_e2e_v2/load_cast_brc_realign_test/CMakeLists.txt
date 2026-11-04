backend_e2e_st_test(load_cast_brc_realign_test
    CODEGEN load_cast_brc_realign_backend_generate.cpp
    KERNEL_SRC
        load_cast_brc_realign_test_kernel.cpp
    TEST_SRC test_e2e_load_cast_brc_realign_expect_kernel.cpp)
