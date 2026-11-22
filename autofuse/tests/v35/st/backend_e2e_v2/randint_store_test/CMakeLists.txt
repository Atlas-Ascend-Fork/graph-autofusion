backend_e2e_st_test(randint_store_test
    CODEGEN randint_store_backend_generator.cpp
    KERNEL_SRC
        randint_store_test_kernel.cpp
        randint_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_randint_store_kernel.cpp)
