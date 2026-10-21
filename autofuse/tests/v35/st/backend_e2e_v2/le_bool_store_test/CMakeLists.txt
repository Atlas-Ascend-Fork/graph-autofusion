backend_e2e_st_test(le_bool_store_test
    CODEGEN le_bool_store_backend_generator.cpp
    KERNEL_SRC
        le_bool_store_test_kernel.cpp
        le_bool_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_le_bool_store_kernel.cpp)
