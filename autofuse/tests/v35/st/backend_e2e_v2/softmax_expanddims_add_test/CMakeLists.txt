backend_e2e_st_test(softmax_expanddims_add_test
        CODEGEN softmax_expanddims_add_backend_generate.cpp
        KERNEL_SRC
        softmax_expanddims_add_test_kernel.cpp
        softmax_expanddims_add_test_tiling.cpp
        autofuse_tiling_data.h
        TEST_SRC test_e2e_softmax_expanddims_add_kernel.cpp)
