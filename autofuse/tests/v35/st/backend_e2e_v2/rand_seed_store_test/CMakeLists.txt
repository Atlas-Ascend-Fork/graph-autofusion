foreach(dtype_suffix IN ITEMS u32 i32 fp16 bf16 fp32)
    backend_e2e_st_test(rand_seed_store_test
        SUFFIX _${dtype_suffix}
    CODEGEN rand_seed_store_backend_generator.cpp
    KERNEL_SRC
        rand_seed_store_test_${dtype_suffix}_kernel.cpp
        rand_seed_store_test_${dtype_suffix}_tiling.cpp
        autofuse_tiling_data.h
        TEST_SRC test_e2e_rand_seed_store_kernel.cpp)
    string(TOUPPER ${dtype_suffix} dtype_macro)
    target_compile_definitions(rand_seed_store_test_${dtype_suffix}_codegen_v2 PRIVATE RAND_E2E_DTYPE_${dtype_macro}=1)
    target_compile_definitions(rand_seed_store_test_${dtype_suffix}_e2e_v2 PRIVATE RAND_E2E_DTYPE_${dtype_macro}=1)
    target_include_directories(rand_seed_store_test_${dtype_suffix}_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
endforeach()
