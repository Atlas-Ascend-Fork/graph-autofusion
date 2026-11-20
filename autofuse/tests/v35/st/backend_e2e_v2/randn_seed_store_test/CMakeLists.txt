foreach(dtype_suffix IN ITEMS u32 i32 fp16 bf16 fp32)
    backend_e2e_st_test(randn_seed_store_test
        SUFFIX _${dtype_suffix}
    CODEGEN randn_seed_store_backend_generator.cpp
    KERNEL_SRC
        randn_seed_store_test_${dtype_suffix}_kernel.cpp
        randn_seed_store_test_${dtype_suffix}_tiling.cpp
        autofuse_tiling_data.h
        TEST_SRC test_e2e_randn_seed_store_kernel.cpp)
    string(TOUPPER ${dtype_suffix} dtype_macro)
    target_compile_definitions(randn_seed_store_test_${dtype_suffix}_codegen_v2 PRIVATE RANDN_E2E_DTYPE_${dtype_macro}=1)
    target_compile_definitions(randn_seed_store_test_${dtype_suffix}_e2e_v2 PRIVATE RANDN_E2E_DTYPE_${dtype_macro}=1)
    target_include_directories(randn_seed_store_test_${dtype_suffix}_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
endforeach()
