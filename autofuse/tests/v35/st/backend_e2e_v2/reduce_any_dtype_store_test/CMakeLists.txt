function(add_reduce_any_dtype_st_test test_name dtype exec_dtype)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN reduce_any_dtype_backend_generate.cpp
        KERNEL_SRC
            ${test_name}_kernel.cpp
            ${test_name}_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_reduce_any_dtype_store_kernel.cpp)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE
        REDUCE_DTYPE=${dtype}
        REDUCE_EXEC_DTYPE=\"${exec_dtype}\")
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE
        KERNEL_NAME=load_logical_reduce_store_test
        ELEMENT_TYPE=${exec_dtype})
endfunction()

add_reduce_any_dtype_st_test(reduce_any_bool_store_test af::DT_BOOL uint8_t)
add_reduce_any_dtype_st_test(reduce_any_uint8_store_test af::DT_UINT8 uint8_t)
add_reduce_any_dtype_st_test(reduce_any_int8_store_test af::DT_INT8 int8_t)
add_reduce_any_dtype_st_test(reduce_any_int16_store_test af::DT_INT16 int16_t)
add_reduce_any_dtype_st_test(reduce_any_int32_store_test af::DT_INT32 int32_t)
add_reduce_any_dtype_st_test(reduce_any_int64_store_test af::DT_INT64 int64_t)
add_reduce_any_dtype_st_test(reduce_any_float_store_test af::DT_FLOAT float)
