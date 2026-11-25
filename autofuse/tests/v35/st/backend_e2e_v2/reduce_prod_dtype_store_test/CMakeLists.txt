function(add_reduce_prod_dtype_st_test test_name dtype input_dtype compute_dtype)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN reduce_prod_dtype_backend_generate.cpp
        KERNEL_SRC
            ${test_name}_kernel.cpp
            ${test_name}_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_reduce_prod_dtype_store_kernel.cpp)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE
        REDUCE_DTYPE=${dtype}
        REDUCE_EXEC_DTYPE=\"${compute_dtype}\")
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE
        KERNEL_NAME=load_prod_reduce_store_test
        ELEMENT_TYPE=${input_dtype})
endfunction()

function(add_reduce_prod_dtype_cols_st_test test_name dtype input_dtype compute_dtype cols)
    add_reduce_prod_dtype_st_test(${test_name} ${dtype} ${input_dtype} ${compute_dtype})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE REDUCE_TEST_COLS=${cols})
endfunction()

function(add_reduce_prod_partial_mask_st_test test_name dtype input_dtype compute_dtype rows cols)
    add_reduce_prod_dtype_st_test(${test_name} ${dtype} ${input_dtype} ${compute_dtype})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE
        REDUCE_TEST_ROWS=${rows}
        REDUCE_TEST_COLS=${cols}
        REDUCE_PARTIAL_MASK_CASE=1)
endfunction()

add_reduce_prod_dtype_st_test(reduce_prod_int8_store_test af::DT_INT8 int8_t int16_t)
add_reduce_prod_dtype_st_test(reduce_prod_int16_store_test af::DT_INT16 int16_t int16_t)
add_reduce_prod_dtype_cols_st_test(reduce_prod_int32_store_test af::DT_INT32 int32_t int32_t 1028)
add_reduce_prod_dtype_cols_st_test(reduce_prod_int64_store_test af::DT_INT64 int64_t int64_t 1028)
add_reduce_prod_dtype_cols_st_test(reduce_prod_float_store_test af::DT_FLOAT float float 1028)
add_reduce_prod_dtype_cols_st_test(reduce_prod_int32_1025_store_test af::DT_INT32 int32_t int32_t 1025)
add_reduce_prod_dtype_cols_st_test(reduce_prod_int64_1025_store_test af::DT_INT64 int64_t int64_t 1025)
add_reduce_prod_dtype_cols_st_test(reduce_prod_int64_129800_store_test af::DT_INT64 int64_t int64_t 129800)
add_reduce_prod_dtype_cols_st_test(reduce_prod_float_1025_store_test af::DT_FLOAT float float 1025)
add_reduce_prod_partial_mask_st_test(reduce_prod_int64_partial_mask_store_test af::DT_INT64 int64_t int64_t 4 3)

set(prod_ra_case_workdir ${CMAKE_CURRENT_BINARY_DIR}/reduce_prod_int8_ra_store_test)
file(MAKE_DIRECTORY ${prod_ra_case_workdir})
do_backend_e2e_st_test(reduce_prod_int8_ra_store_test
    WORKDIR ${prod_ra_case_workdir}
    CODEGEN reduce_prod_dtype_backend_generate.cpp
    KERNEL_SRC
        reduce_prod_int8_ra_store_test_kernel.cpp
        reduce_prod_int8_ra_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_reduce_prod_int8_ra_store_kernel.cpp)
target_compile_definitions(reduce_prod_int8_ra_store_test_codegen_v2 PRIVATE REDUCE_PROD_RA=1)
target_compile_definitions(reduce_prod_int8_ra_store_test_e2e_v2 PRIVATE
    KERNEL_NAME=load_prod_reduce_store_ra_test
    ELEMENT_TYPE=int8_t)
