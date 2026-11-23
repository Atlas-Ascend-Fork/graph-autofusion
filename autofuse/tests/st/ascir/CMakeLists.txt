add_executable(test_ascir_st
    test_main_st.cpp
    reg_func/test_reg_func_broadcast.cpp
    reg_func/test_reg_func_clipbyvalue.cpp
    reg_func/test_reg_func_compare.cpp
    reg_func/test_reg_func_concat.cpp
    reg_func/test_reg_func_div.cpp
    reg_func/test_reg_func_is_nan.cpp
    reg_func/test_reg_func_logic_and.cpp
    reg_func/test_reg_func_logic_not.cpp
    reg_func/test_reg_func_logic_or.cpp
    reg_func/test_reg_func_pow.cpp
    reg_func/test_reg_func_rsqrt.cpp
    reg_func/test_reg_func_sigmoid.cpp
    reg_func/test_reg_func_sign.cpp
    reg_func/test_reg_func_sub.cpp
    reg_func/test_reg_func_tanh.cpp
    reg_func/test_reg_func_truediv.cpp
    reg_func/test_reg_func_where.cpp
    # reg_func/test_reg_func_zeros_like.cpp
    reg_func/test_reg_func_reduce.cpp
    reg_func/test_reg_func_cast.cpp
    reg_func/test_reg_func_isfinite.cpp
    reg_func/test_reg_func_gather.cpp
    reg_func/test_reg_func_erf.cpp
    reg_func/test_reg_func_axpy.cpp
    reg_func/test_reg_func_unsupported.cpp
    reg_func_v2/test_reg_func_sin_v2.cpp
    reg_func_v2/test_reg_func_erf_v2.cpp
    reg_func_v2/test_reg_func_gelu_v2.cpp
    reg_func_v2/test_reg_func_mod_v2.cpp
    reg_func_v2/test_reg_func_pow_v2.cpp
    reg_func_v2/test_reg_func_tanh_v2.cpp
    reg_func_v2/test_reg_func_igamma_v2.cpp
    reg_func_v2/test_reg_func_igammac_v2.cpp
    reg_func_v2/test_reg_func_sinh_v2.cpp
    reg_func_v2/test_reg_func_tan_v2.cpp
    reg_func_v2/test_reg_func_trunc_v2.cpp
    reg_func_v2/test_reg_func_xor_v2.cpp
    reg_func_v2/test_reg_func_bucketize_v2.cpp
    reg_func_v2/test_reg_func_softmax_v2.cpp
    generator/test_rand_dtype_contract_st.cpp
    code_dumper/code_dumper_unittest.cc
    ascir_utils/ascir_utils_st.cpp
)

target_include_directories(test_ascir_st PRIVATE
    ${ASCEND_ROOT}/x86_64-linux/include
    ${ASCEND_ROOT}/opp/built-in/op_proto/inc
    ${CODE_ROOT_DIR})

target_link_libraries(test_ascir_st
    -Wl,--no-as-needed
    error_manager
    aihac_symbolizer_af
    ascir
    aihac_ir_register
    ascir_builtin_ops
    metadef
    #ge_compiler
    GTest::gtest)

add_test(NAME ascir_st_aggregate
    COMMAND $<TARGET_FILE:test_ascir_st>
)
set_tests_properties(ascir_st_aggregate PROPERTIES
    LABELS "st;ascir_st"
)
