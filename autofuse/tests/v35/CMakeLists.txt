file(GLOB_RECURSE OPTIMIZE_UTEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ut/optimize/*.cpp")
file(GLOB_RECURSE CODEGEN_UTEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ut/codegen/*.cpp")
file(GLOB_RECURSE ASCIR_UTEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ut/ascir/*.cpp")
file(GLOB_RECURSE ASCIR_STEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/st/ascir/*.cpp")
file(GLOB_RECURSE OPTIMIZE_STEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/st/optimize/*.cpp")
file(GLOB_RECURSE OPTIMIZE_SRC_FILES "${CODE_ROOT_DIR}/v35/optimize/*.cpp")
file(GLOB_RECURSE CODEGEN_SRC_FILES "${CODE_ROOT_DIR}/v35/codegen/*.cpp")
file(GLOB_RECURSE ATT_SRC_FILES "${CODE_ROOT_DIR}/v35/att/*.cpp")
file(GLOB_RECURSE ATT_ST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/st/att/*.cpp")
file(GLOB_RECURSE ATT_UT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/ut/att/*.cpp")
# Replay physical descriptors through the registered v35 models in both host test builds.
set(ATT_MODEL_SCENARIO_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/att/gen_model_info/api_perf_register/test_calibrated_ascir_perf_v2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/att/gen_model_info/api_perf_register/test_calibrated_nddma_model_v2.cpp
)
file(GLOB_RECURSE ASCIR_REG_FUNC_FILES "${CODE_ROOT_DIR}/v35/ascir/reg_func/*.cpp")

target_sources(optimize_ut PRIVATE
        ${OPTIMIZE_UTEST_FILES}
)

target_include_directories(optimize_ut PRIVATE
        ${CODE_ROOT_DIR}/v35
        ${CODE_ROOT_DIR}/v35/optimize
)

target_sources(optimize_st PRIVATE
        ${OPTIMIZE_STEST_FILES}
)

target_include_directories(optimize_st PRIVATE
        ${CODE_ROOT_DIR}/v35
        ${CODE_ROOT_DIR}/v35/optimize
)

target_include_directories(test_codegen PRIVATE
        ${CODE_ROOT_DIR}/v35/
        ${CODE_ROOT_DIR}/v35/codegen
        ${CODE_ROOT_DIR}/v35/codegen/micro_api_call
        ${CODE_ROOT_DIR}/v35/codegen/reg_api_call
        ${CODE_ROOT_DIR}/v35/codegen/vec_func_call
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/ascir
        ${CODE_ROOT_DIR}/v35/ascir
)

target_sources(test_codegen PRIVATE
        ${CODEGEN_UTEST_FILES}
)

target_link_libraries(test_codegen PRIVATE
        ascendc_api_regbase_extend
        ascendc_api_cube_extend
)

target_link_libraries(codegen_st PRIVATE
        ascendc_api_regbase_extend
        ascendc_api_cube_extend
)

target_sources(codegen_st PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/st/codegen/test_codegen_broadcast_reg_api_call.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/st/codegen/test_codegen_indirect_load_simt.cpp
)

target_include_directories(codegen_st PRIVATE
        ${CODE_ROOT_DIR}/v35/codegen/reg_api_call
)

target_sources(test_ascir_ut PRIVATE
        ${ASCIR_UTEST_FILES}
        ${ASCIR_REG_FUNC_FILES}
)

target_include_directories(test_ascir_ut PRIVATE
        ${CODE_ROOT_DIR}/ascir
        ${CODE_ROOT_DIR}/v35/ascir
        ${CODE_ROOT_DIR}/v35/ascir/reg_func
        ${CODE_ROOT_DIR}/ascir/reg_func
)

target_sources(test_ascir_st PRIVATE
        ${ASCIR_STEST_FILES}
        ${ASCIR_REG_FUNC_FILES}
)

target_include_directories(test_ascir_st PRIVATE
        ${CODE_ROOT_DIR}/ascir
        ${CODE_ROOT_DIR}/v35/ascir
        ${CODE_ROOT_DIR}/v35/ascir/reg_func
        ${CODE_ROOT_DIR}/ascir/reg_func
)

target_include_directories(att_ut PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/v35/
)

target_include_directories(att_st PRIVATE
        ${CODE_ROOT_DIR}/v35/
)

target_sources(optimize PRIVATE
        ${OPTIMIZE_SRC_FILES}
)

target_include_directories(optimize PRIVATE
        ${CODE_ROOT_DIR}/v35/optimize
)

target_sources(att PRIVATE
        ${ATT_SRC_FILES}
)

target_include_directories(att PRIVATE
        ${CODE_ROOT_DIR}/v35/att
)

target_sources(codegen PRIVATE
        ${CODEGEN_SRC_FILES}
)

target_sources(att_st PRIVATE
        ${ATT_ST_FILES}
        ${ATT_MODEL_SCENARIO_SOURCES}
)

target_sources(att_ut PRIVATE
        ${ATT_UT_FILES}
        ${ATT_MODEL_SCENARIO_SOURCES}
)

target_include_directories(codegen PRIVATE
        ${CODE_ROOT_DIR}/v35/
        ${CODE_ROOT_DIR}/v35/codegen
        ${CODE_ROOT_DIR}
)

target_link_libraries(codegen PRIVATE
        ascendc_api_regbase_extend
        ascendc_api_cube_extend
)

add_subdirectory(ut/ascendc)
add_subdirectory(st/backend_e2e_v2)
add_subdirectory(st/codegen/e2e_v2)
