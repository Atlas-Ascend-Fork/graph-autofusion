add_afir_gtest(MlirCodegenSt
  SOURCES
    bridge/CodegenOrchestrationSt.cpp
    bridge/LegacyGraphGuardSt.cpp
    ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph/asc_graph_builder.cpp
  LIBS
    aihac_codegen
    metadef
    MLIRIR
    MLIRLLVMDialect
    AFIRToAscendC
    AfirKernelSource
    ${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET}
    ascendc_api_headers
  LABELS "st;mlir_st;codegen_orchestration_st"
  COMMENT "Running MLIR schedule-to-codegen system test"
)

target_include_directories(MlirCodegenSt PRIVATE
    ${MLIR_CODE_ROOT_DIR}
    ${CODE_ROOT_DIR}/tests
    ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph
    # `st/common/runtime_stub.h` 自己 include 的 `runtime_stub.h` 是 depends 那一份。只加这一条
    # 而不加 `tests/st/common`，是为了让 `LegacyGraphGuardSt.cpp` 里那个带目录的 include 成为
    # 唯一解——两份同名头按路径顺序解析过一次，量错了一整期（见那个文件顶部）。
    ${CODE_ROOT_DIR}/tests/depends/runtime/src
    ${CODE_ROOT_DIR}/optimize
    ${CODE_ROOT_DIR}/codegen
)

target_compile_definitions(MlirCodegenSt PRIVATE
    AUTOFUSE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

# The legacy Python UT/ST share their graph tests. Reuse that source with a
# separate ST registration and process, using the production Python binding.
add_test(NAME LegacyInductorTransposePythonSt
  COMMAND ${Python3_EXECUTABLE}
    ${MLIR_CODE_ROOT_DIR}/test/ut/python/test_legacy_inductor_transpose.py
    --extension $<TARGET_FILE:AutofuseMlirPythonBinding>
    --python-source ${CODE_ROOT_DIR}/compiler/python
)
set_tests_properties(LegacyInductorTransposePythonSt PROPERTIES
  LABELS "mlir_st;inductor_transpose"
  ENVIRONMENT "LD_LIBRARY_PATH=${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH};LD_PRELOAD=$<TARGET_FILE:${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET}>"
)
add_custom_target(check-legacy-inductor-transpose-python-st
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "^LegacyInductorTransposePythonSt$"
  DEPENDS AutofuseMlirPythonBinding ${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET}
)
set_property(GLOBAL APPEND PROPERTY AFIR_GTEST_CHECK_TARGETS_st check-legacy-inductor-transpose-python-st)

# Collect the check-* drivers registered by add_afir_gtest() above.
get_property(AF_ST_CHECK_TARGETS GLOBAL PROPERTY AFIR_GTEST_CHECK_TARGETS_st)

add_custom_target(check-st
  DEPENDS ${AF_ST_CHECK_TARGETS}
  COMMENT "Running MLIR system tests"
)
