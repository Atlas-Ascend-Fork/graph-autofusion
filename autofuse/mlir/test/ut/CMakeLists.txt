add_subdirectory(artifact)
add_subdirectory(symbolic)
add_subdirectory(bridge)
add_subdirectory(target)
add_subdirectory(conversion)
add_subdirectory(dialect)
add_subdirectory(schedule)
add_subdirectory(python)

# Device-session orchestration and result accounting are host-only Python tests.
add_custom_target(check-npu-runner-offline)
foreach(test_name IN ITEMS test_npu_session test_run_entry test_allload_cases)
  set(test_script "${MLIR_CODE_ROOT_DIR}/test/e2e/npu/support/${test_name}.py")
  add_test(NAME ${test_name} COMMAND ${Python3_EXECUTABLE} ${test_script})
  set_tests_properties(${test_name} PROPERTIES LABELS "mlir_ut;npu_runner_offline")
  add_custom_target(check-${test_name}
    COMMAND ${Python3_EXECUTABLE} ${test_script}
    COMMENT "Running ${test_name} without an NPU"
  )
  add_dependencies(check-npu-runner-offline check-${test_name})
endforeach()

add_test(NAME LegacyInductorTransposePython
  COMMAND ${Python3_EXECUTABLE}
    ${MLIR_CODE_ROOT_DIR}/test/ut/python/test_legacy_inductor_transpose.py
    --extension $<TARGET_FILE:AutofuseMlirPythonBinding>
    --python-source ${CODE_ROOT_DIR}/compiler/python
)
set_tests_properties(LegacyInductorTransposePython PROPERTIES
  LABELS "mlir_ut;inductor_transpose"
  ENVIRONMENT "LD_LIBRARY_PATH=${AF_AUTOFUSE_MLIR_TEST_LIBRARY_PATH}:$ENV{LD_LIBRARY_PATH};LD_PRELOAD=$<TARGET_FILE:${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET}>"
)
add_custom_target(check-legacy-inductor-transpose-python
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "^LegacyInductorTransposePython$"
  DEPENDS AutofuseMlirPythonBinding ${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET}
)
set_property(GLOBAL APPEND PROPERTY AFIR_GTEST_CHECK_TARGETS_ut check-legacy-inductor-transpose-python)
set_property(GLOBAL APPEND PROPERTY AFIR_GTEST_CHECK_TARGETS_ut check-npu-runner-offline)

add_test(NAME RegBaseSdkPreparation
  COMMAND ${Python3_EXECUTABLE} ${MLIR_CODE_ROOT_DIR}/test/ut/python/test_reg_base_sdk.py)
set_tests_properties(RegBaseSdkPreparation PROPERTIES LABELS "mlir_ut;cpu_sdk")
add_custom_target(check-reg-base-sdk-preparation
  COMMAND ${Python3_EXECUTABLE} ${MLIR_CODE_ROOT_DIR}/test/ut/python/test_reg_base_sdk.py)
set_property(GLOBAL APPEND PROPERTY AFIR_GTEST_CHECK_TARGETS_ut check-reg-base-sdk-preparation)

# Collect the check-* drivers registered by add_afir_gtest() in each
# subdirectory above.
get_property(AF_UT_CHECK_TARGETS GLOBAL PROPERTY AFIR_GTEST_CHECK_TARGETS_ut)

add_custom_target(check-ut
  DEPENDS ${AF_UT_CHECK_TARGETS}
  COMMENT "Running MLIR unit tests"
)
