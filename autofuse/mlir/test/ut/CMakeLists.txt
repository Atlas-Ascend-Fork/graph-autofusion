add_subdirectory(artifact)
add_subdirectory(symbolic)
add_subdirectory(bridge)
add_subdirectory(target)
add_subdirectory(conversion)
add_subdirectory(dialect)
add_subdirectory(schedule)

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
set_property(GLOBAL APPEND PROPERTY AFIR_GTEST_CHECK_TARGETS_ut check-npu-runner-offline)

# Collect the check-* drivers registered by add_afir_gtest() in each
# subdirectory above.
get_property(AF_UT_CHECK_TARGETS GLOBAL PROPERTY AFIR_GTEST_CHECK_TARGETS_ut)

add_custom_target(check-ut
  DEPENDS ${AF_UT_CHECK_TARGETS}
  COMMENT "Running MLIR unit tests"
)
