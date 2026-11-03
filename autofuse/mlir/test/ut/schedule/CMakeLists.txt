add_afir_gtest(ScheduleSpaceBridgeTest
  SOURCES
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/ScheduleSpaceBridgeTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/FingerprintTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/TilingFieldSchemaTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/PlanSelectorTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/PlanSelectionTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/AttCostEvaluatorTest.cpp
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/AttSchemaAdapterTest.cpp
  LIBS
    AutofuseMlirScheduleBridge
    AutofuseMlirScheduleLegacyAdapter
    aihac_codegen
    metadef
  COMMENT "Running schedule space bridge unit tests"
)

target_include_directories(ScheduleSpaceBridgeTest PRIVATE
  ${MLIR_CODE_ROOT_DIR}
  ${CODE_ROOT_DIR}/att
  ${CODE_ROOT_DIR}/att/base
  ${CODE_ROOT_DIR}/att/gen_model_info
)

# Isolate the ATT model-generation stub from tests that link against real ATT.
add_afir_gtest(PlanSelectorPolicyTest
  SOURCES
    ${MLIR_CODE_ROOT_DIR}/test/ut/schedule/PlanSelectorPolicyTest.cpp
    ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph/asc_graph_builder.cpp
  LIBS
    AutofuseMlirScheduleBridge
    AutofuseMlirScheduleLegacyAdapter
    aihac_codegen
    metadef
  COMMENT "Running plan selector policy unit tests"
)

target_include_directories(PlanSelectorPolicyTest PRIVATE
  ${MLIR_CODE_ROOT_DIR}
  ${CODE_ROOT_DIR}/tests
  ${CODE_ROOT_DIR}/att
  ${CODE_ROOT_DIR}/att/base
  ${CODE_ROOT_DIR}/att/gen_model_info
)
