add_afir_gtest(ScheduleToAfirTest
  SOURCES
    ScheduleToAfirTest.cpp
    BridgeOperationRegistryTest.cpp
    IsolatedPlanEntryTest.cpp
    AutofuseLaunchEmitterTest.cpp
    AggregateLaunchDriverTest.cpp
    CodegenOrchestrationTest.cpp
    ${CODE_ROOT_DIR}/tests/framework/easy_asc_graph/asc_graph_builder.cpp
  LIBS
    aihac_codegen
    metadef
    MLIRIR
    MLIRLLVMDialect
    AFIRToAscendC
    AfirKernelSource
    AfirTilingSource
    ${AF_AUTOFUSE_MLIR_RUNTIME_STUB_TARGET}
    ascendc_api_headers
  COMMENT "Running ScheduleToAfir bridge unit tests"
)

target_include_directories(ScheduleToAfirTest PRIVATE
  ${MLIR_CODE_ROOT_DIR}
  ${CODE_ROOT_DIR}/tests
  ${MLIR_CODE_ROOT_DIR}/test/ut/schedule
  ${CODE_ROOT_DIR}/codegen
  ${CODE_ROOT_DIR}/optimize
)
