add_mlir_library(AfirKernelSource
  Passes.cpp
  GenAggregateTilingData.cpp
  GenAggregateKernelEntry.cpp
  GenAggregateKernelBody.cpp
  GenAggregatePass.cpp
  AggregateLaunchDriver.cpp
  AddKernelIncludes.cpp
  Emit/ApiExtensionInliner.cpp
  Emit/ApiExtensionOrder.cpp
  Emit/TilingDataHeaderEmitter.cpp
  Emit/AutofuseLaunchEmitter.cpp

  LINK_LIBS PUBLIC
  AFIRDialect
  AFPlanDialect
  AFIRToAscendC
  MLIRAsc
  MLIREmitAsc
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRPass
  MLIRTargetAsc
)

target_include_directories(AfirKernelSource PUBLIC
  ${MLIR_CODE_ROOT_DIR}
)

# `ascir/...` 的头文件目录不经 LINK_LIBS 传播，本库直接 include 它们（Asc 方言、
# EmitAsc、Target/Asc 的 translation），通过统一的 headers 目标引入。
target_link_libraries(AfirKernelSource PRIVATE
  ascendc_api_headers
)
