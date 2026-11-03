add_mlir_library(AFIRAnalysis
  BufferAlias.cpp
  CanonicalBindings.cpp
  RaceCheck.cpp
  ReduceScratchLifetime.cpp
  ReducePlan.cpp
  ResourceIds.cpp
  TransferPlan.cpp
  VectorSegmentation.cpp

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
)

target_include_directories(AFIRAnalysis PUBLIC
  ${MLIR_CODE_ROOT_DIR}
)
