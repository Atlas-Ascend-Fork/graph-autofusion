# 装配已下降的 plan modules，生成供外部编译器使用的文本，而非继续转换 IR。
# 按产物分为 KernelSource（kernel、TilingData header、AutofuseLaunch）和
# TilingSource（Inductor host tiling ABI）。两者接收 ArrayRef<ModuleOp>，返回 std::string，
# 由 CodegenOrchestration::FinalizeMlirCodegenResult 依次调用。
# 本层位于 conversion 下游，conversion 不依赖本层。
add_subdirectory(KernelSource)

# --- TilingSource: lowered Plan modules -> Inductor host tiling sources ----
add_mlir_library(AfirTilingSource
  TilingSource/TilingProgramSpec.cpp
  TilingSource/TilingFuncEmitter.cpp

  LINK_LIBS PUBLIC
  AFIRDialect
  AFPlanDialect
  AFIRToAscendC
  AfirKernelSource
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
target_include_directories(AfirTilingSource PUBLIC
  ${MLIR_CODE_ROOT_DIR}
)
