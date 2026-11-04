# 交付物装配：收下降完的 plan modules，产出交给外部编译器的文本。
#
# 这是 mlir/ 里唯一不产出 IR 的一层——别处都是 IR 进 IR 出，这里的出口是 std::string。
# 两个成员按「产出哪一份源码」划分，这也是消费方自己的说法（CodegenOrchestration 里的
# `result.kernel` 与 `assembleTilingSources`）：
#
#   KernelSource   kernel 源码、TilingData header、AutofuseLaunch host wrapper
#   TilingSource   Inductor host tiling ABI
#
# 两者的入参同为 `ArrayRef<ModuleOp>`、出参同为 `std::string`，并在
# bridge/CodegenOrchestration 的 `FinalizeMlirCodegenResult` 里被先后调用。此前一个住
# conversion/、一个住 bridge/，那是按调用方而不是按职责归的类。
#
# 严格在 conversion/ 下游：它消费已经下降完的模块，conversion/ 不认识它。
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
