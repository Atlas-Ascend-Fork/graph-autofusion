# ScheduledResult -> AFIR bridge (autofuse MLIR migration).
#
# Two layers:
#   AfirBuilder     backend-neutral: plain structs -> AFIR module (built here)
#   ScheduleToAfir  in-process adapter: real ScheduledResult -> those structs.
#                   Compiled directly into the codegen lib (see codegen/CMakeLists.txt),
#                   not built as a standalone library here — it needs autofuse
#                   backend headers only available in the full build.
#
# HostTiling 曾经也在这里，现在住 artifact/：它的产物是 host 侧文本而不是 IR，与
# artifact/AfirKernelSource 是同一层的两半（见 artifact/CMakeLists.txt）。留在 bridge/ 的
# 判据是「与外部世界交换数据」，HostTiling 做的是「产出交付物」，两回事。

# --- AfirBuilder core: neutral schedule info -> AFIR module ----------------
add_mlir_library(AfirBuilder
  AfirBuilder/AfirBuilder.cpp
  AfirBuilder/WorkspaceViews.cpp

  LINK_LIBS PUBLIC
  # 归约域的来源证明（`afir::annotateReductionDomains`）住在 AFIR 分析层：bridge 建完模块
  # 之后调它一次。此前 bridge 只用 Analysis 的头（`ResourceIds.h` 全是 inline），没有符号
  # 依赖，所以这条曾经不需要——漏掉它的表现是 `af-afir-gen` 链接期 undefined reference，
  # 而它只在全量构建里编，单独编 af-opt / 各 e2e 生成器都看不见。
  AFIRAnalysis
  AFIRDialect
  AFPlanDialect
  MLIRArithDialect
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  MLIRTensorDialect
)
target_include_directories(AfirBuilder PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
