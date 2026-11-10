# ScheduledResult 到 AFIR 的 bridge，分为两层：
# - AfirBuilder：将后端中立的结构体转换为 AFIR，本目录构建对应库。
# - ScheduleToAfir：将真实 ScheduledResult 适配为这些结构体。它依赖全量 Autofuse
#   后端头，由 codegen/CMakeLists.txt 直接编入 codegen 库，不在此单独构建。
# Host tiling 文本装配属于 artifact 层；bridge 只负责与外部表示交换数据。

# --- AfirBuilder core: neutral schedule info -> AFIR module ----------------
add_mlir_library(AfirBuilder
  AfirBuilder/AfirBuilder.cpp
  AfirBuilder/WorkspaceViews.cpp

  LINK_LIBS PUBLIC
  # bridge 建完模块后调用 AFIR 分析层的 annotateReductionDomains，需链接对应实现。
  # 仅包含 ResourceIds.h 等 inline helper 不需要此符号，但 af-afir-gen 会使用它。
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
