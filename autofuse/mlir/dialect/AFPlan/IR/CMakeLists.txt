# afplan dialect IR: attributes only (no ops, no types).

# tblgen outputs into the AFPlan/ subdir; the Makefile generator (full autofuse
# build) does not auto-create it, unlike Ninja (standalone mlir/ build).
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/AFPlan/IR)

set(LLVM_TARGET_DEFINITIONS AFPlanDialect.td)
mlir_tablegen(AFPlan/IR/AFPlanDialect.h.inc -gen-dialect-decls -dialect=afplan)
mlir_tablegen(AFPlan/IR/AFPlanDialect.cpp.inc -gen-dialect-defs -dialect=afplan)
add_public_tablegen_target(AFPlanDialectIncGen)

set(LLVM_TARGET_DEFINITIONS AFPlanAttrs.td)
mlir_tablegen(AFPlan/IR/AFPlanAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=afplan)
mlir_tablegen(AFPlan/IR/AFPlanAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=afplan)
add_public_tablegen_target(AFPlanAttrsIncGen)
add_dependencies(AFPlanAttrsIncGen AFPlanDialectIncGen)

# 从方言定义生成 AFPlan 属性参考文档（设计文档 5.5），不手工维护。
# afplan 为内部命名空间，不纳入公开 Op reference 门禁。
# 构建：cmake --build build --target AFPlanDocGen
add_mlir_doc(AFPlanAttrs AFPlan Dialects/ -gen-dialect-doc -dialect afplan)
add_dependencies(AFPlanDocGen AFPlanDialectIncGen AFPlanAttrsIncGen)

add_mlir_dialect_library(AFPlanDialect
  AFPlanDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}

  DEPENDS
  AFPlanDialectIncGen
  AFPlanAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

# Expose dirs so headers resolve as "dialect/AFPlan/IR/AFPlan.h" and generated includes.
target_include_directories(AFPlanDialect PUBLIC
  ${MLIR_CODE_ROOT_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
