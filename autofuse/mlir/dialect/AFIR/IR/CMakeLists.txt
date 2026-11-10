# AFIR dialect IR.

# --- TableGen: dialect / enums / attrs / ops -------------------------------
# tblgen outputs into the AFIR/ subdir; the Makefile generator (full autofuse
# build) does not auto-create it, unlike Ninja (standalone mlir/ build).
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/AFIR/IR)

set(LLVM_TARGET_DEFINITIONS AFIRDialect.td)
mlir_tablegen(AFIR/IR/AFIRDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AFIR/IR/AFIRDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AFIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS AFIREnums.td)
mlir_tablegen(AFIR/IR/AFIREnums.h.inc -gen-enum-decls)
mlir_tablegen(AFIR/IR/AFIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(AFIREnumsIncGen)
add_dependencies(AFIREnumsIncGen AFIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS AFIRTypes.td)
mlir_tablegen(AFIR/IR/AFIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AFIR/IR/AFIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(AFIRTypesIncGen)
add_dependencies(AFIRTypesIncGen AFIRDialectIncGen AFIREnumsIncGen)

set(LLVM_TARGET_DEFINITIONS AFIRAttrs.td)
mlir_tablegen(AFIR/IR/AFIRAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AFIR/IR/AFIRAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AFIRAttrsIncGen)
add_dependencies(AFIRAttrsIncGen AFIRDialectIncGen AFIREnumsIncGen)

set(LLVM_TARGET_DEFINITIONS AFIRInterfaces.td)
mlir_tablegen(AFIR/IR/AFIRInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AFIR/IR/AFIRInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AFIRInterfacesIncGen)
add_dependencies(AFIRInterfacesIncGen AFIRDialectIncGen AFIREnumsIncGen AFIRAttrsIncGen)

set(AFIR_SYMBOLIC_OP_DEFINITION ${CMAKE_CURRENT_SOURCE_DIR}/Ops/SymbolicEvalOp.td)
set(AFIR_EXECUTION_OP_DEFINITION ${CMAKE_CURRENT_SOURCE_DIR}/Ops/Execution.td)
set(AFIR_SYNCHRONIZATION_OP_DEFINITION ${CMAKE_CURRENT_SOURCE_DIR}/Ops/Synchronization.td)
set(AFIR_VECTOR_FUNC_OP_DEFINITION ${CMAKE_CURRENT_SOURCE_DIR}/Ops/VectorFunc.td)
set(AFIR_TILING_OP_DEFINITION ${CMAKE_CURRENT_SOURCE_DIR}/Ops/Tiling.td)

set(LLVM_TARGET_DEFINITIONS AFIROps.td)
mlir_tablegen(AFIR/IR/AFIROps.h.inc -gen-op-decls DEPENDS ${AFIR_SYMBOLIC_OP_DEFINITION} ${AFIR_EXECUTION_OP_DEFINITION} ${AFIR_SYNCHRONIZATION_OP_DEFINITION} ${AFIR_VECTOR_FUNC_OP_DEFINITION} ${AFIR_TILING_OP_DEFINITION})
mlir_tablegen(AFIR/IR/AFIROps.cpp.inc -gen-op-defs DEPENDS ${AFIR_SYMBOLIC_OP_DEFINITION} ${AFIR_EXECUTION_OP_DEFINITION} ${AFIR_SYNCHRONIZATION_OP_DEFINITION} ${AFIR_VECTOR_FUNC_OP_DEFINITION} ${AFIR_TILING_OP_DEFINITION})
add_public_tablegen_target(AFIROpsIncGen)
add_dependencies(AFIROpsIncGen AFIRDialectIncGen AFIRAttrsIncGen AFIREnumsIncGen AFIRTypesIncGen)

# --- Op reference 文档 ------------------------------------------------------
# 从 ODS summary / description 生成公开 Op reference，避免手写文档与定义不同步
# （设计文档 5.5）。输出为 ${MLIR_BINARY_DIR}/docs/Dialects/AFIR.md。
# 构建：cmake --build build --target AFIRDocGen
# 公开档位的文档完整性由 tools/af-mlir-tools/check_afir_ods_docs.py 检查。
add_mlir_doc(AFIROps AFIR Dialects/ -gen-dialect-doc -dialect afir)
add_dependencies(AFIRDocGen AFIRDialectIncGen AFIRAttrsIncGen AFIREnumsIncGen AFIRTypesIncGen AFIRInterfacesIncGen)

# --- Library ---------------------------------------------------------------
add_mlir_dialect_library(AFIRDialect
  AFIRDialect.cpp
  AFIRInterfaces.cpp
  AFIRVerifierUtils.cpp
  PlanMetadata.cpp
  SymbolicBuilder.cpp
  Ops/Elementwise.cpp
  Ops/Data.cpp
  Ops/Broadcast.cpp
  Ops/Cast.cpp
  Ops/Condition.cpp
  Ops/Concat.cpp
  Ops/Reduce.cpp
  Ops/SymbolicEvalOp.cpp
  Ops/Synchronization.cpp
  Ops/VectorFunc.cpp
  Ops/Tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}

  DEPENDS
  AFIRDialectIncGen
  AFIREnumsIncGen
  AFIRTypesIncGen
  AFIRAttrsIncGen
  AFIRInterfacesIncGen
  AFIROpsIncGen

  LINK_LIBS PUBLIC
  AFIRSymbolicSupport
  MLIRArithDialect
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  # SymbolicBuilder.cpp 创建 arith.constant，需要显式链接其方言库。
  MLIRArithDialect
)

# Expose dirs so headers resolve as "dialect/AFIR/IR/AFIR.h" and generated includes.
target_include_directories(AFIRDialect PUBLIC
  ${MLIR_CODE_ROOT_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
