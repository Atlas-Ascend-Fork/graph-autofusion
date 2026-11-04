add_library(AFIRSymbolicSupport STATIC
  SymbolicExpression.cpp
)

target_include_directories(AFIRSymbolicSupport PUBLIC
  ${MLIR_CODE_ROOT_DIR}
)

target_link_libraries(AFIRSymbolicSupport PRIVATE
  Boost::boost
  symengine
)

set_target_properties(AFIRSymbolicSupport PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
