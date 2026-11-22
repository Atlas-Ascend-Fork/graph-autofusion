add_afir_gtest(AFIRSymbolicSupportTest
  SOURCES ExpressionTest.cpp
  LIBS AFIRSymbolicSupport
  COMMENT "Running AFIR symbolic support unit tests"
)

add_afir_gtest(AFIRSymbolicTest
  SOURCES AFIRSymbolicTest.cpp
  LIBS AFIRTransforms MLIRFuncDialect
  COMMENT "Running AFIR symbolic materializer unit tests"
)
