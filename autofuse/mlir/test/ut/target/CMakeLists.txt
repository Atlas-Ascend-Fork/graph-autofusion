add_afir_gtest(AscendTargetProfileTest
  SOURCES
    AscendTargetProfileTest.cpp
    TargetSelectionTest.cpp
    TargetProfileLoaderTest.cpp
    TargetProfileRegistryTest.cpp
    CapabilityTest.cpp
    ImplementationRegistryTest.cpp
    ImplementationSelectionTest.cpp
  LIBS
    AFIRDialect
    AscendInstructionCapability
    AscendTargetProfile
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRTransformUtils
  COMMENT "Running Ascend target profile unit tests"
)

add_test(NAME BuiltinTargetProfilesGeneratorTest
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/BuiltinTargetProfilesGeneratorTest.py
)

add_custom_target(check-builtin-target-profiles-generator
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/BuiltinTargetProfilesGeneratorTest.py
  COMMENT "Running builtin target profiles generator tests"
)
set_property(GLOBAL APPEND PROPERTY
  AFIR_GTEST_CHECK_TARGETS_ut check-builtin-target-profiles-generator)
