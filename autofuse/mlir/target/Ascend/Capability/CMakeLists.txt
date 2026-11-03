add_mlir_library(AscendInstructionCapability
  Capability.cpp
  CapabilityQuery.cpp
  ImplementationRegistry.cpp
  ImplementationSelection.cpp
)

target_include_directories(AscendInstructionCapability PUBLIC
  ${MLIR_CODE_ROOT_DIR}
)

target_link_libraries(AscendInstructionCapability PRIVATE
  AscendTargetProfile
  MLIRIR
  MLIRPass
  MLIRSupport
)
