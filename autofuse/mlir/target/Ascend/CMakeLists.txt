set(AF_BUILTIN_TARGET_PROFILE_MANIFEST
  ${CMAKE_CURRENT_SOURCE_DIR}/Profiles/manifest.txt
)
file(GLOB AF_BUILTIN_TARGET_PROFILE_INIS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/Profiles/platform_config/*.ini
)
set(AF_BUILTIN_TARGET_PROFILES_INC
  ${CMAKE_CURRENT_BINARY_DIR}/BuiltinTargetProfiles.inc
)

add_custom_command(
  OUTPUT ${AF_BUILTIN_TARGET_PROFILES_INC}
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Profiles/generate_builtin_profiles.py
    ${AF_BUILTIN_TARGET_PROFILE_MANIFEST}
    ${CMAKE_CURRENT_SOURCE_DIR}/Profiles/platform_config
    ${AF_BUILTIN_TARGET_PROFILES_INC}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/Profiles/generate_builtin_profiles.py
    ${AF_BUILTIN_TARGET_PROFILE_MANIFEST}
    ${AF_BUILTIN_TARGET_PROFILE_INIS}
  VERBATIM
)
set_source_files_properties(${AF_BUILTIN_TARGET_PROFILES_INC}
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
)

add_mlir_library(AscendTargetProfile
  AscendTargetProfile.cpp
  TargetSelection.cpp
  TargetProfileLoader.cpp
  TargetProfileRegistry.cpp
  ${AF_BUILTIN_TARGET_PROFILES_INC}
)

target_include_directories(AscendTargetProfile PUBLIC
  ${MLIR_CODE_ROOT_DIR}
)
target_include_directories(AscendTargetProfile PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(AscendTargetProfile PRIVATE
  LLVMSupport
  MLIRSupport
)

add_subdirectory(Capability)
