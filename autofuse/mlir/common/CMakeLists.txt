# Backend-neutral schedule metadata belongs here. The current schema and
# fingerprint code still uses legacy schedule types, so it is built in
# AutofuseMlirScheduleLegacyAdapter.
add_library(AutofuseMlirScheduleCommon INTERFACE)

target_include_directories(AutofuseMlirScheduleCommon INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CODE_ROOT_DIR}
)
