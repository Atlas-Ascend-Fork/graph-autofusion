# Reserved for backend-neutral schedule metadata.  The current schema and
# fingerprint implementation still consumes legacy schedule types and is
# compiled by AutofuseMlirScheduleLegacyAdapter instead.
add_library(AutofuseMlirScheduleCommon INTERFACE)

target_include_directories(AutofuseMlirScheduleCommon INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CODE_ROOT_DIR}
)
