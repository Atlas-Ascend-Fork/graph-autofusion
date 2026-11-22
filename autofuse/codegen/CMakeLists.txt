set(CODEGEN_DIR ${CODE_ROOT_DIR}/codegen)
file(GLOB_RECURSE CODEGEN_SRCS CONFIGURE_DEPENDS "${CODEGEN_DIR}/*.cpp")

target_sources(aihac_codegen PRIVATE
        ${CODEGEN_SRCS}
)

target_include_directories(aihac_codegen PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/common
        ${CODE_ROOT_DIR}/att
        ${TOP_DIR}/runtime/include/external
)
