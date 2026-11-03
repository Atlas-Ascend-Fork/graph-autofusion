set(ascendc_api_cube_src
        matmul.h
        matmul/matmul_include_headers.h
        matmul/mat_mul_tiling_key.h
        matmul/batch_mat_mul_v3_tiling_key.h
        matmul/mat_mul_pingpong_basic_cmct.h
        matmul/batch_matmul_include_headers.h
        batch_matmul.h
        matmul_dynamic.h
        matmul/mat_mul_tiling_key_dynamic.h
        matmul/batch_mat_mul_v3_tiling_key_dynamic.h
        matmul/mat_mul_pingpong_basic_cmct_dynamic.h
        batch_matmul_dynamic.h
        conv2d.h
        conv2d/conv2d_include_headers.h
        conv2d/conv2d_v2_tilingkey_cv.h
        conv2d/conv_pingpong_basic_atcos.h
)

set(generated_cube_files)

foreach(header ${ascendc_api_cube_src})
    get_filename_component(name_part ${header} NAME_WLE)
    set(str_header "${name_part}_str.h")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND cat ${header}
                | sed "'1i\\R\"===('"
                | sed "'$$a\\)===\"'"
                > ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
        DEPENDS ${header}
    )
    list(APPEND generated_cube_files ${CMAKE_CURRENT_BINARY_DIR}/${str_header})
endforeach()

add_custom_target(generated_cube_files ALL DEPENDS ${generated_cube_files})

add_library(ascendc_api_cube_extend INTERFACE)
target_include_directories(ascendc_api_cube_extend INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(ascendc_api_cube_extend generated_cube_files)
