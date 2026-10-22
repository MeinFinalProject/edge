cmake_minimum_required(VERSION 3.21)

project(FaceRuntime LANGUAGES CXX)

find_program(FXC_EXECUTABLE fxc REQUIRED)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

set(SHADER_HEADERS
    ${CMAKE_CURRENT_BINARY_DIR}/generated/scrfd_tensorize.h
    ${CMAKE_CURRENT_BINARY_DIR}/generated/pad_tensorize.h
    ${CMAKE_CURRENT_BINARY_DIR}/generated/arcface_tensorize.h
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/scrfd_tensorize.h
    COMMAND ${FXC_EXECUTABLE} /nologo /T cs_5_0 /E main /O3
            /I ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess
            /Fh ${CMAKE_CURRENT_BINARY_DIR}/generated/scrfd_tensorize.h
            /Vn g_scrfd_tensorize
            ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/scrfd_tensorize.hlsl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/scrfd_tensorize.hlsl
            ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/nv12_helpers.hlsl
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/pad_tensorize.h
    COMMAND ${FXC_EXECUTABLE} /nologo /T cs_5_0 /E main /O3
            /I ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess
            /Fh ${CMAKE_CURRENT_BINARY_DIR}/generated/pad_tensorize.h
            /Vn g_pad_tensorize
            ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/pad_tensorize.hlsl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/pad_tensorize.hlsl
            ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/nv12_helpers.hlsl
    VERBATIM
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/arcface_tensorize.h
    COMMAND ${FXC_EXECUTABLE} /nologo /T cs_5_0 /E main /O3
            /I ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess
            /Fh ${CMAKE_CURRENT_BINARY_DIR}/generated/arcface_tensorize.h
            /Vn g_arcface_tensorize
            ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/arcface_tensorize.hlsl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/arcface_tensorize.hlsl
            ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/nv12_helpers.hlsl
    VERBATIM
)

add_executable(FaceRuntime 
    src/main.cpp
    src/camera/camera.cpp
    src/gpu/gpu_context.cpp
    src/gpu/camera_texture_import.cpp
    src/preprocess/similarity_transform.cpp
    src/preprocess/preprocess_contracts.cpp
)

target_compile_features(FaceRuntime PRIVATE cxx_std_20)

target_link_libraries(FaceRuntime PRIVATE
    d3d11.lib
    d3d12.lib
    dxgi.lib
)

add_custom_target(preprocess_shaders DEPENDS ${SHADER_HEADERS})
add_dependencies(FaceRuntime preprocess_shaders)
target_sources(FaceRuntime PRIVATE ${SHADER_HEADERS})

target_include_directories(FaceRuntime PRIVATE 
"${CMAKE_CURRENT_SOURCE_DIR}/src"
"${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_compile_definitions(FaceRuntime PRIVATE 
    WIN32_LEAN_AND_MEAN 
    NOMINMAX
    UNICODE
    _UNICODE
)