cmake_minimum_required(VERSION 3.21)

project(FaceRuntime LANGUAGES CXX)

find_program(FXC_EXECUTABLE fxc REQUIRED)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

# Sumber TUNGGAL ukuran thread group: diteruskan ke fxc (/D GROUP) dan ke C++ (FACE_GROUP_SIZE).
set(GROUP_SIZE 16)

# ---------------------------------------------------------------------------
# 1) HLSL -> header C++ (array byte) dengan fxc.exe milik Windows SDK
#    add_hlsl_header(warp_tensorize) -> generated/warp_tensorize.h berisi `const BYTE g_warp_tensorize[]`
# ---------------------------------------------------------------------------
function(add_hlsl_header name)
    set(out ${CMAKE_CURRENT_BINARY_DIR}/generated/${name}.h)
    add_custom_command(
        OUTPUT ${out}
        COMMAND ${FXC_EXECUTABLE} /nologo /T cs_5_0 /E main /O3 /D GROUP=${GROUP_SIZE}
                /I ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess
                /Fh ${out} /Vn g_${name}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/${name}.hlsl
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/${name}.hlsl
                ${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/nv12_helpers.hlsl
        VERBATIM
    )
    set(SHADER_HEADERS ${SHADER_HEADERS} ${out} PARENT_SCOPE)
endfunction()

add_hlsl_header(warp_tensorize)

# ---------------------------------------------------------------------------
# 2) face_core: semua kode kecuali main dan tools (dipakai bersama oleh semua program)
# ---------------------------------------------------------------------------
add_library(face_core STATIC
    src/camera/camera.cpp
    src/gpu/nv12_bridge.cpp
    src/gpu/gpu_context.cpp
    src/gpu/command_recorder.cpp
    src/gpu/compute_kernel.cpp
    src/gpu/tensor_buffer.cpp
    src/inference/onnx_model.cpp
    src/preprocess/similarity_transform.cpp
    src/preprocess/warp.cpp
    src/models/scrfd.cpp
    src/models/liveness.cpp
    src/models/arcface.cpp
    src/tracking/byte_tracker.cpp
    src/pipeline/pipeline.cpp
)

target_compile_features(face_core PUBLIC cxx_std_20)

target_link_libraries(face_core PUBLIC
    d3d11.lib
    d3d12.lib
    dxgi.lib
    strmiids
    ole32
    oleaut32
)

add_custom_target(preprocess_shaders DEPENDS ${SHADER_HEADERS})
add_dependencies(face_core preprocess_shaders)
target_sources(face_core PRIVATE ${SHADER_HEADERS})

target_include_directories(face_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_compile_definitions(face_core PUBLIC
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    UNICODE
    _UNICODE
    FACE_GROUP_SIZE=${GROUP_SIZE}
)

# ---------------------------------------------------------------------------
# 3) Program: FaceRuntime (utama) dan sprt_tool (alat kalibrasi liveness, offline)
# ---------------------------------------------------------------------------
add_executable(FaceRuntime src/main.cpp)
target_link_libraries(FaceRuntime PRIVATE face_core)

# Salin folder models/ ke samping FaceRuntime.exe setiap build.
add_custom_command(TARGET FaceRuntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/models"
            "$<TARGET_FILE_DIR:FaceRuntime>/models"
    COMMENT "Mengopi folder models ke direktori output..."
)

add_executable(sprt_tool tools/sprt_tool.cpp)
target_link_libraries(sprt_tool PRIVATE face_core)
