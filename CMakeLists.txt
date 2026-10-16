cmake_minimum_required(VERSION 3.21)

project(FaceRuntime LANGUAGES CXX)

add_executable(FaceRuntime src/main.cpp
    src/common/frame.hpp
    src/camera/camera.hpp)

target_compile_features(FaceRuntime PRIVATE cxx_std_20)

target_include_directories(FaceRuntime PRIVATE 
"${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_definitions(FaceRuntime PRIVATE 
    WIN32_LEAN_AND_MEAN 
    NOMINMAX
    UNICODE
    _UNICODE
)