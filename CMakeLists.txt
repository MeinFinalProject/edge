cmake_minimum_required(VERSION 3.21)

project(face_runtime LANGUAGES CXX)

add_executable(face_runtime src/main.cpp)

target_compile_features(face_runtime PRIVATE cxx_std_20)

target_include_directories(face_runtime PRIVATE 
"${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_definitions(face_runtime PRIVATE 
    WIN32_LEAN_AND_MEAN 
    NOMINMAX
    UNICODE
    _UNICODE
)