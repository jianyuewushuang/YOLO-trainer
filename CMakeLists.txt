# C++ YOLO 训练工具（与 cpp_infer/CMakeLists.txt 同风格）
#
# Linux / macOS：
#   cmake -B build-cmake -S . -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cmake -j
#   产物：build-cmake/yolo_train
#
# Windows + MinGW：
#   cmake -B build-cmake -S . -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cmake
#   产物：build-cmake/yolo_train.exe
#
# 依赖：仅 C++17 标准库，无需第三方库；训练引擎为项目内 Python 脚本。
cmake_minimum_required(VERSION 3.15)
project(yolo_train VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(yolo_train ${SOURCES})

if (MSVC)
    target_compile_options(yolo_train PRIVATE /W4)
else()
    target_compile_options(yolo_train PRIVATE -Wall -Wextra)
endif()

if (WIN32)
    target_compile_definitions(yolo_train PRIVATE WIN32_LEAN_AND_MEAN)
endif()
if (MINGW)
    # 静态链接运行时，exe 开箱即用
    target_link_options(yolo_train PRIVATE -static -static-libgcc -static-libstdc++)
endif()
