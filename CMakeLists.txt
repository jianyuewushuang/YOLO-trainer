# C++ YOLO 训练工具（与 cpp_infer/CMakeLists.txt 同风格）
#
# Windows + MinGW 构建：
#   cmake -B build-cmake -S . -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-cmake
# 产物：build-cmake/yolo_train.exe
#
# 依赖：仅 C++17 标准库，无需第三方库；训练引擎为项目内 Python 脚本。
cmake_minimum_required(VERSION 3.15)
project(yolo_train LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(yolo_train ${SOURCES})

if (WIN32)
    target_compile_definitions(yolo_train PRIVATE WIN32_LEAN_AND_MEAN)
endif()
if (MINGW)
    # 静态链接运行时，exe 开箱即用
    target_link_options(yolo_train PRIVATE -static -static-libgcc -static-libstdc++)
endif()
