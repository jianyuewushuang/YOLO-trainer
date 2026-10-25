#!/usr/bin/env bash
# ============================================================================
# build.sh —— 用系统 C++ 编译器编译 C++ YOLO 训练工具（Linux / macOS）
#
# 本程序不依赖任何第三方 C++ 库（纯 C++17 标准库，通过管道调用 Python 训练
# 引擎），因此无需额外链接选项。
#
# 用法（在任意目录执行均可，亦可通过 PATH 调用）：
#   ./build.sh               # 用 g++/c++ 编译，产物 build/yolo_train
#   CXX=clang++ ./build.sh   # 指定编译器
# 产物：
#   build/yolo_train
# ============================================================================
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 允许用 CXX 环境变量覆盖编译器，默认 g++（退化到 c++）
if [[ -n "${CXX:-}" ]] && command -v "$CXX" >/dev/null 2>&1; then
    cxx="$CXX"
elif command -v g++ >/dev/null 2>&1; then
    cxx="g++"
elif command -v c++ >/dev/null 2>&1; then
    cxx="c++"
elif command -v clang++ >/dev/null 2>&1; then
    cxx="clang++"
else
    echo "未找到 C++ 编译器，请安装: sudo apt install build-essential" >&2
    exit 3
fi

shopt -s nullglob
srcs=("$root"/src/*.cpp)
shopt -u nullglob
if [[ ${#srcs[@]} -eq 0 ]]; then
    echo "src/ 下没有 .cpp 源文件" >&2
    exit 3
fi

mkdir -p "$root/build"
out="$root/build/yolo_train"

echo "编译: $cxx -> $out"
"$cxx" -std=c++17 -O2 -DNDEBUG -Wall -Wextra \
    "${srcs[@]}" -o "$out"

echo "BUILD OK -> build/yolo_train"
echo "运行 ./build/yolo_train doctor 自检环境"
