# ============================================================================
# build.ps1 —— 用 MinGW g++ (MSYS2 UCRT64) 编译 C++ YOLO 训练工具
#
# 参照 cpp_infer/build.ps1 的工程风格。本程序不依赖任何第三方 C++ 库
# （纯 C++17 标准库，通过管道调用 Python 训练引擎），因此无需拷贝 DLL。
#
# 用法（在任意目录执行均可）：
#   powershell -ExecutionPolicy Bypass -File train_project\build.ps1
# 产物：
#   train_project\build\yolo_train.exe
# ============================================================================
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$gpp = "C:\msys64\ucrt64\bin\g++.exe"

if (-not (Test-Path $gpp)) {
    $found = Get-Command g++ -ErrorAction SilentlyContinue
    if ($found) { $gpp = $found.Source }
}
if (-not (Test-Path $gpp)) {
    throw "未找到 MinGW g++（MSYS2 UCRT64）。请安装 MSYS2 与 ucrt64 工具链，例如执行: pacman -S mingw-w64-ucrt-x86_64-gcc"
}

New-Item -ItemType Directory -Force (Join-Path $root "build") | Out-Null

$srcs = Get-ChildItem -Path (Join-Path $root "src") -Filter *.cpp | ForEach-Object { $_.FullName }
if (-not $srcs) { throw "src\ 下没有 .cpp 源文件" }

$flags = @(
    "-std=c++17", "-O2", "-DNDEBUG", "-Wall", "-Wextra",
    "-DWIN32_LEAN_AND_MEAN",
    "-fexec-charset=UTF-8",          # 源码为 UTF-8，中文日志按 UTF-8 输出
    "-static", "-static-libgcc", "-static-libstdc++"  # 静态链接，exe 开箱即用
)

$out = Join-Path $root "build\yolo_train.exe"
& $gpp @flags @srcs -o $out

if ($LASTEXITCODE -ne 0) {
    throw "编译失败（g++ 退出码 $LASTEXITCODE）"
}

"BUILD OK -> train_project\build\yolo_train.exe"
"运行 `powershell -ExecutionPolicy Bypass -File train_project\build\yolo_train.exe doctor` 自检环境"
