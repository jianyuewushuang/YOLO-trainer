# ============================================================================
# setup.ps1 —— 一键环境搭建（train_project 自包含）
#
# 步骤：
#   1. 自动挑选合适的 Python 3.10~3.12 解释器
#   2. 在 train_project/.venv 创建独立虚拟环境
#   3. 安装训练引擎依赖（engine/requirements.txt）
#   4. 预下载三种模型的预训练权重（yolov8n / yolo11n / yolo26n）
#   5. 编译 C++ 训练工具（build.ps1）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File train_project\setup.ps1
#   powershell -ExecutionPolicy Bypass -File train_project\setup.ps1 -SkipBuild   # 只搭环境
#   powershell -ExecutionPolicy Bypass -File train_project\setup.ps1 -Python C:\path\to\python.exe
#
# 说明：所有产物（.venv / .ultralytics_cfg / weights / logs / runs）都在
#       train_project 内，不会修改项目外部文件。
# ============================================================================
param(
    [string]$Python = "",
    [switch]$SkipBuild,
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# ---------------------------------------------------------------------------
# 1. 挑选 Python 解释器（3.10 ~ 3.12，与 ultralytics/torch 实测兼容）
# ---------------------------------------------------------------------------
function Test-PyVersion([string]$exe) {
    try {
        $ver = & $exe --version 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $ver) { return $false }
        # 形如 "Python 3.11.9"
        if ($ver -match "Python\s+(\d+)\.(\d+)") {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            return ($maj -eq 3 -and $min -ge 10 -and $min -le 12)
        }
    } catch {}
    return $false
}

if (-not $Python) {
    if ($env:YOLO_TRAIN_PYTHON) { $Python = $env:YOLO_TRAIN_PYTHON }
}
if (-not $Python) {
    $candidates = @(
        "$env:LOCALAPPDATA\Python\pythoncore-3.11-64\python.exe",  # 本机实测可用
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c) -and (Test-PyVersion $c)) { $Python = $c; break } }
}
if (-not $Python) {
    # Python Launcher
    try {
        $py = (& py -3.11 --version 2>$null)
        if ($LASTEXITCODE -eq 0 -and $py) { $Python = "py -3.11" }
    } catch {}
}
if (-not $Python) {
    foreach ($c in @("python", "python3")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd -and (Test-PyVersion $cmd.Source)) { $Python = $cmd.Source; break }
    }
}
if (-not $Python) {
    throw "未找到合适的 Python 3.10~3.12 解释器。请安装 Python 3.11 后用 -Python 参数指定。"
}

"使用 Python: $Python"
& $Python --version

# ---------------------------------------------------------------------------
# 2. 创建虚拟环境
# ---------------------------------------------------------------------------
$venv = Join-Path $root ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"

if ((Test-Path $venvPy) -and $ForceRecreate) {
    "强制重建虚拟环境 ..."
    Remove-Item $venv -Recurse -Force
}
if (-not (Test-Path $venvPy)) {
    "创建虚拟环境: $venv"
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败" }
} else {
    "虚拟环境已存在: $venv"
}

# ---------------------------------------------------------------------------
# 3. 安装依赖
# ---------------------------------------------------------------------------
"安装训练引擎依赖（engine/requirements.txt）..."
& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { "pip 自升级失败，继续..." }
& $venvPy -m pip install -r (Join-Path $root "engine\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

# ---------------------------------------------------------------------------
# 4. 预下载预训练权重（失败仅警告：训练时也会自动重试下载）
# ---------------------------------------------------------------------------
"预下载预训练权重（yolov8n / yolo11n / yolo26n）..."
& $venvPy -X utf8 (Join-Path $root "src\train_worker.py") download
if ($LASTEXITCODE -ne 0) {
    "警告：预训练权重预下载失败（可能是网络问题）。首次训练时会自动重试下载。"
}

# ---------------------------------------------------------------------------
# 5. 编译 C++ 训练工具
# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    "编译 C++ 训练工具 ..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root "build.ps1")
    if ($LASTEXITCODE -ne 0) { throw "编译失败" }
}

""
"================ 环境搭建完成 ================"
"下一步："
"  1. 自检环境:   powershell -ExecutionPolicy Bypass -File train_project\build\yolo_train.exe doctor"
"  2. 开始训练:   powershell -ExecutionPolicy Bypass -File train_project\build\yolo_train.exe"
"                （交互菜单选择模型；或 train --model yolov8n）"
