#!/usr/bin/env bash
# ============================================================================
# setup.sh —— 一键环境搭建（Linux / macOS）
#
# 步骤：
#   1. 自动挑选合适的 Python（>=3.10，优先 3.11/3.12）
#   2. 在项目内 .venv 创建独立虚拟环境
#   3. 安装训练引擎依赖（engine/requirements.txt）
#   4. 预下载三种模型的预训练权重（yolov8n / yolo11n / yolo26n）
#   5. 编译 C++ 训练工具（build.sh）
#
# 用法：
#   ./setup.sh                     # 完整搭建
#   ./setup.sh --skip-build        # 只搭 Python 环境
#   ./setup.sh --python /usr/bin/python3.12
#   ./setup.sh --cpu               # 无 NVIDIA GPU 时安装 CPU 版 torch（体积小很多）
#   ./setup.sh --force             # 强制重建虚拟环境
#
# 说明：所有产物（.venv / .ultralytics_cfg / weights / logs / runs）都在项目内，
#       不会修改项目外部文件。
# ============================================================================
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$root/.venv"
venv_py="$venv/bin/python"
python_bin=""
skip_build=0
force=0
cpu_torch=0

usage() {
    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)      python_bin="${2:-}"; shift 2 ;;
        --python=*)    python_bin="${1#*=}"; shift ;;
        --skip-build)  skip_build=1; shift ;;
        --force)       force=1; shift ;;
        --cpu)         cpu_torch=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "未知参数: $1（-h 查看帮助）" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. 挑选 Python 解释器（>=3.10；优先 3.11/3.12，与 ultralytics/torch 实测兼容）
# ---------------------------------------------------------------------------
py_ok() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

if [[ -z "$python_bin" && -n "${YOLO_TRAIN_PYTHON:-}" ]]; then
    python_bin="$YOLO_TRAIN_PYTHON"
fi

if [[ -n "$python_bin" ]]; then
    if ! command -v "$python_bin" >/dev/null 2>&1; then
        echo "指定的 Python 不可执行: $python_bin" >&2
        exit 3
    fi
    if ! py_ok "$python_bin"; then
        echo "指定的 Python 版本过低（需 >=3.10）: $("$python_bin" --version 2>&1)" >&2
        exit 3
    fi
else
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then
            python_bin="$(command -v "$cand")"
            break
        fi
    done
fi

if [[ -z "$python_bin" ]]; then
    echo "未找到合适的 Python（需 >=3.10）。请安装后用 --python 指定。" >&2
    echo "Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip" >&2
    exit 3
fi

echo "使用 Python: $python_bin"
"$python_bin" --version

if ! "$python_bin" -c 'import venv' >/dev/null 2>&1; then
    echo "该 Python 缺少 venv 模块。Ubuntu/Debian: sudo apt install python3-venv" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 2. 创建虚拟环境
# ---------------------------------------------------------------------------
if [[ -d "$venv" && $force -eq 1 ]]; then
    echo "强制重建虚拟环境 ..."
    rm -rf "$venv"
fi
if [[ ! -x "$venv_py" ]]; then
    echo "创建虚拟环境: $venv"
    "$python_bin" -m venv "$venv"
else
    echo "虚拟环境已存在: $venv"
fi

# ---------------------------------------------------------------------------
# 3. 安装依赖
# ---------------------------------------------------------------------------
req="$root/engine/requirements.txt"
echo "安装训练引擎依赖（engine/requirements.txt）..."
"$venv_py" -m pip install --upgrade pip || echo "pip 自升级失败，继续..."

if [[ $cpu_torch -eq 1 ]]; then
    torch_ver="$(sed -n 's/^torch==\([^[:space:];#]*\).*/\1/p' "$req" | head -n1)"
    tv_ver="$(sed -n 's/^torchvision==\([^[:space:];#]*\).*/\1/p' "$req" | head -n1)"
    echo "安装 CPU 版 torch（${torch_ver:-latest} / torchvision ${tv_ver:-latest}）..."
    "$venv_py" -m pip install \
        ${torch_ver:+"torch==$torch_ver"} \
        ${tv_ver:+"torchvision==$tv_ver"} \
        --index-url https://download.pytorch.org/whl/cpu
fi

"$venv_py" -m pip install -r "$req"

# ---------------------------------------------------------------------------
# 4. 预下载预训练权重（失败仅警告：训练时也会自动重试下载）
# ---------------------------------------------------------------------------
echo "预下载预训练权重（yolov8n / yolo11n / yolo26n）..."
if ! "$venv_py" -X utf8 "$root/src/train_worker.py" download; then
    echo "警告：预训练权重预下载失败（可能是网络问题）。首次训练时会自动重试下载。"
fi

# ---------------------------------------------------------------------------
# 5. 编译 C++ 训练工具
# ---------------------------------------------------------------------------
if [[ $skip_build -eq 0 ]]; then
    echo "编译 C++ 训练工具 ..."
    bash "$root/build.sh"
fi

cat <<EOF

================ 环境搭建完成 ================
下一步：
  1. 自检环境:   ./build/yolo_train doctor
  2. 开始训练:   ./build/yolo_train
                （交互菜单选择模型；或 train --model yolov8n）
EOF
