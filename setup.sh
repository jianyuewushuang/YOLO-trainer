#!/usr/bin/env bash
# ============================================================================
# setup.sh —— 一键环境搭建（Linux / macOS）
#
# 步骤：
#   1. 自动挑选合适的 Python（稳定版 >=3.10，排除 alpha/beta/rc；优先 3.12）
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
# 1. 挑选 Python 解释器
#
# 要求：>=3.10 且为正式发布版（releaselevel == final）。
# 预发布版（alpha/beta/rc）不能用于训练：例如 3.11.0rc1 早于
# sys.get_int_max_str_digits 的引入（3.11.0rc2），torch/ultralytics 训练时会
# 直接抛 AttributeError。
# ---------------------------------------------------------------------------
# 可用则输出空串；否则输出不可用原因
py_reject_reason() {
    "$1" - <<'PY' 2>/dev/null
import sys

v = sys.version_info
if v < (3, 10):
    print(f"{sys.version.split()[0]} 版本过低（需 >=3.10）")
elif v.releaselevel != "final":
    print(f"{sys.version.split()[0]} 是预发布版（需正式版，rc/beta 缺少部分标准库 API）")
PY
}

resolve_path() {
    command -v "$1" 2>/dev/null || true
}

if [[ -z "$python_bin" && -n "${YOLO_TRAIN_PYTHON:-}" ]]; then
    python_bin="$YOLO_TRAIN_PYTHON"
fi

if [[ -n "$python_bin" ]]; then
    # 显式指定的解释器：不可用时直接报错，不静默替换成别的版本
    if [[ -z "$(resolve_path "$python_bin")" ]]; then
        echo "指定的 Python 不可执行: $python_bin" >&2
        exit 3
    fi
    python_bin="$(resolve_path "$python_bin")"
    reason="$(py_reject_reason "$python_bin")"
    if [[ -n "$reason" ]]; then
        echo "指定的 Python 不可用: $python_bin —— $reason" >&2
        echo "请换成正式版 Python（如 python3.12）后重试：./setup.sh --python /usr/bin/python3.12 --force" >&2
        exit 3
    fi
else
    # 3.12 优先（与 ultralytics/torch 实测最稳），3.13/3.14 作为备选
    for cand in python3.12 python3.11 python3.10 python3.13 python3.14 python3; do
        exe="$(resolve_path "$cand")"
        [[ -z "$exe" ]] && continue
        reason="$(py_reject_reason "$exe")"
        if [[ -n "$reason" ]]; then
            echo "跳过 $cand ($exe)：$reason" >&2
            continue
        fi
        python_bin="$exe"
        break
    done
fi

if [[ -z "$python_bin" ]]; then
    echo "未找到可用的正式版 Python（需 >=3.10，且不是 alpha/beta/rc）。" >&2
    echo "安装示例：" >&2
    echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv" >&2
    echo "  conda:         conda create -n yolo python=3.12 && conda activate yolo" >&2
    echo "  uv:            uv python install 3.12 && ./setup.sh --python \"\$(uv python find 3.12)\"" >&2
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
