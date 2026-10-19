# C++ YOLO 训练工具

用C++和python进行YOLOv8, YOLO11n和YOLO26nd的训练，在GPU可用时会优先使用GPU。

## 2. 目录结构

```
<path\to\your\project>/
├── README.md                   # 本文档
├── setup.ps1                   # 一键环境搭建（建 .venv + 装依赖 + 预下载权重 + 编译）
├── build.ps1                   # MinGW g++ 编译脚本（产物 build/yolo_train.exe）
├── CMakeLists.txt              # CMake 构建（备选：cmake -B build-cmake -G "MinGW Makefiles"）
├── engine/
│   └── requirements.txt        # 训练引擎（Python）依赖清单
├── src/                        # 全部源代码
│   ├── main.cpp                # C++ CLI 入口与模式分发（train/export/doctor/download/list-models）
│   ├── cli.hpp / cli.cpp       # 参数解析 + 交互式模型选择菜单
│   ├── device.hpp/.cpp         # GPU 探测（nvidia-smi）
│   ├── worker.hpp/.cpp         # 子进程执行（CreateProcess + 管道，实时捕获引擎输出）
│   ├── logger.hpp/.cpp         # 带时间戳的日志（控制台 + 文件双写）
│   ├── workspace.hpp/.cpp      # 项目路径管理
│   └── train_worker.py         # 训练引擎（Ultralytics）：train / val / export / check / download
├── datasets/                   # 数据集（3000 训练 + 600 验证，12 类装甲板）
│   ├── dataset.yaml            # path 为相对路径，项目可整体迁移
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── .venv/                      # <path\to\your\project> 独立虚拟环境（Python 3.11）
├── .ultralytics_cfg/           # Ultralytics 配置目录（重定向，含预训练权重缓存）
│   ├── weights/                #   yolov8n.pt / yolo11n.pt / yolo26n.pt（官方预训练权重）
│   └── resolved_dataset.yaml   #   运行时生成的绝对路径数据集配置
├── weights/                    # 交付产物：每个模型一组 best.pt + best.onnx
│   ├── YOLOv8n/{best.pt, best.onnx}
│   ├── YOLO11n/{best.pt, best.onnx}
│   └── YOLO26n/{best.pt, best.onnx}
├── logs/                       # 训练日志（每次训练一个目录）
│   └── <模型>_<时间戳>/{train.log, metrics.json}
├── runs/                       # Ultralytics 训练过程产物（曲线、混淆矩阵等）
└── build/                      # 编译产物 yolo_train.exe
```

## 3. 环境要求

| 依赖 | 说明 |
| --- | --- |
| Windows | 开发与实测平台 |
| Python 3.10 ~ 3.12 | `setup.ps1` 会自动挑选（测试时使用 3.11.9） |
| MSYS2 UCRT64 g++ | 编译 C++ 工具（`C:\msys64\ucrt64\bin\g++.exe`） |
| CMake（可选） | 第二种构建方式，`build.ps1` 直接调 g++ 不依赖它 |
| 网络 | 首次搭建需联网：pip 装依赖、GitHub 下载预训练权重 |
| NVIDIA GPU（可选） | 有则自动用 GPU 训练；无则用 CPU |

## 4. 快速开始

在 **PowerShell** 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File <path\to\your\project>\setup.ps1
```

成功后自检并开始训练：

```powershell
# 环境自检（Python/torch/ultralytics/数据集/权重缓存/GPU）
powershell -ExecutionPolicy Bypass -File <path\to\your\project>\build\yolo_train.exe doctor

# 交互式训练：菜单选择模型 → 确认参数 → 一键训练
powershell -ExecutionPolicy Bypass -File <path\to\your\project>\build\yolo_train.exe

# 或命令行直接指定模型
powershell -ExecutionPolicy Bypass -File <path\to\your\project>\build\yolo_train.exe train --model yolo26n
```

### 手动分步（不用 setup.ps1 时）

```powershell
# 1. 建虚拟环境并装依赖（用任意 Python 3.10~3.12）
C:\...\python.exe -m venv <path\to\your\project>\.venv
<path\to\your\project>\.venv\Scripts\python.exe -m pip install -r <path\to\your\project>\engine\requirements.txt

# 2. 预下载预训练权重（可选，训练时也会自动下载）
<path\to\your\project>\.venv\Scripts\python.exe -X utf8 <path\to\your\project>\src\train_worker.py download

# 3. 编译
powershell -ExecutionPolicy Bypass -File <path\to\your\project>\build.ps1
```

## 5. 使用说明

### 5.1 训练（train，默认模式）

不带参数运行即进入**交互式模型选择菜单**（每次训练前可自选模型）：

```
请选择要训练的模型：
  [1] YOLOv8n    （Ultralytics YOLOv8 nano，约 3.2M 参数）
  [2] YOLO11n    （Ultralytics YOLO11 nano）
  [3] YOLO26n    （Ultralytics YOLO26 nano）
```

随后显示训练配置摘要并请求确认（`Y` 开始 / `n` 取消）。
脚本化场景请用命令行参数，例如：

```powershell
build\yolo_train.exe train --model yolov8n --epochs 100 --imgsz 512 --batch 16
```

**常用参数：**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | （菜单） | `yolov8n` / `yolo11n` / `yolo26n`（也接受 `YOLOv8n`、`yolov8n.pt`、`v8n` 等写法） |
| `--device` | `auto` | `auto`=有 GPU 用 GPU、无 GPU 用 CPU；也可强制 `cpu` 或 `0` |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 512 | 训练输入尺寸 |
| `--batch` | 16 | 批大小 |
| `--workers` | 自动 | DataLoader 进程数（自动 = min(8, CPU 核数)） |
| `--val-imgsz` | 640 | 验证与 ONNX 导出尺寸 |
| `--data` | `datasets/dataset.yaml` | 数据集配置 |
| `--resume` | 关 | 从最近一次训练继续 |
| `--no-interactive` | 关 | 禁用交互菜单 |
| `--root` | 自动 | 手动指定 <path\to\your\project> 根目录 |

**默认超参数：**

| 项 | 值 | 理由 |
| --- | --- | --- |
| 预训练 | COCO 官方 nano 权重 | 小数据集迁移学习，收敛快、mAP 高 |
| optimizer | AdamW, lr0=0.001 | 小模型小数据集上稳定 |
| cos_lr | True | 余弦退火，后期平滑收敛 |
| warmup_epochs | 3.0 | 避免早期震荡 |
| mosaic / close_mosaic | 1.0 / 10 | 马赛克增强提升小目标泛化，最后 10 轮关闭 |
| hsv_h/s/v | 0.015 / 0.6 / 0.4 | 色彩抖动，模拟场地光照/色偏 |
| degrees / translate / scale / fliplr | 5 / 0.1 / 0.5 / 0.5 | 几何增强 |
| patience | 30 | 连续 30 轮无提升早停，保护机时 |
| amp | 有 GPU 时开 | CPU 上无效，仅 GPU 开启 |

### 5.2 重新导出 ONNX

```powershell
build\yolo_train.exe export --model yolo26n        # 用 weights/YOLO26n/best.pt
build\yolo_train.exe export --weights path\to\best.pt --val-imgsz 640
```

ONNX 导出到 best.pt 同目录。

### 5.3 其他模式

```powershell
build\yolo_train.exe doctor        # 环境与数据集自检
build\yolo_train.exe download      # 预下载三种预训练权重
build\yolo_train.exe list-models   # 查看模型与权重产物状态
build\yolo_train.exe --help        # 完整帮助
```

## 6. 输出布局与日志

一次训练结束后：

```
weights/<模型>/best.pt       # PyTorch 最佳权重
weights/<模型>/best.onnx     # ONNX 最佳权重
logs/<模型>_<时间戳>/
├── train.log                # 详细训练日志
└── metrics.json             # 结构化指标
runs/detect/<模型>_<时间戳>/ # Ultralytics 过程产物
```

日志格式：`[YYYY-MM-DD HH:MM:SS.mmm][tag] 内容`，tag 区分
`config` / `device` / `engine`/ `main`]。

引擎与 C++ 之间通过 `__XXX__` 标记行通信.

## 7. GPU / CPU 说明

- 工具启动时用 `nvidia-smi` 预探测并在日志中记录 GPU 型号；
- PyPI 上的 torch Windows 包为 CPU 版。有 NVIDIA GPU 的机器如需 CUDA 加速：
  1. 到 <https://pytorch.org/get-started/locally/> 安装对应 CUDA 版 torch；
  2. 或在 `.venv` 中执行 `pip install torch==2.14.1 --index-url <CUDA 索引>`（按 PyTorch 官网指引）；
  3. 重新运行 `doctor` 自检，确认 `cuda=True` 后照常训练即可，代码无需改动。

## 10. 常见问题

**Q：首次训练很慢？**
A：首次运行会从 GitHub 下载 COCO 预训练权重（约 5~6 MB，已缓存到
`.ultralytics_cfg/weights/`）。可用 `download` 模式提前下载。

**Q：如何改数据集？**
A：替换 `datasets/images/{train,val}` 与 `datasets/labels/{train,val}` 并更新
`datasets/dataset.yaml`（`nc` 与 `names`），或用 `--data` 指向新的 yaml。
