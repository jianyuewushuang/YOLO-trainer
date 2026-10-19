# train_project —— C++ YOLO 训练工具（YOLOv8n / YOLO11n / YOLO26n）

以**最终交付产品**为目标的一键 YOLO 训练工具。C++ 程序（`build/yolo_train.exe`）负责
**模型选择、GPU/CPU 自动检测、训练流程编排、详细训练日志、产物归档**；底层训练引擎为
`src/train_worker.py`（Ultralytics 8.4.171），负责真正的训练 / 验证 / ONNX 导出。

**任务书见 [REQUEST.md](./REQUEST.md)。**

## 1. 功能特性

- **三种 nano 模型任选**：YOLOv8n / YOLO11n / YOLO26n，每次训练前可通过交互菜单自选；
- **一键流水线**：设备检测 → 训练 → imgsz=640 验证 → 导出 ONNX → 归档
  `weights/<模型>/{best.pt, best.onnx}`；
- **GPU 优先**：有 NVIDIA GPU 自动用 GPU（CUDA），无 GPU 自动用 CPU，无需手动切换；
- **质量优先的默认超参数**：COCO 预训练迁移学习 + AdamW + 余弦退火 + 马赛克增强 + 早停；
- **详细训练日志**：每次训练在 `logs/<模型>_<时间戳>/` 下生成带毫秒时间戳的
  `train.log`（完整训练过程转录）与 `metrics.json`（结构化指标）；
- **完全自包含**：所有产物（虚拟环境、Ultralytics 配置/权重缓存、日志、runs、weights）
  都在 `train_project/` 内，不读写项目外部目录。

## 2. 目录结构

```
train_project/
├── REQUEST.md                  # 任务书
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
├── .venv/                      # train_project 独立虚拟环境（Python 3.11）
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
| Windows 10/11 | 开发与实测平台 |
| Python 3.10 ~ 3.12 | `setup.ps1` 会自动挑选（本机使用 3.11.9） |
| MSYS2 UCRT64 g++ | 编译 C++ 工具（`C:\msys64\ucrt64\bin\g++.exe`） |
| CMake（可选） | 第二种构建方式，`build.ps1` 直接调 g++ 不依赖它 |
| 网络 | 首次搭建需联网：pip 装依赖、GitHub 下载预训练权重 |
| NVIDIA GPU（可选） | 有则自动用 GPU 训练；无则用 CPU |

## 4. 快速开始

在 **PowerShell** 中执行（一键完成：选 Python → 建 `.venv` → 装依赖 → 预下载权重 → 编译）：

```powershell
powershell -ExecutionPolicy Bypass -File train_project\setup.ps1
```

成功后自检并开始训练：

```powershell
# 环境自检（Python/torch/ultralytics/数据集/权重缓存/GPU）
powershell -ExecutionPolicy Bypass -File train_project\build\yolo_train.exe doctor

# 交互式训练：菜单选择模型 → 确认参数 → 一键训练
powershell -ExecutionPolicy Bypass -File train_project\build\yolo_train.exe

# 或命令行直接指定模型
powershell -ExecutionPolicy Bypass -File train_project\build\yolo_train.exe train --model yolo26n
```

### 手动分步（不用 setup.ps1 时）

```powershell
# 1. 建虚拟环境并装依赖（用任意 Python 3.10~3.12）
C:\...\python.exe -m venv train_project\.venv
train_project\.venv\Scripts\python.exe -m pip install -r train_project\engine\requirements.txt

# 2. 预下载预训练权重（可选，训练时也会自动下载）
train_project\.venv\Scripts\python.exe -X utf8 train_project\src\train_worker.py download

# 3. 编译
powershell -ExecutionPolicy Bypass -File train_project\build.ps1
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
| `--epochs` | 100 | 训练轮数（配合 `patience=30` 早停） |
| `--imgsz` | 512 | 训练输入尺寸 |
| `--batch` | 16 | 批大小 |
| `--workers` | 自动 | DataLoader 进程数（自动 = min(8, CPU 核数)） |
| `--val-imgsz` | 640 | 验证与 ONNX 导出尺寸 |
| `--data` | `datasets/dataset.yaml` | 数据集配置 |
| `--resume` | 关 | 从最近一次训练继续 |
| `--no-interactive` | 关 | 禁用交互菜单 |
| `--root` | 自动 | 手动指定 train_project 根目录 |

**默认超参数（质量优先，引擎侧）：**

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

### 5.2 重新导出 ONNX（export）

```powershell
build\yolo_train.exe export --model yolo26n        # 用 weights/YOLO26n/best.pt
build\yolo_train.exe export --weights path\to\best.pt --val-imgsz 640
```

ONNX 导出到 best.pt 同目录（FP32、固定 640×640 输入、onnxslim 简化）。
YOLO26n 为 NMS-free 端到端结构，输出 `(1, 4+12, 8400)`，部署时无需再接 NMS。

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
weights/<模型>/best.pt       # PyTorch 最佳权重（交付物）
weights/<模型>/best.onnx     # ONNX 最佳权重（交付物）
logs/<模型>_<时间戳>/
├── train.log                # 详细训练日志：banner + 配置 + 设备 + 引擎全过程转录 + 汇总
└── metrics.json             # 结构化指标（mAP50、mAP50-95、precision、recall、各类 mAP、超参数、用时、产物路径）
runs/detect/<模型>_<时间戳>/ # Ultralytics 过程产物（results.csv、曲线、混淆矩阵、batch 样例图）
```

日志格式：`[YYYY-MM-DD HH:MM:SS.mmm][tag] 内容`，tag 区分
`config` / `device` / `engine`（引擎输出）/ `main`（编排事件）。
引擎与 C++ 之间通过 `__XXX__` 标记行通信（`__METRICS_JSON__`、`__SUMMARY__` 等），
这些标记行也会原样出现在日志中，便于追溯。

## 7. GPU / CPU 说明

- 工具启动时用 `nvidia-smi` 预探测并在日志中记录 GPU 型号；
- 最终训练设备由引擎按 `torch.cuda.is_available()` 判定：**有 GPU 用 `device=0`（并开启 AMP），
  无 GPU 用 CPU**（AMP 自动关闭，避免无效警告）；
- PyPI 上的 torch Windows 包为 CPU 版。有 NVIDIA GPU 的机器如需 CUDA 加速：
  1. 到 <https://pytorch.org/get-started/locally/> 安装对应 CUDA 版 torch；
  2. 或在 `.venv` 中执行 `pip install torch==2.14.1 --index-url <CUDA 索引>`（按 PyTorch 官网指引）；
  3. 重新运行 `doctor` 自检，确认 `cuda=True` 后照常训练即可，代码无需改动。

## 8. 本次功能验证记录（2026-10-19，CPU 环境）

按任务书要求“完成功能并进行验证”，对三种模型各执行了 **3 epochs** 短程训练
（imgsz=384、batch=8、val@640），完整走通
*设备检测 → 训练 → 验证 → ONNX 导出 → 归档 → 日志/指标落盘* 全流程：

| 模型 | 设备 | mAP50 | mAP50-95 | Precision | Recall | 用时 | 产物 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n | CPU | 0.8386 | 0.6755 | 0.9013 | 0.7659 | 777.5 s | `weights/YOLOv8n/{best.pt,best.onnx}` |
| YOLO11n | CPU | 0.7920 | 0.6450 | 0.7501 | 0.7275 | 838.8 s | `weights/YOLO11n/{best.pt,best.onnx}` |
| YOLO26n | CPU | 0.7818 | 0.6287 | 0.8500 | 0.7353 | 868.2 s | `weights/YOLO26n/{best.pt,best.onnx}` |

> 3 epochs 仅为功能验证；正式使用时按默认（epochs=100 + patience=30）训练即可，
> 指标会显著高于上表。交互菜单、`doctor`、`download`、`list-models`、`export`、
> `build.ps1` 与 CMake 两条构建路径均验证通过。

## 9. 设计说明

- **分工**：C++（`yolo_train.exe`）做产品级编排——交互菜单、参数校验、设备探测、
  子进程管理（管道实时捕获输出）、时间戳日志、产物归档、退出码语义；
  Python（`train_worker.py`）做训练本身。`.pt` 由 Ultralytics 产出，格式与生态完全兼容。
- **路径隔离**：`YOLO_CONFIG_DIR` 重定向到 `train_project/.ultralytics_cfg/`；
  引擎启动时还会自愈 `settings.json`（Ultralytics 默认会按 CWD 向上推断
  weights/runs 目录，可能指到项目外），确保预训练权重、runs、配置全部落在项目内。
- **通信协议**：子进程 stdout/stderr 合并后按行实时写入日志；`__XXX__` 前缀的标记行
  携带结构化数据（设备、指标、产物路径），C++ 解析后完成归档与汇总。

## 10. 常见问题（FAQ）

**Q：首次训练很慢？**
A：首次运行会从 GitHub 下载 COCO 预训练权重（约 5~6 MB，已缓存到
`.ultralytics_cfg/weights/`）。可用 `download` 模式提前下载。

**Q：训练中途报错退出？**
A：查看 `logs/<模型>_<时间戳>/train.log` 末尾的 `__ERROR__` 行；`runs/detect/` 下
保留了断点现场，修复后可用 `train --model X --resume` 续训。

**Q：CPU 上一次训练要多久？**
A：3000 张 @384、batch=8 约 4~5 分钟/epoch（16 核 CPU）。默认 imgsz=512 会更慢，
可按需调小 `--imgsz`/`--batch`；有 GPU 会快一个数量级以上。

**Q：`yolo_train.exe` 提示找不到项目根目录？**
A：exe 通过自身位置（`build/` 的上级）识别 train_project；若移动了目录结构，
用 `--root <train_project 路径>` 指定。

**Q：如何改数据集？**
A：替换 `datasets/images/{train,val}` 与 `datasets/labels/{train,val}` 并更新
`datasets/dataset.yaml`（`nc` 与 `names`），或用 `--data` 指向新的 yaml。
