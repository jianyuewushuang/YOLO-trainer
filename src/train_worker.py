#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_worker.py —— YOLO 训练引擎（Ultralytics 后端）
=====================================================================
本脚本由 train_project 的 C++ 训练工具（build/yolo_train.exe）调用，
负责真正的训练 / 验证 / ONNX 导出；C++ 侧负责模型选择、设备检测、
流程编排、日志落盘与产物归档。

设计要点：
    1. Ultralytics 配置目录重定向到 train_project/.ultralytics_cfg/，
       settings、字体、预训练权重缓存等运行时产物全部写在项目内，
       不污染项目外部目录；
    2. 设备策略：--device auto 时有 CUDA 用 GPU（device 0），否则 CPU；
    3. 训练超参数以“质量优先”为默认（AdamW + 余弦退火 + 马赛克增强 +
       COCO 预训练迁移学习 + 早停），全部可通过命令行覆盖；
    4. 与 C++ 的通信协议：普通输出行直接进日志；以 "__XXX__" 开头的行是
       结构化标记（JSON / 路径 / 摘要），C++ 解析后用于归档与指标落盘。

子命令：
    check     环境与数据集自检
    download  预下载预训练权重（yolov8n / yolo11n / yolo26n）
    train     训练 + 验证（imgsz=val-imgsz）+ 导出 ONNX
    export    由 best.pt 重新导出 ONNX
"""

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 项目路径与 Ultralytics 配置重定向（必须在 import ultralytics 之前设置）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / ".ultralytics_cfg"
CFG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CFG_DIR))


def _ensure_arial_font() -> None:
    """Ultralytics 绘图需要 Arial.ttf；缺失时从系统字体复制，避免联网下载。"""
    dst = CFG_DIR / "Arial.ttf"
    if dst.exists() and dst.stat().st_size > 0:
        return
    for src in (r"C:\Windows\Fonts\arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(src).exists():
            shutil.copy(src, dst)
            return


_ensure_arial_font()


def _bootstrap_settings() -> None:
    """确保 Ultralytics settings.json 指向 train_project 内目录。

    Ultralytics 首次创建 settings.json 时会按“从 CWD 向上寻找含 weights/runs
    的目录”推断默认路径——在本仓库结构下会推断到父项目，导致 runs/ 等产物写到
    项目外。这里在 import ultralytics 之前显式写好项目内路径；若已有配置的
    目录不在项目内（自愈旧配置），则重写。
    """
    cfg = CFG_DIR / "Ultralytics"
    cfg.mkdir(parents=True, exist_ok=True)
    sj = cfg / "settings.json"

    def under_root(p: str) -> bool:
        try:
            return bool(p) and Path(p).resolve().is_relative_to(ROOT.resolve())
        except (ValueError, OSError):
            return False

    ok = False
    if sj.exists():
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            ok = (under_root(data.get("datasets_dir", "")) and
                  under_root(data.get("weights_dir", "")) and
                  under_root(data.get("runs_dir", "")))
        except (ValueError, OSError):
            ok = False
    if not ok:
        sj.write_text(json.dumps({
            "settings_version": "0.0.8",
            "datasets_dir": str(ROOT / "datasets"),
            "weights_dir": str(CFG_DIR / "weights"),
            "runs_dir": str(ROOT / "runs"),
            "uuid": uuid.uuid4().hex,
            "sync": True, "api_key": "", "openai_api_key": "",
            "clearml": True, "comet": True, "dvc": True, "mlflow": True,
            "raytune": True, "tensorboard": False, "wandb": False,
            "vscode_msg": True, "openvino_msg": True,
        }, indent=2), encoding="utf-8")


_bootstrap_settings()


def ensure_pretrained(model_name: str) -> str:
    """确保预训练权重在项目配置目录中，返回其绝对路径。

    直接给 ultralytics 传 "yolo11n.pt" 这类名字时，它会按 CWD 相对路径下载，
    可能把权重写到项目外；这里显式下载到 <项目>/.ultralytics_cfg/weights/ 下。
    """
    from ultralytics.utils import SETTINGS
    from ultralytics.utils.downloads import attempt_download_asset

    wdir = Path(SETTINGS["weights_dir"])
    wdir.mkdir(parents=True, exist_ok=True)
    dst = wdir / f"{model_name}.pt"
    if dst.exists() and dst.stat().st_size > 1e5:
        return str(dst)
    print(f"[weights] 下载预训练权重 {model_name}.pt -> {dst}", flush=True)
    got = Path(attempt_download_asset(str(dst)))
    if not got.exists():
        raise RuntimeError(f"预训练权重下载失败: {model_name}.pt")
    return str(got)

SUPPORTED = ("yolov8n", "yolo11n", "yolo26n")
DIR_NAMES = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo26n": "YOLO26n"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def emit_marker(name: str, payload: str = "") -> None:
    """向 C++ 输出结构化标记行。"""
    print(f"__{name}__{payload}", flush=True)


def fail(msg: str) -> int:
    emit_marker("ERROR", msg)
    return 1


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def load_dataset_info(yaml_path: Path) -> dict:
    """读取 dataset.yaml 并统计图片/标签数量；path 相对路径按 yaml 目录解析。"""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", "."))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    data["path"] = str(base)

    stats = {}
    for split in ("train", "val"):
        img_dir = base / data.get(split, f"images/{split}")
        lbl_dir = base / "labels" / split
        n_img = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) \
            if img_dir.is_dir() else 0
        n_lbl = sum(1 for _ in lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else 0
        stats[split] = {"images": n_img, "labels": n_lbl,
                        "image_dir": str(img_dir), "label_dir": str(lbl_dir)}
    return {"data": data, "stats": stats,
            "nc": int(data.get("nc", 0)), "names": data.get("names", [])}


def write_resolved_yaml(data: dict) -> Path:
    """把解析好绝对路径的数据集配置落到项目配置目录，返回其路径。

    ultralytics 的 train/val 需要 data 为 yaml 路径（不接受 dict），
    这里生成一份 path 已解析为绝对路径的副本，保证项目可整体迁移。
    """
    import yaml

    out = CFG_DIR / "resolved_dataset.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    return out


def resolve_device(device: str, torch_mod):
    """返回 (device, gpu_name)。auto：有 CUDA 用 0，否则 CPU。"""
    if device and device.lower() == "cpu":
        return "cpu", ""
    if device and device != "auto":
        if torch_mod.cuda.is_available():
            return device, torch_mod.cuda.get_device_name(0)
        print(f"[warn] 指定了 GPU 设备 {device} 但 CUDA 不可用，回退 CPU", flush=True)
        return "cpu", ""
    if torch_mod.cuda.is_available():
        return "0", torch_mod.cuda.get_device_name(0)
    return "cpu", ""


# ---------------------------------------------------------------------------
# 子命令：check（环境与数据集自检）
# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    info = {"root": str(ROOT), "cfg_dir": str(CFG_DIR), "python": sys.version.split()[0]}
    try:
        import torch
        import ultralytics
        from ultralytics.utils import SETTINGS

        info["torch"] = torch.__version__
        info["ultralytics"] = ultralytics.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if info["cuda"] else ""
    except Exception as e:  # noqa: BLE001
        return fail(f"引擎依赖不完整: {e}")

    yaml_path = Path(args.data) if args.data else ROOT / "datasets" / "dataset.yaml"
    if not yaml_path.exists():
        return fail(f"数据集配置不存在: {yaml_path}")
    ds = load_dataset_info(yaml_path)
    info["dataset"] = {
        "yaml": str(yaml_path), "nc": ds["nc"], "names": ds["names"],
        "train_images": ds["stats"]["train"]["images"],
        "val_images": ds["stats"]["val"]["images"],
    }

    wdir = Path(SETTINGS["weights_dir"])
    info["weights_cache"] = {m: str(wdir / f"{m}.pt") for m in SUPPORTED}
    info["weights_cached"] = {m: (wdir / f"{m}.pt").exists() for m in SUPPORTED}

    emit_marker("CHECK_JSON", json.dumps(info, ensure_ascii=False))
    print(f"[check] python={info['python']} torch={info['torch']} "
          f"ultralytics={info['ultralytics']}", flush=True)
    print(f"[check] cuda={info['cuda']} gpu={info.get('gpu_name', '') or '-'}", flush=True)
    print(f"[check] 数据集 nc={ds['nc']} train={ds['stats']['train']['images']} "
          f"val={ds['stats']['val']['images']}", flush=True)
    for m in SUPPORTED:
        state = "已缓存" if info["weights_cached"][m] else "未缓存（训练时自动下载）"
        print(f"[check] 预训练权重 {m}.pt: {state}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 子命令：download（预下载预训练权重）
# ---------------------------------------------------------------------------
def normalize_model_name(name: str) -> str:
    """接受 yolov8n / yolov8n.pt / YOLOv8n 等写法，统一为 yolov8n。"""
    s = name.strip().lower()
    if s.endswith(".pt"):
        s = s[:-3]
    return s


def cmd_download(args) -> int:
    models = [normalize_model_name(m) for m in args.models.split(",") if m.strip()] \
        or list(SUPPORTED)
    for m in models:
        if m not in SUPPORTED:
            return fail(f"不支持的模型: {m}（可选: {', '.join(SUPPORTED)}）")
        try:
            p = ensure_pretrained(m)
        except Exception as e:  # noqa: BLE001
            return fail(f"下载 {m}.pt 失败: {e}")
        print(f"[download] {m}.pt 就绪: {p}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 子命令：train（训练 + 验证 + 导出 ONNX）
# ---------------------------------------------------------------------------
def cmd_train(args) -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO

    t0 = time.time()
    model_name = normalize_model_name(args.model)
    if model_name not in SUPPORTED:
        return fail(f"不支持的模型: {model_name}（可选: {', '.join(SUPPORTED)}）")
    dir_name = DIR_NAMES[model_name]
    workers = args.workers if args.workers > 0 else min(8, os.cpu_count() or 1)

    yaml_path = Path(args.data) if args.data else ROOT / "datasets" / "dataset.yaml"
    if not yaml_path.exists():
        return fail(f"数据集配置不存在: {yaml_path}")
    ds = load_dataset_info(yaml_path)
    if ds["stats"]["train"]["images"] == 0:
        return fail(f"训练集为空: {ds['stats']['train']['image_dir']}")
    if ds["stats"]["val"]["images"] == 0:
        return fail(f"验证集为空: {ds['stats']['val']['image_dir']}")

    device, gpu_name = resolve_device(args.device, torch)
    emit_marker("STATUS_JSON", json.dumps({"phase": "init", "device": device, "gpu": gpu_name}))
    print(f"[train] 模型={model_name}.pt 设备={device}({gpu_name or 'CPU'}) "
          f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} workers={workers}",
          flush=True)
    print(f"[train] 数据集: train={ds['stats']['train']['images']} "
          f"val={ds['stats']['val']['images']} nc={ds['nc']}", flush=True)
    resolved_yaml = write_resolved_yaml(ds["data"])

    try:
        # 预训练权重固定在项目配置目录，加载即开始 COCO 迁移学习
        pretrained = ensure_pretrained(model_name)
        model = YOLO(pretrained)
    except Exception as e:  # noqa: BLE001
        return fail(f"获取/加载预训练权重 {model_name}.pt 失败: {e}")

    name = f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}"
    project = str(ROOT / "runs" / "detect")

    if args.resume:
        print("[train] resume=True：从最近一次训练继续", flush=True)
        train_kwargs = {"resume": True}
    else:
        train_kwargs = dict(
            data=str(resolved_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            optimizer=args.optimizer,
            lr0=args.lr0,
            cos_lr=True,
            weight_decay=args.weight_decay,
            patience=args.patience,
            warmup_epochs=args.warmup,
            mosaic=args.mosaic,
            close_mosaic=args.close_mosaic,
            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
            degrees=args.degrees, translate=args.translate,
            scale=args.scale, fliplr=args.fliplr,
            amp=(device != "cpu"),  # CPU 上 AMP 无效，仅 GPU 开启
            verbose=True,
        )
    try:
        results = model.train(**train_kwargs)
    except Exception as e:  # noqa: BLE001
        return fail(f"训练过程失败: {e}")

    run_dir = Path(getattr(results, "save_dir", None) or (Path(project) / name))
    best_pt = run_dir / "weights" / "best.pt"
    emit_marker("RUN_DIR", str(run_dir))
    emit_marker("BEST_PT", str(best_pt))
    if not best_pt.exists():
        return fail(f"未找到 best.pt: {best_pt}")

    # ---- 验证：统一用 best.pt 在 imgsz=val_imgsz 下评测，作为对外指标 ----
    emit_marker("STATUS_JSON", json.dumps({"phase": "val", "imgsz": args.val_imgsz}))
    try:
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(data=str(resolved_yaml), imgsz=args.val_imgsz,
                                 device=device, batch=args.batch, workers=workers,
                                 name=name + "_val640", exist_ok=True)
    except Exception as e:  # noqa: BLE001
        return fail(f"验证失败: {e}")

    box = metrics.box
    map50 = float(box.map50)
    map5095 = float(box.map)
    per_class = {str(best_model.names[int(c)]): float(v)
                 for c, v in zip(box.ap_class_index, box.ap50)}

    # ---- 导出 ONNX（与验证同尺寸，固定输入、FP32） ----
    emit_marker("STATUS_JSON", json.dumps({"phase": "export", "format": "onnx",
                                           "imgsz": args.val_imgsz}))
    try:
        onnx_path = Path(best_model.export(format="onnx", imgsz=args.val_imgsz,
                                           simplify=True))
    except Exception as e:  # noqa: BLE001
        return fail(f"ONNX 导出失败: {e}")
    emit_marker("ONNX", str(onnx_path))

    elapsed = time.time() - t0
    summary = {
        "model": model_name,
        "model_dir": dir_name,
        "device": device,
        "gpu_name": gpu_name,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "imgsz_train": args.imgsz,
        "imgsz_val": args.val_imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "patience": args.patience,
        "mAP50": round(map50, 4),
        "mAP50-95": round(map5095, 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_mAP50": {k: round(v, 4) for k, v in per_class.items()},
        "train_seconds": round(elapsed, 1),
        "run_dir": str(run_dir),
        "best_pt": str(best_pt),
        "best_onnx": str(onnx_path),
        "dataset": {
            "yaml": str(yaml_path), "nc": ds["nc"], "names": ds["names"],
            "train_images": ds["stats"]["train"]["images"],
            "val_images": ds["stats"]["val"]["images"],
        },
    }
    emit_marker("METRICS_JSON", json.dumps(summary, ensure_ascii=False))
    emit_marker("SUMMARY", ";".join(f"{k}={v}" for k, v in [
        ("model", dir_name), ("device", device), ("gpu", gpu_name or "-"),
        ("mAP50", f"{map50:.4f}"), ("mAP50-95", f"{map5095:.4f}"),
        ("precision", f"{float(box.mp):.4f}"), ("recall", f"{float(box.mr):.4f}"),
        ("epochs", args.epochs), ("imgsz", args.imgsz), ("batch", args.batch),
        ("train_seconds", f"{elapsed:.1f}"),
        ("best_pt", best_pt), ("best_onnx", onnx_path),
    ]))
    print(f"[train] 完成，总用时 {elapsed:.1f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 子命令：export（由 best.pt 导出 ONNX）
# ---------------------------------------------------------------------------
def cmd_export(args) -> int:
    from ultralytics import YOLO

    w = Path(args.weights)
    if not w.exists():
        return fail(f"权重文件不存在: {w}")
    emit_marker("STATUS_JSON", json.dumps({"phase": "export", "weights": str(w)}))
    try:
        onnx_path = Path(YOLO(str(w)).export(format="onnx", imgsz=args.imgsz,
                                             simplify=True))
    except Exception as e:  # noqa: BLE001
        return fail(f"ONNX 导出失败: {e}")
    emit_marker("ONNX", str(onnx_path))
    print(f"[export] {w} -> {onnx_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="YOLO 训练引擎（由 C++ 训练工具调用）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("check", help="环境与数据集自检")
    sp.add_argument("--data", default="")

    sp = sub.add_parser("download", help="预下载预训练权重")
    sp.add_argument("--models", default="yolov8n,yolo11n,yolo26n")

    sp = sub.add_parser("train", help="训练 + 验证 + 导出 ONNX")
    sp.add_argument("--model", required=True, help="yolov8n / yolo11n / yolo26n")
    sp.add_argument("--data", default="")
    sp.add_argument("--device", default="auto", help="auto / cpu / 0 / 0,1 ...")
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--imgsz", type=int, default=512)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--workers", type=int, default=0, help="0=自动")
    sp.add_argument("--val-imgsz", type=int, default=640)
    sp.add_argument("--optimizer", default="AdamW")
    sp.add_argument("--lr0", type=float, default=0.001)
    sp.add_argument("--weight-decay", type=float, default=0.0005)
    sp.add_argument("--patience", type=int, default=30)
    sp.add_argument("--warmup", type=float, default=3.0)
    sp.add_argument("--mosaic", type=float, default=1.0)
    sp.add_argument("--close-mosaic", type=int, default=10)
    sp.add_argument("--hsv-h", type=float, default=0.015)
    sp.add_argument("--hsv-s", type=float, default=0.6)
    sp.add_argument("--hsv-v", type=float, default=0.4)
    sp.add_argument("--degrees", type=float, default=5.0)
    sp.add_argument("--translate", type=float, default=0.1)
    sp.add_argument("--scale", type=float, default=0.5)
    sp.add_argument("--fliplr", type=float, default=0.5)
    sp.add_argument("--resume", action="store_true")

    sp = sub.add_parser("export", help="由 best.pt 导出 ONNX")
    sp.add_argument("--weights", required=True)
    sp.add_argument("--imgsz", type=int, default=640)

    args = p.parse_args()
    try:
        if args.cmd == "check":
            return cmd_check(args)
        if args.cmd == "download":
            return cmd_download(args)
        if args.cmd == "train":
            return cmd_train(args)
        if args.cmd == "export":
            return cmd_export(args)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return fail(f"未处理异常: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
