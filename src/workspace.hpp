#pragma once
// workspace.hpp —— 项目路径管理
//
// 统一维护 train_project 内的关键目录（datasets / src / weights / logs /
// runs / .ultralytics_cfg / .venv）。所有产物都约束在项目内，不写外部目录。
#include <filesystem>

namespace fs = std::filesystem;

struct Workspace {
    fs::path root;        // train_project/
    fs::path datasets;    // root/datasets
    fs::path data_yaml;   // root/datasets/dataset.yaml
    fs::path src;         // root/src
    fs::path worker;      // root/src/train_worker.py
    fs::path weights;     // root/weights
    fs::path logs;        // root/logs
    fs::path runs;        // root/runs
    fs::path cfg;         // root/.ultralytics_cfg（Ultralytics 配置/权重缓存）
    fs::path venv_python; // root/.venv/Scripts/python.exe

    bool valid() const { return !root.empty(); }

    // 按优先级探测项目根目录：--root 覆盖 > exe 所在目录的上级 > exe 所在目录 > 当前目录
    static Workspace discover(const fs::path& exe_path, const fs::path& override_root);
};
