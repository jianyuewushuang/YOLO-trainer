#pragma once
// workspace.hpp —— 项目路径管理
//
// 统一维护项目内的关键目录（datasets / src / weights / logs / runs /
// .ultralytics_cfg / .venv）。所有产物都约束在项目内，不写外部目录。
// 支持 Windows 与 Linux/macOS：虚拟环境解释器位置按平台自动选择。
#include <filesystem>

namespace fs = std::filesystem;

struct Workspace {
    fs::path root;        // 项目根目录
    fs::path datasets;    // root/datasets
    fs::path data_yaml;   // root/datasets/dataset.yaml（缺失时自动探测 datasets/*/data.yaml）
    fs::path src;         // root/src
    fs::path worker;      // root/src/train_worker.py
    fs::path weights;     // root/weights
    fs::path logs;        // root/logs
    fs::path runs;        // root/runs
    fs::path cfg;         // root/.ultralytics_cfg（Ultralytics 配置/权重缓存）
    fs::path venv_python; // .venv/Scripts/python.exe（Windows）| .venv/bin/python3（POSIX）

    bool valid() const { return !root.empty(); }

    // 按优先级探测项目根目录：--root 覆盖 > exe 所在目录的上级 > exe 所在目录 > 当前目录
    static Workspace discover(const fs::path& exe_path, const fs::path& override_root);

    // 虚拟环境解释器路径（按平台 + 实际存在的可执行文件选择）
    static fs::path venv_python_path(const fs::path& root);

    // 数据集配置：优先 datasets/dataset.yaml，其次自动探测 datasets/<名称>/data.yaml
    static fs::path find_data_yaml(const fs::path& root);
};
