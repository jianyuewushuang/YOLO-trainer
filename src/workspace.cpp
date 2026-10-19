#include "workspace.hpp"

#include <array>

Workspace Workspace::discover(const fs::path& exe_path, const fs::path& override_root) {
    // 判定“项目根”的标志：同时存在 datasets/dataset.yaml 与 src/train_worker.py
    const auto is_root = [](const fs::path& p) {
        std::error_code ec;
        return fs::exists(p / "datasets" / "dataset.yaml", ec) &&
               fs::exists(p / "src" / "train_worker.py", ec);
    };

    std::array<fs::path, 4> candidates = {
        override_root,
        exe_path.parent_path().parent_path(),  // build/yolo_train.exe -> train_project/
        exe_path.parent_path(),
        fs::current_path(),
    };

    for (const auto& c : candidates) {
        if (c.empty()) continue;
        fs::path root = fs::weakly_canonical(c);
        if (is_root(root)) {
            Workspace w;
            w.root = root;
            w.datasets = root / "datasets";
            w.data_yaml = w.datasets / "dataset.yaml";
            w.src = root / "src";
            w.worker = w.src / "train_worker.py";
            w.weights = root / "weights";
            w.logs = root / "logs";
            w.runs = root / "runs";
            w.cfg = root / ".ultralytics_cfg";
            w.venv_python = root / ".venv" / "Scripts" / "python.exe";
            return w;
        }
    }
    return {};  // root 为空表示未找到
}
