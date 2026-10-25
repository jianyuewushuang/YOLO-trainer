#include "workspace.hpp"

#include <algorithm>
#include <array>
#include <vector>

fs::path Workspace::venv_python_path(const fs::path& root) {
    std::error_code ec;
#ifdef _WIN32
    return root / ".venv" / "Scripts" / "python.exe";
#else
    // POSIX 虚拟环境：bin/python3 优先，兼容只提供 bin/python 的环境
    const fs::path py3 = root / ".venv" / "bin" / "python3";
    if (fs::exists(py3, ec)) return py3;
    const fs::path py = root / ".venv" / "bin" / "python";
    if (fs::exists(py, ec)) return py;
    return py3;  // 不存在时返回默认路径，供错误提示使用
#endif
}

namespace {

// 在 dir 内挑一个数据集 yaml：dir/<固定名> 优先，其次 dir/<子目录>/data.yaml
// 同时兼容 .yml 扩展名（如 Roboflow 有时导出 data.yml）
fs::path pick_data_yaml(const fs::path& dir) {
    std::error_code ec;
    for (const char* name : {"dataset.yaml", "dataset.yml", "data.yaml", "data.yml"}) {
        fs::path cand = dir / name;
        if (fs::is_regular_file(cand, ec)) return cand;
    }
    // Roboflow 导出布局：<数据集名>/data.yaml（多个时取字典序第一个）
    std::vector<fs::path> found;
    if (fs::is_directory(dir, ec)) {
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (!entry.is_directory(ec)) continue;
            for (const char* name : {"data.yaml", "data.yml"}) {
                fs::path cand = entry.path() / name;
                if (fs::is_regular_file(cand, ec)) {
                    found.push_back(cand);
                    break;
                }
            }
        }
    }
    std::sort(found.begin(), found.end());
    return found.empty() ? fs::path{} : found.front();
}

}  // namespace

fs::path Workspace::find_data_yaml(const fs::path& root) {
    return pick_data_yaml(root / "datasets");
}

fs::path Workspace::find_data_yaml_in(const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return dir;  // 已是文件（或不存在）：交给调用方校验

    if (fs::path found = pick_data_yaml(dir); !found.empty()) return found;
    // 传的是项目根目录时，数据集通常在 datasets/ 下
    return pick_data_yaml(dir / "datasets");
}

Workspace Workspace::discover(const fs::path& exe_path, const fs::path& override_root) {
    // 判定“项目根”的标志：存在 src/train_worker.py。
    // 注意：不要求项目内存在数据集配置——数据集可以用 --data 放在任意位置
    //（如 /mnt/data/...），否则外部数据集场景会误判为“未找到项目根目录”。
    const auto is_root = [](const fs::path& p) {
        std::error_code ec;
        return fs::exists(p / "src" / "train_worker.py", ec);
    };

    const auto make_workspace = [](const fs::path& root) {
        Workspace w;
        w.root = root;
        w.datasets = root / "datasets";
        w.data_yaml = find_data_yaml(root);
        w.src = root / "src";
        w.worker = w.src / "train_worker.py";
        w.weights = root / "weights";
        w.logs = root / "logs";
        w.runs = root / "runs";
        w.cfg = root / ".ultralytics_cfg";
        w.venv_python = venv_python_path(root);
        return w;
    };

    // 显式 --root：严格校验，不静默回退到其他位置，避免误用别的项目根
    if (!override_root.empty()) {
        std::error_code ec;
        fs::path root = fs::weakly_canonical(override_root, ec);
        if (ec) root = override_root;
        return is_root(root) ? make_workspace(root) : Workspace{};
    }

    std::array<fs::path, 3> candidates = {
        exe_path.parent_path().parent_path(),  // build/yolo_train -> 项目根
        exe_path.parent_path(),
        fs::current_path(),
    };

    for (const auto& c : candidates) {
        if (c.empty()) continue;
        std::error_code ec;
        fs::path root = fs::weakly_canonical(c, ec);
        if (ec) root = c;
        if (is_root(root)) return make_workspace(root);
    }
    return {};  // root 为空表示未找到
}
