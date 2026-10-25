// ============================================================================
// main.cpp —— C++ YOLO 训练工具 CLI
//
// 定位：以最终交付产品为目标的一键训练工具。负责模型选择（YOLOv8n /
// YOLO11n / YOLO26n）、GPU/CPU 自动检测、训练流程编排、详细训练日志、
// 产物归档；底层训练引擎为 src/train_worker.py（Ultralytics）。
//
// 模式：
//   train       交互/命令行选择模型 -> 训练 -> 验证 -> 导出 ONNX -> 归档
//   export      由 best.pt 重新导出 ONNX
//   doctor      环境与数据集自检
//   download    预下载预训练权重
//   list-models 查看模型与权重产物状态
//
// 产物布局（均在项目根目录内）：
//   weights/<模型>/{best.pt,best.onnx}
//   logs/<模型>_<时间戳>/{train.log,metrics.json}
//   runs/detect/<实验名>/            Ultralytics 训练过程产物
// ============================================================================
#include "cli.hpp"
#include "device.hpp"
#include "logger.hpp"
#include "worker.hpp"
#include "workspace.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static const char* kVersion = "1.0.0";
static const char* const kSupportedModels[] = {"yolov8n", "yolo11n", "yolo26n"};

// 平台相关的提示文案（提示用户运行哪个脚本）
#ifdef _WIN32
static const char* const kSetupScript = "setup.ps1";
#else
static const char* const kSetupScript = "setup.sh";
#endif

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
static std::string now_dir_stamp() {
    const std::time_t t = std::time(nullptr);
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[32] = {0};
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    return buf;
}

// 当前可执行文件路径：优先系统接口，其次 argv[0]，最后当前目录
static fs::path self_exe_path(const char* argv0) {
    std::error_code ec;
#ifdef _WIN32
    std::vector<char> buf(1024);
    for (;;) {
        const DWORD n = GetModuleFileNameA(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) break;
        if (n < buf.size() - 1) return fs::path(std::string(buf.data(), n));
        buf.resize(buf.size() * 2);
    }
#else
    // Linux：/proc/self/exe 是权威路径（可处理从 PATH 调用、符号链接等情况）
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return p;
#endif
    if (argv0 && *argv0) {
        fs::path p = fs::weakly_canonical(fs::path(argv0), ec);
        if (!ec) return p;
    }
    return fs::current_path();
}

// 在 PATH 中查找可执行文件（跨平台），未找到返回空串
static std::string find_in_path(const std::string& name) {
    std::error_code ec;
#ifdef _WIN32
    // MSYS2 UCRT64 / CMake 常见安装位置
    const std::vector<std::string> extra = {
        "C:\\msys64\\ucrt64\\bin\\",
        "C:\\msys64\\mingw64\\bin\\",
        "C:\\Program Files\\CMake\\bin\\",
    };
    for (const auto& dir : extra) {
        fs::path cand = fs::path(dir) / (name + ".exe");
        if (fs::is_regular_file(cand, ec)) return cand.string();
    }
    const char sep = ';';
    const std::vector<std::string> exts = {"", ".exe"};
#else
    const char sep = ':';
    const std::vector<std::string> exts = {""};
#endif
    const char* env = std::getenv("PATH");
    if (!env) return "";
    const std::string paths = env;
    std::size_t pos = 0;
    while (pos <= paths.size()) {
        const std::size_t next = paths.find(sep, pos);
        const std::string dir = paths.substr(
            pos, next == std::string::npos ? std::string::npos : next - pos);
        if (!dir.empty()) {
            for (const auto& ext : exts) {
                fs::path cand = fs::path(dir) / (name + ext);
                if (fs::is_regular_file(cand, ec)) return cand.string();
            }
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return "";
}

// 数据集配置：--data 优先（可为 yaml 文件或数据集目录），否则用项目内自动探测
static fs::path resolve_data_yaml(const Workspace& ws, const Options& opt) {
    if (opt.data.empty()) return ws.data_yaml;
    const fs::path found = Workspace::find_data_yaml_in(fs::path(opt.data));
    std::error_code ec;
    const fs::path abs = fs::weakly_canonical(found, ec);  // 统一为绝对路径，日志更明确
    return ec ? found : abs;
}

// --data 解析失败时的提示文案
static std::string data_yaml_hint(const Options& opt) {
    if (opt.data.empty()) return "项目内未探测到数据集配置";
    std::error_code ec;
    if (fs::is_directory(fs::path(opt.data), ec)) {
        return opt.data + " 下未找到 dataset.yaml / data.yaml";
    }
    return "文件不存在: " + opt.data;
}

// 查找以 prefix 开头的标记行，返回其载荷（不含前缀）
static std::string find_marker(const std::vector<std::string>& lines,
                               const std::string& prefix) {
    for (const auto& l : lines) {
        if (l.rfind(prefix, 0) == 0) return l.substr(prefix.size());
    }
    return "";
}

static bool copy_artifact(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    if (!fs::exists(from, ec)) return false;
    fs::create_directories(to.parent_path(), ec);
    fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
    return !ec;
}

static double file_size_mb(const fs::path& p) {
    std::error_code ec;
    auto sz = fs::file_size(p, ec);
    if (ec) return 0.0;
    return static_cast<double>(sz) / (1024.0 * 1024.0);
}

// 解析 "__SUMMARY__k=v;k=v" 为键值对
static std::vector<std::pair<std::string, std::string>> parse_summary(
    const std::string& payload) {
    std::vector<std::pair<std::string, std::string>> kv;
    std::istringstream ss(payload);
    std::string item;
    while (std::getline(ss, item, ';')) {
        std::size_t eq = item.find('=');
        if (eq == std::string::npos) continue;
        kv.emplace_back(item.substr(0, eq), item.substr(eq + 1));
    }
    return kv;
}

static const std::string* summary_get(
    const std::vector<std::pair<std::string, std::string>>& kv, const std::string& key) {
    for (const auto& p : kv) {
        if (p.first == key) return &p.second;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// list-models
// ---------------------------------------------------------------------------
static int cmd_list_models(const Workspace& ws) {
    Logger& log = Logger::instance();
    log.raw("可用模型（训练产物位于 weights/ 下对应目录）：");
    for (const char* m : kSupportedModels) {
        std::string canonical = m;
        std::string dir = model_dir_name(canonical);
        fs::path pt = ws.weights / dir / "best.pt";
        fs::path onnx = ws.weights / dir / "best.onnx";
        char line[512] = {0};
        std::snprintf(line, sizeof(line), "  %-8s  best.pt: %-8s  best.onnx: %-8s",
                      dir.c_str(),
                      fs::exists(pt) ? "已生成" : "未训练",
                      fs::exists(onnx) ? "已导出" : "未导出");
        log.raw(line);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// download
// ---------------------------------------------------------------------------
static int cmd_download(const Workspace& ws, const Options& opt) {
    (void)opt;  // 预留：后续可按模型过滤下载
    Logger& log = Logger::instance();
    if (!fs::exists(ws.venv_python)) {
        log.line("main", "未找到虚拟环境 Python：" + ws.venv_python.string() +
                             "，请先运行 " + std::string(kSetupScript));
        return 3;
    }
    log.line("main", "预下载预训练权重（yolov8n / yolo11n / yolo26n）...");
    RunResult r = run_process(ws.venv_python,
                              {"-u", "-X", "utf8", ws.worker.string(), "download"},
                              "engine");
    if (r.exit_code != 0) {
        log.line("main", "预下载失败，请检查网络后重试");
        return 4;
    }
    log.line("main", "预训练权重就绪");
    return 0;
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------
static int cmd_doctor(const Workspace& ws, const Options& opt) {
    Logger& log = Logger::instance();
    int problems = 0;

    log.raw("================ 环境自检（doctor） ================");

    // 1. 项目根目录
    if (ws.valid()) {
        log.line("doctor", "项目根目录: " + ws.root.string() + "  [OK]");
    } else {
        log.line("doctor", "未找到项目根目录（需含 src/train_worker.py；数据集可用 --data "
                           "放在任意位置），可用 --root 指定  [FAIL]");
        log.line("doctor", "可执行文件: " + self_exe_path(nullptr).string() + "  [诊断]");
        log.line("doctor", "当前目录:   " + fs::current_path().string() + "  [诊断]");
        return 3;
    }

    // 2. 编译工具链（可选）
    const std::string gxx = find_in_path("g++");
    if (!gxx.empty()) {
        log.line("doctor", "C++ 编译器 g++: " + gxx + "  [OK]");
    } else {
        log.line("doctor", "未找到 g++（编译 yolo_train 需要 C++17 编译器）  [WARN]");
    }
    const std::string cmake = find_in_path("cmake");
    if (!cmake.empty()) {
        log.line("doctor", "CMake: " + cmake + "  [OK]");
    } else {
        log.line("doctor", "CMake 未找到（可选，build 脚本直接调 g++ 不依赖它）  [WARN]");
    }

    // 3. 虚拟环境与引擎脚本
    if (fs::exists(ws.venv_python)) {
        log.line("doctor", "虚拟环境: " + ws.venv_python.string() + "  [OK]");
    } else {
        log.line("doctor", "虚拟环境缺失，请先运行 " + std::string(kSetupScript) + "  [FAIL]");
        ++problems;
    }
    if (fs::exists(ws.worker)) {
        log.line("doctor", "训练引擎: " + ws.worker.string() + "  [OK]");
    } else {
        log.line("doctor", "训练引擎缺失: " + ws.worker.string() + "  [FAIL]");
        ++problems;
    }
    const fs::path data_yaml = resolve_data_yaml(ws, opt);
    const bool data_ok = !data_yaml.empty() && fs::is_regular_file(data_yaml);
    if (data_ok) {
        log.line("doctor", "数据集配置: " + data_yaml.string() + "  [OK]");
    } else {
        log.line("doctor", "数据集配置缺失: " + data_yaml_hint(opt) +
                               "（--data 可指定 yaml 文件或数据集目录）  [FAIL]");
        ++problems;
    }

    // 4. 引擎自检（Python/torch/ultralytics/数据集统计/权重缓存）
    if (fs::exists(ws.venv_python) && fs::exists(ws.worker) && data_ok) {
        RunResult r = run_process(ws.venv_python,
                                  {"-u", "-X", "utf8", ws.worker.string(), "check",
                                   "--data", data_yaml.string()},
                                  "doctor");
        std::string err = find_marker(r.lines, "__ERROR__");
        if (r.exit_code != 0 || !err.empty()) {
            log.line("doctor", "引擎自检失败: " + (err.empty() ? std::string("退出码 " +
                                                     std::to_string(r.exit_code)) : err) +
                                   "  [FAIL]");
            ++problems;
        } else {
            log.line("doctor", "引擎自检  [OK]");
        }
    }

    // 5. GPU 探测
    DeviceInfo gpu = detect_gpu();
    if (gpu.nvidia_smi_ok) {
        log.line("doctor", "GPU: " + gpu.gpu_name + "（训练将优先使用 GPU）  [OK]");
    } else {
        log.line("doctor", "GPU: 未检测到 NVIDIA GPU（训练将使用 CPU）  [INFO]");
    }

    log.raw("====================================================");
    if (problems > 0) {
        log.line("doctor", "自检完成：发现 " + std::to_string(problems) +
                               " 个问题，请先运行 " + std::string(kSetupScript) + " 修复");
        return 3;
    }
    log.line("doctor", "自检完成：一切就绪，可以开始训练");
    return 0;
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------
static int cmd_export(const Workspace& ws, const Options& opt) {
    Logger& log = Logger::instance();

    fs::path src_pt = opt.weights;
    std::string dir_name;
    if (!src_pt.empty()) {
        if (!fs::exists(src_pt)) {
            log.line("main", "权重文件不存在: " + src_pt.string());
            return 3;
        }
        dir_name = model_dir_name(canonical_model(opt.model));
        if (dir_name.empty()) dir_name = src_pt.parent_path().filename().string();
    } else {
        std::string canonical = canonical_model(opt.model);
        if (canonical.empty()) {
            if (opt.interactive) canonical = interactive_select_model();
            if (canonical.empty()) {
                log.line("main", "未指定模型，使用 --model 或交互菜单选择");
                print_usage();
                return 2;
            }
        }
        dir_name = model_dir_name(canonical);
        src_pt = ws.weights / dir_name / "best.pt";
        if (!fs::exists(src_pt)) {
            log.line("main", "未找到 " + src_pt.string() +
                                 "，请先训练该模型或用 --weights 指定权重");
            return 3;
        }
    }

    if (!fs::exists(ws.venv_python)) {
        log.line("main", "未找到虚拟环境 Python：" + ws.venv_python.string() +
                             "，请先运行 " + std::string(kSetupScript));
        return 3;
    }

    std::string stamp = now_dir_stamp();
    fs::path log_dir = ws.logs / (dir_name + "_export_" + stamp);
    fs::create_directories(log_dir);
    log.open(log_dir / "export.log");
    log.raw("============================================================");
    log.raw(" YOLO 训练工具 v" + std::string(kVersion) + " —— ONNX 导出 " + dir_name);
    log.raw("============================================================");
    log.line("export", "源权重: " + src_pt.string());

    RunResult r = run_process(ws.venv_python,
                              {"-u", "-X", "utf8", ws.worker.string(), "export",
                               "--weights", src_pt.string(),
                               "--imgsz", std::to_string(opt.val_imgsz)},
                              "engine");
    if (r.exit_code != 0) {
        std::string err = find_marker(r.lines, "__ERROR__");
        log.line("main", "导出失败: " + (err.empty() ? std::string("退出码 " +
                                            std::to_string(r.exit_code)) : err));
        return 4;
    }

    std::string onnx = find_marker(r.lines, "__ONNX__");
    if (onnx.empty()) {
        log.line("main", "引擎输出缺少 __ONNX__ 标记");
        return 4;
    }
    log.line("main", "ONNX 已生成: " + onnx);

    // 若源权重就在 weights/<模型>/ 下，onnx 已由引擎导出到同目录；否则归档一份
    fs::path canonical_onnx = src_pt.parent_path() / "best.onnx";
    if (fs::path(onnx) != canonical_onnx) {
        if (!copy_artifact(fs::path(onnx), canonical_onnx)) {
            log.line("main", "归档 ONNX 失败: " + canonical_onnx.string());
            return 5;
        }
        log.line("main", "ONNX 已归档: " + canonical_onnx.string());
    }
    log.line("main", "导出完成");
    return 0;
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------
static int cmd_train(const Workspace& ws, const Options& opt) {
    Logger& log = Logger::instance();

    // ---- 模型选择：命令行优先，否则交互菜单 ----
    std::string canonical = canonical_model(opt.model);
    bool menu_used = false;
    if (canonical.empty()) {
        if (opt.interactive) {
            canonical = interactive_select_model();
            menu_used = !canonical.empty();
        }
        if (canonical.empty()) {
            log.line("main", "未指定有效模型（yolov8n / yolo11n / yolo26n），"
                             "请用 --model 指定或通过交互菜单选择");
            print_usage();
            return 2;
        }
    }
    std::string dir_name = model_dir_name(canonical);

    // ---- 数据集配置：--data 优先，否则用项目内自动探测到的数据集 ----
    const fs::path data_yaml = resolve_data_yaml(ws, opt);

    // ---- 环境预检 ----
    if (!ws.valid()) {
        log.line("main", "未找到项目根目录，可用 --root 指定");
        return 3;
    }
    if (!fs::exists(ws.venv_python)) {
        log.line("main", "未找到虚拟环境 Python：" + ws.venv_python.string() +
                             "，请先运行 " + std::string(kSetupScript));
        return 3;
    }
    if (data_yaml.empty() || !fs::is_regular_file(data_yaml)) {
        log.line("main", "数据集配置不存在: " + data_yaml_hint(opt) +
                             "（--data 可指定 yaml 文件或数据集目录）");
        return 3;
    }

    // ---- 交互模式下的参数确认 ----
    if (menu_used) {
        log.raw("");
        log.raw("训练配置：");
        log.raw("  模型:     " + dir_name + "（COCO 预训练迁移学习）");
        log.raw("  设备:     auto（有 GPU 用 GPU，无 GPU 用 CPU）");
        log.raw("  数据集:   " + data_yaml.string());
        log.raw("  超参数:   epochs=" + std::to_string(opt.epochs) +
                " imgsz=" + std::to_string(opt.imgsz) +
                " batch=" + std::to_string(opt.batch) +
                " patience=30 optimizer=AdamW cos_lr");
        std::printf("确认开始训练？[Y/n] ");
        std::fflush(stdout);
        std::string answer;
        if (std::getline(std::cin, answer)) {
            if (!answer.empty() && (answer[0] == 'n' || answer[0] == 'N')) {
                log.line("main", "已取消");
                return 0;
            }
        }
    }

    // ---- 日志目录 logs/<模型>_<时间戳>/ ----
    std::string stamp = now_dir_stamp();
    fs::path log_dir = ws.logs / (dir_name + "_" + stamp);
    fs::create_directories(log_dir);
    log.open(log_dir / "train.log");

    log.raw("============================================================");
    log.raw(" YOLO 训练工具 v" + std::string(kVersion) + " —— " + dir_name);
    log.raw("============================================================");
    log.line("config", "项目根目录: " + ws.root.string());
    log.line("config", "模型: " + canonical + ".pt（COCO 预训练迁移学习）");
    log.line("config", "数据集: " + data_yaml.string());
    log.line("config", "设备策略: " + opt.device + "（有 GPU 优先 GPU，无 GPU 用 CPU）");
    log.line("config", "超参数: epochs=" + std::to_string(opt.epochs) +
                " imgsz=" + std::to_string(opt.imgsz) +
                " batch=" + std::to_string(opt.batch) +
                " workers=" + (opt.workers > 0 ? std::to_string(opt.workers) : "auto") +
                " val_imgsz=" + std::to_string(opt.val_imgsz) +
                " optimizer=AdamW lr0=0.001 cos_lr=true patience=30");
    log.line("config", "训练日志: " + (log_dir / "train.log").string());

    DeviceInfo gpu = detect_gpu();
    if (gpu.nvidia_smi_ok) {
        log.line("device", "检测到 NVIDIA GPU: " + gpu.gpu_name + "，引擎将优先使用 GPU 训练");
    } else {
        log.line("device", "未检测到 NVIDIA GPU，引擎将使用 CPU 训练");
    }

    // ---- 调用训练引擎 ----
    std::vector<std::string> args = {
        "-u", "-X", "utf8", ws.worker.string(),
        "train",
        "--model", canonical + ".pt",
        "--data", data_yaml.string(),
        "--device", opt.device,
        "--epochs", std::to_string(opt.epochs),
        "--imgsz", std::to_string(opt.imgsz),
        "--batch", std::to_string(opt.batch),
        "--workers", std::to_string(opt.workers),
        "--val-imgsz", std::to_string(opt.val_imgsz),
    };
    if (opt.resume) args.push_back("--resume");

    RunResult r = run_process(ws.venv_python, args, "engine");
    if (r.exit_code != 0) {
        std::string err = find_marker(r.lines, "__ERROR__");
        log.line("main", "训练失败: " + (err.empty() ? std::string("引擎退出码 " +
                                            std::to_string(r.exit_code)) : err));
        return 4;
    }

    std::string run_dir = find_marker(r.lines, "__RUN_DIR__");
    std::string best_pt = find_marker(r.lines, "__BEST_PT__");
    std::string onnx = find_marker(r.lines, "__ONNX__");
    std::string metrics = find_marker(r.lines, "__METRICS_JSON__");
    if (best_pt.empty() || onnx.empty() || metrics.empty()) {
        log.line("main", "引擎输出缺少关键标记（best.pt / onnx / metrics）");
        return 4;
    }

    // ---- 归档产物到 weights/<模型>/ ----
    fs::path wdir = ws.weights / dir_name;
    if (!copy_artifact(fs::path(best_pt), wdir / "best.pt")) {
        log.line("main", "归档 best.pt 失败: " + best_pt);
        return 5;
    }
    if (!copy_artifact(fs::path(onnx), wdir / "best.onnx")) {
        log.line("main", "归档 best.onnx 失败: " + onnx);
        return 5;
    }

    // ---- metrics.json 落盘（引擎 JSON 原样保存） ----
    {
        std::ofstream out(log_dir / "metrics.json", std::ios::out | std::ios::trunc);
        out << metrics << "\n";
    }

    // ---- 汇总展示 ----
    std::string summary_payload = find_marker(r.lines, "__SUMMARY__");
    auto kv = parse_summary(summary_payload);
    const std::string* v = nullptr;

    log.raw("");
    log.raw("================ 训练完成 ================");
    log.raw("  模型:       " + dir_name);
    if ((v = summary_get(kv, "device"))) {
        std::string gpu_desc;
        const std::string* g = summary_get(kv, "gpu");
        if (g && *g != "-") gpu_desc = "（" + *g + "）";
        log.raw("  设备:       " + *v + gpu_desc);
    }
    if ((v = summary_get(kv, "mAP50"))) log.raw("  mAP50:      " + *v);
    if ((v = summary_get(kv, "mAP50-95"))) log.raw("  mAP50-95:   " + *v);
    if ((v = summary_get(kv, "precision"))) log.raw("  Precision:  " + *v);
    if ((v = summary_get(kv, "recall"))) log.raw("  Recall:     " + *v);
    if ((v = summary_get(kv, "epochs"))) log.raw("  轮数:       " + *v);
    if ((v = summary_get(kv, "train_seconds"))) log.raw("  总用时:     " + *v + " s");
    char sizes[256] = {0};
    std::snprintf(sizes, sizeof(sizes), "  best.pt:    weights/%s/best.pt（%.1f MB）",
                  dir_name.c_str(), file_size_mb(wdir / "best.pt"));
    log.raw(sizes);
    std::snprintf(sizes, sizeof(sizes), "  best.onnx:  weights/%s/best.onnx（%.1f MB）",
                  dir_name.c_str(), file_size_mb(wdir / "best.onnx"));
    log.raw(sizes);
    log.raw("  训练日志:   " + (log_dir / "train.log").string());
    log.raw("  指标文件:   " + (log_dir / "metrics.json").string());
    if (!run_dir.empty()) log.raw("  过程产物:   " + run_dir);
    log.raw("===========================================");
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
#ifdef _WIN32
    // 控制台与子进程统一 UTF-8，保证中文日志不乱码（Linux/macOS 默认即 UTF-8）
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    Options opt = parse_args(argc, argv);
    if (opt.show_help) {
        print_usage();
        return 0;
    }

    Workspace ws = Workspace::discover(self_exe_path(argc > 0 ? argv[0] : nullptr),
                                       opt.root_override);

    if (opt.mode == "list-models") {
        if (!ws.valid()) {
            Logger::instance().line("main", "未找到项目根目录，可用 --root 指定");
            return 3;
        }
        return cmd_list_models(ws);
    }
    if (opt.mode == "doctor") return cmd_doctor(ws, opt);
    if (opt.mode == "download") {
        if (!ws.valid()) {
            Logger::instance().line("main", "未找到项目根目录，可用 --root 指定");
            return 3;
        }
        return cmd_download(ws, opt);
    }
    if (opt.mode == "export") {
        if (!ws.valid()) {
            Logger::instance().line("main", "未找到项目根目录，可用 --root 指定");
            return 3;
        }
        return cmd_export(ws, opt);
    }
    return cmd_train(ws, opt);
}
