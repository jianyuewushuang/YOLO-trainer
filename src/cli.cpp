#include "cli.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static std::string to_lower(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return r;
}

std::string canonical_model(const std::string& raw) {
    const std::string s = to_lower(raw);
    if (s == "yolov8n" || s == "yolov8-n" || s == "v8n" || s == "8n") return "yolov8n";
    if (s == "yolo11n" || s == "yolo11-n" || s == "v11n" || s == "11n") return "yolo11n";
    if (s == "yolo26n" || s == "yolo26-n" || s == "v26n" || s == "26n") return "yolo26n";
    return "";
}

std::string model_dir_name(const std::string& canonical) {
    if (canonical == "yolov8n") return "YOLOv8n";
    if (canonical == "yolo11n") return "YOLO11n";
    if (canonical == "yolo26n") return "YOLO26n";
    return "";
}

std::vector<std::string> parse_models(const std::string& raw, bool* ok) {
    if (ok) *ok = true;
    std::vector<std::string> models;

    const std::string s = to_lower(raw);
    // 空白与分隔符当作分隔，支持 "yolov8n,yolo11n" / "v8n yolo11n" / "all"
    std::size_t start = 0;
    while (start <= s.size()) {
        std::size_t sep = s.find_first_of(",; \t", start);
        const std::string token = s.substr(
            start, sep == std::string::npos ? std::string::npos : sep - start);
        if (!token.empty()) {
            if (token == "all" || token == "*") {
                for (const char* m : {"yolov8n", "yolo11n", "yolo26n"}) {
                    if (std::find(models.begin(), models.end(), m) == models.end()) {
                        models.push_back(m);
                    }
                }
            } else {
                const std::string c = canonical_model(token);
                if (c.empty()) {
                    if (ok) *ok = false;
                    return {};
                }
                if (std::find(models.begin(), models.end(), c) == models.end()) {
                    models.push_back(c);
                }
            }
        }
        if (sep == std::string::npos) break;
        start = sep + 1;
    }
    return models;
}

static bool is_flag(const std::string& s) {
    return s.size() > 1 && s[0] == '-' && s[1] == '-';
}

static int to_int(const std::string& s, int fallback) {
    try {
        return std::stoi(s);
    } catch (...) {
        return fallback;
    }
}

Options parse_args(int argc, char** argv) {
    Options o;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "缺少参数值: %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--help" || k == "-h") {
            o.show_help = true;
        } else if (k == "--mode") {
            o.mode = need("--mode");
        } else if (k == "--model") {
            // 允许重复 --model：追加为逗号分隔列表（等价于 --model a,b）
            const std::string v = need("--model");
            o.model = o.model.empty() ? v : o.model + "," + v;
        } else if (k == "--device") {
            o.device = need("--device");
        } else if (k == "--data") {
            o.data = need("--data");
        } else if (k == "--weights") {
            o.weights = need("--weights");
        } else if (k == "--epochs") {
            o.epochs = to_int(need("--epochs"), o.epochs);
        } else if (k == "--imgsz") {
            o.imgsz = to_int(need("--imgsz"), o.imgsz);
        } else if (k == "--batch") {
            o.batch = to_int(need("--batch"), o.batch);
        } else if (k == "--workers") {
            o.workers = to_int(need("--workers"), o.workers);
        } else if (k == "--val-imgsz") {
            o.val_imgsz = to_int(need("--val-imgsz"), o.val_imgsz);
        } else if (k == "--root") {
            o.root_override = need("--root");
        } else if (k == "--resume") {
            o.resume = true;
        } else if (k == "--no-interactive") {
            o.interactive = false;
        } else if (!is_flag(k)) {
            positional.push_back(k);  // 位置参数：第一个=mode，第二个=model
        } else {
            std::fprintf(stderr, "未知参数: %s\n", k.c_str());
            std::exit(2);
        }
    }

    if (!positional.empty()) o.mode = positional[0];
    if (positional.size() >= 2) o.model = positional[1];

    if (o.mode != "train" && o.mode != "export" && o.mode != "doctor" &&
        o.mode != "download" && o.mode != "list-models") {
        std::fprintf(stderr, "未知模式: %s（train / export / doctor / download / list-models）\n",
                     o.mode.c_str());
        std::exit(2);
    }
    return o;
}

void print_usage() {
    std::printf(
        "yolo_train —— C++ YOLO 训练工具（YOLOv8n / YOLO11n / YOLO26n）\n"
        "\n"
        "用法：\n"
        "  yolo_train                        # 交互菜单选择模型后开始训练\n"
        "  yolo_train train --model yolov8n  # 命令行直接指定模型训练\n"
        "  yolo_train train --model all      # 依次训练 YOLOv8n / YOLO11n / YOLO26n\n"
        "  yolo_train export --model yolov8n # 由 weights/YOLOv8n/best.pt 重新导出 ONNX\n"
        "  yolo_train doctor                 # 环境与数据集自检\n"
        "  yolo_train download              # 预下载三种模型的预训练权重\n"
        "  yolo_train list-models           # 查看模型与权重产物状态\n"
        "\n"
        "常用参数：\n"
        "  --model   yolov8n | yolo11n | yolo26n | all   要训练的模型（逗号分隔可多选）\n"
        "  --device  auto | cpu | 0 ...           训练设备，auto=有 GPU 用 GPU，否则 CPU\n"
        "  --epochs  N        训练轮数（默认 100，配合早停 patience=30）\n"
        "  --imgsz   N        训练输入尺寸（默认 512）\n"
        "  --batch   N        批大小（默认 16）\n"
        "  --workers N        DataLoader 进程数（默认自动）\n"
        "  --val-imgsz N      验证/导出尺寸（默认 640）\n"
        "  --data    PATH     数据集 yaml 或数据集目录（默认自动探测 datasets/ 下的配置）\n"
        "  --weights PATH     export 模式指定 best.pt\n"
        "  --resume           从最近一次训练继续\n"
        "  --root    PATH     手动指定项目根目录\n"
        "  --no-interactive   禁用交互菜单\n"
        "\n"
        "产物：\n"
        "  weights/<模型>/{best.pt,best.onnx}   训练好的权重与 ONNX\n"
        "  logs/<模型>_<时间戳>/{train.log,metrics.json}   详细训练日志与指标\n"
        "  runs/detect/<实验名>/                Ultralytics 训练过程产物\n");
}

std::string interactive_select_model() {
    std::printf(
        "\n请选择要训练的模型：\n"
        "  [1] YOLOv8n    （Ultralytics YOLOv8 nano，约 3.2M 参数）\n"
        "  [2] YOLO11n    （Ultralytics YOLO11 nano）\n"
        "  [3] YOLO26n    （Ultralytics YOLO26 nano）\n"
        "  [4] 全部依次训练（YOLOv8n → YOLO11n → YOLO26n）\n"
        "请输入编号 [1-4]（或直接输入模型名，Ctrl+C 取消）：");
    std::fflush(stdout);

    std::string line;
    if (!std::getline(std::cin, line)) return "";
    // 去掉首尾空白
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    line.erase(line.begin(), std::find_if(line.begin(), line.end(), not_space));
    line.erase(std::find_if(line.rbegin(), line.rend(), not_space).base(), line.end());

    if (line == "1") return "yolov8n";
    if (line == "2") return "yolo11n";
    if (line == "3") return "yolo26n";
    if (line == "4") return "all";
    if (line == "all" || line == "全部") return "all";
    return canonical_model(line);
}
