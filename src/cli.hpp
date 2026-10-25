#pragma once
// cli.hpp —— 命令行解析、交互式模型选择、用法说明
#include <string>
#include <vector>

struct Options {
    std::string mode = "train";   // train | export | doctor | download | list-models
    // yolov8n | yolo11n | yolo26n | all；也支持 "yolov8n,yolo11n" 或多次 --model
    std::string model;
    std::string device = "auto";  // auto | cpu | 0 | 0,1 ...
    std::string data;             // 数据集 yaml 或数据集目录（空 = 项目内自动探测）
    std::string weights;          // export 模式：best.pt 路径
    int epochs = 100;
    int imgsz = 512;
    int batch = 16;
    int workers = 0;              // 0 = 自动
    int val_imgsz = 640;
    std::string root_override;    // --root 指定项目根目录
    bool resume = false;
    bool interactive = true;      // 未指定模型时是否进入交互菜单
    bool show_help = false;
};

// 规范化模型名（大小写不敏感，支持 v8n/8n 等别名）；不认识返回空串
std::string canonical_model(const std::string& raw);
// 规范模型目录名：YOLOv8n / YOLO11n / YOLO26n
std::string model_dir_name(const std::string& canonical);

// 解析模型列表：支持 all / "*" / 逗号或空格分隔的多个模型；按输入顺序去重
// ok=nullptr 时不报错；遇到无法识别的名字返回空列表并置 *ok=false
std::vector<std::string> parse_models(const std::string& raw, bool* ok = nullptr);

Options parse_args(int argc, char** argv);
void print_usage();
// 交互式选择模型（返回规范名，多选返回 "all"）；用户取消或读取失败返回空串
std::string interactive_select_model();
