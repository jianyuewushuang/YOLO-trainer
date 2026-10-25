#pragma once
// worker.hpp —— 子进程执行：实时捕获 stdout/stderr 并写入日志
//
// Windows 实现：CreateProcess + 匿名管道；POSIX 实现：fork + execvp + 管道。
// 子进程的 stdout 与 stderr 合并后按行实时打印到控制台并写入日志文件，同时
// 收集全部行供调用方解析标记行（如 __METRICS_JSON__{...}）。标记行与 ANSI
// 控制码不会写入日志，保证日志可读。
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct RunResult {
    int exit_code = -1;
    std::vector<std::string> lines;  // 全部输出行（含标记行）
};

// 运行子进程并实时记录输出。log_tag 为空时静默运行（仅收集输出）。
RunResult run_process(const fs::path& exe,
                      const std::vector<std::string>& args,
                      const std::string& log_tag);

// 静默运行一条命令行，返回其标准输出（用于 nvidia-smi 等探测）。
std::string run_capture(const std::string& cmdline);

// 同上，但保留退出码与全部输出行（跨平台判断命令是否真正成功）。
RunResult run_capture_result(const std::string& cmdline);

// 按需给参数加引号（Windows CreateProcess 需要整行命令）。
std::string quote_arg(const std::string& arg);
