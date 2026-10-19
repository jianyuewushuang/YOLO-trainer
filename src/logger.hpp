#pragma once
// logger.hpp —— 统一日志：控制台 + 文件双写，带毫秒时间戳
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

class Logger {
public:
    static Logger& instance();

    // 打开日志文件（截断写）；path 为空则只写控制台
    void open(const fs::path& path);
    void close();
    bool has_file() const { return file_.is_open(); }
    const fs::path& file() const { return file_path_; }

    // 写一行：[YYYY-MM-DD HH:MM:SS.mmm][tag] msg
    void line(const std::string& tag, const std::string& msg);

    // 不带 tag 的原始行（用于 banner / 空行）
    void raw(const std::string& msg);

private:
    Logger() = default;
    fs::path file_path_;
    std::ofstream file_;
};
