#include "logger.hpp"

#include <windows.h>

#include <cstdio>

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

void Logger::open(const fs::path& path) {
    file_path_ = path;
    if (!path.empty()) {
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        file_.open(path, std::ios::out | std::ios::trunc | std::ios::binary);
    }
}

void Logger::close() {
    if (file_.is_open()) file_.close();
}

static std::string now_stamp() {
    SYSTEMTIME st{};
    GetLocalTime(&st);
    char buf[40] = {0};
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  st.wYear, st.wMonth, st.wDay,
                  st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return buf;
}

void Logger::line(const std::string& tag, const std::string& msg) {
    std::string out = "[" + now_stamp() + "][" + tag + "] " + msg + "\n";
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
    if (file_.is_open()) {
        file_ << out;
        file_.flush();
    }
}

void Logger::raw(const std::string& msg) {
    std::fputs(msg.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    if (file_.is_open()) {
        file_ << msg << "\n";
        file_.flush();
    }
}
