#include "logger.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#endif

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

// 本地时间戳：[YYYY-MM-DD HH:MM:SS.mmm]
static std::string now_stamp() {
    char buf[80] = {0};
#ifdef _WIN32
    SYSTEMTIME st{};
    GetLocalTime(&st);
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  st.wYear, st.wMonth, st.wDay,
                  st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
#else
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tmv{};
    localtime_r(&t, &tmv);
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec,
                  static_cast<int>(ms.count()));
#endif
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
