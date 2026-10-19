#include "worker.hpp"

#include "logger.hpp"

#include <windows.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

// 按 '\n' / '\r' 切行（兼容 \r\n 与进度条的裸 \r），剩余不完整内容留在 buffer。
std::vector<std::string> split_lines(std::string& buffer) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        char c = buffer[i];
        if (c != '\n' && c != '\r') continue;
        if (c == '\r' && i + 1 < buffer.size() && buffer[i + 1] == '\n') {
            lines.push_back(buffer.substr(start, i - start));
            ++i;
        } else {
            lines.push_back(buffer.substr(start, i - start));
        }
        start = i + 1;
    }
    buffer.erase(0, start);
    return lines;
}

struct PipePair {
    HANDLE read = nullptr;
    HANDLE write = nullptr;
};

bool create_output_pipe(PipePair& p) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;  // 子进程继承写端
    sa.lpSecurityDescriptor = nullptr;
    if (!CreatePipe(&p.read, &p.write, &sa, 0)) return false;
    // 读端不可继承，否则子进程持有读端会导致 EOF 永不触发
    SetHandleInformation(p.read, HANDLE_FLAG_INHERIT, 0);
    return true;
}

// 内部实现：log_tag 为空 = 静默采集，否则实时写日志。
RunResult run_process_impl(const std::string& exe, const std::vector<std::string>& args,
                           const std::string& log_tag) {
    RunResult result;

    std::string cmdline = quote_arg(exe);
    for (const auto& a : args) cmdline += " " + quote_arg(a);

    PipePair pipe;
    if (!create_output_pipe(pipe)) {
        if (!log_tag.empty()) {
            Logger::instance().line(log_tag, "CreatePipe 失败 (err=" +
                                                 std::to_string(GetLastError()) + ")");
        }
        return result;
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipe.write;
    si.hStdError = pipe.write;  // stderr 合并进 stdout
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE,
                             0, nullptr, nullptr, &si, &pi);
    CloseHandle(pipe.write);  // 父进程关闭写端，读端才能收到 EOF

    if (!ok) {
        CloseHandle(pipe.read);
        if (!log_tag.empty()) {
            Logger::instance().line(log_tag, "CreateProcess 失败: " + cmdline +
                                                 " (err=" + std::to_string(GetLastError()) + ")");
        }
        return result;
    }

    std::string buffer;
    char chunk[8192];
    DWORD got = 0;
    for (;;) {
        BOOL r = ReadFile(pipe.read, chunk, sizeof(chunk), &got, nullptr);
        if (!r || got == 0) break;
        buffer.append(chunk, got);
        for (auto& line : split_lines(buffer)) {
            if (line.empty()) continue;  // 跳过进度条产生的空片段
            result.lines.push_back(line);
            if (!log_tag.empty()) Logger::instance().line(log_tag, line);
        }
    }
    // 收尾：最后一段不完整输出
    if (!buffer.empty()) {
        result.lines.push_back(buffer);
        if (!log_tag.empty()) Logger::instance().line(log_tag, buffer);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    result.exit_code = static_cast<int>(code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(pipe.read);
    return result;
}

}  // namespace

std::string quote_arg(const std::string& arg) {
    if (arg.find_first_of(" \t\"") == std::string::npos) return arg;
    std::string s = "\"";
    for (char c : arg) {
        if (c == '"') s += "\\\"";
        else s += c;
    }
    s += "\"";
    return s;
}

RunResult run_process(const fs::path& exe, const std::vector<std::string>& args,
                      const std::string& log_tag) {
    return run_process_impl(exe.string(), args, log_tag);
}

std::string run_capture(const std::string& cmdline) {
    // cmdline 已含参数；拆成 exe + args 复用同一实现
    std::string exe, rest = cmdline;
    std::size_t sp = rest.find(' ');
    if (sp == std::string::npos) {
        exe = rest;
        rest.clear();
    } else {
        exe = rest.substr(0, sp);
        rest = rest.substr(sp + 1);
    }
    std::vector<std::string> args;
    std::size_t pos = 0;
    while (pos < rest.size()) {
        std::size_t next = rest.find(' ', pos);
        if (next == std::string::npos) {
            args.push_back(rest.substr(pos));
            break;
        }
        args.push_back(rest.substr(pos, next - pos));
        pos = next + 1;
    }
    RunResult r = run_process_impl(exe, args, "");
    std::string out;
    for (const auto& l : r.lines) out += l + "\n";
    return out;
}
