#include "worker.hpp"

#include "logger.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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

// 以 "__" 开头的行为 C++/引擎之间的结构化标记（如 __METRICS_JSON__），
// 只参与解析，不打印到控制台/日志，避免污染人类可读输出。
bool is_marker_line(const std::string& line) {
    return line.size() > 2 && line[0] == '_' && line[1] == '_';
}

// 去掉 ANSI 转义序列（子进程的颜色/光标控制码），保证日志文件可读
std::string strip_ansi(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\x1b' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        const char kind = s[i + 1];
        if (kind == '[') {  // CSI：ESC [ 参数 终止字节(0x40~0x7E)
            i += 2;
            while (i < s.size() && !(s[i] >= '@' && s[i] <= '~')) ++i;
        } else if (kind == ']') {  // OSC：ESC ] ... BEL 或 ST 结束
            i += 2;
            while (i < s.size() && s[i] != '\a' &&
                   !(s[i] == '\x1b' && i + 1 < s.size() && s[i + 1] == '\\')) {
                ++i;
            }
            if (i < s.size() && s[i] == '\x1b') ++i;
        } else {
            ++i;  // 其它两字符转义
        }
    }
    return out;
}

// 把已切分的输出行写入结果集与日志
void emit_line(RunResult& result, const std::string& log_tag, const std::string& line) {
    if (line.empty()) return;  // 跳过进度条产生的空片段
    const std::string clean = strip_ansi(line);
    result.lines.push_back(clean);
    if (!log_tag.empty() && !is_marker_line(clean)) {
        Logger::instance().line(log_tag, clean);
    }
}

#ifdef _WIN32
// ------------------------------- Windows ----------------------------------
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
        for (auto& line : split_lines(buffer)) emit_line(result, log_tag, line);
    }
    // 收尾：最后一段不完整输出
    if (!buffer.empty()) emit_line(result, log_tag, buffer);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    result.exit_code = static_cast<int>(code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(pipe.read);
    return result;
}

#else
// ------------------------------- POSIX ------------------------------------

// 内部实现：fork + execvp，stdout/stderr 合入同一管道实时读取。
RunResult run_process_impl(const std::string& exe, const std::vector<std::string>& args,
                           const std::string& log_tag) {
    RunResult result;

    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
        if (!log_tag.empty()) {
            Logger::instance().line(log_tag, std::string("pipe 失败: ") + std::strerror(errno));
        }
        return result;
    }

    // 子进程会继承父进程的 stdio 缓冲，fork 前先刷出，避免重复输出
    std::fflush(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        const int err = errno;
        ::close(fds[0]);
        ::close(fds[1]);
        if (!log_tag.empty()) {
            Logger::instance().line(log_tag, std::string("fork 失败: ") + std::strerror(err));
        }
        return result;
    }

    if (pid == 0) {
        // ---- 子进程：stdout/stderr 重定向到管道写端 ----
        ::close(fds[0]);
        if (dup2(fds[1], STDOUT_FILENO) < 0 || dup2(fds[1], STDERR_FILENO) < 0) _exit(127);
        if (fds[1] > STDERR_FILENO) ::close(fds[1]);

        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(exe.c_str()));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);

        execvp(exe.c_str(), argv.data());  // 含 '/' 时按路径执行，否则搜索 PATH
        std::fprintf(stderr, "exec %s 失败: %s\n", exe.c_str(), std::strerror(errno));
        _exit(127);
    }

    // ---- 父进程：读管道直到 EOF ----
    ::close(fds[1]);
    std::string buffer;
    char chunk[8192];
    for (;;) {
        const ssize_t got = read(fds[0], chunk, sizeof(chunk));
        if (got < 0) {
            if (errno == EINTR) continue;
            if (!log_tag.empty()) {
                Logger::instance().line(log_tag, std::string("读取管道失败: ") +
                                                     std::strerror(errno));
            }
            break;
        }
        if (got == 0) break;
        buffer.append(chunk, static_cast<std::size_t>(got));
        for (auto& line : split_lines(buffer)) emit_line(result, log_tag, line);
    }
    // 收尾：最后一段不完整输出
    if (!buffer.empty()) emit_line(result, log_tag, buffer);
    ::close(fds[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }
    return result;
}
#endif

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

// cmdline 已含参数；拆成 exe + args 复用同一实现
static RunResult run_cmdline(const std::string& cmdline) {
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
    return run_process_impl(exe, args, "");
}

std::string run_capture(const std::string& cmdline) {
    RunResult r = run_cmdline(cmdline);
    std::string out;
    for (const auto& l : r.lines) out += l + "\n";
    return out;
}

RunResult run_capture_result(const std::string& cmdline) {
    return run_cmdline(cmdline);
}
