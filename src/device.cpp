#include "device.hpp"

#include "worker.hpp"

#include <cctype>

DeviceInfo detect_gpu() {
    DeviceInfo info;
    // nvidia-smi 不存在或执行失败时退出码非 0，视为无 GPU（Windows/Linux 一致）
    RunResult r = run_capture_result("nvidia-smi --query-gpu=name --format=csv,noheader");
    if (r.exit_code != 0) return info;

    // 取第一行非空内容作为 GPU 名称
    for (const auto& line : r.lines) {
        std::size_t begin = 0;
        while (begin < line.size() &&
               std::isspace(static_cast<unsigned char>(line[begin]))) {
            ++begin;
        }
        if (begin >= line.size()) continue;
        std::size_t end = line.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(line[end - 1]))) {
            --end;
        }
        info.nvidia_smi_ok = true;
        info.gpu_name = line.substr(begin, end - begin);
        break;
    }
    return info;
}
