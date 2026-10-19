#include "device.hpp"

#include "worker.hpp"

DeviceInfo detect_gpu() {
    DeviceInfo info;
    // nvidia-smi 不在 PATH 时 CreateProcess 失败，视为无 GPU
    std::string out = run_capture("nvidia-smi --query-gpu=name --format=csv,noheader");
    if (!out.empty()) {
        info.nvidia_smi_ok = true;
        // 取第一行非空内容作为 GPU 名称
        std::size_t begin = out.find_first_not_of(" \t\r\n");
        if (begin != std::string::npos) {
            std::size_t end = out.find_first_of("\r\n", begin);
            info.gpu_name = out.substr(begin, end == std::string::npos ? std::string::npos
                                                                       : end - begin);
        }
    }
    return info;
}
