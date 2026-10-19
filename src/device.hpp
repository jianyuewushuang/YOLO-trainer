#pragma once
// device.hpp —— GPU 探测（仅用于日志展示与自检；最终训练设备由引擎按 torch.cuda 判定）
#include <string>

struct DeviceInfo {
    bool nvidia_smi_ok = false;  // 系统存在可用的 nvidia-smi
    std::string gpu_name;        // GPU 名称（nvidia-smi 查询结果）
};

DeviceInfo detect_gpu();
