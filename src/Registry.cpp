#include <ocl/Registry.hpp>
#include <ocl/Device.hpp>
#include <ocl/Platform.hpp>
#include <iostream>

namespace ocl {

Registry& Registry::instance() {
    static Registry registry;
    if (!registry.initialized_) {
        registry.initialize();
    }
    return registry;
}

void Registry::initialize() {
    if (initialized_) return;
    
    platforms_ = Platform::getAll();
    initialized_ = true;
}

std::vector<Device> Registry::getAllDevices() const {
    std::vector<Device> allDevices;
    for (const auto& platform : platforms_) {
        auto devices = Device::getAll(platform);
        allDevices.insert(allDevices.end(), devices.begin(), devices.end());
    }
    return allDevices;
}

std::vector<Device> Registry::getDevicesByType(cl_device_type type) const {
    std::vector<Device> typedDevices;
    for (const auto& platform : platforms_) {
        auto devices = Device::getAll(platform, type);
        typedDevices.insert(typedDevices.end(), devices.begin(), devices.end());
    }
    return typedDevices;
}

Device Registry::getDefaultDevice() const {
    // Try GPU first
    auto gpus = getDevicesByType(CL_DEVICE_TYPE_GPU);
    if (!gpus.empty()) {
        return gpus[0];
    }
    
    // Fall back to any device
    auto allDevices = getAllDevices();
    if (allDevices.empty()) {
        throw Error(CL_DEVICE_NOT_FOUND, "no devices found in registry");
    }
    return allDevices[0];
}

const Platform& Registry::getPlatform(size_t index) const {
    if (index >= platforms_.size()) {
        throw std::out_of_range("Platform index out of range");
    }
    return platforms_[index];
}

size_t Registry::getDeviceCount() const {
    size_t count = 0;
    for (const auto& platform : platforms_) {
        auto devices = Device::getAll(platform);
        count += devices.size();
    }
    return count;
}

void Registry::printInfo() const {
    std::cout << "OpenCL Registry\n";
    std::cout << "===============\n";
    std::cout << "Platforms: " << platforms_.size() << "\n";
    std::cout << "Total Devices: " << getDeviceCount() << "\n\n";
    
    for (size_t i = 0; i < platforms_.size(); ++i) {
        const auto& platform = platforms_[i];
        std::cout << "Platform " << i << ": " << platform.getName() << "\n";
        std::cout << "  Vendor: " << platform.getVendor() << "\n";
        
        auto devices = Device::getAll(platform);
        std::cout << "  Devices: " << devices.size() << "\n";
        
        for (size_t j = 0; j < devices.size(); ++j) {
            const auto& device = devices[j];
            std::cout << "    Device " << j << ": " << device.getName() << "\n";
            std::cout << "      Memory: " << device.getGlobalMemSize() / (1024 * 1024) << " MB\n";
            std::cout << "      Compute Units: " << device.getMaxComputeUnits() << "\n";
        }
        std::cout << "\n";
    }
}

} // namespace ocl

