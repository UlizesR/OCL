#include <ocl/Context.hpp>
#include <ocl/Device.hpp>

namespace ocl {

Context::Context() : context_(nullptr) {}

Context::Context(const Device& device) {
    if (!device.id()) {
        throw std::invalid_argument("Cannot create context with invalid device");
    }
    
    cl_int err;
    cl_device_id device_id = device.id();
    context_ = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    checkError(err, "creating context");
}

Context::Context(const std::vector<Device>& devices) {
    if (devices.empty()) {
        throw std::invalid_argument("Cannot create context with empty device list");
    }
    
    std::vector<cl_device_id> device_ids;
    device_ids.reserve(devices.size());
    for (const auto& dev : devices) {
        if (!dev.id()) {
            throw std::invalid_argument("Cannot create context with invalid device");
        }
        device_ids.push_back(dev.id());
    }
    
    cl_int err;
    context_ = clCreateContext(nullptr, device_ids.size(), device_ids.data(), 
                              nullptr, nullptr, &err);
    checkError(err, "creating context");
}

Context::~Context() {
    if (context_) {
        clReleaseContext(context_);
    }
}

Context::Context(Context&& other) noexcept : context_(other.context_) {
    other.context_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        if (context_) {
            clReleaseContext(context_);
        }
        context_ = other.context_;
        other.context_ = nullptr;
    }
    return *this;
}

} // namespace ocl

