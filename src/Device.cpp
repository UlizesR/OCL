#include <ocl/Device.hpp>
#include <ocl/Platform.hpp>

namespace ocl {

Device::Device() : id_(nullptr) {}

Device::Device(cl_device_id id) : id_(id) {}

std::vector<Device> Device::getAll(const Platform& platform, cl_device_type type) {
    cl_uint count;
    cl_int err = clGetDeviceIDs(platform.id(), type, 0, nullptr, &count);
    
    if (err == CL_DEVICE_NOT_FOUND) {
        return {};
    }
    checkError(err, "getting device count");
    
    std::vector<cl_device_id> ids(count);
    err = clGetDeviceIDs(platform.id(), type, count, ids.data(), nullptr);
    checkError(err, "getting device IDs");
    
    std::vector<Device> devices;
    devices.reserve(count);
    for (auto id : ids) {
        devices.emplace_back(id);
    }
    return devices;
}

Device Device::getDefault(const Platform& platform) {
    auto devices = getAll(platform, CL_DEVICE_TYPE_GPU);
    if (devices.empty()) {
        devices = getAll(platform, CL_DEVICE_TYPE_ALL);
    }
    if (devices.empty()) {
        throw Error(CL_DEVICE_NOT_FOUND, "no devices found");
    }
    return devices[0];
}

Device Device::getDefault() {
    return getDefault(Platform::getDefault());
}

std::string Device::getName() const {
    return ocl::getInfoString(id_, CL_DEVICE_NAME);
}

std::string Device::getVendor() const {
    return ocl::getInfoString(id_, CL_DEVICE_VENDOR);
}

std::string Device::getVersion() const {
    return ocl::getInfoString(id_, CL_DEVICE_VERSION);
}

cl_device_type Device::getType() const {
    return getInfo<cl_device_type>(CL_DEVICE_TYPE);
}

cl_ulong Device::getGlobalMemSize() const {
    return getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
}

cl_ulong Device::getLocalMemSize() const {
    return getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);
}

cl_uint Device::getMaxComputeUnits() const {
    return getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
}

cl_uint Device::getMaxWorkGroupSize() const {
    return getInfo<cl_uint>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

std::string Device::getInfoString(cl_device_info param) const {
    return ocl::getInfoString(id_, param);
}

bool Device::isGPU() const {
    return (getType() & CL_DEVICE_TYPE_GPU) != 0;
}

bool Device::isCPU() const {
    return (getType() & CL_DEVICE_TYPE_CPU) != 0;
}

bool Device::isAccelerator() const {
    return (getType() & CL_DEVICE_TYPE_ACCELERATOR) != 0;
}

} // namespace ocl

