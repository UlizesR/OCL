#include <ocl/Platform.hpp>

namespace ocl {

Platform::Platform() : id_(nullptr) {}

Platform::Platform(cl_platform_id id) : id_(id) {}

std::vector<Platform> Platform::getAll() {
    cl_uint count;
    cl_int err = clGetPlatformIDs(0, nullptr, &count);
    checkError(err, "getting platform count");
    
    if (count == 0) {
        return {};
    }
    
    std::vector<cl_platform_id> ids(count);
    err = clGetPlatformIDs(count, ids.data(), nullptr);
    checkError(err, "getting platform IDs");
    
    std::vector<Platform> platforms;
    platforms.reserve(count);
    for (auto id : ids) {
        platforms.emplace_back(id);
    }
    return platforms;
}

Platform Platform::getDefault() {
    auto platforms = getAll();
    if (platforms.empty()) {
        throw Error(CL_DEVICE_NOT_FOUND, "no platforms found");
    }
    return platforms[0];
}

std::string Platform::getName() const {
    return ocl::getInfoString(id_, CL_PLATFORM_NAME);
}

std::string Platform::getVendor() const {
    return ocl::getInfoString(id_, CL_PLATFORM_VENDOR);
}

std::string Platform::getVersion() const {
    return ocl::getInfoString(id_, CL_PLATFORM_VERSION);
}

std::string Platform::getInfoString(cl_platform_info param) const {
    return ocl::getInfoString(id_, param);
}

} // namespace ocl

