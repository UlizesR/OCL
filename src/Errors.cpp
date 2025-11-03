#include <ocl/Errors.hpp>
#include <fstream>
#include <sstream>

namespace ocl {

std::string readFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string getInfoString(cl_platform_id platform, cl_platform_info param) {
    size_t size;
    cl_int err = clGetPlatformInfo(platform, param, 0, nullptr, &size);
    checkError(err, "getting platform info size");
    
    std::string result(size, '\0');
    err = clGetPlatformInfo(platform, param, size, &result[0], nullptr);
    checkError(err, "getting platform info");
    
    if (!result.empty() && result.back() == '\0') {
        result.pop_back();
    }
    return result;
}

std::string getInfoString(cl_device_id device, cl_device_info param) {
    size_t size;
    cl_int err = clGetDeviceInfo(device, param, 0, nullptr, &size);
    checkError(err, "getting device info size");
    
    std::string result(size, '\0');
    err = clGetDeviceInfo(device, param, size, &result[0], nullptr);
    checkError(err, "getting device info");
    
    if (!result.empty() && result.back() == '\0') {
        result.pop_back();
    }
    return result;
}

} // namespace ocl

