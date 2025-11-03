#pragma once

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdexcept>
#include <string>

namespace ocl {

// ============================================================================
// OpenCL Error Handling
// ============================================================================

class Error : public std::runtime_error {
public:
    Error(cl_int code, const std::string& operation)
        : std::runtime_error("OpenCL error " + std::to_string(code) + " during: " + operation)
        , error_code_(code) {}
    
    cl_int code() const { return error_code_; }

private:
    cl_int error_code_;
};

// Helper to check errors and throw if needed
inline void checkError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        throw Error(err, operation);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

// Read entire file into a string (defined in Errors.cpp)
std::string readFile(const std::string& filepath);

// Get OpenCL info string (helper for Platform and Device)
std::string getInfoString(cl_platform_id platform, cl_platform_info param);
std::string getInfoString(cl_device_id device, cl_device_info param);

} // namespace ocl

