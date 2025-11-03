#pragma once

#include <ocl/Errors.hpp>
#include <vector>

namespace ocl {

// ============================================================================
// Platform - represents an OpenCL platform
// ============================================================================

class Platform {
public:
    Platform();
    explicit Platform(cl_platform_id id);
    
    // Get all available platforms
    static std::vector<Platform> getAll();
    
    // Get the default (first) platform
    static Platform getDefault();
    
    // Query platform information
    std::string getName() const;
    std::string getVendor() const;
    std::string getVersion() const;
    
    // Get underlying platform ID
    cl_platform_id id() const { return id_; }
    
private:
    cl_platform_id id_;
    std::string getInfoString(cl_platform_info param) const;
};

} // namespace ocl

