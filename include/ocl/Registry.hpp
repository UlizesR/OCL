#pragma once

#include <ocl/Errors.hpp>
#include <vector>

namespace ocl {

// Forward declarations
class Device;
class Platform;

// ============================================================================
// Registry - Central registry for OpenCL platforms and devices
// ============================================================================

class Registry {
public:
    // Get singleton instance
    static Registry& instance();
    
    // Initialize registry (enumerate platforms and devices)
    void initialize();
    
    // Get all platforms
    const std::vector<Platform>& getPlatforms() const { return platforms_; }
    
    // Get all devices (across all platforms)
    std::vector<Device> getAllDevices() const;
    
    // Get devices by type
    std::vector<Device> getDevicesByType(cl_device_type type) const;
    
    // Get default device (first GPU, or first device)
    Device getDefaultDevice() const;
    
    // Get platform by index
    const Platform& getPlatform(size_t index) const;
    
    // Get device count
    size_t getDeviceCount() const;
    
    // Print registry info
    void printInfo() const;
    
private:
    Registry() = default;
    std::vector<Platform> platforms_;
    bool initialized_ = false;
};

} // namespace ocl

