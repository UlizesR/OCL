#pragma once

#include <ocl/Errors.hpp>
#include <vector>

namespace ocl {

// Forward declaration
class Platform;

// ============================================================================
// Device - represents an OpenCL device
// ============================================================================

class Device {
public:
    Device();
    explicit Device(cl_device_id id);
    
    // Get all devices of a specific type from a platform
    static std::vector<Device> getAll(const Platform& platform, 
                                       cl_device_type type = CL_DEVICE_TYPE_ALL);
    
    // Get the default device (first GPU, or first device of any type)
    static Device getDefault(const Platform& platform);
    static Device getDefault();  // Uses default platform
    
    // Query device information
    std::string getName() const;
    std::string getVendor() const;
    std::string getVersion() const;
    cl_device_type getType() const;
    cl_ulong getGlobalMemSize() const;
    cl_ulong getLocalMemSize() const;
    cl_uint getMaxComputeUnits() const;
    cl_uint getMaxWorkGroupSize() const;
    
    // Device type predicates
    bool isGPU() const;
    bool isCPU() const;
    bool isAccelerator() const;
    
    // Get underlying device ID
    cl_device_id id() const { return id_; }
    
private:
    cl_device_id id_;
    
    template<typename T>
    T getInfo(cl_device_info param) const {
        T value;
        cl_int err = clGetDeviceInfo(id_, param, sizeof(T), &value, nullptr);
        checkError(err, "getting device info");
        return value;
    }
    
    std::string getInfoString(cl_device_info param) const;
};

} // namespace ocl

