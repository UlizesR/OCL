#pragma once

#include <ocl/Errors.hpp>
#include <cstddef>
#include <array>
#include <algorithm>

namespace ocl {

// Forward declaration
class Device;
class Kernel;

// ============================================================================
// NDRange - Utilities for work group size calculations
// ============================================================================

class NDRange {
public:
    // Calculate optimal 1D work group size
    static size_t getOptimal1D(const Kernel& kernel, const Device& device, size_t global_size);
    
    // Calculate optimal 2D work group sizes
    static std::array<size_t, 2> getOptimal2D(const Kernel& kernel, const Device& device, 
                                               size_t global_x, size_t global_y);
    
    // Calculate optimal 3D work group sizes
    static std::array<size_t, 3> getOptimal3D(const Kernel& kernel, const Device& device,
                                               size_t global_x, size_t global_y, size_t global_z);
    
    // Round up to nearest multiple (for padding global size)
    static size_t roundUp(size_t value, size_t multiple) {
        if (multiple == 0) return value;
        size_t remainder = value % multiple;
        if (remainder == 0) return value;
        return value + multiple - remainder;
    }
    
    // Find largest divisor <= max_value
    static size_t findBestDivisor(size_t number, size_t max_value);
    
    // Validate that global size is compatible with local size
    static bool isValidWorkSize(size_t global_size, size_t local_size) {
        return local_size > 0 && (global_size % local_size == 0);
    }
    
    // Get the preferred work group size multiple for a kernel
    static size_t getPreferredMultiple(const Kernel& kernel, const Device& device);
    
    // Calculate padded global size
    static size_t getPaddedGlobalSize(size_t desired_size, size_t local_size) {
        return roundUp(desired_size, local_size);
    }
};

} // namespace ocl

