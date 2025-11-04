#include <ocl/NDRange.hpp>
#include <ocl/Device.hpp>
#include <ocl/Kernel.hpp>
#include <algorithm>
#include <cmath>

namespace ocl {

size_t NDRange::getOptimal1D(const Kernel& kernel, const Device& device, size_t global_size) {
    // Get kernel work group info
    size_t max_work_group = kernel.getWorkGroupSize(device);
    size_t preferred_multiple = kernel.getPreferredWorkGroupSizeMultiple(device);
    
    // Start with preferred multiple
    size_t local_size = preferred_multiple;
    
    // Scale up to a reasonable size (but not too large)
    while (local_size * 2 <= max_work_group && local_size * 2 <= 1024) {
        local_size *= 2;
    }
    
    // Find the largest divisor of global_size that's <= local_size
    size_t best_size = findBestDivisor(global_size, local_size);
    
    // Ensure it's a multiple of preferred_multiple
    if (preferred_multiple > 1) {
        best_size = (best_size / preferred_multiple) * preferred_multiple;
        if (best_size == 0) {
            best_size = preferred_multiple;
        }
    }
    
    return std::max(size_t(1), std::min(best_size, max_work_group));
}

std::array<size_t, 2> NDRange::getOptimal2D(const Kernel& kernel, const Device& device,
                                             size_t global_x, size_t global_y) {
    size_t max_work_group = kernel.getWorkGroupSize(device);
    size_t preferred_multiple = kernel.getPreferredWorkGroupSizeMultiple(device);
    
    // Common 2D work group sizes
    std::array<std::array<size_t, 2>, 6> candidates = {{
        {16, 16},   // 256 total - good balance
        {32, 8},    // 256 total - wider
        {8, 32},    // 256 total - taller
        {8, 8},     // 64 total - smaller
        {16, 8},    // 128 total
        {8, 16}     // 128 total
    }};
    
    // Find best candidate
    for (const auto& candidate : candidates) {
        size_t total = candidate[0] * candidate[1];
        if (total <= max_work_group &&
            (global_x % candidate[0] == 0) &&
            (global_y % candidate[1] == 0)) {
            return candidate;
        }
    }
    
    // Fallback: try to find any valid sizes
    size_t local_x = std::min(size_t(16), global_x);
    size_t local_y = std::min(size_t(16), global_y);
    
    // Adjust to make divisible
    while (global_x % local_x != 0 && local_x > 1) local_x--;
    while (global_y % local_y != 0 && local_y > 1) local_y--;
    
    // Ensure total doesn't exceed max
    while (local_x * local_y > max_work_group) {
        if (local_x > local_y) local_x /= 2;
        else local_y /= 2;
    }
    
    return {local_x, local_y};
}

std::array<size_t, 3> NDRange::getOptimal3D(const Kernel& kernel, const Device& device,
                                             size_t global_x, size_t global_y, size_t global_z) {
    size_t max_work_group = kernel.getWorkGroupSize(device);
    
    // For 3D, use smaller work groups
    std::array<std::array<size_t, 3>, 4> candidates = {{
        {8, 8, 4},   // 256 total
        {4, 8, 8},   // 256 total
        {8, 4, 4},   // 128 total
        {4, 4, 4}    // 64 total
    }};
    
    for (const auto& candidate : candidates) {
        size_t total = candidate[0] * candidate[1] * candidate[2];
        if (total <= max_work_group &&
            (global_x % candidate[0] == 0) &&
            (global_y % candidate[1] == 0) &&
            (global_z % candidate[2] == 0)) {
            return candidate;
        }
    }
    
    // Fallback
    size_t local_x = std::min(size_t(4), global_x);
    size_t local_y = std::min(size_t(4), global_y);
    size_t local_z = std::min(size_t(4), global_z);
    
    while (global_x % local_x != 0 && local_x > 1) local_x--;
    while (global_y % local_y != 0 && local_y > 1) local_y--;
    while (global_z % local_z != 0 && local_z > 1) local_z--;
    
    while (local_x * local_y * local_z > max_work_group) {
        if (local_x >= local_y && local_x >= local_z) local_x = std::max(size_t(1), local_x / 2);
        else if (local_y >= local_z) local_y = std::max(size_t(1), local_y / 2);
        else local_z = std::max(size_t(1), local_z / 2);
    }
    
    return {local_x, local_y, local_z};
}

size_t NDRange::findBestDivisor(size_t number, size_t max_value) {
    // Find largest divisor of number that is <= max_value
    for (size_t i = std::min(number, max_value); i >= 1; i--) {
        if (number % i == 0) {
            return i;
        }
    }
    return 1;
}

size_t NDRange::getPreferredMultiple(const Kernel& kernel, const Device& device) {
    return kernel.getPreferredWorkGroupSizeMultiple(device);
}

} // namespace ocl

