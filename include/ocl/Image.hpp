#pragma once

#include <ocl/Errors.hpp>

namespace ocl {

// Forward declarations
class Context;
class CommandQueue;

// ============================================================================
// Image - OpenCL image/texture wrapper (placeholder for future implementation)
// ============================================================================

class Image {
public:
    Image();
    // Future: Add image creation, reading, writing methods
    // Example: Image(Context, width, height, format)
    // Example: void write(CommandQueue, data)
    // Example: void read(CommandQueue, data)
    
private:
    cl_mem image_;
};

} // namespace ocl
