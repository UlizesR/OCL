#pragma once

#include <ocl/Errors.hpp>
#include <vector>

namespace ocl {

// Forward declarations
class Device;

// ============================================================================
// Context - manages OpenCL context with RAII
// ============================================================================

class Context {
public:
    Context();
    explicit Context(const Device& device);
    explicit Context(const std::vector<Device>& devices);
    
    ~Context();
    
    // Disable copying
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    
    // Enable moving
    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;
    
    // Get underlying context
    cl_context get() const { return context_; }
    
private:
    cl_context context_;
};

} // namespace ocl

