#pragma once

#include <ocl/Errors.hpp>

namespace ocl {

// Forward declarations
class Context;
class Device;

// ============================================================================
// CommandQueue - manages OpenCL command queue with RAII
// ============================================================================

class CommandQueue {
public:
    CommandQueue();
    CommandQueue(const Context& context, const Device& device, 
                 cl_command_queue_properties properties = 0);
    
    ~CommandQueue();
    
    // Disable copying
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;
    
    // Enable moving
    CommandQueue(CommandQueue&& other) noexcept;
    CommandQueue& operator=(CommandQueue&& other) noexcept;
    
    // Queue operations
    void finish();
    void flush();
    
    // Get underlying queue
    cl_command_queue get() const { return queue_; }
    
private:
    cl_command_queue queue_;
};

} // namespace ocl

