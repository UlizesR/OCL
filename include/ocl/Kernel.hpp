#pragma once

#include <ocl/Errors.hpp>
#include <string>

namespace ocl {

// Forward declarations
class Program;
class CommandQueue;
template<typename T> class Buffer;

// ============================================================================
// Kernel - manages OpenCL kernel with RAII
// ============================================================================

class Kernel {
public:
    Kernel();
    Kernel(const Program& program, const std::string& name);
    
    ~Kernel();
    
    // Disable copying
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    
    // Enable moving
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;
    
    // Set argument by index (works for scalars)
    template<typename T>
    void setArg(cl_uint index, const T& value) {
        cl_int err = clSetKernelArg(kernel_, index, sizeof(T), &value);
        checkError(err, "setting kernel arg " + std::to_string(index));
    }
    
    // Set argument for cl_mem (buffers pass their cl_mem handle)
    void setArg(cl_uint index, cl_mem mem);
    
    // Set local memory argument (size in bytes)
    void setLocalArg(cl_uint index, size_t size_in_bytes);
    
    // Execute kernel (1D)
    void execute(const CommandQueue& queue, size_t global_work_size,
                 size_t local_work_size = 0);
    
    // Execute kernel (2D)
    void execute2D(const CommandQueue& queue, 
                   size_t global_width, size_t global_height,
                   size_t local_width = 0, size_t local_height = 0);
    
    // Get underlying kernel
    cl_kernel get() const { return kernel_; }
    
private:
    cl_kernel kernel_;
};

} // namespace ocl

