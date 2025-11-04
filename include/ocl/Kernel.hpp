#pragma once

#include <ocl/Errors.hpp>
#include <string>
#include <utility>

namespace ocl {

// Forward declarations
class Program;
class CommandQueue;
class Device;
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
    
    // Set argument for Buffer<T> (convenience overload)
    template<typename T>
    void setArg(cl_uint index, const Buffer<T>& buffer) {
        setArg(index, buffer.get());
    }
    
    // Set local memory argument (size in bytes)
    void setLocalArg(cl_uint index, size_t size_in_bytes);
    
    // Set multiple arguments at once (variadic)
    template<typename... Args>
    void setArgs(Args&&... args) {
        setArgsImpl(0, std::forward<Args>(args)...);
    }
    
    // Execute kernel (1D)
    void execute(const CommandQueue& queue, size_t global_work_size, size_t local_work_size = 0);
    
    // Execute kernel (2D)
    void execute2D(const CommandQueue& queue, size_t global_width, size_t global_height,size_t local_width = 0, size_t local_height = 0);
    
    // Execute kernel (3D)
    void execute3D(const CommandQueue& queue, size_t global_x, size_t global_y, size_t global_z, size_t local_x = 0, size_t local_y = 0, size_t local_z = 0);
    
    // Get kernel work group info for a device
    size_t getWorkGroupSize(const Device& device) const;
    size_t getPreferredWorkGroupSizeMultiple(const Device& device) const;
    cl_ulong getLocalMemSize(const Device& device) const;
    
    // Get underlying kernel
    cl_kernel get() const { return kernel_; }
    
private:
    cl_kernel kernel_;
    
    // Helper for variadic setArgs - base case
    void setArgsImpl(cl_uint) {}
    
    // Helper for variadic setArgs - recursive case
    template<typename First, typename... Rest>
    void setArgsImpl(cl_uint index, First&& first, Rest&&... rest) {
        setArg(index, std::forward<First>(first));
        setArgsImpl(index + 1, std::forward<Rest>(rest)...);
    }
};

} // namespace ocl

