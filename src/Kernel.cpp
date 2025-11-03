#include <ocl/Kernel.hpp>
#include <ocl/Program.hpp>
#include <ocl/CommandQueue.hpp>

namespace ocl {

Kernel::Kernel() : kernel_(nullptr) {}

Kernel::Kernel(const Program& program, const std::string& name) {
    cl_int err;
    kernel_ = clCreateKernel(program.get(), name.c_str(), &err);
    checkError(err, "creating kernel: " + name);
}

Kernel::~Kernel() {
    if (kernel_) {
        clReleaseKernel(kernel_);
    }
}

Kernel::Kernel(Kernel&& other) noexcept : kernel_(other.kernel_) {
    other.kernel_ = nullptr;
}

Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        if (kernel_) {
            clReleaseKernel(kernel_);
        }
        kernel_ = other.kernel_;
        other.kernel_ = nullptr;
    }
    return *this;
}

void Kernel::setArg(cl_uint index, cl_mem mem) {
    cl_int err = clSetKernelArg(kernel_, index, sizeof(cl_mem), &mem);
    checkError(err, "setting kernel mem arg " + std::to_string(index));
}

void Kernel::setLocalArg(cl_uint index, size_t size_in_bytes) {
    cl_int err = clSetKernelArg(kernel_, index, size_in_bytes, nullptr);
    checkError(err, "setting kernel local memory arg " + std::to_string(index));
}

void Kernel::execute(const CommandQueue& queue, size_t global_work_size,
                     size_t local_work_size) {
    const size_t* local = (local_work_size > 0) ? &local_work_size : nullptr;
    cl_int err = clEnqueueNDRangeKernel(queue.get(), kernel_, 1, nullptr,
                                       &global_work_size, local, 0, nullptr, nullptr);
    checkError(err, "executing kernel");
}

void Kernel::execute2D(const CommandQueue& queue, 
                       size_t global_width, size_t global_height,
                       size_t local_width, size_t local_height) {
    size_t global[2] = {global_width, global_height};
    size_t local[2] = {local_width, local_height};
    const size_t* local_ptr = (local_width > 0 && local_height > 0) ? local : nullptr;
    
    cl_int err = clEnqueueNDRangeKernel(queue.get(), kernel_, 2, nullptr,
                                       global, local_ptr, 0, nullptr, nullptr);
    checkError(err, "executing kernel 2D");
}

} // namespace ocl

