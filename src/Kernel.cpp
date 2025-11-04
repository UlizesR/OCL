#include <ocl/Kernel.hpp>
#include <ocl/Program.hpp>
#include <ocl/CommandQueue.hpp>
#include <ocl/Device.hpp>

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

void Kernel::execute(const CommandQueue& queue, size_t global_work_size, size_t local_work_size) {
    // Validate work group size
    if (local_work_size > 0 && global_work_size % local_work_size != 0) {
        throw std::invalid_argument("Global work size must be a multiple of local work size");
    }
    
    const size_t* local = (local_work_size > 0) ? &local_work_size : nullptr;
    cl_int err = clEnqueueNDRangeKernel(queue.get(), kernel_, 1, nullptr, &global_work_size, local, 0, nullptr, nullptr);
    checkError(err, "executing kernel");
}

void Kernel::execute2D(const CommandQueue& queue, size_t global_width, size_t global_height,size_t local_width, size_t local_height) {
    size_t global[2] = {global_width, global_height};
    size_t local[2] = {local_width, local_height};
    const size_t* local_ptr = (local_width > 0 && local_height > 0) ? local : nullptr;
    
    cl_int err = clEnqueueNDRangeKernel(queue.get(), kernel_, 2, nullptr, global, local_ptr, 0, nullptr, nullptr);
    checkError(err, "executing kernel 2D");
}

void Kernel::execute3D(const CommandQueue& queue, size_t global_x, size_t global_y, size_t global_z, size_t local_x, size_t local_y, size_t local_z) {
    size_t global[3] = {global_x, global_y, global_z};
    size_t local[3] = {local_x, local_y, local_z};
    const size_t* local_ptr = (local_x > 0 && local_y > 0 && local_z > 0) ? local : nullptr;
    
    cl_int err = clEnqueueNDRangeKernel(queue.get(), kernel_, 3, nullptr, global, local_ptr, 0, nullptr, nullptr);
    checkError(err, "executing kernel 3D");
}

size_t Kernel::getWorkGroupSize(const Device& device) const {
    size_t size;
    cl_int err = clGetKernelWorkGroupInfo(kernel_, device.id(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &size, nullptr);
    checkError(err, "getting kernel work group size");
    return size;
}

size_t Kernel::getPreferredWorkGroupSizeMultiple(const Device& device) const {
    size_t size;
    cl_int err = clGetKernelWorkGroupInfo(kernel_, device.id(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &size, nullptr);
    checkError(err, "getting preferred work group size multiple");
    return size;
}

cl_ulong Kernel::getLocalMemSize(const Device& device) const {
    cl_ulong size;
    cl_int err = clGetKernelWorkGroupInfo(kernel_, device.id(), CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, nullptr);
    checkError(err, "getting kernel local mem size");
    return size;
}

} // namespace ocl

