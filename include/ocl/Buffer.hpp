#pragma once

#include <ocl/Errors.hpp>
#include <vector>

namespace ocl {

// Forward declarations
class Context;
class CommandQueue;

// ============================================================================
// Templated Buffer - Type-safe OpenCL memory buffer
// ============================================================================

template<typename T>
class Buffer {
public:
    // Default constructor
    Buffer() : buffer_(nullptr), size_(0), capacity_(0) {}
    
    // Create buffer with specified size
    // Usage: Buffer<float> buf(context, 1024);
    Buffer(const Context& context, size_t count, cl_mem_flags flags = CL_MEM_READ_WRITE);
    
    // Create buffer and initialize from std::vector
    // Usage: Buffer<float> buf(context, myVector);
    Buffer(const Context& context, const std::vector<T>& data, 
           cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
    // Destructor
    ~Buffer() {
        if (buffer_) {
            clReleaseMemObject(buffer_);
        }
    }
    
    // Disable copying
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    // Enable moving
    Buffer(Buffer&& other) noexcept 
        : buffer_(other.buffer_)
        , size_(other.size_)
        , capacity_(other.capacity_) {
        other.buffer_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (buffer_) {
                clReleaseMemObject(buffer_);
            }
            buffer_ = other.buffer_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.buffer_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Write entire vector to buffer
    // Usage: buf.write(myVector);
    void write(const CommandQueue& queue, const std::vector<T>& data, bool blocking = true);
    
    // Write partial data to buffer
    // Usage: buf.write(myVector, offset);
    void write(const CommandQueue& queue, const std::vector<T>& data, size_t offset, bool blocking = true);
    
    // Write raw pointer
    void write(const CommandQueue& queue, const T* data, size_t count, size_t offset = 0, bool blocking = true);
    
    // Read entire buffer into vector
    // Usage: buf.read(myVector);
    void read(const CommandQueue& queue, std::vector<T>& data, bool blocking = true);
    
    // Read partial buffer
    void read(const CommandQueue& queue, std::vector<T>& data, size_t count, size_t offset = 0, bool blocking = true);
    
    // Read into raw pointer
    void read(const CommandQueue& queue, T* data, size_t count, size_t offset = 0, bool blocking = true);
    
    // Get underlying OpenCL buffer
    cl_mem get() const { return buffer_; }
    
    // Get size in elements
    size_t size() const { return size_; }
    
    // Get capacity in elements
    size_t capacity() const { return capacity_; }
    
    // Get size in bytes
    size_t sizeBytes() const { return size_ * sizeof(T); }
    
    // Get capacity in bytes
    size_t capacityBytes() const { return capacity_ * sizeof(T); }

private:
    cl_mem buffer_;
    size_t size_;      // Current size in elements
    size_t capacity_;  // Total capacity in elements
};

// ============================================================================
// Implementation
// ============================================================================

// Forward declarations for implementation
namespace detail {
    cl_context getContextHandle(const Context& ctx);
    cl_command_queue getQueueHandle(const CommandQueue& queue);
}

// Implementation will be provided when Context and CommandQueue are fully defined

template<typename T>
Buffer<T>::Buffer(const Context& context, size_t count, cl_mem_flags flags) : size_(count), capacity_(count) {
    
    cl_int err;
    buffer_ = clCreateBuffer(detail::getContextHandle(context), 
                            flags, 
                            count * sizeof(T), 
                            nullptr, 
                            &err);
    checkError(err, "creating buffer");
}

template<typename T>
Buffer<T>::Buffer(const Context& context, const std::vector<T>& data, cl_mem_flags flags) : size_(data.size()), capacity_(data.size()) {
    
    cl_int err;
    buffer_ = clCreateBuffer(detail::getContextHandle(context), 
                            flags, 
                            data.size() * sizeof(T), 
                            const_cast<T*>(data.data()),  // clCreateBuffer needs non-const
                            &err);
    checkError(err, "creating buffer with data");
}

template<typename T>
void Buffer<T>::write(const CommandQueue& queue, const std::vector<T>& data, bool blocking) {
    if (data.size() > capacity_) {
        throw std::runtime_error("Data size exceeds buffer capacity");
    }
    
    size_ = data.size();
    cl_int err = clEnqueueWriteBuffer(detail::getQueueHandle(queue), 
                                     buffer_, 
                                     blocking ? CL_TRUE : CL_FALSE,
                                     0, 
                                     data.size() * sizeof(T), 
                                     data.data(), 
                                     0, nullptr, nullptr);
    checkError(err, "writing buffer");
}

template<typename T>
void Buffer<T>::write(const CommandQueue& queue, const std::vector<T>& data, size_t offset, bool blocking) {
    if (offset + data.size() > capacity_) {
        throw std::runtime_error("Write would exceed buffer capacity");
    }
    
    cl_int err = clEnqueueWriteBuffer(detail::getQueueHandle(queue), 
                                     buffer_, 
                                     blocking ? CL_TRUE : CL_FALSE,
                                     offset * sizeof(T), 
                                     data.size() * sizeof(T), 
                                     data.data(), 
                                     0, nullptr, nullptr);
    checkError(err, "writing buffer with offset");
}

template<typename T>
void Buffer<T>::write(const CommandQueue& queue, const T* data, size_t count, 
                     size_t offset, bool blocking) {
    if (offset + count > capacity_) {
        throw std::runtime_error("Write would exceed buffer capacity");
    }
    
    cl_int err = clEnqueueWriteBuffer(detail::getQueueHandle(queue), 
                                     buffer_, 
                                     blocking ? CL_TRUE : CL_FALSE,
                                     offset * sizeof(T), 
                                     count * sizeof(T), 
                                     data, 
                                     0, nullptr, nullptr);
    checkError(err, "writing buffer from pointer");
}

template<typename T>
void Buffer<T>::read(const CommandQueue& queue, std::vector<T>& data, bool blocking) {
    data.resize(size_);
    
    cl_int err = clEnqueueReadBuffer(detail::getQueueHandle(queue), 
                                    buffer_, 
                                    blocking ? CL_TRUE : CL_FALSE,
                                    0, 
                                    size_ * sizeof(T), 
                                    data.data(), 
                                    0, nullptr, nullptr);
    checkError(err, "reading buffer");
}

template<typename T>
void Buffer<T>::read(const CommandQueue& queue, std::vector<T>& data, size_t count, size_t offset, bool blocking) {
    if (offset + count > size_) {
        throw std::runtime_error("Read would exceed buffer size");
    }
    
    data.resize(count);
    
    cl_int err = clEnqueueReadBuffer(detail::getQueueHandle(queue), 
                                    buffer_, 
                                    blocking ? CL_TRUE : CL_FALSE,
                                    offset * sizeof(T), 
                                    count * sizeof(T), 
                                    data.data(), 
                                    0, nullptr, nullptr);
    checkError(err, "reading buffer with offset");
}

template<typename T>
void Buffer<T>::read(const CommandQueue& queue, T* data, size_t count, size_t offset, bool blocking) {
    if (offset + count > size_) {
        throw std::runtime_error("Read would exceed buffer size");
    }
    
    cl_int err = clEnqueueReadBuffer(detail::getQueueHandle(queue), 
                                    buffer_, 
                                    blocking ? CL_TRUE : CL_FALSE,
                                    offset * sizeof(T), 
                                    count * sizeof(T), 
                                    data, 
                                    0, nullptr, nullptr);
    checkError(err, "reading buffer to pointer");
}

} // namespace ocl

