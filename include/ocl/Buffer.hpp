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
    Buffer(const Context& context, const std::vector<T>& data, cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    
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
    Buffer(Buffer&& other) noexcept : buffer_(other.buffer_), size_(other.size_), capacity_(other.capacity_) {
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
    
    // Write with event (async) - note: include Event.hpp before using
    void writeAsync(const CommandQueue& queue, const std::vector<T>& data, cl_event& event);
    
    // Write partial data to buffer
    // Usage: buf.write(myVector, offset);
    void write(const CommandQueue& queue, const std::vector<T>& data, size_t offset, bool blocking = true);
    
    // Write raw pointer
    void write(const CommandQueue& queue, const T* data, size_t count, size_t offset = 0, bool blocking = true);
    
    // Read entire buffer into vector
    // Usage: buf.read(myVector);
    void read(const CommandQueue& queue, std::vector<T>& data, bool blocking = true);
    
    // Read with event (async) - note: include Event.hpp before using
    void readAsync(const CommandQueue& queue, std::vector<T>& data, cl_event& event);
    
    // Read partial buffer
    void read(const CommandQueue& queue, std::vector<T>& data, size_t count, size_t offset = 0, bool blocking = true);
    
    // Read into raw pointer
    void read(const CommandQueue& queue, T* data, size_t count, size_t offset = 0, bool blocking = true);
    
    // Fill buffer with a value
    void fill(const CommandQueue& queue, const T& value, bool blocking = true);
    
    // Copy from another buffer
    void copyFrom(const CommandQueue& queue, const Buffer<T>& src, size_t count, size_t src_offset = 0, size_t dst_offset = 0, bool blocking = true);
    
    // Copy to another buffer
    void copyTo(const CommandQueue& queue, Buffer<T>& dst, size_t count, size_t src_offset = 0, size_t dst_offset = 0, bool blocking = true);
    
    // Map buffer for direct access (zero-copy)
    T* map(const CommandQueue& queue, cl_map_flags flags = CL_MAP_READ | CL_MAP_WRITE, bool blocking = true);
    
    // Unmap buffer
    void unmap(const CommandQueue& queue, T* mapped_ptr);
    
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
                                     0, 
                                     nullptr, 
                                     nullptr);
    checkError(err, "writing buffer");
}

// Async write
template<typename T>
void Buffer<T>::writeAsync(const CommandQueue& queue, const std::vector<T>& data, cl_event& event) {
    if (data.size() > capacity_) {
        throw std::runtime_error("Data size exceeds buffer capacity");
    }
    
    size_ = data.size();
    cl_int err = clEnqueueWriteBuffer(detail::getQueueHandle(queue), 
                                     buffer_, 
                                     CL_FALSE,  // Non-blocking
                                     0, 
                                     data.size() * sizeof(T), 
                                     data.data(), 
                                     0, nullptr, &event);  // Use address directly
    checkError(err, "writing buffer async");
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
void Buffer<T>::write(const CommandQueue& queue, const T* data, size_t count, size_t offset, bool blocking) {
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

// Async read
template<typename T>
void Buffer<T>::readAsync(const CommandQueue& queue, std::vector<T>& data, cl_event& event) {
    data.resize(size_);
    
    cl_int err = clEnqueueReadBuffer(detail::getQueueHandle(queue), 
                                    buffer_, 
                                    CL_FALSE,  // Non-blocking
                                    0, 
                                    size_ * sizeof(T), 
                                    data.data(), 
                                    0, nullptr, &event);  // Use address directly
    checkError(err, "reading buffer async");
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

template<typename T>
void Buffer<T>::fill(const CommandQueue& queue, const T& value, bool blocking) {
    // Create a temp vector and write it (more portable than clEnqueueFillBuffer)
    std::vector<T> temp(size_, value);
    write(queue, temp, blocking);
}

template<typename T>
void Buffer<T>::copyFrom(const CommandQueue& queue, const Buffer<T>& src, size_t count, size_t src_offset, size_t dst_offset, bool blocking) {
    if (src_offset + count > src.size()) {
        throw std::runtime_error("Copy would exceed source buffer size");
    }
    if (dst_offset + count > capacity_) {
        throw std::runtime_error("Copy would exceed destination buffer capacity");
    }
    
    cl_int err = clEnqueueCopyBuffer(detail::getQueueHandle(queue),
                                    src.get(),
                                    buffer_,
                                    src_offset * sizeof(T),
                                    dst_offset * sizeof(T),
                                    count * sizeof(T),
                                    0, nullptr, nullptr);
    checkError(err, "copying buffer");
    
    if (blocking) {
        err = clFinish(detail::getQueueHandle(queue));
        checkError(err, "finishing copy operation");
    }
}

template<typename T>
void Buffer<T>::copyTo(const CommandQueue& queue, Buffer<T>& dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) {
    dst.copyFrom(queue, *this, count, src_offset, dst_offset, blocking);
}

template<typename T>
T* Buffer<T>::map(const CommandQueue& queue, cl_map_flags flags, bool blocking) {
    cl_int err;
    void* ptr = clEnqueueMapBuffer(detail::getQueueHandle(queue),
                                  buffer_,
                                  blocking ? CL_TRUE : CL_FALSE,
                                  flags,
                                  0,
                                  size_ * sizeof(T),
                                  0, nullptr, nullptr,
                                  &err);
    checkError(err, "mapping buffer");
    return static_cast<T*>(ptr);
}

template<typename T>
void Buffer<T>::unmap(const CommandQueue& queue, T* mapped_ptr) {
    cl_int err = clEnqueueUnmapMemObject(detail::getQueueHandle(queue),
                                        buffer_,
                                        mapped_ptr,
                                        0, nullptr, nullptr);
    checkError(err, "unmapping buffer");
}

} // namespace ocl

