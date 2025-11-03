#include <ocl/CommandQueue.hpp>
#include <ocl/Context.hpp>
#include <ocl/Device.hpp>

namespace ocl {

CommandQueue::CommandQueue() : queue_(nullptr) {}

CommandQueue::CommandQueue(const Context& context, const Device& device, 
                           cl_command_queue_properties properties) {
    cl_int err;
    // Suppress deprecation warning for clCreateCommandQueue (still used on macOS)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    queue_ = clCreateCommandQueue(context.get(), device.id(), properties, &err);
#pragma clang diagnostic pop
    checkError(err, "creating command queue");
}

CommandQueue::~CommandQueue() {
    if (queue_) {
        clReleaseCommandQueue(queue_);
    }
}

CommandQueue::CommandQueue(CommandQueue&& other) noexcept : queue_(other.queue_) {
    other.queue_ = nullptr;
}

CommandQueue& CommandQueue::operator=(CommandQueue&& other) noexcept {
    if (this != &other) {
        if (queue_) {
            clReleaseCommandQueue(queue_);
        }
        queue_ = other.queue_;
        other.queue_ = nullptr;
    }
    return *this;
}

void CommandQueue::finish() {
    cl_int err = clFinish(queue_);
    checkError(err, "finishing command queue");
}

void CommandQueue::flush() {
    cl_int err = clFlush(queue_);
    checkError(err, "flushing command queue");
}

} // namespace ocl

