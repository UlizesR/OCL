#include <ocl/Buffer.hpp>
#include <ocl/Context.hpp>
#include <ocl/CommandQueue.hpp>

namespace ocl {
namespace detail {

cl_context getContextHandle(const Context& ctx) {
    return ctx.get();
}

cl_command_queue getQueueHandle(const CommandQueue& queue) {
    return queue.get();
}

} // namespace detail
} // namespace ocl

