#include <ocl/Event.hpp>

namespace ocl {

Event::Event() : event_(nullptr) {}

Event::Event(cl_event event) : event_(event) {}

Event::~Event() {
    if (event_) {
        clReleaseEvent(event_);
    }
}

Event::Event(Event&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (event_) {
            clReleaseEvent(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void Event::wait() {
    if (!event_) {
        throw std::runtime_error("Cannot wait on invalid event");
    }
    
    cl_int err = clWaitForEvents(1, &event_);
    checkError(err, "waiting for event");
}

cl_int Event::getStatus() const {
    if (!event_) {
        return CL_INVALID_EVENT;
    }
    
    cl_int status;
    cl_int err = clGetEventInfo(event_, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, nullptr);
    checkError(err, "getting event status");
    return status;
}

bool Event::isComplete() const {
    return getStatus() == CL_COMPLETE;
}

cl_ulong Event::getProfilingStart() const {
    if (!event_) {
        throw std::runtime_error("Cannot get profiling info from invalid event");
    }
    
    cl_ulong time;
    cl_int err = clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time, nullptr);
    checkError(err, "getting profiling start time");
    return time;
}

cl_ulong Event::getProfilingEnd() const {
    if (!event_) {
        throw std::runtime_error("Cannot get profiling info from invalid event");
    }
    
    cl_ulong time;
    cl_int err = clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time, nullptr);
    checkError(err, "getting profiling end time");
    return time;
}

cl_ulong Event::getProfilingDuration() const {
    return getProfilingEnd() - getProfilingStart();
}

double Event::getProfilingDurationMs() const {
    return getProfilingDuration() / 1000000.0;  // nanoseconds to milliseconds
}

void waitForEvents(const std::vector<Event>& events) {
    if (events.empty()) {
        return;
    }
    
    std::vector<cl_event> event_handles;
    event_handles.reserve(events.size());
    for (const auto& event : events) {
        if (event.isValid()) {
            event_handles.push_back(event.get());
        }
    }
    
    if (!event_handles.empty()) {
        cl_int err = clWaitForEvents(event_handles.size(), event_handles.data());
        checkError(err, "waiting for events");
    }
}

} // namespace ocl

