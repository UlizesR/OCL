#pragma once

#include <ocl/Errors.hpp>
#include <vector>

namespace ocl {

// ============================================================================
// Event - manages OpenCL event with RAII
// ============================================================================

class Event {
public:
    Event();
    explicit Event(cl_event event);
    
    ~Event();
    
    // Disable copying
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    
    // Enable moving
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    
    // Wait for event to complete
    void wait();
    
    // Get event status
    cl_int getStatus() const;
    
    // Check if event is complete
    bool isComplete() const;
    
    // Get profiling information (requires CL_QUEUE_PROFILING_ENABLE)
    cl_ulong getProfilingStart() const;
    cl_ulong getProfilingEnd() const;
    cl_ulong getProfilingDuration() const;  // In nanoseconds
    double getProfilingDurationMs() const;   // In milliseconds
    
    // Get underlying event
    cl_event get() const { return event_; }
    cl_event* ptr() { return &event_; }
    
    // Check if event is valid
    bool isValid() const { return event_ != nullptr; }
    
private:
    cl_event event_;
};

// Wait for multiple events
void waitForEvents(const std::vector<Event>& events);

} // namespace ocl

