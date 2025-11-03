#pragma once

#include <ocl/Errors.hpp>
#include <string>
#include <chrono>
#include <unordered_map>

namespace ocl {

// ============================================================================
// Profiler - Performance profiling utilities
// ============================================================================

class Profiler {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    using Duration = std::chrono::duration<double, std::milli>;
    
    // Start timing an operation
    void start(const std::string& name);
    
    // Stop timing an operation
    void stop(const std::string& name);
    
    // Get elapsed time for an operation
    double getElapsed(const std::string& name) const;
    
    // Print all profiling results
    void printResults() const;
    
    // Reset all timers
    void reset();
    
    // Get singleton instance
    static Profiler& instance();
    
private:
    Profiler() = default;
    
    struct TimingData {
        TimePoint start;
        double total_ms = 0.0;
        size_t count = 0;
    };
    
    std::unordered_map<std::string, TimingData> timings_;
};

} // namespace ocl

