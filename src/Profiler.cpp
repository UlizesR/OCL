#include <ocl/Profiler.hpp>
#include <iostream>
#include <iomanip>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>

namespace ocl {

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::start(const std::string& name) {
    timings_[name].start = std::chrono::high_resolution_clock::now();
}

void Profiler::stop(const std::string& name) {
    auto end = std::chrono::high_resolution_clock::now();
    auto& timing = timings_[name];
    
    Duration elapsed = end - timing.start;
    timing.total_ms += elapsed.count();
    timing.count++;
}

double Profiler::getElapsed(const std::string& name) const {
    auto it = timings_.find(name);
    if (it == timings_.end()) {
        return 0.0;
    }
    return it->second.total_ms;
}

void Profiler::printResults() const {
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  Profiling Results\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << std::left << std::setw(30) << "Operation" 
              << std::right << std::setw(12) << "Total (ms)" 
              << std::setw(10) << "Count" 
              << std::setw(12) << "Avg (ms)" << "\n";
    std::cout << "───────────────────────────────────────────────────────\n";
    
    for (const auto& pair : timings_) {
        const auto& name = pair.first;
        const auto& timing = pair.second;
        double avg = timing.count > 0 ? timing.total_ms / timing.count : 0.0;
        std::cout << std::left << std::setw(30) << name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << timing.total_ms
                  << std::setw(10) << timing.count
                  << std::setw(12) << avg << "\n";
    }
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "\n";
}

void Profiler::reset() {
    timings_.clear();
}

} // namespace ocl

