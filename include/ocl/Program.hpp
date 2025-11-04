#pragma once

#include <ocl/Errors.hpp>
#include <string>

namespace ocl {

// Forward declarations
class Context;
class Device;

// ============================================================================
// Program - manages OpenCL program with RAII
// ============================================================================

class Program {
public:
    Program();
    Program(const Context& context, const std::string& source);
    
    // Create program from file
    static Program fromFile(const Context& context, const std::string& filepath);
    
    ~Program();
    
    // Disable copying
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    
    // Enable moving
    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;
    
    // Build program
    void build(const Device& device, const std::string& options = "");
    
    // Save compiled binary to file
    void saveBinary(const Device& device, const std::string& filepath);
    
    // Load program from binary file
    static Program fromBinary(const Context& context, const Device& device, const std::string& filepath);
    
    // Build with optimization flags
    void buildOptimized(const Device& device);
    
    // Build with debug flags
    void buildDebug(const Device& device);
    
    // Get build log (useful for debugging)
    std::string getBuildLog(const Device& device) const;
    
    // Get underlying program
    cl_program get() const { return program_; }
    
private:
    cl_program program_;
};

} // namespace ocl

