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
    
    // Get underlying program
    cl_program get() const { return program_; }
    
private:
    cl_program program_;
};

} // namespace ocl

