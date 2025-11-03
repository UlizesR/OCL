#include <ocl/Program.hpp>
#include <ocl/Context.hpp>
#include <ocl/Device.hpp>

namespace ocl {

Program::Program() : program_(nullptr) {}

Program::Program(const Context& context, const std::string& source) {
    if (source.empty()) {
        throw std::invalid_argument("Cannot create program with empty source");
    }
    
    const char* src = source.c_str();
    size_t length = source.length();
    
    cl_int err;
    program_ = clCreateProgramWithSource(context.get(), 1, &src, &length, &err);
    checkError(err, "creating program");
}

Program Program::fromFile(const Context& context, const std::string& filepath) {
    std::string source = readFile(filepath);
    return Program(context, source);
}

Program::~Program() {
    if (program_) {
        clReleaseProgram(program_);
    }
}

Program::Program(Program&& other) noexcept : program_(other.program_) {
    other.program_ = nullptr;
}

Program& Program::operator=(Program&& other) noexcept {
    if (this != &other) {
        if (program_) {
            clReleaseProgram(program_);
        }
        program_ = other.program_;
        other.program_ = nullptr;
    }
    return *this;
}

void Program::build(const Device& device, const std::string& options) {
    cl_device_id device_id = device.id();
    cl_int err = clBuildProgram(program_, 1, &device_id, 
                               options.empty() ? nullptr : options.c_str(),
                               nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        // Get build log on failure
        size_t log_size;
        clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 
                             0, nullptr, &log_size);
        
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG,
                             log_size, &log[0], nullptr);
        
        throw Error(err, "building program: " + log);
    }
}

} // namespace ocl

