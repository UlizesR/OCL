#include <ocl/Program.hpp>
#include <ocl/Context.hpp>
#include <ocl/Device.hpp>
#include <fstream>
#include <vector>

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
        throw Error(err, "building program: " + getBuildLog(device));
    }
}

std::string Program::getBuildLog(const Device& device) const {
    size_t log_size;
    clGetProgramBuildInfo(program_, device.id(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    
    std::string log(log_size, '\0');
    clGetProgramBuildInfo(program_, device.id(), CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
    
    // Remove trailing nulls
    while (!log.empty() && log.back() == '\0') {
        log.pop_back();
    }
    return log;
}

void Program::saveBinary(const Device&, const std::string& filepath) {
    size_t binary_size;
    cl_int err = clGetProgramInfo(program_, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
    checkError(err, "getting program binary size");
    
    std::vector<unsigned char> binary(binary_size);
    unsigned char* binary_ptr = binary.data();
    err = clGetProgramInfo(program_, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary_ptr, nullptr);
    checkError(err, "getting program binary");
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    file.write(reinterpret_cast<const char*>(binary.data()), binary_size);
}

Program Program::fromBinary(const Context& context, const Device& device, const std::string& filepath) {
    // Read binary from file
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + filepath);
    }
    
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> binary(file_size);
    file.read(reinterpret_cast<char*>(binary.data()), file_size);
    
    if (!file.good()) {
        throw std::runtime_error("Failed to read binary file: " + filepath);
    }
    
    // Create program from binary
    Program prog;
    cl_int err, binary_status;
    cl_device_id device_id = device.id();
    const unsigned char* binary_ptr = binary.data();
    
    prog.program_ = clCreateProgramWithBinary(context.get(), 1, &device_id, &file_size, &binary_ptr, &binary_status, &err);
    checkError(err, "creating program from binary");
    checkError(binary_status, "binary status");
    
    // Build the program (required even for binaries)
    err = clBuildProgram(prog.program_, 1, &device_id, nullptr, nullptr, nullptr);
    checkError(err, "building program from binary");
    
    return prog;
}

void Program::buildOptimized(const Device& device) {
    // Common optimization flags for OpenCL
    std::string opts = "-cl-fast-relaxed-math "   // Fast math
                       "-cl-mad-enable "            // Multiply-add fusion
                       "-cl-no-signed-zeros "       // Assume no signed zeros
                       "-cl-finite-math-only";      // Assume finite math
    build(device, opts);
}

void Program::buildDebug(const Device& device) {
    // Debug flags
    std::string opts = "-g "                        // Debug info
                       "-cl-opt-disable";           // Disable optimizations
    build(device, opts);
}

} // namespace ocl

