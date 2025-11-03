# OCL - Modern C++ OpenCL Library

A lightweight, modular C++ library that provides a modern RAII interface for OpenCL with type-safe memory management.

## Features

✅ **Modular Architecture** - Each class in separate header/source files  
✅ **Type-Safe Buffers** - `Buffer<T>` with automatic sizing and type checking  
✅ **RAII Design** - Automatic resource cleanup, no manual memory management  
✅ **File-Based Kernels** - Load kernels from `.cl` files with `Program::fromFile()`  
✅ **Exception-Based Errors** - Clear error reporting with error codes  
✅ **Move Semantics** - Efficient resource transfer  
✅ **Cross-Platform** - Works on macOS, Linux, Windows  
✅ **Modern C++14** - Clean, professional API  
✅ **Static Library** - Links as `libocl.a`  
✅ **Comprehensive Examples** - 4 tested algorithms included  

## Project Structure

```
ocl/
├── include/ocl/          # Public API (12 headers)
│   ├── Errors.hpp        # Error handling & utilities
│   ├── Platform.hpp      # OpenCL platform
│   ├── Device.hpp        # Device selection & queries
│   ├── Context.hpp       # Context management
│   ├── CommandQueue.hpp  # Command queue operations
│   ├── Program.hpp       # Kernel compilation
│   ├── Kernel.hpp        # Kernel execution
│   ├── Buffer.hpp        # Type-safe memory buffers
│   ├── Registry.hpp      # Platform/device registry
│   ├── Profiler.hpp      # Performance profiling
│   ├── Image.hpp         # Image support (placeholder)
│   └── ocl.hpp           # Main include (includes all)
│
├── src/                  # Implementation (10 sources)
│   ├── Errors.cpp
│   ├── Platform.cpp
│   ├── Device.cpp
│   ├── Context.cpp
│   ├── CommandQueue.cpp
│   ├── Program.cpp
│   ├── Kernel.cpp
│   ├── Buffer.cpp
│   ├── Registry.cpp
│   └── Profiler.cpp
│
├── kernels/              # OpenCL kernels (4 algorithms)
│   ├── vector_add.cl     # Element-wise addition
│   ├── matmul_tiled.cl   # Tiled matrix multiplication
│   ├── reduction.cl      # Parallel sum reduction
│   └── scan.cl           # Prefix sum (exclusive scan)
│
├── examples/             # Working examples (4 programs)
│   ├── vec_add.cpp       # Vector addition
│   ├── matmul.cpp        # Matrix multiplication
│   ├── reduction.cpp     # Parallel reduction
│   └── scan.cpp          # Prefix sum
│
└── CMakeLists.txt        # Cross-platform build
```

## Quick Start

### Build and Run

```bash
# Clone or navigate to project
cd ocl

# Configure
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build .

# Run examples
./examples/vec_add
./examples/matmul
./examples/reduction
./examples/scan
```

## API Reference

### Platform

```cpp
// Get all platforms
auto platforms = ocl::Platform::getAll();

// Get default platform
auto platform = ocl::Platform::getDefault();

// Query information
std::string name = platform.getName();
std::string vendor = platform.getVendor();
std::string version = platform.getVersion();
```

### Device

```cpp
// Get all devices from a platform
auto devices = ocl::Device::getAll(platform, CL_DEVICE_TYPE_GPU);

// Get default device (first GPU, or first available)
auto device = ocl::Device::getDefault();

// Query device information
std::string name = device.getName();
std::string vendor = device.getVendor();
std::string version = device.getVersion();
cl_device_type type = device.getType();
cl_ulong globalMem = device.getGlobalMemSize();
cl_ulong localMem = device.getLocalMemSize();
cl_uint computeUnits = device.getMaxComputeUnits();
cl_uint maxWorkGroup = device.getMaxWorkGroupSize();
```

### Context

```cpp
// Create context for single device
ocl::Context ctx(device);

// Create context for multiple devices
std::vector<ocl::Device> devices = {...};
ocl::Context ctx(devices);

// Context automatically released when destroyed
```

### CommandQueue

```cpp
// Create command queue
ocl::CommandQueue queue(ctx, device);

// With properties (e.g., profiling)
ocl::CommandQueue queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);

// Operations
queue.finish();  // Wait for all commands
queue.flush();   // Flush the queue
```

### Buffer<T> - Type-Safe Memory

```cpp
// Create buffer with size
ocl::Buffer<float> buf(ctx, 1024);

// Create and initialize from vector
std::vector<int> data = {1, 2, 3, 4, 5};
ocl::Buffer<int> buf(ctx, data);

// Write vector to buffer
buf.write(queue, myVector);

// Write with offset
buf.write(queue, partialData, offset, blocking);

// Read from buffer
std::vector<float> result;
buf.read(queue, result);

// Query properties
size_t elements = buf.size();
size_t bytes = buf.sizeBytes();
cl_mem handle = buf.get();
```

### Program & Kernel

```cpp
// Load kernel from file (recommended)
ocl::Program prog = ocl::Program::fromFile(ctx, "my_kernel.cl");
prog.build(device);

// Or from string
ocl::Program prog(ctx, kernel_source_string);
prog.build(device, "-DDEBUG");  // Optional build flags

// Create kernel
ocl::Kernel kernel(prog, "kernel_function_name");

// Set arguments
kernel.setArg(0, buf.get());           // Buffer (cl_mem)
kernel.setArg(1, 42);                  // Scalar value
kernel.setLocalArg(2, 256 * sizeof(float));  // Local memory

// Execute (1D)
kernel.execute(queue, global_size, local_size);

// Execute (2D)
kernel.execute2D(queue, width, height, tile_w, tile_h);
```

### Registry (Device Discovery)

```cpp
// Get singleton registry
auto& registry = ocl::Registry::instance();

// Get all platforms and devices
const auto& platforms = registry.getPlatforms();
auto allDevices = registry.getAllDevices();
auto gpus = registry.getDevicesByType(CL_DEVICE_TYPE_GPU);

// Print system info
registry.printInfo();
```

### Profiler (Performance Timing)

```cpp
auto& profiler = ocl::Profiler::instance();

// Time an operation
profiler.start("kernel_execution");
kernel.execute(queue, N);
queue.finish();
profiler.stop("kernel_execution");

// Get results
double elapsed = profiler.getElapsed("kernel_execution");
profiler.printResults();  // Print all timings
profiler.reset();         // Clear all timings
```

## Building

### Standard Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Release Build (Optimized)

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Build Without Examples

```bash
cmake -DBUILD_EXAMPLES=OFF ..
cmake --build .
```

## Installation

```bash
cd build
sudo cmake --install .
```

This installs:
- Headers to `/usr/local/include/ocl/`
- Static library to `/usr/local/lib/libocl.a`
- CMake config to `/usr/local/lib/cmake/ocl/`

### Using Installed Library

```cmake
# In your CMakeLists.txt
find_package(ocl REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE OCL::ocl)
```

## Platform-Specific Setup

### macOS

```bash
# Install OpenCL headers
brew install opencl-headers

# OpenCL framework included with macOS
# No additional drivers needed for Apple Silicon
```

### Linux (Ubuntu/Debian)

```bash
# Install headers and ICD loader
sudo apt install opencl-headers ocl-icd-opencl-dev

# Install GPU-specific drivers:
# NVIDIA:
sudo apt install nvidia-cuda-toolkit

# AMD:
sudo apt install rocm-opencl-runtime

# Intel:
sudo apt install intel-opencl-icd
```

### Windows

1. Install OpenCL SDK from your GPU vendor:
   - NVIDIA: CUDA Toolkit
   - AMD: AMD APP SDK
   - Intel: Intel OpenCL SDK
2. CMake will automatically find OpenCL
3. Use Visual Studio or MinGW for compilation

## Architecture

### Core Modules

| Module | Header | Source | Purpose |
|--------|--------|--------|---------|
| **Errors** | ✓ | ✓ | Error handling, file I/O |
| **Platform** | ✓ | ✓ | Platform enumeration |
| **Device** | ✓ | ✓ | Device selection & queries |
| **Context** | ✓ | ✓ | Context lifecycle |
| **CommandQueue** | ✓ | ✓ | Queue management |
| **Program** | ✓ | ✓ | Kernel compilation |
| **Kernel** | ✓ | ✓ | Kernel execution |
| **Buffer<T>** | ✓ | ✓ | Type-safe buffers |
| **Registry** | ✓ | ✓ | Device registry |
| **Profiler** | ✓ | ✓ | Performance timing |
| **Image** | ✓ | - | Future image support |

### Design Principles

1. **RAII Everywhere** - All resources automatically managed
2. **No Raw Pointers** - Modern C++ smart patterns
3. **Type Safety** - Templates prevent type mismatches
4. **Exception-Based** - Clear error reporting
5. **Modular** - Easy to extend and maintain
6. **Zero Overhead** - Minimal abstraction cost

## Advanced Features

### Local Memory Support

```cpp
// Allocate local memory for work-group
kernel.setLocalArg(argIndex, sizeInBytes);

// Example: reduction with local scratch memory
kernel.setArg(0, input_buffer.get());
kernel.setArg(1, output_buffer.get());
kernel.setLocalArg(2, 256 * sizeof(float));  // Local scratch
kernel.setArg(3, N);
kernel.execute(queue, global_size, local_size);
```

### Multi-Device Contexts

```cpp
auto devices = ocl::Device::getAll(platform);
ocl::Context ctx(devices);  // Context spans multiple devices
```

### Kernel Build Options

```cpp
// Build with compiler flags
program.build(device, "-DTILE_SIZE=16 -DDEBUG");
```

### Buffer Variants

```cpp
// Read-only buffer
ocl::Buffer<float> buf(ctx, N, CL_MEM_READ_ONLY);

// Write-only buffer
ocl::Buffer<float> buf(ctx, N, CL_MEM_WRITE_ONLY);

// Initialize with data
ocl::Buffer<float> buf(ctx, myVector, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

// Partial I/O
buf.write(queue, data, offset, blocking);
buf.read(queue, result, count, offset, blocking);
```

## Error Handling

All OpenCL errors throw `ocl::Error` with descriptive messages:

```cpp
try {
    auto device = ocl::Device::getDefault();
    ocl::Context ctx(device);
    // ... OpenCL operations ...
    
} catch (const ocl::Error& e) {
    std::cerr << "OpenCL error: " << e.what() << "\n";
    std::cerr << "Error code: " << e.code() << "\n";
    return 1;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
```

## Performance Profiling

```cpp
auto& prof = ocl::Profiler::instance();

// Time operations
prof.start("data_transfer");
buffer.write(queue, data);
prof.stop("data_transfer");

prof.start("kernel_execution");
kernel.execute(queue, N);
queue.finish();
prof.stop("kernel_execution");

// Print results
prof.printResults();
// Output:
// Operation              Total (ms)  Count   Avg (ms)
// data_transfer          5.234       1       5.234
// kernel_execution       2.156       1       2.156
```

## Complete API Quick Reference

### Platform
```cpp
Platform::getAll()               // All platforms
Platform::getDefault()           // First platform
platform.getName()               // Platform name
platform.getVendor()             // Vendor name
platform.getVersion()            // OpenCL version
```

### Device
```cpp
Device::getAll(platform, type)   // Get devices by type
Device::getDefault()             // Default device
device.getName()                 // Device name
device.getVendor()               // Vendor
device.getVersion()              // OpenCL version
device.getType()                 // GPU/CPU type
device.getGlobalMemSize()        // Global memory (bytes)
device.getLocalMemSize()         // Local memory (bytes)
device.getMaxComputeUnits()      // Compute units
device.getMaxWorkGroupSize()     // Max work group size
```

### Context
```cpp
Context(device)                  // Single device
Context(devices)                 // Multiple devices
ctx.get()                        // Get cl_context
```

### CommandQueue
```cpp
CommandQueue(ctx, device)        // Create queue
CommandQueue(ctx, dev, props)    // With properties
queue.finish()                   // Wait for completion
queue.flush()                    // Flush queue
```

### Buffer<T>
```cpp
Buffer<float> buf(ctx, N)                    // Allocate
Buffer<int> buf(ctx, vec)                    // From vector
Buffer<T> buf(ctx, N, CL_MEM_READ_ONLY)     // With flags
buf.write(queue, vector)                     // Write
buf.read(queue, vector)                      // Read
buf.size()                                   // Elements
buf.sizeBytes()                              // Bytes
```

### Program
```cpp
Program::fromFile(ctx, "kernel.cl")   // From file
Program(ctx, source_string)            // From string
prog.build(device)                     // Build
prog.build(device, "-O3")              // With options
```

### Kernel
```cpp
Kernel(program, "kernel_name")        // Create
kernel.setArg(i, value)               // Scalar arg
kernel.setArg(i, buffer.get())        // Buffer arg
kernel.setLocalArg(i, bytes)          // Local memory
kernel.execute(queue, N)              // 1D execution
kernel.execute(queue, N, workgroup)   // 1D with local size
kernel.execute2D(queue, W, H)         // 2D execution
kernel.execute2D(queue, W, H, LW, LH) // 2D with local size
```

### Registry
```cpp
auto& reg = Registry::instance();
reg.getPlatforms()                    // All platforms
reg.getAllDevices()                   // All devices
reg.getDevicesByType(type)            // Devices by type
reg.getDefaultDevice()                // Default device
reg.printInfo()                       // Print system info
```

### Profiler
```cpp
auto& prof = Profiler::instance();
prof.start("operation")               // Start timer
prof.stop("operation")                // Stop timer
prof.getElapsed("operation")          // Get time (ms)
prof.printResults()                   // Print all
prof.reset()                          // Clear all
```

## Requirements

- **CMake:** 3.10 or later
- **C++ Standard:** C++14 or later
- **OpenCL:** 1.2 or later (headers and runtime)
- **Compiler:** GCC 5+, Clang 3.4+, MSVC 2015+

## Tested Platforms

- ✅ macOS 14+ (Apple Silicon M4) - OpenCL 1.2
- ✅ Linux (Ubuntu 20.04+) - NVIDIA/AMD/Intel
- ⚠️ Windows (should work, not tested)

