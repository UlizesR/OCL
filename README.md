# OCL - Modern C++ OpenCL Library

A lightweight, header-friendly C++ library that provides a modern RAII interface for OpenCL, making GPU computing simple and safe.

[![C++14](https://img.shields.io/badge/C%2B%2B-14-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B14)
[![OpenCL](https://img.shields.io/badge/OpenCL-1.2%2B-green.svg)](https://www.khronos.org/opencl/)

## Features

### Core Functionality
- ✅ **RAII Resource Management** - Automatic cleanup, no memory leaks
- ✅ **Type-Safe Buffers** - `Buffer<T>` with compile-time type checking
- ✅ **Variadic Kernel Arguments** - `kernel.setArgs(a, b, c, d)` 
- ✅ **Automatic Work Group Sizing** - `NDRange::getOptimal1D/2D/3D()`
- ✅ **1D/2D/3D Kernel Execution** - Full ND-range support
- ✅ **Human-Readable Errors** - `CL_INVALID_VALUE (-30)` instead of just `-30`

### Advanced Features
- ✅ **Async Buffer Operations** - `writeAsync()`, `readAsync()` with events
- ✅ **Buffer Mapping** - Zero-copy access with `map()`/`unmap()`
- ✅ **GPU-Side Buffer Copy** - Fast device-to-device transfers
- ✅ **Program Binary Caching** - Save/load compiled kernels
- ✅ **Compilation Flags** - `buildOptimized()`, `buildDebug()`, custom flags
- ✅ **Kernel Introspection** - Query work group sizes and memory usage
- ✅ **Device Type Predicates** - `device.isGPU()`, `device.isCPU()`
- ✅ **Event Management** - Async operation tracking

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ocl.git
cd ocl
mkdir build && cd build
cmake ..
cmake --build .
```

### Your First Program

```cpp
#include <ocl/ocl.hpp>
#include <vector>

int main() {
    // Initialize
    auto device = ocl::Device::getDefault();
    ocl::Context ctx(device);
    ocl::CommandQueue queue(ctx, device);
    
    // Create data
    std::vector<float> a(1000, 1.0f);
    std::vector<float> b(1000, 2.0f);
    std::vector<float> c;
    
    // Create GPU buffers
    ocl::Buffer<float> buf_a(ctx, a);
    ocl::Buffer<float> buf_b(ctx, b);
    ocl::Buffer<float> buf_c(ctx, 1000);
    
    // Compile kernel
    ocl::Program prog = ocl::Program::fromFile(ctx, "kernel.cl");
    prog.buildOptimized(device);
    ocl::Kernel kernel(prog, "vector_add");
    
    // Execute
    kernel.setArgs(buf_a, buf_b, buf_c, 1000);
    kernel.execute(queue, 1000);
    
    // Read results
    buf_c.read(queue, c);
    
    return 0;
}
```

**kernel.cl:**
```c
__kernel void vector_add(__global float* a, __global float* b, 
                         __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}
```

## Examples

The library includes 6 comprehensive examples:

### Algorithms
- **vec_add** - Vector addition with automatic work group sizing
- **matmul** - Tiled matrix multiplication (optimized)
- **reduction** - Parallel sum with local memory
- **scan** - Prefix sum (exclusive scan)

### Utilities
- **benchmark** - Performance benchmarks (4 categories)
- **comprehensive_test** - Full feature validation (10 tests)

Run examples:
```bash
cd build/examples
./vec_add
./benchmark
./comprehensive_test
```

## API Overview

### Device Selection

```cpp
// Get default device
auto device = ocl::Device::getDefault();

// Or select by type
auto platforms = ocl::Platform::getAll();
for (auto& platform : platforms) {
    for (auto& device : platform.getDevices()) {
        if (device.isGPU()) {
            // Use this GPU
        }
    }
}
```

### Buffer Management

```cpp
// Create buffers
ocl::Buffer<float> buf(ctx, 1000);              // Allocate
ocl::Buffer<float> buf(ctx, host_vector);       // Initialize

// Data transfer
buf.write(queue, data);                          // Host → Device
buf.read(queue, data);                           // Device → Host
buf.writeAsync(queue, data, event);              // Async
buf.readAsync(queue, data, event);               // Async

// Operations
buf.fill(queue, 3.14f);                          // Fill with value
src.copyTo(queue, dst, count);                   // GPU-side copy

// Zero-copy access
float* ptr = buf.map(queue, CL_MAP_WRITE);
// ... modify data ...
buf.unmap(queue, ptr);
```

### Kernel Execution

```cpp
// Compile
ocl::Program prog = ocl::Program::fromFile(ctx, "kernel.cl");
prog.buildOptimized(device);  // Fast math optimizations
// OR
prog.buildDebug(device);      // Debug build
// OR
prog.build(device, "-Werror -cl-mad-enable");  // Custom flags

// Create kernel
ocl::Kernel kernel(prog, "my_kernel");

// Set arguments (clean variadic API)
kernel.setArgs(buf_a, buf_b, buf_c, N);

// Execute with automatic work group sizing
size_t local = ocl::NDRange::getOptimal1D(kernel, device, N);
size_t global = ocl::NDRange::getPaddedGlobalSize(N, local);
kernel.execute(queue, global, local);

// Or 2D/3D execution
kernel.execute2D(queue, width, height, local_x, local_y);
kernel.execute3D(queue, w, h, d, lx, ly, lz);
```

### Error Handling

```cpp
try {
    // OpenCL operations
} catch (const ocl::Error& e) {
    // Prints: "CL_INVALID_VALUE (-30) during: creating buffer"
    std::cerr << e.what() << "\n";
    std::cerr << "Error code: " << e.code() << "\n";
}
```

### Program Caching

```cpp
// First run - compile and save
ocl::Program prog = ocl::Program::fromFile(ctx, "kernel.cl");
prog.build(device);
prog.saveBinary(device, "kernel.bin");

// Subsequent runs - load from cache (10-100x faster!)
ocl::Program prog = ocl::Program::fromBinary(ctx, device, "kernel.bin");
```

### NDRange Utilities

```cpp
// Automatic optimal work group sizing
size_t local_1d = ocl::NDRange::getOptimal1D(kernel, device, N);

auto local_2d = ocl::NDRange::getOptimal2D(kernel, device, width, height);
// Returns: {local_x, local_y}

auto local_3d = ocl::NDRange::getOptimal3D(kernel, device, w, h, d);
// Returns: {local_x, local_y, local_z}

// Utilities
size_t global = ocl::NDRange::getPaddedGlobalSize(N, local);
bool valid = ocl::NDRange::isValidWorkSize(global, local);
```

## Project Structure

```
ocl/
├── include/ocl/           # Public headers
│   ├── ocl.hpp           # Main header (includes all)
│   ├── Errors.hpp        # Error handling (40+ error codes)
│   ├── Platform.hpp      # Platform abstraction
│   ├── Device.hpp        # Device abstraction + predicates
│   ├── Context.hpp       # Context management
│   ├── CommandQueue.hpp  # Command queue
│   ├── Event.hpp         # Event wrapper (async ops)
│   ├── Program.hpp       # Program compilation + caching
│   ├── Kernel.hpp        # Kernel execution
│   ├── Buffer.hpp        # Type-safe buffers
│   ├── NDRange.hpp       # Work group utilities
│   ├── Profiler.hpp      # Performance profiling
│   └── Registry.hpp      # Platform/device discovery
├── src/                  # Implementation
├── examples/             # 6 comprehensive examples
├── kernels/              # OpenCL kernel files
└── CMakeLists.txt        # Build configuration
```

## Performance

Benchmarks on Apple M4 GPU (1M elements):

| Operation | Time | Notes |
|-----------|------|-------|
| Host → Device | 1.1 ms | 4 MB transfer |
| Device → Host | 0.6 ms | 4 MB transfer |
| Kernel execution | 0.6 ms | With optimal work group |
| GPU-side copy | 0.4 ms | 2x faster than CPU |
| Binary load | 0.06 ms | 1.3x faster than compile |

## Requirements

- **C++14** or later
- **CMake 3.10** or later
- **OpenCL 1.2** or later

### Platform-Specific

**macOS:**
```bash
# OpenCL is built-in
brew install opencl-headers  # For development
```

**Linux:**
```bash
sudo apt install opencl-headers ocl-icd-opencl-dev
# Plus vendor-specific drivers (NVIDIA/AMD/Intel)
```

**Windows:**
```bash
# Install vendor SDK (NVIDIA CUDA, AMD APP SDK, or Intel SDK)
```

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .

# Run tests
cd examples
./comprehensive_test

# Run benchmarks
./benchmark
```

### CMake Integration

```cmake
add_subdirectory(ocl)
target_link_libraries(your_app PRIVATE OCL::ocl)
```

## Advanced Usage

### Async Workflows

```cpp
cl_event write_event, kernel_event, read_event;

// Start async operations
buf_in.writeAsync(queue, input_data, write_event);
clWaitForEvents(1, &write_event);

kernel.execute(queue, N);

buf_out.readAsync(queue, output_data, read_event);
clWaitForEvents(1, &read_event);

// Cleanup
clReleaseEvent(write_event);
clReleaseEvent(read_event);
```

### Local Memory

```cpp
// Allocate local memory for work group
const size_t LOCAL_SIZE = 256;
kernel.setArg(0, buf_input);
kernel.setArg(1, buf_output);
kernel.setLocalArg(2, LOCAL_SIZE * sizeof(float));  // Local memory
kernel.setArg(3, N);

kernel.execute(queue, global_size, LOCAL_SIZE);
```

### Kernel Introspection

```cpp
// Query kernel properties
size_t max_wg = kernel.getWorkGroupSize(device);
size_t preferred = kernel.getPreferredWorkGroupSizeMultiple(device);
cl_ulong local_mem = kernel.getLocalMemSize(device);

std::cout << "Max work group: " << max_wg << "\n";
std::cout << "Preferred multiple: " << preferred << "\n";
std::cout << "Local memory used: " << local_mem << " bytes\n";
```
