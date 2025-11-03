#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Setup OpenCL
        auto device = ocl::Device::getDefault();
        ocl::Context context(device);
        ocl::CommandQueue queue(context, device);
        
        std::cout << "Device: " << device.getName() << "\n";
        
        // Initialize data
        const size_t N = 1024;
        std::vector<float> a(N), b(N), c(N);
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i * 2);
        }
        
        // Create and populate buffers
        ocl::Buffer<float> buf_a(context, a);
        ocl::Buffer<float> buf_b(context, b);
        ocl::Buffer<float> buf_c(context, N);
        
        // Load and build kernel
        std::cout << "Loading kernel: vector_add.cl\n";
        ocl::Program program = ocl::Program::fromFile(context, "vector_add.cl");
        program.build(device);
        ocl::Kernel kernel(program, "vector_add");
        
        // Execute kernel
        kernel.setArg(0, buf_a.get());
        kernel.setArg(1, buf_b.get());
        kernel.setArg(2, buf_c.get());
        kernel.setArg(3, static_cast<int>(N));
        kernel.execute(queue, N);
        
        // Read results
        buf_c.read(queue, c);
        
        // Verify
        for (size_t i = 0; i < N; ++i) {
            if (c[i] != a[i] + b[i]) {
                std::cerr << "Verification failed at index " << i << "\n";
                return 1;
            }
        }
        
        std::cout << "✓ Vector addition successful!\n";
        std::cout << "  Computed " << N << " elements\n";
        std::cout << "  Example: " << a[0] << " + " << b[0] << " = " << c[0] << "\n";
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (code: " << e.code() << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

