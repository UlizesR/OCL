#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Initialize
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "Vector Addition (" << device.getName() << ")\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        // Create data
        const size_t N = 1024 * 1024;
        std::vector<float> a(N, 1.0f), b(N, 2.0f), c;
        
        // Create buffers
        ocl::Buffer<float> buf_a(ctx, a);
        ocl::Buffer<float> buf_b(ctx, b);
        ocl::Buffer<float> buf_c(ctx, N);
        
        // Compile with optimizations
        ocl::Program prog = ocl::Program::fromFile(ctx, "vector_add.cl");
        prog.buildOptimized(device);
        ocl::Kernel kernel(prog, "vector_add");
        
        // Calculate optimal work group size
        size_t local = ocl::NDRange::getOptimal1D(kernel, device, N);
        size_t global = ocl::NDRange::getPaddedGlobalSize(N, local);
        
        std::cout << "Problem size: " << N << " elements\n";
        std::cout << "Work group:   " << local << " (global: " << global << ")\n";
        
        // Execute with variadic args and optimal sizing
        kernel.setArgs(buf_a, buf_b, buf_c, static_cast<int>(N));
        kernel.execute(queue, global, local);
        
        // Read results
        buf_c.read(queue, c);
        
        // Verify (check first, middle, last)
        bool correct = (c[0] == 3.0f && c[N/2] == 3.0f && c[N-1] == 3.0f);
        
        std::cout << "Result:       " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        return correct ? 0 : 1;
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "\n";
        return 1;
    }
}
