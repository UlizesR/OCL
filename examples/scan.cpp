#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

int main() {
    try {
        // Setup OpenCL
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "Device: " << device.getName() << "\n";
        
        // Problem size (must be power of 2 for this scan algorithm)
        const size_t N = 256;  // Small size for demo
        const size_t WORK_GROUP_SIZE = N / 2;
        
        std::cout << "Scan (prefix sum) of " << N << " elements\n";
        
        // Initialize input data
        std::vector<float> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i] = 1.0f;  // Prefix sum: [0, 1, 2, 3, ...]
        }
        
        // Expected output (exclusive scan)
        std::vector<float> expected(N);
        expected[0] = 0.0f;
        for (size_t i = 1; i < N; ++i) {
            expected[i] = expected[i-1] + input[i-1];
        }
        
        // Create buffers
        ocl::Buffer<float> buf_input(ctx, input);
        ocl::Buffer<float> buf_output(ctx, N);
        
        // Load and build kernel
        std::cout << "Loading kernel: scan.cl\n";
        ocl::Program program = ocl::Program::fromFile(ctx, "scan.cl");
        program.build(device);
        ocl::Kernel kernel(program, "scan_inclusive");
        
        // Set kernel arguments
        kernel.setArg(0, buf_input.get());
        kernel.setArg(1, buf_output.get());
        kernel.setLocalArg(2, N * sizeof(float));  // Local memory
        kernel.setArg(3, static_cast<int>(N));
        
        // Execute scan kernel
        kernel.execute(queue, WORK_GROUP_SIZE, WORK_GROUP_SIZE);
        
        // Read results
        std::vector<float> output;
        buf_output.read(queue, output);
        
        // Verify first few elements
        bool success = true;
        const size_t VERIFY_COUNT = 10;
        for (size_t i = 0; i < VERIFY_COUNT && i < N; ++i) {
            if (std::abs(output[i] - expected[i]) > 1e-3f) {
                std::cerr << "Verification failed at index " << i << "\n";
                std::cerr << "  Expected: " << expected[i] << ", Got: " << output[i] << "\n";
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "✓ Scan (prefix sum) successful!\n";
            std::cout << "  First 10 results: [";
            for (size_t i = 0; i < 10; ++i) {
                std::cout << output[i];
                if (i < 9) std::cout << ", ";
            }
            std::cout << "]\n";
            std::cout << "  Expected:         [";
            for (size_t i = 0; i < 10; ++i) {
                std::cout << expected[i];
                if (i < 9) std::cout << ", ";
            }
            std::cout << "]\n";
        } else {
            return 1;
        }
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

