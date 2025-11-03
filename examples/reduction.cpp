#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

int main() {
    try {
        // Setup OpenCL
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "Device: " << device.getName() << "\n";
        
        // Problem size
        const size_t N = 1024;
        const size_t WORK_GROUP_SIZE = 256;
        const size_t NUM_GROUPS = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
        
        std::cout << "Reducing " << N << " elements\n";
        std::cout << "Work group size: " << WORK_GROUP_SIZE << "\n";
        std::cout << "Number of groups: " << NUM_GROUPS << "\n";
        
        // Initialize input data
        std::vector<float> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i] = 1.0f;  // Sum should be N
        }
        
        // Create buffers
        ocl::Buffer<float> buf_input(ctx, input);
        ocl::Buffer<float> buf_output(ctx, NUM_GROUPS);
        
        // Load and build kernel
        std::cout << "Loading kernel: reduction.cl\n";
        ocl::Program program = ocl::Program::fromFile(ctx, "reduction.cl");
        program.build(device);
        ocl::Kernel kernel(program, "reduce_sum");
        
        // Set kernel arguments
        kernel.setArg(0, buf_input.get());
        kernel.setArg(1, buf_output.get());
        kernel.setLocalArg(2, WORK_GROUP_SIZE * sizeof(float));  // Local memory
        kernel.setArg(3, static_cast<int>(N));
        
        // Execute reduction kernel
        kernel.execute(queue, NUM_GROUPS * WORK_GROUP_SIZE, WORK_GROUP_SIZE);
        
        // Read partial results
        std::vector<float> partial_sums(NUM_GROUPS);
        buf_output.read(queue, partial_sums);
        
        // Final reduction on CPU (or could do another GPU pass)
        float gpu_result = 0.0f;
        for (float val : partial_sums) {
            gpu_result += val;
        }
        
        // CPU verification
        float cpu_result = std::accumulate(input.begin(), input.end(), 0.0f);
        
        // Verify
        float diff = std::abs(gpu_result - cpu_result);
        if (diff < 1e-3f) {
            std::cout << "✓ Reduction successful!\n";
            std::cout << "  GPU result: " << gpu_result << "\n";
            std::cout << "  CPU result: " << cpu_result << "\n";
            std::cout << "  Difference: " << diff << "\n";
        } else {
            std::cerr << "✗ Verification failed!\n";
            std::cerr << "  GPU: " << gpu_result << ", CPU: " << cpu_result << "\n";
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

