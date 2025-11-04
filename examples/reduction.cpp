#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    try {
        // Initialize
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "Parallel Reduction (" << device.getName() << ")\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        // Create data
        const size_t N = 1024 * 1024;
        const size_t WORK_GROUP_SIZE = 256;
        const size_t NUM_GROUPS = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
        
        std::vector<float> data(N, 1.0f);  // Sum should be N
        std::vector<float> partial_sums(NUM_GROUPS, 0.0f);
        
        // Create buffers
        ocl::Buffer<float> buf_input(ctx, data);
        ocl::Buffer<float> buf_output(ctx, NUM_GROUPS);
        
        // Compile with optimizations
        ocl::Program prog = ocl::Program::fromFile(ctx, "reduction.cl");
        prog.buildOptimized(device);
        ocl::Kernel kernel(prog, "reduce_sum");
        
        std::cout << "Problem size: " << N << " elements\n";
        std::cout << "Work group:   " << WORK_GROUP_SIZE << "\n";
        std::cout << "Num groups:   " << NUM_GROUPS << "\n";
        
        // Set arguments (including local memory)
        kernel.setArg(0, buf_input);
        kernel.setArg(1, buf_output);
        kernel.setLocalArg(2, WORK_GROUP_SIZE * sizeof(float));
        kernel.setArg(3, static_cast<int>(N));
        
        // Execute
        kernel.execute(queue, NUM_GROUPS * WORK_GROUP_SIZE, WORK_GROUP_SIZE);
        
        // Read partial results
        buf_output.read(queue, partial_sums);
        
        // Final reduction on CPU
        float gpu_sum = std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0f);
        float expected = static_cast<float>(N);
        
        bool correct = (std::abs(gpu_sum - expected) < 1.0f);
        
        std::cout << "Expected sum: " << expected << "\n";
        std::cout << "GPU sum:      " << gpu_sum << "\n";
        std::cout << "Result:       " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        return correct ? 0 : 1;
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "\n";
        return 1;
    }
}
