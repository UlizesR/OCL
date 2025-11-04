#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Initialize
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "Prefix Sum (Scan) (" << device.getName() << ")\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        // Create data (power of 2 for simplicity)
        const size_t N = 256;
        const size_t WORK_GROUP_SIZE = N / 2;
        
        std::vector<float> input(N);
        for (size_t i = 0; i < N; ++i) input[i] = 1.0f;
        
        std::vector<float> output(N, 0.0f);
        
        // Create buffers
        ocl::Buffer<float> buf_input(ctx, input);
        ocl::Buffer<float> buf_output(ctx, N);
        
        // Compile with optimizations
        ocl::Program prog = ocl::Program::fromFile(ctx, "scan.cl");
        prog.buildOptimized(device);
        ocl::Kernel kernel(prog, "scan_inclusive");
        
        std::cout << "Problem size: " << N << " elements\n";
        std::cout << "Work group:   " << WORK_GROUP_SIZE << "\n";
        
        // Set arguments (including local memory)
        kernel.setArg(0, buf_input);
        kernel.setArg(1, buf_output);
        kernel.setLocalArg(2, 2 * N * sizeof(float));
        kernel.setArg(3, static_cast<int>(N));
        
        // Execute
        kernel.execute(queue, WORK_GROUP_SIZE, WORK_GROUP_SIZE);
        
        // Read results
        buf_output.read(queue, output);
        
        // Verify (exclusive scan of all 1s: [0, 1, 2, ..., N-1])
        bool correct = (output[0] == 0.0f && output[N/2] == static_cast<float>(N/2) && output[N-1] == static_cast<float>(N-1));
        
        std::cout << "First 5:      [" << output[0] << ", " << output[1] << ", " 
                  << output[2] << ", " << output[3] << ", " << output[4] << "]\n";
        std::cout << "Last:         " << output[N-1] << " (expected: " << (N-1) << ")\n";
        std::cout << "Result:       " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        return correct ? 0 : 1;
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "\n";
        return 1;
    }
}
