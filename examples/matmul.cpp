#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Initialize
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "Tiled Matrix Multiplication (" << device.getName() << ")\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        // Matrix dimensions
        const size_t M = 1024, N = 1024, K = 1024;
        const size_t TILE_SIZE = 16;
        
        // Create matrices
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 2.0f);
        std::vector<float> C(M * N, 0.0f);
        
        // Create buffers
        ocl::Buffer<float> buf_A(ctx, A);
        ocl::Buffer<float> buf_B(ctx, B);
        ocl::Buffer<float> buf_C(ctx, M * N);
        
        // Compile with optimizations
        ocl::Program prog = ocl::Program::fromFile(ctx, "matmul_tiled.cl");
        prog.buildOptimized(device);
        ocl::Kernel kernel(prog, "matmul_tiled");
        
        // Calculate optimal 2D work group
        auto local_2d = ocl::NDRange::getOptimal2D(kernel, device, M, N);
        
        std::cout << "Matrix size:  " << M << " x " << K << " x " << K << " x " << N << "\n";
        std::cout << "Tile size:    " << TILE_SIZE << " x " << TILE_SIZE << "\n";
        std::cout << "Work group:   [" << local_2d[0] << ", " << local_2d[1] << "]\n";
        
        // Set arguments and execute
        kernel.setArgs(buf_A, buf_B, buf_C, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
        kernel.execute2D(queue, M, N, local_2d[0], local_2d[1]);
        
        // Read results
        buf_C.read(queue, C);
        
        // Verify (expected value: K * 1.0 * 2.0 = 2*K)
        float expected = static_cast<float>(K * 2);
        bool correct = (std::abs(C[0] - expected) < 1.0f && std::abs(C[M*N-1] - expected) < 1.0f);
        
        std::cout << "Expected:     " << expected << "\n";
        std::cout << "Got:          " << C[0] << "\n";
        std::cout << "Result:       " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << "\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        
        return correct ? 0 : 1;
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "\n";
        return 1;
    }
}
