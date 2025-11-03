#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>

int main() {
    try {
        // Setup OpenCL
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "Device: " << device.getName() << "\n";
        
        // Matrix dimensions (M x K) * (K x N) = (M x N)
        const size_t M = 512, K = 512, N = 512;
        std::cout << "Matrix size: " << M << " x " << K << " x " << N << "\n";
        
        // Initialize matrices with random values
        std::vector<float> A(M * K), B(K * N), C(M * N);
        std::srand(42);
        for (auto& val : A) val = static_cast<float>(std::rand()) / RAND_MAX;
        for (auto& val : B) val = static_cast<float>(std::rand()) / RAND_MAX;
        
        // Create buffers  
        ocl::Buffer<float> buf_A(ctx, A);
        ocl::Buffer<float> buf_B(ctx, B);
        ocl::Buffer<float> buf_C(ctx, M * N);
        
        // Load and build kernel
        std::cout << "Loading kernel: matmul_tiled.cl\n";
        ocl::Program program = ocl::Program::fromFile(ctx, "matmul_tiled.cl");
        program.build(device);
        ocl::Kernel kernel(program, "matmul_tiled");
        
        // Execute kernel (16x16 tiles as defined in kernel)
        const size_t TILE_SIZE = 16;
        kernel.setArg(0, buf_A.get());
        kernel.setArg(1, buf_B.get());
        kernel.setArg(2, buf_C.get());
        kernel.setArg(3, static_cast<int>(M));
        kernel.setArg(4, static_cast<int>(K));
        kernel.setArg(5, static_cast<int>(N));
        kernel.execute2D(queue, M, N, TILE_SIZE, TILE_SIZE);
        
        // Read results
        buf_C.read(queue, C);
        
        // Verify sample elements
        const size_t VERIFY_COUNT = 5;
        for (size_t i = 0; i < VERIFY_COUNT && i < M; ++i) {
            for (size_t j = 0; j < VERIFY_COUNT && j < N; ++j) {
                float expected = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    expected += A[i * K + k] * B[k * N + j];
                }
                if (std::abs(C[i * N + j] - expected) > 1e-3f) {
                    std::cerr << "Verification failed at [" << i << "," << j << "]\n";
                    return 1;
                }
            }
        }
        
        std::cout << "✓ Matrix multiplication successful!\n";
        std::cout << "  Size: " << M << " x " << K << " x " << N << "\n";
        std::cout << "  Tile size: " << TILE_SIZE << " x " << TILE_SIZE << "\n";
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

