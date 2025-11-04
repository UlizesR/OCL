#include <ocl/ocl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

template<typename Func>
double benchmark(const std::string& name, int iterations, Func func) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = Clock::now();
    Duration elapsed = end - start;
    double avg = elapsed.count() / iterations;
    
    std::cout << std::left << std::setw(40) << name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << elapsed.count() << " ms"
              << std::setw(12) << avg << " ms/iter\n";
    
    return avg;
}

int main() {
    try {
        auto device = ocl::Device::getDefault();
        ocl::Context ctx(device);
        ocl::CommandQueue queue(ctx, device);
        
        std::cout << "\nPerformance Benchmarks\n";
        std::cout << "══════════════════════════════════════════════════════════════════\n";
        std::cout << "Device:         " << device.getName() << "\n";
        std::cout << "Type:           " << (device.isGPU() ? "GPU" : "CPU") << "\n";
        std::cout << "Compute Units:  " << device.getMaxComputeUnits() << "\n";
        std::cout << "══════════════════════════════════════════════════════════════════\n\n";
        
        // Buffer Transfer Benchmarks
        std::cout << "1. Buffer Transfer Performance (4 MB)\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        const size_t N = 1024 * 1024;  // 1M floats = 4MB
        std::vector<float> hostData(N, 1.0f);
        std::vector<float> resultData;
        ocl::Buffer<float> buf(ctx, N);
        
        std::cout << std::left << std::setw(40) << "Operation"
                  << std::right << std::setw(12) << "Total"
                  << std::setw(12) << "Average\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        benchmark("Host → Device (write)", 10, [&]() {
            buf.write(queue, hostData);
        });
        
        benchmark("Device → Host (read)", 10, [&]() {
            buf.read(queue, resultData);
        });
        
        benchmark("Round-trip (write + read)", 10, [&]() {
            buf.write(queue, hostData);
            buf.read(queue, resultData);
        });
        
        benchmark("Buffer fill", 10, [&]() {
            buf.fill(queue, 0.0f);
        });
        
        // Kernel Execution Benchmarks
        std::cout << "\n2. Kernel Execution Performance (1M elements)\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        ocl::Buffer<float> buf_a(ctx, N);
        ocl::Buffer<float> buf_b(ctx, N);
        ocl::Buffer<float> buf_c(ctx, N);
        
        buf_a.fill(queue, 1.0f);
        buf_b.fill(queue, 2.0f);
        
        // Test with optimized build
        ocl::Program prog = ocl::Program::fromFile(ctx, "vector_add.cl");
        prog.buildOptimized(device);
        ocl::Kernel kernel(prog, "vector_add");
        kernel.setArgs(buf_a, buf_b, buf_c, static_cast<int>(N));
        
        // Calculate optimal work group
        size_t local = ocl::NDRange::getOptimal1D(kernel, device, N);
        size_t global = ocl::NDRange::getPaddedGlobalSize(N, local);
        
        std::cout << std::left << std::setw(40) << "Operation"
                  << std::right << std::setw(12) << "Total"
                  << std::setw(12) << "Average\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        benchmark("Kernel (auto work group " + std::to_string(local) + ")", 100, [&]() {
            kernel.execute(queue, global, local);
            queue.finish();
        });
        
        // Buffer Copy Benchmarks
        std::cout << "\n3. Buffer Copy Performance (GPU vs CPU)\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        ocl::Buffer<float> src(ctx, N);
        ocl::Buffer<float> dst(ctx, N);
        src.fill(queue, 1.0f);
        
        std::cout << std::left << std::setw(40) << "Operation"
                  << std::right << std::setw(12) << "Total"
                  << std::setw(12) << "Average\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        double gpu_time = benchmark("GPU-side copy", 10, [&]() {
            src.copyTo(queue, dst, N);
            queue.finish();
        });
        
        std::vector<float> temp;
        double cpu_time = benchmark("CPU round-trip", 10, [&]() {
            src.read(queue, temp);
            dst.write(queue, temp);
        });
        
        double speedup = cpu_time / gpu_time;
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(1) 
                  << speedup << "x (GPU-side copy vs CPU round-trip)\n";
        
        // Compilation Benchmarks
        std::cout << "\n4. Program Compilation Performance\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        const std::string cache_file = "bench_cache.bin";
        
        std::cout << std::left << std::setw(40) << "Operation"
                  << std::right << std::setw(12) << "Total"
                  << std::setw(12) << "Average\n";
        std::cout << "──────────────────────────────────────────────────────────────────\n";
        
        double compile_time = benchmark("Compile from source", 5, [&]() {
            ocl::Program p = ocl::Program::fromFile(ctx, "vector_add.cl");
            p.build(device);
        });
        
        // Save binary
        ocl::Program prog_for_save = ocl::Program::fromFile(ctx, "vector_add.cl");
        prog_for_save.build(device);
        prog_for_save.saveBinary(device, cache_file);
        
        double binary_time = benchmark("Load from binary", 5, [&]() {
            ocl::Program p = ocl::Program::fromBinary(ctx, device, cache_file);
        });
        
        std::remove(cache_file.c_str());
        
        speedup = compile_time / binary_time;
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(1)
                  << speedup << "x (binary cache vs recompiling)\n";
        
        // Summary
        std::cout << "\n══════════════════════════════════════════════════════════════════\n";
        std::cout << "Benchmark Complete!\n";
        std::cout << "══════════════════════════════════════════════════════════════════\n\n";
        
    } catch (const ocl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
