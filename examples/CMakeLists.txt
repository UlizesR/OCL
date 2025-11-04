# Examples CMakeLists.txt

# Core algorithm examples
add_executable(vec_add vec_add.cpp)
target_link_libraries(vec_add PRIVATE OCL::ocl)

add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE OCL::ocl)

add_executable(reduction reduction.cpp)
target_link_libraries(reduction PRIVATE OCL::ocl)

add_executable(scan scan.cpp)
target_link_libraries(scan PRIVATE OCL::ocl)

# Utilities
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE OCL::ocl)

add_executable(comprehensive_test comprehensive_test.cpp)
target_link_libraries(comprehensive_test PRIVATE OCL::ocl)

# Copy kernel files to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/vector_add.cl
    ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cl
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/reduction.cl
    ${CMAKE_CURRENT_BINARY_DIR}/reduction.cl
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/matmul_tiled.cl
    ${CMAKE_CURRENT_BINARY_DIR}/matmul_tiled.cl
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/scan.cl
    ${CMAKE_CURRENT_BINARY_DIR}/scan.cl
    COPYONLY
)

message(STATUS "Building examples:")
message(STATUS "  [Algorithms]")
message(STATUS "    • vec_add - Vector addition")
message(STATUS "    • matmul - Tiled matrix multiplication")
message(STATUS "    • reduction - Parallel sum")
message(STATUS "    • scan - Prefix sum")
message(STATUS "  [Utilities]")
message(STATUS "    • benchmark - Performance benchmarks")
message(STATUS "    • comprehensive_test - Full feature test suite")
