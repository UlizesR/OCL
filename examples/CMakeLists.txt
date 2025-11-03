# Examples CMakeLists.txt

# Vector addition example
add_executable(vec_add vec_add.cpp)
target_link_libraries(vec_add PRIVATE OCL::ocl)

# Matrix multiplication example  
add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE OCL::ocl)

# Reduction example
add_executable(reduction reduction.cpp)
target_link_libraries(reduction PRIVATE OCL::ocl)

# Scan (prefix sum) example
add_executable(scan scan.cpp)
target_link_libraries(scan PRIVATE OCL::ocl)

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
message(STATUS "  - vec_add (vector addition)")
message(STATUS "  - matmul (matrix multiplication)")
message(STATUS "  - reduction (parallel sum)")
message(STATUS "  - scan (prefix sum)")
