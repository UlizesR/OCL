cmake_minimum_required(VERSION 3.10)
project(OCL VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Find OpenCL
# ============================================================================
find_package(OpenCL REQUIRED)

if(OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
    message(STATUS "  Include: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "  Library: ${OpenCL_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenCL not found!")
endif()

# ============================================================================
# Platform-specific settings
# ============================================================================

# On macOS with Homebrew, help find OpenCL headers
if(APPLE)
    execute_process(
        COMMAND brew --prefix opencl-headers
        OUTPUT_VARIABLE OPENCL_HEADERS_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(OPENCL_HEADERS_PREFIX)
        include_directories(${OPENCL_HEADERS_PREFIX}/include)
        message(STATUS "Found Homebrew OpenCL headers: ${OPENCL_HEADERS_PREFIX}/include")
    endif()
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# ============================================================================
# OCL Library (static library with source files)
# ============================================================================

# Collect source files
set(OCL_SOURCES
    src/Errors.cpp
    src/Platform.cpp
    src/Device.cpp
    src/Context.cpp
    src/CommandQueue.cpp
    src/Event.cpp
    src/Program.cpp
    src/Kernel.cpp
    src/Buffer.cpp
    src/NDRange.cpp
    src/Registry.cpp
    src/Profiler.cpp
)

# Collect header files (for IDE support)
set(OCL_HEADERS
    include/ocl/Errors.hpp
    include/ocl/Platform.hpp
    include/ocl/Device.hpp
    include/ocl/Context.hpp
    include/ocl/CommandQueue.hpp
    include/ocl/Event.hpp
    include/ocl/Program.hpp
    include/ocl/Kernel.hpp
    include/ocl/Buffer.hpp
    include/ocl/NDRange.hpp
    include/ocl/Registry.hpp
    include/ocl/Profiler.hpp
    include/ocl/ocl.hpp
)

# Create static library
add_library(ocl STATIC ${OCL_SOURCES} ${OCL_HEADERS})

target_include_directories(ocl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ocl PUBLIC OpenCL::OpenCL)

# Set C++ standard for the library
target_compile_features(ocl PUBLIC cxx_std_14)

# Alias for consistent naming
add_library(OCL::ocl ALIAS ocl)

# ============================================================================
# Examples (optional)
# ============================================================================

option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install headers
install(DIRECTORY include/ocl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install library target
install(TARGETS ocl
    EXPORT oclTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install CMake config files
install(EXPORT oclTargets
    FILE oclTargets.cmake
    NAMESPACE OCL::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ocl
)

# ============================================================================
# Configuration summary
# ============================================================================

message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  OCL Configuration Summary")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
