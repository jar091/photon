cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0079 NEW)

# Kokkos CUDA uses COMPILE_AS_CMAKE_LANGUAGE mode: CMake's native CUDA language
# support compiles .cpp files as CUDA via nvcc with MSVC as the host compiler.
# Only photon source files are compiled as CUDA — ANARI SDK files are NOT
# (CUDA's float4 conflicts with linalg::float4).

option(OPENCODE_ENABLE_CUDA "Enable Kokkos CUDA backend for GPU shading" ON)

if(OPENCODE_ENABLE_CUDA)
  project(opencode_pathtracer LANGUAGES CXX CUDA)
  # Ensure CUDA standard matches CXX
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  project(opencode_pathtracer LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

set(BUILD_TESTING ON) # this project

# The tests and most apps use internal symbols of the device library, which a
# Windows DLL does not export.
if(WIN32)
  set(PHOTON_BUILD_EXTRAS_DEFAULT OFF)
else()
  set(PHOTON_BUILD_EXTRAS_DEFAULT ON)
endif()
option(PHOTON_BUILD_TESTS "Build the unit tests" ${PHOTON_BUILD_EXTRAS_DEFAULT})
option(PHOTON_BUILD_APPS "Build the command line apps" ${PHOTON_BUILD_EXTRAS_DEFAULT})
option(PHOTON_BUILD_TOOLS "Build the standalone rendering regression test" ON)
set(PHOTON_RUNTIME_DLLS "" CACHE STRING
  "DLLs installed next to the device library on Windows (e.g. embree4.dll;tbb12.dll)")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectOptions)
include(Dependencies)

add_subdirectory(src)

if(BUILD_TESTING AND PHOTON_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(PHOTON_BUILD_APPS)
  add_subdirectory(app)
endif()
if(PHOTON_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
