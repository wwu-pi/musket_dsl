cmake_minimum_required(VERSION 3.10)
project(nbody_float VERSION 1.0.0 LANGUAGES CXX)

# required macros
SET( CMAKE_CXX_FLAGS_DEV "-g -O0 -Minfo=accel -ta=tesla,nollvm -Mcudalib=curand" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
SET( CMAKE_CXX_FLAGS_TEST "-gopt -fast -⁠Mipa=fast -O4 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
SET( CMAKE_CXX_FLAGS_VTUNE "-gopt -fast -⁠Mipa=fast -O4 -w" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARK "-fast -⁠Mipa=fast -O4 -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-fast -⁠Mipa=fast -O4 -tp -ta:tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Palma." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-fast -⁠Mipa=fast -O4 -tp -ta:tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Taurus." FORCE )
		
# output path for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# packages

find_package(OpenMP REQUIRED)

find_package(OpenACC REQUIRED)

find_package(CUDA REQUIRED)

add_executable(nbody_float_0 ${PROJECT_SOURCE_DIR}/src/nbody_float_0.cpp)
target_compile_features(nbody_float_0 PRIVATE cxx_std_14)
target_include_directories(nbody_float_0 PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(nbody_float_0 PRIVATE)
target_compile_options(nbody_float_0 PRIVATE ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
target_link_libraries(nbody_float_0 PRIVATE ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
