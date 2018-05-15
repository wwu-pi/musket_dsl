cmake_minimum_required(VERSION 3.5)
project(nbody_float-n-1-c-12 VERSION 1.0.0 LANGUAGES CXX)

# required macros
SET( CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
SET( CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
SET( CMAKE_CXX_FLAGS_VTUNE "-O3 -g -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Palma." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-O3 -DNDEBUG -march=haswell -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Taurus." FORCE )
		
# output path for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# packages

find_package(OpenMP REQUIRED)	

add_executable(nbody_float-n-1-c-12 ${PROJECT_SOURCE_DIR}/src/nbody_float-n-1-c-12.cpp)
    target_include_directories(nbody_float-n-1-c-12 PRIVATE ${PROJECT_SOURCE_DIR}/include/)
    target_compile_options(nbody_float-n-1-c-12 PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(nbody_float-n-1-c-12 PRIVATE ${OpenMP_CXX_FLAGS})
