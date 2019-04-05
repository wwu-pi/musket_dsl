cmake_minimum_required(VERSION 3.5)
project(matrix VERSION 1.0.0 LANGUAGES CXX)

# required macros
SET( CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
SET( CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -fopt-info-vec-optimized -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
SET( CMAKE_CXX_FLAGS_VTUNE "-O3 -g -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Palma." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-O3 -DNDEBUG -march=haswell -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Taurus." FORCE )
SET( CMAKE_CXX_FLAGS_CALLGRIND "-O3 -g -DNDEBUG -march=haswell -DMPICH_IGNORE_CXX_SEEK -std=c++14" CACHE STRING "Flags used by the C++ compiler during Callgrind builds for Taurus." FORCE )
		
# output path for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# packages

find_package(OpenMP REQUIRED)

add_executable(matrix_0 ${PROJECT_SOURCE_DIR}/src/matrix_0.cpp)
target_include_directories(matrix_0 PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_compile_options(matrix_0 PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(matrix_0 PRIVATE ${OpenMP_CXX_FLAGS})
