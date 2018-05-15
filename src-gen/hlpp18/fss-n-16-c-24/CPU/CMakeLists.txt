cmake_minimum_required(VERSION 3.5)
project(fss-n-16-c-24 VERSION 1.0.0 LANGUAGES CXX)

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
find_package(MPI REQUIRED)
### this is a cmake bug: MPI link flags are preceeded by two whitespaces, which leads to one leading whitespace, which is now an error according to policy CMP0004.
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)

find_package(OpenMP REQUIRED)	

add_executable(fss-n-16-c-24 ${PROJECT_SOURCE_DIR}/src/fss-n-16-c-24.cpp)
    target_include_directories(fss-n-16-c-24 PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(fss-n-16-c-24 PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_link_libraries(fss-n-16-c-24 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})
