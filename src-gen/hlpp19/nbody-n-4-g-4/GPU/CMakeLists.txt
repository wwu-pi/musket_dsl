cmake_minimum_required(VERSION 3.10)
project(nbody-n-4-g-4 VERSION 1.0.0 LANGUAGES CXX)

# required macros
SET( CMAKE_CXX_FLAGS_DEV "-g -O0 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
SET( CMAKE_CXX_FLAGS_TEST "-gopt -fast -⁠Mipa=fast -O4 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
SET( CMAKE_CXX_FLAGS_VTUNE "-gopt -fast -⁠Mipa=fast -O4 -w" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARK "-fast -⁠Mipa=fast -O4 -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-fast -⁠Mipa=fast -O4 -tp -ta:tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Palma." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-fast -O4 -tp=haswell -ta:tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Taurus." FORCE )
		
# output path for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# packages
find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)

find_package(OpenACC REQUIRED)

add_executable(nbody-n-4-g-4_0 ${PROJECT_SOURCE_DIR}/src/nbody-n-4-g-4_0.cpp)
target_compile_features(nbody-n-4-g-4_0 PRIVATE cxx_std_14)
target_include_directories(nbody-n-4-g-4_0 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(nbody-n-4-g-4_0 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
target_compile_options(nbody-n-4-g-4_0 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
target_link_libraries(nbody-n-4-g-4_0 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
add_executable(nbody-n-4-g-4_1 ${PROJECT_SOURCE_DIR}/src/nbody-n-4-g-4_1.cpp)
target_compile_features(nbody-n-4-g-4_1 PRIVATE cxx_std_14)
target_include_directories(nbody-n-4-g-4_1 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(nbody-n-4-g-4_1 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
target_compile_options(nbody-n-4-g-4_1 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
target_link_libraries(nbody-n-4-g-4_1 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
add_executable(nbody-n-4-g-4_2 ${PROJECT_SOURCE_DIR}/src/nbody-n-4-g-4_2.cpp)
target_compile_features(nbody-n-4-g-4_2 PRIVATE cxx_std_14)
target_include_directories(nbody-n-4-g-4_2 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(nbody-n-4-g-4_2 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
target_compile_options(nbody-n-4-g-4_2 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
target_link_libraries(nbody-n-4-g-4_2 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
add_executable(nbody-n-4-g-4_3 ${PROJECT_SOURCE_DIR}/src/nbody-n-4-g-4_3.cpp)
target_compile_features(nbody-n-4-g-4_3 PRIVATE cxx_std_14)
target_include_directories(nbody-n-4-g-4_3 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(nbody-n-4-g-4_3 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
target_compile_options(nbody-n-4-g-4_3 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
target_link_libraries(nbody-n-4-g-4_3 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
