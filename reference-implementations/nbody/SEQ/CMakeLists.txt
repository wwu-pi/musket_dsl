cmake_minimum_required(VERSION 3.5)
project(nbody VERSION 1.0.0 LANGUAGES CXX)

SET( CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -Wall -Wextra -Wpedantic" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
SET( CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -Wall -Wextra -Wpedantic" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
SET( CMAKE_CXX_FLAGS_VTUNE "-O3 -g -DNDEBUG -march=native" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -march=native" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )

# status messages
message(STATUS "Host System: " ${CMAKE_HOST_SYSTEM})
message(STATUS "Build Type: " ${CMAKE_BUILD_TYPE})
  
message(STATUS "Compiler Id: " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "Compiler flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "Compiler flags debug: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "Compiler flags release: " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "Compiler flags minsizerel: " ${CMAKE_CXX_FLAGS_MINSIZEREL})
message(STATUS "Compiler flags relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
message(STATUS "Compiler flags dev: " ${CMAKE_CXX_FLAGS_DEV})
message(STATUS "Compiler flags test: " ${CMAKE_CXX_FLAGS_TEST})
message(STATUS "Compiler flags vtune: " ${CMAKE_CXX_FLAGS_VTUNE})
message(STATUS "Compiler flags benchmark: " ${CMAKE_CXX_FLAGS_BENCHMARK})

# required macros

# output path for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

add_executable(nbody ${PROJECT_SOURCE_DIR}/src/nbody.cpp)
    target_include_directories(nbody PRIVATE ${PROJECT_SOURCE_DIR}/include/)
    target_compile_features(nbody PRIVATE cxx_auto_type cxx_lambdas cxx_nullptr cxx_uniform_initialization)
