cmake_minimum_required(VERSION 3.5)
project(optimizations VERSION 1.0.0 LANGUAGES CXX)

SET( CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -m64 -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
SET( CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -m64 -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
SET( CMAKE_CXX_FLAGS_VTUNE "-O3 -g -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
SET( CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -march=haswell -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )

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

# packages		

## MPI
find_package(MPI REQUIRED)
### this is a cmake bug: MPI link flags are preceeded by two whitespaces, which leads to one leading whitespace, which is now an error according to policy CMP0004.
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)

message(STATUS "MPI compiler: " ${MPI_CXX_COMPILER})
message(STATUS "MPI compiler flags: " ${MPI_CXX_COMPILE_FLAGS})
message(STATUS "MPI link flags: " ${MPI_CXX_LINK_FLAGS})
message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})

## OpenMP
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP compiler flags: " ${OpenMP_CXX_FLAGS})

add_executable(map ${PROJECT_SOURCE_DIR}/src/map.cpp)
    target_include_directories(map PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(map PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(map PRIVATE cxx_std_14)
    target_link_libraries(map PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(map_opt ${PROJECT_SOURCE_DIR}/src/map-opt.cpp)
    target_include_directories(map_opt PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(map_opt PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(map_opt PRIVATE cxx_std_14)
    target_link_libraries(map_opt PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(map_fold ${PROJECT_SOURCE_DIR}/src/map-fold.cpp)
    target_include_directories(map_fold PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(map_fold PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(map_fold PRIVATE cxx_std_14)
    target_link_libraries(map_fold PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(map_fold_opt ${PROJECT_SOURCE_DIR}/src/map-fold-opt.cpp)
    target_include_directories(map_fold_opt PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(map_fold_opt PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(map_fold_opt PRIVATE cxx_std_14)
    target_link_libraries(map_fold_opt PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(ref ${PROJECT_SOURCE_DIR}/src/ref.cpp)
    target_include_directories(ref PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(ref PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(ref PRIVATE cxx_std_14)
    target_link_libraries(ref PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(ref_opt ${PROJECT_SOURCE_DIR}/src/ref-opt.cpp)
    target_include_directories(ref_opt PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(ref_opt PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(ref_opt PRIVATE cxx_std_14)
    target_link_libraries(ref_opt PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(data ${PROJECT_SOURCE_DIR}/src/data.cpp)
    target_include_directories(data PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(data PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(data PRIVATE cxx_std_14)
    target_link_libraries(data PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(data_opt ${PROJECT_SOURCE_DIR}/src/data-opt.cpp)
    target_include_directories(data_opt PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(data_opt PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(data_opt PRIVATE cxx_std_14)
    target_link_libraries(data_opt PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})

add_executable(data_opt_move ${PROJECT_SOURCE_DIR}/src/data-opt-move.cpp)
    target_include_directories(data_opt_move PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(data_opt_move PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
    target_compile_features(data_opt_move PRIVATE cxx_std_14)
    target_link_libraries(data_opt_move PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})