	cmake_minimum_required(VERSION 3.10)
	project(frobenius-n-1-g-2 VERSION 1.0.0 LANGUAGES CXX)
	
	# required macros
	SET( CMAKE_CXX_FLAGS_DEV "-g -O0 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
	SET( CMAKE_CXX_FLAGS_TEST "-gopt -fast -ta:tesla:cc60,pinned -O4 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
	SET( CMAKE_CXX_FLAGS_VTUNE "-gopt -fast -O4 -w" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
	SET( CMAKE_CXX_FLAGS_BENCHMARK "-fast -O4 -ta:tesla:cc60,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
	SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-fast -O4 -tp -ta:tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Palma." FORCE )
	SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-fast -O4 -tp=haswell -ta=tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Taurus." FORCE )
		
	# output path for binaries and libraries
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
	
	# packages
	find_package(MPI REQUIRED)
	
	find_package(OpenMP REQUIRED)
	
	find_package(OpenACC REQUIRED)

	
	add_executable(frobenius-n-1-g-2_0 ${PROJECT_SOURCE_DIR}/src/frobenius-n-1-g-2_0.cpp)
	target_compile_features(frobenius-n-1-g-2_0 PRIVATE cxx_std_14)
	target_include_directories(frobenius-n-1-g-2_0 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
	target_compile_definitions(frobenius-n-1-g-2_0 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
	target_compile_options(frobenius-n-1-g-2_0 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
	target_link_libraries(frobenius-n-1-g-2_0 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
	add_executable(frobenius-n-1-g-2_1 ${PROJECT_SOURCE_DIR}/src/frobenius-n-1-g-2_1.cpp)
	target_compile_features(frobenius-n-1-g-2_1 PRIVATE cxx_std_14)
	target_include_directories(frobenius-n-1-g-2_1 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
	target_compile_definitions(frobenius-n-1-g-2_1 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
	target_compile_options(frobenius-n-1-g-2_1 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
	target_link_libraries(frobenius-n-1-g-2_1 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
	add_executable(frobenius-n-1-g-2_2 ${PROJECT_SOURCE_DIR}/src/frobenius-n-1-g-2_2.cpp)
	target_compile_features(frobenius-n-1-g-2_2 PRIVATE cxx_std_14)
	target_include_directories(frobenius-n-1-g-2_2 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
	target_compile_definitions(frobenius-n-1-g-2_2 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
	target_compile_options(frobenius-n-1-g-2_2 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
	target_link_libraries(frobenius-n-1-g-2_2 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
	add_executable(frobenius-n-1-g-2_3 ${PROJECT_SOURCE_DIR}/src/frobenius-n-1-g-2_3.cpp)
	target_compile_features(frobenius-n-1-g-2_3 PRIVATE cxx_std_14)
	target_include_directories(frobenius-n-1-g-2_3 PRIVATE ${PROJECT_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_DIRS})
	target_compile_definitions(frobenius-n-1-g-2_3 PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
	target_compile_options(frobenius-n-1-g-2_3 PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
	target_link_libraries(frobenius-n-1-g-2_3 PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
