	cmake_minimum_required(VERSION 3.10)
	project(nbody-n-1-g-1 VERSION 1.0.0 LANGUAGES CUDA CXX)
	
	# packages
	find_package(OpenMP REQUIRED)

	SET(CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -Wall -Wextra -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET(CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -fopt-info-vec-optimized -Wall -Wextra -DMPICH_IGNORE_CXX_SEEK -std=c++14 " )
	SET( CMAKE_CXX_FLAGS_VTUNE "-O3 -g -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET( CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-O3 -DNDEBUG -march=haswell -DMPICH_IGNORE_CXX_SEEK -std=c++14")

		# workaround bug with MPI and OpenMP:
		# append MPI and OpenMP flags, which are actually for the host compiler, to the host_flags variables
		
		 # use -Xcompiler to propagate flags to nvcc host compiler
		 #whitespaces are required here (otherwise part of the previous flag, and first flag to propagate)
		set(CMAKE_CUDA_HOST_FLAGS " -Xcompiler ")
		
		    foreach (flag ${OpenMP_CXX_FLAGS})
			    string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
			endforeach (flag ${OpenMP_CXX_FLAGS})
		
			# append host flags to "normal" flags
			string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_FLAGS})

	SET( CMAKE_CUDA_FLAGS_DEV "-g -G -O0 -arch=compute_61 -code=sm_61 -use_fast_math -restrict -Xptxas -O0 -Xcompiler -O0,-g,-march=native,-Wall,-Wextra,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
	SET( CMAKE_CUDA_FLAGS_TEST "-g -G -O0 -arch=compute_61 -code=sm_61 -use_fast_math -restrict -Xptxas -O0 -Xcompiler -O3,-g,-march=native,-fopt-info-vec-optimized,-Wall,-Wextra,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
	SET( CMAKE_CUDA_FLAGS_VTUNE "-g -G -pg -O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-g,-DNDEBUG,-march=native,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
	SET( CMAKE_CUDA_FLAGS_BENCHMARK "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3" )
	SET( CMAKE_CUDA_FLAGS_BENCHMARKPALMA "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-DNDEBUG,-march=broadwell,-DMPICH_IGNORE_CXX_SEEK,-std=c++14" )
	SET( CMAKE_CUDA_FLAGS_BENCHMARKTAURUS "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-DNDEBUG,-march=haswell,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
		



	# output path for binaries and libraries
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
	
	
	

	
	add_executable(nbody-n-1-g-1_0 ${PROJECT_SOURCE_DIR}/src/nbody-n-1-g-1_0.cu)
	target_compile_features(nbody-n-1-g-1_0 PRIVATE cxx_std_14)
	target_include_directories(nbody-n-1-g-1_0 PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	target_compile_definitions(nbody-n-1-g-1_0 PRIVATE)
	target_compile_options(nbody-n-1-g-1_0 PRIVATE )
	target_link_libraries(nbody-n-1-g-1_0 PRIVATE ${OpenMP_CXX_LIBRARIES})
