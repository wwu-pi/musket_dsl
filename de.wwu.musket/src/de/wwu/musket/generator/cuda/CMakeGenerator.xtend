package de.wwu.musket.generator.cuda

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

/**
 * Generates CMake file.
 * <p>
 * Entry point is the method generateCMake(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context).
 * The generator creates a file called CMakeLists.txt, which is required to build the generated project.
 * The CMake file is designed in such a way that new build types are added, which will only work with gcc.
 * Other compilers can be used, but then the default types (release, debug) have to be used.
 * Additional flags can then be set in the cmake call.
 */
class CMakeGenerator {
	private static final Logger logger = LogManager.getLogger(CMakeGenerator)

	/**
	 * Starting point for the CMakeGenerator.
	 */
	def static void generateCMake(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate CMakeLists.txt.")
		resource.generateCMakeListstxt(fsa)
		logger.info("Generation of CMakeLists.txt done.")
	}

	/**
	 * Creates the file CMakeLists.txt in the current base_path.
	 */
	def static void generateCMakeListstxt(Resource resource, IFileSystemAccess2 fsa) {
		fsa.generateFile(Config.base_path + "CMakeLists.txt", CMakeListstxtContent(resource))
	}

	/**
	 * Generates the content of the CMakeLists.txt file.
	 */
	def static CMakeListstxtContent(Resource resource) '''
	cmake_minimum_required(VERSION 3.10)
	project(«resource.ProjectName» VERSION 1.0.0 LANGUAGES CXX CUDA)
	
	SET(CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -Wall -Wextra -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET(CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -fopt-info-vec-optimized -Wall -Wextra -DMPICH_IGNORE_CXX_SEEK -std=c++14 " )
	SET(CMAKE_CXX_FLAGS_VTUNE "-O3 -g -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET(CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DNDEBUG -march=native -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET(CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	SET(CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-O3 -DNDEBUG -march=haswell -DMPICH_IGNORE_CXX_SEEK -std=c++14")
	
	set(CMAKE_CUDA_HOST_FLAGS " -Xcompiler ")
	set(CMAKE_CUDA_HOST_LINKER_FLAGS " -Xlinker ")
	
	# packages
	«IF Config.processes > 1»
		find_package(MPI REQUIRED)
		string(REPLACE " " "," MPI_CXX_LINK_FLAG ${MPI_CXX_LINK_FLAGS})
		
		foreach (flag ${MPI_CXX_COMPILE_OPTIONS})
			string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
		endforeach (flag ${MPI_CXX_COMPILE_OPTIONS})
		foreach (flag ${MPI_CXX_LINK_FLAG})
			string(APPEND CMAKE_CUDA_HOST_LINKER_FLAGS ",${flag}")
		endforeach (flag ${MPI_CXX_LINK_FLAG})
	«ENDIF»
	«IF Config.cores > 1»
		find_package(OpenMP REQUIRED)
		foreach (flag ${OpenMP_CXX_FLAGS})
			string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
			string(APPEND CMAKE_CUDA_HOST_LINKER_FLAGS ",${flag}")
		endforeach (flag ${OpenMP_CXX_FLAGS})
	«ENDIF»

	# append host flags to "normal" flags
	string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_FLAGS})
	string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_LINKER_FLAGS})

	SET( CMAKE_CUDA_FLAGS_DEV "-g -G -O0 -arch=compute_61 -code=sm_61 -use_fast_math -restrict -Xptxas -O0 -Xcompiler -O0,-g,-march=native,-Wall,-Wextra,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
	SET( CMAKE_CUDA_FLAGS_TEST "-g -G -O0 -arch=compute_61 -code=sm_61 -use_fast_math -restrict -Xptxas -O0 -Xcompiler -O3,-g,-march=native,-fopt-info-vec-optimized,-Wall,-Wextra,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
	SET( CMAKE_CUDA_FLAGS_VTUNE "-g -G -pg -O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-g,-DNDEBUG,-march=native,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")
	SET( CMAKE_CUDA_FLAGS_BENCHMARK "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3" )
	SET( CMAKE_CUDA_FLAGS_BENCHMARKPALMA "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-DNDEBUG,-march=broadwell,-DMPICH_IGNORE_CXX_SEEK,-std=c++14" )
	SET( CMAKE_CUDA_FLAGS_BENCHMARKTAURUS "-O3 -arch=compute_35 -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-DNDEBUG,-march=haswell,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")	
	
	# output path for binaries and libraries
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

	«FOR processId : 0 ..< Config.processes»
		add_executable(«resource.ProjectName»_«processId» ${PROJECT_SOURCE_DIR}/src/«resource.ProjectName»_«processId»«Config.source_extension»)
		target_compile_features(«resource.ProjectName»_«processId» PRIVATE cxx_std_14)
		target_include_directories(«resource.ProjectName»_«processId» PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}«IF Config.processes > 1» ${MPI_CXX_INCLUDE_DIRS}«ENDIF»)
		target_compile_definitions(«resource.ProjectName»_«processId» PRIVATE«IF Config.processes > 1» ${MPI_CXX_COMPILE_DEFINITIONS}«ENDIF»)
		target_compile_options(«resource.ProjectName»_«processId» PRIVATE )
		target_link_libraries(«resource.ProjectName»_«processId» PRIVATE«IF Config.processes > 1» ${MPI_CXX_LIBRARIES}«ENDIF»«IF Config.cores > 1» ${OpenMP_CXX_LIBRARIES}«ENDIF»)
	«ENDFOR»
	'''
}
