package de.wwu.musket.generator.gpu

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
		
		# required macros
		SET( CMAKE_CXX_FLAGS_DEV "-g -O0 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
		SET( CMAKE_CXX_FLAGS_TEST "-gopt -fast -ta:tesla:cc60,pinned,nollvm -O4 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
		SET( CMAKE_CXX_FLAGS_VTUNE "-gopt -fast -O4 -w" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
		SET( CMAKE_CXX_FLAGS_BENCHMARK "-fast -O4 -ta:tesla:cc60,pinned,nollvm -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
		SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-fast -O4 -tp -ta:tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Palma." FORCE )
		SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-fast -O4 -tp=haswell -ta:tesla:cc35,pinned,nollvm -w --c++14" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Taurus." FORCE )
		SET( CMAKE_CUDA_FLAGS_BENCHMARKTAURUS "--std=c++14 -arch=compute_35 -code=sm_35")
			
		# output path for binaries and libraries
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
		
		# packages
		«IF Config.processes > 1»
			find_package(MPI REQUIRED)
«««			### this is a cmake bug: MPI link flags are preceeded by two whitespaces, which leads to one leading whitespace, which is now an error according to policy CMP0004.
«««			string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
		«ENDIF»
		
		find_package(OpenMP REQUIRED)
		
		find_package(OpenACC REQUIRED)
		
		find_package(CUDA REQUIRED)
		
		«FOR processId : 0 ..< Config.processes»
			add_executable(«resource.ProjectName»_«processId» ${PROJECT_SOURCE_DIR}/src/«resource.ProjectName»_«processId»«Config.source_extension»)
			target_compile_features(«resource.ProjectName»_«processId» PRIVATE cxx_std_14)
			target_include_directories(«resource.ProjectName»_«processId» PRIVATE ${PROJECT_SOURCE_DIR}/include«IF Config.processes > 1» ${MPI_CXX_INCLUDE_DIRS}«ENDIF» ${CUDA_INCLUDE_DIRS})
			target_compile_definitions(«resource.ProjectName»_«processId» PRIVATE«IF Config.processes > 1» ${MPI_CXX_COMPILE_DEFINITIONS}«ENDIF»)
			target_compile_options(«resource.ProjectName»_«processId» PRIVATE«IF Config.processes > 1» ${MPI_CXX_COMPILE_OPTIONS}«ENDIF» ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
			target_link_libraries(«resource.ProjectName»_«processId» PRIVATE«IF Config.processes > 1» ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES}«ENDIF» ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS} ${CUDA_LIBRARIES} ${CUDA_cudart_static_LIBRARY} ${CUDA_cudadevrt_LIBRARY} ${CUDA_curand_LIBRARY})
		«ENDFOR»
	'''
}
