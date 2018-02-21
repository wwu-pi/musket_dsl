package de.wwu.musket.generator.cpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

class CMakeGenerator {
	private static final Logger logger = LogManager.getLogger(CMakeGenerator)

	def static void generateCMake(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate CMakeLists.txt.")
		resource.generateCMakeListstxt(fsa)
		logger.info("Generation of CMakeLists.txt done.")
	}

	def static void generateCMakeListstxt(Resource resource, IFileSystemAccess2 fsa) {
		fsa.generateFile(Config.base_path + "CMakeLists.txt", CMakeListstxtContent(resource))
	}

	def static CMakeListstxtContent(Resource resource) '''
		cmake_minimum_required(VERSION 3.5)
		project(«resource.ProjectName» VERSION 1.0.0 LANGUAGES CXX)
		
		# required macros
		SET( CMAKE_CXX_FLAGS_DEV "-O0 -g -march=native -m64 -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
		SET( CMAKE_CXX_FLAGS_TEST "-O3 -g -march=native -m64 -Wall -Wextra -Wpedantic -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
		SET( CMAKE_CXX_FLAGS_VTUNE "-O3 -g -DNDEBUG -march=native -m64 -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
		SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -m64 -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
		SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-O3 -DNDEBUG -march=haswell -m64 -DMPICH_IGNORE_CXX_SEEK" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
				
		# output path for binaries and libraries
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
		
		# packages		
		find_package(MPI REQUIRED)
		### this is a cmake bug: MPI link flags are preceeded by two whitespaces, which leads to one leading whitespace, which is now an error according to policy CMP0004.
		string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
		
		find_package(OpenMP REQUIRED)
				
		add_executable(«resource.ProjectName» ${PROJECT_SOURCE_DIR}/src/«resource.ProjectName».cpp)
		    target_include_directories(«resource.ProjectName» PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
		    target_compile_options(«resource.ProjectName» PRIVATE ${COMPILER_OPTIONS} ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
		    target_compile_features(«resource.ProjectName» PRIVATE cxx_auto_type cxx_lambdas cxx_nullptr cxx_uniform_initialization)
		    target_link_libraries(«resource.ProjectName» PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})
	'''
}
