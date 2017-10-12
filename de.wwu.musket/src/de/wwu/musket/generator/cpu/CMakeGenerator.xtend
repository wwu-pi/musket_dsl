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
		project(«resource.ProjectName»)
		
		# required macros
		include(CheckCXXCompilerFlag)
		
		# output path for binaries and libraries
		set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
		set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
		
		# packages		
		find_package(MPI REQUIRED)
		### this is a cmake bug: MPI link flags are preceeded by two whitespaces, which leads to one leading whitespace, which is now an error according to policy CMP0004.
		string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)
		
		find_package(OpenMP REQUIRED)
		
		# check supported flags
		CHECK_CXX_COMPILER_FLAG(-Wall compiler_flag_wall)
		CHECK_CXX_COMPILER_FLAG(-m64 compiler_flag_m64)
		CHECK_CXX_COMPILER_FLAG(-fno-strict-aliasing compiler_flag_fno_strict_aliasing)
		CHECK_CXX_COMPILER_FLAG(-DMPICH_IGNORE_CXX_SEEK compiler_flag_dmpich_ignore_cxx_seek)
		
		# set the supported flags
		if(compiler_flag_wall)
		    set(COMPILER_OPTIONS ${COMPILER_OPTIONS} -Wall)
		endif(compiler_flag_wall)
		
		if(compiler_flag_m64)
		    set(COMPILER_OPTIONS ${COMPILER_OPTIONS} -m64)
		endif(compiler_flag_m64)
		
		if(compiler_flag_fno_strict_aliasing)
		    set(COMPILER_OPTIONS ${COMPILER_OPTIONS} -fno-strict-aliasing)
		endif(compiler_flag_fno_strict_aliasing)
		
		if(compiler_flag_dmpich_ignore_cxx_seek)
		    set(COMPILER_OPTIONS ${COMPILER_OPTIONS} -DMPICH_IGNORE_CXX_SEEK)
		endif(compiler_flag_dmpich_ignore_cxx_seek)
		
		add_executable(«resource.ProjectName» ${PROJECT_SOURCE_DIR}/src/«resource.ProjectName».cpp)
		    target_include_directories(«resource.ProjectName» PRIVATE ${PROJECT_SOURCE_DIR}/include/ ${MPI_CXX_INCLUDE_PATH})
		    target_compile_options(«resource.ProjectName» PRIVATE ${COMPILER_OPTIONS} ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
		    target_compile_features(«resource.ProjectName» PRIVATE cxx_auto_type cxx_lambdas cxx_nullptr cxx_uniform_initialization)
		    target_link_libraries(«resource.ProjectName» PRIVATE ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_FLAGS})
	'''
}
