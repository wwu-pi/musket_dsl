package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.MusketFunctionName
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType
import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.FoldSkeletonGenerator.*
import static de.wwu.musket.generator.cpu.LogicGenerator.*
import static de.wwu.musket.generator.cpu.MapSkeletonGenerator.*
import static de.wwu.musket.generator.cpu.RngGenerator.*

import static extension de.wwu.musket.generator.cpu.DataGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

/** 
 * Generates the source file of the project.
 * <p>
 * The generator is split into smaller functions that generates sections of the source file, such as generateGlobalConstants.
 * The entry point is the function generateSourceFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context), which is called by the CPU generator.
 */
class SourceFileGenerator {
	private static final Logger logger = LogManager.getLogger(HeaderFileGenerator)

	/**
	 * Creates the source file in the source folder of the project.
	 */
	def static void generateSourceFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate source file.")
		fsa.generateFile(Config.base_path + Config.source_path + resource.ProjectName + Config.source_extension,
			sourceFileContent(resource))
		logger.info("Generation of source file done.")
	}

	/**
	 * Generates the content of the source file. It calls several smaller functions that generate certain parts.
	 * @param resource the resource object
	 * @return content of the source file
	 */
	def static sourceFileContent(Resource resource) '''
			«generateIncludes»
			#include "../«Config.include_path + resource.ProjectName + Config.header_extension»"
			
			«generateGlobalConstants»
			«generateGlobalVariables»
			«generateTmpVariables»
			
		
			«FOR d : resource.Data»
				«d.generateObjectDefinition»
			«ENDFOR»
			
			«generateMPIFoldFunction(resource.SkeletonExpressions)»
		
			«generateMainFunction(resource)»
	'''

	/** 
	 * Generate required imports.
	 * 
	 * TODO: find a more sophisticated way to generate imports. At the moment here are simple all imports, which could be required, but they might not actyally be.
	 */
	def static generateIncludes() '''
		#include <mpi.h>
		#include <omp.h>
		#include <array>
		#include <sstream>
		#include <chrono>
		#include <random>
	'''

	/**
	 * Generates global constants, which are required but not in the model.
	 */
	def static generateGlobalConstants() '''
		const size_t «Config.var_np» = «Config.processes»;
	'''

	/**
	 * Generates global variables, which are required but not in the model.
	 */
	def static generateGlobalVariables() '''
		int «Config.var_pid» = -1;
	'''

	/**
	 * Generates temporary variable, which are required but not in the model.
	 */
	def static generateTmpVariables() '''
		size_t «Config.tmp_size_t» = 0;
	'''

	/** 
	 * Generate content of the main function in the cpp source file.
	 * 
	 * @param resource the resource object
	 */
	def static generateMainFunction(Resource resource) '''
		int main(int argc, char** argv) {
			«generateInitialization»
			
			if(«Config.var_pid» == 0){
				printf("Run «resource.ProjectName.toFirstUpper»\n\n");			
			}
			
			«IF resource.MusketFunctionCalls.exists[it.value == MusketFunctionName.RAND]»
				«generateRandomEnginesArray(resource.ConfigBlock.cores, resource.ConfigBlock.mode)»
			«ENDIF»
			
			«val rcs = resource.MusketFunctionCalls.filter[it.value == MusketFunctionName.RAND].toList»
			«generateDistributionArrays(rcs, resource.ConfigBlock.cores)»
			
			«generateInitializeDataStructures(resource)»
			
			«generateReductionDeclarations(resource)»
			«generateMPIFoldOperators(resource)»
			«generateTmpFoldResults(resource)»
			«generateOffsetVariableDeclarations(resource.SkeletonExpressions)»
			
			std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
			
			«generateLogic(resource.Model.main)»
			
			std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
			double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		
			if(«Config.var_pid» == 0){
			printf("Execution time: %.5fs\n", seconds);
			printf("Threads: %i\n", omp_get_max_threads());
			printf("Processes: %i\n", «Config.var_mpi_procs»);
			}
			
			«generateFinalization»
		}
	'''

	/**
	 * Generates boilerplate code, which is required for initialization.
	 */
	def static generateInitialization() '''
		MPI_Init(&argc, &argv);
		
		int «Config.var_mpi_procs» = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &«Config.var_mpi_procs»);
		
		if(«Config.var_mpi_procs» != «Config.var_np»){
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		
		MPI_Comm_rank(MPI_COMM_WORLD, &«Config.var_pid»);
	'''

	/**
	 * Generates boilerplate code, which is required for finalization.
	 */
	def static generateFinalization() '''
		MPI_Finalize();
		return EXIT_SUCCESS;
	'''

	/** 
	 * Generates the initialization of data structures. The method generates an if-clause that checks for the process id.
	 * Within each case all data structures of the respective process are initialized. 
	 * Struct arrays and matrices are rejected, since std::arrays, which are used, are always initialized with calls to the default constructor.
	 * 
	 * @param resource the resource object
	 * @return generated code
	 */
	def static String generateInitializeDataStructures(Resource resource) {
		var result = ""

		if (resource.Arrays.reject[it.type instanceof StructArrayType].exists[it.ValuesAsString.size > 1]) {
			for (var p = 0; p < Config.processes; p++) {
				result += '''if(«Config.var_pid» == «p»){
				'''
				for (a : resource.Arrays.reject[it.type instanceof StructArrayType]) {
					val values = a.ValuesAsString
					if (values.size > 1) {
						val sizeLocal = a.type.sizeLocal
						result += a.generateArrayInitializationForProcess(p, values.drop(sizeLocal * p).take(sizeLocal))
					}
				}
				result += '''}«IF p != Config.processes - 1» else «ENDIF»'''
			}
		}

		for (a : resource.CollectionObjects.reject [
			it.type instanceof StructArrayType || it.type instanceof StructMatrixType
		].filter[it.ValuesAsString.size < 2]) {
			result += "\n"
			result += a.generateInitializationWithSingleValue
		}

		return result
	}

}
