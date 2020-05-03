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
import static extension de.wwu.musket.generator.cpu.StructGenerator.*

import static extension de.wwu.musket.generator.cpu.DataGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.cpu.util.ObjectExtension.*
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.DistributionMode

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
		
		«FOR s : resource.Structs»
			«s.generateStructDefaultConstructor»
		«ENDFOR»
		
		«IF Config.processes > 1»
			«generateMPIFoldFunction(resource.SkeletonExpressions)»
		«ENDIF»
		«generateMainFunction(resource)»
	'''

	/** 
	 * Generate required imports.
	 * 
	 * TODO: find a more sophisticated way to generate imports. At the moment here are simple all imports, which could be required, but they might not actyally be.
	 */
	def static generateIncludes() '''
		«IF Config.processes > 1»
			#include <mpi.h>
		«ENDIF»
		
		«IF Config.cores > 1»
			#include <omp.h>
		«ENDIF»
		#include <array>
		#include <sstream>
		#include <chrono>
		#include <random>
		#include <limits>
		#include <memory>
	'''

	/**
	 * Generates global constants, which are required but not in the model.
	 */
	def static generateGlobalConstants() '''
		«IF Config.processes > 1»
			const size_t «Config.var_np» = «Config.processes»;
		«ENDIF»
	'''

	/**
	 * Generates global variables, which are required but not in the model.
	 */
	def static generateGlobalVariables() '''
		«IF Config.processes > 1»
			int «Config.var_pid» = -1;
		«ENDIF»
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
			
			«IF Config.processes > 1»
				if(«Config.var_pid» == 0){
			«ENDIF»
			printf("Run «resource.ProjectName.toFirstUpper»\n\n");			
			«IF Config.processes > 1»
				}
			«ENDIF»
			
			«IF resource.MusketFunctionCalls.exists[it.value == MusketFunctionName.RAND]»
				«generateRandomEnginesArray(resource.ConfigBlock.cores, resource.ConfigBlock.mode)»
			«ENDIF»
			
			«val rcs = resource.MusketFunctionCalls.filter[it.value == MusketFunctionName.RAND].toList»
			«generateDistributionArrays(rcs, resource.ConfigBlock.cores)»
			
			«generateInitializeDataStructures(resource)»
			«IF Config.cores > 1»
				«generateReductionDeclarations(resource)»
			«ENDIF»
			
			«IF Config.processes > 1»
				«generateMPIFoldOperators(resource)»			
				«generateTmpFoldResults(resource)»
				«generateOffsetVariableDeclarations(resource.SkeletonExpressions)»
			«ENDIF»
			
			«generateLogic(resource.Model.main)»
			
			«IF Config.processes > 1»		
				if(«Config.var_pid» == 0){
			«ENDIF»
			«IF resource.Model.main.content.exists[it instanceof MusketFunctionCall && (it as MusketFunctionCall).value == MusketFunctionName.ROI_START]»
				printf("Execution time: %.5fs\n", seconds);
			«ENDIF»
			printf("Threads: %i\n", «IF Config.cores > 1»omp_get_max_threads()«ELSE»«Config.cores»«ENDIF»);
			printf("Processes: %i\n", «IF Config.processes > 1»«Config.var_mpi_procs»«ELSE»«Config.processes»«ENDIF»);
			«IF Config.processes > 1»	
				}
			«ENDIF»
			
			«generateFinalization»
		}
	'''

	/**
	 * Generates boilerplate code, which is required for initialization.
	 */
	def static generateInitialization() '''
		«IF Config.processes > 1»
			MPI_Init(&argc, &argv);
			
			int «Config.var_mpi_procs» = 0;
			MPI_Comm_size(MPI_COMM_WORLD, &«Config.var_mpi_procs»);
			
			if(«Config.var_mpi_procs» != «Config.var_np»){
				MPI_Finalize();
				return EXIT_FAILURE;
			}
			
			MPI_Comm_rank(MPI_COMM_WORLD, &«Config.var_pid»);
		«ENDIF»
	'''

	/**
	 * Generates boilerplate code, which is required for finalization.
	 */
	def static generateFinalization() '''
		«IF Config.processes > 1»
			MPI_Finalize();
		«ENDIF»
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

		// init distributed arrays with values
		if (resource.Arrays.reject [
			it.type instanceof StructArrayType || it.type.distributionMode == DistributionMode.COPY ||
				it.type.distributionMode == DistributionMode.LOC
		].exists [
			it.ValuesAsString.size > 1
		] && Config.processes > 1) {
			result += "switch(" + Config.var_pid + "){\n"
			for (var p = 0; p < Config.processes; p++) {
				result += "case " + p + ": {\n"
				for (a : resource.Arrays.reject [
					it.type instanceof StructArrayType || it.type.distributionMode == DistributionMode.COPY
				]) {
					val values = a.ValuesAsString
					if (values.size > 1) {
						val sizeLocal = a.type.sizeLocal
						result += a.generateArrayInitializationForProcess(values.drop((sizeLocal * p) as int).take(sizeLocal as int))
					}
				}
				result += "break;\n}\n"
			}
			result += "}\n\n"
		} else {
			for (a : resource.Arrays.reject[it.type instanceof StructArrayType]) {
				val values = a.ValuesAsString
				if (values.size > 1) {
					result += a.generateArrayInitializationForProcess(values)
				}
			}
		}

		// init copy dist data structures with init list
		for (a : resource.Arrays.reject[it.type instanceof StructArrayType].filter [
			it.type.distributionMode == DistributionMode.COPY
		]) {
			val values = a.ValuesAsString
			if (values.size > 1) {
				result += a.generateArrayInitializationForProcess(values)
			}
		}

		// init data structures with single value
		for (a : resource.CollectionObjects.reject [
			it.type instanceof StructArrayType || it.type instanceof StructMatrixType
		].filter[it.ValuesAsString.size < 2]) {
			result += "\n"
			result += a.generateInitializationWithSingleValue
		}

		return result
	}

}
