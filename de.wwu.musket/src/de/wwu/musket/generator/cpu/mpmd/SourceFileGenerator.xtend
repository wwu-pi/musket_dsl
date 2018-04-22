package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketFunctionName
import de.wwu.musket.musket.StructArrayType
import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.mpmd.FoldSkeletonGenerator.*
import static de.wwu.musket.generator.cpu.mpmd.LogicGenerator.*
import static de.wwu.musket.generator.cpu.mpmd.MapSkeletonGenerator.*
import static de.wwu.musket.generator.cpu.mpmd.RngGenerator.*

import static extension de.wwu.musket.generator.cpu.mpmd.DataGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.StructGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

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
	def static void generateSourceFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context, int processId) {
		logger.info("Generate source file.")
		fsa.generateFile(Config.base_path + Config.source_path + resource.ProjectName +  '_' + processId + Config.source_extension,
			sourceFileContent(resource, processId))
		logger.info("Generation of source file done.")
	}

	/**
	 * Generates the content of the source file. It calls several smaller functions that generate certain parts.
	 * @param resource the resource object
	 * @return content of the source file
	 */
	def static sourceFileContent(Resource resource, int processId) '''
		«generateIncludes»
		#include "../«Config.include_path + resource.ProjectName + "_" + processId + Config.header_extension»"
		
		«generateGlobalConstants(processId)»
		«generateGlobalVariables»
		«generateTmpVariables»
		
		
		«FOR d : resource.Data»
			«d.generateObjectDefinition(processId)»
		«ENDFOR»
		
		«FOR s : resource.Structs»
			«s.generateStructDefaultConstructor»
		«ENDFOR»
		
		«IF Config.processes > 1»
			«generateMPIFoldFunction(resource.SkeletonExpressions, processId)»
		«ENDIF»
		«generateMainFunction(resource, processId)»
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
		
		#include <omp.h>
		#include <array>
		#include <vector>
		#include <sstream>
		#include <chrono>
		#include <random>
		#include <limits>
		#include <memory>
	'''

	/**
	 * Generates global constants, which are required but not in the model.
	 */
	def static generateGlobalConstants(int processId) '''
		«IF Config.processes > 1»
			const size_t «Config.var_np» = «Config.processes»;
			const size_t «Config.var_pid» = «processId»;
		«ENDIF»
	'''

	/**
	 * Generates global variables, which are required but not in the model.
	 */
	def static generateGlobalVariables() '''
		«IF Config.processes > 1»
			int «Config.var_mpi_rank» = -1;
			int «Config.var_mpi_procs» = 0;
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
	def static generateMainFunction(Resource resource, int processId) '''
		int main(int argc, char** argv) {
			«generateInitialization»
			
			«IF Config.processes > 1 && processId == 0»
				printf("Run «resource.ProjectName.toFirstUpper»\n\n");			
			«ENDIF»
			
«««			TODO: as global variable
			«IF resource.MusketFunctionCalls.exists[it.value == MusketFunctionName.RAND]»
				«generateRandomEnginesArray(resource.ConfigBlock.cores, resource.ConfigBlock.mode)»
			«ENDIF»
			
			«val rcs = resource.MusketFunctionCalls.filter[it.value == MusketFunctionName.RAND].toList»
			«generateDistributionArrays(rcs, resource.ConfigBlock.cores)»
«««			rng end
			
			«generateInitializeDataStructures(resource, processId)»
			«generateReductionDeclarations(resource, processId)»
			
			«IF Config.processes > 1»
				«generateMPIFoldOperators(resource)»
				«generateTmpFoldResults(resource)»
				«generateOffsetVariableDeclarations(resource.SkeletonExpressions)»
			«ENDIF»
			
			«generateLogic(resource.Model.main, processId)»
			
			«IF processId == 0»		
				«IF resource.Model.main.content.exists[it instanceof MusketFunctionCall && (it as MusketFunctionCall).value == MusketFunctionName.ROI_START]»
					printf("Execution time: %.5fs\n", seconds);
				«ENDIF»
				printf("Threads: %i\n", «IF Config.cores > 1»omp_get_max_threads()«ELSE»«Config.cores»«ENDIF»);
				printf("Processes: %i\n", «IF Config.processes > 1»«Config.var_mpi_procs»«ELSE»«Config.processes»«ENDIF»);
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
			
			MPI_Comm_size(MPI_COMM_WORLD, &«Config.var_mpi_procs»);
			MPI_Comm_rank(MPI_COMM_WORLD, &«Config.var_mpi_rank»);
			
			if(«Config.var_mpi_procs» != «Config.var_np» || «Config.var_mpi_rank» != «Config.var_pid»){
				MPI_Finalize();
				return EXIT_FAILURE;
			}			
			
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
	def static String generateInitializeDataStructures(Resource resource, int processId) {
		var result = ""

		// init distributed arrays with values
		if (resource.Arrays.reject [
			it.type instanceof StructArrayType || it.type.distributionMode == DistributionMode.COPY ||
				it.type.distributionMode == DistributionMode.LOC
		].exists [
			it.ValuesAsString.size > 1
		] && Config.processes > 1) {
//		distributed
				for (a : resource.Arrays.reject [
					it.type instanceof StructArrayType || it.type.distributionMode == DistributionMode.COPY
				]) {
					val values = a.ValuesAsString
					if (values.size > 1) {
						val sizeLocal = a.type.sizeLocal(processId)
						result += a.generateArrayInitializationForProcess(values.drop((sizeLocal * processId) as int).take(sizeLocal as int))
					}
				}
		} else {
//		copy distributed
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

		return result
	}

}
