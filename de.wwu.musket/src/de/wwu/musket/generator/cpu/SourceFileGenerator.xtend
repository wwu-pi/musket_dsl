package de.wwu.musket.generator.cpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.LogicGenerator.*
import static de.wwu.musket.generator.cpu.FoldSkeletonGenerator.*

import static extension de.wwu.musket.generator.cpu.DataGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.Array

class SourceFileGenerator {
	private static final Logger logger = LogManager.getLogger(HeaderFileGenerator)

	def static void generateSourceFile(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate source file.")
		fsa.generateFile(Config.base_path + Config.source_path + resource.ProjectName + Config.source_extension,
			sourceFileContent(resource))
		logger.info("Generation of source file done.")
	}

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

	def static generateGlobalConstants() '''
		const size_t «Config.var_np» = «Config.processes»;
	'''

	def static generateGlobalVariables() '''
		int «Config.var_pid» = -1;
	'''

	def static generateTmpVariables() '''
		size_t «Config.tmp_size_t» = 0;
	'''

	def static generateMainFunction(Resource resource) '''
		int main(int argc, char** argv) {
			«generateInitialization»
			
			if(«Config.var_pid» == 0){
				printf("Run «resource.ProjectName.toFirstUpper»\n\n");			
			}
			
			«generateInitializeDataStructures(resource)»
			
			«generateReductionDeclarations(resource)»
			«generateMPIFoldOperators(resource)»
			«generateTmpFoldResults(resource)»
			
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

	def static generateIncludes() '''
		#include <mpi.h>
		#include <omp.h>
		#include <array>
		#include <sstream>
		#include <chrono>
	'''

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

	def static generateFinalization() '''
		MPI_Finalize();
		return EXIT_SUCCESS;
	'''

	def static String generateInitializeDataStructures(Resource resource) {
		var result = ""

		for (var p = 0; p < Config.processes; p++) {
			result += '''if(«Config.var_pid» == «p»){
			'''
			for (a : resource.Arrays) {
				if (a.ValuesAsString.size > 1) {
					val sizeLocal = a.sizeLocal
					result +=
						a.generateArrayInitializationForProcess(p, a.ValuesAsString.drop(sizeLocal * p).take(sizeLocal))
				}
			}
			result += '''}«IF p != Config.processes - 1» else «ENDIF»'''
		}

		for (a : resource.Arrays.filter[a|a.ValuesAsString.size < 2]) {
			result += "\n"
			result += a.generateArrayInitializationWithSingleValue
		}

		return result
	}

}
