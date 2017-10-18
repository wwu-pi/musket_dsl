package de.wwu.musket.generator.cpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.LogicGenerator.*

import static extension de.wwu.musket.generator.cpu.DataGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

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
		
		«««		«FOR f : resource.Functions»
«««			«f.generateFunctorDefinition»
«««		«ENDFOR»
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

	def static generateInitializeDataStructures(Resource resource) '''
		«FOR d : resource.Arrays»
			«d.generateArrayInitialization»
		«ENDFOR»
	'''

}
