package de.wwu.musket.generator.cpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.cpu.DataGenerator.*
import static extension de.wwu.musket.generator.cpu.FunctorGenerator.*

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
		#include "../«Config.include_path + resource.ProjectName + Config.header_extension»"

		«generateGlobalConstants»
		«generateGlobalVariables»

		«FOR d : resource.Data»
			«d.generateObjectDefinition»
		«ENDFOR»
		
«««		«FOR f : resource.Functions»
«««			«f.generateFunctorDefinition»
«««		«ENDFOR»
		«generateMainFunction(resource)»
	'''
	
	def static generateGlobalConstants()'''
		const size_t «Config.var_np» = «Config.processes»;
	'''
	
	def static generateGlobalVariables()'''
		size_t «Config.var_pid»;
	'''
	
	def static generateMainFunction(Resource resource)'''
		int main(int argc, char** argv) {
			«generateInitialization»
			«generateInitializeDataStructures(resource)»
			«generateFinalization»
		}
	'''
	
	def static generateInitialization()'''
		MPI_Init(&argc, &argv);
		
		size_t «Config.var_mpi_procs»;
		MPI_Comm_size( MPI_COMM_WORLD, &«Config.var_mpi_procs» );
		
		if(«Config.var_mpi_procs» != «Config.processes»){
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		
		MPI_Comm_rank(MPI_COMM_WORLD, &«Config.var_pid»);
	'''
	
	def static generateFinalization()'''
		MPI_Finalize();
		return EXIT_SUCCESS;
	'''
	
	def static generateInitializeDataStructures(Resource resource)'''
		«FOR d : resource.Arrays»
			«d.generateArrayInitialization»
		«ENDFOR»
	'''

	
}
