package de.wwu.musket.generator.cuda

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext
import de.wwu.musket.musket.Mode

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

/**
 * Generates bash scripts that build and run the application.
 * <p>
 * Entry point is the method generateRunScript(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context).
 * There are two files generated, called build-and-run.sh and build-and-submit.sh, which create the directories, call cmake, make, and mpirun/sbatch.
 */
class RunScriptGenerator {
	private static final Logger logger = LogManager.getLogger(RunScriptGenerator)

	/**
	 * Starting point for the RunFile generator. It creates a script in the base_path.
	 */
	def static void generateRunScript(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate run scripts.")
		fsa.generateFile(Config.base_path + "build-and-run.sh", BuildAndRunScriptContent(resource))
		fsa.generateFile(Config.base_path + "build-and-submit.sh", BuildAndSubmitScriptContent(resource))
		fsa.generateFile(Config.base_path + "nvprof.sh", NvprofScriptContent(resource))
		logger.info("Generation of run scripts done.")
	}

	/**
	 * Generates the content of the run script.
	 *  
	 * @param resource the resource object
	 * @return the content of the build-and-run script
	 */
	def static BuildAndRunScriptContent(Resource resource) '''
		#!/bin/bash
		
		# remove files and create folder
		rm -rf -- «Config.build_folder» && \
		mkdir «Config.build_folder» && \
		
		# run cmake
		cd «Config.build_folder» && \
		cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=«IF Config.mode == Mode.DEBUG»Dev«ELSE»Test«ENDIF» -D CMAKE_CXX_COMPILER=g++ ../ && \
		
		«FOR p: 0 ..< Config.processes»
			make «resource.ProjectName»_«p» && \
		«ENDFOR»
		
		«IF Config.processes > 1»
			mpirun «FOR p : 0 ..< Config.processes SEPARATOR " : "»-np 1 bin/«resource.ProjectName»_«p»«ENDFOR»
		«ELSE»
			bin/«resource.ProjectName»_0
		«ENDIF»
	'''

	/**
	 * Generates the content of the build-and-submit script.
	 * The file job.sh is generated in the SlurmGenerator.
	 * 
	 * @param resource the resource object
	 * @return the content of the build-and-submit script
	 */
	def static BuildAndSubmitScriptContent(Resource resource) '''
		#!/bin/bash
		
		source_folder=${PWD} && \
		
		# remove files and create folder
		mkdir -p «Config.out_path» && \
		rm -rf -- «Config.build_path»benchmark && \
		mkdir -p «Config.build_path»benchmark && \
		
		# run cmake
		cd «Config.build_path»benchmark && \
		cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

		«FOR p: 0 ..< Config.processes»
			make «resource.ProjectName»_«p» && \
		«ENDFOR»
		cd ${source_folder} && \

		sbatch job.sh
	'''
	
	def static NvprofScriptContent(Resource resource) '''
		#!/bin/bash
		
		source_folder=${PWD} && \
		
		# remove files and create folder
		mkdir -p «Config.out_path» && \
		rm -rf -- «Config.build_path»nvprof && \
		mkdir -p «Config.build_path»nvprof && \
		
		# run cmake
		cd «Config.build_path»nvprof && \
		cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

		«FOR p: 0 ..< Config.processes»
			make «resource.ProjectName»_«p» && \
		«ENDFOR»
		cd ${source_folder} && \

		sbatch nvprof-job.sh
	'''

}
