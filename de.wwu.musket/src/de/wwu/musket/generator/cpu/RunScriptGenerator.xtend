package de.wwu.musket.generator.cpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

/**
 * Generates bash scripts that build and run the application.
 * <p>
 * Entry point is the method generateRunScript(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context).
 * There is one file generated, called build-and-run.sh, which creates the directories, calls cmake, make, and mpirun.
 * 
 * TODO: since cmake just has to be called once, and since it might be necessary to build a project and not to run it, this might be split into more separate scripts.
 */
class RunScriptGenerator {
	private static final Logger logger = LogManager.getLogger(RunScriptGenerator)

	/**
	 * Starting point for the RunFile generator. It creates a script in the base_path.
	 */
	def static void generateRunScript(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate build-and-run.sh.")
		fsa.generateFile(Config.base_path + "build-and-run.sh", RunScriptContent(resource))
		logger.info("Generation of build-and-run.sh done.")
	}

	/**
	 * Generates the content of the run script.
	 * 
	 * TODO: adjust build type based on config in the model.
	 * 
	 * @param resource the resource object
	 * @return the content of the run script
	 */
	def static RunScriptContent(Resource resource) '''
		#!/bin/bash
		
		# remove files and create folder
		rm -rf -- build && \
		mkdir build && \
		
		# run cmake
		cd build/ && \
		cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Release ../ && \
		
		make «resource.ProjectName» && \
		mpirun -np «Config.processes» bin/«resource.ProjectName» 
	'''
}
