package de.wwu.musket.generator.cpu

import org.apache.log4j.Logger
import org.apache.log4j.LogManager
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

class RunScriptGenerator {
	private static final Logger logger = LogManager.getLogger(RunScriptGenerator)

	def static void generateRunScript(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate build-and-run.sh.")
		fsa.generateFile(Config.base_path + "build-and-run.sh", RunScriptContent(resource))
		logger.info("Generation of build-and-run.sh done.")
	}

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
