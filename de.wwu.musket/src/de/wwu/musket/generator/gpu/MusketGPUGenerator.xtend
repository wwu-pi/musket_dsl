package de.wwu.musket.generator.gpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.gpu.CMakeGenerator.generateCMake
import static de.wwu.musket.generator.gpu.HeaderFileGenerator.generateHeaderFile
import static de.wwu.musket.generator.gpu.HeaderFileGenerator.generateProcessHeaderFiles
import static de.wwu.musket.generator.gpu.RunScriptGenerator.generateRunScript
import static de.wwu.musket.generator.gpu.SourceFileGenerator.generateSourceFile
import static de.wwu.musket.generator.gpu.SlurmGenerator.generateSlurmJob
import static de.wwu.musket.generator.gpu.lib.Musket.generateMusketHeaderFile

/** 
 * This is the start of the GPU generator.
 * <p>
 * In this class all other generators are called, so that a working generated project is the result.
 */
class MusketGPUGenerator {

	// general generator info
	
	private static final Logger logger = LogManager.getLogger(MusketGPUGenerator)

/**
 * This is the starting point for the GPU generator.
 * All other generators are called from this method.
 * There are file-related generators (such as headerFileGenerator) and functionality-related generators (such as skeletonGenerator).
 * First all the file-related generators are called in this method.
 * They call the other generators later as required.
 */
	def static void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Start generation for GPU Platform.")

		// config
		Config.init(resource)

		// run scripts 
		generateRunScript(resource, fsa, context)
		
		generateSlurmJob(resource, fsa, context)

		// build files
		generateCMake(resource, fsa, context)
		
		// lib header files
		generateMusketHeaderFile(resource, fsa, context)
		generateHeaderFile(resource, fsa, context)
		//generateDArrayHeaderFile(resource, fsa, context)
		//generateDMatrixHeaderFile(resource, fsa, context)
		
				
		// source code
		for(var i = 0; i < Config.processes; i++){
			generateProcessHeaderFiles(resource, fsa, context, i)		
			generateSourceFile(resource, fsa, context, i)
		}
		logger.info("Generation for GPU platform done.")
	}
}
