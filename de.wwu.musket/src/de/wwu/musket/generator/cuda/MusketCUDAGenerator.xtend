package de.wwu.musket.generator.cuda

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cuda.CMakeGenerator.generateCMake
import static de.wwu.musket.generator.cuda.HeaderFileGenerator.generateHeaderFile
import static de.wwu.musket.generator.cuda.HeaderFileGenerator.generateProcessHeaderFiles
import static de.wwu.musket.generator.cuda.RunScriptGenerator.generateRunScript
import static de.wwu.musket.generator.cuda.SourceFileGenerator.generateSourceFile
import static de.wwu.musket.generator.cuda.SlurmGenerator.generateSlurmJob
import static de.wwu.musket.generator.cuda.lib.Musket.generateMusketHeaderFile
import static de.wwu.musket.generator.cuda.lib.Kernel.generateKernelHeaderFile

/** 
 * This is the start of the GPU generator.
 * <p>
 * In this class all other generators are called, so that a working generated project is the result.
 */
class MusketCUDAGenerator {

	// general generator info
	
	private static final Logger logger = LogManager.getLogger(MusketCUDAGenerator)

/**
 * This is the starting point for the GPU generator.
 * All other generators are called from this method.
 * There are file-related generators (such as headerFileGenerator) and functionality-related generators (such as skeletonGenerator).
 * First all the file-related generators are called in this method.
 * They call the other generators later as required.
 */
	def static void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Start generation for CUDA Platform.")

		// config
		Config.init(resource)

		// run scripts 
		generateRunScript(resource, fsa, context)
		
		generateSlurmJob(resource, fsa, context)

		// build files
		generateCMake(resource, fsa, context)
		
		// lib header files
		generateMusketHeaderFile(resource, fsa, context)
		generateKernelHeaderFile(resource, fsa, context)
		generateHeaderFile(resource, fsa, context)
		//generateDArrayHeaderFile(resource, fsa, context)
		//generateDMatrixHeaderFile(resource, fsa, context)
		
				
		// source code
		for(var i = 0; i < Config.processes; i++){
			generateProcessHeaderFiles(resource, fsa, context, i)		
			generateSourceFile(resource, fsa, context, i)
		}
		logger.info("Generation for CUDA platform done.")
	}
}
