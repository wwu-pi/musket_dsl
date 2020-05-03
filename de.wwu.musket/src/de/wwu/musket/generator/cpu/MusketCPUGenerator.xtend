package de.wwu.musket.generator.cpu

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.CMakeGenerator.generateCMake
import static de.wwu.musket.generator.cpu.HeaderFileGenerator.generateHeaderFile
import static de.wwu.musket.generator.cpu.RunScriptGenerator.generateRunScript
import static de.wwu.musket.generator.cpu.SourceFileGenerator.generateSourceFile
import static de.wwu.musket.generator.cpu.SlurmGenerator.generateSlurmJob

/** 
 * This is the start of the CPU generator.
 * <p>
 * In this class all other generators are called, so that a working generated project is the result.
 */
class MusketCPUGenerator {

	// general generator info
	
	private static final Logger logger = LogManager.getLogger(MusketCPUGenerator)

/**
 * This is the starting point for the CPU generator.
 * All other generators are called from this method.
 * There are file-related generators (such as headerFileGenerator) and functionality-related generators (such as skeletonGenerator).
 * First all the file-related generators are called in this method.
 * They call the other generators later as required.
 */
	def static void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Start generation for CPU Platform.")

		// config
		Config.init(resource)

		// run scripts 
		generateRunScript(resource, fsa, context)
		
		generateSlurmJob(resource, fsa, context)

		// build files
		generateCMake(resource, fsa, context)
		
		// source code
		generateHeaderFile(resource, fsa, context)
		
		generateSourceFile(resource, fsa, context)

		logger.info("Generation for CPU platform done.")
	}
}
