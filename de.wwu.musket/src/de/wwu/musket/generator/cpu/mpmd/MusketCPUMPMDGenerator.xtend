package de.wwu.musket.generator.cpu.mpmd

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.mpmd.CMakeGenerator.generateCMake
import static de.wwu.musket.generator.cpu.mpmd.HeaderFileGenerator.generateHeaderFile
import static de.wwu.musket.generator.cpu.mpmd.RunScriptGenerator.generateRunScript
import static de.wwu.musket.generator.cpu.mpmd.SourceFileGenerator.generateSourceFile
import static de.wwu.musket.generator.cpu.mpmd.SlurmGenerator.generateSlurmJob
import static de.wwu.musket.generator.cpu.mpmd.lib.DArray.generateDArrayHeaderFile
import static de.wwu.musket.generator.cpu.mpmd.lib.DMatrix.generateDMatrixHeaderFile
import static de.wwu.musket.generator.cpu.mpmd.lib.Musket.generateMusketHeaderFile

/** 
 * This is the start of the CPU generator.
 * <p>
 * In this class all other generators are called, so that a working generated project is the result.
 */
class MusketCPUMPMDGenerator {

	// general generator info
	
	private static final Logger logger = LogManager.getLogger(MusketCPUMPMDGenerator)

/**
 * This is the starting point for the CPU generator.
 * All other generators are called from this method.
 * There are file-related generators (such as headerFileGenerator) and functionality-related generators (such as skeletonGenerator).
 * First all the file-related generators are called in this method.
 * They call the other generators later as required.
 */
	def static void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Start generation for CPU Platform with MPMD.")

		// config
		Config.init(resource)

		// run scripts 
		generateRunScript(resource, fsa, context)
		
		generateSlurmJob(resource, fsa, context)

		// build files
		generateCMake(resource, fsa, context)
		
		// lib header files
		generateMusketHeaderFile(resource, fsa, context)
		//generateDArrayHeaderFile(resource, fsa, context)
		//generateDMatrixHeaderFile(resource, fsa, context)
		
				
		// source code
		for(var i = 0; i < Config.processes; i++){
			generateHeaderFile(resource, fsa, context, i)		
			generateSourceFile(resource, fsa, context, i)
		}
		logger.info("Generation for CPU platform wit MPMD done.")
	}
}
