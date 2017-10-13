package de.wwu.musket.generator.cpu

import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.RunScriptGenerator.generateRunScript
import static de.wwu.musket.generator.cpu.CMakeGenerator.generateCMake
import static de.wwu.musket.generator.cpu.HeaderFileGenerator.generateHeaderFile
import static de.wwu.musket.generator.cpu.SourceFileGenerator.generateSourceFile

import org.apache.log4j.Logger
import org.apache.log4j.LogManager

import de.wwu.musket.generator.cpu.Config

class MusketCPUGenerator {

	// general generator info
	
	private static final Logger logger = LogManager.getLogger(MusketCPUGenerator)

	def static void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		// this is the starting point for the CPU generator
		logger.info("Start generation for CPU Platform.")

		// config
		Config.init(resource)

		generateRunScript(resource, fsa, context)

		// CMake files
		generateCMake(resource, fsa, context)
		
		generateHeaderFile(resource, fsa, context)
		
		generateSourceFile(resource, fsa, context)

		logger.info("Generation for CPU platform done.")
	}
}
