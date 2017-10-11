package de.wwu.musket.generator.cpu

import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static de.wwu.musket.generator.cpu.CMakeGenerator.generateCMake
import org.apache.log4j.Logger
import org.apache.log4j.LogManager

class MusketCPUGenerator {

	// general generator info
	public static final String base_path = "CPU/"

	private static final Logger logger = LogManager.getLogger(MusketCPUGenerator)

	def static void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		// this is the starting point for the CPU generator
		logger.info("Start generation for CPU Platform.")

		// CMake files
		generateCMake(resource, fsa, context)

		logger.info("Generation for CPU platform done.")
	}
}
