package de.wwu.musket.generator

import de.wwu.musket.generator.cpu.MusketCPUGenerator
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.AbstractGenerator
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import org.apache.log4j.Logger
import org.apache.log4j.LogManager

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import de.wwu.musket.generator.cpu.mpmd.MusketCPUMPMDGenerator
import de.wwu.musket.generator.gpu.MusketGPUGenerator
import de.wwu.musket.generator.preprocessor.MusketPreprocessor

/**
 * Generates code from your model files on save.
 * 
 * See https://www.eclipse.org/Xtext/documentation/303_runtime_concepts.html#code-generation
 */
class MusketGenerator extends AbstractGenerator {

	private static final Logger logger = LogManager.getLogger(MusketGenerator)

	override void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Start model preprocessing.")
		// Perform model transformations
		val preprocessedModel = MusketPreprocessor.getPreprocessedModel(resource)
		logger.info("Preprocessing done.")
		
		logger.info("Start the Musket generator.")

		if (resource.isPlatformCPU) {
			MusketCPUGenerator.doGenerate(preprocessedModel, fsa, context)
		}
		
		if (resource.isPlatformCPUMPMD) {
			MusketCPUMPMDGenerator.doGenerate(preprocessedModel, fsa, context)
		}
		
		if (resource.isPlatformGPU) {
			MusketGPUGenerator.doGenerate(preprocessedModel, fsa, context)
		}
		
		logger.info("Musket generator done.")
		logger.info("Done.")
	}

}
