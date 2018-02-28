package de.wwu.musket.generator

import de.wwu.musket.generator.cpu.MusketCPUGenerator
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.AbstractGenerator
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import org.apache.log4j.Logger
import org.apache.log4j.LogManager

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

/**
 * Generates code from your model files on save.
 * 
 * See https://www.eclipse.org/Xtext/documentation/303_runtime_concepts.html#code-generation
 */
class MusketGenerator extends AbstractGenerator {

	private static final Logger logger = LogManager.getLogger(MusketGenerator)

	override void doGenerate(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Start the Musket generator.")

		if (resource.isPlatformCPU) {
			MusketCPUGenerator.doGenerate(resource, fsa, context)
		}

		logger.info("Musket generator done.")
		logger.info("Done.")
	}

}
