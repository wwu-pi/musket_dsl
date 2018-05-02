package de.wwu.musket.generator

import de.wwu.musket.MusketStandaloneSetup
import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.common.util.URI
import org.eclipse.xtext.generator.GeneratorDelegate
import org.eclipse.xtext.generator.JavaIoFileSystemAccess
import org.eclipse.xtext.resource.XtextResource
import org.eclipse.xtext.resource.XtextResourceSet
import org.eclipse.xtext.util.CancelIndicator
import org.eclipse.xtext.validation.CheckMode

/**
 * Starting point for the MusketStandaloneGenerator.
 * <p>
 * This can be used to generate models without opening an additional eclipse instance.
 * Just choose run as > java application.
 * If models should be added, include them in the list "val models = #[....]"
 * Remember to use relative paths.
 */
class MusketStandaloneGenerator {
	private static final Logger logger = LogManager.getLogger(MusketStandaloneGenerator)

	def static void main(String[] args) {

		logger.info("Start standalone Musket generator.")
		val injector = new MusketStandaloneSetup().createInjectorAndDoEMFRegistration()

		// obtain a resourceset from the injector
		val resourceSet = injector.getInstance(XtextResourceSet)

		val models = #[
//			'../de.wwu.musket.models/src/double.musket',
//			'../de.wwu.musket.models/src/fold.musket',
			'../de.wwu.musket.models/src/frobenius.musket',
//			'../de.wwu.musket.models/src/frobenius_float.musket',
			'../de.wwu.musket.models/src/fss.musket',
//			'../de.wwu.musket.models/src/matmult.musket',
			'../de.wwu.musket.models/src/matmult_float.musket',
			'../de.wwu.musket.models/src/matmult_float_transposed.musket',
//			'../de.wwu.musket.models/src/nbody.musket',
			'../de.wwu.musket.models/src/nbody_float.musket'
//			'../de.wwu.musket.models/src/plus-row.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/array.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/matrix.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/lambda.musket'
		]

		for (String s : models) {
			logger.info("Generate: " + s + '.')
			// load a resource by URI, in this case from the file system
			val resource = resourceSet.getResource(URI.createFileURI(s), true)

			// Validation
			val validator = (resource as XtextResource).getResourceServiceProvider().getResourceValidator()
			val issues = validator.validate(resource, CheckMode.ALL, CancelIndicator.NullImpl)
			for (issue : issues) {
				logger.error(issue.getMessage())
			}

			// Code Generator
			val generator = injector.getInstance(GeneratorDelegate)
			val fsa = injector.getInstance(JavaIoFileSystemAccess)
			fsa.setOutputPath("../src-gen/")
			generator.doGenerate(resource, fsa)
			logger.info("Generate: " + s + '. Done.')
		}
		logger.info("Musket standalone generator done.")
	}
}
