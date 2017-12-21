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

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.ProjectName

class MusketStandaloneGenerator {
	private static final Logger logger = LogManager.getLogger(MusketStandaloneGenerator)

	def static void main(String[] args) {
		
		logger.info("Start standalone Musket generator.")
		val injector = new MusketStandaloneSetup().createInjectorAndDoEMFRegistration()

		// obtain a resourceset from the injector
		val resourceSet = injector.getInstance(XtextResourceSet)

		// load a resource by URI, in this case from the file system
		val resource = resourceSet.getResource(
			URI.createFileURI("/home/fabian/gitlab/lspi-research/musket-dsl/de.wwu.musket.models/src/frobenius.musket"),
			true)

			// Validation
			val validator = (resource as XtextResource).getResourceServiceProvider().getResourceValidator()
			val issues = validator.validate(resource, CheckMode.ALL, CancelIndicator.NullImpl)
			for (issue : issues) {
				logger.error(issue.getMessage())
			}

			// Code Generator
			val generator = injector.getInstance(GeneratorDelegate)
			val fsa = injector.getInstance(JavaIoFileSystemAccess)
			fsa.setOutputPath("../src-gen/" + resource.ProjectName.toString + "/")			
			generator.doGenerate(resource, fsa)

			logger.info("Musket standalone generator done.")
		}
	}
	