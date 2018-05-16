package de.wwu.musket.generator.preprocessor.transformations

import org.eclipse.emf.ecore.resource.Resource

interface PreprocessorTransformation {
	
	def void run(Resource input);
}