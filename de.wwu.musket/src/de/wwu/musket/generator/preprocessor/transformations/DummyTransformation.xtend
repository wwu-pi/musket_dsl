package de.wwu.musket.generator.preprocessor.transformations

import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory

class DummyTransformation extends PreprocessorTransformation {
	
	new(MusketComplexElementFactory factory) {
		super(factory)
	}
	
	override run(Resource input) {
		// Do nothing
	}
	
}