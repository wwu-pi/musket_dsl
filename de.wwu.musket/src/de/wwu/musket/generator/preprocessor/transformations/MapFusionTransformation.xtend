package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import org.eclipse.emf.ecore.resource.Resource

class MapFusionTransformation extends PreprocessorTransformation {
	
	new(MusketComplexElementFactory factory) {
		super(factory)
	}
	
	override run(Resource input) {
		// TODO
		
		// Rename duplicate attribute names
		
		// Replace return expression with temporary variable
		 
		// Copy function content (so far, only one return is allowed at the end of the function)
		
		// Replace parameter reference with temporary variable
	}
	
}