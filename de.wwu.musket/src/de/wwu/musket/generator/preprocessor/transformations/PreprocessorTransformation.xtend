package de.wwu.musket.generator.preprocessor.transformations

import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory

abstract class PreprocessorTransformation {
	
	protected MusketComplexElementFactory factory;
	
	new(MusketComplexElementFactory factory){
		this.factory = factory
	}
	
	abstract def void run(Resource input);
}