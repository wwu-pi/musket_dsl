package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import org.eclipse.emf.ecore.resource.Resource

/**
 * Dependencies:
 * - should be run after MapFusion
 */
class SkeletonFusionTransformation extends PreprocessorTransformation {
	
	new(MusketComplexElementFactory factory) {
		super(factory)
	}
	
	/**
	 * Combine map and fold functions on the same data in order to avoid temporary 
	 * data structures. 
	 */
	override run(Resource input) {
		// TODO
		// If map is followed by fold on same data structure (which not further used otherwise)
		// 
	}
	
}