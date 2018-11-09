package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.MainBlock
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.SkeletonExpression
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.emf.ecore.util.EcoreUtil

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
		// Get all map skeleton calls
		val maps = input.allContents.filter(SkeletonExpression).filter[it.skeleton instanceof MapSkeleton].toList
		
		maps.forEach[
			val map = it
			val mainBlock = input.allContents.filter(MainBlock).head
			
			val nextSkelExpressions = mainBlock.eAllContents.filter(SkeletonExpression).dropWhile[it !== map].filter[it.obj === (map.eContainer as MusketAssignment).^var.value].toList
			val nextRefs = mainBlock.eAllContents.dropWhile[it !== map].drop(1).filter(ObjectRef).filter[it.value === (map.eContainer as MusketAssignment).^var.value].toList
			
			// If current map is followed by fold skeleton on same data structure
			val nextSkel = nextSkelExpressions.head?.skeleton
			if(nextSkel instanceof FoldSkeleton){
				
				// Ensure data structure is not further used in skeletons AND not further referenced
				if(nextSkelExpressions.size == 1 && nextRefs.size == 0){
					val mapFold = factory.createMapFoldSkeleton(map.skeleton as MapSkeleton, nextSkel)
					
					// Replace map + fold and insert combined statement 
					EcoreUtil.remove(map)
					EcoreUtil.replace(nextSkel, mapFold)
				}
			}
		]
	}
	
}