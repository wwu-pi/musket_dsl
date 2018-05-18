package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.FoldSkeletonVariants
import de.wwu.musket.musket.MapSkeletonVariants
import static extension de.wwu.musket.generator.preprocessor.util.PreprocessorUtil.*
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.MapLocalIndexSkeleton
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.MainFunctionStatement
import de.wwu.musket.musket.MainBlock

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
		// Check all defined data structures
//		val allDataStructures = input.allContents.filter(CollectionObject)
//		val allSkeletonExpressions = input.allContents.filter(SkeletonExpression).toList
//		
//		allDataStructures.forEach[
//			val currentDataStructure = it
//			// Get all skeleton calls
//			val callsOnDataStructure = allSkeletonExpressions.filter[it.obj === currentDataStructure]
//			
//			// If map is followed by fold on same data structure
//			var SkeletonExpression lastExpression = null 
//			for(expr : callsOnDataStructure){
//				if(lastExpression.skeleton instanceof MapSkeletonVariants &&
//					expr.skeleton instanceof FoldSkeletonVariants){
//						
//					if(expr.eContainer instanceof MusketAssignment) {
//					// Ensure data structure is not further used
//					val next = findNextUsage(expr.eContainer as MusketAssignment)
//					val nexts = findAllSubsequentUsages(expr.eContainer as MusketAssignment)
//					}
//					lastExpression = null
//				} else {
//					lastExpression = expr
//				}
//			}
//		]
		val maps = input.allContents.filter(SkeletonExpression).filter[it.skeleton instanceof MapSkeleton ||
			it.skeleton instanceof MapIndexSkeleton || it.skeleton instanceof MapLocalIndexSkeleton]
		
		maps.forEach[
			val map = it
			val x = input.allContents.filter(MainBlock).head
			val y = x.eAllContents.dropWhile[it !== map].drop(1).toList
			val next = y.filter(ObjectRef).filter[it.value === map.obj].toList
			
			//val next = findNextUsage(it.eContainer as MusketAssignment)
			println(next)
		]
	}
	
}