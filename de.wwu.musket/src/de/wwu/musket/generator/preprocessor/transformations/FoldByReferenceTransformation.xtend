package de.wwu.musket.generator.preprocessor.transformations

import static extension de.wwu.musket.generator.preprocessor.util.PreprocessorUtil.*
import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import de.wwu.musket.musket.FoldSkeletonVariants
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.LambdaFunction
import de.wwu.musket.musket.PrimitiveTypeLiteral
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.SkeletonExpression
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.emf.ecore.util.EcoreUtil
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.MapFoldSkeleton

/**
 * Dependencies:
 * - none
 */
 class FoldByReferenceTransformation extends PreprocessorTransformation {
	
	new(MusketComplexElementFactory factory) {
		super(factory)
	}
	
	/**
	 * Replace return statement of fold user functions to allow for fold function generation by reference
	 * (i.e., reusing the identity value for intermediate calculations instead of copying values). 
	 * Example:
	 *  
	 * int sum(int a, int b){     -->	void sum(int& a, int b){ 
	 * 		int result = a + b;				int result = a + b;
	 * 		return result;					a = result;
	 * }								}
	 */
	override run(Resource input) {
		// Get all fold skeleton calls
		val folds = input.allContents.filter(SkeletonExpression).filter[it.skeleton instanceof FoldSkeletonVariants].toList
		
		folds.forEach[
			val fold = it
			
			val skeletonParam = fold.skeleton.param
			val Function userFunction = switch (skeletonParam){
				InternalFunctionCall: skeletonParam.value
				LambdaFunction: skeletonParam
			}
			
			// Check if other non-fold skeletons also call this user function
			if(input.allContents.filter(InternalFunctionCall).filter[it !== skeletonParam].filter[it.value === userFunction]
				.map[it.eContainerOfType(Skeleton) as Skeleton].filter[!(it instanceof FoldSkeletonVariants) || // either fold skeletons are ok
					(it instanceof MapFoldSkeleton && ((it as MapFoldSkeleton).mapFunction instanceof InternalFunctionCall) // mapFold must not use this as map function
						&& ((it as MapFoldSkeleton).mapFunction as InternalFunctionCall).value === userFunction
					)
				].size == 0){
				
				val returnStatement = userFunction.statement.last
				if(returnStatement instanceof ReturnStatement){
					// Set function return type to void and use parameter by reference
					userFunction.returnType = factory.createPrimitiveType(PrimitiveTypeLiteral.AUTO)
					userFunction.params.head.reference = true
					
					// Build assignment to write return value to first user function parameter 
					val target = factory.createObjectRef
					target.value = userFunction.params.head
					
					val assignment = factory.createAssignment
					assignment.^var = target
					assignment.value = returnStatement.value
					assignment.operator = '='
					
					EcoreUtil.replace(returnStatement, assignment)
				}
			}
		]
	}
	
}