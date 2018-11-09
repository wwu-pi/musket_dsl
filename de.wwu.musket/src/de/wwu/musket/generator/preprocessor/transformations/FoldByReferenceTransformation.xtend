package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.LambdaFunction
import de.wwu.musket.musket.PrimitiveTypeLiteral
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.SkeletonExpression
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.emf.ecore.util.EcoreUtil

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
	 * int sum(int a, int b){     -->	void sum(int a, int b){ 
	 * 		int result = a + b;				int result = a + b;
	 * 		return result;					a = result;
	 * }								}
	 */
	override run(Resource input) {
		// Get all fold skeleton calls
		val maps = input.allContents.filter(SkeletonExpression).filter[it.skeleton instanceof FoldSkeleton].toList
		
		maps.forEach[
			val fold = it
			
			var Function userFunction;
			switch (fold.skeleton.param){
				InternalFunctionCall: userFunction = (fold.skeleton.param as InternalFunctionCall).value
				LambdaFunction: userFunction = (fold.skeleton.param as LambdaFunction)
			}
			
			val returnStatement = userFunction.statement.last
			if(returnStatement instanceof ReturnStatement){
				// Set function return type to void
				userFunction.returnType = factory.createPrimitiveType(PrimitiveTypeLiteral.AUTO)
				
				// Build assignment to write return value to first user function parameter 
				val target = factory.createObjectRef
				target.value = userFunction.params.head
				
				val assignment = factory.createAssignment
				assignment.^var = target
				assignment.value = returnStatement.value
				assignment.operator = '='
				
				EcoreUtil.replace(returnStatement, assignment)
			}
		]
	}
	
}