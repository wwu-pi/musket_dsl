package de.wwu.musket.generator.preprocessor.transformations

import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.MapInPlaceSkeleton
import org.eclipse.emf.ecore.resource.Resource

import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.PrimitiveTypeLiteral
import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.IndividualParameter
import de.wwu.musket.musket.StructType

class MapVariantTransformation extends PreprocessorTransformation {
	
	new(MusketComplexElementFactory factory) {
		super(factory)
	}
	
	override run(Resource input) {
		
		transformMapInPlace(input)
		
		transformMapWithSideEffects(input)
	}
	
	def transformMapWithSideEffects(Resource resource) {
		val maps = resource.allContents.filter(MapSkeleton)
		
		maps.forEach[
			if(it.param instanceof InternalFunctionCall){
				val functionCall = it.param as InternalFunctionCall
				val targetFunction = functionCall.value
								
				// Check if other skeletons also call this user function
				if(resource.allContents.filter(InternalFunctionCall).filter[it !== functionCall].map[it.value].filter[it === functionCall.value].size == 0){
					// Dealing with struct?
					if(targetFunction.params.last.calculateType.isStruct){
						val struct = targetFunction.params.last as IndividualParameter 
						
						// Add copy constructor to statements
						val newCopy = factory.createStructVariable
						newCopy.type = (struct.type as StructType).type
						newCopy.name = "_" + struct.name
						val newRef = factory.createObjectRef
						newRef.value = struct
						newCopy.copyFrom = newRef
						
						targetFunction.statement.add(0, newCopy)
						
						// Replace all old references to parameter
						targetFunction.eAllContents.filter[it !== newRef].filter(ObjectRef).forEach[
							it.value = newCopy
						]
					}
				}
				// TODO else duplicate function
			}
		]
	}
	
	/**
	 * Replace a return statement within a map user function that deals with modifying a struct, e.g.
	 * SomeStruct square(SomeStruct s){     -->		void square(SomeStruct s){ 
	 * 		s.val = s.val * s.val;						s.val = s.val * s.val;
	 * 		return s;								}
	 * }
	 */
	def transformMapInPlace(Resource resource) {
		val maps = resource.allContents.filter(MapInPlaceSkeleton)
		
		maps.forEach[
			if(it.param instanceof InternalFunctionCall){
				val functionCall = it.param as InternalFunctionCall
				val targetFunction = functionCall.value
								
				// Check if other skeletons also call this user function
				if(resource.allContents.filter(InternalFunctionCall).filter[it !== functionCall].map[it.value].filter[it === functionCall.value].size == 0){
					// Dealing with struct?
					if(targetFunction.params.last.calculateType.isStruct){
						// Remove return statement and return type if only main object is returned
						if(targetFunction.statement.last instanceof ReturnStatement 
							&& (targetFunction.statement.last as ReturnStatement).value instanceof CompareExpression
							&& ((targetFunction.statement.last as ReturnStatement).value as CompareExpression).eqLeft instanceof ObjectRef
							&& (((targetFunction.statement.last as ReturnStatement).value as CompareExpression).eqLeft as ObjectRef).value === targetFunction.params.last){
							
							targetFunction.statement.remove(targetFunction.statement.size - 1)
							targetFunction.returnType = factory.createPrimitiveType(PrimitiveTypeLiteral.AUTO)
						}
					}
				}
				// TODO else duplicate function
			}
		]
	}
}