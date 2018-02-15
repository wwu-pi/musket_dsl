package de.wwu.musket.validation

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.FoldIndexSkeleton
import de.wwu.musket.musket.FoldOption
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FoldSkeletonVariants
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.MapOption
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MapSkeletonVariants
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.MusketIteratorForLoop
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.ParameterInput
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.ZipInPlaceSkeleton
import de.wwu.musket.musket.ZipIndexSkeleton
import de.wwu.musket.musket.ZipOption
import de.wwu.musket.musket.ZipSkeleton
import de.wwu.musket.musket.ZipSkeletonVariants
import de.wwu.musket.util.MusketType
import java.util.Collection
import org.eclipse.emf.ecore.EObject
import org.eclipse.xtext.validation.Check

import static extension de.wwu.musket.util.CollectionHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.Modulo

class MusketTypeValidator extends AbstractMusketValidator {

	public static val INVALID_TYPE = 'invalidType'
	public static val INVALID_OPTIONS = 'invalidOptions'
	public static val INVALID_PARAMS = 'invalidParameters'
	public static val INCOMPLETE_DECLARATION = 'incompleteDeclaration'
	public static val INVALID_STATEMENT = 'invalidStatement'
	
	// Native parameters to skeletons
	static val zipParamsMin = 0
	static val mapParamsOut = 1
	static val zipParamsOut = 2
	static val foldParamsOut = 1
	static val shiftParamsOut = 1
	
	// Check skeleton options
	@Check
	def checkSkeletonOptions(MapSkeleton skel) {
		if(skel.options.exists[it == MapOption.INDEX] && skel.options.exists[it == MapOption.LOCAL_INDEX]) {
			error('Skeleton cannot contain both index and localIndex option!', 
				MusketPackage.eINSTANCE.mapSkeleton_Options,
				INVALID_OPTIONS)
		}
	}
	
	@Check
	def checkSkeletonOptions(ZipSkeleton skel) {
		if(skel.options.exists[it == ZipOption.INDEX] && skel.options.exists[it == ZipOption.LOCAL_INDEX]) {
			error('Skeleton cannot contain both index and localIndex option!', 
				MusketPackage.eINSTANCE.mapSkeleton_Options,
				INVALID_OPTIONS)
		}
	}
	
	// Check parameter match when calling skeletons with internal functions
	@Check
	def checkSkeletonFunctionParameterCount(Skeleton skel) {
		if (skel.param instanceof InternalFunctionCall){
			val call = skel.param as InternalFunctionCall
			// check type of objects on which the skeleton is called: Array has 1, Matrix 2 index parameters
			val indexParams = if ((skel.eContainer as SkeletonExpression).obj instanceof Array) 1 else 2
			
			switch skel {
				MapSkeleton case !skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
				MapInPlaceSkeleton: 
					if(call.value.params.size < mapParamsOut){
						// Check minimum amount of parameters in target function
						error('Referenced function requires at least ' + mapParamsOut + ' parameters, ' + call.value.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					} else if(call.params.size !== call.value.params.size-mapParamsOut){
						// Check provided argument count matches target function parameter count
						error('Skeleton function call requires ' + (call.value.params.size-mapParamsOut) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				MapSkeleton case skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
				MapIndexSkeleton,
				MapIndexInPlaceSkeleton,
				MapLocalIndexInPlaceSkeleton:
					if(call.value.params.size < indexParams+mapParamsOut){
						// Check minimum amount of parameters in target function
						error('Referenced function requires at least ' + (indexParams+mapParamsOut) + ' parameters, ' + call.value.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					} else if(call.params.size !== call.value.params.size-indexParams-mapParamsOut){
						// Check provided argument count matches target function parameter count
						error('Skeleton function call requires ' + (call.value.params.size-indexParams-mapParamsOut) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				
				ZipSkeleton case !skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
				ZipInPlaceSkeleton: {
						if(call.params.size < zipParamsMin){
							error('Skeleton function call requires at least ' + zipParamsMin + ' arguments, ' + call.params.size + ' given!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						if(call.value.params.size < zipParamsOut){
							// Check minimum amount of parameters in target function
							error('Referenced function requires at least ' + (indexParams+zipParamsOut) + ' parameters, ' + call.value.params.size + ' given!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						} else if(call.params.size !== call.value.params.size-zipParamsOut+zipParamsMin){
							// Check provided argument count matches target function parameter count
							error('Skeleton function call requires ' + (call.value.params.size-zipParamsOut+zipParamsMin) + ' arguments, ' + call.params.size + ' given!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
					}
				ZipSkeleton case skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
				ZipIndexSkeleton: {
						if(call.params.size < zipParamsMin){
							// Check minimum amount of arguments in function call
							error('Skeleton function call requires at least ' + zipParamsMin + ' arguments, ' + call.params.size + ' given!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						if(call.value.params.size < indexParams+zipParamsOut){
							// Check minimum amount of parameters in target function
							error('Referenced function requires at least ' + (indexParams+zipParamsOut) + ' parameters, ' + call.value.params.size + ' given!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						} else if(call.params.size !== call.value.params.size-indexParams-zipParamsOut+zipParamsMin){
							// Check provided argument count matches target function parameter count
							error('Skeleton function call requires ' + (call.value.params.size-indexParams-zipParamsOut+zipParamsMin) + ' arguments, ' + call.params.size + ' given!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
					}
					
				FoldSkeleton case !skel.options.exists[it == FoldOption.INDEX]: 
					if(call.value.params.size < foldParamsOut){
						// Check minimum amount of parameters in target function
						error('Referenced function requires at least ' + foldParamsOut + ' parameters, ' + call.value.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					} else if(call.params.size !== call.value.params.size-indexParams-foldParamsOut){
						// Check provided argument count matches target function parameter count
						error('Skeleton function call requires ' + (call.value.params.size-foldParamsOut) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				FoldSkeleton case skel.options.exists[it == FoldOption.INDEX],
				FoldIndexSkeleton: 
					if(call.value.params.size < indexParams+foldParamsOut){
						// Check minimum amount of parameters in target function
						error('Referenced function requires at least ' + (indexParams+foldParamsOut) + ' parameters, ' + call.value.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					} else if(call.params.size !== call.value.params.size-indexParams-foldParamsOut){
						// Check provided argument count matches target function parameter count
						error('Skeleton function call requires ' + (call.value.params.size-indexParams-foldParamsOut) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				
				GatherSkeleton:
					if(call.params.size !== 0){ 
						// gather has exactly zero arguments
						error('Skeleton has no arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				ShiftPartitionsHorizontallySkeleton,
				ShiftPartitionsVerticallySkeleton: 
					if(call.value.params.size < shiftParamsOut){
						// Check minimum amount of parameters in target function
						error('Referenced function requires at least ' + shiftParamsOut + ' parameters, ' + call.value.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					} else if(call.params.size !== call.value.params.size-shiftParamsOut){
						// Check provided argument count matches target function parameter count
						error('Skeleton function call requires ' + (call.value.params.size-shiftParamsOut) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
			}
		}
	}
	
	@Check
	def checkSkeletonFunctionImplicitParameterType(Skeleton skel) {
		if (skel.param instanceof InternalFunctionCall){
			val call = skel.param as InternalFunctionCall
			val callingType = (skel.eContainer as SkeletonExpression).obj.calculateCollectionType
			
			// Check skeleton type
			switch skel {
				MapSkeletonVariants: 
					if(callingType != call.value.params.last?.calculateType){
						error('Calling type ' + callingType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				ZipSkeletonVariants: {
						if(callingType != call.value.params.last?.calculateType){
							error('Calling type ' + callingType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!',
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						// zipWith parameter needs to match second but last parameters for zip skeleton, e.g. ints.zip(doubles, f(...)) -> f(..., double, int)
						if(!skel.zipWith.calculateType?.collection){
							error('First argument needs to be a collection!',
								MusketPackage.eINSTANCE.zipSkeletonVariants_ZipWith,
								INVALID_PARAMS)
						} else if(skel.zipWith.value?.calculateCollectionType != call.value.params.get(call.value.params.size-2).calculateType){
							error('Argument type ' + skel.zipWith.value?.calculateCollectionType + ' does not match expected parameter type ' + call.value.params.get(call.value.params.size-2).calculateType + '!',
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
					}
					
				FoldSkeletonVariants: {
						// Last two parameters need to match for fold skeleton
						if(callingType != call.value.params.last?.calculateType || callingType != call.value.params.get(call.value.params.size-2)?.calculateType){
							error('Calling type ' + callingType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						// Check identity value parameter matches
						if(call.value.params.last?.calculateType != skel.identity.calculateType){
							error('Identity value of type ' + skel.identity.calculateType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!', 
								MusketPackage.eINSTANCE.foldSkeletonVariants_Identity,
								INVALID_PARAMS)
						}
						// Fold function needs to return same type as its input
						if(call.value.calculateType != callingType){
							error('Return type ' + new MusketType(call.value) + ' needs to match the input type ' + callingType + 'for fold skeletons!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
					}
				
				ShiftPartitionsHorizontallySkeleton,
				ShiftPartitionsVerticallySkeleton: {
						// Shifting functions need exactly one int parameter
						if(call.value.params.last?.calculateType != MusketType.INT){
							error('The function\'s last argument of type ' + call.value.params.last?.calculateType + ' does not match the expected skeleton parameter type int!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						// Shifting functions return one int value
						if(call.value.calculateType != MusketType.INT){
							error('Return type ' + new MusketType(call.value) + ' must be int for this skeleton!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
					}
			}
		}
	}
	
	@Check
	def checkSkeletonFunctionIndexParameterType(Skeleton skel) {
		if (skel.param instanceof InternalFunctionCall){
			val call = skel.param as InternalFunctionCall
			val isArray = (skel.eContainer as SkeletonExpression).obj instanceof Array
			
			// Check skeleton type
			switch skel {
				// Those have 1 final parameter after index parameters
				MapSkeleton case skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
				MapIndexSkeleton,
				MapIndexInPlaceSkeleton,
				MapLocalIndexInPlaceSkeleton: 
					if(call.value.params.size >= 2 && call.value.params.get(call.value.params.size - 2)?.calculateType != MusketType.INT &&
						(isArray || call.value.params.get(call.value.params.size - 3)?.calculateType != MusketType.INT)
					){
						error('Referenced function does not have correct amount or type of index parameters!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				
				// Those have 2 final parameters after index parameters
				ZipSkeleton case skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
				ZipIndexSkeleton:
					if(call.value.params.size >= 3 && call.value.params.get(call.value.params.size - 3)?.calculateType != MusketType.INT &&
						(isArray || call.value.params.get(call.value.params.size - 4)?.calculateType != MusketType.INT)
					){
						error('Referenced function does not have correct amount or type of index parameters!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				// Those have index parameters as final parameters
				FoldSkeleton,
				FoldIndexSkeleton: 
					if(call.value.params.size >= 1 && call.value.params.get(call.value.params.size - 1)?.calculateType != MusketType.INT &&
						(isArray || call.value.params.get(call.value.params.size - 2)?.calculateType != MusketType.INT)
					){
						error('Referenced function does not have correct amount or type of index parameters!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
			}
		}
	}
	
	@Check
	def checkSkeletonFunctionCustomParameterType(Skeleton skel) {
		if (skel.param instanceof InternalFunctionCall){
			val call = skel.param as InternalFunctionCall
			// check type of objects on which the skeleton is called: Array has 1, Matrix 2 index parameters
			var indexParams = 0; 
			switch skel {
				MapSkeleton case !skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
				MapInPlaceSkeleton,
				ZipSkeleton case !skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
				ZipInPlaceSkeleton,
				FoldSkeleton case !skel.options.exists[it == FoldOption.INDEX],
				GatherSkeleton,
				ShiftPartitionsHorizontallySkeleton,
				ShiftPartitionsVerticallySkeleton:
					indexParams = 0
				default: indexParams = if ((skel.eContainer as SkeletonExpression).obj instanceof Array) {
						1 
					} else if ((skel.eContainer as SkeletonExpression).obj instanceof Matrix) {
						2
					} else { 
						0
					}
				}
				
			switch skel {
				MapSkeletonVariants: 
					if(call.value.params.size >= indexParams+mapParamsOut && call.value.params.size > mapParamsOut - indexParams && call.params.size > 0){
						validateParamType(call.params.take(call.params.size), call.value.params.take(call.value.params.size))
					}
					
				ZipSkeletonVariants:
					if(call.value.params.size > zipParamsOut - indexParams && call.params.size > 0){
						validateParamType(call.params.take(call.params.size), call.value.params.take(call.value.params.size))
					}
					
				FoldSkeletonVariants: 
					if(call.value.params.size > foldParamsOut - indexParams && call.params.size > 0){
						validateParamType(call.params.take(call.params.size), call.value.params.take(call.value.params.size))
					}
					
				ShiftPartitionsHorizontallySkeleton,
				ShiftPartitionsVerticallySkeleton: 
					if(call.value.params.size > shiftParamsOut - indexParams && call.params.size > 0){
						validateParamType(call.params.take(call.params.size), call.value.params.take(call.value.params.size))
					}
			}
		}
	}

	private def validateParamType(Iterable<ParameterInput> input, Iterable<Parameter> target){
		for(var i=0; i < input.size; i++){
			if(input.get(i).calculateType != target.get(i).calculateType){
				error('Parameter does not match expected type ' + target.get(i).calculateType + '!', 
					input.get(i).eContainer, input.get(i).eContainingFeature, i)
			}
		}
	} 
		
	// Check return type of functions is correct
	@Check
	def checkFunctionReturnType(ReturnStatement stmt) {
		// Move to top level of nested statements to get function
		var EObject obj = stmt
		do {
			obj = obj.eContainer
		} while(!(obj instanceof Function) && obj.eContainer !== null)
		
		// Check return type
		if((obj as Function).calculateType != stmt.value.calculateType){
			error('Expression of type ' + stmt.value.calculateType + ' does not match specified return type ' + new MusketType(obj as Function) + '!', 
				MusketPackage.eINSTANCE.returnStatement_Value,
				INVALID_TYPE)
		}
			
	}
	
	// Check function return type is present and correct 
	@Check
	def checkReturnTypeForInPlaceSkeletons(Skeleton skel) {
		if (skel.param instanceof InternalFunctionCall){
			val call = skel.param as InternalFunctionCall
			
			switch skel {
				MapSkeleton case skel.options.exists[it == MapOption.IN_PLACE],
				MapIndexInPlaceSkeleton,
				MapInPlaceSkeleton,
				ZipSkeleton case skel.options.exists[it == ZipOption.IN_PLACE],
				ZipInPlaceSkeleton:
					if(new MusketType(call.value) != (skel.eContainer as SkeletonExpression).obj.calculateCollectionType){
						error('In place skeleton requires return type ' + (skel.eContainer as SkeletonExpression).obj.structType + ', ' + new MusketType(call.value) + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_TYPE)
					} 
			}
		}
	}
	
	@Check
	def checkReturnStatement(Function func) {
		if(func.statement.size > 0 && func.statement.filter(ReturnStatement).toList == emptyList){
			//  Return statement missing
			error('Function has no return statement!', 
				MusketPackage.eINSTANCE.function_Statement,
				func.statement.size-1,
				INCOMPLETE_DECLARATION)
		} else if (func.statement.size > 0 && (func.statement.last instanceof ReturnStatement) && func.calculateType != (func.statement.last as ReturnStatement).value.calculateType){
			error('Return type ' + (func.statement.last as ReturnStatement).value.calculateType + ' does not match specified type ' + new MusketType(func) + '!', 
				MusketPackage.eINSTANCE.function_Statement,
				func.statement.size-1,
				INVALID_TYPE)
		}
	}
	
	// Check IteratorForLoop parameter type matches
	@Check
	def checkIteratorForLoopParameterType(IteratorForLoop loop) {
		if(loop.iter.calculateType != loop.dataStructure.calculateCollectionType){
			error('Iterator element type ' + loop.iter.calculateType + ' does not match collection type ' + loop.dataStructure.calculateCollectionType + '!', 
				MusketPackage.eINSTANCE.iteratorForLoop_Iter,
				INVALID_TYPE)
		}
	}
	
	@Check
	def checkMusketIteratorForLoopParameterType(MusketIteratorForLoop loop) {
		if(loop.iter.calculateType != loop.dataStructure.calculateCollectionType){
			error('Iterator element type ' + loop.iter.calculateType + ' does not match collection type ' + loop.dataStructure.calculateCollectionType + '!', 
				MusketPackage.eINSTANCE.iteratorForLoop_Iter,
				INVALID_TYPE)
		}
	}
	
	// Check assignment type
	@Check
	def checkAssignmentType(Assignment assign) {
		if(assign.value?.calculateType != assign.^var?.calculateType){
			error('Expression of type ' + assign.value.calculateType + ' cannot be assigned to variable of type ' + assign.^var.calculateType + '!', 
				MusketPackage.eINSTANCE.assignment_Value,
				INVALID_TYPE)
		}
	}
	
	// Check modulo Operator only works on ints
	@Check
	def checkModuloOperator(Modulo modulo) {
		if(modulo.left.calculateType != MusketType.INT || modulo.right.calculateType != MusketType.INT){
			error('Modulo operator requires two int values, ' + modulo.left.calculateType + ' and ' + modulo.right.calculateType + ' given!', 
				modulo, null, null)
		}
	}
}
