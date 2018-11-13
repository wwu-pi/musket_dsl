package de.wwu.musket.validation

import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.FoldLocalSkeleton
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FoldSkeletonVariants
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.LambdaFunction
import de.wwu.musket.musket.MapFoldSkeleton
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.MapLocalIndexSkeleton
import de.wwu.musket.musket.MapOption
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MapSkeletonVariants
import de.wwu.musket.musket.Modulo
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketFunctionName
import de.wwu.musket.musket.MusketIteratorForLoop
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.Ref
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.ScatterSkeleton
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.ZipInPlaceSkeleton
import de.wwu.musket.musket.ZipIndexInPlaceSkeleton
import de.wwu.musket.musket.ZipIndexSkeleton
import de.wwu.musket.musket.ZipLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.ZipLocalIndexSkeleton
import de.wwu.musket.musket.ZipOption
import de.wwu.musket.musket.ZipSkeleton
import de.wwu.musket.musket.ZipSkeletonVariants
import de.wwu.musket.util.MusketType
import org.eclipse.emf.ecore.EObject
import org.eclipse.xtext.validation.Check

import static extension de.wwu.musket.util.TypeHelper.*

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
	static val foldParamsOut = 2
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
		val isFunctionCall = skel.param instanceof InternalFunctionCall
		val callFunction = if (isFunctionCall){
			(skel.param as InternalFunctionCall).value
		} else {
			skel.param as LambdaFunction
		}
		
		// check type of objects on which the skeleton is called: Array has 1, Matrix 2 index parameters
		val indexParams = if ((skel.eContainer as SkeletonExpression).obj.calculateType.isArray) 1 else 2
		
		switch skel {
			MapSkeleton case !skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
			MapInPlaceSkeleton: 
				if(callFunction.params.size < mapParamsOut){
					// Check minimum amount of parameters in target function
					error('Referenced function requires at least ' + mapParamsOut + ' parameters, ' + callFunction.params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				} else if(isFunctionCall && (skel.param as InternalFunctionCall).params.size !== callFunction.params.size-mapParamsOut){
					// Check provided argument count matches target function parameter count
					error('Skeleton function call requires ' + (callFunction.params.size-mapParamsOut) + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
				
			MapSkeleton case skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
			MapIndexSkeleton,
			MapLocalIndexSkeleton,
			MapIndexInPlaceSkeleton,
			MapLocalIndexInPlaceSkeleton:
				if(callFunction.params.size < indexParams+mapParamsOut){
					// Check minimum amount of parameters in target function
					error('Referenced function requires at least ' + (indexParams+mapParamsOut) + ' parameters, ' + callFunction.params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				} else if(isFunctionCall && (skel.param as InternalFunctionCall).params.size !== callFunction.params.size-indexParams-mapParamsOut){
					// Check provided argument count matches target function parameter count
					error('Skeleton function call requires ' + (callFunction.params.size-indexParams-mapParamsOut) + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
			
			ZipSkeleton case !skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
			ZipInPlaceSkeleton: {
					if(isFunctionCall && (skel.param as InternalFunctionCall).params.size < zipParamsMin){
						error('Skeleton function call requires at least ' + zipParamsMin + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					if(callFunction.params.size < zipParamsOut){
						// Check minimum amount of parameters in target function
						error('Referenced function requires at least ' + (indexParams+zipParamsOut) + ' parameters, ' + callFunction.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					} else if(isFunctionCall && (skel.param as InternalFunctionCall).params.size !== callFunction.params.size-zipParamsOut+zipParamsMin){
						// Check provided argument count matches target function parameter count
						error('Skeleton function call requires ' + (callFunction.params.size-zipParamsOut+zipParamsMin) + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				}
			ZipSkeleton case skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
			ZipIndexSkeleton,
			ZipLocalIndexSkeleton,
			ZipIndexInPlaceSkeleton,
			ZipLocalIndexInPlaceSkeleton: {
					if(isFunctionCall && (skel.param as InternalFunctionCall).params.size < zipParamsMin){
						// Check minimum amount of arguments in function call
						error('Skeleton function call requires at least ' + zipParamsMin + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					if(callFunction.params.size < indexParams+zipParamsOut){
						// Check minimum amount of parameters in target function
						error('Referenced function requires at least ' + (indexParams+zipParamsOut) + ' parameters, ' + callFunction.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					} else if(isFunctionCall && (skel.param as InternalFunctionCall).params.size !== callFunction.params.size-indexParams-zipParamsOut+zipParamsMin){
						// Check provided argument count matches target function parameter count
						error('Skeleton function call requires ' + (callFunction.params.size-indexParams-zipParamsOut+zipParamsMin) + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				}
				
			FoldSkeleton,
			FoldLocalSkeleton: 
				if(callFunction.params.size < foldParamsOut){
					// Check minimum amount of parameters in target function
					error('Referenced function requires at least ' + foldParamsOut + ' parameters, ' + callFunction.params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				} else if(isFunctionCall && (skel.param as InternalFunctionCall).params.size !== callFunction.params.size-foldParamsOut){
					// Check provided argument count matches target function parameter count
					error('Skeleton function call requires ' + (callFunction.params.size-foldParamsOut) + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
			
			MapFoldSkeleton: 
				// First parameter used for map -> less required for main fold
				if(callFunction.params.size < indexParams+foldParamsOut-1){
					// Check minimum amount of parameters in target function
					error('Referenced function requires at least ' + (indexParams+foldParamsOut-1) + ' parameters, ' + callFunction.params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				} else if(isFunctionCall && (skel.param as InternalFunctionCall).params.size-1 !== callFunction.params.size-indexParams-foldParamsOut){
					// Check provided argument count matches target function parameter count
					error('Skeleton function call requires ' + (callFunction.params.size-indexParams-foldParamsOut) + ' arguments, ' + ((skel.param as InternalFunctionCall).params.size-1) + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
				
			GatherSkeleton,
			ScatterSkeleton:
				// gather/scatter has exactly zero arguments
				if(isFunctionCall && (skel.param as InternalFunctionCall).params.size !== 0){ 
					error('Skeleton has no arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				} else if (!isFunctionCall && (skel.param as LambdaFunction).params.size !== 0){ 
					error('Skeleton has no arguments, ' + (skel.param as LambdaFunction).params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
				
			ShiftPartitionsHorizontallySkeleton,
			ShiftPartitionsVerticallySkeleton: 
				if(callFunction.params.size < shiftParamsOut){
					// Check minimum amount of parameters in target function
					error('Referenced function requires at least ' + shiftParamsOut + ' parameters, ' + callFunction.params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				} else if(isFunctionCall && (skel.param as InternalFunctionCall).params.size !== callFunction.params.size-shiftParamsOut){
					// Check provided argument count matches target function parameter count
					error('Skeleton function call requires ' + (callFunction.params.size-shiftParamsOut) + ' arguments, ' + (skel.param as InternalFunctionCall).params.size + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
		}
	}
	
	@Check
	def checkSkeletonFunctionImplicitParameterType(Skeleton skel) {
		val callFunction = if (skel.param instanceof InternalFunctionCall){
			(skel.param as InternalFunctionCall).value
		} else {
			skel.param as LambdaFunction
		}
		val callingType = (skel.eContainer as SkeletonExpression).obj.calculateCollectionType
		
		// Check skeleton type
		switch skel {
			MapSkeletonVariants: 
				if(callingType != callFunction.params.last?.calculateType){
					error('Calling type ' + callingType + ' does not match expected parameter type ' + callFunction.params.last?.calculateType + '!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
				
			ZipSkeletonVariants: {
					if(callingType != callFunction.params.last?.calculateType){
						error('Calling type ' + callingType + ' does not match expected parameter type ' + callFunction.params.last?.calculateType + '!',
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					// zipWith parameter needs to match second but last parameters for zip skeleton, e.g. ints.zip(doubles, f(...)) -> f(..., double, int)
					if(!skel.zipWith.calculateType?.collection){
						error('First argument needs to be a collection!',
							MusketPackage.eINSTANCE.zipSkeletonVariants_ZipWith,
							INVALID_PARAMS)
					} else if(skel.zipWith.value?.calculateCollectionType != callFunction.params.get(callFunction.params.size-2).calculateType){
						error('Argument type ' + skel.zipWith.value?.calculateCollectionType + ' does not match expected parameter type ' + callFunction.params.get(callFunction.params.size-2).calculateType + '!',
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				}
				
			FoldSkeletonVariants: {
					// user function: 	T func(..., T t, T t)
					// call:			Ts.func(t, func(...))
					
					// Referenced function parameters need to match calling type (or mapped type) for fold skeleton
					if(skel instanceof MapFoldSkeleton){
						if(!(skel.mapFunction.calculateType.equalsIgnoreDistribution(callFunction.params.last?.calculateType))){
							error('Mapped type ' + (skel as MapFoldSkeleton).mapFunction.calculateType + ' does not match expected parameter type ' + callFunction.params.last?.calculateType + '!', 
							MusketPackage.eINSTANCE.mapFoldSkeleton_MapFunction,
							INVALID_PARAMS)
						}
						if(skel.mapFunction instanceof LambdaFunction && ((skel.mapFunction as LambdaFunction).params.size != 1 || callingType != (skel.mapFunction as LambdaFunction).params.get(0).calculateType)){
							error('Calling type ' + callingType + ' does not match expected function input ' + (skel.mapFunction as LambdaFunction).params.get(0).calculateType + '!', 
							MusketPackage.eINSTANCE.mapFoldSkeleton_MapFunction,
							INVALID_PARAMS)
						}
					} else if(!(skel instanceof MapFoldSkeleton) && callingType != callFunction.params.last?.calculateType){
						// Regular folds
						error('Calling type ' + callingType + ' does not match expected parameter type ' + callFunction.params.last?.calculateType + '!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					// Check identity value parameter matches two last values
					if(callFunction.params.size >= 2 && !callFunction.params.get(callFunction.params.size-2)?.calculateType.equalsIgnoreDistribution(skel.identity.calculateType)){
						error('Identity value of type ' + skel.identity.calculateType + ' does not match expected parameter type ' + callFunction.params.get(callFunction.params.size-2)?.calculateType + '!', 
							MusketPackage.eINSTANCE.foldSkeletonVariants_Identity,
							INVALID_PARAMS)
					}
					if(callFunction.params.size >= 2 && !callFunction.params.get(callFunction.params.size-1)?.calculateType.equalsIgnoreDistribution(skel.identity.calculateType)){
						error('Identity value of type ' + skel.identity.calculateType + ' does not match expected parameter type ' + callFunction.params.get(callFunction.params.size-1)?.calculateType + '!', 
							MusketPackage.eINSTANCE.foldSkeletonVariants_Identity,
							INVALID_PARAMS)
					}
					// Fold function needs to return same type as its identity
					if(!callFunction.calculateType.equalsIgnoreDistribution(skel.identity.calculateType)){
						error('Return type ' + new MusketType(callFunction) + ' needs to match the identity type ' + skel.identity.calculateType + ' for fold skeletons!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					// Fold function needs to return same type as second but last value
					if(callFunction.params.size >= 2 && !callFunction.calculateType.equalsIgnoreDistribution(callFunction.params.get(callFunction.params.size-2)?.calculateType)){
						error('Return type ' + new MusketType(callFunction) + ' needs to match the second but last parameter type ' + callFunction.params.get(callFunction.params.size-2)?.calculateType + ' for fold skeletons!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				}
			
			ShiftPartitionsHorizontallySkeleton,
			ShiftPartitionsVerticallySkeleton: {
					// Shifting functions need exactly one int parameter
					if(callFunction.params.last?.calculateType != MusketType.INT){
						error('The function\'s last argument of type ' + callFunction.params.last?.calculateType + ' does not match the expected skeleton parameter type int!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					// Shifting functions return one int value
					if(callFunction.calculateType != MusketType.INT){
						error('Return type ' + new MusketType(callFunction) + ' must be int for this skeleton!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				}
		}
	}
	
	@Check
	def checkSkeletonFunctionIndexParameterType(Skeleton skel) {
		val callFunction = if (skel.param instanceof InternalFunctionCall){
			(skel.param as InternalFunctionCall).value
		} else {
			skel.param as LambdaFunction
		}		
		val isArray = (skel.eContainer as SkeletonExpression).obj.calculateType.isArray
			
		// Check skeleton type
		switch skel {
			// Those have 1 final parameter after index parameters
			MapSkeleton case skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
			MapIndexSkeleton,
			MapLocalIndexSkeleton,
			MapIndexInPlaceSkeleton,
			MapLocalIndexInPlaceSkeleton: 
				if(callFunction.params.size >= 2 && callFunction.params.get(callFunction.params.size - 2)?.calculateType != MusketType.INT &&
					(isArray || callFunction.params.get(callFunction.params.size - 3)?.calculateType != MusketType.INT)
				){
					error('Referenced function does not have correct amount or type of index parameters!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
				}
			
			// Those have 2 final parameters after index parameters
			ZipSkeleton case skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
			ZipIndexSkeleton,
			ZipIndexInPlaceSkeleton,
			ZipLocalIndexInPlaceSkeleton: 
				if(callFunction.params.size >= 3 && callFunction.params.get(callFunction.params.size - 3)?.calculateType != MusketType.INT &&
					(isArray || callFunction.params.get(callFunction.params.size - 4)?.calculateType != MusketType.INT)
				){
					error('Referenced function does not have correct amount or type of index parameters!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_PARAMS)
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
				FoldSkeleton,
				FoldLocalSkeleton,
				GatherSkeleton,
				ScatterSkeleton,
				ShiftPartitionsHorizontallySkeleton,
				ShiftPartitionsVerticallySkeleton:
					indexParams = 0
				default: indexParams = if ((skel.eContainer as SkeletonExpression).obj.calculateType.isArray) {
						1 
					} else if ((skel.eContainer as SkeletonExpression).obj.calculateType.isMatrix) {
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

	private def validateParamType(Iterable<Expression> input, Iterable<Parameter> target){
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
		if(!((obj as Function).calculateType.equalsIgnoreDistribution(stmt.value.calculateType))){
			error('Expression of type ' + stmt.value.calculateType + ' does not match specified return type ' + new MusketType(obj as Function) + '!', 
				MusketPackage.eINSTANCE.returnStatement_Value,
				INVALID_TYPE)
		}		
	}
	
	// Check function return type is present and correct 
	@Check
	def checkReturnTypeForInPlaceSkeletons(Skeleton skel) {
		val callFunction = if (skel.param instanceof InternalFunctionCall){
			(skel.param as InternalFunctionCall).value
		} else {
			skel.param as LambdaFunction
		}
			
		switch skel {
			MapSkeleton case skel.options.exists[it == MapOption.IN_PLACE],
			MapIndexInPlaceSkeleton,
			MapInPlaceSkeleton,
			ZipSkeleton case skel.options.exists[it == ZipOption.IN_PLACE],
			ZipInPlaceSkeleton:
				if(new MusketType(callFunction) != (skel.eContainer as SkeletonExpression).obj.calculateCollectionType){
					error('In place skeleton requires return type ' + (skel.eContainer as SkeletonExpression).obj.calculateCollectionType + ', ' + new MusketType(callFunction) + ' given!', 
						MusketPackage.eINSTANCE.skeleton_Param,
						INVALID_TYPE)
				} 
		}
	}
	
	@Check
	def checkReturnStatement(Function func) {
		if(func.returnType !== null && func.statement.filter(ReturnStatement).toList == emptyList){
			//  Return statement missing
			error('Function has no return statement!', 
				MusketPackage.eINSTANCE.function_ReturnType,
				INCOMPLETE_DECLARATION)
		} else if (func.statement.size > 0 && (func.statement.last instanceof ReturnStatement) && !func.calculateType.equalsIgnoreDistribution((func.statement.last as ReturnStatement).value.calculateType)){
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
	
	// Check collection access expression is int
	@Check
	def checkCollectionAccessIsNumeric(Ref ref) {
		if(ref.localCollectionIndex?.size > 0 && ref.localCollectionIndex.exists[it.calculateType != MusketType.INT]){
			error('Collection element expression must be int!', 
				MusketPackage.eINSTANCE.ref_LocalCollectionIndex,
				INVALID_TYPE)
		}
		if(ref.globalCollectionIndex?.size > 0 && ref.globalCollectionIndex.exists[it.calculateType != MusketType.INT]){
			error('Collection element expression must be int!', 
				MusketPackage.eINSTANCE.ref_GlobalCollectionIndex,
				INVALID_TYPE)
		}
	}
	
	// Check collection objects are instantiated with correct (primitive) types
	@Check
	def checkCollectionObjectInstantiationType(CollectionObject obj) {
		if(obj.values?.size == 0) return;
		
		obj.values.forEach[if(it.calculateType != obj.calculateCollectionType){
			error('Collection element type must be of type ' + obj.calculateCollectionType + '!', 
				MusketPackage.eINSTANCE.collectionObject_Values,
				obj.values.indexOf(it))
		}]
	}
	
	// Rand function in MusketFunctionCall must have two numeric parameters of equal type
	@Check
	def checkRandFunctionParameters(MusketFunctionCall call) {
		if(call.value !== MusketFunctionName.RAND) return;
		
		if(call.params?.size !== 2){
			error('Musket rand function expects 2 parameters, ' + call.params?.size + ' given!', 
				MusketPackage.eINSTANCE.musketFunctionCall_Params,
				INVALID_PARAMS)
		}
		
		call.params.forEach[ if(!it.calculateType.isNumeric) {
			error('Musket rand function parameters must have numeric type, ' + it.calculateType + ' given!', 
				MusketPackage.eINSTANCE.musketFunctionCall_Params,
				call.params.indexOf(it))
		}]
	}
	
	@Check
	def checkRandFunctionParameterType(MusketFunctionCall call) {
		if(call.value !== MusketFunctionName.RAND) return;
		
		call.params.forEach[ if(it.calculateType != call.params.head.calculateType) {
			error('Musket rand function parameters must have the same type!', 
				MusketPackage.eINSTANCE.musketFunctionCall_Params,
				INVALID_PARAMS)
		}]
	}
	
	// Scatter skeleton must assign from copy-distributed to distributed data structure
	@Check
	def checkScatterAssignmentSourceIsCopyDistributed(SkeletonExpression skel) {
		if(skel.skeleton instanceof ScatterSkeleton && skel.obj.calculateType.distributionMode != DistributionMode.COPY){
			error('A scatter operation can only be called on a copy distributed data structures!', 
				MusketPackage.eINSTANCE.skeletonExpression_Obj,
				INVALID_TYPE)
		}
	}
	
	@Check
	def checkScatterAssignmentIsDistributed(MusketAssignment assignment) {
		if(assignment.value instanceof SkeletonExpression && (assignment.value as SkeletonExpression).skeleton instanceof ScatterSkeleton &&
			assignment.^var.value instanceof CollectionObject && (assignment.^var.value as CollectionObject).type.distributionMode == DistributionMode.COPY
		){
			error('The result of a scatter operation can only be assigned to a distributed data structures!', 
				MusketPackage.eINSTANCE.musketAssignment_Var,
				INVALID_STATEMENT)
		}
	}
}
