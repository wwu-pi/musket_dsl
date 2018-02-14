package de.wwu.musket.validation

import de.wwu.musket.musket.Function
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.ReturnStatement
import org.eclipse.emf.ecore.EObject
import org.eclipse.xtext.validation.Check

import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.MapOption
import de.wwu.musket.musket.ZipSkeleton
import de.wwu.musket.musket.ZipOption
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.Array
import de.wwu.musket.musket.ZipInPlaceSkeleton
import de.wwu.musket.musket.ZipIndexSkeleton
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FoldOption
import de.wwu.musket.musket.FoldIndexSkeleton
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.RotatePartitionsVerticallySkeleton
import de.wwu.musket.musket.RotatePartitionsHorizontallySkeleton
import de.wwu.musket.musket.Type
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.MusketIteratorForLoop

class MusketTypeValidator extends AbstractMusketValidator {

	public static val INVALID_TYPE = 'invalidType'
	public static val INVALID_OPTIONS = 'invalidOptions'
	public static val INVALID_PARAMS = 'invalidParameters'
	
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
			
			// Native parameters to skeletons
			val zipParamsMin = 1
			val mapParamsOut = 1
			val zipParamsOut = 2
			val foldParamsOut = 1
			val shiftParamsOut = 1
			
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
					
				RotatePartitionsHorizontallySkeleton,
				RotatePartitionsVerticallySkeleton: 
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
				MapSkeleton,
				MapInPlaceSkeleton,
				MapIndexSkeleton,
				MapIndexInPlaceSkeleton,
				MapLocalIndexInPlaceSkeleton: 
					if(callingType !== call.value.params.last?.calculateType){
						error('Calling type ' + callingType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				ZipSkeleton,
				ZipInPlaceSkeleton,
				ZipIndexSkeleton: {
						if(callingType !== call.value.params.last?.calculateType){
							error('Calling type ' + callingType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!',
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						// Last given argument needs to match second but last parameters for zip skeleton, e.g. ints.zip(f(doubles)) -> f(double, int)
						if(!call.params.last?.calculateType?.collection){
							error('Last argument needs to be a collection!',
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						} else if(call.params.last?.calculateCollectionType !== call.value.params.get(call.value.params.size-2).calculateType){
							error('Argument type ' + call.params.last?.calculateCollectionType + ' does not match expected parameter type ' + call.value.params.get(call.value.params.size-2).calculateType + '!',
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
					}
					
				FoldSkeleton,
				FoldIndexSkeleton: {
						// Last two parameters need to match for fold skeleton
						if(callingType !== call.value.params.last?.calculateType || callingType !== call.value.params.get(call.value.params.size-2)?.calculateType){
							error('Calling type ' + callingType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						// Check identity value parameter matches
						if((skel instanceof FoldSkeleton) && call.value.params.last?.calculateType !== (skel as FoldSkeleton).identity.calculateType){
							error('Identity value of type ' + (skel as FoldSkeleton).identity.calculateType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!', 
								MusketPackage.eINSTANCE.foldSkeleton_Identity,
								INVALID_PARAMS)
						} else if((skel instanceof FoldIndexSkeleton) && call.value.params.last?.calculateType !== (skel as FoldIndexSkeleton).identity.calculateType){
							error('Identity value of type ' + (skel as FoldIndexSkeleton).identity.calculateType + ' does not match expected parameter type ' + call.value.params.last?.calculateType + '!', 
								MusketPackage.eINSTANCE.foldIndexSkeleton_Identity,
								INVALID_PARAMS)
						}
						// Fold function needs to return same type as its input
						if(call.value.calculateType !== callingType){
							error('Return type ' + call.value.getReadableType + ' needs to match the input type ' + callingType + 'for fold skeletons!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
					}
				
				RotatePartitionsHorizontallySkeleton,
				RotatePartitionsVerticallySkeleton: {
						// Shifting functions need exactly one int parameter
						if(call.value.params.last?.calculateType !== Type.INT){
							error('The function\'s last argument of type ' + call.value.params.last?.calculateType + ' does not match the expected skeleton parameter type int!', 
								MusketPackage.eINSTANCE.skeleton_Param,
								INVALID_PARAMS)
						}
						// Shifting functions return one int value
						if(call.value.calculateType !== Type.INT){
							error('Return type ' + call.value.getReadableType + ' must be int for this skeleton!', 
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
					if(call.value.params.size >= 2 && call.value.params.get(call.value.params.size - 2)?.calculateType !== Type.INT &&
						(isArray || call.value.params.get(call.value.params.size - 3)?.calculateType !== Type.INT)
					){
						error('Referenced function does not have correct amount or type of index parameters!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				
				// Those have 2 final parameters after index parameters
				ZipSkeleton case skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
				ZipIndexSkeleton:
					if(call.value.params.size >= 3 && call.value.params.get(call.value.params.size - 3)?.calculateType !== Type.INT &&
						(isArray || call.value.params.get(call.value.params.size - 4)?.calculateType !== Type.INT)
					){
						error('Referenced function does not have correct amount or type of index parameters!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				// Those have index parameters as final parameters
				FoldSkeleton,
				FoldIndexSkeleton: 
					if(call.value.params.size >= 1 && call.value.params.get(call.value.params.size - 1)?.calculateType !== Type.INT &&
						(isArray || call.value.params.get(call.value.params.size - 2)?.calculateType !== Type.INT)
					){
						error('Referenced function does not have correct amount or type of index parameters!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
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
		if((obj as Function).calculateType !== stmt.value.calculateType){
			error('Expression of type ' + stmt.value.calculateType + ' does not match specified return type ' + (obj as Function).getReadableType + '!', 
				MusketPackage.eINSTANCE.returnStatement_Value,
				INVALID_TYPE)
		}
			
	}
	
	// Check function parameter types are correct in call
	
	// Check function return type is correct in call 
	
	// Check IteratorForLoop parameter type matches
	@Check
	def checkIteratorForLoopParameterType(IteratorForLoop loop) {
		if(loop.iter.calculateType !== loop.dataStructure.calculateCollectionType){
			error('Iterator element type  ' + loop.iter.calculateType + ' does not match collection type ' + loop.dataStructure.calculateCollectionType + '!', 
				MusketPackage.eINSTANCE.iteratorForLoop_Iter,
				INVALID_TYPE)
		}
	}
	
	@Check
	def checkMusketIteratorForLoopParameterType(MusketIteratorForLoop loop) {
		if(loop.iter.calculateType !== loop.dataStructure.calculateCollectionType){
			error('Iterator element type  ' + loop.iter.calculateType + ' does not match collection type ' + loop.dataStructure.calculateCollectionType + '!', 
				MusketPackage.eINSTANCE.iteratorForLoop_Iter,
				INVALID_TYPE)
		}
	}
}
