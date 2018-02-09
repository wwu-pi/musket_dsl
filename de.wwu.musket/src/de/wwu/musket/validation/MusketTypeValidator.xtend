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
			val mapParams = 1
			val zipParams = 2
			val foldParams = 1
			val shiftParams = 1
			
			switch skel {
				MapSkeleton case !skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
				MapInPlaceSkeleton: 
					if(call.params.size !== call.value.params.size-mapParams){
						error('Skeleton requires ' + (call.value.params.size-mapParams) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				MapSkeleton case skel.options.exists[it == MapOption.INDEX || it == MapOption.LOCAL_INDEX],
				MapIndexSkeleton,
				MapIndexInPlaceSkeleton,
				MapLocalIndexInPlaceSkeleton: 
					if(call.params.size !== call.value.params.size-indexParams-mapParams){
						error('Skeleton requires ' + (call.value.params.size-indexParams-mapParams) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				ZipSkeleton case !skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
				ZipInPlaceSkeleton: 
					if(call.params.size !== call.value.params.size-indexParams-zipParams){
						error('Skeleton requires ' + (call.value.params.size-zipParams) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				ZipSkeleton case skel.options.exists[it == ZipOption.INDEX || it == ZipOption.LOCAL_INDEX],
				ZipIndexSkeleton: 
					if(call.params.size !== call.value.params.size-indexParams-zipParams){
						error('Skeleton requires ' + (call.value.params.size-indexParams-zipParams) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				
				FoldSkeleton case !skel.options.exists[it == FoldOption.INDEX]: 
					if(call.params.size !== call.value.params.size-indexParams-foldParams){
						error('Skeleton requires ' + (call.value.params.size-foldParams) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				FoldSkeleton case skel.options.exists[it == FoldOption.INDEX],
				FoldIndexSkeleton: 
					if(call.params.size !== call.value.params.size-indexParams-foldParams){
						error('Skeleton requires ' + (call.value.params.size-indexParams-foldParams) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
				
				GatherSkeleton:
					if(call.params.size !== 0){ // gather has exactly zero arguments
						error('Skeleton requires no arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
					
				RotatePartitionsHorizontallySkeleton,
				RotatePartitionsVerticallySkeleton: 
					if(call.params.size !== call.value.params.size-shiftParams){
						error('Skeleton requires ' + (call.value.params.size-shiftParams) + ' arguments, ' + call.params.size + ' given!', 
							MusketPackage.eINSTANCE.skeleton_Param,
							INVALID_PARAMS)
					}
			}
		}
	}
	
	// Check correct types when calculating values 
	
	// Check return type of functions is correct
	@Check
	def checkFunctionReturnType(ReturnStatement stmt) {
		// Move to top level of nested statements to get function
		var EObject obj = stmt
		do {
			obj = obj.eContainer
		} while(!(obj instanceof Function) && obj.eContainer !== null)
		
		// Check return type
		if((obj as Function).returnType !== stmt.value.calculateType){
			error('Expression of type ' + stmt.value.calculateType + ' does not match specified return type ' + (obj as Function).returnType + '!', 
				MusketPackage.eINSTANCE.returnStatement_Value,
				INVALID_TYPE)
		}
			
	}
	
	// Check function parameter types are correct in call
	
	// Check function return type is correct in call 
	
}
