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

/**
 * This class contains custom validation rules. 
 *
 * See https://www.eclipse.org/Xtext/documentation/303_runtime_concepts.html#validation
 */
class MusketTypeValidator extends AbstractMusketValidator {

	public static val INVALID_TYPE = 'invalidType'
	public static val INVALID_OPTIONS = 'invalidOptions'

	// Check skeleton options
	@Check
	def checkSkeletonOptions(MapSkeleton skel) {
		if(skel.options.exists[it == MapOption.INDEX] && skel.options.exists[it == MapOption.LOCAL_INDEX]) {
			error('Skeleton cannot contain both index and localIndex option', 
				MusketPackage.eINSTANCE.mapSkeleton_Options,
				INVALID_OPTIONS)
		}
	}
	
	@Check
	def checkSkeletonOptions(ZipSkeleton skel) {
		if(skel.options.exists[it == ZipOption.INDEX] && skel.options.exists[it == ZipOption.LOCAL_INDEX]) {
			error('Skeleton cannot contain both index and localIndex option', 
				MusketPackage.eINSTANCE.mapSkeleton_Options,
				INVALID_OPTIONS)
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
			error('Expression of type ' + stmt.value.calculateType + ' does not match specified return type ' + (obj as Function).returnType, 
				MusketPackage.eINSTANCE.returnStatement_Value,
				INVALID_TYPE)
		}
			
	}
	
	// Check function parameter types are correct in call
	
	// Check function return type is correct in call 
	
}
