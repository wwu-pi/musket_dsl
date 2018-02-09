package de.wwu.musket.validation

import de.wwu.musket.musket.Model
import de.wwu.musket.musket.MusketObject
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.Parameter
import org.eclipse.xtext.validation.Check
import de.wwu.musket.musket.Variable
import org.eclipse.emf.ecore.EObject
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.ReferableObject
import de.wwu.musket.musket.MainBlock

class MusketModelValidator extends AbstractMusketValidator {
	
	public static val INVALID_ID = 'invalidIdentifier'
	
	// Check variable/constant names are unique
	@Check
	def checkMusketObjectNamesUnique(MusketObject obj) {
		if(obj.eContainer.eContents.filter(MusketObject).exists[it !== obj && it.name == obj.name]) {
			error('Duplicate name ' + obj.name + '!', 
				MusketPackage.eINSTANCE.referableObject_Name,
				INVALID_ID)
		}
	}
	
	// Check function parameter names are unique
	@Check
	def checkFunctionParameterNamesUnique(Parameter param) {
		if(param.eContainer.eContents.filter(Parameter).exists[it !== param && it.name == param.name]) {
			error('Duplicate name ' + param.name + '!', 
				MusketPackage.eINSTANCE.referableObject_Name,
				INVALID_ID)
		}
	}
	
	// Check if function parameter names overwrite global object names
	@Check
	def checkFunctionParameterNamesOverwriteGlobals(Parameter param) {
		if((param.eResource.allContents.filter(Model).next as Model).data.exists[it !== param && it.name == param.name]) {
			warning('Parameter ' + param.name + ' overwrites global object with the same name!', 
				MusketPackage.eINSTANCE.referableObject_Name,
				INVALID_ID)
		}
	}
	
	@Check
	def checkVariableNamesOverwriteGlobals(Variable variable) {
		// Check if variable name overwrites global object names
		if((variable.eResource.allContents.filter(Model).next as Model).data.exists[it !== variable && it.name == variable.name]) {
			warning('Parameter ' + variable.name + ' overwrites global object with the same name!', 
				MusketPackage.eINSTANCE.referableObject_Name,
				INVALID_ID)
		}
	}
	
	@Check
	def checkVariableNamesOverwriteFunctionObjects(ReferableObject variable) {	
		// Move to top level of nested statements to get function
		var EObject obj = variable
		do {
			obj = obj.eContainer
		} while(!(obj instanceof Function) && obj.eContainer !== null)
		
		// Only continue if we are in function scope
		if(!(obj instanceof Function)) return;
		
		// Check if variable name overwrites other local name
		if((obj as Function).eAllContents.filter(ReferableObject).exists[it !== variable && it.name == variable.name]) {
			error('Duplicate declaration of ' + variable.name + '!', 
				MusketPackage.eINSTANCE.referableObject_Name,
				INVALID_ID)
		}
	}
	
	@Check
	def checkVariableNamesOverwriteMainBlockObjects(ReferableObject variable) {	
		// Move to top level of nested statements to get function
		var EObject obj = variable
		do {
			obj = obj.eContainer
		} while(!(obj instanceof MainBlock) && obj.eContainer !== null)
		
		// Only continue if we are in function scope
		if(!(obj instanceof MainBlock)) return;
		
		// Check if variable name overwrites other local name
		if((obj as MainBlock).eAllContents.filter(ReferableObject).exists[it !== variable && it.name == variable.name]) {
			error('Duplicate declaration of ' + variable.name + '!', 
				MusketPackage.eINSTANCE.referableObject_Name,
				INVALID_ID)
		}
	}
}