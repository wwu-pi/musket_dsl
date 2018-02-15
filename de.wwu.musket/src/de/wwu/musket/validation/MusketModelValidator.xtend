package de.wwu.musket.validation

import de.wwu.musket.musket.Model
import de.wwu.musket.musket.MusketObject
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.ReferableObject
import java.util.Collection
import org.eclipse.emf.ecore.EObject
import org.eclipse.xtext.validation.Check
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.Constant
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.CollectionFunctionName
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.ReturnStatement

class MusketModelValidator extends AbstractMusketValidator {
	
	public static val INVALID_ID = 'invalidIdentifier'
	public static val INVALID_OPERATION = 'invalidOperation'
	
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
		// Already checked in checkVariableNamesOverwritePrevious() and marked as error
//		if((param.eResource.allContents.filter(Model).next as Model).data.exists[it !== param && it.name == param.name]) {
//			warning('Parameter ' + param.name + ' overwrites global object with the same name!', 
//				MusketPackage.eINSTANCE.referableObject_Name,
//				INVALID_ID)
//		}
	}
	
	// Check if variable name overwrites other name defined in the scope
	@Check
	def checkVariableNamesOverwritePrevious(ReferableObject variable) {	
		// Move to top level of nested statements to get function
		var EObject obj = variable
		var Collection<ReferableObject> inScope = newArrayList()
		do {
			obj = obj.eContainer
			// collect available elements in scope on this level
			inScope.addAll(obj.eContents.filter(ReferableObject).toList)
		} while(!(obj instanceof Model) && obj.eContainer !== null)
		
		// Check if variable name overwrites other local name
		if(inScope.exists[it !== variable && it.name == variable.name]) {
			error('Duplicate declaration of ' + variable.name + '!', 
				MusketPackage.eINSTANCE.referableObject_Name,
				INVALID_ID)
		}
	}
	
	// Check that constants are not reassigned
	@Check
	def checkAssignmentToConstant(Assignment assignment) {
		if(assignment.^var?.value instanceof Constant){
			error('Value cannot be assigned to constant!', 
				MusketPackage.eINSTANCE.assignment_Var,
				INVALID_OPERATION)
		}
	}
	
	@Check
	def checkMusketAssignmentToConstant(MusketAssignment assignment) {
		if(assignment.^var?.value instanceof Constant){
			error('Value cannot be assigned to constant!', 
				MusketPackage.eINSTANCE.musketAssignment_Var,
				INVALID_OPERATION)
		}
	}
	
	// Check collectionFunctionCalls match with collection type
	@Check
	def checkValidCollectionFunctionCall(CollectionFunctionCall call) {
		val matrixFunctions = #[CollectionFunctionName.ROWS, CollectionFunctionName.ROWS_LOCAL, CollectionFunctionName.COLUMNS, CollectionFunctionName.COLUMNS_LOCAL, CollectionFunctionName.BLOCKS_IN_COLUMN, CollectionFunctionName.BLOCKS_IN_ROW]
		if(!(call.^var instanceof Matrix) && matrixFunctions.contains(call.function)){
			error(call.function + ' can only be applied to matrices!', 
				MusketPackage.eINSTANCE.collectionFunctionCall_Function,
				INVALID_OPERATION)
		}
	}
	
	// Ensure no constants are defined in structs
	@Check
	def checkNoConstantsInStructs(Constant const) {
		if(const.eContainer instanceof Struct){
			error('Constants are not allowed in structs!', 
				const, null)
		}
	}
	
	// Check for unreachable code
	@Check
	def checkUnreachableCodeAfterReturn(Function func) {
		//  check for unreachable code
		val iter = func.statement.iterator
		var foundReturn = false
		var counter = 0
		while (iter.hasNext){
			val currentStatement = iter.next
			
			if (!foundReturn && currentStatement instanceof ReturnStatement) {
				foundReturn = true
			} else if (foundReturn) {
				error('Unreachable code!', 
					func,
					currentStatement.eContainingFeature,
					counter)
			}
			
			counter++
		}
	}
}