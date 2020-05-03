package de.wwu.musket.validation

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionFunctionName
import de.wwu.musket.musket.Constant
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Model
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.MusketObject
import de.wwu.musket.musket.MusketPackage
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.Ref
import de.wwu.musket.musket.ReferableObject
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.Struct
import java.util.Collection
import org.eclipse.emf.ecore.EObject
import org.eclipse.xtext.validation.Check

import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.StructVariable

class MusketModelValidator extends AbstractMusketValidator {
	
	public static val INVALID_ID = 'invalidIdentifier'
	public static val INVALID_OPERATION = 'invalidOperation'
	public static val INVALID_PARAMETER = 'invalidParameter'
	
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
		if(!call.^var.calculateType.isMatrix && matrixFunctions.contains(call.function)){
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
	
	// Check collection access expression matches dimensions
	@Check
	def checkCollectionAccessIsNumeric(Ref ref) {
		val dimensions = if (ref.value?.calculateType?.isArray) 1 
					else if (ref.value?.calculateType?.isMatrix) 2 else 0
		
		val errorText = if(dimensions == 1) 'Array element access expects 1 dimension, ' else 'Matrix element access expects 2 dimensions, '
		
		if(ref.localCollectionIndex?.size > 0 && ref.localCollectionIndex?.size !== dimensions){
			error(errorText + ref.localCollectionIndex?.size + ' given!', 
				MusketPackage.eINSTANCE.ref_LocalCollectionIndex,
				INVALID_PARAMETER)
		}
		if(ref.globalCollectionIndex?.size > 0 && ref.globalCollectionIndex?.size !== dimensions){
			error(errorText + ref.globalCollectionIndex?.size + ' given!', 
				MusketPackage.eINSTANCE.ref_GlobalCollectionIndex,
				INVALID_PARAMETER)
		}
	}
	
	// Ensure constants in array/matrix dimensions are positive
	@Check
	def checkPositiveValuesInCollectionDimensions(ArrayType array){
		if(array.size?.ref?.value < 0) {
			error('Array dimensions must be positive!', 
				MusketPackage.eINSTANCE.arrayType_Size,
				INVALID_PARAMETER)
		}
		
	}
	
	@Check
	def checkPositiveValuesInCollectionDimensions(MatrixType matrix){
		if(matrix.rows?.ref?.value < 0) {
			error('Matrix dimensions must be positive!', 
				MusketPackage.eINSTANCE.matrixType_Rows,
				INVALID_PARAMETER)
		}
		if(matrix.cols?.ref?.value < 0) {
			error('Matrix dimensions must be positive!', 
				MusketPackage.eINSTANCE.matrixType_Cols,
				INVALID_PARAMETER)
		}
	}
	
	// Ensure copy constructor for structs gets struct input
	@Check
	def checkCopyConstructorInput(StructVariable s){
		if(s.copyFrom !== null && !s.copyFrom.calculateType.isStruct){
			error('Copy constructor expects struct input, ' + s.copyFrom.calculateType + ' given!', 
				MusketPackage.eINSTANCE.structVariable_CopyFrom,
				INVALID_PARAMETER)
		}
	} 
}