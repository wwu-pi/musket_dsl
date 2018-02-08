package de.wwu.musket.util

import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.And
import de.wwu.musket.musket.BoolConstant
import de.wwu.musket.musket.BoolParameter
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.BoolVariable
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.DoubleConstant
import de.wwu.musket.musket.DoubleParameter
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.DoubleVariable
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.IntConstant
import de.wwu.musket.musket.IntParameter
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.MusketBoolVariable
import de.wwu.musket.musket.MusketDoubleVariable
import de.wwu.musket.musket.MusketIntVariable
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PreDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.Subtraction
import de.wwu.musket.musket.Type
import org.eclipse.emf.ecore.EObject

class TypeHelper {
	
	static def isNumeric(Type type){
		return type === Type.INT || type === Type.DOUBLE
	}
	
	// Helper to check the expression type (hard to see within type hierarchy)
	// TODO collections?? missing
	static dispatch def Type calculateType(IntVal exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleVal exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolVal exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(StringVal exp){
		return Type.STRING
	}
	
	// TODO check functionCall content
	static dispatch def Type calculateType(FunctionCall exp){
		return null
	}
	
	static dispatch def Type calculateType(IntConstant exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleConstant exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolConstant exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(MusketIntVariable exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(MusketDoubleVariable exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(MusketBoolVariable exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(IntVariable exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleVariable exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolVariable exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(IntParameter exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleParameter exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolParameter exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(ObjectRef exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(SignedArithmetic exp){
		return exp.expression.calculateType
	}
	
	static dispatch def Type calculateType(PostIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(PostDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(PreIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(PreDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(Addition exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType === Type.DOUBLE) return Type.DOUBLE // anything plus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Subtraction exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType === Type.DOUBLE) return Type.DOUBLE // anything minus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Multiplication exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType === Type.DOUBLE) return Type.DOUBLE // anything times double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Division exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			return exp.right.calculateType // division result depends on right side 
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Not exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(And exp){
		if(exp.rightExpression !== null) return Type.BOOL
		return exp.leftExpression.calculateType 
	}
	
	static dispatch def Type calculateType(Or exp){
		if(exp.rightExpression !== null) return Type.BOOL
		return exp.leftExpression.calculateType
	}
	
	static dispatch def Type calculateType(CompareExpression exp){
		if(exp.eqRight !== null) return Type.BOOL
		return exp.eqLeft?.calculateType
	}
	
	static dispatch def Type calculateType(EObject exp){ // Else case
		return null
	}
}