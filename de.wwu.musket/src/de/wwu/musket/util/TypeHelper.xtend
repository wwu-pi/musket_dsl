package de.wwu.musket.util

import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.And
import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.Subtraction
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.PreDecrement

class TypeHelper {
	
	// Helper to check if an expression is a boolean expression (hard to see due to type hierarchy)
	// TODO check functionCall content
	static def isBoolean(Expression exp){
		if((exp as CompareExpression).eqRight !== null){
			// Comparison
			return true;
			
		} else if(((exp as CompareExpression).eqLeft instanceof Or) && ((exp as CompareExpression).eqLeft as Or).rightExpression !== null){
			// OR condition
			return true;
			
		} else if(((exp as CompareExpression).eqLeft instanceof And) && ((exp as CompareExpression).eqLeft as And).rightExpression !== null){
			// AND condition
			return true;
		
		} else if((exp as CompareExpression).eqLeft instanceof BoolVal){
			// Boolean value
			return true;
		
		} else if((exp as CompareExpression).eqLeft instanceof Not){
			// NOT operation
			return true;
			
		} else if((exp as CompareExpression).eqLeft instanceof FunctionCall){
			// external call
			return true;
		}
			
		return false;
	}
	
	// Helper to check if an expression is a numeric expression (hard to see due to type hierarchy)
	// TODO check functionCall content
	static def isNumeric(Expression exp){
		if( ((exp as CompareExpression).eqLeft instanceof Addition) ||
			((exp as CompareExpression).eqLeft instanceof Subtraction) ||
			((exp as CompareExpression).eqLeft instanceof Multiplication) ||
			((exp as CompareExpression).eqLeft instanceof Division) ||
			((exp as CompareExpression).eqLeft instanceof IntVal) ||
			((exp as CompareExpression).eqLeft instanceof DoubleVal) ||
			((exp as CompareExpression).eqLeft instanceof SignedArithmetic) ||
			((exp as CompareExpression).eqLeft instanceof PostIncrement) ||
			((exp as CompareExpression).eqLeft instanceof PostDecrement) ||
			((exp as CompareExpression).eqLeft instanceof PreIncrement) ||
			((exp as CompareExpression).eqLeft instanceof PreDecrement) ||
			((exp as CompareExpression).eqLeft instanceof FunctionCall)){
			return true;
		}
		
		return false;
	}
	
	// Helper to check if an expression is a string expression (hard to see due to type hierarchy)
	// TODO check functionCall content ((exp as CompareExpression).eqLeft instanceof FunctionCall)
	static def isString(Expression exp){
		if( ((exp as CompareExpression).eqLeft instanceof Addition) && 
			((exp as CompareExpression).eqLeft as Addition).right instanceof StringVal){
			// String concatenation
			return true;
			
		} else if((exp as CompareExpression).eqLeft instanceof FunctionCall){
			return true;
		}
		
		return false;
	}
}