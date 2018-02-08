package de.wwu.musket.util

import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.And

class TypeHelper {
	
	// Helper to check if an expression is a boolean expression (hard to see due to type hierarchy)
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
		}
			
		return false;
	}
}