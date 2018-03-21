package de.wwu.musket.util

import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.And
import de.wwu.musket.musket.BoolArrayType
import de.wwu.musket.musket.BoolConstant
import de.wwu.musket.musket.BoolMatrixType
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.BoolVariable
import de.wwu.musket.musket.CollectionInstantiation
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.CollectionParameter
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.DoubleArrayType
import de.wwu.musket.musket.DoubleConstant
import de.wwu.musket.musket.DoubleMatrixType
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.DoubleVariable
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.IndividualParameter
import de.wwu.musket.musket.IntArrayType
import de.wwu.musket.musket.IntConstant
import de.wwu.musket.musket.IntMatrixType
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.Modulo
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.MusketBoolVariable
import de.wwu.musket.musket.MusketDoubleVariable
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketIntVariable
import de.wwu.musket.musket.MusketStructVariable
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PreDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.PrimitiveType
import de.wwu.musket.musket.Ref
import de.wwu.musket.musket.ReferableObject
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType
import de.wwu.musket.musket.StructVariable
import de.wwu.musket.musket.Subtraction
import de.wwu.musket.musket.TypeCast
import org.eclipse.emf.ecore.EObject

import static extension de.wwu.musket.util.CollectionHelper.*
import de.wwu.musket.musket.FloatArrayType
import de.wwu.musket.musket.FloatMatrixType
import de.wwu.musket.musket.FloatVal
import de.wwu.musket.musket.FloatConstant
import de.wwu.musket.musket.MusketFloatVariable
import de.wwu.musket.musket.FloatVariable

class TypeHelper {
	static dispatch def MusketType calculateCollectionType(IntArrayType obj){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateCollectionType(DoubleArrayType obj){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateCollectionType(FloatArrayType obj){
		return MusketType.FLOAT
	}
	
	static dispatch def MusketType calculateCollectionType(BoolArrayType obj){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateCollectionType(StructArrayType obj){
		return new MusketType(obj.type)
	}
	
	static dispatch def MusketType calculateCollectionType(IntMatrixType obj){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateCollectionType(DoubleMatrixType obj){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateCollectionType(FloatMatrixType obj){
		return MusketType.FLOAT
	}
	
	static dispatch def MusketType calculateCollectionType(BoolMatrixType obj){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateCollectionType(StructMatrixType obj){
		return new MusketType(obj.type).toSingleValued
	}
	
	static dispatch def MusketType calculateCollectionType(CollectionParameter obj){
		return new MusketType(obj.type).toSingleValued
	}
	
	static dispatch def MusketType calculateCollectionType(CollectionObject obj){
		return new MusketType(obj.type).toSingleValued
	}
	
	static dispatch def MusketType calculateCollectionType(ObjectRef obj){
		if(obj.isCollectionElementRef) {
			// A collection _element_ has no collection type
			return null
		}
		
		return obj.value.calculateCollectionType
	}
	
	static dispatch def MusketType calculateCollectionType(MusketType t){
		return t.toSingleValued;
	}
	
	static dispatch def MusketType calculateCollectionType(ReferableObject obj){
		if(obj.eIsProxy) return null
		
		println("try to calculate collection type for " + obj)
		return MusketType.AUTO;
	}
	
	// calculate type
	static dispatch def MusketType calculateType(PrimitiveType exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(IntVal exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(DoubleVal exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(FloatVal exp){
		return MusketType.FLOAT
	}
	
	static dispatch def MusketType calculateType(BoolVal exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(StringVal exp){
		return MusketType.STRING
	}
	
	static dispatch def MusketType calculateType(InternalFunctionCall exp){
		return new MusketType(exp.value)
	}
	
	static dispatch def MusketType calculateType(MusketFunctionCall exp){
		if (exp.value === null) return null;
		
		switch exp.value {			
			case RAND: return exp.params?.head?.calculateType
			case DOUBLE_MIN: return MusketType.DOUBLE
			case DOUBLE_MAX: return MusketType.DOUBLE
			case FLOAT_MIN: return MusketType.FLOAT
			case FLOAT_MAX: return MusketType.FLOAT
			default: return MusketType.STRING
		}
	}
	
	static dispatch def MusketType calculateType(IntConstant exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(DoubleConstant exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(FloatConstant exp){
		return MusketType.FLOAT
	}
	
	static dispatch def MusketType calculateType(BoolConstant exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(MusketIntVariable exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(MusketDoubleVariable exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(MusketFloatVariable exp){
		return MusketType.FLOAT
	}
	
	static dispatch def MusketType calculateType(MusketBoolVariable exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(MusketStructVariable exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(IntVariable exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(DoubleVariable exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(FloatVariable exp){
		return MusketType.FLOAT
	}
	
	static dispatch def MusketType calculateType(StructVariable exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(BoolVariable exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(IndividualParameter exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(CollectionParameter exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(CollectionObject exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(IntArrayType exp){
		return MusketType.INT_ARRAY
	}
	
	static dispatch def MusketType calculateType(DoubleArrayType exp){
		return MusketType.DOUBLE_ARRAY
	}
	
	static dispatch def MusketType calculateType(FloatArrayType exp){
		return MusketType.FLOAT_ARRAY
	}
	
	static dispatch def MusketType calculateType(BoolArrayType exp){
		return MusketType.BOOL_ARRAY
	}
	
	static dispatch def MusketType calculateType(StructArrayType exp){
		return new MusketType(exp.type).toArray
	}
	
	static dispatch def MusketType calculateType(IntMatrixType exp){
		return MusketType.INT_MATRIX
	}
	
	static dispatch def MusketType calculateType(DoubleMatrixType exp){
		return MusketType.DOUBLE_MATRIX
	}
	
	static dispatch def MusketType calculateType(FloatMatrixType exp){
		return MusketType.FLOAT_MATRIX
	}
	
	static dispatch def MusketType calculateType(BoolMatrixType exp){
		return MusketType.BOOL_MATRIX
	}
	
	static dispatch def MusketType calculateType(StructMatrixType exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(Ref exp){
		// Go down nested reference structure
		if(exp.tail !== null) return exp.tail.calculateType
		
		// Type of collection _element_ 
		if(exp.isCollectionElementRef) return exp.value.calculateCollectionType
		
		// Other object references
		return exp.value.calculateType
	}
		
	static dispatch def MusketType calculateType(SignedArithmetic exp){
		return exp.expression.calculateType
	}
	
	static dispatch def MusketType calculateType(TypeCast exp){
		return exp.targetType.calculateType
	}
	
	static dispatch def MusketType calculateType(PostIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(PostDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(PreIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(PreDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(Addition exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType == MusketType.DOUBLE) return MusketType.DOUBLE // anything plus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Subtraction exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType == MusketType.DOUBLE) return MusketType.DOUBLE // anything minus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Multiplication exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType == MusketType.DOUBLE) return MusketType.DOUBLE // anything times double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Division exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			return exp.right.calculateType // division result depends on right side 
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Modulo exp){
		if(exp.right !== null) {
			if(exp.left.calculateType != MusketType.INT || exp.right.calculateType != MusketType.INT) return null // Modulo requires two ints 
		}
		return exp.left.calculateType
	}
	
	static dispatch def MusketType calculateType(Not exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(And exp){
		if(exp.rightExpression !== null) return MusketType.BOOL
		return exp.leftExpression.calculateType 
	}
	
	static dispatch def MusketType calculateType(Or exp){
		if(exp.rightExpression !== null) return MusketType.BOOL
		return exp.leftExpression.calculateType
	}
	
	static dispatch def MusketType calculateType(CompareExpression exp){
		if(exp.eqRight !== null) return MusketType.BOOL
		return exp.eqLeft?.calculateType
	}
	
	static dispatch def MusketType calculateType(Function exp){
		return new MusketType(exp)
	}
	
	static dispatch def MusketType calculateType(ReturnStatement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(ExternalFunctionCall call){
		// We cannot know this
		return MusketType.AUTO
	}
	
	static dispatch def MusketType calculateType(CollectionInstantiation exp){
		return new MusketType(exp.type).toLocalCollection
	}
	
	static dispatch def MusketType calculateType(EObject exp){ // Else case
		if(exp.eIsProxy) return null
		
		println("try to calculate type for unknown object " + exp)
		return MusketType.AUTO
	}
}