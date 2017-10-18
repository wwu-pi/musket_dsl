package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.And
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.IndividualObject
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.Subtraction
import java.util.Map

class ExpressionGenerator {
	def static String generateExpression(Expression expression, Map<String, String> param_map){
		switch expression{
			Addition: '''(«expression.left.generateExpression(param_map)» + «expression.right.generateExpression(param_map)»)'''
			Subtraction:'''(«expression.left.generateExpression(param_map)» - «expression.right.generateExpression(param_map)»)'''
			Multiplication:'''(«expression.left.generateExpression(param_map)» * «expression.right.generateExpression(param_map)»)'''
			Division:'''(«expression.left.generateExpression(param_map)» / «expression.right.generateExpression(param_map)»)'''
			CompareExpression case expression.eqRight === null: '''«expression.eqLeft.generateExpression(param_map)»'''
			CompareExpression case expression.eqRight !== null: '''(«expression.eqLeft.generateExpression(param_map)» «expression.op» «expression.eqRight.generateExpression(param_map)»)'''
			SignedArithmetic: '''-(«expression.expression.generateExpression(param_map)»)'''	
			Not:'''!«expression.expression.generateExpression(param_map)»'''
			And: '''(«expression.leftExpression.generateExpression(param_map)» && «expression.rightExpression.generateExpression(param_map)»)'''
			Or: '''(«expression.leftExpression.generateExpression(param_map)» || «expression.rightExpression.generateExpression(param_map)»)'''
			ObjectRef: '''«expression.value.generateObjectRef(param_map)»'''
			IntVal: '''«expression.value»'''
			DoubleVal: '''«expression.value»'''
			ExternalFunctionCall: ''''''
			default: ''''''
		}
	}
	
	def static dispatch generateObjectRef(CollectionObject c, Map<String, String> param_map)'''
	'''
	
	def static dispatch generateObjectRef(IndividualObject i, Map<String, String> param_map)'''
	'''
	
	def static dispatch generateObjectRef(Parameter p, Map<String, String> param_map)'''«param_map.get(p.name)»'''
}