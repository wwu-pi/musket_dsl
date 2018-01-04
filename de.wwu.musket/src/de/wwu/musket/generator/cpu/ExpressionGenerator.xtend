package de.wwu.musket.generator.cpu

import static extension de.wwu.musket.generator.cpu.ArrayFunctions.*

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
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.Variable
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.PreDecrement
import de.wwu.musket.musket.CollectionFunctionCall

class ExpressionGenerator {
	def static String generateExpression(Expression expression, Map<String, String> param_map) {
		switch expression {
			Addition: '''(«expression.left.generateExpression(param_map)» + «expression.right.generateExpression(param_map)»)'''
			Subtraction: '''(«expression.left.generateExpression(param_map)» - «expression.right.generateExpression(param_map)»)'''
			Multiplication: '''(«expression.left.generateExpression(param_map)» * «expression.right.generateExpression(param_map)»)'''
			Division: '''(«expression.left.generateExpression(param_map)» / «expression.right.generateExpression(param_map)»)'''
			CompareExpression case expression.eqRight === null: '''«expression.eqLeft.generateExpression(param_map)»'''
			CompareExpression case expression.eqRight !==
				null: '''(«expression.eqLeft.generateExpression(param_map)» «expression.op» «expression.eqRight.generateExpression(param_map)»)'''
			SignedArithmetic: '''-(«expression.expression.generateExpression(param_map)»)'''
			Not: '''!«expression.expression.generateExpression(param_map)»'''
			And: '''(«expression.leftExpression.generateExpression(param_map)» && «expression.rightExpression.generateExpression(param_map)»)'''
			Or: '''(«expression.leftExpression.generateExpression(param_map)» || «expression.rightExpression.generateExpression(param_map)»)'''
			ObjectRef: '''«expression.value.generateObjectRef(param_map)»'''
			IntVal: '''«expression.value»'''
			DoubleVal: '''«expression.value»'''
			ExternalFunctionCall:
				throw new UnsupportedOperationException("ExpressionGenerator: ExternalFunctionCall")
			CollectionFunctionCall: '''«expression.generateCollectionFunctionCall»'''
			PostIncrement: '''«expression.value.generateObjectRef(param_map)»++'''
			PostDecrement: '''«expression.value.generateObjectRef(param_map)»--'''
			PreIncrement: '''++«expression.value.generateObjectRef(param_map)»'''
			PreDecrement: '''--«expression.value.generateObjectRef(param_map)»'''
			default: {throw new UnsupportedOperationException("ExpressionGenerator: ran into default case")}
		}
	}

	def static dispatch generateObjectRef(CollectionObject c, Map<String, String> param_map) {
		throw new UnsupportedOperationException("ExpressionGenerator: generateObjectRef")
	}

	def static dispatch generateObjectRef(IndividualObject i, Map<String, String> param_map) '''«i.name»'''

//	def static dispatch generateObjectRef(Variable v, Map<String, String> param_map)'''
//		It's an int
//	'''
	def static dispatch generateObjectRef(Parameter p, Map<String, String> param_map) '''«IF param_map !== null && param_map.containsKey(p.name)»«param_map.get(p.name)»«ELSE»«p.name»«ENDIF»'''
}
