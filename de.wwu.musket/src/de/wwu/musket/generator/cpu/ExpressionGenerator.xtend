package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.And
import de.wwu.musket.musket.Array
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.IndividualObject
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PreDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.Subtraction
import java.util.Map
import static extension de.wwu.musket.generator.cpu.ArrayFunctions.*
import static extension de.wwu.musket.generator.cpu.MusketFunctionCalls.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.cpu.StandardFunctionCalls.*
import de.wwu.musket.musket.TypeCast
import de.wwu.musket.musket.Modulo
import de.wwu.musket.musket.StandardFunctionCall
import static extension de.wwu.musket.util.CollectionHelper.*
import de.wwu.musket.musket.DistributionMode

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
			Modulo: '''(«expression.left.generateExpression(param_map)» % «expression.right.generateExpression(param_map)»)'''
			Not: '''!«expression.expression.generateExpression(param_map)»'''
			And: '''(«expression.leftExpression.generateExpression(param_map)» && «expression.rightExpression.generateExpression(param_map)»)'''
			Or: '''(«expression.leftExpression.generateExpression(param_map)» || «expression.rightExpression.generateExpression(param_map)»)'''
			ObjectRef case expression.isCollectionElementRef: '''«expression.generateCollectionElementRef(param_map).toString.removeLineBreak»'''
			ObjectRef: '''«expression.value.generateObjectRef(param_map)»«expression?.tail.generateTail»'''
			IntVal: '''«expression.value»'''
			DoubleVal: '''«expression.value»'''
			ExternalFunctionCall:
				throw new UnsupportedOperationException("ExpressionGenerator: ExternalFunctionCall")
			CollectionFunctionCall: '''«expression.generateCollectionFunctionCall»'''
			PostIncrement: '''«expression.value.generateObjectRef(param_map)»++'''
			PostDecrement: '''«expression.value.generateObjectRef(param_map)»--'''
			PreIncrement: '''++«expression.value.generateObjectRef(param_map)»'''
			PreDecrement: '''--«expression.value.generateObjectRef(param_map)»'''
			MusketFunctionCall: '''«expression.generateMusketFunctionCall»'''
			StandardFunctionCall: '''«expression.generateStandardFunctionCall»'''
			TypeCast: '''static_cast<«expression.targetType»>(«expression.expression.generateExpression(param_map)»)'''
			default: '''/* WARNING: ExpressionGenerator: ran into default case") */'''
		}
	}

	def static generateCollectionElementRef(ObjectRef cer, Map<String, String> param_map)'''
«««		ARRAY
		«IF cer.value instanceof Array»
«««			LOCAL REF
			«IF cer.localCollectionIndex.size == 1»
				«cer.value.name»[«cer.localCollectionIndex.head.generateExpression(param_map)»]«cer?.tail.generateTail»
«««			GLOBAL REF
			«ELSE»
«««				COPY
				«IF (cer.value as Array).distributionMode == DistributionMode.COPY»
					«cer.value.name»[«cer.globalCollectionIndex.head.generateExpression(param_map)»]«cer?.tail.generateTail»
«««				DIST
				«ELSE»
					// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
				«ENDIF»
			«ENDIF»
«««		MATRIX
		«ELSEIF cer.value instanceof Matrix»
«««			LOCAL REF
			«IF cer.localCollectionIndex.size == 2»
				«cer.value.name»[«cer.localCollectionIndex.head.generateExpression(param_map)» * «(cer.value as Matrix).colsLocal» + «cer.localCollectionIndex.drop(1).head.generateExpression(param_map)»]«cer?.tail.generateTail»
«««			GLOBAL REF
			«ELSEIF cer.globalCollectionIndex.size == 2»
«««					COPY
					«IF (cer.value as Matrix).distributionMode == DistributionMode.COPY»
						«cer.value.name»[«cer.globalCollectionIndex.head.generateExpression(param_map)» * «(cer.value as Matrix).colsLocal» + «cer.globalCollectionIndex.drop(1).head.generateExpression(param_map)»]«cer?.tail.generateTail»
«««					DIST
					«ELSE»
						//TODO: ExpressionGenerator.generateCollectionElementRef: Matrix, global indices, distributed
					«ENDIF»				
			«ENDIF»
		«ELSE»
			«cer.value.name»«cer?.tail.generateTail»
		«ENDIF»
	'''

	def static dispatch generateObjectRef(CollectionObject co, Map<String, String> param_map) '''«co.name»'''

	def static dispatch generateObjectRef(IndividualObject i, Map<String, String> param_map) '''«i.name»'''

	def static dispatch generateObjectRef(de.wwu.musket.musket.Parameter p, Map<String, String> param_map) '''«IF param_map !== null && param_map.containsKey(p.name)»«param_map.get(p.name)»«ELSE»«p.name»«ENDIF»'''
}
