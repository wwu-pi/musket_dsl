package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.And
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.IndividualObject
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Modulo
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
import de.wwu.musket.musket.TypeCast
import java.util.Map

import static extension de.wwu.musket.generator.cpu.mpmd.ExternalFunctionCallGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.MusketFunctionCalls.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.util.CollectionHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.generator.cpu.CollectionFunctionsGenerator.*
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.BoolVal
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import de.wwu.musket.musket.FloatVal
import de.wwu.musket.musket.CollectionObjectOrParam

/**
 * Generates expressions, such as 1+1.
 * <p>
 * Expressions can be used in many places, for example also as parameter input.
 * Consequently, this generator generates expressions as a single line, even if that means that certain 
 * expressions, which could be evaluated once in the cpp program and be stored in a variable, are generated in such a way
 * that they have to be re-evaluated in the cpp program.
 */
class ExpressionGenerator {
	
	private static final Logger logger = LogManager.getLogger(ExpressionGenerator)
	
	/**
	 * Starting point for the expression generator.
	 * 
	 * @param expression the expression
	 * @param param_map the param_map. It maps a function parameter with the input so that it can be replaced accordingly. If there is no param_map, pass null.
	 * @return the generated code
	 */
	def static String generateExpression(Expression expression, Map<String, String> param_map, int processId) {
		switch expression {
			Addition: '''(«expression.left.generateExpression(param_map, processId)» + «expression.right.generateExpression(param_map, processId)»)'''
			Subtraction: '''(«expression.left.generateExpression(param_map, processId)» - «expression.right.generateExpression(param_map, processId)»)'''
			Multiplication: '''(«expression.left.generateExpression(param_map, processId)» * «expression.right.generateExpression(param_map, processId)»)'''
			Division: '''(«expression.left.generateExpression(param_map, processId)» / «expression.right.generateExpression(param_map, processId)»)'''
			CompareExpression case expression.eqRight === null: '''«expression.eqLeft.generateExpression(param_map, processId)»'''
			CompareExpression case expression.eqRight !==
				null: '''(«expression.eqLeft.generateExpression(param_map, processId)» «expression.op» «expression.eqRight.generateExpression(param_map, processId)»)'''
			SignedArithmetic: '''-(«expression.expression.generateExpression(param_map, processId)»)'''
			Modulo: '''(«expression.left.generateExpression(param_map, processId)» % «expression.right.generateExpression(param_map, processId)»)'''
			Not: '''!«expression.expression.generateExpression(param_map, processId)»'''
			And: '''(«expression.leftExpression.generateExpression(param_map, processId)» && «expression.rightExpression.generateExpression(param_map, processId)»)'''
			Or: '''(«expression.leftExpression.generateExpression(param_map, processId)» || «expression.rightExpression.generateExpression(param_map, processId)»)'''
			ObjectRef case expression.isCollectionElementRef: '''«expression.generateCollectionElementRef(param_map, processId).toString.removeLineBreak»'''
			ObjectRef: '''(«expression.value.generateObjectRef(param_map)»)«expression?.tail.generateTail»'''
			IntVal: '''«expression.value»'''
			DoubleVal: '''«expression.value»'''
			FloatVal: '''«expression.value»f'''
			StringVal: '''"«expression.value.replaceAll("\n", "\\\\n").replaceAll("\t", "\\\\t")»"''' // this is necessary so that the line break remains as \n in the generated code
			BoolVal: '''«expression.value»'''
			ExternalFunctionCall: '''«expression.generateExternalFunctionCall(param_map, processId)»'''
			CollectionFunctionCall: '''«expression.generateCollectionFunctionCall»'''
			PostIncrement: '''«expression.value.generateObjectRef(param_map)»++'''
			PostDecrement: '''«expression.value.generateObjectRef(param_map)»--'''
			PreIncrement: '''++«expression.value.generateObjectRef(param_map)»'''
			PreDecrement: '''--«expression.value.generateObjectRef(param_map)»'''
			MusketFunctionCall: '''«expression.generateMusketFunctionCall(processId)»'''
			TypeCast: '''static_cast<«expression.targetType.calculateType.cppType»>(«expression.expression.generateExpression(param_map, processId)»)'''
			default: {logger.error("Expression Generator ran into default case!"); ""}
		}
	}

/**
 * Generate a reference to a collection element.
 * The function considers different cases, based on:
 * array or matrix
 * global or local index
 * distributed or copy
 * 
 * @param or the object ref object 
 * @param param_map the param map
 * @return the generated code
 */
	def static generateCollectionElementRef(ObjectRef or, Map<String, String> param_map, int processId)'''
		«val orName = if(param_map !== null && param_map.containsKey(or.value.name)){param_map.get(or.value.name)}else{or.value.name}»
«««		ARRAY
		«IF or.value.calculateType.isArray»
«««			LOCAL REF
			«IF or.localCollectionIndex.size == 1»
				(«orName»).get_local(«or.localCollectionIndex.head.generateExpression(param_map, processId)»)«or?.tail.generateTail»
«««			GLOBAL REF
			«ELSE»
«««				COPY or LOC
				«IF (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.LOC»
					(«orName»)[«or.globalCollectionIndex.head.generateExpression(param_map, processId)»]«or?.tail.generateTail»
«««             COPY or LOC
				«ELSEIF (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.COPY »
					(«orName»).get_local(«or.globalCollectionIndex.head.generateExpression(param_map, processId)»)«or?.tail.generateTail»
«««				DIST
				«ELSE»
					// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
				«ENDIF»
			«ENDIF»
«««		MATRIX
		«ELSEIF or.value.calculateType.isMatrix»
«««			LOCAL REF
			«IF or.localCollectionIndex.size == 2»
				(«orName»)[«or.localCollectionIndex.head.generateExpression(param_map, processId)» * «((or.value as CollectionObject).type as MatrixType).colsLocal» + «or.localCollectionIndex.drop(1).head.generateExpression(param_map, processId)»]«or?.tail.generateTail»
«««			GLOBAL REF
			«ELSEIF or.globalCollectionIndex.size == 2»
«««					COPY
					«IF (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.COPY || (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.LOC»
						(«orName»)[«or.globalCollectionIndex.head.generateExpression(param_map, processId)» * «((or.value as CollectionObject).type as MatrixType).colsLocal» + «or.globalCollectionIndex.drop(1).head.generateExpression(param_map, processId)»]«or?.tail.generateTail»
«««					DIST
					«ELSE»
						//TODO: ExpressionGenerator.generateCollectionElementRef: Matrix, global indices, distributed
					«ENDIF»				
			«ENDIF»
		«ELSE»
			(«orName»)«or?.tail.generateTail»
		«ENDIF»
	'''


// dispatch methods for generation of OjbectReference

	def static dispatch generateObjectRef(CollectionObject co, Map<String, String> param_map) '''«co.name»'''

	def static dispatch generateObjectRef(IndividualObject i, Map<String, String> param_map) '''«i.name»'''

	def static dispatch generateObjectRef(de.wwu.musket.musket.Parameter p, Map<String, String> param_map) '''«IF param_map !== null && param_map.containsKey(p.name)»«param_map.get(p.name)»«ELSE»«p.name»«ENDIF»'''
}
