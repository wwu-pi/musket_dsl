package de.wwu.musket.generator.cuda

import de.wwu.musket.musket.Object
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.MainBlock
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.MusketBoolVariable
import de.wwu.musket.musket.MusketConditionalForLoop
import de.wwu.musket.musket.MusketDoubleVariable
import de.wwu.musket.musket.MusketFloatVariable
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketIfClause
import de.wwu.musket.musket.MusketIntVariable
import de.wwu.musket.musket.MusketIteratorForLoop
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.cuda.CollectionFunctionsGenerator.*
import static extension de.wwu.musket.generator.cuda.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cuda.MusketFunctionCalls.*
import static extension de.wwu.musket.generator.cuda.SkeletonGenerator.*
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.StructVariable

/**
 * Generates the content of the main block.
 * <p>
 * The entry point is the function generateLogic(MainBlock mainBlock).
 * The musket statements (assignment, loop, if ...) are handled by separate dispatch methods, which are called from the generate logic method.
 * The generator is called by the source file generator.
 */
class LogicGenerator {
	/**
	 * Generates the main logic.
	 * Called by source file generator.
	 * The generation of the musket statments is handled by separate dispatch methods.
	 * 
	 * @param mainBlock the main block object
	 * @return the generated code
	 */
	def static generateLogic(MainBlock mainBlock, int processId) '''
		«FOR s : mainBlock.content»
			«generateStatement(s, processId)»
		«ENDFOR»
	'''

// dispatch methods for musket statements
	def static dispatch generateStatement(MusketConditionalForLoop s, int processId) '''
		for(«s.init.calculateType.cppType» «s.init.name» = «s.init.initExpression.generateExpression(null, processId)»; «s.condition.generateExpression(null, processId)»; «s.increment.generateExpression(null, processId)»){
			«FOR mainstatement : s.statements»
				«mainstatement.generateStatement(processId)»
			«ENDFOR»
		}
	'''

	def static dispatch generateStatement(MusketIteratorForLoop s, int processId) '''
		// TODO: LogicGenerator: generateStatement(MusketIteratorForLoop)
	'''

	def static dispatch generateStatement(MusketIfClause s, int processId) '''
		«FOR ifs : s.ifClauses SEPARATOR "\n} else " AFTER "}"»
			if(«ifs.condition.generateExpression(null, processId)»){
				«FOR statement: ifs.statements»
					«statement.generateStatement(processId)»
				«ENDFOR»
		«ENDFOR»
		«IF !s.elseStatements.nullOrEmpty»}else{
			«FOR statement : s.elseStatements»
				«statement.generateStatement(processId)»
			«ENDFOR»
		}
		«ENDIF»
	'''

	def static dispatch generateStatement(SkeletonExpression s, int processId) '''
		«s.generateSkeletonExpression(null, processId)»
	'''

	def static dispatch generateStatement(MusketIntVariable s, int processId) '''
		int «s.name» = 0;
		«s?.initExpression.generateSkeletonExpression(s, processId)»
	'''

	def static dispatch generateStatement(MusketDoubleVariable s, int processId) '''
		double «s.name» = 0.0;
		«s?.initExpression.generateSkeletonExpression(s, processId)»
	'''

	def static dispatch generateStatement(MusketFloatVariable s, int processId) '''
		float «s.name» = 0.0f;
		«s?.initExpression.generateSkeletonExpression(s, processId)»
	'''

	def static dispatch generateStatement(MusketBoolVariable s, int processId) '''
		bool «s.name» = true;
		«s?.initExpression.generateSkeletonExpression(s, processId)»
	'''

	def static dispatch generateStatement(Variable v, int processId) '''
		«v.calculateType.cppType» «v.name»«IF v.initExpression !== null» = «v.initExpression.generateExpression(null, processId)»«ELSEIF v instanceof StructVariable && (v as StructVariable).copyFrom !== null»{«(v as StructVariable).copyFrom.value.name»}«ENDIF»;
	'''

	def static dispatch generateStatement(Assignment s, int processId) '''
		// TODO: LogicGenerator: generateStatement(Assignment)
	'''

	def static dispatch generateStatement(CollectionFunctionCall s, int processId) '''
		«s.generateCollectionFunctionCall(processId)»
	'''

	def static dispatch generateStatement(ExternalFunctionCall s, int processId) '''
		// TODO: LogicGenerator: generateStatement(ExternalFunctionCall)
	'''

	def static dispatch generateStatement(MusketFunctionCall s, int processId) {
		s.generateMusketFunctionCall(processId)
	}

	def static dispatch generateStatement(MusketAssignment s, int processId) {
		switch s.value {
			Expression: '''«s.^var.value.name» = «(s.value as Expression).generateExpression(null, processId)»;'''
			SkeletonExpression:
				(s.value as SkeletonExpression).generateSkeletonExpression(s.^var.value as Object, processId)
			default: '''// TODO: LogicGenerator: generateStatement: MusketAssignment'''
		}
	}
}
