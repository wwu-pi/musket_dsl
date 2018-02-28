package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.MainBlock
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.MusketBoolVariable
import de.wwu.musket.musket.MusketConditionalForLoop
import de.wwu.musket.musket.MusketDoubleVariable
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketIfClause
import de.wwu.musket.musket.MusketIntVariable
import de.wwu.musket.musket.MusketIteratorForLoop
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.cpu.CollectionFunctionsGenerator.*
import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.MusketFunctionCalls.*
import static extension de.wwu.musket.generator.cpu.SkeletonGenerator.*
import static extension de.wwu.musket.util.TypeHelper.*

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
	def static generateLogic(MainBlock mainBlock) '''
		«FOR s : mainBlock.content»
			«generateStatement(s)»
		«ENDFOR»
	'''

// dispatch methods for musket statements
	def static dispatch generateStatement(MusketConditionalForLoop s) '''
		for(«s.init.calculateType.cppType» «s.init.name» = «s.init.initExpression.generateExpression(null)»; «s.condition.generateExpression(null)»; «s.increment.generateExpression(null)»){
			«FOR mainstatement : s.statements»
				«mainstatement.generateStatement()»
			«ENDFOR»
		}
	'''

	def static dispatch generateStatement(MusketIteratorForLoop s) '''
		// TODO: LogicGenerator: generateStatement(MusketIteratorForLoop)
	'''

	def static dispatch generateStatement(MusketIfClause s) '''
		// TODO: LogicGenerator: generateStatement(MusketIfClause)
	'''

	def static dispatch generateStatement(SkeletonExpression s) '''
		«s.generateSkeletonExpression(null)»
	'''

	def static dispatch generateStatement(MusketIntVariable s) '''
		int «s.name» = 0;
		«s.initExpression.generateSkeletonExpression(s.name)»
	'''

	def static dispatch generateStatement(MusketDoubleVariable s) '''
		double «s.name» = 0.0;
		«s.initExpression.generateSkeletonExpression(s.name)»
	'''

	def static dispatch generateStatement(MusketBoolVariable s) '''
		bool «s.name» = true;
		«s.initExpression.generateSkeletonExpression(s.name)»
	'''

	def static dispatch generateStatement(Variable v) '''
		«v.calculateType.cppType» «v.name» «IF v.initExpression !== null» = «v.initExpression.generateExpression(null)»«ENDIF»;
	'''

	def static dispatch generateStatement(Assignment s) '''
		// TODO: LogicGenerator: generateStatement(Assignment)
	'''

	def static dispatch generateStatement(CollectionFunctionCall s) '''
		«s.generateCollectionFunctionCall»
	'''

	def static dispatch generateStatement(ExternalFunctionCall s) '''
		// TODO: LogicGenerator: generateStatement(ExternalFunctionCall)
	'''

	def static dispatch generateStatement(MusketFunctionCall s) {
		s.generateMusketFunctionCall
	}

	def static dispatch generateStatement(MusketAssignment s) {
		switch s.value {
			Expression: '''«s.^var.value.name» = «(s.value as Expression).generateExpression(null)»;'''
			SkeletonExpression:
				(s.value as SkeletonExpression).generateSkeletonExpression(s.^var.value.name)
			default: '''// TODO: LogicGenerator: generateStatement: MusketAssignment'''
		}
	}
}
