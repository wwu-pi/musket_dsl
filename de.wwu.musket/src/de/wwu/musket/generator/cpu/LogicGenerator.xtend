package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.CollectionFunctionCall
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
import de.wwu.musket.musket.StandardFunctionCall

import static extension de.wwu.musket.generator.cpu.ArrayFunctions.*
import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.MusketFunctionCalls.*
import static extension de.wwu.musket.generator.cpu.SkeletonGenerator.*
import static extension de.wwu.musket.generator.cpu.StandardFunctionCalls.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

class LogicGenerator {
	def static generateLogic(MainBlock mainBlock) '''
		«FOR s : mainBlock.content»
			«generateStatement(s)»
		«ENDFOR»
	'''

	def static dispatch generateStatement(MusketConditionalForLoop s) '''
		for(«s.init.CppPrimitiveTypeAsString» «s.init.name» = «s.init.initExpression.generateExpression(null)»; «s.condition.generateExpression(null)»; «s.increment.generateExpression(null)»){
			«FOR mainstatement : s.statements»
				«mainstatement.generateStatement()»
			«ENDFOR»
		}
	'''

	def static dispatch generateStatement(MusketIteratorForLoop s) '''
	'''

	def static dispatch generateStatement(MusketIfClause s) '''
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

	def static dispatch CharSequence generateStatement(MusketBoolVariable s) '''
		bool «s.name» = true;
		«s.initExpression.generateSkeletonExpression(s.name)»
	'''

	def static dispatch generateStatement(Assignment s) '''
	'''

	def static dispatch generateStatement(CollectionFunctionCall s) '''
		«s.generateCollectionFunctionCall»
	'''

	def static dispatch generateStatement(StandardFunctionCall s) '''
		«s.generateStandardFunctionCall»
	'''

	def static dispatch generateStatement(ExternalFunctionCall s) '''
	'''

	def static dispatch generateStatement(MusketFunctionCall s) {
		s.generateMusketFunctionCall
	}

	def static dispatch generateStatement(MusketAssignment s) '''
	'''
}
