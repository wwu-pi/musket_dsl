package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ArrayFunctionCall
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.MainBlock
import de.wwu.musket.musket.MusketConditionalForLoop
import de.wwu.musket.musket.MusketIfClause
import de.wwu.musket.musket.MusketIteratorForLoop

import static extension de.wwu.musket.generator.cpu.ArrayFunctions.*
import static extension de.wwu.musket.generator.cpu.SkeletonGenerator.*
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.musket.MusketBoolVariable
import de.wwu.musket.musket.MusketIntVariable
import de.wwu.musket.musket.MusketDoubleVariable

class LogicGenerator {
	def static generateLogic(MainBlock mainBlock) '''
		«FOR s : mainBlock.content»
			«generateStatement(s)»
		«ENDFOR»
	'''

	def static dispatch generateStatement(MusketConditionalForLoop s) '''
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

	def static dispatch generateStatement(MusketBoolVariable s) '''
		bool «s.name» = true;
		«s.initExpression.generateSkeletonExpression(s.name)»
	'''

	def static dispatch generateStatement(Assignment s) '''
	'''

	def static dispatch generateStatement(ArrayFunctionCall s) '''
		«s.generateArrayFunctionCall»
	'''

	def static dispatch generateStatement(ExternalFunctionCall s) '''
	'''

	def static dispatch generateStatement(MusketAssignment s) '''
	'''
}
