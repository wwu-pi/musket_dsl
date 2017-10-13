package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.MainBlock
import de.wwu.musket.musket.MusketConditionalForLoop
import de.wwu.musket.musket.MusketIfClause
import de.wwu.musket.musket.MusketIteratorForLoop
import de.wwu.musket.musket.SkeletonStatement

import static extension de.wwu.musket.generator.cpu.SkeletonGenerator.*

class LogicGenerator {
	def static generateLogic(MainBlock mainBlock)'''
		«FOR s : mainBlock.content»
			«generateStatement(s)»
		«ENDFOR»
	'''
	
	def static dispatch generateStatement(MusketConditionalForLoop s)'''
	'''
	
	def static dispatch generateStatement(MusketIteratorForLoop s)'''
	'''
	
	def static dispatch generateStatement(MusketIfClause s)'''
	'''
	
	def static dispatch generateStatement(SkeletonStatement s)'''
		«s.generateSkeletonStatement»
	'''
}