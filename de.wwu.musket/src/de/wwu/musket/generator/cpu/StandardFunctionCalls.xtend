package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.StandardFunctionCall
import static extension de.wwu.musket.generator.cpu.Parameter.*

class StandardFunctionCalls {
	def static generateStandardFunctionCall(StandardFunctionCall sfc) //{
		'''«sfc.value.literal»«FOR p : sfc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateParameterInput)»«ENDFOR»;'''
}
