package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ExternalFunctionCall
import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import java.util.Map

class ExternalFunctionCallGenerator {
	def static generateExternalFunctionCall(ExternalFunctionCall efc, Map<String, String> param_map) //{
		'''«efc.namespace»::«efc.function»«FOR p : efc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateExpression(param_map))»«ENDFOR»'''
}
