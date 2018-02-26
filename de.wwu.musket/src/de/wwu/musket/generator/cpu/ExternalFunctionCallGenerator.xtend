package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ExternalFunctionCall
import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import java.util.Map

class ExternalFunctionCallGenerator {
	// TODO template parameters in namespace chain
	def static generateExternalFunctionCall(ExternalFunctionCall efc, Map<String, String> param_map) //{
		'''«FOR namespace : efc.namespaces SEPARATOR "::"»«namespace.name»«ENDFOR»«efc.function»«FOR p : efc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateExpression(param_map))»«ENDFOR»'''
}
