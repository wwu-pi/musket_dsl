package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.ExternalFunctionCall
import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import java.util.Map

/**
 * Generate calls to functions, which are not part of musket.
 * This can include functions of the std library, or any functions defined in cpp files outside of musket.
 */
class ExternalFunctionCallGenerator {

	/**
	 * Generates the external function call.
	 * TODO: template parameters in namespace chain
	 * 
	 * @param efc the external function call
	 * @param param_map the param map, if call happens within a function, otherwise null.
	 * @return the generated code
	 */
	def static generateExternalFunctionCall(ExternalFunctionCall efc,
		Map<String, String> param_map, int processId) '''«FOR namespace : efc.namespaces SEPARATOR "::" AFTER "::"»«namespace.name»«ENDFOR»«efc.function»«FOR p : efc.params BEFORE '(' SEPARATOR ', ' AFTER ')'»«(p.generateExpression(param_map, processId))»«ENDFOR»'''
}
