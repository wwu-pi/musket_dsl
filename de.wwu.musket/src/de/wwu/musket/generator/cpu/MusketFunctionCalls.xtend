package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.util.TypeHelper.*

/**
 * Generates all musket function calls, that is all function that are specific for musket, such as print or rand.
 * <p>
 * For some functions as print it is necessary that the result is a block, in this case an if-clause with the print expression.
 * However, musket functions might be a part of an expression, which makes it necessary that the result is just one line that can be used in a more complex expression.
 */
class MusketFunctionCalls {
	def static generateMusketFunctionCall(MusketFunctionCall mfc){
		switch mfc.value{
			case PRINT: generatePrint(mfc)
			case RAND: generateRand(mfc)
			default: ''''''
		}
	}
	
	def static generatePrint(MusketFunctionCall mfc)'''
		if(«Config.var_pid» == 0){
			printf«FOR p : mfc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateExpression(null))»«ENDFOR»;
		}
	'''	
	
	def static generateRand(MusketFunctionCall mfc)'''rand_dist_«mfc.params.head.calculateType.cppType»_«mfc.params.head.ValueAsString.replace('.', '_')»_«mfc.params.get(1).ValueAsString.replace('.', '_')»[omp_get_thread_num()](«Config.var_rng_array»[omp_get_thread_num()])'''	
}