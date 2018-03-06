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
	/**
	 * This is the starting point of the Musket function call generator.
	 * <p>
	 * It switches over the type of function and calls the correct method to generate the code or directly returns it for simple cases.
	 * 
	 * @param mfc the musket function call.
	 * @return the generated code for the function call.
	 */
	def static generateMusketFunctionCall(MusketFunctionCall mfc) {
		switch mfc.value {
			case PRINT:
				generatePrint(mfc)
			case RAND:
				generateRand(mfc)
			case FLOAT_MIN: '''std::numeric_limits<float>::min()'''
			case FLOAT_MAX: '''std::numeric_limits<float>::max()'''
			case DOUBLE_MIN: '''std::numeric_limits<double>::min()'''
			case DOUBLE_MAX: '''std::numeric_limits<double>::max()'''
			case ROI_START:
				generateRoiStart(mfc)
			case ROI_END:
				generateRoiEnd(mfc)
			default: ''''''
		}
	}

	/**
	 * Generates the code for the musket print function.
	 * This function cannot be called within expressions.
	 * 
	 * @param mfc the musket function call
	 * @return the generated code
	 */
	def static generatePrint(MusketFunctionCall mfc) '''
		«IF Config.processes > 1»
			if(«Config.var_pid» == 0){
		«ENDIF»
			printf«FOR p : mfc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateExpression(null))»«ENDFOR»;
		«IF Config.processes > 1»
			}
		«ENDIF»
	'''

	/**
	 * Generates the code for the musket rand function.
	 * This function can be called within expressions, but it calls the omp procedure 'omp_get_thread_num()' multiple times, which could be avoided by storing the value before.
	 * But then it is not a one-liner anymore.
	 * 
	 * @param mfc the musket function call
	 * @return the generated code
	 */
	def static generateRand(
		MusketFunctionCall mfc) '''rand_dist_«mfc.params.head.calculateType.cppType»_«mfc.params.head.ValueAsString.replace('.', '_')»_«mfc.params.get(1).ValueAsString.replace('.', '_')»[«IF Config.cores > 1»omp_get_thread_num()«ELSE»0«ENDIF»](«Config.var_rng_array»[«IF Config.cores > 1»omp_get_thread_num()«ELSE»0«ENDIF»])'''

	/**
	 * Generates the code for the musket roi start function. (Region of Interest)
	 * This function cannot be called within expressions.
	 * It starts a timer to measure the execution time of the roi.
	 * 
	 * @param mfc the musket function call
	 * @return the generated code
	 */
	def static generateRoiStart(MusketFunctionCall mfc) '''
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	'''

	/**
	 * Generates the code for the musket roi end function. (Region of Interest)
	 * This function cannot be called within expressions.
	 * It stops the timer to measure the execution time of the roi.
	 * Must be called after mkt::roi_start() in the model.
	 * 
	 * @param mfc the musket function call
	 * @return the generated code
	 */
	def static generateRoiEnd(MusketFunctionCall mfc) '''
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	'''

}
