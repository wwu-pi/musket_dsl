package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.util.TypeHelper.*
import java.util.List
import de.wwu.musket.musket.MusketFunctionName

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
	def static generateMusketFunctionCall(MusketFunctionCall mfc, int processId) {
		switch mfc.value {
			case PRINT:
				generatePrint(mfc, processId)
			case RAND:
				generateRand(mfc)
			case SQRT:
				'''std::sqrt(«mfc.params.get(0).generateExpression(null, processId)», «mfc.params.get(1).generateExpression(null, processId)»)'''
			case POW:
				'''std::pow(«mfc.params.head.generateExpression(null, processId)»)'''
			case FLOAT_MIN: '''std::numeric_limits<float>::lowest()'''
			case FLOAT_MAX: '''std::numeric_limits<float>::max()'''
			case DOUBLE_MIN: '''std::numeric_limits<double>::lowest()'''
			case DOUBLE_MAX: '''std::numeric_limits<double>::max()'''
			case ROI_START:
				generateRoiStart(mfc, processId)
			case ROI_END:
				generateRoiEnd(mfc, processId)
			case TIMER_START:
				generateTimerStart(mfc, processId)
			case TIMER_STOP:
				generateTimerStop(mfc, processId)
			case TIMER_RESUME:
				generateTimerResume(mfc, processId)
			case TIMER_SHOW:
				generateTimerShow(mfc, processId)
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
	def static generatePrint(MusketFunctionCall mfc, int processId) '''
		«IF processId == 0»			
			printf«FOR p : mfc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateExpression(null, processId))»«ENDFOR»;
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
		MusketFunctionCall mfc) '''rand_dist_«mfc.params.head.calculateType.cppType»_«mfc.params.head.ValueAsString.toCXXIdentifier»_«mfc.params.get(1).ValueAsString.toCXXIdentifier»[«IF Config.cores > 1»omp_get_thread_num()«ELSE»0«ENDIF»](«Config.var_rng_array»[«IF Config.cores > 1»omp_get_thread_num()«ELSE»0«ENDIF»])'''

	/**
	 * Generates the code for the musket roi start function. (Region of Interest)
	 * This function cannot be called within expressions.
	 * It starts a timer to measure the execution time of the roi.
	 * 
	 * @param mfc the musket function call
	 * @return the generated code
	 */
	def static generateRoiStart(MusketFunctionCall mfc, int processId) '''
		«IF processId == 0»
			std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		«ENDIF»
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
	 def static generateRoiEnd(MusketFunctionCall mfc, int processId) '''
		«IF processId == 0»
			std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
			double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		«ENDIF»
	'''
	
	// Functions to generate Timer
	
	def static generateAllTimerGlobalVars(List<MusketFunctionCall> mfcs, int processId){
		var result = '' 
		var timers = newArrayList
		for(mfc : mfcs.filter[it.value == MusketFunctionName::TIMER_START || it.value == MusketFunctionName::TIMER_RESUME || it.value == MusketFunctionName::TIMER_STOP || it.value == MusketFunctionName::TIMER_SHOW]){
			val name = mfc.params.head.generateExpression(null, processId).toCXXIdentifier
			if(!timers.contains(name)){
				result += generateTimerGlobalVars(mfc, processId)
				timers.add(name)
			}
		}
		return result
	}
	
	def static generateTimerGlobalVars(MusketFunctionCall mfc, int processId)'''
		«val name = mfc.params.head.generateExpression(null, processId).toCXXIdentifier»
		std::chrono::high_resolution_clock::time_point «name»_start;
		std::chrono::high_resolution_clock::time_point «name»_end;
		double «name»_elapsed = 0.0;
	'''
	
	def static generateTimerStart(MusketFunctionCall mfc, int processId) '''
		«val name = mfc.params.head.generateExpression(null, processId).toCXXIdentifier»
		«IF processId == 0»
			«name»_elapsed = 0.0;
			«name»_start = std::chrono::high_resolution_clock::now();
		«ENDIF»
	'''
	
	def static generateTimerStop(MusketFunctionCall mfc, int processId) '''
		«val name = mfc.params.head.generateExpression(null, processId).toCXXIdentifier»
		«IF processId == 0»
			«name»_end = std::chrono::high_resolution_clock::now();
			«name»_elapsed += std::chrono::duration<double>(«name»_end - «name»_start).count();
		«ENDIF»
	'''
	
	def static generateTimerResume(MusketFunctionCall mfc, int processId) '''
		«val name = mfc.params.head.generateExpression(null, processId).toCXXIdentifier»
		«IF processId == 0»
			«name»_start = std::chrono::high_resolution_clock::now();
		«ENDIF»
	'''
	
	def static generateTimerShow(MusketFunctionCall mfc, int processId) '''
		«val name = mfc.params.head.generateExpression(null, processId).toCXXIdentifier»
		«IF processId == 0»
			printf("Elapsed time in seconds for timer %s: %.5f\n", "«name»", «name»_elapsed);
		«ENDIF»
	'''
	
	

}
