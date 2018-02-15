package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.cpu.Parameter.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*

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
			printf«FOR p : mfc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateParameterInput)»«ENDFOR»;
		}
	'''	
	
	def static generateRand(MusketFunctionCall mfc)'''rand_dist_«mfc.params.head.CppPrimitiveTypeAsString»_«mfc.params.head.ValueAsString.replace('.', '_')»_«mfc.params.get(1).ValueAsString.replace('.', '_')»[omp_get_thread_num()](«Config.var_rng_array»[omp_get_thread_num()])'''	
}