package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.MusketFunctionCall
import static extension de.wwu.musket.generator.cpu.Parameter.*

class MusketFunctionCalls {
	def static generateMusketFunctionCall(MusketFunctionCall mfc){
		switch mfc.value{
			case PRINT: generatePrint(mfc)
			default: ''''''
		}
	}
	
	def static generatePrint(MusketFunctionCall mfc)'''
		if(«Config.var_pid» == 0){
			printf«FOR p : mfc.params BEFORE '(' SEPARATOR ',' AFTER ')'»«(p.generateParameterInput)»«ENDFOR»;
		}
	'''	
}