package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.ControlStructure
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.FunctionStatement
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.Statement
import de.wwu.musket.musket.Variable
import java.util.Map

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*

class FunctionGenerator {
	def static generateInternalFunctionCallForSkeleton(InternalFunctionCall ifc, Array a, Map<String, String> param_map)'''
		«FOR s : ifc.value.statement»
			«s.generateFunctionStatement(a, param_map)»
		«ENDFOR»
	'''
	
	def static generateFunctionStatement(FunctionStatement functionStatement, Array a, Map<String, String> param_map){
		switch functionStatement{
			ControlStructure: ''''''
			Statement: functionStatement.generateStatement(a, param_map)
			default: ''''''
		}
	}
	
	def static dispatch generateStatement(Assignment assignment, Array a, Map<String, String> param_map)'''
	'''
	
	def static dispatch generateStatement(ReturnStatement returnStatement, Array a, Map<String, String> param_map)'''
		«a.name»[«Config.var_loop_counter»] = «returnStatement.value.generateExpression(param_map)»;
	'''
	
	def static dispatch generateStatement(Variable variable, Array a, Map<String, String> param_map)'''
	'''
	
	def static dispatch generateStatement(FunctionCall functionCall, Array a, Map<String, String> param_map)'''
	'''
}
