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
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.GatherSkeleton

class FunctionGenerator {
	def static generateInternalFunctionCallForSkeleton(InternalFunctionCall ifc, Skeleton skeleton, Array a,
		Map<String, String> param_map) '''
		«FOR s : ifc.value.statement»
			«s.generateFunctionStatement(skeleton, a, param_map)»
		«ENDFOR»
	'''

	def static generateFunctionStatement(FunctionStatement functionStatement, Skeleton skeleton, Array a,
		Map<String, String> param_map) {
		switch functionStatement {
			ControlStructure: ''''''
			Statement:
				functionStatement.generateStatement(skeleton, a, param_map)
			default: ''''''
		}
	}

	def static dispatch generateStatement(Assignment assignment, Skeleton skeleton, Array a,
		Map<String, String> param_map) '''
	'''

	def static dispatch generateStatement(ReturnStatement returnStatement, Skeleton skeleton, Array a,
		Map<String, String> param_map) '''
	«getReturnString(returnStatement, skeleton, param_map)»«returnStatement.value.generateExpression(param_map)»;
	'''

	def static dispatch generateStatement(Variable variable, Skeleton skeleton, Array a,
		Map<String, String> param_map) '''
	'''

	def static dispatch generateStatement(FunctionCall functionCall, Skeleton skeleton, Array a,
		Map<String, String> param_map) '''
	'''

	// helper
	def static String getReturnString(ReturnStatement returnStatement, Skeleton skeleton,
		Map<String, String> param_map) {
		val params = (returnStatement.eContainer as Function).params

		switch skeleton {
			MapSkeleton: param_map.get(params.get(params.size - 1).name) + ' = '
			MapInPlaceSkeleton: param_map.get(params.get(params.size - 1).name) + ' = '
			FoldSkeleton: param_map.get(params.get(params.size - 2).name) + ' = '
			GatherSkeleton: '''GATHER!!!'''
			default: '''return '''
		}

	}
}
