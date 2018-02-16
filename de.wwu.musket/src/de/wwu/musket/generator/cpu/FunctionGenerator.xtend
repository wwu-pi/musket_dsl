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
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.util.CollectionHelper.*
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.ConditionalForLoop
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.IfClause
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.ReferableObject
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton

import static extension de.wwu.musket.util.CollectionHelper.*

class FunctionGenerator {
	def static generateInternalFunctionCallForSkeleton(InternalFunctionCall ifc, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
		«FOR s : ifc.value.statement»
			«s.generateFunctionStatement(skeleton, a, param_map)»
		«ENDFOR»
	'''

	def static CharSequence generateFunctionStatement(FunctionStatement functionStatement, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) {
		switch functionStatement {
			ControlStructure: functionStatement.generateControlStructure(skeleton, a, param_map)
			Statement:
				functionStatement.generateStatement(skeleton, a, param_map)
			default: '''//TODO: FunctionGenerator.generateFunctionStatement: Default Case'''
		}
	}


// Statements
	def static dispatch generateStatement(Assignment assignment, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
		«IF param_map.containsKey(assignment.^var.value.name)»«param_map.get(assignment.^var.value.name)»«ELSE»«assignment.^var.value.name»«ENDIF»«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(param_map)»;
	'''

	def static dispatch generateStatement(ReturnStatement returnStatement, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
	«getReturnString(returnStatement, skeleton, param_map)»«returnStatement.value.generateExpression(param_map)»;
	'''

	def static dispatch generateStatement(Variable variable, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
		«variable.CppPrimitiveTypeAsString» «variable.name»«IF variable.initExpression !== null» = «variable.initExpression.generateExpression(param_map)»«ENDIF»;
	'''

	def static dispatch generateStatement(FunctionCall functionCall, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
		//TODO: FunctionGenerator.generateStatement: FunctionCall
	'''

	// helper
	def static String getReturnString(ReturnStatement returnStatement, Skeleton skeleton,
		Map<String, String> param_map) {
		val params = (returnStatement.eContainer as Function).params

		switch skeleton {
			MapSkeleton: param_map.get(params.get(params.size - 1).name) + ' = '
			MapInPlaceSkeleton: param_map.get(params.get(params.size - 1).name) + ' = '
			MapIndexInPlaceSkeleton: param_map.get(params.get(params.size - 1).name) + ' = '
			MapLocalIndexInPlaceSkeleton: param_map.get(params.get(params.size - 1).name) + ' = '
			FoldSkeleton: param_map.get(params.get(params.size - 2).name) + ' = '
			GatherSkeleton: '''GATHER!!!'''
			ShiftPartitionsHorizontallySkeleton: '''«Config.var_shift_steps» = '''
			ShiftPartitionsVerticallySkeleton: '''«Config.var_shift_steps» = '''
			default: '''return '''
		}
	}
	
	def static generateObjectRef(ObjectRef or, Map<String, String> param_map) {
		switch or {
			case or.isCollectionRef: '''«or.generateCollectionElementRef(param_map)»'''
			ReferableObject: '''//TODO: FunctionGenerator.generateObjectRef: ReferableObject'''
			default: '''//TODO: FunctionGenerator.generateObjectRef: default case'''
		}
	}
	
	def static generateCollectionElementRef(ObjectRef cer, Map<String, String> param_map) {
		switch cer.value {		
			Array: '''//TODO: FunctionGenerator.generateCollectionElementRef: array'''
			Matrix: '''//TODO: FunctionGenerator.generateCollectionElementRef: matrix'''
			default: '''//TODO: FunctionGenerator.generateCollectionElementRef: default case'''
		}
	}
	
	// ControlStructures	
	def static dispatch generateControlStructure(ConditionalForLoop cfl, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
		for(«cfl.init.CppPrimitiveTypeAsString» «cfl.init.name» = «cfl.init.initExpression.generateExpression(null)»; «cfl.condition.generateExpression(null)»; «cfl.increment.generateExpression(null)»){
			«FOR statement : cfl.statements»
				«statement.generateFunctionStatement(skeleton, a, param_map)»
			«ENDFOR»
		}
	'''
	
	def static dispatch generateControlStructure(IteratorForLoop ifl, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
		//TODO: FunctionGenerator.generateControlStructure: IteratorForLoop
	'''
	
	def static dispatch generateControlStructure(IfClause ic, Skeleton skeleton, CollectionObject a,
		Map<String, String> param_map) '''
		if(«ic.condition.generateExpression(param_map)»){
			«FOR s : ic.statements»
				«s.generateFunctionStatement(skeleton, a, param_map)»
			«ENDFOR»
		} «IF !ic.elseStatements.empty» else {
			«FOR es : ic.elseStatements»
				«es.generateFunctionStatement(skeleton, a, param_map)»
			«ENDFOR»
		}
		«ENDIF»
	'''
}
