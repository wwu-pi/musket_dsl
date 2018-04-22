package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.ConditionalForLoop
import de.wwu.musket.musket.ControlStructure
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.FunctionStatement
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.IfClause
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.Skeleton
import de.wwu.musket.musket.Statement
import de.wwu.musket.musket.Variable
import java.util.Map

import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.LambdaFunction
import de.wwu.musket.musket.MapFoldSkeleton

/**
 * Generates a function call.
 * <p>
 * For example, if a map skeleton is called with function f as argument, then this generator generates f as main
 * body of the map skeleton.
 * <p>
 * The entry point is the method generateInternalFunctionCallForSkeleton(InternalFunctionCall ifc, Skeleton skeleton, CollectionObject a,
 * 		Map<String, String> param_map).
 * The methods are mostly called from the skeleton generator.
 */
class FunctionGenerator {

	/**
	 * Generates all statments of a function.
	 * 
	 * @param ifc the internal function call
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param target where the results should end up, important for skeletons, which do not work in place
	 * @param param_map the param_map
	 * @return the generated function statement
	 */
	def static generateFunctionCallForSkeleton(SkeletonParameterInput spi, Skeleton skeleton, CollectionObject a,
		String target, Map<String, String> param_map, int processId) '''
		«val statements = if(spi instanceof LambdaFunction) spi.statement else (spi as InternalFunctionCall).value.statement»
		
		«FOR s : statements»
			«s.generateFunctionStatement(skeleton, a, target, param_map, processId)»
		«ENDFOR»
	'''

	/**
	 * Generates a single function statement. A function statement is either a statement, or a control structure.
	 * 
	 * @param functionStatement the function statement
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param target where the results should end up, important for skeletons, which do not work in place
	 * @param param_map the param_map
	 * @return the generated function call
	 */
	def static CharSequence generateFunctionStatement(FunctionStatement functionStatement, Skeleton skeleton,
		CollectionObject a, String target, Map<String, String> param_map, int processId) {
		switch functionStatement {
			Statement:
				functionStatement.generateStatement(skeleton, a, target, param_map, processId)
			ControlStructure:
				functionStatement.generateControlStructure(skeleton, a, target, param_map, processId)
			default: '''//TODO: FunctionGenerator.generateFunctionStatement: Default Case'''
		}
	}

// Statements
	/**
	 * Generates a assignment.
	 * 
	 * @param assignment the assignment
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param target where the results should end up, important for skeletons, which do not work in place
	 * @param param_map the param_map
	 * @return the generated code
	 */
	def static dispatch generateStatement(Assignment assignment, Skeleton skeleton, CollectionObject a, String target,
		Map<String, String> param_map, int processId) '''
		«val targetName = param_map.getOrDefault(assignment.^var.value.name, null) ?: assignment.^var.value.name»
		«««	collection with local ref	
		«IF !assignment.^var.localCollectionIndex.nullOrEmpty»
			«targetName»[«assignment.^var.value.collectionType.convertLocalCollectionIndex(assignment.^var.localCollectionIndex, param_map)»]«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(param_map, processId)»;
		«««	collection with global ref
		«ELSEIF !assignment.^var.globalCollectionIndex.nullOrEmpty»
			«targetName»[«assignment.^var.value.collectionType.convertGlobalCollectionIndex(assignment.^var.globalCollectionIndex, param_map)»]«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(param_map, processId)»;
		«««	no collection ref
		«ELSE»
			«targetName»«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(param_map, processId)»;
		«ENDIF»
	'''

	/**
	 * Generates a return statement.
	 * If the left and right side of the statement are the same, an empty string is returned
	 * 
	 * @param returnStatement the return statement
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param target where the results should end up, important for skeletons, which do not work in place
	 * @param param_map the param_map
	 * @return the generated code
	 */
	def static dispatch generateStatement(ReturnStatement returnStatement, Skeleton skeleton, CollectionObject a,
		String target, Map<String, String> param_map, int processId) {
		val lhs = getReturnString(returnStatement, skeleton, target, param_map)
		val rhs = returnStatement.value.generateExpression(param_map, processId)
		if (lhs != rhs.replace("(", "").replace(")", "") + ' = ') {
			return lhs + rhs + ';'
		} else {
			return ''
		}
	}

	/**
	 * Generates the string for a single return statement. Used by the method generateStatement(ReturnStatement returnStatement, Skeleton skeleton, CollectionObject a,
	 * 	Map<String, String> param_map)
	 * 
	 * @param returnStatement the return statement
	 * @param skeleton the skeleton in which the function is used
	 * @param param_map the param_map
	 * @return the return string
	 */
	def static String getReturnString(ReturnStatement returnStatement, Skeleton skeleton, String target,
		Map<String, String> param_map) {
		val params = (returnStatement.eContainer as Function).params

		switch skeleton {
			MapSkeleton:
				target + '[' + Config.var_loop_counter + '] = '
			MapInPlaceSkeleton:
				param_map.get(params.get(params.size - 1).name) + ' = '
			MapIndexInPlaceSkeleton:
				param_map.get(params.get(params.size - 1).name) + ' = '
			MapLocalIndexInPlaceSkeleton:
				param_map.get(params.get(params.size - 1).name) + ' = '
			FoldSkeleton:
				param_map.get(params.get(params.size - 2).name) + ' = '
			MapFoldSkeleton: '''«param_map.get("return")» = '''
			GatherSkeleton: ''''''
			ShiftPartitionsHorizontallySkeleton: '''«Config.var_shift_steps» = '''
			ShiftPartitionsVerticallySkeleton: '''«Config.var_shift_steps» = '''
			default: '''return '''
		}
	}

	/**
	 * Generates a variable. The init expression is only generated if it is not null.
	 * 
	 * @param variable the variable
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param param_map the param_map
	 * @return the generated function call
	 */
	def static dispatch generateStatement(Variable variable, Skeleton skeleton, CollectionObject a, String target,
		Map<String, String> param_map, int processId) '''
		«variable.calculateType.cppType» «variable.name»«IF variable.initExpression !== null» = «variable.initExpression.generateExpression(param_map, processId)»«ENDIF»;
	'''

	/**
	 * Generates a function call.
	 * 
	 * TODO: not yet supported. And possibly never fully will, since only external function calls could be allowed here. To be discussed.
	 * 
	 * @param assignment the assignment
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param param_map the param_map
	 * @return the generated function call
	 */
	def static dispatch generateStatement(FunctionCall functionCall, Skeleton skeleton, CollectionObject a,
		String target, Map<String, String> param_map, int processId) '''
		//TODO: FunctionGenerator.generateStatement: FunctionCall
	'''

	// ControlStructures	
	/**
	 * Generates a conditional for loop.
	 *  
	 * @param cfl the for loop
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param param_map the param_map
	 * @return the generated conditional for loop
	 */
	def static dispatch generateControlStructure(ConditionalForLoop cfl, Skeleton skeleton, CollectionObject a,
		String target, Map<String, String> param_map, int processId) '''
		for(«cfl.init.calculateType.cppType» «cfl.init.name» = «cfl.init.initExpression.generateExpression(param_map, processId)»; «cfl.condition.generateExpression(param_map, processId)»; «cfl.increment.generateExpression(param_map, processId)»){
			«FOR statement : cfl.statements»
				«statement.generateFunctionStatement(skeleton, a, target, param_map, processId)»
			«ENDFOR»
		}
	'''

	/**
	 * Generates a iterator for loop.
	 * 
	 * TODO: not yet implemented
	 *  
	 * @param ifl the iterator for loop
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param param_map the param_map
	 * @return the generated iterator for loop
	 */
	def static dispatch generateControlStructure(IteratorForLoop ifl, Skeleton skeleton, CollectionObject a,
		String target, Map<String, String> param_map, int processId) '''
		//TODO: FunctionGenerator.generateControlStructure: IteratorForLoop
	'''

	/**
	 * Generates a if clause.
	 *  
	 * @param ic the if clause
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param param_map the param_map
	 * @return the generated if clause
	 */
	def static dispatch generateControlStructure(IfClause ic, Skeleton skeleton, CollectionObject a, String target,
		Map<String, String> param_map, int processId) '''
		
		«FOR ifs : ic.ifClauses SEPARATOR "\n} else " AFTER "}"»
			if(«ifs.condition.generateExpression(param_map, processId)»){
			«FOR statement: ifs.statements»
				«statement.generateFunctionStatement(skeleton, a, target, param_map, processId)»
			«ENDFOR»
		«ENDFOR»
		
		«IF !ic.elseStatements.empty» else {
			«FOR es : ic.elseStatements»
				«es.generateFunctionStatement(skeleton, a, target, param_map, processId)»
			«ENDFOR»
		}
		«ENDIF»
	'''
}
