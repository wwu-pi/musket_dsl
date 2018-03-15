package de.wwu.musket.generator.cpu

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

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.LambdaFunction

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
		String target, Map<String, String> param_map) '''
		«val statements = if(spi instanceof LambdaFunction) spi.statement else (spi as InternalFunctionCall).value.statement»
		
		«FOR s : statements»
			«s.generateFunctionStatement(skeleton, a, target, param_map)»
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
		CollectionObject a, String target, Map<String, String> param_map) {
		switch functionStatement {
			Statement:
				functionStatement.generateStatement(skeleton, a, target, param_map)
			ControlStructure:
				functionStatement.generateControlStructure(skeleton, a, target, param_map)
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
		Map<String, String> param_map) '''
		«IF param_map.containsKey(assignment.^var.value.name)»«param_map.get(assignment.^var.value.name)»«ELSE»«assignment.^var.value.name»«ENDIF»«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(param_map)»;
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
		String target, Map<String, String> param_map) {
		val lhs = getReturnString(returnStatement, skeleton, target, param_map)
		val rhs = returnStatement.value.generateExpression(param_map)
		if (lhs != rhs + ' = ') {
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
		Map<String, String> param_map) '''
		«variable.calculateType.cppType» «variable.name»«IF variable.initExpression !== null» = «variable.initExpression.generateExpression(param_map)»«ENDIF»;
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
		String target, Map<String, String> param_map) '''
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
		String target, Map<String, String> param_map) '''
		for(«cfl.init.calculateType.cppType» «cfl.init.name» = «cfl.init.initExpression.generateExpression(param_map)»; «cfl.condition.generateExpression(param_map)»; «cfl.increment.generateExpression(param_map)»){
			«FOR statement : cfl.statements»
				«statement.generateFunctionStatement(skeleton, a, target, param_map)»
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
		String target, Map<String, String> param_map) '''
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
		Map<String, String> param_map) '''
		if(«ic.condition.generateExpression(param_map)»){
			«FOR s : ic.statements»
				«s.generateFunctionStatement(skeleton, a, target, param_map)»
			«ENDFOR»
		} «IF !ic.elseStatements.empty» else {
									«FOR es : ic.elseStatements»
										«es.generateFunctionStatement(skeleton, a, target, param_map)»
									«ENDFOR»
			}
		«ENDIF»
	'''
}
