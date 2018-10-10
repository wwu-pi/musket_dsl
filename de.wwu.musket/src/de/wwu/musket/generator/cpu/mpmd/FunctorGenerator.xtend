package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.ConditionalForLoop
import de.wwu.musket.musket.ControlStructure
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.FunctionStatement
import de.wwu.musket.musket.IfClause
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.Statement
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.CollectionType
import de.wwu.musket.generator.cpu.mpmd.lib.Musket

class FunctorGenerator {
	
	def static generateFunctorInstantiation(Function f, String skelName, int processId) '''
		«f.name.toFirstUpper»_«skelName»_functor «f.name.toFirstLower»_«skelName»_functor{};
	'''

	def static generateFunctor(Function f, String skelName, int freeParameter, int processId) '''
		struct «f.name.toFirstUpper»_«skelName»_functor{
			auto operator()(«FOR p : f.params.drop(freeParameter) SEPARATOR ", "»«p.generateParameter»«ENDFOR») const{
				«FOR s : f.statement»
					«s.generateFunctionStatement(processId)»
				«ENDFOR»
			}
			
			«FOR p : f.params.take(freeParameter)»
				«p.generateParameter»;
			«ENDFOR»
		};
	'''
	
	def static generateParameter(de.wwu.musket.musket.Parameter p)'''«p.calculateType.cppType.replace("0", p.calculateType.collectionType?.size.toString)» «p.name»'''

	def static generateFunction(Function f, int processId) '''
		// generate Function
		inline auto «f.name.toFirstLower»_function(«FOR p : f.params SEPARATOR ", "»«p.calculateType.cppType.replace("0", p.calculateType.collectionType?.size.toString)» «p.name»«ENDFOR»){
			«FOR s : f.statement»
				«s.generateFunctionStatement(processId)»
			«ENDFOR»
		}
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
	def static CharSequence generateFunctionStatement(FunctionStatement functionStatement, int processId) {
		switch functionStatement {
			Statement:
				functionStatement.generateStatement(processId)
			ControlStructure:
				functionStatement.generateControlStructure(processId)
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
	def static dispatch generateStatement(Assignment assignment, int processId) '''
		«val targetName = assignment.^var.value.name»
		«««	collection with local ref	
		«IF !assignment.^var.localCollectionIndex.nullOrEmpty»
			«targetName»[«assignment.^var.value.collectionType.convertLocalCollectionIndex(assignment.^var.localCollectionIndex, null)»]«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(null, processId)»;
		«««	collection with global ref
		«ELSEIF !assignment.^var.globalCollectionIndex.nullOrEmpty»
			«targetName»[«assignment.^var.value.collectionType.convertGlobalCollectionIndex(assignment.^var.globalCollectionIndex, null)»]«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(null, processId)»;
		«««	no collection ref
		«ELSE»
			«targetName»«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(null, processId)»;
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
	def static dispatch generateStatement(ReturnStatement returnStatement, int processId) '''
		return «returnStatement.value.generateExpression(null, processId)»;
	'''

	/**
	 * Generates a variable. The init expression is only generated if it is not null.
	 * 
	 * @param variable the variable
	 * @param skeleton the skeleton in which the function is used
	 * @param a the collection object on which the skeleton is used
	 * @param param_map the param_map
	 * @return the generated function call
	 */
	def static dispatch generateStatement(Variable variable, int processId) '''
		«variable.calculateType.cppType» «variable.name»«IF variable.initExpression !== null» = «variable.initExpression.generateExpression(null, processId)»«ENDIF»;
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
	def static dispatch generateStatement(FunctionCall functionCall, int processId) '''
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
	def static dispatch generateControlStructure(ConditionalForLoop cfl, int processId) '''
		for(«cfl.init.calculateType.cppType» «cfl.init.name» = «cfl.init.initExpression.generateExpression(null, processId)»; «cfl.condition.generateExpression(null, processId)»; «cfl.increment.generateExpression(null, processId)»){
			«FOR statement : cfl.statements»
				«statement.generateFunctionStatement(processId)»
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
	def static dispatch generateControlStructure(IteratorForLoop ifl, int processId) '''
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
	def static dispatch generateControlStructure(IfClause ic, int processId) '''
		
		«FOR ifs : ic.ifClauses SEPARATOR "\n} else " AFTER "}"»
			if(«ifs.condition.generateExpression(null, processId)»){
			«FOR statement: ifs.statements»
				«statement.generateFunctionStatement(processId)»
			«ENDFOR»
		«ENDFOR»
		
		«IF !ic.elseStatements.empty» else {
				«FOR es : ic.elseStatements»
					«es.generateFunctionStatement(processId)»
				«ENDFOR»
			}
		«ENDIF»
	'''
}
