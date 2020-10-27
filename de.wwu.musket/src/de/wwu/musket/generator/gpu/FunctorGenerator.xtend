package de.wwu.musket.generator.gpu

import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.And
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionInstantiation
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.CollectionObjectOrParam
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.ConditionalForLoop
import de.wwu.musket.musket.ControlStructure
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.ExternalFunctionCall
import de.wwu.musket.musket.FloatVal
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.FunctionStatement
import de.wwu.musket.musket.IfClause
import de.wwu.musket.musket.IndividualObject
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.IteratorForLoop
import de.wwu.musket.musket.LeftShift
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Modulo
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PreDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.RightShift
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.Statement
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.StructVariable
import de.wwu.musket.musket.Subtraction
import de.wwu.musket.musket.TypeCast
import de.wwu.musket.musket.Variable

import static extension de.wwu.musket.generator.gpu.CollectionFunctionsGenerator.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.gpu.ExternalFunctionCallGenerator.*
import static extension de.wwu.musket.generator.gpu.MusketFunctionCalls.*
import static extension de.wwu.musket.generator.gpu.util.DataHelper.*
import static extension de.wwu.musket.util.CollectionHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import java.util.List
import java.util.Set
import de.wwu.musket.musket.MusketFunctionName
import de.wwu.musket.musket.ConditionalWhileLoop

class FunctorGenerator {
	
	static final Logger logger = LogManager.getLogger(FunctorGenerator)
	
	def static generateFunctorInstantiation(SkeletonExpression se, SkeletonParameterInput spi, int processId) '''
		«val f = spi.toFunction»
		«val referencedCollections = f.referencedCollections»
		«se.getFunctorName(spi)» «se.getFunctorObjectName(spi)»{«FOR co : referencedCollections SEPARATOR ", "»«co.name»«ENDFOR»«IF f.containsRandCall && !referencedCollections.empty», «ENDIF»«IF f.containsRandCall»«Config.var_rns_pointers»«ENDIF»};
	'''

	def static generateFunctor(Function f, String skelName, String coName, int freeParameter, int processId) '''
		«val referencedCollections = f.referencedCollections»
		struct «f.name.toFirstUpper»_«skelName»_«coName»_functor{
			
			«f.name.toFirstUpper»_«skelName»_«coName»_functor(«generateConstructorParameter(f)»)«FOR co : referencedCollections BEFORE " : " SEPARATOR ", "»«co.generateCollectionObjectInitListEntry»«ENDFOR»{
				«IF f.containsRandCall»
					for(int gpu = 0; gpu < «Config.gpus»; gpu++){
					 	_rns_pointers[gpu] = rns_pointers[gpu];
					}
					_rns_index = 0;
				«ENDIF»
			}
			
			~«f.name.toFirstUpper»_«skelName»_«coName»_functor() {}
			
			auto operator()(«FOR p : f.params.drop(freeParameter) SEPARATOR ", "»«p.generateParameter»«ENDFOR»){
				«IF f.containsRandCall»
					size_t local_rns_index  = _gang + _worker + _vector + _rns_index; // this can probably be improved
					local_rns_index  = (local_rns_index + 0x7ed55d16) + (local_rns_index << 12);
					local_rns_index = (local_rns_index ^ 0xc761c23c) ^ (local_rns_index >> 19);
					local_rns_index = (local_rns_index + 0x165667b1) + (local_rns_index << 5);
					local_rns_index = (local_rns_index + 0xd3a2646c) ^ (local_rns_index << 9);
					local_rns_index = (local_rns_index + 0xfd7046c5) + (local_rns_index << 3);
					local_rns_index = (local_rns_index ^ 0xb55a4f09) ^ (local_rns_index >> 16);
					local_rns_index = local_rns_index % «Config.number_of_random_numbers»;
					_rns_index++;
				«ENDIF»
				«FOR s : f.statement»
					«s.generateFunctionStatement(processId)»
				«ENDFOR»
			}

			void init(int gpu){
				«FOR co : referencedCollections»
					«co.generateInitFunctionCall»;
				«ENDFOR»
				«IF f.containsRandCall»
					_rns = _rns_pointers[gpu];
					std::random_device rd{};
					std::mt19937 d_rng_gen(rd());
					std::uniform_int_distribution<> d_rng_dis(0, «Config.number_of_random_numbers»);
					_rns_index = d_rng_dis(d_rng_gen);
				«ENDIF»
			}
			
			void set_id(int gang, int worker, int vector){
				_gang = gang;
				_worker = worker;
				_vector = vector;
			}
			
			«FOR p : f.params.take(freeParameter)»
				«p.generateMember»;
			«ENDFOR»
			
			«FOR co : referencedCollections»
				«co.generateCollectionMember»;
			«ENDFOR»
			
			«IF f.containsRandCall»
				float* _rns;
				std::array<float*, «Config.gpus»> _rns_pointers;
				size_t _rns_index;
			«ENDIF»
			
			int _gang;
			int _worker;
			int _vector;
		};
	'''
	
	def static generateConstructorParameter(Function f)'''«val referencedCollections = f.referencedCollections»«FOR co : referencedCollections SEPARATOR ", "»«co.generateCollectionObjectConstructorArgument»«ENDFOR»«IF f.containsRandCall && !referencedCollections.empty», «ENDIF»«IF f.containsRandCall»std::array<float*, «Config.gpus»> «Config.var_rns_pointers»«ENDIF»'''
	
	def static generateParameter(de.wwu.musket.musket.Parameter p)'''«IF p.const»const «ENDIF»«p.calculateType.cppType.replace("0", p.calculateType.collectionType?.size.toString)»«IF p.reference»&«ENDIF» «p.name»'''
	def static generateMember(de.wwu.musket.musket.Parameter p)'''«p.calculateType.cppType.replace("0", p.calculateType.collectionType?.size.toString)» «p.name»'''

	def static containsRandCall(Function function){
		function.eAllContents.exists[it instanceof MusketFunctionCall && (it as MusketFunctionCall).value === MusketFunctionName.RAND]
	}

	def static getReferencedCollections(Function function){
		function.eAllContents.filter(ObjectRef).filter[it.collectionElementRef].map[it.value as CollectionObject].toSet
	}
	
	def static generateCollectionObjectConstructorArgument(CollectionObject co)'''const «co.calculateType.cppType.replace("0", co.calculateType.collectionType?.size.toString)»& _«co.name»'''
	def static generateCollectionObjectInitListEntry(CollectionObject co)'''«co.name»(_«co.name»)'''
	
	def static generateInitFunctionCall(CollectionObject co)'''«co.name».init(gpu)'''
	
	def static generateCollectionMember(CollectionObject co)'''mkt::Device«IF co.calculateType.isArray»Array«ELSE»Matrix«ENDIF»<«co.calculateCollectionType.cppType»> «co.name»'''
	
	def static generateFunction(Function f, int processId) '''
		// generate Function
		auto «f.name.toFirstLower»_function(«FOR p : f.params SEPARATOR ", "»«p.generateParameter»«ENDFOR»){
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
			«targetName»[«assignment.^var.value.collectionType.convertLocalCollectionIndex(assignment.^var.localCollectionIndex, null)»]«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(processId)»;
		«««	collection with global ref
		«ELSEIF !assignment.^var.globalCollectionIndex.nullOrEmpty»
			«targetName»[«assignment.^var.value.collectionType.convertGlobalCollectionIndex(assignment.^var.globalCollectionIndex, null)»]«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(processId)»;
		«««	no collection ref
		«ELSE»
			«targetName»«assignment.^var?.tail.generateTail» «assignment.operator» «assignment.value.generateExpression(processId)»;
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
		return «returnStatement.value.generateExpression(processId)»;
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
		«variable.calculateType.cppType» «variable.name»«IF variable.initExpression !== null» = «variable.initExpression.generateExpression(processId)»«ELSEIF variable instanceof StructVariable && (variable as StructVariable).copyFrom !== null»{«(variable as StructVariable).copyFrom.value.name»}«ENDIF»;
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
		for(«cfl.init.calculateType.cppType» «cfl.init.name» = «cfl.init.initExpression.generateExpression(processId)»; «cfl.condition.generateExpression(processId)»; «cfl.increment.generateExpression(processId)»){
			«FOR statement : cfl.statements»
				«statement.generateFunctionStatement(processId)»
			«ENDFOR»
		}
	'''
	def static dispatch generateControlStructure(ConditionalWhileLoop cfl, int processId) '''
		while( «cfl.condition.generateExpression(processId)» «FOR condition : cfl.moreconditions»«condition.connection.generateConnection()» «condition.furtherconditions.generateExpression(processId)»«ENDFOR»){
				«FOR statement : cfl.statements»
					«statement.generateFunctionStatement(processId)»
				«ENDFOR»
		}
	'''
	def static generateConnection(String c) '''«IF c.equals('and')»	&&	«ENDIF»	«IF c.equals('or')» || «ENDIF»'''
	
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
			if(«ifs.condition.generateExpression(processId)»){
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
	
	def static String generateExpression(Expression expression, int processId) {
		switch expression {
			Addition: '''(«expression.left.generateExpression(processId)» + «expression.right.generateExpression(processId)»)'''
			Subtraction: '''(«expression.left.generateExpression(processId)» - «expression.right.generateExpression(processId)»)'''
			Multiplication: '''(«expression.left.generateExpression(processId)» * «expression.right.generateExpression(processId)»)'''
			Division: '''(«expression.left.generateExpression(processId)» / «expression.right.generateExpression(processId)»)'''
			LeftShift: '''(«expression.left.generateExpression(processId)» << «expression.right.generateExpression(processId)»)'''
			RightShift: '''(«expression.left.generateExpression(processId)» >> «expression.right.generateExpression(processId)»)'''
			CompareExpression case expression.eqRight === null: '''«expression.eqLeft.generateExpression(processId)»'''
			CompareExpression case expression.eqRight !==
				null: '''(«expression.eqLeft.generateExpression(processId)» «expression.op» «expression.eqRight.generateExpression(processId)»)'''
			SignedArithmetic: '''-(«expression.expression.generateExpression(processId)»)'''
			Modulo: '''(«expression.left.generateExpression(processId)» % «expression.right.generateExpression(processId)»)'''
			Not: '''!«expression.expression.generateExpression(processId)»'''
			And: '''(«expression.leftExpression.generateExpression(processId)» && «expression.rightExpression.generateExpression(processId)»)'''
			Or: '''(«expression.leftExpression.generateExpression(processId)» || «expression.rightExpression.generateExpression(processId)»)'''
			ObjectRef case expression.isCollectionElementRef: '''«expression.generateCollectionElementRef(processId).toString.removeLineBreak»'''
			ObjectRef: '''(«expression.value.generateObjectRef()»)«expression?.tail.generateTail»'''
			CollectionInstantiation: '''«expression.generateCollectionInstantiation»'''
			IntVal: '''«expression.value»'''
			DoubleVal: '''«expression.value»'''
			FloatVal: '''«expression.value»f'''
			StringVal: '''"«expression.value.replaceAll("\n", "\\\\n").replaceAll("\t", "\\\\t")»"''' // this is necessary so that the line break remains as \n in the generated code
			BoolVal: '''«expression.value»'''
			ExternalFunctionCall: '''«expression.generateExternalFunctionCall(null, processId)»'''
			CollectionFunctionCall: '''«expression.generateCollectionFunctionCall(processId)»'''
			PostIncrement: '''«expression.value.generateObjectRef()»++'''
			PostDecrement: '''«expression.value.generateObjectRef()»--'''
			PreIncrement: '''++«expression.value.generateObjectRef()»'''
			PreDecrement: '''--«expression.value.generateObjectRef()»'''
			MusketFunctionCall: '''«expression.generateMusketFunctionCall(processId)»'''
			TypeCast: '''static_cast<«expression.targetType.calculateType.cppType»>(«expression.expression.generateExpression(processId)»)'''
			default: {logger.error("Functor Expression Generator ran into default case!"); ""}
		}
	}

/**
 * Generate a reference to a collection element.
 * The function considers different cases, based on:
 * array or matrix
 * global or local index
 * distributed or copy
 * 
 * @param or the object ref object 
 * @param param_map the param map
 * @return the generated code
 */
	def static generateCollectionElementRef(ObjectRef or, int processId)'''
		«val orName = or.value.name»
«««		ARRAY
		«IF or.value.calculateType.isArray»
«««			LOCAL REF
			«IF or.localCollectionIndex.size == 1»
				«orName».get_data_local(«or.localCollectionIndex.head.generateExpression(processId)»)«or?.tail.generateTail»
«««			GLOBAL REF
			«ELSE»
«««				COPY or LOC
				«IF (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.LOC»
					«orName»[«or.globalCollectionIndex.head.generateExpression(processId)»]«or?.tail.generateTail»
«««             COPY or LOC
				«ELSEIF (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.COPY »
					«orName».get_data_local(«or.globalCollectionIndex.head.generateExpression(processId)»)«or?.tail.generateTail»
«««				DIST
				«ELSE»
					// TODO: ExpressionGenerator.generateCollectionElementRef: Array, global indices, distributed
				«ENDIF»
			«ENDIF»
«««		MATRIX
		«ELSEIF or.value.calculateType.isMatrix»
«««			LOCAL REF
			«IF or.localCollectionIndex.size == 2»
				«orName».get_data_local(«or.localCollectionIndex.head.generateExpression(processId)», «or.localCollectionIndex.drop(1).head.generateExpression(processId)»)«or?.tail.generateTail»
«««			GLOBAL REF
			«ELSEIF or.globalCollectionIndex.size == 2»
«««					COPY
					«IF (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.COPY || (or.value as CollectionObjectOrParam).collectionType.distributionMode == DistributionMode.LOC»
						«orName»[«or.localCollectionIndex.head.generateExpression(processId)» * «(or.value.collectionType as MatrixType).colsLocal» + «or.localCollectionIndex.drop(1).head.generateExpression(processId)»]«or?.tail.generateTail»
«««					DIST
					«ELSE»
						//TODO: ExpressionGenerator.generateCollectionElementRef: Matrix, global indices, distributed
					«ENDIF»				
			«ENDIF»
		«ELSE»
			(«orName»)«or?.tail.generateTail»
		«ENDIF»
	'''


// dispatch methods for generation of OjbectReference

	def static dispatch generateObjectRef(CollectionObject co) '''«co.name»'''

	def static dispatch generateObjectRef(IndividualObject i) '''«i.name»'''

	def static dispatch generateObjectRef(de.wwu.musket.musket.Parameter p) '''«p.name»'''

	def static generateCollectionInstantiation(CollectionInstantiation ci)'''«val type = ci.calculateType»«IF type.isArray && type.distributionMode == DistributionMode.LOC»«type.cppType»{}«ENDIF»'''
	
}
