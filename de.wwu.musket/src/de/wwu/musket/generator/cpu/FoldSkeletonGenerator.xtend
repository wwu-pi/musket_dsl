package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FoldSkeletonVariants
import de.wwu.musket.musket.MapFoldSkeleton
import de.wwu.musket.musket.SkeletonExpression
import java.util.HashMap
import java.util.List
import java.util.Map
import org.eclipse.emf.ecore.resource.Resource

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.FunctionGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*

/**
 * Generate everything required for the fold skeleton, except for the actual fold skeleton call.
 * That is done in the skeleton generator.
 * <p>
 * The fold skeleton requires two things:
 * First, the OpenMP reduction declaration.
 * http://scc.ustc.edu.cn/zlsc/tc4600/intel/2016.0.109/compiler_c/common/core/GUID-7312910C-D175-4544-99C5-29C12D980744.htm
 * 
 * Second, the MPI fold function and the operator. 
 * http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node107.htm
 * 
 * TODO: some functions take the resource object, some a list of skeleton calls --> unify
 * 
 * The methods are called by the source file generator.
 */
class FoldSkeletonGenerator {

// OpenMP part
	/**
	 * Generates OpenMP reduction declaration.
	 * It iterates through all skeleton statements and generates the reduction declaration
	 * only once for each function that is used in a fold skeleton.
	 * 
	 * @param sleketons list of all skeleton expressions.
	 * @return generated code
	 */
	def static generateReductionDeclarations(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeletonVariants) {
				val alreadyProcessed = processed.exists [
					it.skeleton.param.functionName == se.skeleton.param.functionName
				]

				if (!alreadyProcessed) {
					result += generateReductionDeclaration(se.skeleton as FoldSkeletonVariants, se.obj)
					processed.add(se)
				}
			}
		}

		return result
	}

	/**
	 * Generate single reduction declaration.
	 * 
	 * @param s the fold skeleton
	 * @param a the collection object on which the fold skeleton is used
	 * @return generated code
	 */
	def static generateReductionDeclaration(FoldSkeletonVariants s, CollectionObject a) '''
		«val param_map_red = createParameterLookupTableFoldReductionClause(s.param.functionParameters, s.param.functionArguments)»
		#pragma omp declare reduction(«s.param.functionName» : «s.identity.calculateType.cppType» : omp_out = [&](){«(s.param.generateFunctionCallForSkeleton(null, a, null, param_map_red)).toString.removeLineBreaks»}()) initializer(omp_priv = omp_orig)
	'''

	/**
	 * Create parameter map for OpenMP reduction clause.
	 * 
	 * @param parameters the parameters
	 * @param inputs the input expressions
	 * @return the param map
	 */
	def static Map<String, String> createParameterLookupTableFoldReductionClause(
		Iterable<de.wwu.musket.musket.Parameter> parameters, Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''omp_out''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''omp_in''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

// MPI part
	/**
	 * Generates MPI fold functions. It iterates through all skeleton statements and generates the MPI fold function
	 * only once for each function that is used in a fold skeleton.
	 * 
	 * @param sleketons list of all skeleton expressions.
	 * @return generated code
	 */
	def static generateMPIFoldFunction(Iterable<SkeletonExpression> skeletons) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : skeletons) {
			if (se.skeleton instanceof FoldSkeleton || se.skeleton instanceof MapFoldSkeleton) {
				val alreadyProcessed = processed.exists [
					it.skeleton.param.functionName == se.skeleton.param.functionName
				]

				if (!alreadyProcessed) {
					result += generateMPIFoldFunction(se.skeleton as FoldSkeletonVariants, se.obj)
					processed.add(se)
				}
			}
		}

		return result
	}

	/**
	 * Generates a single MPI fold function.
	 * 
	 * @param foldSkeleton the fold skeleton
	 * @param a the collection object on which the fold skeleton is used.
	 * @return generated code
	 */
	def static generateMPIFoldFunction(FoldSkeletonVariants foldSkeleton, CollectionObject a) '''
		void «foldSkeleton.param.functionName»(void *in, void *inout, int *len, MPI_Datatype *dptr){
			«val type = foldSkeleton.identity.calculateType.cppType»
			«type»* inv = static_cast<«type»*>(in);
			«type»* inoutv = static_cast<«type»*>(inout);
			«val param_map = createParameterLookupTable(foldSkeleton.param.functionParameters, foldSkeleton.param.functionArguments)»
			«foldSkeleton.param.generateFunctionCallForSkeleton(foldSkeleton, a, null, param_map)»
		} 
	'''

	/**
	 * Generates MPI fold Operators. It iterates through all skeleton statements and generates the MPI fold operator
	 * only once for each function that is used in a fold skeleton.
	 * 
	 * @param resource the resource object
	 * @return generated code
	 */
	def static generateMPIFoldOperators(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton || se.skeleton instanceof MapFoldSkeleton) {
				val alreadyProcessed = processed.exists [
					it.skeleton.param.functionName == se.skeleton.param.functionName
				]

				if (!alreadyProcessed) {
					result += generateMPIFoldOperator(se.skeleton as FoldSkeletonVariants)
					processed.add(se)
				}
			}
		}

		return result
	}

	/**
	 * Generates a single MPI fold operator.
	 * 
	 * @param foldSkeleton the fold skeleton
	 * @return generated code
	 */
	def static generateMPIFoldOperator(FoldSkeletonVariants s) '''
		«val name = s.param.functionName»
		MPI_Op «name»«Config.mpi_op_suffix»;
		MPI_Op_create( «name», 0, &«name»«Config.mpi_op_suffix» );
	'''

	/**
	 * Generates required temporary variables used to store intermediate fold results.
	 * 
	 * @param resource the resource object
	 * @return generated code
	 */
	def static generateTmpFoldResults(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton || se.skeleton instanceof MapFoldSkeleton) {
				val alreadyProcessed = processed.exists [
					(it.skeleton as FoldSkeletonVariants).identity.calculateType.cppType ==
						(se.skeleton as FoldSkeletonVariants).identity.calculateType.cppType
				]

				if (!alreadyProcessed) {
					val type = (se.skeleton as FoldSkeletonVariants).identity.calculateType.cppType

					result +=
						'''«type» «Config.var_fold_result»_«type»;'''
				}
				processed.add(se)
			}
		}

		return result
	}

	/**
	 * create parameter map for MPI fold function.
	 * 
	 * @param parameters the parameters
	 * @param inputs the input expressions
	 * @return the param map
	 */
	def static Map<String, String> createParameterLookupTable(Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''*inoutv''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''*inv''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}

		// for mapFoldSkeleton
		param_map.put("return", '''*inoutv''')

		return param_map
	}
}
