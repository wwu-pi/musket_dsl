package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.BoolArrayType
import de.wwu.musket.musket.BoolMatrixType
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DoubleArrayType
import de.wwu.musket.musket.DoubleMatrixType
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.IntArrayType
import de.wwu.musket.musket.IntMatrixType
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.RegularFunction
import de.wwu.musket.musket.SkeletonExpression
import java.util.HashMap
import java.util.List
import java.util.Map
import org.eclipse.emf.ecore.resource.Resource

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
import static extension de.wwu.musket.generator.cpu.FunctionGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.FloatArrayType
import de.wwu.musket.musket.FloatMatrixType

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
	 * only once for each function that is used in a fold skeleton and only if the return type
	 * and the input types for both values is the same, otherwise reduction is done without
	 * declare reduction.
	 * 
	 * @param sleketons list of all skeleton expressions.
	 * @return generated code
	 */
	def static generateReductionDeclarations(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton && (se.skeleton as FoldSkeleton).identity.calculateType == ((se.skeleton as FoldSkeleton).param as InternalFunctionCall).value.params.last?.calculateType) {
				val alreadyProcessed = processed.exists [
					((it.skeleton.param as InternalFunctionCall).value as RegularFunction).name ==
						((se.skeleton.param as InternalFunctionCall).value as RegularFunction).name
				]

				if (!alreadyProcessed) {
					result += generateReductionDeclaration(se.skeleton as FoldSkeleton, se.obj)
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
	def static generateReductionDeclaration(FoldSkeleton s, CollectionObject a) '''
		«val param_map_red = createParameterLookupTableFoldReductionClause((s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
		#pragma omp declare reduction(«((s.param as InternalFunctionCall).value as RegularFunction).name» : «a.calculateCollectionType.cppType» : omp_out = [&](){«((s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(null, a, param_map_red)).toString.removeLineBreak»}()) initializer(omp_priv = omp_orig)
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
			if (se.skeleton instanceof FoldSkeleton) {
				val alreadyProcessed = processed.exists [
					((it.skeleton.param as InternalFunctionCall).value as RegularFunction).name ==
						((se.skeleton.param as InternalFunctionCall).value as RegularFunction).name
				]

				if (!alreadyProcessed) {
					result += generateMPIFoldFunction(se.skeleton as FoldSkeleton, se.obj)
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
	def static generateMPIFoldFunction(FoldSkeleton foldSkeleton, CollectionObject a) '''
		void «((foldSkeleton.param as InternalFunctionCall).value as RegularFunction).name»(void *in, void *inout, int *len, MPI_Datatype *dptr){
			«val type = a.calculateCollectionType.cppType»
			«type»* inv = static_cast<«type»*>(in);
			«type»* inoutv = static_cast<«type»*>(inout);
			«val param_map = createParameterLookupTable((foldSkeleton.param as InternalFunctionCall).value.params, (foldSkeleton.param as InternalFunctionCall).params)»
			«(foldSkeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(foldSkeleton, a, param_map)»
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
			if (se.skeleton instanceof FoldSkeleton) {
				val alreadyProcessed = processed.exists [
					((it.skeleton.param as InternalFunctionCall).value as RegularFunction).name ==
						((se.skeleton.param as InternalFunctionCall).value as RegularFunction).name
				]

				if (!alreadyProcessed) {
					result += generateMPIFoldOperator(se.skeleton as FoldSkeleton)
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
	def static generateMPIFoldOperator(FoldSkeleton s) '''
		«val name = ((s.param as InternalFunctionCall).value as RegularFunction).name»
		MPI_Op «name»«Config.mpi_op_suffix»;
		MPI_Op_create( «name», 0, &«name»«Config.mpi_op_suffix» );
	'''

	/**
	 * Generates required temporary variables used to store intermediate fold results.
	 * 
	 * TODO: only supports int, double, float, and bool at the moment
	 * 
	 * @param resource the resource object
	 * @return generated code
	 */
	def static generateTmpFoldResults(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton) {
				val alreadyProcessed = processed.exists [
					it.obj.calculateCollectionType.cppType == se.obj.calculateCollectionType.cppType
				]

				if (!alreadyProcessed) {
					val obj = se.obj
					switch obj.type {
						IntArrayType,
						IntMatrixType:
							result +=
								'''«obj.calculateCollectionType.cppType» «Config.var_fold_result»_«obj.calculateCollectionType.cppType» = 0;'''
						BoolArrayType,
						BoolMatrixType:
							result +=
								'''«obj.calculateCollectionType.cppType» «Config.var_fold_result»_«obj.calculateCollectionType.cppType» = true;'''
						DoubleArrayType,
						DoubleMatrixType:
							result +=
								'''«obj.calculateCollectionType.cppType» «Config.var_fold_result»_«obj.calculateCollectionType.cppType» = 0.0;'''
						FloatArrayType,
						FloatMatrixType:
							result +=
								'''«obj.calculateCollectionType.cppType» «Config.var_fold_result»_«obj.calculateCollectionType.cppType» = 0.0f;'''
					}
					processed.add(se)
				}
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
		return param_map
	}
}
