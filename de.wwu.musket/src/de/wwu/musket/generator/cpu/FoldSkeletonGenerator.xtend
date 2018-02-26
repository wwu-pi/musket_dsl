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

class FoldSkeletonGenerator {

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

	def static generateMPIFoldFunction(FoldSkeleton foldSkeleton, CollectionObject a) '''
		void «((foldSkeleton.param as InternalFunctionCall).value as RegularFunction).name»(void *in, void *inout, int *len, MPI_Datatype *dptr){
			«val type = a.calculateCollectionType.cppType»
			«type»* inv = static_cast<«type»*>(in);
			«type»* inoutv = static_cast<«type»*>(inout);
			«val param_map = createParameterLookupTable((foldSkeleton.param as InternalFunctionCall).value.params, (foldSkeleton.param as InternalFunctionCall).params)»
			«(foldSkeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(foldSkeleton, a, param_map)»
		} 
	'''

	def static generateReductionDeclarations(Resource resource) {
		var result = ""
		var List<SkeletonExpression> processed = newArrayList
		for (SkeletonExpression se : resource.SkeletonExpressions) {
			if (se.skeleton instanceof FoldSkeleton) {
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

	def static generateReductionDeclaration(FoldSkeleton s, CollectionObject a) '''
		«val param_map_red = createParameterLookupTableFoldReductionClause((s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
		#pragma omp declare reduction(«((s.param as InternalFunctionCall).value as RegularFunction).name» : «a.calculateCollectionType.cppType» : omp_out = [&](){«((s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(null, a, param_map_red)).toString.removeLineBreak»}()) initializer(omp_priv = omp_orig)
	'''

	def static Map<String, String> createParameterLookupTableFoldReductionClause(Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''omp_out''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''omp_in''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

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

	def static generateMPIFoldOperator(FoldSkeleton s) '''
		«val name = ((s.param as InternalFunctionCall).value as RegularFunction).name»
		MPI_Op «name»«Config.mpi_op_suffix»;
		MPI_Op_create( «name», 0, &«name»«Config.mpi_op_suffix» );
	'''

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
						IntArrayType, IntMatrixType: result +=
							'''«obj.calculateCollectionType.cppType» «Config.var_fold_result»_«obj.calculateCollectionType.cppType» = 0;'''
						BoolArrayType, BoolMatrixType: result +=
							'''«obj.calculateCollectionType.cppType» «Config.var_fold_result»_«obj.calculateCollectionType.cppType» = true;'''
						DoubleArrayType, DoubleMatrixType: result +=
							'''«obj.calculateCollectionType.cppType» «Config.var_fold_result»_«obj.calculateCollectionType.cppType» = 0.0;'''
					}
					processed.add(se)
				}
			}
		}

		return result
	}

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
