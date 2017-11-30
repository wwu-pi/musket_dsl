package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Array
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.DoubleArray
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.FunctionCall
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Parameter
import de.wwu.musket.musket.ParameterInput
import de.wwu.musket.musket.RegularFunction
import de.wwu.musket.musket.SkeletonExpression
import java.util.HashMap
import java.util.Map
import java.util.Map.Entry

import static extension de.wwu.musket.generator.cpu.FunctionGenerator.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*

class SkeletonGenerator {
	def static generateSkeletonExpression(SkeletonExpression s, String target) {
		switch s.skeleton {
			MapInPlaceSkeleton: generateMapInPlaceSkeleton(s)
			FoldSkeleton: generateFoldSkeleton(s, target)
			default: ''''''
		}
	}

// MapInPlace
	def static generateMapInPlaceSkeleton(SkeletonExpression s) {
		switch s.obj {
			Array: generateArrayMapInPlaceSkeleton(s, s.obj as Array)
		}
	}

	def static generateArrayMapInPlaceSkeleton(SkeletonExpression s, Array a) '''
		«««	create lookup table for parameters
		«val param_map = createParameterLookupTable(a, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
		«FOR p : param_map.entrySet»
			
		«ENDFOR»
		#pragma omp parallel for
		for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, a, param_map)»
		}
	'''

	def static generateParameter(Entry<String, ParameterInput> set) {
		switch set.value {
			DoubleArray: '''double «set.key» = «(set.value as DoubleArray).name»[«»];'''
			default: ''''''
		}
	}

	def static Map<String, String> createParameterLookupTable(Array a, Iterable<Parameter> parameters,
		Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		if(parameters.length > inputs.size){
			param_map.put(parameters.drop(inputs.size).head.name, '''«a.name»[«Config.var_loop_counter»]''')
		}
		
		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}

	def static String asString(ParameterInput pi) {
		switch pi {
			FunctionCall: '''ERROR FUNCTION CALL'''
			ObjectRef: '''«pi.value»'''
			IntVal: '''«pi.value»'''
			DoubleVal: '''«pi.value»'''
			BoolVal: '''«pi.value»'''
			default: ''''''
		}
	}

// Fold
	def static generateFoldSkeleton(SkeletonExpression s, String target) {
		switch s.obj {
			Array: generateArrayFoldSkeleton(s.skeleton as FoldSkeleton, s.obj as Array, target)
		}
	}

	def static generateArrayFoldSkeleton(FoldSkeleton s, Array a, String target) '''		
		«val param_map_red = createParameterLookupTableFoldReductionClause(a, (s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
			
		#pragma omp declare reduction(«((s.param as InternalFunctionCall).value as RegularFunction).name» : «a.CppPrimitiveTypeAsString» : omp_out = [&](){«((s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(null, a, param_map_red)).toString.removeLineBreak»}()) initializer(omp_priv = omp_orig)
		
		«a.CppPrimitiveTypeAsString» «Config.var_fold_result» = «s.identity.ValueAsString»;
		
			#pragma omp parallel for reduction(«((s.param as InternalFunctionCall).value as RegularFunction).name»:«Config.var_fold_result»)
			for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«val param_map = createParameterLookupTableFold(a, (s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
			«(s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s, a, param_map)»

		}
		
		MPI_Op myOp;
		MPI_Op_create( «((s.param as InternalFunctionCall).value as RegularFunction).name», 0, &myOp );
		
		printf("local fold result %i: %i\n", «Config.var_pid», «Config.var_fold_result»);
		
		MPI_Allreduce(&«Config.var_fold_result», &«target», sizeof(«a.CppPrimitiveTypeAsString»), MPI_BYTE, myOp, MPI_COMM_WORLD); 
		
		printf("global fold result %i: %i\n", «Config.var_pid», «target»);
		
	'''

	def static Map<String, String> createParameterLookupTableFold(Array a, Iterable<Parameter> parameters,
		Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_fold_result»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«a.name»[«Config.var_loop_counter»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}
	
	def static Map<String, String> createParameterLookupTableFoldReductionClause(Array a, Iterable<Parameter> parameters,
		Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''omp_out''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''omp_in''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}
}
