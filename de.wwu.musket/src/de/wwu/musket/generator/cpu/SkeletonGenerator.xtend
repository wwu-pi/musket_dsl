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
import static extension de.wwu.musket.generator.cpu.Parameter.*
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.Matrix
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode

class SkeletonGenerator {
	def static generateSkeletonExpression(SkeletonExpression s, String target) {
		switch s.skeleton {
			MapInPlaceSkeleton: generateMapInPlaceSkeleton(s)
			MapIndexInPlaceSkeleton: generateMapIndexInPlaceSkeleton(s, s.obj)
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
	
	def static Map<String, String> createParameterLookupTable(CollectionObject co, Iterable<Parameter> parameters,
		Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		if (parameters.length > inputs.size) {
			param_map.put(parameters.drop(inputs.size).head.name, '''«co.name»[«Config.var_loop_counter»]''')
		}

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}
	
	def static Map<String, String> createParameterLookupTableMapIndexSkeleton(CollectionObject co, Iterable<Parameter> parameters,
		Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''(«Config.var_row_offset» + «Config.var_loop_counter_rows»)''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''(«Config.var_col_offset» + «Config.var_loop_counter_cols»)''')
		param_map.put(parameters.drop(inputs.size + 2).head.name, '''«co.name»[«Config.var_loop_counter_rows» * «co.» + «Config.var_loop_counter_cols»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateParameterInput.toString)
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

// MapIndexInPlace
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, Array a) '''
	// TODO mapIndexInPlace for array
	'''
	
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, Matrix m) '''
		«IF m.distributionMode == DistributionMode.COPY»
			«Config.var_row_offset» = 0;
			«Config.var_col_offset» = 0;
		«ELSE»
			«FOR p : 0..<Config.processes BEFORE 'if' SEPARATOR 'else if' AFTER ''»
				(«Config.var_pid» == «p»){
					«Config.var_row_offset» = «p / m.blocksInColumn * m.rowsLocal»;
					«Config.var_col_offset» = «p % m.blocksInRow * m.colsLocal»;
				}
			«ENDFOR»
		«ENDIF»
		«««	create lookup table for parameters
		«val param_map = createParameterLookupTable(m, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
		#pragma omp parallel for
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			#pragma omp simd
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, m, param_map)»
			}
		}
	'''

// Fold
	def static generateFoldSkeleton(SkeletonExpression s, String target) {
		switch s.obj {
			Array: generateArrayFoldSkeleton(s.skeleton as FoldSkeleton, s.obj as Array, target)
		}
	}

	def static generateArrayFoldSkeleton(FoldSkeleton s, Array a, String target) '''		

		«Config.var_fold_result»_«a.CppPrimitiveTypeAsString»  = «s.identity.ValueAsString»;
		«val foldName = ((s.param as InternalFunctionCall).value as RegularFunction).name»
		
			#pragma omp parallel for reduction(«foldName»:«Config.var_fold_result»_«a.CppPrimitiveTypeAsString»)
			for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«val param_map = createParameterLookupTableFold(a, (s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
			«(s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s, a, param_map)»
		
		}		
		
«««		printf("local fold result %i: %i\n", «Config.var_pid», «Config.var_fold_result»_«a.CppPrimitiveTypeAsString»);
		
		MPI_Allreduce(&«Config.var_fold_result»_«a.CppPrimitiveTypeAsString», &«target», sizeof(«a.CppPrimitiveTypeAsString»), MPI_BYTE, «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
		
«««		printf("global fold result %i: %i\n", «Config.var_pid», «target»);
		
	'''

	def static Map<String, String> createParameterLookupTableFold(Array a, Iterable<Parameter> parameters,
		Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_fold_result»_«a.CppPrimitiveTypeAsString»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«a.name»[«Config.var_loop_counter»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}

	def static Map<String, String> createParameterLookupTableFoldReductionClause(Array a,
		Iterable<Parameter> parameters, Iterable<ParameterInput> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''omp_out''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''omp_in''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).asString)
		}
		return param_map
	}
}
