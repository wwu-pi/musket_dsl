package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.MapOption
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.RegularFunction
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.SkeletonExpression
import java.util.HashMap
import java.util.Map

import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIAllgather
import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIIrecv
import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIIsend
import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIWaitall

import static extension de.wwu.musket.generator.cpu.FunctionGenerator.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.util.TypeHelper.*

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*

class SkeletonGenerator {
	
	static var ShiftCounter = 0

	def static generateSkeletonExpression(SkeletonExpression s, String target) {
		switch s.skeleton {
			MapSkeleton: {
				if((s.skeleton as MapSkeleton).options.exists[it == MapOption.LOCAL_INDEX]){
					'''// TODO: Options for skeletons'''
				}
			}
			MapInPlaceSkeleton: generateMapInPlaceSkeleton(s)
			MapIndexInPlaceSkeleton: generateMapIndexInPlaceSkeleton(s, s.obj.type)
			MapLocalIndexInPlaceSkeleton: generateMapLocalIndexInPlaceSkeleton(s, s.obj.type)
			FoldSkeleton: generateFoldSkeleton(s.skeleton as FoldSkeleton, s.obj, target)
			ShiftPartitionsHorizontallySkeleton: generateShiftPartitionsHorizontallySkeleton(s.skeleton as ShiftPartitionsHorizontallySkeleton, s.obj.type as MatrixType)
			ShiftPartitionsVerticallySkeleton: generateShiftPartitionsVerticallySkeleton(s.skeleton as ShiftPartitionsVerticallySkeleton, s.obj.type as MatrixType)
			GatherSkeleton: generateGatherSkeleton(s, s.skeleton as GatherSkeleton, target)
			default: '''// TODO: SkeletonGenerator.generateSkeletonExpression: default case'''
		}
	}

	def static generateMapSkeleton(SkeletonExpression s, String target) '''
«««		«val a = s.obj»
«««		«val param_map = createParameterLookupTable(a, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
«««		#pragma omp parallel for simd
«««		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
«««			«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, a, param_map)»
«««		}
		//TODO: SkeletonGenerator.generateMapSkeleton: map skeleton
	'''

	def static generateMapInPlaceSkeleton(SkeletonExpression s) '''
		«val a = s.obj»
		«««	create lookup table for parameters
		«val param_map = createParameterLookupTable(a, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
		#pragma omp parallel for simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.type.sizeLocal»; ++«Config.var_loop_counter»){
			«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, a, param_map)»
		}
	'''
	
	def static Map<String, String> createParameterLookupTable(CollectionObject co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		if (parameters.length > inputs.size) {
			param_map.put(parameters.drop(inputs.size).head.name, '''«co.name»[«Config.var_loop_counter»]''')
		}

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
	def static dispatch Map<String, String> createParameterLookupTableMapIndexSkeleton(ArrayType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''(«Config.var_elem_offset» + «Config.var_loop_counter»)''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter»]''')
		
		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
	def static dispatch Map<String, String> createParameterLookupTableMapIndexSkeleton(MatrixType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''(«Config.var_row_offset» + «Config.var_loop_counter_rows»)''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''(«Config.var_col_offset» + «Config.var_loop_counter_cols»)''')
		param_map.put(parameters.drop(inputs.size + 2).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter_rows» * «co.colsLocal» + «Config.var_loop_counter_cols»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

// MapIndexInPlace
// Array
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a) '''
		«IF a.distributionMode == DistributionMode.COPY»
					«Config.var_elem_offset» = 0;
				«ELSE»
					«FOR p : 0..<Config.processes BEFORE 'if' SEPARATOR 'else if' AFTER ''»
						(«Config.var_pid» == «p»){
							«Config.var_elem_offset» = «p * a.sizeLocal»;
						}
					«ENDFOR»
				«ENDIF»
		«val param_map = createParameterLookupTableMapIndexSkeleton(a, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
		#pragma omp parallel for simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, a.eContainer as CollectionObject, param_map)»
		}
	'''
	
// Matrix
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m) '''
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
		«val param_map = createParameterLookupTableMapIndexSkeleton(m, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
		#pragma omp parallel for
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			#pragma omp simd
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, m.eContainer as CollectionObject, param_map)»
			}
		}
	'''

// map local index in place
	def static dispatch generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a) '''
		«val param_map = createParameterLookupTableMapLocalIndexSkeleton(a, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
		#pragma omp parallel for simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, a.eContainer as CollectionObject, param_map)»
		}
	'''
	
	def static dispatch generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m) '''
		«««	create lookup table for parameters
		«val param_map = createParameterLookupTableMapLocalIndexSkeleton(m, (s.skeleton.param as InternalFunctionCall).value.params, (s.skeleton.param as InternalFunctionCall).params)»
		#pragma omp parallel for
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			#pragma omp simd
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				«(s.skeleton.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s.skeleton, m.eContainer as CollectionObject, param_map)»
			}
		}
	'''
	
	def static dispatch Map<String, String> createParameterLookupTableMapLocalIndexSkeleton(ArrayType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_loop_counter»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter»]''')
		
		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

	def static dispatch Map<String, String> createParameterLookupTableMapLocalIndexSkeleton(MatrixType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_loop_counter_rows»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«Config.var_loop_counter_cols»''')
		param_map.put(parameters.drop(inputs.size + 2).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter_rows» * «co.colsLocal» + «Config.var_loop_counter_cols»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

// Fold
//	def static generateFoldSkeleton(SkeletonExpression s, String target) {
//		switch s.obj {
//			Array: generateArrayFoldSkeleton(s.skeleton as FoldSkeleton, s.obj as Array, target)
//		}
//	}

	def static generateFoldSkeleton(FoldSkeleton s, CollectionObject a, String target) '''	
		«Config.var_fold_result»_«a.calculateType.cppType»  = «s.identity.ValueAsString»;
		«val foldName = ((s.param as InternalFunctionCall).value as RegularFunction).name»
		
			#pragma omp parallel for simd reduction(«foldName»:«Config.var_fold_result»_«a.calculateCollectionType.cppType»)
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.type.sizeLocal»; ++«Config.var_loop_counter»){
			«val param_map = createParameterLookupTableFold(a, (s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
			«(s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s, a, param_map)»
		
		}		
		
		MPI_Allreduce(&«Config.var_fold_result»_«a.calculateCollectionType.cppType», &«target», sizeof(«a.calculateCollectionType.cppType»), MPI_BYTE, «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
	'''

	def static Map<String, String> createParameterLookupTableFold(CollectionObject a, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_fold_result»_«a.calculateCollectionType.cppType»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«a.name»[«Config.var_loop_counter»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

	def static Map<String, String> createParameterLookupTableFoldReductionClause(CollectionObject a,
		Iterable<de.wwu.musket.musket.Parameter> parameters, Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''omp_out''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''omp_in''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
	// Shift partitions
	def static generateShiftPartitionsHorizontallySkeleton(ShiftPartitionsHorizontallySkeleton s, MatrixType m) '''		
		«FOR pid : 0 ..< Config.processes BEFORE 'if' SEPARATOR 'else if' AFTER ''»
			(«Config.var_pid» == «pid»){
				«val pos = m.partitionPosition(pid)»				
				size_t «Config.var_shift_source» = «pid»;
				size_t «Config.var_shift_target» = «pid»;
«««				generate Function Call
				«val param_map = createParameterLookupTableShiftPartitionsHorizontally(m, pid, (s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
				size_t «(s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s, m.eContainer as CollectionObject, param_map)»
				«Config.var_shift_target» = ((«pid» + «Config.var_shift_steps») % «m.blocksInRow») + «pos.key * m.blocksInRow»;
				«Config.var_shift_source» = ((«pid» - «Config.var_shift_steps») % «m.blocksInRow») + «pos.key * m.blocksInRow»;
				
«««				shifting is happening
				if(«Config.var_shift_target» != «pid»){
					MPI_Request requests[2];
					MPI_Status statuses[2] ;
					«val buffer_name = Config.tmp_shift_buffer + '_' + ShiftCounter»
					std::array<«m.calculateCollectionType.cppType», «m.sizeLocal»> «buffer_name»;
					«generateMPIIrecv(pid, buffer_name + '.data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_source, "&requests[1]")»
					«generateMPIIsend(pid, (m.eContainer as CollectionObject).name + '.data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_target, "&requests[0]")»
					«generateMPIWaitall(2, "requests", "statuses")»
					
					#pragma omp parallel for simd
					for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «m.sizeLocal»; ++«Config.var_loop_counter»){
						«(m.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «buffer_name»[«Config.var_loop_counter»];
					}			
				}
			}
		«ENDFOR»
	'''
	

	def static Map<String, String> createParameterLookupTableShiftPartitionsHorizontally(MatrixType m, int pid,
		Iterable<de.wwu.musket.musket.Parameter> parameters, Iterable<Expression> inputs) {

		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«m.partitionPosition(pid).key»''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
	def static generateShiftPartitionsVerticallySkeleton(ShiftPartitionsVerticallySkeleton s, MatrixType m) '''		
		«FOR pid : 0 ..< Config.processes BEFORE 'if' SEPARATOR 'else if' AFTER ''»
			(«Config.var_pid» == «pid»){
				«val pos = m.partitionPosition(pid)»			
				size_t «Config.var_shift_source» = «pid»;
				size_t «Config.var_shift_target» = «pid»;
«««				generate Function Call
				«val param_map = createParameterLookupTableShiftPartitionsVertically(m, pid, (s.param as InternalFunctionCall).value.params, (s.param as InternalFunctionCall).params)»
				size_t «(s.param as InternalFunctionCall).generateInternalFunctionCallForSkeleton(s, m.eContainer as CollectionObject, param_map)»
				«Config.var_shift_target» = ((«pid / m.blocksInColumn» + «Config.var_shift_steps») % «m.blocksInColumn») * «m.blocksInRow» + «pos.value»;
				«Config.var_shift_source» = ((«pid / m.blocksInColumn» - «Config.var_shift_steps») % «m.blocksInColumn») * «m.blocksInRow» + «pos.value»;
				
«««				shifting is happening
				if(«Config.var_shift_target» != «pid»){
					MPI_Request requests[2];
					MPI_Status statuses[2] ;
					«val buffer_name = Config.tmp_shift_buffer + '_' + ShiftCounter»
					std::array<«m.calculateCollectionType.cppType», «m.sizeLocal»> «buffer_name»;
					«generateMPIIrecv(pid, buffer_name + '.data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_source, "&requests[1]")»
					«generateMPIIsend(pid, (m.eContainer as CollectionObject).name + '.data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_target, "&requests[0]")»
					«generateMPIWaitall(2, "requests", "statuses")»
					
					#pragma omp parallel for simd
					for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «m.sizeLocal»; ++«Config.var_loop_counter»){
						«(m.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «buffer_name»[«Config.var_loop_counter»];
					}			
				}
			}
		«ENDFOR»
	'''
	
	def static Map<String, String> createParameterLookupTableShiftPartitionsVertically(MatrixType m, int pid,
		Iterable<de.wwu.musket.musket.Parameter> parameters, Iterable<Expression> inputs) {

		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«m.partitionPosition(pid).value»''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
	def static generateGatherSkeleton(SkeletonExpression se, GatherSkeleton gs, String target) '''		
		«generateMPIAllgather(se.obj.name + '.data()', se.obj.type.sizeLocal, se.obj.calculateCollectionType.cppType, target + '.data()')»
	'''
}
