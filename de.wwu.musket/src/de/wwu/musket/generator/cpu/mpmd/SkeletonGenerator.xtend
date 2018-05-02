package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Expression
import de.wwu.musket.musket.FoldSkeleton
import de.wwu.musket.musket.GatherSkeleton
import de.wwu.musket.musket.MapInPlaceSkeleton
import de.wwu.musket.musket.MapIndexInPlaceSkeleton
import de.wwu.musket.musket.MapLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.MapSkeleton
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton
import de.wwu.musket.musket.SkeletonExpression
import de.wwu.musket.musket.MapIndexSkeleton
import de.wwu.musket.musket.MapLocalIndexSkeleton
import de.wwu.musket.musket.ZipSkeleton
import de.wwu.musket.musket.ZipInPlaceSkeleton
import de.wwu.musket.musket.ZipIndexSkeleton
import de.wwu.musket.musket.ZipLocalIndexSkeleton
import de.wwu.musket.musket.ZipIndexInPlaceSkeleton
import de.wwu.musket.musket.ZipLocalIndexInPlaceSkeleton
import de.wwu.musket.musket.FoldLocalSkeleton
import de.wwu.musket.musket.MapFoldSkeleton
import de.wwu.musket.musket.CollectionInstantiation
import de.wwu.musket.musket.CompareExpression

import java.util.HashMap
import java.util.Map

import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIAllgather
import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIIrecv
import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIIsend
import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIWaitall

import static extension de.wwu.musket.generator.cpu.mpmd.FunctionGenerator.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*



/**
 * Generates the skeleton calls.
 * <p>
 * Entry point is the method generateSkeletonExpression(SkeletonExpression s, String target). 
 * It is called by the LogicGenerator.
 */
class SkeletonGenerator {
/**
 * Starting point for the skeleton generator.
 * It switches over the skeletons and calls the correct function.
 * 
 * @param s the skeleton expression
 * @param target where to write the result of the skeleton expression
 * @return generated skeleton call
 */
	def static generateSkeletonExpression(SkeletonExpression s, String target, String target_type, int processId) {
		switch s.skeleton {
			MapSkeleton: generateMapSkeleton(s, target, target_type, processId)
			MapInPlaceSkeleton: generateMapInPlaceSkeleton(s, processId)
			MapIndexSkeleton: '''// TODO: MapIndexSkeleton''' //generateMapIndexSkeleton(s)			
			MapLocalIndexSkeleton: '''// TODO: MapLocalIndexSkeleton''' //generateMapLocalIndexSkeleton(s)
			MapIndexInPlaceSkeleton: generateMapIndexInPlaceSkeleton(s, s.obj.type, processId)
			MapLocalIndexInPlaceSkeleton: generateMapLocalIndexInPlaceSkeleton(s, s.obj.type, processId)
			FoldSkeleton: generateFoldSkeleton(s.skeleton as FoldSkeleton, s.obj, target, processId)
			FoldLocalSkeleton: '''// TODO: FoldLocalSkeleton'''
			MapFoldSkeleton: generateMapFoldSkeleton(s.skeleton as MapFoldSkeleton, s.obj, target, processId)
			ZipSkeleton:  '''// TODO: ZipSkeleton'''
			ZipInPlaceSkeleton:  '''// TODO: ZipInPlaceSkeleton'''
			ZipIndexSkeleton: '''// TODO: ZipIndexSkeleton'''		
			ZipLocalIndexSkeleton: '''// TODO: ZipLocalIndexSkeleton'''
			ZipIndexInPlaceSkeleton:  '''// TODO: ZipIndexInPlaceSkeleton'''
			ZipLocalIndexInPlaceSkeleton:  '''// TODO: ZipLocalIndexInPlaceSkeleton'''
			ShiftPartitionsHorizontallySkeleton: if(Config.processes > 1){generateShiftPartitionsHorizontallySkeleton(s.skeleton as ShiftPartitionsHorizontallySkeleton, s.obj.type as MatrixType, processId)}
			ShiftPartitionsVerticallySkeleton: if(Config.processes > 1){generateShiftPartitionsVerticallySkeleton(s.skeleton as ShiftPartitionsVerticallySkeleton, s.obj.type as MatrixType, processId)}
			GatherSkeleton: generateGatherSkeleton(s, s.skeleton as GatherSkeleton, target, processId)
			default: '''// TODO: SkeletonGenerator.generateSkeletonExpression: default case'''
		}
	}


	def static generateMapSkeleton(SkeletonExpression s, String target, String target_type, int processId) '''
		«val a = s.obj»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.type.sizeLocal(processId)»; ++«Config.var_loop_counter»){
«««			TODO: «target_type» «Config.var_map_input» = «a.name»[«Config.var_loop_counter»];
		}
	'''

/**
 * Generates the MapInPlace skeleton.
 * <p>
 * The method does not need a target, since it is a mapInPlace call.
 * It is the same for matrices and arrays, because both are generated as std::arrays and therefore,
 * a single loop is sufficient.
 * 
 * @param s the skeleton expression
 * @return generated skeleton call
 */
	def static generateMapInPlaceSkeleton(SkeletonExpression s, int processId) '''
		«val a = s.obj»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.type.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«a.name»[«Config.var_loop_counter»] = «s.skeleton.param.functionName.toFirstLower»_functor(«FOR arg : s.skeleton.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«a.name»[«Config.var_loop_counter»]);
		}
	'''
	
// MapIndexInPlace

/**
 * Generates the mapIndexInPlace skeleton for arrays.
 * <p>
 * First, the offset variable is set for each process.
 * Second, the loop iterates over the array.
 * 
 * @param s the skeleton expression
 * @param a the array on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a, int processId) '''
		«IF a.distributionMode == DistributionMode.DIST && Config.processes > 1»
			«Config.var_elem_offset» = «a.globalOffset(processId)»;
		«ELSEIF a.distributionMode == DistributionMode.COPY && Config.processes > 1»
			«Config.var_elem_offset» = 0;
		«ENDIF»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«s.obj.name»[«Config.var_loop_counter»] = «s.skeleton.param.functionName.toFirstLower»_functor(«FOR arg : s.skeleton.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«Config.var_elem_offset» + «Config.var_loop_counter», «s.obj.name»[«Config.var_loop_counter»]);
		}
	'''
	
/**
 * Generates the mapIndexInPlace skeleton for matrices.
 * <p>
 * First, the offset variables are set for each process.
 * Second, the loop iterates over the array.
 * 
 * @param s the skeleton expression
 * @param m the matrix on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m, int processId) '''
		«IF m.distributionMode == DistributionMode.DIST && Config.processes > 1»
			«Config.var_row_offset» = «processId / m.blocksInColumn * m.rowsLocal»;
			«Config.var_col_offset» = «processId % m.blocksInRow * m.colsLocal»;
		«ELSEIF m.distributionMode == DistributionMode.COPY && Config.processes > 1»
			«Config.var_row_offset» = 0;
			«Config.var_col_offset» = 0;
		«ENDIF»
		#pragma omp«IF Config.cores > 1» parallel for«ELSE» simd«ENDIF» 
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			«IF Config.cores > 1»
				#pragma omp simd
			«ENDIF»
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				size_t «Config.var_loop_counter» = «Config.var_loop_counter_rows» * «m.colsLocal» + «Config.var_loop_counter_cols»;
				«s.obj.name»[«Config.var_loop_counter»] = «s.skeleton.param.functionName.toFirstLower»_functor(«FOR arg : s.skeleton.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«Config.var_row_offset» + «Config.var_loop_counter_rows», «Config.var_col_offset» + «Config.var_loop_counter_cols», «s.obj.name»[«Config.var_loop_counter»]);
			}
		}
	'''


// map local index in place
/**
 * Generates the mapLocalIndexInPlace skeleton for arrays.
 * 
 * @param s the skeleton expression
 * @param a the array on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static dispatch generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a, int processId) '''
		«val param_map = createParameterLookupTableMapLocalIndexSkeleton(a, s.skeleton.param.functionParameters, s.skeleton.param.functionArguments, processId)»
		
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«s.obj.name»[«Config.var_loop_counter»] = «s.skeleton.param.functionName.toFirstLower»_functor(«FOR arg : s.skeleton.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«Config.var_loop_counter», «s.obj.name»[«Config.var_loop_counter»]);
		}
	'''
	
/**
 * Generates the mapLocalIndexInPlace skeleton for matrices.
 * 
 * @param s the skeleton expression
 * @param m the matrix on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static dispatch generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m, int processId) '''
		#pragma omp«IF Config.cores > 1» parallel for«ELSE» simd«ENDIF» 
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			«IF Config.cores > 1»
				#pragma omp simd
			«ENDIF»
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				size_t «Config.var_loop_counter» = «Config.var_loop_counter_rows» * «m.colsLocal» + «Config.var_loop_counter_cols»;
				«s.obj.name»[«Config.var_loop_counter»] = «s.skeleton.param.functionName.toFirstLower»_functor(«FOR arg : s.skeleton.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«Config.var_loop_counter_rows», «Config.var_loop_counter_cols», «s.obj.name»[«Config.var_loop_counter»]);
			}
		}
	'''
	
	/**
	 * Creates the param map for the mapLocalIndex skeleton for arrays.
	 * There is another version for matrices, because there are two index parameters.
	 * 
	 * @param co the array the skeleton is used on
	 * @param parameters the parameters of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static dispatch Map<String, String> createParameterLookupTableMapLocalIndexSkeleton(ArrayType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs, int processId) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_loop_counter»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter»]''')
		
		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null, processId))
		}
		return param_map
	}

	/**
	 * Creates the param map for the mapLocalIndex skeleton for matrices.
	 * There is another version for arrays, because there is only one index parameters.
	 * 
	 * @param co the matrix the skeleton is used on
	 * @param parameters the parameters of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static dispatch Map<String, String> createParameterLookupTableMapLocalIndexSkeleton(MatrixType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs, int processId) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_loop_counter_rows»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«Config.var_loop_counter_cols»''')
		param_map.put(parameters.drop(inputs.size + 2).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter_rows» * «co.colsLocal» + «Config.var_loop_counter_cols»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null, processId))
		}
		return param_map
	}

// Fold
/**
 * Generates the fold skeleton.
 * <p>
 * It is the same for arrays and matrices, because both are generated as std::vectors. 
 * <p>
 * ASSUMPTION: the function passed to the fold skeleton is associative and commutative
 * 
 * @param s the skeleton expression
 * @param a the array on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static generateFoldSkeleton(FoldSkeleton s, CollectionObject co, String target, int processId) '''
	«val foldResultType = s.identity.calculateType.cppType.toCXXIdentifier»

	«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»
		«Config.var_fold_result»_«foldResultType» = «s.identity.ValueAsString»;
	«ELSE»
		«target» = «s.identity.ValueAsString»;
	«ENDIF»
	«val foldName = s.param.functionName + "_reduction"»
	
	#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd reduction(«foldName»:«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»«Config.var_fold_result»_«foldResultType»«ELSE»«target»«ENDIF»)
	for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «co.type.sizeLocal(processId)»; ++«Config.var_loop_counter»){
		«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»
			«Config.var_fold_result»_«foldResultType» = «s.param.functionName.toFirstLower»_functor(«FOR arg : s.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«Config.var_fold_result»_«foldResultType», «co.name»[«Config.var_loop_counter»]);
		«ELSE»
			«target» = «s.param.functionName.toFirstLower»_functor(«FOR arg : s.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«target», «co.name»[«Config.var_loop_counter»]);
		«ENDIF»
	}
	
	«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»
		MPI_Allreduce(&«Config.var_fold_result»_«foldResultType», &«target», «Config.processes», «s.identity.calculateType.MPIType», «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
	«ENDIF»
	'''


	/**
 * Generates the MapFold skeleton.
 * <p>
 * It is the same for arrays and matrices, because both are generated as std::vectors. 
 * <p>
 * ASSUMPTION: the function passed to the fold skeleton is associative and commutative
 * 
 * @param s the skeleton expression
 * @param s the skeleton expression
 * @param a the array on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static generateMapFoldSkeleton(MapFoldSkeleton s, CollectionObject co, String target, int processId) '''
		«val foldResultTypeIdentifier = s.identity.calculateType.cppType.replace("0", s.mapFunction.calculateType.collectionType?.size.toString).toCXXIdentifier»
		«val foldResultType = s.identity.calculateType»
		«val name = if (Config.processes > 1 && co.distributionMode != DistributionMode.COPY) {Config.var_fold_result + "_" + foldResultTypeIdentifier } else {target}»
		«IF foldResultType.collection»
			«val ci = ((s.identity as CompareExpression).eqLeft as CollectionInstantiation)»
			«IF ci.values.size == 1»
				«name».fill(«ci.values.head.ValueAsString»);
			«ELSEIF ci.values.size == foldResultType.collectionType.sizeLocal(processId)»
				«FOR i : 0 ..< ci.values.size»
					«name»[«i»] = «ci.values.get(i)»;
				«ENDFOR»
			«ELSE»
				«name».fill(«foldResultType.collectionType.CXXPrimitiveDefaultValue»);
			«ENDIF»
		«ELSE»
			«name» = «s.identity.ValueAsString»;
		«ENDIF»
		
		«val foldName = s.param.functionName»
		
		
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd reduction(«foldName»_reduction:«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»«Config.var_fold_result»_«foldResultTypeIdentifier»«ELSE»«target»«ENDIF»)
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «co.type.sizeLocal(processId)»; ++«Config.var_loop_counter»){
«««			map part
			«s.mapFunction.calculateType.cppType.replace("0", s.mapFunction.calculateType.collectionType?.size.toString)» «Config.var_map_fold_tmp» = «s.mapFunction.functionName.toFirstLower»_functor(«FOR arg : s.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«co.name»[«Config.var_loop_counter»]);

«««			fold part
			«Config.var_fold_result»_«s.identity.calculateType.cppType.replace("0", s.identity.calculateType.collectionType?.size.toString).toCXXIdentifier» = «s.param.functionName.toFirstLower»_functor(«FOR arg : s.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«Config.var_fold_result»_«s.identity.calculateType.cppType.replace("0", s.identity.calculateType.collectionType?.size.toString).toCXXIdentifier», «Config.var_map_fold_tmp»);
		}		
		
		«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY && co.distributionMode != DistributionMode.LOC»
«««			array as result
			«IF s.identity.calculateType.collection»
				MPI_Allreduce(«Config.var_fold_result»_«foldResultTypeIdentifier».data(), «target».data(), «s.identity.calculateType.collectionType.sizeLocal(processId)», «s.identity.calculateType.calculateCollectionType.MPIType», «foldName»_reduction«Config.mpi_op_suffix», MPI_COMM_WORLD); 
			«ELSE »
				MPI_Allreduce(&«Config.var_fold_result»_«foldResultTypeIdentifier», &«target», «Config.processes», «foldResultType.MPIType», «foldName»_reduction«Config.mpi_op_suffix», MPI_COMM_WORLD); 
			«ENDIF»			
		«ENDIF»
	'''
		
	// Shift partitions
	
	/**
	 * Generates the shift partitions horizontally skeleton.
	 * <p>
	 * The skeleton only exists for matrices. First, the source and target have to be calculated.
	 * Source: where the new values come from
	 * Target: where the current values go to
	 * The result of the passed function gives the number of steps. So the second step has to determine the partition id.
	 * Partitions are considered as a cirlcle --> 0,1,2,0,1,2 and so on.
	 * 
	 * The buffer is required because non-blocking communication is used.
	 * 
	 * @param co the collection object the skeleton is used on
	 * @param parameters the parameters of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static generateShiftPartitionsHorizontallySkeleton(ShiftPartitionsHorizontallySkeleton s, MatrixType m, int processId) '''		
				«val pos = m.partitionPosition(processId)»
«««				generate Function Call
				«Config.var_shift_steps» = «s.param.functionName.toFirstLower»_functor(«FOR arg : s.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«pos.key»);
				
				«Config.var_shift_target» = ((((«pos.value» + «Config.var_shift_steps») % «m.blocksInRow») + «m.blocksInRow» ) % «m.blocksInRow») + «pos.key * m.blocksInRow»;
				«Config.var_shift_source» = ((((«pos.value» - «Config.var_shift_steps») % «m.blocksInRow») + «m.blocksInRow» ) % «m.blocksInRow») + «pos.key * m.blocksInRow»;
				
«««				shifting is happening
				if(«Config.var_shift_target» != «processId»){
					MPI_Request requests[2];
					MPI_Status statuses[2];
					«val buffer_name = Config.tmp_shift_buffer»
					auto «buffer_name» = std::make_unique<std::vector<«m.calculateCollectionType.cppType»>>(«m.sizeLocal(processId)»);
					«generateMPIIrecv(processId, buffer_name + '->data()', m.sizeLocal(processId), m.calculateCollectionType, Config.var_shift_source, "&requests[1]")»
					«generateMPIIsend(processId, (m.eContainer as CollectionObject).name + '.data()', m.sizeLocal(processId), m.calculateCollectionType, Config.var_shift_target, "&requests[0]")»
					«generateMPIWaitall(2, "requests", "statuses")»
					
					std::move(«buffer_name»->begin(), «buffer_name»->end(), «(m.eContainer as CollectionObject).name».begin());
				}
	'''
	
	/**
	 * Generates the shift partitions vertically skeleton.
	 * <p>
	 * The skeleton only exists for matrices. First, the source and target have to be calculated.
	 * Source: where the new values come from
	 * Target: where the current values go to
	 * The result of the passed function gives the number of steps. So the second step has to determine the partition id.
	 * Partitions are considered as a cirlcle --> 0,1,2,0,1,2 and so on.
	 * 
	 * The buffer is required because non-blocking communication is used.
	 * 
	 * @param co the collection object the skeleton is used on
	 * @param parameters the parameters of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static generateShiftPartitionsVerticallySkeleton(ShiftPartitionsVerticallySkeleton s, MatrixType m, int processId) '''		
				«val pos = m.partitionPosition(processId)»
«««				generate Function Call
				«Config.var_shift_steps» = «s.param.functionName.toFirstLower»_functor(«FOR arg : s.param.functionArguments SEPARATOR ", " AFTER ", "»«arg.generateExpression(null, processId)»«ENDFOR»«pos.value»);
				«Config.var_shift_target» = ((((«pos.key» + «Config.var_shift_steps») % «m.blocksInColumn») + «m.blocksInColumn» ) % «m.blocksInColumn») * «m.blocksInRow» + «pos.value»;
				«Config.var_shift_source» = ((((«pos.key» - «Config.var_shift_steps») % «m.blocksInColumn») + «m.blocksInColumn» ) % «m.blocksInColumn») * «m.blocksInRow» + «pos.value»;

«««				shifting is happening
				if(«Config.var_shift_target» != «processId»){
					MPI_Request requests[2];
					MPI_Status statuses[2];
					«val buffer_name = Config.tmp_shift_buffer»
					auto «buffer_name» = std::make_unique<std::vector<«m.calculateCollectionType.cppType»>>(«m.sizeLocal(processId)»);
					«generateMPIIrecv(processId, buffer_name + '->data()', m.sizeLocal(processId), m.calculateCollectionType, Config.var_shift_source, "&requests[1]")»
					«generateMPIIsend(processId, (m.eContainer as CollectionObject).name + '.data()', m.sizeLocal(processId), m.calculateCollectionType, Config.var_shift_target, "&requests[0]")»
					«generateMPIWaitall(2, "requests", "statuses")»
					
					std::move(«buffer_name»->begin(), «buffer_name»->end(), «(m.eContainer as CollectionObject).name».begin());		
				}
	'''
		
	/**
 * Generates the gather skeleton.
 * <p>
 * It is the same for arrays and matrices, because both are generated as std::arrays. 
 * <p>
 * It is generated as a MPIAllgather routine so that each process gets the values of the copy distributed array.
 * 
 * Assumption: it is forbidden in the model to call gather for copy distributed data structure, and boundaries are already checked
 * 
 * @param se the skeleton expression
 * @param target the target where the results should be written
 * @return the generated skeleton code 
 */
	def static generateGatherSkeleton(SkeletonExpression se, GatherSkeleton gs, String target, int processId) '''
		«IF Config.processes > 1»
			«generateMPIAllgather(se.obj.name + '.data()', se.obj.type.sizeLocal(processId), se.obj.calculateCollectionType, target + '.data()')»
		«ELSE»
			std::copy(«se.obj.name».begin(), «se.obj.name».end(), «target».begin());
		«ENDIF»
	'''
}
