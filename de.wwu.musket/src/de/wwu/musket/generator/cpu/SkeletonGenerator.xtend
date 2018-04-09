package de.wwu.musket.generator.cpu

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

import java.util.HashMap
import java.util.Map

import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIAllgather
import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIIrecv
import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIIsend
import static de.wwu.musket.generator.cpu.MPIRoutines.generateMPIWaitall

import static extension de.wwu.musket.generator.cpu.FunctionGenerator.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*

import static extension de.wwu.musket.generator.cpu.ExpressionGenerator.*
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
	def static generateSkeletonExpression(SkeletonExpression s, String target, String target_type) {
		switch s.skeleton {
			MapSkeleton: generateMapSkeleton(s, target, target_type)
			MapInPlaceSkeleton: generateMapInPlaceSkeleton(s)
			MapIndexSkeleton: '''// TODO: MapIndexSkeleton''' //generateMapIndexSkeleton(s)			
			MapLocalIndexSkeleton: '''// TODO: MapLocalIndexSkeleton''' //generateMapLocalIndexSkeleton(s)
			MapIndexInPlaceSkeleton: generateMapIndexInPlaceSkeleton(s, s.obj.type)
			MapLocalIndexInPlaceSkeleton: generateMapLocalIndexInPlaceSkeleton(s, s.obj.type)
			FoldSkeleton: generateFoldSkeleton(s.skeleton as FoldSkeleton, s.obj, target)
			FoldLocalSkeleton: '''// TODO: FoldLocalSkeleton'''
			MapFoldSkeleton: generateMapFoldSkeleton(s.skeleton as MapFoldSkeleton, s.obj, target)
			ZipSkeleton:  '''// TODO: ZipSkeleton'''
			ZipInPlaceSkeleton:  '''// TODO: ZipInPlaceSkeleton'''
			ZipIndexSkeleton: '''// TODO: ZipIndexSkeleton'''		
			ZipLocalIndexSkeleton: '''// TODO: ZipLocalIndexSkeleton'''
			ZipIndexInPlaceSkeleton:  '''// TODO: ZipIndexInPlaceSkeleton'''
			ZipLocalIndexInPlaceSkeleton:  '''// TODO: ZipLocalIndexInPlaceSkeleton'''
			ShiftPartitionsHorizontallySkeleton: if(Config.processes > 1){generateShiftPartitionsHorizontallySkeleton(s.skeleton as ShiftPartitionsHorizontallySkeleton, s.obj.type as MatrixType)}
			ShiftPartitionsVerticallySkeleton: if(Config.processes > 1){generateShiftPartitionsVerticallySkeleton(s.skeleton as ShiftPartitionsVerticallySkeleton, s.obj.type as MatrixType)}
			GatherSkeleton: generateGatherSkeleton(s, s.skeleton as GatherSkeleton, target)
			default: '''// TODO: SkeletonGenerator.generateSkeletonExpression: default case'''
		}
	}


	def static generateMapSkeleton(SkeletonExpression s, String target, String target_type) '''
		«val a = s.obj»
		«val param_map = createParameterLookupTableMap(a, s.skeleton.param.functionParameters, s.skeleton.param.functionArguments)»
				#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
				for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.type.sizeLocal»; ++«Config.var_loop_counter»){
					«target_type» «Config.var_map_input» = «a.name»[«Config.var_loop_counter»];
					«s.skeleton.param.generateFunctionCallForSkeleton(s.skeleton, a, target, param_map)»
				}
	'''

/**
 * Creates the param map for the mapInPlace skeleton.
 * 
 * @param co the collection object the skeleton is used on
 * @param parameters the parameters of the skeleton
 * @param inputs the parameter inputs
 * @return the param map
 */
	def static Map<String, String> createParameterLookupTableMap(CollectionObject co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>
		
		param_map.put(parameters.drop(inputs.size).head.name, Config.var_map_input)
		
		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

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
	def static generateMapInPlaceSkeleton(SkeletonExpression s) '''
		«val a = s.obj»
		«««	create lookup table for parameters
		«val param_map = createParameterLookupTable(a, s.skeleton.param.functionParameters, s.skeleton.param.functionArguments)»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.type.sizeLocal»; ++«Config.var_loop_counter»){
			«s.skeleton.param.generateFunctionCallForSkeleton(s.skeleton, a, null, param_map)»
		}
	'''
	
/**
 * Creates the param map for the mapInPlace skeleton.
 * 
 * @param co the collection object the skeleton is used on
 * @param parameters the parameters of the skeleton
 * @param inputs the parameter inputs
 * @return the param map
 */
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
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a) '''
		«IF !(a.distributionMode == DistributionMode.COPY || Config.processes == 1)»
			«FOR p : 0..<Config.processes BEFORE 'switch(' + Config.var_pid + '){\n' SEPARATOR '' AFTER '}'»
				case «p»: {
					«Config.var_elem_offset» = «p * a.sizeLocal»;
					break;
				}
			«ENDFOR»
		«ENDIF»
		«val param_map = createParameterLookupTableMapIndexSkeleton(a, s.skeleton.param.functionParameters, s.skeleton.param.functionArguments)»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«s.skeleton.param.generateFunctionCallForSkeleton(s.skeleton, a.eContainer as CollectionObject, null, param_map)»
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
	def static dispatch generateMapIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m) '''
		«IF !(m.distributionMode == DistributionMode.COPY || Config.processes == 1)»
			«FOR p : 0..<Config.processes BEFORE 'switch(' + Config.var_pid + '){\n' SEPARATOR '' AFTER '}'»
				case «p»: {
					«Config.var_row_offset» = «p / m.blocksInColumn * m.rowsLocal»;
					«Config.var_col_offset» = «p % m.blocksInRow * m.colsLocal»;
					break;
				}
			«ENDFOR»
		«ENDIF»
		«««	create lookup table for parameters
		«val param_map = createParameterLookupTableMapIndexSkeleton(m, s.skeleton.param.functionParameters, s.skeleton.param.functionArguments)»
		#pragma omp«IF Config.cores > 1» parallel for«ELSE» simd«ENDIF» 
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			«IF Config.cores > 1»
				#pragma omp simd
			«ENDIF»
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				«s.skeleton.param.generateFunctionCallForSkeleton(s.skeleton, m.eContainer as CollectionObject, null, param_map)»
			}
		}
	'''
	
/**
 * Creates the param map for the mapIndex skeleton for arrays.
 * There is another version for matrices, because there are two index parameters.
 * 
 * @param co the array the skeleton is used on
 * @param parameters the parameters of the skeleton
 * @param inputs the parameter inputs
 * @return the param map
 */
	def static dispatch Map<String, String> createParameterLookupTableMapIndexSkeleton(ArrayType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		if(co.distributionMode == DistributionMode.COPY || Config.processes == 1){
			param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_loop_counter»''')
		}else{
			param_map.put(parameters.drop(inputs.size).head.name, '''(«Config.var_elem_offset» + «Config.var_loop_counter»)''')
		}
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter»]''')
		
		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
	/**
	 * Creates the param map for the mapIndex skeleton for matrices.
	 * There is another version for arrays, because there is only one index parameters.
	 * 
	 * @param co the matrix the skeleton is used on
	 * @param parameters the parameters of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static dispatch Map<String, String> createParameterLookupTableMapIndexSkeleton(MatrixType co, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		if(co.distributionMode == DistributionMode.COPY || Config.processes == 1){
			param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_loop_counter_rows»''')
			param_map.put(parameters.drop(inputs.size + 1).head.name, '''«Config.var_loop_counter_cols»''')
		}else{
			param_map.put(parameters.drop(inputs.size).head.name, '''(«Config.var_row_offset» + «Config.var_loop_counter_rows»)''')
			param_map.put(parameters.drop(inputs.size + 1).head.name, '''(«Config.var_col_offset» + «Config.var_loop_counter_cols»)''')
		}
		param_map.put(parameters.drop(inputs.size + 2).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter_rows» * «co.colsLocal» + «Config.var_loop_counter_cols»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}

// map local index in place
/**
 * Generates the mapLocalIndexInPlace skeleton for arrays.
 * 
 * @param s the skeleton expression
 * @param a the array on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static dispatch generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a) '''
		«val param_map = createParameterLookupTableMapLocalIndexSkeleton(a, s.skeleton.param.functionParameters, s.skeleton.param.functionArguments)»
		
		#pragma omp «IF Config.cores > 1»parallel for «ENDIF»simd		
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal»; ++«Config.var_loop_counter»){
			«s.skeleton.param.generateFunctionCallForSkeleton(s.skeleton, a.eContainer as CollectionObject, null, param_map)»
		}
	'''
	
/**
 * Generates the mapLocalIndexInPlace skeleton for matrices.
 * 
 * @param s the skeleton expression
 * @param m the matrix on which the skeleton is used
 * @return the generated skeleton code 
 */
	def static dispatch generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m) '''
		«««	create lookup table for parameters
		«val param_map = createParameterLookupTableMapLocalIndexSkeleton(m, s.skeleton.param.functionParameters, s.skeleton.param.functionArguments)»
		#pragma omp«IF Config.cores > 1» parallel for«ELSE» simd«ENDIF» 
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			«IF Config.cores > 1»
				#pragma omp simd
			«ENDIF»
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				«s.skeleton.param.generateFunctionCallForSkeleton(s.skeleton, m.eContainer as CollectionObject, null, param_map)»
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
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_loop_counter»''')
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«(co.eContainer as CollectionObject).name»[«Config.var_loop_counter»]''')
		
		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
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
	def static generateFoldSkeleton(FoldSkeleton s, CollectionObject co, String target) '''
		«val foldResultType = s.identity.calculateType.cppType.toCXXIdentifier»
	
		«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»
			«Config.var_fold_result»_«foldResultType»  = «s.identity.ValueAsString»;
		«ELSE»
			«target» = «s.identity.ValueAsString»;
		«ENDIF»
		«val foldName = s.param.functionName»
		
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd reduction(«foldName»:«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»«Config.var_fold_result»_«foldResultType»«ELSE»«target»«ENDIF»)
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «co.type.sizeLocal»; ++«Config.var_loop_counter»){
			«val param_map = createParameterLookupTableFold(s, co, target, s.param.functionParameters, s.param.functionArguments)»
			«s.param.generateFunctionCallForSkeleton(s, co, null, param_map)»
		}		
		
		«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»
			MPI_Allreduce(&«Config.var_fold_result»_«foldResultType», &«target», sizeof(«foldResultType»), MPI_BYTE, «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
		«ENDIF»
	'''

	/**
	 * Creates the param map for the fold skeleton.
	 * 
	 * @param a the collection object the skeleton is used on
	 * @param parameters the parameters of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static Map<String, String> createParameterLookupTableFold(FoldSkeleton s, CollectionObject a, String target, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		if(Config.processes > 1){
			param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_fold_result»_«s.identity.calculateType.cppType.toCXXIdentifier»''')
		}else{
			param_map.put(parameters.drop(inputs.size).head.name, target)
		}
		
		param_map.put(parameters.drop(inputs.size + 1).head.name, '''«a.name»[«Config.var_loop_counter»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}


	
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
	def static generateMapFoldSkeleton(MapFoldSkeleton s, CollectionObject co, String target) '''
		«val foldResultTypeIdentifier = s.identity.calculateType.cppType.toCXXIdentifier»
		«val foldResultType = s.identity.calculateType»
		«val name = if (Config.processes > 1 && co.distributionMode != DistributionMode.COPY) {Config.var_fold_result + "_" + foldResultTypeIdentifier } else {target}»
		«IF foldResultType.collection»
			«val ci = ((s.identity as CompareExpression).eqLeft as CollectionInstantiation)»
			«IF ci.values.size == 1»
				«name».assign(«foldResultType.collectionType.sizeLocal», «ci.values.head.ValueAsString»);
			«ELSEIF ci.values.size == foldResultType.collectionType.sizeLocal»
				«name».assign(«foldResultType.collectionType.sizeLocal»);
				«FOR i : 0 ..< ci.values.size»
					«name»[«i»] = «ci.values.get(i)»;
				«ENDFOR»
			«ELSE»
				«name».assign(«foldResultType.collectionType.sizeLocal», «foldResultType.collectionType.CXXPrimitiveDefaultValue»);
			«ENDIF»
		«ELSE»
			«name» = «s.identity.ValueAsString»;
		«ENDIF»
		
		«val foldName = s.param.functionName»
		
		
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd reduction(«foldName»:«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY»«Config.var_fold_result»_«foldResultTypeIdentifier»«ELSE»«target»«ENDIF»)
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «co.type.sizeLocal»; ++«Config.var_loop_counter»){
«««			map part
			«s.mapFunction.calculateType.cppType» «Config.var_map_fold_tmp»;
			«val param_map_map = createParameterLookupTableMapFoldMap(co, target, s.mapFunction.functionParameters, s.mapFunction.functionArguments)»
			«s.mapFunction.generateFunctionCallForSkeleton(s, co, null, param_map_map)»

«««			fold part
			«val param_map_fold = createParameterLookupTableMapFoldFold(s, co, target, s.param.functionParameters, s.param.functionArguments)»
			«s.param.generateFunctionCallForSkeleton(s, co, null, param_map_fold)»
		}		
		
		«IF Config.processes > 1 && co.distributionMode != DistributionMode.COPY && co.distributionMode != DistributionMode.LOC»
«««			array as result
			«IF s.identity.calculateType.collection»
				MPI_Allreduce(«Config.var_fold_result»_«foldResultTypeIdentifier».data(), «target».data(), «s.identity.calculateType.collectionType.sizeLocal» * sizeof(«s.identity.calculateType.calculateCollectionType.cppType»), MPI_BYTE, «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
			«ELSE »
				MPI_Allreduce(&«Config.var_fold_result»_«foldResultTypeIdentifier», &«target», sizeof(«foldResultTypeIdentifier»), MPI_BYTE, «foldName»«Config.mpi_op_suffix», MPI_COMM_WORLD); 
			«ENDIF»			
		«ENDIF»
	'''
	
		def static Map<String, String> createParameterLookupTableMapFoldMap(CollectionObject a, String target, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«a.name»[«Config.var_loop_counter»]''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		
		param_map.put("return", Config.var_map_fold_tmp)
		
		return param_map
	}
	
		def static Map<String, String> createParameterLookupTableMapFoldFold(MapFoldSkeleton s, CollectionObject a, String target, Iterable<de.wwu.musket.musket.Parameter> parameters,
		Iterable<Expression> inputs) {
		val param_map = new HashMap<String, String>

		if(Config.processes > 1){
			param_map.put(parameters.drop(inputs.size).head.name, '''«Config.var_fold_result»_«s.identity.calculateType.cppType.toCXXIdentifier»''')
			param_map.put("return", '''«Config.var_fold_result»_«s.identity.calculateType.cppType.toCXXIdentifier»''')
		}else{
			param_map.put(parameters.drop(inputs.size).head.name, target)
			param_map.put("return", target)
		}
		
		param_map.put(parameters.drop(inputs.size + 1).head.name, Config.var_map_fold_tmp)

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		
		
		
		return param_map
	}
	
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
	def static generateShiftPartitionsHorizontallySkeleton(ShiftPartitionsHorizontallySkeleton s, MatrixType m) '''		
		«FOR pid : 0 ..< Config.processes BEFORE 'switch(' + Config.var_pid + '){\n' SEPARATOR '' AFTER '}'»
			case «pid»:	{
				«val pos = m.partitionPosition(pid)»				
				int «Config.var_shift_source» = «pid»;
				int «Config.var_shift_target» = «pid»;
«««				generate Function Call
				«val param_map = createParameterLookupTableShiftPartitionsHorizontally(m, pid, s.param.functionParameters, s.param.functionArguments)»
				int «s.param.generateFunctionCallForSkeleton(s, m.eContainer as CollectionObject, null, param_map)»
				
				«Config.var_shift_target» = ((((«pos.value» + «Config.var_shift_steps») % «m.blocksInRow») + «m.blocksInRow» ) % «m.blocksInRow») + «pos.key * m.blocksInRow»;
				«Config.var_shift_source» = ((((«pos.value» - «Config.var_shift_steps») % «m.blocksInRow») + «m.blocksInRow» ) % «m.blocksInRow») + «pos.key * m.blocksInRow»;
				
«««				shifting is happening
				if(«Config.var_shift_target» != «pid»){
					MPI_Request requests[2];
					MPI_Status statuses[2];
					«val buffer_name = Config.tmp_shift_buffer»
					auto «buffer_name» = std::make_unique<std::vector<«m.calculateCollectionType.cppType»>>(«m.sizeLocal»);
					«generateMPIIrecv(pid, buffer_name + '->data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_source, "&requests[1]")»
					«generateMPIIsend(pid, (m.eContainer as CollectionObject).name + '.data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_target, "&requests[0]")»
					«generateMPIWaitall(2, "requests", "statuses")»
					
					std::move(«buffer_name»->begin(), «buffer_name»->end(), «(m.eContainer as CollectionObject).name».begin());
				}
				break;
			}
		«ENDFOR»
	'''
	
	/**
	 * Creates the param map for the ShiftPartitionsVertically skeleton.
	 * 
	 * @param m the matrix the skeleton is used on
	 * @param pid the process id
	 * @param inputs the parameter of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static Map<String, String> createParameterLookupTableShiftPartitionsHorizontally(MatrixType m, int pid,
		Iterable<de.wwu.musket.musket.Parameter> parameters, Iterable<Expression> inputs) {

		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«m.partitionPosition(pid).key»''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
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
	def static generateShiftPartitionsVerticallySkeleton(ShiftPartitionsVerticallySkeleton s, MatrixType m) '''		
		«FOR pid : 0 ..< Config.processes BEFORE 'switch(' + Config.var_pid + '){\n' SEPARATOR '' AFTER '}'»
			case «pid»:	{
				«val pos = m.partitionPosition(pid)»			
				int «Config.var_shift_source» = «pid»;
				int «Config.var_shift_target» = «pid»;
«««				generate Function Call
				«val param_map = createParameterLookupTableShiftPartitionsVertically(m, pid, s.param.functionParameters, s.param.functionArguments)»
				int «s.param.generateFunctionCallForSkeleton(s, m.eContainer as CollectionObject, null, param_map)»
				
				«Config.var_shift_target» = ((((«pos.key» + «Config.var_shift_steps») % «m.blocksInColumn») + «m.blocksInColumn» ) % «m.blocksInColumn») * «m.blocksInRow» + «pos.value»;
				«Config.var_shift_source» = ((((«pos.key» - «Config.var_shift_steps») % «m.blocksInColumn») + «m.blocksInColumn» ) % «m.blocksInColumn») * «m.blocksInRow» + «pos.value»;

«««				shifting is happening
				if(«Config.var_shift_target» != «pid»){
					MPI_Request requests[2];
					MPI_Status statuses[2];
					«val buffer_name = Config.tmp_shift_buffer»
					auto «buffer_name» = std::make_unique<std::vector<«m.calculateCollectionType.cppType»>>(«m.sizeLocal»);
					«generateMPIIrecv(pid, buffer_name + '->data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_source, "&requests[1]")»
					«generateMPIIsend(pid, (m.eContainer as CollectionObject).name + '.data()', m.sizeLocal, m.calculateCollectionType.cppType, Config.var_shift_target, "&requests[0]")»
					«generateMPIWaitall(2, "requests", "statuses")»
					
					std::move(«buffer_name»->begin(), «buffer_name»->end(), «(m.eContainer as CollectionObject).name».begin());		
				}
				break;
			}
		«ENDFOR»
	'''
	
	/**
	 * Creates the param map for the ShiftPartitionsVertically skeleton.
	 * 
	 * @param m the matrix the skeleton is used on
	 * @param pid the process id
	 * @param inputs the parameter of the skeleton
	 * @param inputs the parameter inputs
	 * @return the param map
	 */
	def static Map<String, String> createParameterLookupTableShiftPartitionsVertically(MatrixType m, int pid,
		Iterable<de.wwu.musket.musket.Parameter> parameters, Iterable<Expression> inputs) {

		val param_map = new HashMap<String, String>

		param_map.put(parameters.drop(inputs.size).head.name, '''«m.partitionPosition(pid).value»''')

		for (var i = 0; i < inputs.size; i++) {
			param_map.put(parameters.get(i).name, inputs.get(i).generateExpression(null))
		}
		return param_map
	}
	
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
	def static generateGatherSkeleton(SkeletonExpression se, GatherSkeleton gs, String target) '''
		«IF Config.processes > 1»
			«generateMPIAllgather(se.obj.name + '.data()', se.obj.type.sizeLocal, se.obj.calculateCollectionType.cppType, target + '.data()')»
		«ELSE»
			std::copy(«se.obj.name».begin(), «se.obj.name».end(), «target».begin());
		«ENDIF»
	'''
}
