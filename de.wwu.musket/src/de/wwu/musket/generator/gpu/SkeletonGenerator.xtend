package de.wwu.musket.generator.gpu

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.Object
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
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

import static de.wwu.musket.generator.gpu.MPIRoutines.generateMPIAllgather
import static de.wwu.musket.generator.gpu.MPIRoutines.generateMPIIrecv
import static de.wwu.musket.generator.gpu.MPIRoutines.generateMPIIsend
import static de.wwu.musket.generator.gpu.MPIRoutines.generateMPIWaitall

import static extension de.wwu.musket.generator.gpu.util.DataHelper.*
import static de.wwu.musket.generator.gpu.FunctionGenerator.generateFunctionCall
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.gpu.util.DataHelper.*
import static extension de.wwu.musket.generator.gpu.ExpressionGenerator.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import de.wwu.musket.musket.ScatterSkeleton
import org.eclipse.emf.common.util.BasicEList
import de.wwu.musket.musket.Expression
import org.eclipse.emf.common.util.EList
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.SkeletonParameterInput
import de.wwu.musket.musket.ReductionSkeleton
import de.wwu.musket.musket.ReductionOperation
import de.wwu.musket.musket.MapReductionSkeleton

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
	def static generateSkeletonExpression(SkeletonExpression s, Object target, int processId) {
		val skel = s.skeleton
		switch skel {
			MapSkeleton: generateMapSkeleton(s, (target as CollectionObject), processId)
			MapInPlaceSkeleton: generateMapInPlaceSkeleton(s, processId)
			MapIndexSkeleton: generateMapIndexSkeleton(s, (target as CollectionObject), processId)			
			MapLocalIndexSkeleton: generateMapLocalIndexSkeleton(s, (target as CollectionObject), processId)
			MapIndexInPlaceSkeleton: generateMapIndexInPlaceSkeleton(s, s.obj, processId)
			MapLocalIndexInPlaceSkeleton: generateMapLocalIndexInPlaceSkeleton(s, s.obj, processId)
			FoldSkeleton: generateFoldSkeleton(s, s.obj, target, processId)
			FoldLocalSkeleton: '''// TODO: FoldLocalSkeleton''' // this is future work
			MapFoldSkeleton: generateMapFoldSkeleton(s, s.obj, target, processId)
			ReductionSkeleton: generateReductionSkeleton(s, s.obj, target, processId)
			MapReductionSkeleton: generateMapReductionSkeleton(s, s.obj, target, processId)
			ZipSkeleton:  generateZipSkeleton(s, (target as CollectionObject).name, processId)
			ZipInPlaceSkeleton:  generateZipInPlaceSkeleton(s, processId)
			ZipIndexSkeleton: generateZipIndexSkeleton(s, s.obj.type, (target as CollectionObject).name, processId)		
			ZipLocalIndexSkeleton: generateZipLocalIndexSkeleton(s, s.obj.type, (target as CollectionObject).name, processId)
			ZipIndexInPlaceSkeleton:  generateZipIndexInPlaceSkeleton(s, s.obj.type, processId)
			ZipLocalIndexInPlaceSkeleton: generateZipLocalIndexInPlaceSkeleton(s, s.obj.type, processId)
			ShiftPartitionsHorizontallySkeleton: if(Config.processes > 1){generateShiftPartitionsHorizontallySkeleton(s, processId)}
			ShiftPartitionsVerticallySkeleton: if(Config.processes > 1){generateShiftPartitionsVerticallySkeleton(s, processId)}
			GatherSkeleton: generateGatherSkeleton(s, target, processId)
			ScatterSkeleton: generateScatterSkeleton(s, target, processId)
			default: '''// TODO: SkeletonGenerator.generateSkeletonExpression: default case'''
		}
	}

	def static generateSetValuesInFunctor(SkeletonExpression s, SkeletonParameterInput spi){
		var result = ""
		val numberOfFreeParams = getNumberOfFreeParameters(s, spi.toFunction)
		val parameters = spi.functionParameters
		val arguments = spi.functionArguments
		for(var i = 0; i < numberOfFreeParams; i++){
			result += '''«s.getFunctorObjectName(spi)».«parameters.get(i).name» = «arguments.get(i).generateExpression(null, 0)»;'''
		}
		return result
	}

	def static generateMapSkeleton(SkeletonExpression s, CollectionObject target, int processId) '''
		«val a = s.obj»
		«val aType = a.calculateCollectionType.cppType»
		«val tType = target.calculateCollectionType.cppType»
		«val skel = s.skeleton as MapSkeleton»
		«generateSetValuesInFunctor(s, skel.param)»
		mkt::map<«aType», «tType», «s.getFunctorName(skel.param)»>(«a.collectionName», «target.name», «s.getFunctorObjectName(skel.param)»);
	'''
	
	def static generateMapIndexSkeleton(SkeletonExpression s, CollectionObject target, int processId) '''
		«val a = s.obj»
		«val aType = a.calculateCollectionType.cppType»
		«val tType = target.calculateCollectionType.cppType»
		«val skel = s.skeleton as MapIndexSkeleton»
		«generateSetValuesInFunctor(s, skel.param)»
		mkt::map_index<«aType», «tType», «s.getFunctorName(skel.param)»>(«a.collectionName», «target.name», «s.getFunctorObjectName(skel.param)»);
	'''
	
	def static generateMapLocalIndexSkeleton(SkeletonExpression s, CollectionObject target, int processId) '''
		«val a = s.obj»
		«val aType = a.calculateCollectionType.cppType»
		«val tType = target.calculateCollectionType.cppType»
		«val skel = s.skeleton as MapLocalIndexSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param)»
		mkt::map_local_index<«aType», «tType», «s.getFunctorName(skel.param)»>(«a.collectionName», «target.name», «s.getFunctorObjectName(skel.param)»);
	'''

	def static generateMapInPlaceSkeleton(SkeletonExpression s, int processId) '''
		«val a = s.obj»
		«val skel = s.skeleton as MapInPlaceSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param)»
		mkt::map_in_place<«s.obj.calculateCollectionType.cppType», «s.getFunctorName(skel.param)»>(«a.name», «s.getFunctorObjectName(skel.param)»);
	'''

	def static generateMapIndexInPlaceSkeleton(SkeletonExpression s, CollectionObject co, int processId) '''
		«val skel = s.skeleton as MapIndexInPlaceSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param)»
		mkt::map_index_in_place<«s.obj.calculateCollectionType.cppType», «s.getFunctorName(skel.param)»>(«co.name», «s.getFunctorObjectName(skel.param)»);
	'''

	def static generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, CollectionObject co, int processId) '''
		«val skel = s.skeleton as MapLocalIndexInPlaceSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param)»
		mkt::map_local_index_in_place<«co.calculateCollectionType.cppType», «s.getFunctorName(skel.param)»>(«co.name», «s.getFunctorObjectName(skel.param)»);
	'''

// Zip
	def static generateZipSkeleton(SkeletonExpression s, String target, int processId) '''
		// Zip skeleton start
		«val obj = s.obj»
		«val skel = s.skeleton as ZipSkeleton»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «obj.type.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«target»[«Config.var_loop_counter»] = «generateFunctionCall(skel, obj, skel.zipWith.value as CollectionObject, processId)»
		}
		// Zip skeleton end
	'''
	
// ZipInPlace
	def static generateZipInPlaceSkeleton(SkeletonExpression s, int processId) '''
		// Zip skeleton start
		«val obj = s.obj»
		«val skel = s.skeleton as ZipInPlaceSkeleton»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «obj.type.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«obj.name»[«Config.var_loop_counter»] = «generateFunctionCall(skel, obj, skel.zipWith.value as CollectionObject, processId)»
		}
		// Zip skeleton end
	'''

// ZipIndex
	def static dispatch generateZipIndexSkeleton(SkeletonExpression s, ArrayType a, String target, int processId) '''
		// ZipIndexSkeleton Array Start
		«val skel = s.skeleton as ZipIndexSkeleton»
		
		«IF a.distributionMode == DistributionMode.DIST && Config.processes > 1»
			«Config.var_elem_offset» = «a.globalOffset(processId)»;
		«ELSEIF a.distributionMode == DistributionMode.COPY && Config.processes > 1»
			«Config.var_elem_offset» = 0;
		«ENDIF»
		
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«target»[«Config.var_loop_counter»] = «generateFunctionCall(skel, s.obj.type, (skel.zipWith.value as CollectionObject).type, processId)»
		}
		// ZipIndexSkeleton Array End
	'''
	
	def static dispatch generateZipIndexSkeleton(SkeletonExpression s, MatrixType m, String target, int processId) '''
		// ZipIndexSkeleton Matrix Start
		«val skel = s.skeleton as ZipIndexSkeleton»
		
		«IF m.distributionMode == DistributionMode.DIST && Config.processes > 1»
			«Config.var_row_offset» = «m.globalRowOffset(processId)»;
			«Config.var_col_offset» = «m.globalColOffset(processId)»;
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
				«target»[«Config.var_loop_counter»] = «generateFunctionCall(skel, m, (skel.zipWith.value as CollectionObject).type, processId)»
			}
		}
		// ZipIndexSkeleton Matrix End
	'''

// ZipLocalIndex
	def static dispatch generateZipLocalIndexSkeleton(SkeletonExpression s, ArrayType a, String target, int processId) '''
		// ZipLocalIndexSkeleton Array Start
		«val skel = s.skeleton as ZipLocalIndexSkeleton»
		
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«target»[«Config.var_loop_counter»] = «generateFunctionCall(skel, s.obj.type, (skel.zipWith.value as CollectionObject).type, processId)»
		}
		// ZipLocalIndexSkeleton Array End
	'''
	
	def static dispatch generateZipLocalIndexSkeleton(SkeletonExpression s, MatrixType m, String target, int processId) '''
		// ZipLocalIndexSkeleton Matrix Start
		«val skel = s.skeleton as ZipLocalIndexSkeleton»

		#pragma omp«IF Config.cores > 1» parallel for«ELSE» simd«ENDIF» 
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			«IF Config.cores > 1»
				#pragma omp simd
			«ENDIF»
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				size_t «Config.var_loop_counter» = «Config.var_loop_counter_rows» * «m.colsLocal» + «Config.var_loop_counter_cols»;
				«target»[«Config.var_loop_counter»] = «generateFunctionCall(skel, m, (skel.zipWith.value as CollectionObject).type, processId)»
			}
		}
		// ZipLocalIndexSkeleton Matrix End
	'''

// ZipIndexInPlace
	def static dispatch generateZipIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a, int processId) '''
		// ZipIndexInPlaceSkeleton Array Start
		«val skel = s.skeleton as ZipIndexInPlaceSkeleton»
		«val obj = s.obj»
		«IF a.distributionMode == DistributionMode.DIST && Config.processes > 1»
			«Config.var_elem_offset» = «a.globalOffset(processId)»;
		«ELSEIF a.distributionMode == DistributionMode.COPY && Config.processes > 1»
			«Config.var_elem_offset» = 0;
		«ENDIF»
		
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«obj.name»[«Config.var_loop_counter»] = «generateFunctionCall(skel, s.obj.type, (skel.zipWith.value as CollectionObject).type, processId)»
		}
		// ZipIndexInPlaceSkeleton Array End
	'''
	
	def static dispatch generateZipIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m, int processId) '''
		// ZipIndexInPlaceSkeleton Matrix Start
		«val skel = s.skeleton as ZipIndexInPlaceSkeleton»
		«val obj = s.obj»
		«IF m.distributionMode == DistributionMode.DIST && Config.processes > 1»
			«Config.var_row_offset» = «m.globalRowOffset(processId)»;
			«Config.var_col_offset» = «m.globalColOffset(processId)»;
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
				«obj.name»[«Config.var_loop_counter»] = «generateFunctionCall(skel, m, (skel.zipWith.value as CollectionObject).type, processId)»
			}
		}
		// ZipIndexInPlaceSkeleton Matrix End
	'''

// ZipLocalIndexInPlace
	def static dispatch generateZipLocalIndexInPlaceSkeleton(SkeletonExpression s, ArrayType a, int processId) '''
		// ZipLocalIndexInPlaceSkeleton Array Start
		«val skel = s.skeleton as ZipLocalIndexInPlaceSkeleton»
		«val obj = s.obj»
		#pragma omp«IF Config.cores > 1» parallel for«ENDIF» simd
		for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «a.sizeLocal(processId)»; ++«Config.var_loop_counter»){
			«obj.name»[«Config.var_loop_counter»] = «generateFunctionCall(skel, s.obj.type, (skel.zipWith.value as CollectionObject).type, processId)»
		}
		// ZipLocalIndexInPlaceSkeleton Array End
	'''
	
	def static dispatch generateZipLocalIndexInPlaceSkeleton(SkeletonExpression s, MatrixType m, int processId) '''
		// ZipLocalIndexInPlaceSkeleton Matrix Start
		«val skel = s.skeleton as ZipLocalIndexInPlaceSkeleton»
		«val obj = s.obj»
		#pragma omp«IF Config.cores > 1» parallel for«ELSE» simd«ENDIF» 
		for(size_t «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «m.rowsLocal»; ++«Config.var_loop_counter_rows»){
			«IF Config.cores > 1»
				#pragma omp simd
			«ENDIF»
			for(size_t «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «m.colsLocal»; ++«Config.var_loop_counter_cols»){
				size_t «Config.var_loop_counter» = «Config.var_loop_counter_rows» * «m.colsLocal» + «Config.var_loop_counter_cols»;
				«obj.name»[«Config.var_loop_counter»] = «generateFunctionCall(skel, m, (skel.zipWith.value as CollectionObject).type, processId)»
			}
		}
		// ZipLocalIndexInPlaceSkeleton Matrix End
	'''


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
	def static generateFoldSkeleton(SkeletonExpression s, CollectionObject co, Object target, int processId) '''
		«val skel = s.skeleton as FoldSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param)»
		mkt::fold«IF co.type.distributionMode == DistributionMode.COPY»_copy«ENDIF»<«co.calculateCollectionType.cppType», «s.getFunctorName(skel.param)»>(«co.name», «target.name», «skel.identity.generateExpression(null, processId)»,«s.getFunctorObjectName(skel.param)»);
	'''

	def static generateReductionSkeleton(SkeletonExpression s, CollectionObject co, Object target, int processId) '''
		«val skel = s.skeleton as ReductionSkeleton»
		«target.name» = mkt::reduce_«(skel.param as ReductionOperation).getName»<«co.calculateCollectionType.cppType»>(«co.name»);
	'''

	def static generateMapReductionSkeleton(SkeletonExpression s, CollectionObject co, Object target, int processId) '''
		«val skel = s.skeleton as MapReductionSkeleton»
		«generateSetValuesInFunctor(s, skel.mapFunction)»
		«target.name» = mkt::map_reduce_«(skel.param as ReductionOperation).getName»<«co.calculateCollectionType.cppType», «skel.mapFunction.calculateType.cppType», «s.getFunctorName(skel.mapFunction)»>(«co.name», «s.getFunctorObjectName(skel.mapFunction)»);
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
	def static generateMapFoldSkeleton(SkeletonExpression s, CollectionObject co, Object target, int processId) '''
		«val skel = s.skeleton as MapFoldSkeleton»
		«generateSetValuesInFunctor(s, (s.skeleton as MapFoldSkeleton).mapFunction)»
		«generateSetValuesInFunctor(s, s.skeleton.param)»
		«IF target.calculateType.isArray»
			mkt::map_fold«IF co.type.distributionMode == DistributionMode.COPY»_copy«ENDIF»<«co.calculateCollectionType.cppType», «target.calculateCollectionType.cppType», «skel.identity.calculateType.cppType», «s.getFunctorName(skel.mapFunction)», «s.getFunctorName(skel.param)»>(«co.name», «target.name», «s.getFunctorObjectName(skel.mapFunction)», «skel.identity.generateExpression(null, processId)», «s.getFunctorObjectName(skel.param)»);
		«ELSE»
			mkt::map_fold«IF co.type.distributionMode == DistributionMode.COPY»_copy«ENDIF»<«co.calculateCollectionType.cppType», «skel.identity.calculateType.cppType», «s.getFunctorName(skel.mapFunction)», «s.getFunctorName(skel.param)»>(«co.name», «target.name», «s.getFunctorObjectName(skel.mapFunction)», «skel.identity.generateExpression(null, processId)», «s.getFunctorObjectName(skel.param)»);
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
	def static generateShiftPartitionsHorizontallySkeleton(SkeletonExpression se, int processId) '''		
		«val skel = se.skeleton as ShiftPartitionsHorizontallySkeleton»
		«generateSetValuesInFunctor(se, skel.param)»
		mkt::shift_partitions_horizontally<«se.obj.calculateCollectionType.cppType», «se.getFunctorName(skel.param)»>(«se.obj.name», «se.getFunctorObjectName(skel.param)»);
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
	def static generateShiftPartitionsVerticallySkeleton(SkeletonExpression se, int processId) '''		
		«val skel = se.skeleton as ShiftPartitionsVerticallySkeleton»
		«generateSetValuesInFunctor(se, skel.param)»
		mkt::shift_partitions_vertically<«se.obj.calculateCollectionType.cppType», «se.getFunctorName(skel.param)»>(«se.obj.name», «se.getFunctorObjectName(skel.param)»);
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
	def static  generateGatherSkeleton(SkeletonExpression se, Object output, int processId) '''
		«IF se.obj.calculateType.array»
			mkt::gather<«se.obj.calculateCollectionType.cppType»>(«se.obj.name», «output.name»);
		«ELSE»
			mkt::gather<«se.obj.calculateCollectionType.cppType»>(«se.obj.name», «output.name», «se.obj.name»_partition_type_resized);
		«ENDIF»		
	'''
	
	def static generateScatterSkeleton(SkeletonExpression se, Object output, int processId) '''
		mkt::scatter<«se.obj.calculateCollectionType.cppType»>(«se.obj.name», «output.name»);
	'''
}
