package de.wwu.musket.generator.cpu.mpmd

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

import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIAllgather
import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIIrecv
import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIIsend
import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.generateMPIWaitall

import static de.wwu.musket.generator.cpu.mpmd.FunctionGenerator.generateFunctionCall
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.generator.cpu.mpmd.ExpressionGenerator.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import de.wwu.musket.musket.ScatterSkeleton
import org.eclipse.emf.common.util.BasicEList
import de.wwu.musket.musket.Expression
import org.eclipse.emf.common.util.EList
import de.wwu.musket.musket.Function

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
			FoldSkeleton: generateFoldSkeleton(skel, s.obj, (target as Object).name, processId)
			FoldLocalSkeleton: '''// TODO: FoldLocalSkeleton''' // this is future work
			MapFoldSkeleton: generateMapFoldSkeleton(skel, s.obj, (target as Object).name, processId)
			ZipSkeleton:  generateZipSkeleton(s, (target as CollectionObject).name, processId)
			ZipInPlaceSkeleton:  generateZipInPlaceSkeleton(s, processId)
			ZipIndexSkeleton: generateZipIndexSkeleton(s, s.obj.type, (target as CollectionObject).name, processId)		
			ZipLocalIndexSkeleton: generateZipLocalIndexSkeleton(s, s.obj.type, (target as CollectionObject).name, processId)
			ZipIndexInPlaceSkeleton:  generateZipIndexInPlaceSkeleton(s, s.obj.type, processId)
			ZipLocalIndexInPlaceSkeleton: generateZipLocalIndexInPlaceSkeleton(s, s.obj.type, processId)
			ShiftPartitionsHorizontallySkeleton: if(Config.processes > 1){generateShiftPartitionsHorizontallySkeleton(skel, s.obj.type as MatrixType, processId)}
			ShiftPartitionsVerticallySkeleton: if(Config.processes > 1){generateShiftPartitionsVerticallySkeleton(skel, s.obj.type as MatrixType, processId)}
			GatherSkeleton: generateGatherSkeleton(s, skel, s.obj.type, (target as CollectionObject).type, processId)
			ScatterSkeleton: generateScatterSkeleton(s, skel, s.obj.type, (target as CollectionObject).type, processId)
			default: '''// TODO: SkeletonGenerator.generateSkeletonExpression: default case'''
		}
	}

	def static generateSetValuesInFunctor(SkeletonExpression s, Function f){
		var result = ""
		val numberOfFreeParams = getNumberOfFreeParameters(s, f)
		val parameters = s.skeleton.param.functionParameters
		val arguments = s.skeleton.param.functionArguments
		for(var i = 0; i < numberOfFreeParams; i++){
			result += '''«s.functorObjectName».«parameters.get(i).name» = «arguments.get(i).generateExpression(null, 0)»;'''
		}
		return result
	}

	def static generateMapSkeleton(SkeletonExpression s, CollectionObject target, int processId) '''
		«val a = s.obj»
		«val aType = a.calculateCollectionType.cppType»
		«val tType = target.calculateCollectionType.cppType»
		«val skel = s.skeleton as MapSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param.toFunction)»
		mkt::map<«aType», «tType», «s.functorName»>(«a.collectionName», «target.name», «s.functorObjectName»);
	'''
	
	def static generateMapIndexSkeleton(SkeletonExpression s, CollectionObject target, int processId) '''
		«val a = s.obj»
		«val aType = a.calculateCollectionType.cppType»
		«val tType = target.calculateCollectionType.cppType»
		«val skel = s.skeleton as MapIndexSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param.toFunction)»
		mkt::map_index<«aType», «tType», «s.functorName»>(«a.collectionName», «target.name», «s.functorObjectName»);
	'''
	
	def static generateMapLocalIndexSkeleton(SkeletonExpression s, CollectionObject target, int processId) '''
		«val a = s.obj»
		«val aType = a.calculateCollectionType.cppType»
		«val tType = target.calculateCollectionType.cppType»
		«val skel = s.skeleton as MapLocalIndexSkeleton»
		«generateSetValuesInFunctor(s, s.skeleton.param.toFunction)»
		mkt::map_local_index<«aType», «tType», «s.functorName»>(«a.collectionName», «target.name», «s.functorObjectName»);
	'''

	def static generateMapInPlaceSkeleton(SkeletonExpression s, int processId) '''
		«val a = s.obj»
		«generateSetValuesInFunctor(s, s.skeleton.param.toFunction)»
		mkt::map_in_place<«s.obj.calculateCollectionType.cppType», «s.functorName»>(«a.name», «s.functorObjectName»);
	'''

	def static generateMapIndexInPlaceSkeleton(SkeletonExpression s, CollectionObject co, int processId) '''
		«generateSetValuesInFunctor(s, s.skeleton.param.toFunction)»
		mkt::map_index_in_place<«s.obj.calculateCollectionType.cppType», «s.functorName»>(«co.name», «s.functorObjectName»);
	'''

	def static generateMapLocalIndexInPlaceSkeleton(SkeletonExpression s, CollectionObject co, int processId) '''
		«generateSetValuesInFunctor(s, s.skeleton.param.toFunction)»
		mkt::map_local_index_in_place<«co.calculateCollectionType.cppType», «s.functorName»>(«co.name», «s.functorObjectName»);
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
	def static dispatch generateGatherSkeleton(SkeletonExpression se, GatherSkeleton gs, ArrayType input, ArrayType output, int processId) '''
		«IF Config.processes > 1»
			«generateMPIAllgather(se.obj.name + '.data()', se.obj.type.sizeLocal(processId), se.obj.calculateCollectionType, (output.eContainer as CollectionObject).name + '.data()')»
		«ELSE»
			#pragma omp«IF Config.cores > 1» parallel for «ENDIF»simd
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «output.sizeLocal(processId)»; ++«Config.var_loop_counter»){
				«(output.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «se.obj.name»[«Config.var_loop_counter»];
			}
		«ENDIF»
	'''
	
	def static dispatch generateGatherSkeleton(SkeletonExpression se, GatherSkeleton gs, MatrixType input, MatrixType output, int processId) '''
		«IF Config.processes > 1»
			MPI_Allgatherv(«se.obj.name».data(), «se.obj.type.sizeLocal(processId)», «se.obj.calculateCollectionType.MPIType», «(output.eContainer as CollectionObject).name».data(), (std::array<int, «Config.processes»>{«FOR i: 0 ..< Config.processes SEPARATOR ', '»1«ENDFOR»}).data(), (std::array<int, «Config.processes»>{«FOR i: 0 ..< Config.processes SEPARATOR ', '»«input.sizeLocal(i) * input.partitionPosition(i).key + input.partitionPosition(i).value»«ENDFOR»}).data(), «se.obj.name»_partition_type_resized, MPI_COMM_WORLD);
			//«generateMPIAllgather(se.obj.name + '.data()', se.obj.type.sizeLocal(processId), se.obj.calculateCollectionType, (output.eContainer as CollectionObject).name + '.data()', 1l, input)»
		«ELSE»
			#pragma omp«IF Config.cores > 1» parallel for «ENDIF»simd
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «output.sizeLocal(processId)»; ++«Config.var_loop_counter»){
				«(output.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «se.obj.name»[«Config.var_loop_counter»];
			}
		«ENDIF»
	'''
	
	def static dispatch generateScatterSkeleton(SkeletonExpression se, ScatterSkeleton gs, ArrayType input, ArrayType output, int processId) '''
		«IF Config.processes > 1»
			«Config.var_elem_offset» = «output.globalOffset(processId)»;
			#pragma omp«IF Config.cores > 1» parallel for «ENDIF»simd
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «output.sizeLocal(processId)»; ++«Config.var_loop_counter»){
				«(output.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «se.obj.name»[«Config.var_elem_offset» + «Config.var_loop_counter»];
			}
		«ELSE»
			#pragma omp«IF Config.cores > 1» parallel for «ENDIF»simd
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «output.sizeLocal(processId)»; ++«Config.var_loop_counter»){
				«(output.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «se.obj.name»[«Config.var_loop_counter»];
			}
		«ENDIF»
	'''
	
	def static dispatch generateScatterSkeleton(SkeletonExpression se, ScatterSkeleton gs, MatrixType input, MatrixType output, int processId) '''
		«IF Config.processes > 1»
			«Config.var_row_offset» = «output.globalRowOffset(processId)»;
			«Config.var_col_offset» = «output.globalColOffset(processId)»;
			#pragma omp«IF Config.cores > 1» parallel for «ENDIF»simd
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «output.sizeLocal(processId)»; ++«Config.var_loop_counter»){
				«(output.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «se.obj.name»[(«Config.var_loop_counter» / «output.colsLocal») * «input.cols.concreteValue» + «Config.var_row_offset» * «input.cols.concreteValue» + («Config.var_loop_counter» % «output.colsLocal» + «Config.var_col_offset»)];
			}
		«ELSE»
			#pragma omp«IF Config.cores > 1» parallel for «ENDIF»simd
			for(size_t «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «se.obj.type.sizeLocal(processId)»; ++«Config.var_loop_counter»){
				«(output.eContainer as CollectionObject).name»[«Config.var_loop_counter»] = «se.obj.name»[«Config.var_loop_counter»];
			}
		«ENDIF»
	'''
}
