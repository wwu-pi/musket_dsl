package de.wwu.musket.generator.gpu

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

import static extension de.wwu.musket.generator.gpu.ExpressionGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.gpu.util.DataHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.CollectionHelper.*
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.generator.gpu.lib.Musket
import de.wwu.musket.musket.ReductionSkeleton
import de.wwu.musket.musket.ReductionOperation
import de.wwu.musket.util.MusketType
import de.wwu.musket.musket.PlusReduction
import de.wwu.musket.musket.MultiplyReduction
import de.wwu.musket.musket.MaxReduction
import de.wwu.musket.musket.MinReduction

class ReductionSkeletonGenerator {

	def static generateReductionSkeletonArrayFunctionDeclarations() '''
		template<typename T>
		T reduce_plus(mkt::DArray<T>& a);
		
		template<typename T>
		T reduce_multiply(mkt::DArray<T>& a);
				
		template<typename T>
		T reduce_max(mkt::DArray<T>& a);
						
		template<typename T>
		T reduce_min(mkt::DArray<T>& a);
		
		template<typename T>
		T reduce_plus_copy(mkt::DArray<T>& a);
		
		template<typename T>
		T reduce_multiply_copy(mkt::DArray<T>& a);
				
		template<typename T>
		T reduce_max_copy(mkt::DArray<T>& a);
						
		template<typename T>
		T reduce_min_copy(mkt::DArray<T>& a);
	'''
	
	def static generateReductionSkeletonMatrixFunctionDeclarations() '''
		template<typename T>
		T reduce_plus(mkt::DMatrix<T>& m);
		
		template<typename T>
		T reduce_multiply(mkt::DMatrix<T>& m);
				
		template<typename T>
		T reduce_max(mkt::DMatrix<T>& m);
						
		template<typename T>
		T reduce_min(mkt::DMatrix<T>& m);
		
		template<typename T>
		T reduce_plus_copy(mkt::DMatrix<T>& m);
		
		template<typename T>
		T reduce_multiply_copy(mkt::DMatrix<T>& m);
				
		template<typename T>
		T reduce_max_copy(mkt::DMatrix<T>& m);
						
		template<typename T>
		T reduce_min_copy(mkt::DMatrix<T>& m);
	'''
	
	def static generateReductionSkeletonFunctionDefinitions(Resource resource) {
		var result = ''''''
		var typeOperatorPairs = newArrayList
		for(se : resource.SkeletonExpressions.filter[it.skeleton instanceof ReductionSkeleton]){
			val mktType = se.obj.calculateCollectionType
			val type = mktType.cppType
			val operator = (se.skeleton.param as ReductionOperation)
			val operatorName = operator.name.toString
			val pair = type -> operatorName
			if(!typeOperatorPairs.contains(pair)){
				if(resource.Arrays.size() > 0)
					result += generateReductionSkeletonFunctionDefinition(mktType, operator, "DArray")
				if(resource.Matrices.size() > 0)
					result += generateReductionSkeletonFunctionDefinition(mktType, operator, "DMatrix")
				typeOperatorPairs.add(pair)
			}			
		}
		return result
	}
	
	def static generateReductionSkeletonFunctionDefinition(MusketType type, ReductionOperation ro, String dataStructure) '''
		«val cppType = type.cppType»
		«val mpiType = type.MPIType»
		template<>
		«cppType» mkt::reduce_«ro.getName»<«cppType»>(mkt::«dataStructure»<«cppType»>& a){
			«cppType» local_result = «getIdentity(type, ro)»;
			«IF Config.processes > 1»
				«cppType» global_result = «getIdentity(type, ro)»;
			«ENDIF»
			
			«IF Config.gpus > 1»
				«IF Config.cores > 1»#pragma omp parallel for reduction(«ro.sign»:local_result)«ENDIF»
				for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
					acc_set_device_num(gpu, acc_device_not_host);
					«cppType»* devptr = a.get_device_pointer(gpu);
					const int gpu_elements = a.get_size_gpu();
					«cppType» gpu_result = «getIdentity(type, ro)»;
					
					#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(«ro.sign»:gpu_result) async(0)
					for (int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
						#pragma acc cache(gpu_result)
						gpu_result = gpu_result + devptr[«Config.var_loop_counter»];
					}
					acc_wait(0);
					local_result = local_result + gpu_result;
				}
			«ELSE»
				acc_set_device_num(0, acc_device_not_host);
				«cppType»* devptr = a.get_device_pointer(0);
				const int gpu_elements = a.get_size_gpu();
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(«ro.sign»:local_result) async(0)
				for(int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
					#pragma acc cache(local_result)
					local_result = local_result + devptr[«Config.var_loop_counter»];
				}
				acc_wait(0);
			«ENDIF»
			
			«IF Config.processes > 1»
				MPI_Allreduce(&local_result, &global_result, 1, «mpiType», «ro.MPIReduction», MPI_COMM_WORLD);
				return global_result;
			«ELSE»
				return local_result;
			«ENDIF»
		}
	'''
	
	def static getIdentity(MusketType type, ReductionOperation ro) {
		if(type.type.literal == 'int' && ro instanceof PlusReduction)
			'''0'''
		else if(type.type.literal == "int" && ro instanceof MultiplyReduction)
			'''1'''
		else if(type.type.literal == "int" && ro instanceof MaxReduction)
			'''std::numeric_limits<int>::lowest()'''
		else if(type.type.literal == "int" && ro instanceof MinReduction)
			'''std::numeric_limits<int>::max()'''
		else if(type.type.literal == "float" && ro instanceof PlusReduction)
			'''0.0f'''
		else if(type.type.literal == "float" && ro instanceof MultiplyReduction)
			'''1.0f'''
		else if(type.type.literal == "float" && ro instanceof MaxReduction)
			'''std::numeric_limits<float>::lowest()'''
		else if(type.type.literal == "float" && ro instanceof MinReduction)
			'''std::numeric_limits<float>::max()'''
		else if(type.type.literal == "double" && ro instanceof PlusReduction)
			'''0.0'''
		else if(type.type.literal == "double" && ro instanceof MultiplyReduction)
			'''1.0'''
		else if(type.type.literal == "double" && ro instanceof MaxReduction)
			'''std::numeric_limits<double>::lowest()'''
		else if(type.type.literal == "double" && ro instanceof MinReduction)
			'''std::numeric_limits<double>::max()'''
		else
			'''0'''		
	}

}
