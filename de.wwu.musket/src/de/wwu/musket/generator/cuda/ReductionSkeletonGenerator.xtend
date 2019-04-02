package de.wwu.musket.generator.cuda

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

import static extension de.wwu.musket.generator.cuda.ExpressionGenerator.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static extension de.wwu.musket.generator.extensions.StringExtension.*
import static extension de.wwu.musket.generator.cuda.util.DataHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.CollectionHelper.*
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.Assignment
import de.wwu.musket.musket.MusketAssignment
import de.wwu.musket.generator.cuda.lib.Musket
import de.wwu.musket.musket.ReductionSkeleton
import de.wwu.musket.musket.ReductionOperation
import de.wwu.musket.util.MusketType
import de.wwu.musket.musket.PlusReduction
import de.wwu.musket.musket.MultiplyReduction
import de.wwu.musket.musket.MaxReduction
import de.wwu.musket.musket.MinReduction
import de.wwu.musket.musket.MapReductionSkeleton

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
	'''
	
	def static generateMapReductionSkeletonArrayFunctionDeclarations() '''
		template<typename T, typename R, typename Functor>
		R map_reduce_plus(mkt::DArray<T>& a, Functor f);
		
		template<typename T, typename R, typename Functor>
		R map_reduce_multiply(mkt::DArray<T>& a, Functor f);
				
		template<typename T, typename R, typename Functor>
		R map_reduce_max(mkt::DArray<T>& a, Functor f);
						
		template<typename T, typename R, typename Functor>
		R map_reduce_min(mkt::DArray<T>& a, Functor f);
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
	'''
	
def static generateMapReductionSkeletonMatrixFunctionDeclarations() '''
		template<typename T, typename R, typename Functor>
		R map_reduce_plus(mkt::DMatrix<T>& m, Functor f);
		
		template<typename T, typename R, typename Functor>
		R map_reduce_multiply(mkt::DMatrix<T>& m, Functor f);
				
		template<typename T, typename R, typename Functor>
		R map_reduce_max(mkt::DMatrix<T>& m, Functor f);
						
		template<typename T, typename R, typename Functor>
		R map_reduce_min(mkt::DMatrix<T>& m, Functor f);
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
	
	def static generateMapReductionSkeletonFunctionDefinitions(Resource resource) {
		var result = ''''''
		var typeOperatorPairs = newArrayList
		for(se : resource.SkeletonExpressions.filter[it.skeleton instanceof MapReductionSkeleton]){
			val in_type = se.obj.calculateCollectionType
			val out_type = (se.skeleton as MapReductionSkeleton).mapFunction.calculateType
			val inCPPtype = in_type.cppType
			val outCPPtype = out_type.cppType
			val functorName = se.getFunctorName((se.skeleton as MapReductionSkeleton).mapFunction)
			val operator = (se.skeleton.param as ReductionOperation)
			val operatorName = operator.name.toString
			val pair = inCPPtype -> outCPPtype -> functorName -> operatorName
			if(!typeOperatorPairs.contains(pair)){
				if(resource.Arrays.size() > 0)
					result += generateMapReductionSkeletonFunctionDefinition(in_type, out_type, operator, functorName, "DArray")
				if(resource.Matrices.size() > 0)
					result += generateMapReductionSkeletonFunctionDefinition(in_type, out_type, operator, functorName, "DMatrix")
				typeOperatorPairs.add(pair)
			}			
		}
		return result
	}
	
	// TODO also special cases for array and structs
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
				if(a.get_device_distribution() == mkt::Distribution::DIST){
					«IF Config.cores > 1»#pragma omp parallel for reduction(«ro.sign»:local_result)«ENDIF»
					for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
						acc_set_device_num(gpu, acc_device_not_host);
						«cppType»* devptr = a.get_device_pointer(gpu);
						const int gpu_elements = a.get_size_gpu();
						«cppType» gpu_result = «getIdentity(type, ro)»;
						
						#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(«ro.sign»:gpu_result) async(0)
						for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
							#pragma acc cache(gpu_result)
							gpu_result = «generateReductionOperation("gpu_result", "devptr[" + Config.var_loop_counter + "]" , ro)»;
						}
						acc_wait(0);
						local_result = «generateReductionOperation("local_result", "gpu_result" , ro)»;
					}
				}else if(a.get_device_distribution() == mkt::Distribution::COPY){
					acc_set_device_num(0, acc_device_not_host);
					«cppType»* devptr = a.get_device_pointer(0);
					const int gpu_elements = a.get_size_gpu();
					
					#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(«ro.sign»:local_result) async(0)
					for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
						#pragma acc cache(local_result)
						local_result = «generateReductionOperation("local_result", "devptr[" + Config.var_loop_counter + "]" , ro)»;
					}
					acc_wait(0);
				}
			«ELSE»
				acc_set_device_num(0, acc_device_not_host);
				«cppType»* devptr = a.get_device_pointer(0);
				const int gpu_elements = a.get_size_gpu();
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(«ro.sign»:local_result) async(0)
				for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
					#pragma acc cache(local_result)					
					local_result = «generateReductionOperation("local_result", "devptr[" + Config.var_loop_counter + "]" , ro)»;
				}
				acc_wait(0);
			«ENDIF»
			
			«IF Config.processes > 1»
				if(a.get_distribution() == mkt::Distribution::DIST){
					MPI_Allreduce(&local_result, &global_result, 1, «mpiType», «ro.MPIReduction», MPI_COMM_WORLD);
					return global_result;
				}else if(a.get_distribution() == mkt::Distribution::COPY){
					return local_result;
				}
			«ELSE»
				return local_result;
			«ENDIF»
		}
	'''
	
	def static generateMapReductionSkeletonFunctionDefinition(MusketType input_type, MusketType output_type, ReductionOperation ro, String functorType, String dataStructure) '''
		«val in_cppType = input_type.cppType»
		«val out_cppType = output_type.cppType»
		«val mpiType = output_type.MPIType»
		template<>
		«out_cppType» mkt::map_reduce_«ro.getName»<«in_cppType», «out_cppType», «functorType»>(mkt::«dataStructure»<«in_cppType»>& a, «functorType» f){
			«IF output_type.array»
				«generateMapReductionArray(input_type, output_type, ro)»
			«ELSEIF output_type.struct»
				«generateMapReductionStruct()»
			«ELSE»
				«generateMapReductionScalar(input_type, output_type, ro)»
			«ENDIF»
		}
	'''
	
	def static generateMapReductionScalar(MusketType input_type, MusketType output_type, ReductionOperation ro)'''
		«val in_cppType = input_type.cppType»
		«val out_cppType = output_type.cppType»
		«val mpiType = output_type.MPIType»
		«out_cppType» local_result = «getIdentity(output_type, ro)»;
		«IF Config.processes > 1»
			«out_cppType» global_result = «getIdentity(output_type, ro)»;
		«ENDIF»
		
		«IF Config.gpus > 1»
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
					acc_set_device_num(gpu, acc_device_not_host);
					f.init(gpu);
					«in_cppType»* devptr = a.get_device_pointer(gpu);
					const int gpu_elements = a.get_size_gpu();
					«out_cppType» gpu_result = «getIdentity(output_type, ro)»;
					
					#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(«ro.sign»:gpu_result) async(0)
					for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
						#pragma acc cache(gpu_result, devptr[0:gpu_elements])
						«out_cppType» map_result = f(devptr[«Config.var_loop_counter»]);
						gpu_result = «generateReductionOperation("gpu_result", "map_result", ro)»;
					}
					acc_wait(0);
					local_result = «generateReductionOperation("local_result", "gpu_result", ro)»;
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){
				acc_set_device_num(0, acc_device_not_host);
				f.init(0);
				«in_cppType»* devptr = a.get_device_pointer(0);
				const int gpu_elements = a.get_size_gpu();
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(«ro.sign»:local_result) async(0)
				for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
					#pragma acc cache(local_result, devptr[0:gpu_elements])
					«out_cppType» map_result = f(devptr[«Config.var_loop_counter»]);
					local_result = «generateReductionOperation("local_result", "map_result", ro)»;
				}
				acc_wait(0);
			}
		«ELSE»
			acc_set_device_num(0, acc_device_not_host);
			«in_cppType»* devptr = a.get_device_pointer(0);
			f.init(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(«ro.sign»:local_result) async(0)
			for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < gpu_elements; ++«Config.var_loop_counter») {
				#pragma acc cache(local_result, devptr[0:gpu_elements])
				«out_cppType» map_result = f(devptr[«Config.var_loop_counter»]);
				local_result = «generateReductionOperation("local_result", "map_result", ro)»;
			}
			acc_wait(0);
		«ENDIF»
		
		«IF Config.processes > 1»
			if(a.get_distribution() == mkt::Distribution::DIST){
				MPI_Allreduce(&local_result, &global_result, 1, «mpiType», «ro.MPIReduction», MPI_COMM_WORLD);
				return global_result;
			}else if(a.get_distribution() == mkt::Distribution::COPY){
				return local_result;
			}				
		«ELSE»
			return local_result;
		«ENDIF»
	'''
	
	def static generateMapReductionArray(MusketType input_type, MusketType output_type, ReductionOperation ro)'''
		«val in_cppType = input_type.cppType»
		«val out_cppType = output_type.cppType»
		«val scalar_out_cppType = output_type.calculateCollectionType.cppType»
		«val out_size = output_type.size»
		«val mpiType = output_type.MPIType»
		«out_cppType» local_result;
		local_result.fill(«getIdentity(output_type, ro)»);
		«IF Config.processes > 1»
			«out_cppType» global_result;
			global_result.fill(«getIdentity(output_type, ro)»);
		«ENDIF»
		
		«IF Config.gpus > 1»
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				for(int gpu = 0; gpu < «Config.gpus»; ++gpu){
					acc_set_device_num(gpu, acc_device_not_host);
					f.init(gpu);
					«in_cppType»* devptr = a.get_device_pointer(gpu);
					const int gpu_elements = a.get_size_gpu();
					«out_cppType» gpu_result;
					gpu_result.fill(«getIdentity(output_type, ro)»);
					
					#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) async(0)
					for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «out_size»; ++«Config.var_loop_counter») {
						«scalar_out_cppType» element_result = «getIdentity(output_type, ro)»;
						#pragma acc loop reduction(«ro.sign»:element_result)
						for(unsigned int inner_«Config.var_loop_counter» = 0; inner_«Config.var_loop_counter» < gpu_elements; ++inner_«Config.var_loop_counter») {
							«scalar_out_cppType» map_result = (f(devptr[inner_«Config.var_loop_counter»]))[«Config.var_loop_counter»]; // this is actually calculate more often than necessary
							element_result = «generateReductionOperation("element_result", "map_result", ro)»;
						}
						gpu_result[«Config.var_loop_counter»] = «generateReductionOperation("gpu_result[" + Config.var_loop_counter + "]", "element_result", ro)»;
					}
					acc_wait(0);
					
					for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «out_size»; ++«Config.var_loop_counter»){
						local_result[«Config.var_loop_counter»] = «generateReductionOperation("local_result[" + Config.var_loop_counter + "]", "gpu_result[" + Config.var_loop_counter + "]", ro)»;
					}
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){
				acc_set_device_num(0, acc_device_not_host);
				f.init(0);
				«in_cppType»* devptr = a.get_device_pointer(0);
				const int gpu_elements = a.get_size_gpu();
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) async(0)
				for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «out_size»; ++«Config.var_loop_counter») {
					«scalar_out_cppType» element_result = «getIdentity(output_type, ro)»;
					#pragma acc loop reduction(«ro.sign»:element_result)
					for(unsigned int inner_«Config.var_loop_counter» = 0; inner_«Config.var_loop_counter» < gpu_elements; ++inner_«Config.var_loop_counter») {
						«scalar_out_cppType» map_result = (f(devptr[inner_«Config.var_loop_counter»]))[«Config.var_loop_counter»];
						element_result = «generateReductionOperation("element_result", "map_result", ro)»;
					}
					local_result[«Config.var_loop_counter»] = «generateReductionOperation("local_result[" + Config.var_loop_counter + "]", "element_result", ro)»;
				}
				acc_wait(0);
			}
		«ELSE»
			acc_set_device_num(0, acc_device_not_host);
			f.init(0);
			«in_cppType»* devptr = a.get_device_pointer(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) async(0)
			for(unsigned int «Config.var_loop_counter» = 0; «Config.var_loop_counter» < «out_size»; ++«Config.var_loop_counter») {
				«scalar_out_cppType» element_result = «getIdentity(output_type, ro)»;
				#pragma acc loop reduction(«ro.sign»:element_result)
				for(unsigned int inner_«Config.var_loop_counter» = 0; inner_«Config.var_loop_counter» < gpu_elements; ++inner_«Config.var_loop_counter») {
					«scalar_out_cppType» map_result = (f(devptr[inner_«Config.var_loop_counter»]))[«Config.var_loop_counter»];
					element_result = «generateReductionOperation("element_result", "map_result", ro)»;
				}
				local_result[«Config.var_loop_counter»] = «generateReductionOperation("local_result[" + Config.var_loop_counter + "]", "element_result", ro)»;
			}
			acc_wait(0);
		«ENDIF»
		
		«IF Config.processes > 1»
			if(a.get_distribution() == mkt::Distribution::DIST){
				MPI_Allreduce(local_result.data(), global_result.data(), «out_size», «mpiType», «ro.MPIReduction», MPI_COMM_WORLD);
				return global_result;
			}else if(a.get_distribution() == mkt::Distribution::COPY){
				return local_result;
			}				
		«ELSE»
			return local_result;
		«ENDIF»
	'''
	
	def static generateMapReductionStruct()'''
«««	TODO
	'''
	
	def static generateReductionOperation(String result, String input, ReductionOperation ro){
		switch ro{
			PlusReduction: '''«result» + «input»'''
			MultiplyReduction: '''«result» * «input»'''
			MaxReduction: '''«result» > «input» ? «result» : «input»'''
			MinReduction: '''«result» < «input» ? «result» : «input»'''
			default: ''''''
		}
	}
	
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
