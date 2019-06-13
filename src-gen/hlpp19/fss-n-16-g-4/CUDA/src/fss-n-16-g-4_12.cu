	#include <mpi.h>
	#include <cuda.h>
	#include <omp.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <curand_kernel.h>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	
	
	#include "../include/musket.cuh"
	#include "../include/fss-n-16-g-4_12.cuh"
	
	const size_t number_of_processes = 16;
	const size_t process_id = 12;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	
	const double PI = 3.141592653589793;
	const double EULER = 2.718281828459045;
	const double UPPER_BOUND = 5.12;
	const double LOWER_BOUND = -5.12;
	const double PROBLEM_RANGE = 10.24;
	const double INIT_UPPER_BOUND = 5.12;
	const double INIT_LOWER_BOUND = -5.12;
	const double WEIGHT_UPPER_BOUND = 5000.0;
	const double WEIGHT_LOWER_BOUND = 1.0;
	const double STEP_SIZE_INITIAL = 0.1;
	const double STEP_SIZE_FINAL = 1.0E-5;
	const double STEP_SIZE_VOLITIVE_INITIAL = 0.2;
	const double STEP_SIZE_VOLITIVE_FINAL = 2.0E-5;
	const int NUMBER_OF_FISH = 2048;
	const int ITERATIONS = 5000;
	const int DIMENSIONS = 512;
	
	//Fish::Fish() : position(0, 0.0), fitness(), candidate_position(0, 0.0), candidate_fitness(), displacement(0, 0.0), fitness_variation(), weight(), best_position(0, 0.0), best_fitness() {}
	

	
	struct InitFish_map_in_place_array_functor{
		
		InitFish_map_in_place_array_functor(){}
		
		~InitFish_map_in_place_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			curandState_t curand_state; // performance could be improved by creating states before
			size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			curand_init(clock64(), id, 0, &curand_state);
			fi.fitness = std::numeric_limits<double>::lowest();
			fi.candidate_fitness = std::numeric_limits<double>::lowest();
			fi.weight = (WEIGHT_LOWER_BOUND);
			fi.fitness_variation = 0.0;
			fi.best_fitness = std::numeric_limits<double>::lowest();
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				fi.position[(i)] = static_cast<double>(curand_uniform(&curand_state) * ((INIT_UPPER_BOUND) - (INIT_LOWER_BOUND)) + (INIT_LOWER_BOUND));
				fi.candidate_position[(i)] = 0.0;
				fi.displacement[(i)] = 0.0;
				fi.best_position[(i)] = 0.0;
			}
			return (fi);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct EvaluateFitness_map_in_place_array_functor{
		
		EvaluateFitness_map_in_place_array_functor(){}
		
		~EvaluateFitness_map_in_place_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			double sum = 0.0;
			for(int j = 0; ((j) < (DIMENSIONS)); ++j){
				double value = (fi).position[(j)];
				sum += (std::pow((value), 2) - (10 * std::cos(((2 * (PI)) * (value)))));
			}
			fi.fitness = -(((10 * (DIMENSIONS)) + (sum)));
			
			if(((fi).fitness > (fi).best_fitness)){
			fi.best_fitness = (fi).fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				fi.best_position[(k)] = (fi).position[(k)];
			}
			}
			return (fi);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct IndividualMovement_map_in_place_array_functor{
		
		IndividualMovement_map_in_place_array_functor(){}
		
		~IndividualMovement_map_in_place_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			curandState_t curand_state; // performance could be improved by creating states before
			size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			curand_init(clock64(), id, 0, &curand_state);
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double rand_factor = static_cast<double>(curand_uniform(&curand_state) * (1.0 - -(1.0)) + -(1.0));
				double direction = (((rand_factor) * (step_size)) * ((UPPER_BOUND) - (LOWER_BOUND)));
				double new_value = ((fi).position[(i)] + (direction));
				
				if(((new_value) < (LOWER_BOUND))){
				new_value = (LOWER_BOUND);
				} else 
				if(((new_value) > (UPPER_BOUND))){
				new_value = (UPPER_BOUND);
				}
				fi.candidate_position[(i)] = (new_value);
			}
			double sum = 0.0;
			for(int j = 0; ((j) < (DIMENSIONS)); ++j){
				double value = (fi).candidate_position[(j)];
				sum += (std::pow((value), 2) - (10 * std::cos(((2 * (PI)) * (value)))));
			}
			fi.candidate_fitness = -(((10 * (DIMENSIONS)) + (sum)));
			
			if(((fi).candidate_fitness > (fi).fitness)){
			fi.fitness_variation = ((fi).candidate_fitness - (fi).fitness);
			fi.fitness = (fi).candidate_fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				fi.displacement[(k)] = ((fi).candidate_position[(k)] - (fi).position[(k)]);
				fi.position[(k)] = (fi).candidate_position[(k)];
			}
			
			if(((fi).fitness > (fi).best_fitness)){
			fi.best_fitness = (fi).fitness;
			for(int k = 0; ((k) < (DIMENSIONS)); ++k){
				fi.best_position[(k)] = (fi).position[(k)];
			}
			}
			}
			 else {
					fi.fitness_variation = 0.0;
					for(int k = 0; ((k) < (DIMENSIONS)); ++k){
						fi.displacement[(k)] = 0.0;
					}
				}
			return (fi);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		double step_size;
		
	};
	struct Feeding_map_in_place_array_functor{
		
		Feeding_map_in_place_array_functor(){}
		
		~Feeding_map_in_place_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			
			if(((max_fitness_variation) != 0.0)){
			double result = ((fi).weight + ((fi).fitness_variation / (max_fitness_variation)));
			
			if(((result) > (WEIGHT_UPPER_BOUND))){
			result = (WEIGHT_UPPER_BOUND);
			} else 
			if(((result) < (WEIGHT_LOWER_BOUND))){
			result = (WEIGHT_LOWER_BOUND);
			}
			fi.weight = (result);
			}
			return (fi);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		double max_fitness_variation;
		
	};
	struct CalcDisplacementMap_map_in_place_array_functor{
		
		CalcDisplacementMap_map_in_place_array_functor(){}
		
		~CalcDisplacementMap_map_in_place_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				fi.displacement[(i)] *= (fi).fitness_variation;
			}
			return (fi);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct CalcInstinctiveMovementVector_map_in_place_array_functor{
		
		CalcInstinctiveMovementVector_map_in_place_array_functor(){}
		
		~CalcInstinctiveMovementVector_map_in_place_array_functor() {}
		
		__device__
		auto operator()(double x){
			double result = (x);
			
			if(((sum_fitness_variation) != 0.0)){
			result = ((x) / (sum_fitness_variation));
			}
			x = (result);
			return (x);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		double sum_fitness_variation;
		
	};
	struct InstinctiveMovement_map_in_place_array_functor{
		
		InstinctiveMovement_map_in_place_array_functor(const mkt::DArray<double>& _instinctive_movement_vector_copy) : instinctive_movement_vector_copy(_instinctive_movement_vector_copy){}
		
		~InstinctiveMovement_map_in_place_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double new_position = ((fi).position[(i)] + instinctive_movement_vector_copy.get_data_local((i)));
				
				if(((new_position) < (LOWER_BOUND))){
				new_position = (LOWER_BOUND);
				} else 
				if(((new_position) > (UPPER_BOUND))){
				new_position = (UPPER_BOUND);
				}
				fi.position[(i)] = (new_position);
			}
			return (fi);
		}
	
		void init(int device){
			instinctive_movement_vector_copy.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
		mkt::DeviceArray<double> instinctive_movement_vector_copy;
	};
	struct CalcWeightedFish_map_array_functor{
		
		CalcWeightedFish_map_array_functor(){}
		
		~CalcWeightedFish_map_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				fi.position[(i)] *= (fi).weight;
			}
			return (fi);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct CalcBarycenterMap_map_in_place_array_functor{
		
		CalcBarycenterMap_map_in_place_array_functor(){}
		
		~CalcBarycenterMap_map_in_place_array_functor() {}
		
		__device__
		auto operator()(double x){
			double result = (x);
			
			if(((sum_weight) != 0)){
			result = ((x) / (sum_weight));
			}
			x = (result);
			return (x);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		double sum_weight;
		
	};
	struct VolitiveMovement_map_in_place_array_functor{
		
		VolitiveMovement_map_in_place_array_functor(const mkt::DArray<double>& _barycenter_copy) : barycenter_copy(_barycenter_copy){}
		
		~VolitiveMovement_map_in_place_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			curandState_t curand_state; // performance could be improved by creating states before
			size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			curand_init(clock64(), id, 0, &curand_state);
			double distance = 0.0;
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				distance += (((fi).position[(i)] - barycenter_copy.get_data_local((i))) * ((fi).position[(i)] - barycenter_copy.get_data_local((i))));
			}
			distance = sqrt((distance));
			
			if(((distance) != 0.0)){
			double rand_factor = static_cast<double>(curand_uniform(&curand_state) * (1.0 - 0.0) + 0.0);
			for(int i = 0; ((i) < (DIMENSIONS)); ++i){
				double direction = ((((rand_factor) * (step_size)) * ((UPPER_BOUND) - (LOWER_BOUND))) * (((fi).position[(i)] - barycenter_copy.get_data_local((i))) / (distance)));
				double new_position = (fi).position[(i)];
				
				if(((sum_weight) > (sum_weight_last_iteration))){
				new_position -= (direction);
				}
				 else {
						new_position += (direction);
					}
				
				if(((new_position) < (LOWER_BOUND))){
				new_position = (LOWER_BOUND);
				} else 
				if(((new_position) > (UPPER_BOUND))){
				new_position = (UPPER_BOUND);
				}
				fi.position[(i)] = (new_position);
			}
			}
			return (fi);
		}
	
		void init(int device){
			barycenter_copy.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		double step_size;
		double sum_weight;
		double sum_weight_last_iteration;
		
		mkt::DeviceArray<double> barycenter_copy;
	};
	struct Lambda56_map_reduce_array_functor{
		
		Lambda56_map_reduce_array_functor(){}
		
		~Lambda56_map_reduce_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			return (fi).weight;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Lambda57_map_reduce_array_functor{
		
		Lambda57_map_reduce_array_functor(){}
		
		~Lambda57_map_reduce_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			return (fi).fitness_variation;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Lambda58_map_reduce_array_functor{
		
		Lambda58_map_reduce_array_functor(){}
		
		~Lambda58_map_reduce_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			return (fi).fitness_variation;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Lambda59_map_reduce_array_functor{
		
		Lambda59_map_reduce_array_functor(){}
		
		~Lambda59_map_reduce_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			return (fi).displacement;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Lambda60_map_reduce_array_functor{
		
		Lambda60_map_reduce_array_functor(){}
		
		~Lambda60_map_reduce_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			return (fi).weight;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Lambda61_map_reduce_array_functor{
		
		Lambda61_map_reduce_array_functor(){}
		
		~Lambda61_map_reduce_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			return (fi).position;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Lambda62_map_reduce_array_functor{
		
		Lambda62_map_reduce_array_functor(){}
		
		~Lambda62_map_reduce_array_functor() {}
		
		__device__
		auto operator()(Fish fi){
			return (fi).best_fitness;
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	
	
	
	
	template<>
	double mkt::map_reduce_plus<Fish, double, Lambda56_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda56_map_reduce_array_functor f){
			double local_result = 0.0;
			double global_result = 0.0;
						
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				std::array<double*,4> d_odata;
				std::array<double, 4> gpu_results;
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
		
				for(int gpu = 0; gpu < 4; ++gpu){
					cudaSetDevice(gpu);
					cudaMalloc((void**) &d_odata[gpu], blocks * sizeof(double));
					Fish* devptr = a.get_device_pointer(gpu);
					
					mkt::kernel::map_reduce_plus_call<Fish, double, Lambda56_map_reduce_array_functor>(gpu_elements, devptr, d_odata[gpu], threads, blocks, f, mkt::cuda_streams[gpu], gpu);
				}
				mkt::sync_streams();
				
				// fold on gpus: step 2
				while(blocks > 1){
			      int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
			      int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  for(int gpu = 0; gpu < 4; ++gpu){
				      cudaSetDevice(gpu);
				      mkt::kernel::reduce_plus_call<double>(blocks, d_odata[gpu], d_odata[gpu], threads_2, blocks_2, mkt::cuda_streams[gpu], gpu);
				  }
				  blocks = blocks_2;
			  	  mkt::sync_streams();
			  	}
				
				// copy final sum from device to host
				  for (int gpu = 0; gpu < 4; ++gpu) {
				    cudaSetDevice(gpu);
				    cudaMemcpyAsync(&gpu_results[gpu], d_odata[gpu], sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
				  }
				  mkt::sync_streams();
				  
				  for(int gpu = 0; gpu < 4; ++gpu) {
					cudaSetDevice(gpu);
					cudaFree(d_odata[gpu]);
				  }
				
				for(int gpu = 0; gpu < 4; ++gpu){
					local_result = local_result + gpu_results[gpu];
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){ // use only gpu 0, since all have the same data
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
				cudaSetDevice(0);
				double* d_odata;
				cudaMalloc((void**) &d_odata, blocks * sizeof(double));
				Fish* devptr = a.get_device_pointer(0);
				
				mkt::kernel::map_reduce_plus_call<Fish, double, Lambda56_map_reduce_array_functor>(gpu_elements, devptr, d_odata, threads, blocks, f, mkt::cuda_streams[0], 0);
				
				// fold on gpus: step 2
				while(blocks > 1){
				  int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
				  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  mkt::kernel::reduce_plus_call<double>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
				  blocks = blocks_2;
				}
				
				// copy final sum from device to host
				  cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
				  mkt::sync_streams();
				cudaFree(d_odata);
			}
			
			if(a.get_distribution() == mkt::Distribution::DIST){
				MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				return global_result;
			}else if(a.get_distribution() == mkt::Distribution::COPY){
				return local_result;
			}
	}
	template<>
	double mkt::map_reduce_max<Fish, double, Lambda57_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda57_map_reduce_array_functor f){
			double local_result = std::numeric_limits<double>::lowest();
			double global_result = std::numeric_limits<double>::lowest();
						
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				std::array<double*,4> d_odata;
				std::array<double, 4> gpu_results;
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
		
				for(int gpu = 0; gpu < 4; ++gpu){
					cudaSetDevice(gpu);
					cudaMalloc((void**) &d_odata[gpu], blocks * sizeof(double));
					Fish* devptr = a.get_device_pointer(gpu);
					
					mkt::kernel::map_reduce_max_call<Fish, double, Lambda57_map_reduce_array_functor>(gpu_elements, devptr, d_odata[gpu], threads, blocks, f, mkt::cuda_streams[gpu], gpu);
				}
				mkt::sync_streams();
				
				// fold on gpus: step 2
				while(blocks > 1){
			      int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
			      int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  for(int gpu = 0; gpu < 4; ++gpu){
				      cudaSetDevice(gpu);
				      mkt::kernel::reduce_max_call<double>(blocks, d_odata[gpu], d_odata[gpu], threads_2, blocks_2, mkt::cuda_streams[gpu], gpu);
				  }
				  blocks = blocks_2;
			  	  mkt::sync_streams();
			  	}
				
				// copy final sum from device to host
				  for (int gpu = 0; gpu < 4; ++gpu) {
				    cudaSetDevice(gpu);
				    cudaMemcpyAsync(&gpu_results[gpu], d_odata[gpu], sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
				  }
				  mkt::sync_streams();
				  
				  for(int gpu = 0; gpu < 4; ++gpu) {
					cudaSetDevice(gpu);
					cudaFree(d_odata[gpu]);
				  }
				
				for(int gpu = 0; gpu < 4; ++gpu){
					local_result = local_result > gpu_results[gpu] ? local_result : gpu_results[gpu];
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){ // use only gpu 0, since all have the same data
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
				cudaSetDevice(0);
				double* d_odata;
				cudaMalloc((void**) &d_odata, blocks * sizeof(double));
				Fish* devptr = a.get_device_pointer(0);
				
				mkt::kernel::map_reduce_max_call<Fish, double, Lambda57_map_reduce_array_functor>(gpu_elements, devptr, d_odata, threads, blocks, f, mkt::cuda_streams[0], 0);
				
				// fold on gpus: step 2
				while(blocks > 1){
				  int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
				  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  mkt::kernel::reduce_max_call<double>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
				  blocks = blocks_2;
				}
				
				// copy final sum from device to host
				  cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
				  mkt::sync_streams();
				cudaFree(d_odata);
			}
			
			if(a.get_distribution() == mkt::Distribution::DIST){
				MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				return global_result;
			}else if(a.get_distribution() == mkt::Distribution::COPY){
				return local_result;
			}
	}
	template<>
	double mkt::map_reduce_plus<Fish, double, Lambda58_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda58_map_reduce_array_functor f){
			double local_result = 0.0;
			double global_result = 0.0;
						
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				std::array<double*,4> d_odata;
				std::array<double, 4> gpu_results;
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
		
				for(int gpu = 0; gpu < 4; ++gpu){
					cudaSetDevice(gpu);
					cudaMalloc((void**) &d_odata[gpu], blocks * sizeof(double));
					Fish* devptr = a.get_device_pointer(gpu);
					
					mkt::kernel::map_reduce_plus_call<Fish, double, Lambda58_map_reduce_array_functor>(gpu_elements, devptr, d_odata[gpu], threads, blocks, f, mkt::cuda_streams[gpu], gpu);
				}
				mkt::sync_streams();
				
				// fold on gpus: step 2
				while(blocks > 1){
			      int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
			      int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  for(int gpu = 0; gpu < 4; ++gpu){
				      cudaSetDevice(gpu);
				      mkt::kernel::reduce_plus_call<double>(blocks, d_odata[gpu], d_odata[gpu], threads_2, blocks_2, mkt::cuda_streams[gpu], gpu);
				  }
				  blocks = blocks_2;
			  	  mkt::sync_streams();
			  	}
				
				// copy final sum from device to host
				  for (int gpu = 0; gpu < 4; ++gpu) {
				    cudaSetDevice(gpu);
				    cudaMemcpyAsync(&gpu_results[gpu], d_odata[gpu], sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
				  }
				  mkt::sync_streams();
				  
				  for(int gpu = 0; gpu < 4; ++gpu) {
					cudaSetDevice(gpu);
					cudaFree(d_odata[gpu]);
				  }
				
				for(int gpu = 0; gpu < 4; ++gpu){
					local_result = local_result + gpu_results[gpu];
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){ // use only gpu 0, since all have the same data
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
				cudaSetDevice(0);
				double* d_odata;
				cudaMalloc((void**) &d_odata, blocks * sizeof(double));
				Fish* devptr = a.get_device_pointer(0);
				
				mkt::kernel::map_reduce_plus_call<Fish, double, Lambda58_map_reduce_array_functor>(gpu_elements, devptr, d_odata, threads, blocks, f, mkt::cuda_streams[0], 0);
				
				// fold on gpus: step 2
				while(blocks > 1){
				  int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
				  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  mkt::kernel::reduce_plus_call<double>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
				  blocks = blocks_2;
				}
				
				// copy final sum from device to host
				  cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
				  mkt::sync_streams();
				cudaFree(d_odata);
			}
			
			if(a.get_distribution() == mkt::Distribution::DIST){
				MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				return global_result;
			}else if(a.get_distribution() == mkt::Distribution::COPY){
				return local_result;
			}
	}
	template<>
	std::array<double,512> mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda59_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda59_map_reduce_array_functor f){
		std::array<double,512> local_result;
		local_result.fill(0.0);
		std::array<double,512> global_result;
		global_result.fill(0.0);
		
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			for(int gpu = 0; gpu < 4; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				f.init(gpu);
				Fish* devptr = a.get_device_pointer(gpu);
				const int gpu_elements = a.get_size_gpu();
				std::array<double,512> gpu_result;
				gpu_result.fill(0.0);
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) async(0)
				for(unsigned int counter = 0; counter < 512; ++counter) {
					double element_result = 0.0;
					#pragma acc loop reduction(+:element_result)
					for(unsigned int inner_counter = 0; inner_counter < gpu_elements; ++inner_counter) {
						double map_result = (f(devptr[inner_counter]))[counter]; // this is actually calculate more often than necessary
						element_result = element_result + map_result;
					}
					gpu_result[counter] = gpu_result[counter] + element_result;
				}
				acc_wait(0);
				
				for(unsigned int counter = 0; counter < 512; ++counter){
					local_result[counter] = local_result[counter] + gpu_result[counter];
				}
			}
		}else if(a.get_device_distribution() == mkt::Distribution::COPY){
			acc_set_device_num(0, acc_device_not_host);
			f.init(0);
			Fish* devptr = a.get_device_pointer(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) async(0)
			for(unsigned int counter = 0; counter < 512; ++counter) {
				double element_result = 0.0;
				#pragma acc loop reduction(+:element_result)
				for(unsigned int inner_counter = 0; inner_counter < gpu_elements; ++inner_counter) {
					double map_result = (f(devptr[inner_counter]))[counter];
					element_result = element_result + map_result;
				}
				local_result[counter] = local_result[counter] + element_result;
			}
			acc_wait(0);
		}
		
		if(a.get_distribution() == mkt::Distribution::DIST){
			MPI_Allreduce(local_result.data(), global_result.data(), 512, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			return global_result;
		}else if(a.get_distribution() == mkt::Distribution::COPY){
			return local_result;
		}				
		 // TODO
	}
	template<>
	double mkt::map_reduce_plus<Fish, double, Lambda60_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda60_map_reduce_array_functor f){
			double local_result = 0.0;
			double global_result = 0.0;
						
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				std::array<double*,4> d_odata;
				std::array<double, 4> gpu_results;
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
		
				for(int gpu = 0; gpu < 4; ++gpu){
					cudaSetDevice(gpu);
					cudaMalloc((void**) &d_odata[gpu], blocks * sizeof(double));
					Fish* devptr = a.get_device_pointer(gpu);
					
					mkt::kernel::map_reduce_plus_call<Fish, double, Lambda60_map_reduce_array_functor>(gpu_elements, devptr, d_odata[gpu], threads, blocks, f, mkt::cuda_streams[gpu], gpu);
				}
				mkt::sync_streams();
				
				// fold on gpus: step 2
				while(blocks > 1){
			      int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
			      int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  for(int gpu = 0; gpu < 4; ++gpu){
				      cudaSetDevice(gpu);
				      mkt::kernel::reduce_plus_call<double>(blocks, d_odata[gpu], d_odata[gpu], threads_2, blocks_2, mkt::cuda_streams[gpu], gpu);
				  }
				  blocks = blocks_2;
			  	  mkt::sync_streams();
			  	}
				
				// copy final sum from device to host
				  for (int gpu = 0; gpu < 4; ++gpu) {
				    cudaSetDevice(gpu);
				    cudaMemcpyAsync(&gpu_results[gpu], d_odata[gpu], sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
				  }
				  mkt::sync_streams();
				  
				  for(int gpu = 0; gpu < 4; ++gpu) {
					cudaSetDevice(gpu);
					cudaFree(d_odata[gpu]);
				  }
				
				for(int gpu = 0; gpu < 4; ++gpu){
					local_result = local_result + gpu_results[gpu];
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){ // use only gpu 0, since all have the same data
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
				cudaSetDevice(0);
				double* d_odata;
				cudaMalloc((void**) &d_odata, blocks * sizeof(double));
				Fish* devptr = a.get_device_pointer(0);
				
				mkt::kernel::map_reduce_plus_call<Fish, double, Lambda60_map_reduce_array_functor>(gpu_elements, devptr, d_odata, threads, blocks, f, mkt::cuda_streams[0], 0);
				
				// fold on gpus: step 2
				while(blocks > 1){
				  int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
				  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  mkt::kernel::reduce_plus_call<double>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
				  blocks = blocks_2;
				}
				
				// copy final sum from device to host
				  cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
				  mkt::sync_streams();
				cudaFree(d_odata);
			}
			
			if(a.get_distribution() == mkt::Distribution::DIST){
				MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				return global_result;
			}else if(a.get_distribution() == mkt::Distribution::COPY){
				return local_result;
			}
	}
	template<>
	std::array<double,512> mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda61_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda61_map_reduce_array_functor f){
		std::array<double,512> local_result;
		local_result.fill(0.0);
		std::array<double,512> global_result;
		global_result.fill(0.0);
		
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			for(int gpu = 0; gpu < 4; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				f.init(gpu);
				Fish* devptr = a.get_device_pointer(gpu);
				const int gpu_elements = a.get_size_gpu();
				std::array<double,512> gpu_result;
				gpu_result.fill(0.0);
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) async(0)
				for(unsigned int counter = 0; counter < 512; ++counter) {
					double element_result = 0.0;
					#pragma acc loop reduction(+:element_result)
					for(unsigned int inner_counter = 0; inner_counter < gpu_elements; ++inner_counter) {
						double map_result = (f(devptr[inner_counter]))[counter]; // this is actually calculate more often than necessary
						element_result = element_result + map_result;
					}
					gpu_result[counter] = gpu_result[counter] + element_result;
				}
				acc_wait(0);
				
				for(unsigned int counter = 0; counter < 512; ++counter){
					local_result[counter] = local_result[counter] + gpu_result[counter];
				}
			}
		}else if(a.get_device_distribution() == mkt::Distribution::COPY){
			acc_set_device_num(0, acc_device_not_host);
			f.init(0);
			Fish* devptr = a.get_device_pointer(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) async(0)
			for(unsigned int counter = 0; counter < 512; ++counter) {
				double element_result = 0.0;
				#pragma acc loop reduction(+:element_result)
				for(unsigned int inner_counter = 0; inner_counter < gpu_elements; ++inner_counter) {
					double map_result = (f(devptr[inner_counter]))[counter];
					element_result = element_result + map_result;
				}
				local_result[counter] = local_result[counter] + element_result;
			}
			acc_wait(0);
		}
		
		if(a.get_distribution() == mkt::Distribution::DIST){
			MPI_Allreduce(local_result.data(), global_result.data(), 512, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			return global_result;
		}else if(a.get_distribution() == mkt::Distribution::COPY){
			return local_result;
		}				
		 // TODO
	}
	template<>
	double mkt::map_reduce_max<Fish, double, Lambda62_map_reduce_array_functor>(mkt::DArray<Fish>& a, Lambda62_map_reduce_array_functor f){
			double local_result = std::numeric_limits<double>::lowest();
			double global_result = std::numeric_limits<double>::lowest();
						
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				std::array<double*,4> d_odata;
				std::array<double, 4> gpu_results;
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
		
				for(int gpu = 0; gpu < 4; ++gpu){
					cudaSetDevice(gpu);
					cudaMalloc((void**) &d_odata[gpu], blocks * sizeof(double));
					Fish* devptr = a.get_device_pointer(gpu);
					
					mkt::kernel::map_reduce_max_call<Fish, double, Lambda62_map_reduce_array_functor>(gpu_elements, devptr, d_odata[gpu], threads, blocks, f, mkt::cuda_streams[gpu], gpu);
				}
				mkt::sync_streams();
				
				// fold on gpus: step 2
				while(blocks > 1){
			      int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
			      int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  for(int gpu = 0; gpu < 4; ++gpu){
				      cudaSetDevice(gpu);
				      mkt::kernel::reduce_max_call<double>(blocks, d_odata[gpu], d_odata[gpu], threads_2, blocks_2, mkt::cuda_streams[gpu], gpu);
				  }
				  blocks = blocks_2;
			  	  mkt::sync_streams();
			  	}
				
				// copy final sum from device to host
				  for (int gpu = 0; gpu < 4; ++gpu) {
				    cudaSetDevice(gpu);
				    cudaMemcpyAsync(&gpu_results[gpu], d_odata[gpu], sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
				  }
				  mkt::sync_streams();
				  
				  for(int gpu = 0; gpu < 4; ++gpu) {
					cudaSetDevice(gpu);
					cudaFree(d_odata[gpu]);
				  }
				
				for(int gpu = 0; gpu < 4; ++gpu){
					local_result = local_result > gpu_results[gpu] ? local_result : gpu_results[gpu];
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){ // use only gpu 0, since all have the same data
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
				cudaSetDevice(0);
				double* d_odata;
				cudaMalloc((void**) &d_odata, blocks * sizeof(double));
				Fish* devptr = a.get_device_pointer(0);
				
				mkt::kernel::map_reduce_max_call<Fish, double, Lambda62_map_reduce_array_functor>(gpu_elements, devptr, d_odata, threads, blocks, f, mkt::cuda_streams[0], 0);
				
				// fold on gpus: step 2
				while(blocks > 1){
				  int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
				  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  mkt::kernel::reduce_max_call<double>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
				  blocks = blocks_2;
				}
				
				// copy final sum from device to host
				  cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
				  mkt::sync_streams();
				cudaFree(d_odata);
			}
			
			if(a.get_distribution() == mkt::Distribution::DIST){
				MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				return global_result;
			}else if(a.get_distribution() == mkt::Distribution::COPY){
				return local_result;
			}
	}
	
	
	
	int main(int argc, char** argv) {
		MPI_Init(&argc, &argv);
		
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		
		if(mpi_world_size != number_of_processes || mpi_rank != process_id){
			MPI_Finalize();
			return EXIT_FAILURE;
		}				
		mkt::init();
		
		
		mkt::sync_streams();
		
		mkt::DArray<Fish> population(12, 2048, 128, Fish{}, 4, 12, 1536, mkt::DIST, mkt::COPY);
		mkt::DArray<double> instinctive_movement_vector_copy(12, 512, 512, 0.0, 1, 12, 0, mkt::COPY, mkt::COPY);
		mkt::DArray<Fish> weighted_fishes(12, 2048, 128, Fish{}, 4, 12, 1536, mkt::DIST, mkt::COPY);
		mkt::DArray<double> barycenter_copy(12, 512, 512, 0.0, 1, 12, 0, mkt::COPY, mkt::COPY);
		
		InitFish_map_in_place_array_functor initFish_map_in_place_array_functor{};
		EvaluateFitness_map_in_place_array_functor evaluateFitness_map_in_place_array_functor{};
		IndividualMovement_map_in_place_array_functor individualMovement_map_in_place_array_functor{};
		Feeding_map_in_place_array_functor feeding_map_in_place_array_functor{};
		CalcDisplacementMap_map_in_place_array_functor calcDisplacementMap_map_in_place_array_functor{};
		CalcInstinctiveMovementVector_map_in_place_array_functor calcInstinctiveMovementVector_map_in_place_array_functor{};
		InstinctiveMovement_map_in_place_array_functor instinctiveMovement_map_in_place_array_functor{instinctive_movement_vector_copy};
		CalcWeightedFish_map_array_functor calcWeightedFish_map_array_functor{};
		CalcBarycenterMap_map_in_place_array_functor calcBarycenterMap_map_in_place_array_functor{};
		VolitiveMovement_map_in_place_array_functor volitiveMovement_map_in_place_array_functor{barycenter_copy};
		Lambda56_map_reduce_array_functor lambda56_map_reduce_array_functor{};
		Lambda57_map_reduce_array_functor lambda57_map_reduce_array_functor{};
		Lambda58_map_reduce_array_functor lambda58_map_reduce_array_functor{};
		Lambda59_map_reduce_array_functor lambda59_map_reduce_array_functor{};
		Lambda60_map_reduce_array_functor lambda60_map_reduce_array_functor{};
		Lambda61_map_reduce_array_functor lambda61_map_reduce_array_functor{};
		Lambda62_map_reduce_array_functor lambda62_map_reduce_array_functor{};
		
		
				
			MPI_Datatype Fish_mpi_type_temp;
			MPI_Type_create_struct(9, (std::array<int,9>{1, 1, 1, 1, 1, 1, 1, 1, 1}).data(), (std::array<MPI_Aint,9>{static_cast<MPI_Aint>(offsetof(struct Fish, position)), static_cast<MPI_Aint>(offsetof(struct Fish, fitness)), static_cast<MPI_Aint>(offsetof(struct Fish, candidate_position)), static_cast<MPI_Aint>(offsetof(struct Fish, candidate_fitness)), static_cast<MPI_Aint>(offsetof(struct Fish, displacement)), static_cast<MPI_Aint>(offsetof(struct Fish, fitness_variation)), static_cast<MPI_Aint>(offsetof(struct Fish, weight)), static_cast<MPI_Aint>(offsetof(struct Fish, best_position)), static_cast<MPI_Aint>(offsetof(struct Fish, best_fitness))}).data(), (std::array<MPI_Datatype,9>{MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}).data(), &Fish_mpi_type_temp);
			MPI_Type_create_resized(Fish_mpi_type_temp, 0, sizeof(Fish), &Fish_mpi_type);
			MPI_Type_free(&Fish_mpi_type_temp);
			MPI_Type_commit(&Fish_mpi_type);
			
			
		
			
		
		
		mkt::sync_streams();
		mkt::map_in_place<Fish, InitFish_map_in_place_array_functor>(population, initFish_map_in_place_array_functor);
		double step_size = (STEP_SIZE_INITIAL);
		double step_size_vol = (STEP_SIZE_VOLITIVE_INITIAL);
		double sum_weight_last_iteration = 0.0;
		sum_weight_last_iteration = mkt::map_reduce_plus<Fish, double, Lambda56_map_reduce_array_functor>(population, lambda56_map_reduce_array_functor);
		for(int iteration = 0; ((iteration) < (ITERATIONS)); ++iteration){
			mkt::map_in_place<Fish, EvaluateFitness_map_in_place_array_functor>(population, evaluateFitness_map_in_place_array_functor);
			if(((iteration) > 0)){
				step_size = ((step_size) - (((STEP_SIZE_INITIAL) - (STEP_SIZE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
				step_size_vol = ((step_size_vol) - (((STEP_SIZE_VOLITIVE_INITIAL) - (STEP_SIZE_VOLITIVE_FINAL)) / static_cast<double>(((ITERATIONS) - 1))));
			}
			individualMovement_map_in_place_array_functor.step_size = (step_size);
			mkt::map_in_place<Fish, IndividualMovement_map_in_place_array_functor>(population, individualMovement_map_in_place_array_functor);
			double max_fitness_variation = 0.0;
			max_fitness_variation = mkt::map_reduce_max<Fish, double, Lambda57_map_reduce_array_functor>(population, lambda57_map_reduce_array_functor);
			feeding_map_in_place_array_functor.max_fitness_variation = (max_fitness_variation);
			mkt::map_in_place<Fish, Feeding_map_in_place_array_functor>(population, feeding_map_in_place_array_functor);
			double sum_fitness_variation = 0.0;
			sum_fitness_variation = mkt::map_reduce_plus<Fish, double, Lambda58_map_reduce_array_functor>(population, lambda58_map_reduce_array_functor);
			mkt::map_in_place<Fish, CalcDisplacementMap_map_in_place_array_functor>(population, calcDisplacementMap_map_in_place_array_functor);
			instinctive_movement_vector_copy = mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda59_map_reduce_array_functor>(population, lambda59_map_reduce_array_functor);
			calcInstinctiveMovementVector_map_in_place_array_functor.sum_fitness_variation = (sum_fitness_variation);
			mkt::map_in_place<double, CalcInstinctiveMovementVector_map_in_place_array_functor>(instinctive_movement_vector_copy, calcInstinctiveMovementVector_map_in_place_array_functor);
			mkt::map_in_place<Fish, InstinctiveMovement_map_in_place_array_functor>(population, instinctiveMovement_map_in_place_array_functor);
			double sum_weight = 0.0;
			sum_weight = mkt::map_reduce_plus<Fish, double, Lambda60_map_reduce_array_functor>(population, lambda60_map_reduce_array_functor);
			mkt::map<Fish, Fish, CalcWeightedFish_map_array_functor>(population, weighted_fishes, calcWeightedFish_map_array_functor);
			barycenter_copy = mkt::map_reduce_plus<Fish, std::array<double,512>, Lambda61_map_reduce_array_functor>(weighted_fishes, lambda61_map_reduce_array_functor);
			calcBarycenterMap_map_in_place_array_functor.sum_weight = (sum_weight);
			mkt::map_in_place<double, CalcBarycenterMap_map_in_place_array_functor>(barycenter_copy, calcBarycenterMap_map_in_place_array_functor);
			volitiveMovement_map_in_place_array_functor.step_size = (step_size_vol);volitiveMovement_map_in_place_array_functor.sum_weight = (sum_weight);volitiveMovement_map_in_place_array_functor.sum_weight_last_iteration = (sum_weight_last_iteration);
			mkt::map_in_place<Fish, VolitiveMovement_map_in_place_array_functor>(population, volitiveMovement_map_in_place_array_functor);
			sum_weight_last_iteration = (sum_weight);
		}
		double global_best_fitness = 0.0;
		global_best_fitness = mkt::map_reduce_max<Fish, double, Lambda62_map_reduce_array_functor>(population, lambda62_map_reduce_array_functor);
		mkt::sync_streams();
		
		mkt::sync_streams();
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
