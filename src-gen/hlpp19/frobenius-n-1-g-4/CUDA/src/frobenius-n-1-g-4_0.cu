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
	#include "../include/frobenius-n-1-g-4_0.cuh"
	
	
	
	const int dim = 16384;
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		
		Init_map_index_in_place_matrix_functor(){}
		
		~Init_map_index_in_place_matrix_functor() {}
		
		__device__
		auto operator()(int x, int y, double a){
			a = static_cast<double>((((x) + (y)) + 1));
			return (a);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Square_map_reduce_matrix_functor{
		
		Square_map_reduce_matrix_functor(){}
		
		~Square_map_reduce_matrix_functor() {}
		
		__device__
		auto operator()(double a){
			a = ((a) * (a));
			return (a);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	
	
	
	
	template<>
	double mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(mkt::DMatrix<double>& a, Square_map_reduce_matrix_functor f){
			double local_result = 0.0;
						
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				std::array<double*,4> d_odata;
				std::array<double, 4> gpu_results;
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
		
				for(int gpu = 0; gpu < 4; ++gpu){
					cudaSetDevice(gpu);
					cudaMalloc((void**) &d_odata[gpu], blocks * sizeof(double));
					double* devptr = a.get_device_pointer(gpu);
					
					mkt::kernel::map_reduce_plus_call<double, double, Square_map_reduce_matrix_functor>(gpu_elements, devptr, d_odata[gpu], threads, blocks, f, mkt::cuda_streams[gpu], gpu);
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
				double* devptr = a.get_device_pointer(0);
				
				mkt::kernel::map_reduce_plus_call<double, double, Square_map_reduce_matrix_functor>(gpu_elements, devptr, d_odata, threads, blocks, f, mkt::cuda_streams[0], 0);
				
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
			
			return local_result;
	}
	
	
	
	int main(int argc, char** argv) {
		mkt::init();
		
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point complete_timer_start = std::chrono::high_resolution_clock::now();
		
		mkt::DMatrix<double> as(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 0.0, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
		
		Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
		Square_map_reduce_matrix_functor square_map_reduce_matrix_functor{};
		
		
				
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		double fn = 0.0;
		fn = mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(as, square_map_reduce_matrix_functor);
		fn = std::sqrt((fn));
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Frobenius norm is %.5f.\n",(fn));
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point complete_timer_end = std::chrono::high_resolution_clock::now();
		double complete_seconds = std::chrono::duration<double>(complete_timer_end - complete_timer_start).count();
		printf("Complete execution time: %.5fs\n", complete_seconds);
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
