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
	#include "../include/matmult-n-1-g-4_0.cuh"
	
	
			
	

	
	struct DotProduct_map_local_index_in_place_matrix_functor{
		
		DotProduct_map_local_index_in_place_matrix_functor(const mkt::DMatrix<float>& _as, const mkt::DMatrix<float>& _bs) : as(_as), bs(_bs){}
		
		~DotProduct_map_local_index_in_place_matrix_functor() {}
		
		__device__
		auto operator()(int i, int j, float Cij){
			float sum = 0.0f;
			for(int k = 0; ((k) < 16384); k++){
				sum += (as.get_data_local((i), (k)) * bs.get_data_local((k), (j)));
			}
			Cij += (sum);
			return (Cij);
		}
	
		void init(int device){
			as.init(device);
			bs.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
		mkt::DeviceMatrix<float> as;
		mkt::DeviceMatrix<float> bs;
	};
	struct Square_map_in_place_matrix_functor{
		
		Square_map_in_place_matrix_functor(){}
		
		~Square_map_in_place_matrix_functor() {}
		
		__device__
		auto operator()(float a){
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
	float mkt::reduce_plus<float>(mkt::DMatrix<float>& a){
		float local_result = 0.0f;
					
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			std::array<float*,4> d_odata;
			std::array<float, 4> gpu_results;
			const int gpu_elements = a.get_size_gpu();
			int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
			int blocks = (gpu_elements + threads - 1) / threads;
	
			for(int gpu = 0; gpu < 4; ++gpu){
				cudaSetDevice(gpu);
				cudaMalloc((void**) &d_odata[gpu], blocks * sizeof(float));
				float* devptr = a.get_device_pointer(gpu);
				
				mkt::kernel::reduce_plus_call<float>(gpu_elements, devptr, d_odata[gpu], threads, blocks, mkt::cuda_streams[gpu], gpu);
			}
			mkt::sync_streams();
			
			// fold on gpus: step 2
			while(blocks > 1){
		      int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
		      int blocks_2 = (blocks + threads_2 - 1) / threads_2;
			  for(int gpu = 0; gpu < 4; ++gpu){
			      cudaSetDevice(gpu);
			      mkt::kernel::reduce_plus_call<float>(blocks, d_odata[gpu], d_odata[gpu], threads_2, blocks_2, mkt::cuda_streams[gpu], gpu);
			  }
			  blocks = blocks_2;
		  	  mkt::sync_streams();
		  	}
			
			// copy final sum from device to host
			  for (int gpu = 0; gpu < 4; ++gpu) {
			    cudaSetDevice(gpu);
			    cudaMemcpyAsync(&gpu_results[gpu], d_odata[gpu], sizeof(float), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
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
			float* d_odata;
			cudaMalloc((void**) &d_odata, blocks * sizeof(float));
			float* devptr = a.get_device_pointer(0);
			
			mkt::kernel::reduce_plus_call<float>(gpu_elements, devptr, d_odata, threads, blocks, mkt::cuda_streams[0], 0);
			
			// fold on gpus: step 2
			while(blocks > 1){
			  int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
			  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
			  mkt::kernel::reduce_plus_call<float>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
			  blocks = blocks_2;
			}
			
			// copy final sum from device to host
			  cudaMemcpyAsync(&local_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
			  mkt::sync_streams();
			cudaFree(d_odata);
		}
		
		return local_result;
	}
	
	
	
	int main(int argc, char** argv) {
		mkt::init();
		
		
		const int dim = 16384;
		mkt::DMatrix<float> as(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 1.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
		mkt::DMatrix<float> bs(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 0.001f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::COPY);
		mkt::DMatrix<float> cs(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 0.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
		
		DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{as, bs};
		Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
				
		
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 1); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
		}
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		mkt::map_in_place<float, Square_map_in_place_matrix_functor>(cs, square_map_in_place_matrix_functor);
		double fn = 0.0;
		fn = mkt::reduce_plus<float>(cs);
		fn = std::sqrt((fn));
		printf("Frobenius norm of cs is %.5f.\n",(fn));
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
