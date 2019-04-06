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
	#include "../include/matmult_float_0.cuh"
	
	
			
	const int dim = 4;
	mkt::DMatrix<float> as(0, 4, 4, 4, 4, 16, 16, 1.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	mkt::DMatrix<float> bs(0, 4, 4, 4, 4, 16, 16, 0.001f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DMatrix<float> cs(0, 4, 4, 4, 4, 16, 16, 0.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	
	

	
	struct InitA_map_index_in_place_matrix_functor{
		
		InitA_map_index_in_place_matrix_functor(){}
		
		~InitA_map_index_in_place_matrix_functor() {}
		
		__device__
		auto operator()(int a, int b, float& x){
			x = ((static_cast<float>((a)) * 4) + (b));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct InitB_map_index_in_place_matrix_functor{
		
		InitB_map_index_in_place_matrix_functor(){}
		
		~InitB_map_index_in_place_matrix_functor() {}
		
		__device__
		auto operator()(int a, int b, float& x){
			x = ((static_cast<float>(16) + ((a) * 4)) + (b));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct DotProduct_map_local_index_in_place_matrix_functor{
		
		DotProduct_map_local_index_in_place_matrix_functor(const mkt::DMatrix<float>& _as, const mkt::DMatrix<float>& _bs) : as(_as), bs(_bs){}
		
		~DotProduct_map_local_index_in_place_matrix_functor() {}
		
		__device__
		auto operator()(int i, int j, float& Cij){
			for(int k = 0; ((k) < 4); k++){
				Cij += (as.get_data_local((i), (k)) * bs.get_data_local((k), (j)));
			}
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
		auto operator()(float& a){
			a = ((a) * (a));
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
		
		return local_result;
	}
	
	
	
	int main(int argc, char** argv) {
		mkt::init();
		
		
		InitA_map_index_in_place_matrix_functor initA_map_index_in_place_matrix_functor{};
		InitB_map_index_in_place_matrix_functor initB_map_index_in_place_matrix_functor{};
		DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{as, bs};
		Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
				
		
		mkt::map_index_in_place<float, InitA_map_index_in_place_matrix_functor>(as, initA_map_index_in_place_matrix_functor);
		mkt::map_index_in_place<float, InitB_map_index_in_place_matrix_functor>(bs, initB_map_index_in_place_matrix_functor);
		as.update_self();
		mkt::print("as", as);
		bs.update_self();
		mkt::print("bs", bs);
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 1); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
		}
		mkt::sync_streams();
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		cs.update_self();
		mkt::print("cs", cs);
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