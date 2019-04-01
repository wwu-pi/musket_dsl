	
	#include <omp.h>
	#include <openacc.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	//#include <cuda.h>
	//#include <openacc_curand.h>
	
	#include "../include/musket.hpp"
	#include "../include/matmult-n-1-g-4_0.hpp"
	
	
	
			
	const int dim = 16384;
	mkt::DMatrix<float> as(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 1.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	mkt::DMatrix<float> bs(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 0.001f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DMatrix<float> cs(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 0.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	
	

	
	struct DotProduct_map_local_index_in_place_matrix_functor{
		
		DotProduct_map_local_index_in_place_matrix_functor(const mkt::DMatrix<float>& _as, const mkt::DMatrix<float>& _bs) : m_as(_as), m_bs(_bs){
		}
		
		~DotProduct_map_local_index_in_place_matrix_functor() {}
		
		auto operator()(int i, int j, float& Cij, float* p_as, float* p_bs){
			for(int k = 0; ((k) < 16384); k++){
				Cij += p_as[m_as.get_index(i, k)] * p_bs[m_bs.get_index(i,k)]; //(as.get_data_local((i), (k)) * bs.get_data_local((k), (j)));
			}
		}
	
		void init(int gpu){
			m_as.init(gpu);
			m_bs.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		mkt::DeviceMatrix<float> m_as;
		mkt::DeviceMatrix<float> m_bs;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct Square_map_in_place_matrix_functor{
		
		Square_map_in_place_matrix_functor(){
		}
		
		~Square_map_in_place_matrix_functor() {}
		
		auto operator()(float& a){
			a = ((a) * (a));
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	
	
	
	
	template<>
	float mkt::reduce_plus<float>(mkt::DMatrix<float>& a){
		float local_result = 0.0f;
		
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			#pragma omp parallel for reduction(+:local_result)
			for(int gpu = 0; gpu < 4; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				float* devptr = a.get_device_pointer(gpu);
				const int gpu_elements = a.get_size_gpu();
				float gpu_result = 0.0f;
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(+:gpu_result) async(0) vector_length(1024)
				for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
					#pragma acc cache(gpu_result)
					gpu_result = gpu_result + devptr[counter];
				}
				acc_wait(0);
				local_result = local_result + gpu_result;
			}
		}else if(a.get_device_distribution() == mkt::Distribution::COPY){
			acc_set_device_num(0, acc_device_not_host);
			float* devptr = a.get_device_pointer(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result) async(0) vector_length(1024)
			for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
				#pragma acc cache(local_result)
				local_result = local_result + devptr[counter];
			}
			acc_wait(0);
		}
		
		return local_result;
	}
	
	void wait_all_gpus(){
		#pragma omp parallel for
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
	}
	
template<typename T, typename Functor>
void mkt::map_local_index_in_place(mkt::DMatrix<T>& m, Functor f){
	unsigned int columns_local = m.get_number_of_columns_local();

  	unsigned int gpu_elements = m.get_size_gpu();
  	unsigned int rows_on_gpu = m.get_rows_gpu();
  	
	//#pragma omp parallel for shared(f)
	for(int gpu = 0; gpu < 4; ++gpu){
		acc_set_device_num(gpu, acc_device_not_host);
		f.init(gpu);
		
		T* devptr = m.get_device_pointer(gpu);
		T* devptr_as = as.get_device_pointer(gpu);
		T* devptr_bs = bs.get_device_pointer(gpu);

		unsigned int gpu_row_offset = 0;
		if(m.get_device_distribution() == mkt::Distribution::DIST){
			gpu_row_offset = gpu * rows_on_gpu;
		}
		
		#pragma acc parallel loop deviceptr(devptr,devptr_as,devptr_bs) firstprivate(f) async(0) vector_length(1024)
		for(unsigned int i = 0; i < gpu_elements; ++i) {
			#pragma acc cache(devptr[0:2147483642], devptr_as[0:1], devptr_bs[0:1])
			f.set_id(__pgi_gangidx(), __pgi_workeridx(),__pgi_vectoridx());
			unsigned int row_index = gpu_row_offset + (i / columns_local);
			unsigned int column_index = i % columns_local;
			f(row_index, column_index, devptr[i], devptr_as, devptr_bs);
		}
	}
}

	int main(int argc, char** argv) {
		
		
		
		DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{as, bs};
		Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
				
		
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 1); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
		}
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
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
