	
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
	#include "../include/matmult-n-1-g-1_0.hpp"
	
	
	
			
	const int dim = 8192;
	mkt::DMatrix<float> as(0, 8192, 8192, 8192, 8192, 67108864, 67108864, 1.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	mkt::DMatrix<float> bs(0, 8192, 8192, 8192, 8192, 67108864, 67108864, 0.001f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DMatrix<float> cs(0, 8192, 8192, 8192, 8192, 67108864, 67108864, 0.0f, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	
	

	
	struct InitA_map_index_in_place_matrix_functor{
		
		InitA_map_index_in_place_matrix_functor(){
		}
		
		~InitA_map_index_in_place_matrix_functor() {}
		
		auto operator()(int a, int b, float& x){
			x = ((static_cast<float>((a)) * 4) + (b));
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
	struct InitB_map_index_in_place_matrix_functor{
		
		InitB_map_index_in_place_matrix_functor(){
		}
		
		~InitB_map_index_in_place_matrix_functor() {}
		
		auto operator()(int a, int b, float& x){
			x = ((static_cast<float>(16) + ((a) * 4)) + (b));
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
	struct DotProduct_map_local_index_in_place_matrix_functor{
		
		DotProduct_map_local_index_in_place_matrix_functor(const mkt::DMatrix<float>& _as, const mkt::DMatrix<float>& _bs) : as(_as), bs(_bs){
		}
		
		~DotProduct_map_local_index_in_place_matrix_functor() {}
		
		auto operator()(int i, int j, float& Cij){
			for(int k = 0; ((k) < 8192); k++){
				Cij += (as.get_data_local((i), (k)) * bs.get_data_local((k), (j)));
			}
		}
	
		void init(int gpu){
			as.init(gpu);
			bs.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		mkt::DeviceMatrix<float> as;
		mkt::DeviceMatrix<float> bs;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	
	
	
	template<>
	float mkt::reduce_plus<float>(mkt::DMatrix<float>& a){
		float local_result = 0.0f;
		
		float* devptr = a.get_device_pointer(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result)
		for(int counter = 0; counter < gpu_elements; ++counter) {
			#pragma acc cache(local_result)
			local_result = local_result + devptr[counter];
		}
		
		return local_result;
	}
	
	
	
	int main(int argc, char** argv) {
		
		
		
		InitA_map_index_in_place_matrix_functor initA_map_index_in_place_matrix_functor{};
		InitB_map_index_in_place_matrix_functor initB_map_index_in_place_matrix_functor{};
		Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{as, bs};
		
		
				
		
		mkt::map_index_in_place<float, InitA_map_index_in_place_matrix_functor>(as, initA_map_index_in_place_matrix_functor);
		mkt::map_index_in_place<float, InitB_map_index_in_place_matrix_functor>(bs, initB_map_index_in_place_matrix_functor);
		mkt::map_in_place<float, Square_map_in_place_matrix_functor>(as, square_map_in_place_matrix_functor);
		double fna = 0.0;
		fna = mkt::reduce_plus<float>(as);
		fna = std::sqrt((fna));
		mkt::map_in_place<float, Square_map_in_place_matrix_functor>(bs, square_map_in_place_matrix_functor);
		double fnb = 0.0;
		fnb = mkt::reduce_plus<float>(bs);
		fnb = std::sqrt((fnb));
		printf("Frobenius norm of as is %.5f and of bs is %.5f.\n",(fna),(fnb));
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 1); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
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
