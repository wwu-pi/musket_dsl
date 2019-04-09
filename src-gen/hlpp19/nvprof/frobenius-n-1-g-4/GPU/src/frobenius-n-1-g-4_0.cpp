	
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
	#include "../include/frobenius-n-1-g-4_0.hpp"
	
	
	
			
	const int dim = 1024;
	mkt::DMatrix<double> as(0, 1024, 1024, 1024, 1024, 1048576, 1048576, 0.0, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		
		Init_map_index_in_place_matrix_functor(){
		}
		
		~Init_map_index_in_place_matrix_functor() {}
		
		auto operator()(int x, int y, double& a){
			a = static_cast<double>((((x) + (y)) + 1));
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
	struct Square_map_reduce_matrix_functor{
		
		Square_map_reduce_matrix_functor(){
		}
		
		~Square_map_reduce_matrix_functor() {}
		
		auto operator()(double a){
			a = ((a) * (a));
			return (a);
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
	double mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(mkt::DMatrix<double>& a, Square_map_reduce_matrix_functor f){
		double local_result = 0.0;
		
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			for(int gpu = 0; gpu < 4; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				f.init(gpu);
				double* devptr = a.get_device_pointer(gpu);
				const int gpu_elements = a.get_size_gpu();
				double gpu_result = 0.0;
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(+:gpu_result) async(0) gang vector
				for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
					#pragma acc cache(gpu_result, devptr[0:gpu_elements])
					double map_result = f(devptr[counter]);
					gpu_result = gpu_result + map_result;
				}
				acc_wait(0);
				local_result = local_result + gpu_result;
			}
		}else if(a.get_device_distribution() == mkt::Distribution::COPY){
			acc_set_device_num(0, acc_device_not_host);
			f.init(0);
			double* devptr = a.get_device_pointer(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result) async(0)
			for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
				#pragma acc cache(local_result, devptr[0:gpu_elements])
				double map_result = f(devptr[counter]);
				local_result = local_result + map_result;
			}
			acc_wait(0);
		}
		
		return local_result;
	}
	
	
	
	int main(int argc, char** argv) {
		
		
		
		Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
		Square_map_reduce_matrix_functor square_map_reduce_matrix_functor{};
		
		
				
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		double fn = 0.0;
		fn = mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(as, square_map_reduce_matrix_functor);
		fn = std::sqrt((fn));
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Frobenius norm is %.5f.\n",(fn));
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}