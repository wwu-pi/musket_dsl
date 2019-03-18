	
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
	#include "../include/frobenius-n-1-g-1_0.hpp"
	
	
	

	
	const int dim = 32768;
	mkt::DMatrix<double> as(0, 32768, 32768, 32768, 32768, 1073741824, 1073741824, 0.0, 1, 1, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		
		Init_map_index_in_place_matrix_functor() {}
		
		auto operator()(int x, int y, double& a) const{
			a = static_cast<double>((((x) + (y)) + 1));
		}
	
		void init(int gpu){
		}
		
		
	};
	struct Square_map_in_place_matrix_functor{
		
		Square_map_in_place_matrix_functor() {}
		
		auto operator()(double& a) const{
			a = ((a) * (a));
		}
	
		void init(int gpu){
		}
		
		
	};
	
	
	
	template<>
	double mkt::reduce_plus<double>(mkt::DMatrix<double>& a){
		double local_result = 0.0;
		
		double* devptr = a.get_device_pointer(0);
		const int gpu_elements = a.get_size_gpu();
		
		#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result)
		for(int counter = 0; counter < gpu_elements; ++counter) {
			#pragma acc cache(local_result)
			local_result = local_result + devptr[counter];
		}
		
		return local_result;
	}
	
	
	
	int main(int argc, char** argv) {
		
		
				Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
				Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
		
				
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		mkt::map_in_place<double, Square_map_in_place_matrix_functor>(as, square_map_in_place_matrix_functor);
		double fn = 0.0;
		fn = mkt::reduce_plus<double>(as);
		fn = std::sqrt((fn));
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Frobenius norm is %.5f.\n",(fn));
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}