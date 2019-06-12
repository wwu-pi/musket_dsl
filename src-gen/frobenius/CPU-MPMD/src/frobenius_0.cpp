	
	#include <omp.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	
	#include "../include/musket.hpp"
	#include "../include/frobenius_0.hpp"
	
	
	

	
	const int dim = 8192;
	mkt::DMatrix<double> as(0, 8192, 8192, 8192, 8192, 67108864, 67108864, 0.0, 1, 1, 0, 0, 0, 0, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		auto operator()(int x, int y, double a) const{
			a = static_cast<double>((((x) + (y)) + 1));
			return (a);
		}
		
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
		
		
		
				
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		double fn = 0.0;
		// TODO: SkeletonGenerator.generateSkeletonExpression: default case
		fn = std::sqrt((fn));
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Frobenius norm is %.5f.\n",(fn));
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
