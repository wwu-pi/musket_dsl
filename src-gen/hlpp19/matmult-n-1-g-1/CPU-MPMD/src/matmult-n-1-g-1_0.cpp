	
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
	#include "../include/matmult-n-1-g-1_0.hpp"
	
	
	

	
	const int dim = 16384;
	mkt::DMatrix<float> as(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 1.0f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<float> bs(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 0.001f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<float> cs(0, 16384, 16384, 16384, 16384, 268435456, 268435456, 0.0f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	
	

	
	struct DotProduct_map_local_index_in_place_matrix_functor{
		auto operator()(int i, int j, float Cij) const{
			float sum = 0.0f;
			for(int k = 0; ((k) < 4096); k++){
				sum += (as[(i) * 16384 + (k)] * bs[(k) * 16384 + (j)]);
			}
			Cij = (sum);
			return (Cij);
		}
		
	};
	struct Square_map_in_place_matrix_functor{
		auto operator()(float a) const{
			a = ((a) * (a));
			return (a);
		}
		
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{};
				Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
		
				
		
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 4); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		mkt::map_in_place<float, Square_map_in_place_matrix_functor>(cs, square_map_in_place_matrix_functor);
		double fn = 0.0;
		// TODO: SkeletonGenerator.generateSkeletonExpression: default case
		fn = std::sqrt((fn));
		printf("Frobenius norm of cs is %.5f.\n",(fn));
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
