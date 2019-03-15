	
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
	#include "../include/matmult-n-1-g-2_0.hpp"
	
	
	

	
	const int dim = 8192;
	mkt::DMatrix<float> as(0, 8192, 8192, 8192, 8192, 67108864, 67108864, 1.0f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<float> bs(0, 8192, 8192, 8192, 8192, 67108864, 67108864, 0.001f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<float> cs(0, 8192, 8192, 8192, 8192, 67108864, 67108864, 0.0f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	
	

	
	struct InitA_map_index_in_place_matrix_functor{
		auto operator()(int a, int b, float& x) const{
			x = ((static_cast<float>((a)) * 4) + (b));
		}
		
	};
	struct InitB_map_index_in_place_matrix_functor{
		auto operator()(int a, int b, float& x) const{
			x = ((static_cast<float>(16) + ((a) * 4)) + (b));
		}
		
	};
	struct DotProduct_map_local_index_in_place_matrix_functor{
		auto operator()(int i, int j, float& Cij) const{
			for(int k = 0; ((k) < 8192); k++){
				Cij += (as[(i) * 8192 + (k)] * bs[(k) * 8192 + (j)]);
			}
		}
		
	};
	struct Square_map_in_place_matrix_functor{
		auto operator()(float& a) const{
			a = ((a) * (a));
		}
		
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				InitA_map_index_in_place_matrix_functor initA_map_index_in_place_matrix_functor{};
				InitB_map_index_in_place_matrix_functor initB_map_index_in_place_matrix_functor{};
				DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{};
				Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
		
				
		
		mkt::map_index_in_place<float, InitA_map_index_in_place_matrix_functor>(as, initA_map_index_in_place_matrix_functor);
		mkt::map_index_in_place<float, InitB_map_index_in_place_matrix_functor>(bs, initB_map_index_in_place_matrix_functor);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 1); ++i){
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
