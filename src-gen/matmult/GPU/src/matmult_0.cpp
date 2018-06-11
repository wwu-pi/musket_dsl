	
	#include <omp.h>
	#include <openacc.h>
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
	#include "../include/matmult_0.hpp"
	
	
	

	
	mkt::DMatrix<double> as(0, 4, 4, 4, 4, 16, 16, 1.0, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<double> bs(0, 4, 4, 4, 4, 16, 16, 0.001, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<double> cs(0, 4, 4, 4, 4, 16, 16, 0.0, 1, 1, 0, 0, 0, 0, mkt::DIST);
	
	

	
	struct InitA_map_index_in_place_matrix_functor{
		auto operator()(int a, int b, double& x) const{
			x = ((static_cast<double>((a)) * 4) + (b));
		}
		
	};
	struct InitB_map_index_in_place_matrix_functor{
		auto operator()(int a, int b, double& x) const{
			x = ((static_cast<double>(16) + ((a) * 4)) + (b));
		}
		
	};
	struct DotProduct_map_local_index_in_place_matrix_functor{
		auto operator()(int i, int j, double& Cij) const{
			for(int k = 0; ((k) < 4); k++){
				Cij = 42;
			}
		}
		
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				InitA_map_index_in_place_matrix_functor initA_map_index_in_place_matrix_functor{};
				InitB_map_index_in_place_matrix_functor initB_map_index_in_place_matrix_functor{};
				DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{};
		
		
		
				
		
		mkt::map_index_in_place<double, InitA_map_index_in_place_matrix_functor>(as, initA_map_index_in_place_matrix_functor);
		mkt::map_index_in_place<double, InitB_map_index_in_place_matrix_functor>(bs, initB_map_index_in_place_matrix_functor);
		as.update_self();
		mkt::print("as", as);
		bs.update_self();
		mkt::print("bs", bs);
		cs.update_self();
		mkt::print("cs", cs);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 1); ++i){
			mkt::map_local_index_in_place<double, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
		}
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		cs.update_self();
		mkt::print("cs", cs);
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
