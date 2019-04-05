	
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
	#include "../include/matrix_0.hpp"
	
	
	

	
	mkt::DMatrix<int> ads(0, 4, 4, 4, 4, 16, 16, 7, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<int> bds(0, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<int> acs(0, 4, 4, 4, 4, 16, 16, 7, 1, 1, 0, 0, 0, 0, mkt::COPY);
	mkt::DMatrix<int> bcs(0, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::COPY);
	mkt::DMatrix<int> r_ads(0, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<int> r_bds(0, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<int> r_acs(0, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::COPY);
	mkt::DMatrix<int> r_bcs(0, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::COPY);
	
	

	
	struct Init_map_index_matrix_functor{
		auto operator()(const int row, const int col, const int x) const{
			return (((row) * 4) + (col));
		}
		
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				Init_map_index_matrix_functor init_map_index_matrix_functor{};
		
		
		
				
		
		mkt::print("ads", ads);
		mkt::print("bds", bds);
		mkt::print("acs", acs);
		mkt::print("bcs", bcs);
		printf("\n mapIndexInPlace \n");
		mkt::map_index<int, int, Init_map_index_matrix_functor>(ads, r_ads, init_map_index_matrix_functor);
		mkt::map_index<int, int, Init_map_index_matrix_functor>(bds, r_bds, init_map_index_matrix_functor);
		mkt::map_index<int, int, Init_map_index_matrix_functor>(acs, r_acs, init_map_index_matrix_functor);
		mkt::map_index<int, int, Init_map_index_matrix_functor>(bcs, r_bcs, init_map_index_matrix_functor);
		mkt::print("r_ads", r_ads);
		mkt::print("r_bds", r_bds);
		mkt::print("r_acs", r_acs);
		mkt::print("r_bcs", r_bcs);
		
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
