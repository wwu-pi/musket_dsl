	
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
	#include "../include/array_0.hpp"
	
	
	

	
	const int dim = 16;
	mkt::DArray<int> ads(0, 16, 16, 1, 1, 0, 0, mkt::DIST);
	mkt::DArray<int> bds(0, 16, 16, 0, 1, 0, 0, mkt::DIST);
	mkt::DArray<int> acs(0, 16, 16, 7, 1, 0, 0, mkt::COPY);
	mkt::DArray<int> bcs(0, 16, 16, 0, 1, 0, 0, mkt::COPY);
	mkt::DArray<int> temp(0, 16, 16, 0, 1, 0, 0, mkt::DIST);
	mkt::DArray<int> temp_copy(0, 16, 16, 0, 1, 0, 0, mkt::COPY);
	
	

	
	struct PlusX_map_array_functor{
		auto operator()(int v) const{
			return ((x) + (v));
		}
		
		int x;
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				PlusX_map_array_functor plusX_map_array_functor{};
		
		
		
				
		
		mkt::print("ads", ads);
		mkt::print("bds", bds);
		mkt::print("acs", acs);
		mkt::print("bcs", bcs);
		plusX_map_array_functor.x = 17;
		mkt::map<int, int, PlusX_map_array_functor>(ads, temp, plusX_map_array_functor);
		mkt::print("temp", temp);
		plusX_map_array_functor.x = 42;
		mkt::map<int, int, PlusX_map_array_functor>(bcs, temp_copy, plusX_map_array_functor);
		mkt::print("temp_copy", temp_copy);
		
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
