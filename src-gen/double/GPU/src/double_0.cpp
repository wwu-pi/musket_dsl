	
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
	#include "../include/double_0.hpp"
	
	
	

	
	mkt::DArray<int> b(0, 4, 4, 0, 1, 0, 0, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_array_functor{
		auto operator()(int i, int& x) const{
			x = (i);
		}
		
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				Init_map_index_in_place_array_functor init_map_index_in_place_array_functor{};
		
		
		
				
		
		b.update_self();
		mkt::print("b", b);
		mkt::map_index_in_place<int, Init_map_index_in_place_array_functor>(b, init_map_index_in_place_array_functor);
		b.update_self();
		mkt::print("b", b);
		
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
