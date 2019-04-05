	
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
	#include "../include/array_0.hpp"
	
	
	
			
	const int dim = 16;
	mkt::DArray<int> ads(0, 16, 16, 1, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<int> bds(0, 16, 16, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<int> acs(0, 16, 16, 7, 1, 0, 0, mkt::COPY, mkt::COPY);
	mkt::DArray<int> bcs(0, 16, 16, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
	mkt::DArray<int> temp(0, 16, 16, 0, 1, 0, 0, mkt::DIST, mkt::COPY);
	mkt::DArray<int> temp_copy(0, 16, 16, 0, 1, 0, 0, mkt::COPY, mkt::COPY);
	
	

	
	struct PlusX_map_array_functor{
		
		PlusX_map_array_functor(){
		}
		
		~PlusX_map_array_functor() {}
		
		auto operator()(const int v){
			return ((x) + (v));
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		int x;
		
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
		
		PlusX_map_array_functor plusX_map_array_functor{};
		
		
				
		
		ads.update_self();
		mkt::print("ads", ads);
		bds.update_self();
		mkt::print("bds", bds);
		acs.update_self();
		mkt::print("acs", acs);
		bcs.update_self();
		mkt::print("bcs", bcs);
		plusX_map_array_functor.x = 17;
		mkt::map<int, int, PlusX_map_array_functor>(ads, temp, plusX_map_array_functor);
		temp.update_self();
		mkt::print("temp", temp);
		plusX_map_array_functor.x = 42;
		mkt::map<int, int, PlusX_map_array_functor>(bcs, temp_copy, plusX_map_array_functor);
		temp_copy.update_self();
		mkt::print("temp_copy", temp_copy);
		
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", 1);
		
		return EXIT_SUCCESS;
		}
