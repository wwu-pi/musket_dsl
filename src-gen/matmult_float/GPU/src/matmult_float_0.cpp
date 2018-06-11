	
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
	#include "../include/matmult_float_0.hpp"
	
	
	

	
	const int dim = 4;
	mkt::DMatrix<float> as(0, 4, 4, 4, 4, 16, 16, 1.0f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<float> bs(0, 4, 4, 4, 4, 16, 16, 0.001f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<float> cs(0, 4, 4, 4, 4, 16, 16, 0.0f, 1, 1, 0, 0, 0, 0, mkt::DIST);
	
	

	
	struct InitA_map_index_in_place_matrix_functor{
		
		InitA_map_index_in_place_matrix_functor() {}
		
		auto operator()(int a, int b, float& x) const{
			x = ((static_cast<float>((a)) * 4) + (b));
		}
	
		void init(int gpu){
		}
		
		
	};
	struct InitB_map_index_in_place_matrix_functor{
		
		InitB_map_index_in_place_matrix_functor() {}
		
		auto operator()(int a, int b, float& x) const{
			x = ((static_cast<float>(16) + ((a) * 4)) + (b));
		}
	
		void init(int gpu){
		}
		
		
	};
	struct DotProduct_map_local_index_in_place_matrix_functor{
		
		DotProduct_map_local_index_in_place_matrix_functor(const mkt::DMatrix<float>& _as, const mkt::DMatrix<float>& _bs) : as(_as), bs(_bs) {}
		
		auto operator()(int i, int j, float& Cij) const{
			for(int k = 0; ((k) < 4); k++){
				Cij += (as.get_data_local((i), (k)) * bs.get_data_local((k), (j)));
			}
		}
	
		void init(int gpu){
			as.init(gpu);
			bs.init(gpu);
		}
		
		
		mkt::DeviceMatrix<float> as;
		mkt::DeviceMatrix<float> bs;
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		
		
				InitA_map_index_in_place_matrix_functor initA_map_index_in_place_matrix_functor{};
				InitB_map_index_in_place_matrix_functor initB_map_index_in_place_matrix_functor{};
				DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{as, bs};
		
		
		
				
		
		mkt::map_index_in_place<float, InitA_map_index_in_place_matrix_functor>(as, initA_map_index_in_place_matrix_functor);
		mkt::map_index_in_place<float, InitB_map_index_in_place_matrix_functor>(bs, initB_map_index_in_place_matrix_functor);
		as.update_self();
		mkt::print("as", as);
		bs.update_self();
		mkt::print("bs", bs);
		cs.update_self();
		mkt::print("cs", cs);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		for(int i = 0; ((i) < 1); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
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
