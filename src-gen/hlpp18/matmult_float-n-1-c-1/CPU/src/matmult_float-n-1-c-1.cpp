
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-1-c-1.hpp"

size_t tmp_size_t = 0;


std::vector<float> as(268435456);
std::vector<float> bs(268435456);
std::vector<float> cs(268435456);


int main(int argc, char** argv) {
	
	printf("Run Matmult_float-n-1-c-1\n\n");			
	
	
	
	
	for(size_t counter = 0; counter  < 268435456; ++counter){
		as[counter] = 1.0f;
	}
	
	for(size_t counter = 0; counter  < 268435456; ++counter){
		bs[counter] = 0.001f;
	}
	
	for(size_t counter = 0; counter  < 268435456; ++counter){
		cs[counter] = 0.0f;
	}
	
	
	#pragma omp simd 
	for(size_t counter_rows = 0; counter_rows < 16384; ++counter_rows){
		for(size_t counter_cols = 0; counter_cols < 16384; ++counter_cols){
			
			as[counter_rows * 16384 + counter_cols] = ((static_cast<float>((counter_rows)) * 4) + (counter_cols));
		}
	}
	#pragma omp simd 
	for(size_t counter_rows = 0; counter_rows < 16384; ++counter_rows){
		for(size_t counter_cols = 0; counter_cols < 16384; ++counter_cols){
			
			bs[counter_rows * 16384 + counter_cols] = ((static_cast<float>(16) + ((counter_rows) * 4)) + (counter_cols));
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	for(int i = 0; ((i) < 1); ++i){
		#pragma omp simd 
		for(size_t counter_rows = 0; counter_rows < 16384; ++counter_rows){
			for(size_t counter_cols = 0; counter_cols < 16384; ++counter_cols){
				
				float sum = (cs[counter_rows * 16384 + counter_cols]);
				for(int k = 0; ((k) < 16384); k++){
					sum += ((as)[(counter_rows) * 16384 + (k)] * (bs)[(k) * 16384 + (counter_cols)]);
				}
				cs[counter_rows * 16384 + counter_cols] = (sum);
			}
		}
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", 1);
	printf("Processes: %i\n", 1);
	
	return EXIT_SUCCESS;
}
