
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/frobenius-n-1-c-1.hpp"

size_t tmp_size_t = 0;


std::vector<double> as(1073741824);


int main(int argc, char** argv) {
	
	printf("Run Frobenius-n-1-c-1\n\n");			
	
	
	
	
	for(size_t counter = 0; counter  < 1073741824; ++counter){
		as[counter] = 0;
	}
	
	
	#pragma omp simd 
	for(size_t counter_rows = 0; counter_rows < 32768; ++counter_rows){
		for(size_t counter_cols = 0; counter_cols < 32768; ++counter_cols){
			
			as[counter_rows * 32768 + counter_cols] = ((static_cast<double>((counter_rows)) + (counter_cols)) + 1);
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp simd
	for(size_t counter = 0; counter < 1073741824; ++counter){
		
		as[counter] = ((as[counter]) * (as[counter]));
	}
	double fn = 0.0;
	
		fn = 0.0;
		
		#pragma omp simd reduction(sum:fn)
		for(size_t counter = 0; counter < 1073741824; ++counter){
			
			fn = ((fn) + (as[counter]));
		}		
		
	fn = std::sqrt((fn));
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Frobenius norm is %.5f.\n",(fn));
	
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", 1);
	printf("Processes: %i\n", 1);
	
	return EXIT_SUCCESS;
}
