#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>

size_t tmp_size_t = 0;


std::vector<double> as(4294967296);


int main(int argc, char** argv) {
	
	printf("Run Frobenius_mapfold inline on one node\n\n");			
		
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4294967296; ++counter){
		as[counter] = 0;
	}
	#pragma omp declare reduction(sum : double : omp_out = [&](){return ((omp_out) + (omp_in));}()) initializer(omp_priv = omp_orig)

	#pragma omp parallel for
	for(size_t counter_rows = 0; counter_rows < 65536; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 65536; ++counter_cols){
			
			as[counter_rows * 65536 + counter_cols] = static_cast<double>(counter_rows + counter_cols + 1.53);
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		
	double fn = 0.0;	
	
	#pragma omp parallel for simd reduction(sum:fn)
	for(size_t counter = 0; counter < 4294967296; ++counter){
		double map_fold_tmp = as[counter] * as[counter];
		fn += map_fold_tmp;
	}		
	
	fn = std::sqrt(fn);
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();

	printf("Frobenius norm is %.5f.\n",(fn));	

	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);

	return EXIT_SUCCESS;
}
