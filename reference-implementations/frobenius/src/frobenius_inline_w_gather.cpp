#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<double> as(1073741824);

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	int mpi_world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	
	if(mpi_world_size != number_of_processes){
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	
	if(process_id == 0){
	printf("Run Frobenius inline w gather\n\n");			
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 1073741824; ++counter){
		as[counter] = 0;
	}
	#pragma omp declare reduction(sum : double : omp_out = [&](){return ((omp_out) + (omp_in));}()) initializer(omp_priv = omp_orig)

	double fold_result_double;
	size_t row_offset = 0;size_t col_offset = 0;
	
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 32768;
		break;
	}
	case 2: {
		row_offset = 32768;
		col_offset = 0;
		break;
	}
	case 3: {
		row_offset = 32768;
		col_offset = 32768;
		break;
	}
	}		
	#pragma omp parallel for
	for(size_t counter_rows = 0; counter_rows < 32768; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 32768; ++counter_cols){
			
			as[counter_rows * 32768 + counter_cols] = ((static_cast<double>(((row_offset + counter_rows))) + ((col_offset + counter_cols))) + 1.53);
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for
	for(size_t counter = 0; counter < 1073741824; ++counter){
		
		as[counter] = ((as[counter]) * (as[counter]));
	}
	
	double fn = 0.0;
	std::array<double, 4> local_results;
		fold_result_double  = 0.0;
	
		#pragma omp parallel for reduction(sum:fold_result_double)
		for(size_t counter = 0; counter < 1073741824; ++counter){
			
			fold_result_double = fold_result_double + as[counter];
		}		
		
		MPI_Allgather(&fold_result_double, sizeof(double), MPI_BYTE, local_results.data(), sizeof(double), MPI_BYTE, MPI_COMM_WORLD); 
		
		for(int k = 0; k < 4; ++k){
			fn += local_results[k];
		}
		
	fn = std::sqrt((fn));
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	if(process_id == 0){
		printf("Frobenius norm is %.5f.\n",(fn));
	}
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
