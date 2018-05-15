#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/plus-row.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<int> m(4);


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
	printf("Run Plus-row\n\n");			
	}
	
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4; ++counter){
		m[counter] = 1;
	}
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 2;
		break;
	}
	case 2: {
		row_offset = 2;
		col_offset = 0;
		break;
	}
	case 3: {
		row_offset = 2;
		col_offset = 2;
		break;
	}
	}		
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 2; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 2; ++counter_cols){
			
			m[counter_rows * 2 + counter_cols] = ((m[counter_rows * 2 + counter_cols]) + ((row_offset + counter_rows)));
		}
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
