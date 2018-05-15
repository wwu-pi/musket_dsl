#include <mpi.h>

#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/frobenius-n-16-c-1.hpp"

const size_t number_of_processes = 16;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<double> as(268435456);


void sum(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	*inoutv = ((*inoutv) + (*inv));
} 
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
	printf("Run Frobenius-n-16-c-1\n\n");			
	}
	
	
	
	
	for(size_t counter = 0; counter  < 268435456; ++counter){
		as[counter] = 0;
	}
	
	MPI_Op sum_mpi_op;
	MPI_Op_create( sum, 0, &sum_mpi_op );
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
		col_offset = 16384;
		break;
	}
	case 2: {
		row_offset = 0;
		col_offset = 32768;
		break;
	}
	case 3: {
		row_offset = 0;
		col_offset = 49152;
		break;
	}
	case 4: {
		row_offset = 16384;
		col_offset = 0;
		break;
	}
	case 5: {
		row_offset = 16384;
		col_offset = 16384;
		break;
	}
	case 6: {
		row_offset = 16384;
		col_offset = 32768;
		break;
	}
	case 7: {
		row_offset = 16384;
		col_offset = 49152;
		break;
	}
	case 8: {
		row_offset = 32768;
		col_offset = 0;
		break;
	}
	case 9: {
		row_offset = 32768;
		col_offset = 16384;
		break;
	}
	case 10: {
		row_offset = 32768;
		col_offset = 32768;
		break;
	}
	case 11: {
		row_offset = 32768;
		col_offset = 49152;
		break;
	}
	case 12: {
		row_offset = 49152;
		col_offset = 0;
		break;
	}
	case 13: {
		row_offset = 49152;
		col_offset = 16384;
		break;
	}
	case 14: {
		row_offset = 49152;
		col_offset = 32768;
		break;
	}
	case 15: {
		row_offset = 49152;
		col_offset = 49152;
		break;
	}
	}		
	#pragma omp simd 
	for(size_t counter_rows = 0; counter_rows < 16384; ++counter_rows){
		for(size_t counter_cols = 0; counter_cols < 16384; ++counter_cols){
			
			as[counter_rows * 16384 + counter_cols] = ((static_cast<double>(((row_offset + counter_rows))) + ((col_offset + counter_cols))) + 1);
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp simd
	for(size_t counter = 0; counter < 268435456; ++counter){
		
		as[counter] = ((as[counter]) * (as[counter]));
	}
	double fn = 0.0;
	
		fold_result_double  = 0.0;
		
		#pragma omp simd reduction(sum:fold_result_double)
		for(size_t counter = 0; counter < 268435456; ++counter){
			
			fold_result_double = ((fold_result_double) + (as[counter]));
		}		
		
		MPI_Allreduce(&fold_result_double, &fn, sizeof(double), MPI_BYTE, sum_mpi_op, MPI_COMM_WORLD); 
	fn = std::sqrt((fn));
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	if(process_id == 0){
		printf("Frobenius norm is %.5f.\n",(fn));
	}
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", 1);
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
