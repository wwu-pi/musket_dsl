#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/frobenius_float.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<float> as(67108864);


void sum(void *in, void *inout, int *len, MPI_Datatype *dptr){
	float* inv = static_cast<float*>(in);
	float* inoutv = static_cast<float*>(inout);
	
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
	printf("Run Frobenius_float\n\n");			
	}
	
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 67108864; ++counter){
		as[counter] = 0;
	}
	#pragma omp declare reduction(sum : float : omp_out = [&](){return ((omp_out) + (omp_in));}()) initializer(omp_priv = omp_orig)
	
	MPI_Op sum_mpi_op;
	MPI_Op_create( sum, 0, &sum_mpi_op );
	float fold_result_float;
	size_t row_offset = 0;size_t col_offset = 0;
	
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 8192;
		break;
	}
	case 2: {
		row_offset = 8192;
		col_offset = 0;
		break;
	}
	case 3: {
		row_offset = 8192;
		col_offset = 8192;
		break;
	}
	}		
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 8192; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 8192; ++counter_cols){
			
			as[counter_rows * 8192 + counter_cols] = ((static_cast<float>(((row_offset + counter_rows))) + ((col_offset + counter_cols))) + 1);
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 67108864; ++counter){
		
		as[counter] = ((as[counter]) * (as[counter]));
	}
	float fn = 0.0f;
	
		fold_result_float  = 0.0f;
		
		#pragma omp parallel for simd reduction(sum:fold_result_float)
		for(size_t counter = 0; counter < 67108864; ++counter){
			
			fold_result_float = ((fold_result_float) + (as[counter]));
		}		
		
		MPI_Allreduce(&fold_result_float, &fn, sizeof(float), MPI_BYTE, sum_mpi_op, MPI_COMM_WORLD); 
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
