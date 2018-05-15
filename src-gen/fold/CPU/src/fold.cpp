#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/fold.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<int> as(4);
std::vector<int> bs(4);
std::vector<int> cs(4);


void sum(void *in, void *inout, int *len, MPI_Datatype *dptr){
	int* inv = static_cast<int*>(in);
	int* inoutv = static_cast<int*>(inout);
	
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
	printf("Run Fold\n\n");			
	}
	
	
	
	switch(process_id){
	case 0: {
	cs[0] = 1;
	cs[1] = 2;
	cs[2] = 3;
	cs[3] = 4;
	break;
	}
	case 1: {
	cs[0] = 5;
	cs[1] = 6;
	cs[2] = 7;
	cs[3] = 8;
	break;
	}
	case 2: {
	cs[0] = 9;
	cs[1] = 10;
	cs[2] = 11;
	cs[3] = 12;
	break;
	}
	case 3: {
	cs[0] = 13;
	cs[1] = 14;
	cs[2] = 15;
	cs[3] = 16;
	break;
	}
	}
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4; ++counter){
		as[counter] = 7;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 4; ++counter){
		bs[counter] = 0;
	}
	#pragma omp declare reduction(sum : int : omp_out = [&](){return ((omp_out) + (omp_in));}()) initializer(omp_priv = omp_orig)
	
	MPI_Op sum_mpi_op;
	MPI_Op_create( sum, 0, &sum_mpi_op );
	int fold_result_int;
	
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	int a  = 0;
	int b  = 0;
	int c  = 0;
	for(int i = 0; ((i) < 3); i++){
		
			fold_result_int  = 0;
			
			#pragma omp parallel for simd reduction(sum:fold_result_int)
			for(size_t counter = 0; counter < 4; ++counter){
				
				fold_result_int = ((fold_result_int) + (as[counter]));
			}		
			
			MPI_Allreduce(&fold_result_int, &a, sizeof(int), MPI_BYTE, sum_mpi_op, MPI_COMM_WORLD); 
		
			fold_result_int  = 0;
			
			#pragma omp parallel for simd reduction(sum:fold_result_int)
			for(size_t counter = 0; counter < 4; ++counter){
				
				fold_result_int = ((fold_result_int) + (bs[counter]));
			}		
			
			MPI_Allreduce(&fold_result_int, &b, sizeof(int), MPI_BYTE, sum_mpi_op, MPI_COMM_WORLD); 
		
			fold_result_int  = 0;
			
			#pragma omp parallel for simd reduction(sum:fold_result_int)
			for(size_t counter = 0; counter < 4; ++counter){
				
				fold_result_int = ((fold_result_int) + (cs[counter]));
			}		
			
			MPI_Allreduce(&fold_result_int, &c, sizeof(int), MPI_BYTE, sum_mpi_op, MPI_COMM_WORLD); 
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	if(process_id == 0){
		printf("a = %i; b = %i; c = %i\n",(a),(b),(c));
	}
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
