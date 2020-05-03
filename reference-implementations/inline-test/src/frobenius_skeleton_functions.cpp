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
MPI_Op sum_mpi_op;

std::vector<double> as(1073741824);

inline void map_square(){
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 1073741824; ++counter){
			as[counter] = as[counter] * as[counter];
	}
}

inline double fold_sum(){
		double fold_result_double  = 0.0;
		double fn;
		#pragma omp declare reduction(sum : double : omp_out += omp_in) initializer(omp_priv = omp_orig)
		#pragma omp parallel for simd reduction(sum:fold_result_double)
		for(size_t counter = 0; counter < 1073741824; ++counter){			
			fold_result_double = fold_result_double + as[counter];
		}		
		
		MPI_Allreduce(&fold_result_double, &fn, sizeof(double), MPI_BYTE, sum_mpi_op, MPI_COMM_WORLD);
		return fn;
}


void sum_mpi(void *in, void *inout, int *len, MPI_Datatype *dptr){
	double* inv = static_cast<double*>(in);
	double* inoutv = static_cast<double*>(inout);
	
	*inoutv = (*inoutv + *inv);
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
	printf("Run Frobenius skeleton functions\n\n");			
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 1073741824; ++counter){
		as[counter] = 0;
	}
	
	
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
	
	MPI_Op_create( sum_mpi, 0, &sum_mpi_op );
	
	map_square();
	
	double fn = fold_sum(); 
	
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
