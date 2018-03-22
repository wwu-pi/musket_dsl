	// reference implementation for nbody simulation with 2 processes and 4 cores per process
	
	#include <mpi.h>
	#include <omp.h>
	#include <array>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include "../include/mpi.hpp"
	
	
	// constants and global vars
	const size_t number_of_processes = 4;
	const int vector_size = 8;
	int process_id = -1;
	size_t tmp_size_t = 0;
	
	void vector_sum(void *in, void *inout, int *len, MPI_Datatype *dptr){
		int* inv = static_cast<int*>(in);
		int* inoutv = static_cast<int*>(inout);
	
		//*inoutv += *inv;
	
		for(int i = 0; i < vector_size; ++i){
			inoutv[i] += inv[i];
		}
	} 
	

	int main(int argc, char** argv) {
	
		// mpi setup
		MPI_Init(&argc, &argv);
		
		int mpi_world_size = 0;
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
		
		if(mpi_world_size != number_of_processes){
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		
		MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
		
		MPI_Op vector_sum_op;
		MPI_Op_create( vector_sum, 0, &vector_sum_op );
		
		if(process_id == 0){
			printf("Run MPI allreduce test\n\n");			
		}
			
		std::vector<int> vs; 
			
		switch(process_id){
		case 0 : {vs.assign(vector_size, 1);
		break;}
		case 1 : {vs.assign(vector_size, 2);
		break;}
		case 2 :{ vs.assign(vector_size, 3);
		break;}
		case 3 :{ vs.assign(vector_size, 4);
		break;}
		}
			
		if(process_id == 0){
			printf("vs:\n");
			for(int i = 0; i < vector_size; ++i){
				printf("%i, ", vs[i]);
			} 
		}
		
		
		int local_fold_result = 0;
		
		for(int i = 0; i < vector_size; ++i){
			local_fold_result += vs[i];
		}
		
		printf("local fold result on P%i: %i\n", process_id, local_fold_result);
		
		int global_fold_result = 0;
		MPI_Allreduce(&local_fold_result, &global_fold_result, sizeof(int), MPI_BYTE, MPI_SUM, MPI_COMM_WORLD);
		//MPI_Allreduce(&fold_result_double, &global_best_fitness, sizeof(double), MPI_BYTE, getBestSolution_mpi_op, MPI_COMM_WORLD); 
				
		printf("global fold result on P%i: %i\n", process_id, global_fold_result);
		
		
		std::vector<int> vector_fold_result(vector_size, 0);
		
		printf("vs.size() = %i and vector_fold_result.size() = %i\n", vs.size(), vector_fold_result.size());
		
		//MPI_Reduce(vs.data(), vector_fold_result.data(), vector_size * sizeof(int), MPI_BYTE, vector_sum_op, 0, MPI_COMM_WORLD);

		MPI_Allreduce(vs.data(), vector_fold_result.data(), vector_size * sizeof(int), MPI_BYTE, vector_sum_op, MPI_COMM_WORLD);
		
		if(process_id == 0){
			printf("vector_fold_result:\n");
			for(int i = 0; i < vector_size; ++i){
				printf("%i, ", vector_fold_result[i]);
			} 
		}
		MPI_Finalize();
		return EXIT_SUCCESS;
	}
