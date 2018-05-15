#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/double.hpp"

const size_t number_of_processes = 4;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<int> a(1);


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
	printf("Run Double\n\n");			
	}
	
	
	
	switch(process_id){
	case 0: {
	a[0] = 1;
	break;
	}
	case 1: {
	a[0] = 2;
	break;
	}
	case 2: {
	a[0] = 3;
	break;
	}
	case 3: {
	a[0] = 4;
	break;
	}
	}
	
	
	
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 1; ++counter){
		
		a[counter] = (((a[counter]) + (a[counter])) + (42));
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
