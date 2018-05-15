#include <mpi.h>

#include <omp.h>
#include <array>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include <cstddef>
#include "../include/double_0.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 0;
int mpi_rank = -1;
int mpi_world_size = 0;

size_t tmp_size_t = 0;


std::vector<int> a(4);




struct Double_values_functor{
	auto operator()(int j, int i) const{
		return (((i) + (i)) + (j));
	}
};


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	
	if(mpi_world_size != number_of_processes || mpi_rank != process_id){
		MPI_Finalize();
		return EXIT_FAILURE;
	}			
	
	
	printf("Run Double\n\n");			
	
	Double_values_functor double_values_functor{};
	
	
	
	a[0] = 1;
	a[1] = 2;
	a[2] = 3;
	a[3] = 4;
	
	
	

	
	
	
	std::array<int, 16> temp0{};
	MPI_Gather(a.data(), 4, MPI_INT, temp0.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s0;
	s0 << "a: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s0 << temp0[i];
	s0 << "; ";
	}
	s0 << temp0[15] << "]" << std::endl;
	s0 << std::endl;
	printf("%s", s0.str().c_str());
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		a[counter] = double_values_functor(42, a[counter]);
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	std::array<int, 16> temp1{};
	MPI_Gather(a.data(), 4, MPI_INT, temp1.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	std::ostringstream s1;
	s1 << "a: " << std::endl << "[";
	for (int i = 0; i < 15; i++) {
	s1 << temp1[i];
	s1 << "; ";
	}
	s1 << temp1[15] << "]" << std::endl;
	s1 << std::endl;
	printf("%s", s1.str().c_str());
	
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
