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
#include "../include/double_2.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 2;
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
	
	
	
	Double_values_functor double_values_functor{};
	
	
	
	a[0] = 9;
	a[1] = 10;
	a[2] = 11;
	a[3] = 12;
	
	
	

	
	
	
	MPI_Gather(a.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter < 4; ++counter){
		a[counter] = double_values_functor(42, a[counter]);
	}
	MPI_Gather(a.data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
	
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
