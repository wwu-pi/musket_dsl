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
	#include <type_traits>
	
	#include "../include/musket.hpp"
	#include "../include/array_3.hpp"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 3;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	

	
	const int dim = 16;
	mkt::DArray<int> ads(3, 16, 4, 1, 2, 3, 12, mkt::DIST);
	mkt::DArray<int> bds(3, 16, 4, 0, 2, 3, 12, mkt::DIST);
	mkt::DArray<int> acs(3, 16, 16, 7, 1, 3, 0, mkt::COPY);
	mkt::DArray<int> bcs(3, 16, 16, 0, 1, 3, 0, mkt::COPY);
	mkt::DArray<int> temp(3, 16, 4, 0, 2, 3, 12, mkt::DIST);
	mkt::DArray<int> temp_copy(3, 16, 16, 0, 1, 3, 0, mkt::COPY);
	
	

	
	struct PlusX_map_array_functor{
		auto operator()(const int v) const{
			return ((x) + (v));
		}
		
		int x;
	};
	
	
	
	
	
	
	int main(int argc, char** argv) {
		MPI_Init(&argc, &argv);
		
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		
		if(mpi_world_size != number_of_processes || mpi_rank != process_id){
			MPI_Finalize();
			return EXIT_FAILURE;
		}			
		
		
		
				PlusX_map_array_functor plusX_map_array_functor{};
		
		
		
				
			
			
		
			
		
		
		MPI_Gather(ads.get_data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(bds.get_data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		// show array (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		// show array (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		plusX_map_array_functor.x = 17;
		mkt::map<int, int, PlusX_map_array_functor>(ads, temp, plusX_map_array_functor);
		MPI_Gather(temp.get_data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		plusX_map_array_functor.x = 42;
		mkt::map<int, int, PlusX_map_array_functor>(bcs, temp_copy, plusX_map_array_functor);
		// show array (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
