	#include <mpi.h>
	#include <cuda.h>
	#include <omp.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <random>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	
	
	#include "../include/musket.cuh"
	#include "../include/array_2.cuh"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 2;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
			
	const int dim = 16;
	mkt::DArray<int> ads(2, 16, 4, 1, 2, 2, 8, mkt::DIST, mkt::COPY);
	mkt::DArray<int> bds(2, 16, 4, 0, 2, 2, 8, mkt::DIST, mkt::COPY);
	mkt::DArray<int> acs(2, 16, 16, 7, 1, 2, 0, mkt::COPY, mkt::COPY);
	mkt::DArray<int> bcs(2, 16, 16, 0, 1, 2, 0, mkt::COPY, mkt::COPY);
	mkt::DArray<int> temp(2, 16, 4, 0, 2, 2, 8, mkt::DIST, mkt::COPY);
	mkt::DArray<int> temp_copy(2, 16, 16, 0, 1, 2, 0, mkt::COPY, mkt::COPY);
	
	

	
	struct PlusX_map_array_functor{
		
		PlusX_map_array_functor(){}
		
		~PlusX_map_array_functor() {}
		
		__device__
		auto operator()(const int v){
			return ((x) + (v));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
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
		
		mkt::init();
		
		
		PlusX_map_array_functor plusX_map_array_functor{};
		
		
				
			
			
		
			
		
		
		ads.update_self();
		MPI_Gather(ads.get_data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		bds.update_self();
		MPI_Gather(bds.get_data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		// show array (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		// show array (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		plusX_map_array_functor.x = 17;
		mkt::map<int, int, PlusX_map_array_functor>(ads, temp, plusX_map_array_functor);
		temp.update_self();
		MPI_Gather(temp.get_data(), 4, MPI_INT, nullptr, 4, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		plusX_map_array_functor.x = 42;
		mkt::map<int, int, PlusX_map_array_functor>(bcs, temp_copy, plusX_map_array_functor);
		// show array (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
