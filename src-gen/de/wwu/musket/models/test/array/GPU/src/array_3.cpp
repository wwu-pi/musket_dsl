	#include <mpi.h>
	
	#include <omp.h>
	#include <openacc.h>
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
	//#include <cuda.h>
	//#include <openacc_curand.h>
	
	#include "../include/musket.hpp"
	#include "../include/array_3.hpp"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 3;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	
			
	const int dim = 16;
	mkt::DArray<int> ads(3, 16, 4, 1, 2, 3, 12, mkt::DIST, mkt::COPY);
	mkt::DArray<int> bds(3, 16, 4, 0, 2, 3, 12, mkt::DIST, mkt::COPY);
	mkt::DArray<int> acs(3, 16, 16, 7, 1, 3, 0, mkt::COPY, mkt::COPY);
	mkt::DArray<int> bcs(3, 16, 16, 0, 1, 3, 0, mkt::COPY, mkt::COPY);
	mkt::DArray<int> temp(3, 16, 4, 0, 2, 3, 12, mkt::DIST, mkt::COPY);
	mkt::DArray<int> temp_copy(3, 16, 16, 0, 1, 3, 0, mkt::COPY, mkt::COPY);
	
	

	
	struct PlusX_map_array_functor{
		
		PlusX_map_array_functor(){
		}
		
		~PlusX_map_array_functor() {}
		
		auto operator()(const int v){
			return ((x) + (v));
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		int x;
		
		
		
		int _gang;
		int _worker;
		int _vector;
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
