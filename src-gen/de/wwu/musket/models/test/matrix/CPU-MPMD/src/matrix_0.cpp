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
	#include "../include/matrix_0.hpp"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 0;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	

	
	mkt::DMatrix<int> ads(0, 4, 4, 2, 2, 16, 4, 7, 2, 2, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<int> bds(0, 4, 4, 2, 2, 16, 4, 0, 2, 2, 0, 0, 0, 0, mkt::DIST);
	mkt::DMatrix<int> acs(0, 4, 4, 4, 4, 16, 16, 7, 1, 1, 0, 0, 0, 0, mkt::COPY);
	mkt::DMatrix<int> bcs(0, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::COPY);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		auto operator()(int row, int col, int x) const{
			return (((row) * 4) + (col));
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
		
		
		printf("Run Matrix\n\n");
		
				Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
		
		
		
				
			
			
			MPI_Datatype ads_partition_type;
			MPI_Type_vector(2, 2, 4, MPI_INT, &ads_partition_type);
			MPI_Type_create_resized(ads_partition_type, 0, sizeof(int) * 2, &ads_partition_type_resized);
			MPI_Type_free(&ads_partition_type);
			MPI_Type_commit(&ads_partition_type_resized);
			MPI_Datatype bds_partition_type;
			MPI_Type_vector(2, 2, 4, MPI_INT, &bds_partition_type);
			MPI_Type_create_resized(bds_partition_type, 0, sizeof(int) * 2, &bds_partition_type_resized);
			MPI_Type_free(&bds_partition_type);
			MPI_Type_commit(&bds_partition_type_resized);
		
			
		
		
		mkt::print_dist_ads(ads, ads_partition_type_resized);
		MPI_Barrier(MPI_COMM_WORLD);
		mkt::print_dist_bds(bds, bds_partition_type_resized);
		MPI_Barrier(MPI_COMM_WORLD);
		mkt::print("acs", acs);
		MPI_Barrier(MPI_COMM_WORLD);
		mkt::print("bcs", bcs);
		MPI_Barrier(MPI_COMM_WORLD);
		mkt::map_index_in_place<int, Init_map_index_in_place_matrix_functor>(ads, init_map_index_in_place_matrix_functor);
		mkt::map_index_in_place<int, Init_map_index_in_place_matrix_functor>(acs, init_map_index_in_place_matrix_functor);
		mkt::print_dist_ads(ads, ads_partition_type_resized);
		MPI_Barrier(MPI_COMM_WORLD);
		mkt::print("acs", acs);
		MPI_Barrier(MPI_COMM_WORLD);
		
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", mpi_world_size);
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
