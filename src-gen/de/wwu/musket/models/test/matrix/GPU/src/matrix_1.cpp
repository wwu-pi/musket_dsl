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
	#include "../include/matrix_1.hpp"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 1;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	
			
	mkt::DMatrix<int> ads(1, 4, 4, 2, 2, 16, 4, 7, 2, 2, 0, 1, 0, 2, mkt::DIST, mkt::COPY);
	mkt::DMatrix<int> bds(1, 4, 4, 2, 2, 16, 4, 0, 2, 2, 0, 1, 0, 2, mkt::DIST, mkt::COPY);
	mkt::DMatrix<int> acs(1, 4, 4, 4, 4, 16, 16, 7, 1, 1, 0, 0, 0, 0, mkt::COPY, mkt::COPY);
	mkt::DMatrix<int> bcs(1, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::COPY, mkt::COPY);
	mkt::DMatrix<int> r_ads(1, 4, 4, 2, 2, 16, 4, 0, 2, 2, 0, 1, 0, 2, mkt::DIST, mkt::COPY);
	mkt::DMatrix<int> r_bds(1, 4, 4, 2, 2, 16, 4, 0, 2, 2, 0, 1, 0, 2, mkt::DIST, mkt::COPY);
	mkt::DMatrix<int> r_acs(1, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::COPY, mkt::COPY);
	mkt::DMatrix<int> r_bcs(1, 4, 4, 4, 4, 16, 16, 0, 1, 1, 0, 0, 0, 0, mkt::COPY, mkt::COPY);
	
	

	
	struct Init_map_index_matrix_functor{
		
		Init_map_index_matrix_functor(){
		}
		
		~Init_map_index_matrix_functor() {}
		
		auto operator()(const int row, const int col, const int x){
			return (((row) * 4) + (col));
		}
	
		void init(int gpu){
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		
		
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
		
		
		
		
		Init_map_index_matrix_functor init_map_index_matrix_functor{};
		
		
				
			
			
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
			MPI_Datatype r_ads_partition_type;
			MPI_Type_vector(2, 2, 4, MPI_INT, &r_ads_partition_type);
			MPI_Type_create_resized(r_ads_partition_type, 0, sizeof(int) * 2, &r_ads_partition_type_resized);
			MPI_Type_free(&r_ads_partition_type);
			MPI_Type_commit(&r_ads_partition_type_resized);
			MPI_Datatype r_bds_partition_type;
			MPI_Type_vector(2, 2, 4, MPI_INT, &r_bds_partition_type);
			MPI_Type_create_resized(r_bds_partition_type, 0, sizeof(int) * 2, &r_bds_partition_type_resized);
			MPI_Type_free(&r_bds_partition_type);
			MPI_Type_commit(&r_bds_partition_type_resized);
		
			
		
		
		// show matrix dist
		ads.update_self();
		MPI_Gatherv(ads.get_data(), 4, MPI_INT, nullptr, nullptr, nullptr, nullptr, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		// show matrix dist
		bds.update_self();
		MPI_Gatherv(bds.get_data(), 4, MPI_INT, nullptr, nullptr, nullptr, nullptr, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		// show matrix (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		// show matrix (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		mkt::map_index<int, int, Init_map_index_matrix_functor>(ads, r_ads, init_map_index_matrix_functor);
		mkt::map_index<int, int, Init_map_index_matrix_functor>(bds, r_bds, init_map_index_matrix_functor);
		mkt::map_index<int, int, Init_map_index_matrix_functor>(acs, r_acs, init_map_index_matrix_functor);
		mkt::map_index<int, int, Init_map_index_matrix_functor>(bcs, r_bcs, init_map_index_matrix_functor);
		// show matrix dist
		r_ads.update_self();
		MPI_Gatherv(r_ads.get_data(), 4, MPI_INT, nullptr, nullptr, nullptr, nullptr, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		// show matrix dist
		r_bds.update_self();
		MPI_Gatherv(r_bds.get_data(), 4, MPI_INT, nullptr, nullptr, nullptr, nullptr, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		// show matrix (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		// show matrix (copy) --> only in p0
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
