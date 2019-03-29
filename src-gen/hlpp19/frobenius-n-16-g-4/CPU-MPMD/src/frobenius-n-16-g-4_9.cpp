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
	#include "../include/frobenius-n-16-g-4_9.hpp"
	
	const size_t number_of_processes = 16;
	const size_t process_id = 9;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	

	
	const int dim = 16384;
	mkt::DMatrix<double> as(9, 16384, 16384, 4096, 4096, 268435456, 16777216, 0.0, 4, 4, 2, 1, 8192, 4096, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		auto operator()(int x, int y, double& a) const{
			a = static_cast<double>((((x) + (y)) + 1));
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
		
		
		
				Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
		
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(4096, 4096, 16384, MPI_DOUBLE, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(double) * 4096, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
		
			
		
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		double fn = 0.0;
		// TODO: SkeletonGenerator.generateSkeletonExpression: default case
		fn = std::sqrt((fn));
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}