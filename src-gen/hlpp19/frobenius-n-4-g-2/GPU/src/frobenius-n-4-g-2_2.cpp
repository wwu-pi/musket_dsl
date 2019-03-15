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
	#include <cuda.h>
	#include <openacc_curand.h>
	
	#include "../include/musket.hpp"
	#include "../include/frobenius-n-4-g-2_2.hpp"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 2;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	

	
	const int dim = 32768;
	mkt::DMatrix<double> as(2, 32768, 32768, 16384, 16384, 1073741824, 268435456, 0.0, 2, 2, 1, 0, 16384, 0, mkt::DIST, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		
		Init_map_index_in_place_matrix_functor() {}
		
		auto operator()(int x, int y, double& a) const{
			a = static_cast<double>((((x) + (y)) + 1));
		}
	
		void init(int gpu){
		}
		
		
	};
	struct Square_map_in_place_matrix_functor{
		
		Square_map_in_place_matrix_functor() {}
		
		auto operator()(double& a) const{
			a = ((a) * (a));
		}
	
		void init(int gpu){
		}
		
		
	};
	
	
	
	template<>
	double mkt::reduce_plus<double>(mkt::DMatrix<double>& a){
		double local_result = 0.0;
		double global_result = 0.0;
		
		#pragma omp parallel for reduction(+:local_result)
		for(int gpu = 0; gpu < 2; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			double* devptr = a.get_device_pointer(gpu);
			const int gpu_elements = a.get_size_gpu();
			double gpu_result = 0.0;
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(+:gpu_result)
			for (int counter = 0; counter < gpu_elements; ++counter) {
				#pragma acc cache(gpu_result)
				gpu_result = gpu_result + devptr[counter];
			}
			local_result = local_result + gpu_result;
		}
		
		MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		return global_result;
	}
	
	
	
	int main(int argc, char** argv) {
		MPI_Init(&argc, &argv);
		
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		
		if(mpi_world_size != number_of_processes || mpi_rank != process_id){
			MPI_Finalize();
			return EXIT_FAILURE;
		}			
		
		
		
				Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
				Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(16384, 16384, 32768, MPI_DOUBLE, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(double) * 16384, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
		
			
		
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		mkt::map_in_place<double, Square_map_in_place_matrix_functor>(as, square_map_in_place_matrix_functor);
		double fn = 0.0;
		fn = mkt::reduce_plus<double>(as);
		fn = std::sqrt((fn));
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
