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
	#include "../include/frobenius-n-4-g-2_0.hpp"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 0;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	
			
	const int dim = 16384;
	mkt::DMatrix<double> as(0, 16384, 16384, 8192, 8192, 268435456, 67108864, 0.0, 2, 2, 0, 0, 0, 0, mkt::DIST, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		
		Init_map_index_in_place_matrix_functor(){
		}
		
		~Init_map_index_in_place_matrix_functor() {}
		
		auto operator()(int x, int y, double& a){
			a = static_cast<double>((((x) + (y)) + 1));
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
	struct Square_map_in_place_matrix_functor{
		
		Square_map_in_place_matrix_functor(){
		}
		
		~Square_map_in_place_matrix_functor() {}
		
		auto operator()(double& a){
			a = ((a) * (a));
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
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(+:gpu_result) async(0)
			for (int counter = 0; counter < gpu_elements; ++counter) {
				#pragma acc cache(gpu_result)
				gpu_result = gpu_result + devptr[counter];
			}
			acc_wait(0);
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
		
		
		printf("Run Frobenius-n-4-g-2\n\n");
		
		
		Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
		Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(8192, 8192, 16384, MPI_DOUBLE, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(double) * 8192, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
		
			
		
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
		mkt::map_in_place<double, Square_map_in_place_matrix_functor>(as, square_map_in_place_matrix_functor);
		double fn = 0.0;
		fn = mkt::reduce_plus<double>(as);
		fn = std::sqrt((fn));
		std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
		double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
		printf("Frobenius norm is %.5f.\n",(fn));
		
		printf("Execution time: %.5fs\n", seconds);
		printf("Threads: %i\n", omp_get_max_threads());
		printf("Processes: %i\n", mpi_world_size);
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
