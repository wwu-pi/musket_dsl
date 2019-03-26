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
	#include "../include/frobenius-n-16-g-4_14.hpp"
	
	const size_t number_of_processes = 16;
	const size_t process_id = 14;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	
			
	const int dim = 16384;
	mkt::DMatrix<double> as(14, 16384, 16384, 4096, 4096, 268435456, 16777216, 0.0, 4, 4, 3, 2, 12288, 8192, mkt::DIST, mkt::DIST);
	
	

	
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
	struct Square_map_reduce_matrix_functor{
		
		Square_map_reduce_matrix_functor(){
		}
		
		~Square_map_reduce_matrix_functor() {}
		
		auto operator()(double a){
			a = ((a) * (a));
			return (a);
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
	double mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(mkt::DMatrix<double>& a, Square_map_reduce_matrix_functor f){
		double local_result = 0.0;
		double global_result = 0.0;
		
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			for(int gpu = 0; gpu < 4; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				f.init(gpu);
				double* devptr = a.get_device_pointer(gpu);
				const int gpu_elements = a.get_size_gpu();
				double gpu_result = 0.0;
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(+:gpu_result) async(0)
				for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
					#pragma acc cache(gpu_result, devptr[0:gpu_elements])
					double map_result = f(devptr[counter]);
					gpu_result = gpu_result + map_result;
				}
				acc_wait(0);
				local_result = local_result + gpu_result;
			}
		}else if(a.get_device_distribution() == mkt::Distribution::COPY){
			acc_set_device_num(0, acc_device_not_host);
			f.init(0);
			double* devptr = a.get_device_pointer(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result) async(0)
			for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
				#pragma acc cache(local_result, devptr[0:gpu_elements])
				double map_result = f(devptr[counter]);
				local_result = local_result + map_result;
			}
			acc_wait(0);
		}
		
		if(a.get_distribution() == mkt::Distribution::DIST){
			MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			return global_result;
		}else if(a.get_distribution() == mkt::Distribution::COPY){
			return local_result;
		}				
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
		Square_map_reduce_matrix_functor square_map_reduce_matrix_functor{};
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(4096, 4096, 16384, MPI_DOUBLE, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(double) * 4096, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
		
			
		
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		double fn = 0.0;
		fn = mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(as, square_map_reduce_matrix_functor);
		fn = std::sqrt((fn));
		for(int gpu = 0; gpu < 4; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
