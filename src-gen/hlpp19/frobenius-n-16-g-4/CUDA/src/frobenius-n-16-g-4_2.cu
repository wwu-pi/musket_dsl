	#include <mpi.h>
	#include <cuda.h>
	#include <omp.h>
	#include <stdlib.h>
	#include <math.h>
	#include <array>
	#include <vector>
	#include <sstream>
	#include <chrono>
	#include <curand_kernel.h>
	#include <limits>
	#include <memory>
	#include <cstddef>
	#include <type_traits>
	
	
	#include "../include/musket.cuh"
	#include "../include/frobenius-n-16-g-4_2.cuh"
	
	const size_t number_of_processes = 16;
	const size_t process_id = 2;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
			
	const int dim = 16384;
	mkt::DMatrix<double> as(2, 16384, 16384, 4096, 4096, 268435456, 16777216, 0.0, 4, 4, 0, 2, 0, 8192, mkt::DIST, mkt::DIST);
	
	

	
	struct Init_map_index_in_place_matrix_functor{
		
		Init_map_index_in_place_matrix_functor(){}
		
		~Init_map_index_in_place_matrix_functor() {}
		
		__device__
		auto operator()(int x, int y, double& a){
			a = static_cast<double>((((x) + (y)) + 1));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Square_map_reduce_matrix_functor{
		
		Square_map_reduce_matrix_functor(){}
		
		~Square_map_reduce_matrix_functor() {}
		
		__device__
		auto operator()(double a){
			a = ((a) * (a));
			return (a);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	
	
	
	
	template<>
	double mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(mkt::DMatrix<double>& a, Square_map_reduce_matrix_functor f){
			double local_result = 0.0;
			double global_result = 0.0;
						
			if(a.get_device_distribution() == mkt::Distribution::DIST){
				std::array<double*,4> d_odata;
				std::array<double, 4> gpu_results;
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 256 ? gpu_elements : 256; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
		
				for(int gpu = 0; gpu < 4; ++gpu){
					cudaSetDevice(gpu);
					cudaMalloc((void**) &d_odata[i], blocks * sizeof(double));
					double* devptr = a.get_device_pointer(gpu);
					
					mkt::kernel::map_reduce_plus_call<double, double, Square_map_reduce_matrix_functor>(gpu_elements, devptr, d_odata[gpu], threads, blocks, f, mkt::cuda_streams[gpu], gpu);
				}
				mkt::sync_streams();
				
				// fold on gpus: step 2
				while(blocks > 1){
			      int threads_2 = blocks < 256 ? blocks : 256; // nextPow2
			      int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  for(int gpu = 0; gpu < 4; ++gpu){
				      cudaSetDevice(gpu);
				      mkt::kernel::reduce_plus_call<double>(blocks, d_odata[gpu], d_odata[gpu], threads_2, blocks_2, mkt::cuda_streams[gpu], gpu);
				  }
				  blocks = blocks_2;
			  	  mkt::sync_streams();
			  	}
				
				// copy final sum from device to host
				  for (int gpu = 0; gpu < 4; ++gpu) {
				    cudaSetDevice(gpu);
				    cudaMemcpyAsync(&gpu_results[gpu], d_odata[gpu], sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
				  }
				  mkt::sync_streams();
				  
				  for(int gpu = 0; gpu < 4; ++gpu) {
					cudaSetDevice(gpu);
					cudaFree(d_odata[gpu]);
				  }
				
				for(int gpu = 0; gpu < 4; ++gpu){
					local_result = local_result + gpu_results[gpu];
				}
			}else if(a.get_device_distribution() == mkt::Distribution::COPY){ // use only gpu 0, since all have the same data
				const int gpu_elements = a.get_size_gpu();
				int threads = gpu_elements < 256 ? gpu_elements : 256; // nextPow2
				int blocks = (gpu_elements + threads - 1) / threads;
				cudaSetDevice(0);
				double* d_odata;
				cudaMalloc((void**) &d_odata, blocks * sizeof(double));
				double* devptr = a.get_device_pointer(0);
				
				mkt::kernel::map_reduce_plus_call<double, double, Square_map_reduce_matrix_functor>(gpu_elements, devptr, d_odata, threads, blocks, f, mkt::cuda_streams[0], 0);
				
				// fold on gpus: step 2
				while(blocks > 1){
				  int threads_2 = blocks < 256 ? blocks : 256; // nextPow2
				  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
				  mkt::kernel::reduce_plus_call<double>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
				  blocks = blocks_2;
				}
				
				// copy final sum from device to host
				  cudaMemcpyAsync(&local_result, d_odata, sizeof(double), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
				  mkt::sync_streams();
				cudaFree(d_odata);
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
		mkt::init();
		
		
		Init_map_index_in_place_matrix_functor init_map_index_in_place_matrix_functor{};
		Square_map_reduce_matrix_functor square_map_reduce_matrix_functor{};
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(4096, 4096, 16384, MPI_DOUBLE, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(double) * 4096, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
		
			
		
		
		mkt::map_index_in_place<double, Init_map_index_in_place_matrix_functor>(as, init_map_index_in_place_matrix_functor);
		mkt::sync_streams();
		double fn = 0.0;
		fn = mkt::map_reduce_plus<double, double, Square_map_reduce_matrix_functor>(as, square_map_reduce_matrix_functor);
		fn = std::sqrt((fn));
		mkt::sync_streams();
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
