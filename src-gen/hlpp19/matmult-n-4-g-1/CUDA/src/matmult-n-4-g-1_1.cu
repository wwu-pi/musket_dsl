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
	#include "../include/matmult-n-4-g-1_1.cuh"
	
	const size_t number_of_processes = 4;
	const size_t process_id = 1;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
			
	const int dim = 16384;
	mkt::DMatrix<float> as(1, 16384, 16384, 8192, 8192, 268435456, 67108864, 1.0f, 2, 2, 0, 1, 0, 8192, mkt::DIST, mkt::DIST);
	mkt::DMatrix<float> bs(1, 16384, 16384, 8192, 8192, 268435456, 67108864, 0.001f, 2, 2, 0, 1, 0, 8192, mkt::DIST, mkt::COPY);
	mkt::DMatrix<float> cs(1, 16384, 16384, 8192, 8192, 268435456, 67108864, 0.0f, 2, 2, 0, 1, 0, 8192, mkt::DIST, mkt::DIST);
	
	

	
	struct Negate_shift_partitions_horizontally_matrix_functor{
		
		Negate_shift_partitions_horizontally_matrix_functor(){}
		
		~Negate_shift_partitions_horizontally_matrix_functor() {}
		
		__host__
		auto operator()(int a){
			return -((a));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Negate_shift_partitions_vertically_matrix_functor{
		
		Negate_shift_partitions_vertically_matrix_functor(){}
		
		~Negate_shift_partitions_vertically_matrix_functor() {}
		
		__host__
		auto operator()(int a){
			return -((a));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct DotProduct_map_local_index_in_place_matrix_functor{
		
		DotProduct_map_local_index_in_place_matrix_functor(const mkt::DMatrix<float>& _as, const mkt::DMatrix<float>& _bs) : as(_as), bs(_bs){}
		
		~DotProduct_map_local_index_in_place_matrix_functor() {}
		
		__device__
		auto operator()(int i, int j, float& Cij){
			float sum = 0.0f;
			for(int k = 0; ((k) < 8192); k++){
				sum += (as.get_data_local((i), (k)) * bs.get_data_local((k), (j)));
			}
			Cij = (sum);
		}
	
		void init(int device){
			as.init(device);
			bs.init(device);
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
		mkt::DeviceMatrix<float> as;
		mkt::DeviceMatrix<float> bs;
	};
	struct MinusOne_shift_partitions_horizontally_matrix_functor{
		
		MinusOne_shift_partitions_horizontally_matrix_functor(){}
		
		~MinusOne_shift_partitions_horizontally_matrix_functor() {}
		
		__host__
		auto operator()(int a){
			return -(1);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct MinusOne_shift_partitions_vertically_matrix_functor{
		
		MinusOne_shift_partitions_vertically_matrix_functor(){}
		
		~MinusOne_shift_partitions_vertically_matrix_functor() {}
		
		__host__
		auto operator()(int a){
			return -(1);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Identity_shift_partitions_horizontally_matrix_functor{
		
		Identity_shift_partitions_horizontally_matrix_functor(){}
		
		~Identity_shift_partitions_horizontally_matrix_functor() {}
		
		__host__
		auto operator()(int a){
			return (a);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Identity_shift_partitions_vertically_matrix_functor{
		
		Identity_shift_partitions_vertically_matrix_functor(){}
		
		~Identity_shift_partitions_vertically_matrix_functor() {}
		
		__host__
		auto operator()(int a){
			return (a);
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	struct Square_map_in_place_matrix_functor{
		
		Square_map_in_place_matrix_functor(){}
		
		~Square_map_in_place_matrix_functor() {}
		
		__device__
		auto operator()(float& a){
			a = ((a) * (a));
		}
	
		void init(int device){
		}
		
		size_t get_smem_bytes(){
			size_t result = 0;
			return result;
		}
		
		
	};
	
	
	
	
	template<>
	float mkt::reduce_plus<float>(mkt::DMatrix<float>& a){
		float local_result = 0.0f;
		float global_result = 0.0f;
					
		const int gpu_elements = a.get_size_gpu();
		int threads = gpu_elements < 1024 ? gpu_elements : 1024; // nextPow2
		int blocks = (gpu_elements + threads - 1) / threads;
		cudaSetDevice(0);
		float* d_odata;
		cudaMalloc((void**) &d_odata, blocks * sizeof(float));
		float* devptr = a.get_device_pointer(0);
		
		mkt::kernel::reduce_plus_call<float>(gpu_elements, devptr, d_odata, threads, blocks, mkt::cuda_streams[0], 0);
		
		// fold on gpus: step 2
		while(blocks > 1){
		  int threads_2 = blocks < 1024 ? blocks : 1024; // nextPow2
		  int blocks_2 = (blocks + threads_2 - 1) / threads_2;
		  mkt::kernel::reduce_plus_call<float>(blocks, d_odata, d_odata, threads_2, blocks_2, mkt::cuda_streams[0], 0);
		  blocks = blocks_2;
		}
		
		// copy final sum from device to host
		  cudaMemcpyAsync(&local_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
		  mkt::sync_streams();
		cudaFree(d_odata);
		
		if(a.get_distribution() == mkt::Distribution::DIST){
			MPI_Allreduce(&local_result, &global_result, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			return global_result;
		}else if(a.get_distribution() == mkt::Distribution::COPY){
			return local_result;
		}
	}
	
	template<>
	void mkt::shift_partitions_horizontally<float, Negate_shift_partitions_horizontally_matrix_functor>(mkt::DMatrix<float>& m, Negate_shift_partitions_horizontally_matrix_functor& f){
		int steps = f(m.get_partition_x_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		
		int target = ((((m.get_partition_y_pos() + steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
		int source = ((((m.get_partition_y_pos() - steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
			
		if(target != mpi_rank){
			m.update_self();
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.begin());
			m.update_devices();
		}
	}
	template<>
	void mkt::shift_partitions_horizontally<float, MinusOne_shift_partitions_horizontally_matrix_functor>(mkt::DMatrix<float>& m, MinusOne_shift_partitions_horizontally_matrix_functor& f){
		int steps = f(m.get_partition_x_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		
		int target = ((((m.get_partition_y_pos() + steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
		int source = ((((m.get_partition_y_pos() - steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
			
		if(target != mpi_rank){
			m.update_self();
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.begin());
			m.update_devices();
		}
	}
	template<>
	void mkt::shift_partitions_horizontally<float, Identity_shift_partitions_horizontally_matrix_functor>(mkt::DMatrix<float>& m, Identity_shift_partitions_horizontally_matrix_functor& f){
		int steps = f(m.get_partition_x_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		
		int target = ((((m.get_partition_y_pos() + steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
		int source = ((((m.get_partition_y_pos() - steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
			
		if(target != mpi_rank){
			m.update_self();
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.begin());
			m.update_devices();
		}
	}
	
	template<>
	void mkt::shift_partitions_vertically<float, Negate_shift_partitions_vertically_matrix_functor>(mkt::DMatrix<float>& m, Negate_shift_partitions_vertically_matrix_functor& f){
		int steps = f(m.get_partition_y_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		int partitions_in_column = m.get_partitions_in_column();
		
		int target = ((((m.get_partition_x_pos() + steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		int source = ((((m.get_partition_x_pos() - steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		
		
		if(target != mpi_rank){
			m.update_self();
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.get_data());
			m.update_devices();
		}
	}
	template<>
	void mkt::shift_partitions_vertically<float, MinusOne_shift_partitions_vertically_matrix_functor>(mkt::DMatrix<float>& m, MinusOne_shift_partitions_vertically_matrix_functor& f){
		int steps = f(m.get_partition_y_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		int partitions_in_column = m.get_partitions_in_column();
		
		int target = ((((m.get_partition_x_pos() + steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		int source = ((((m.get_partition_x_pos() - steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		
		
		if(target != mpi_rank){
			m.update_self();
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.get_data());
			m.update_devices();
		}
	}
	template<>
	void mkt::shift_partitions_vertically<float, Identity_shift_partitions_vertically_matrix_functor>(mkt::DMatrix<float>& m, Identity_shift_partitions_vertically_matrix_functor& f){
		int steps = f(m.get_partition_y_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		int partitions_in_column = m.get_partitions_in_column();
		
		int target = ((((m.get_partition_x_pos() + steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		int source = ((((m.get_partition_x_pos() - steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		
		
		if(target != mpi_rank){
			m.update_self();
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.get_data());
			m.update_devices();
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
		
		
		Negate_shift_partitions_horizontally_matrix_functor negate_shift_partitions_horizontally_matrix_functor{};
		Negate_shift_partitions_vertically_matrix_functor negate_shift_partitions_vertically_matrix_functor{};
		DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{as, bs};
		MinusOne_shift_partitions_horizontally_matrix_functor minusOne_shift_partitions_horizontally_matrix_functor{};
		MinusOne_shift_partitions_vertically_matrix_functor minusOne_shift_partitions_vertically_matrix_functor{};
		Identity_shift_partitions_horizontally_matrix_functor identity_shift_partitions_horizontally_matrix_functor{};
		Identity_shift_partitions_vertically_matrix_functor identity_shift_partitions_vertically_matrix_functor{};
		Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(8192, 8192, 16384, MPI_FLOAT, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(float) * 8192, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
			MPI_Datatype bs_partition_type;
			MPI_Type_vector(8192, 8192, 16384, MPI_FLOAT, &bs_partition_type);
			MPI_Type_create_resized(bs_partition_type, 0, sizeof(float) * 8192, &bs_partition_type_resized);
			MPI_Type_free(&bs_partition_type);
			MPI_Type_commit(&bs_partition_type_resized);
			MPI_Datatype cs_partition_type;
			MPI_Type_vector(8192, 8192, 16384, MPI_FLOAT, &cs_partition_type);
			MPI_Type_create_resized(cs_partition_type, 0, sizeof(float) * 8192, &cs_partition_type_resized);
			MPI_Type_free(&cs_partition_type);
			MPI_Type_commit(&cs_partition_type_resized);
		
			
		
		
		mkt::sync_streams();
		mkt::shift_partitions_horizontally<float, Negate_shift_partitions_horizontally_matrix_functor>(as, negate_shift_partitions_horizontally_matrix_functor);
		mkt::shift_partitions_vertically<float, Negate_shift_partitions_vertically_matrix_functor>(bs, negate_shift_partitions_vertically_matrix_functor);
		for(int i = 0; ((i) < 2); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
			mkt::shift_partitions_horizontally<float, MinusOne_shift_partitions_horizontally_matrix_functor>(as, minusOne_shift_partitions_horizontally_matrix_functor);
			mkt::shift_partitions_vertically<float, MinusOne_shift_partitions_vertically_matrix_functor>(bs, minusOne_shift_partitions_vertically_matrix_functor);
		}
		mkt::shift_partitions_horizontally<float, Identity_shift_partitions_horizontally_matrix_functor>(as, identity_shift_partitions_horizontally_matrix_functor);
		mkt::shift_partitions_vertically<float, Identity_shift_partitions_vertically_matrix_functor>(bs, identity_shift_partitions_vertically_matrix_functor);
		mkt::sync_streams();
		mkt::map_in_place<float, Square_map_in_place_matrix_functor>(cs, square_map_in_place_matrix_functor);
		double fn = 0.0;
		fn = mkt::reduce_plus<float>(cs);
		fn = std::sqrt((fn));
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}