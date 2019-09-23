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
	#include "../include/matmult-n-16-g-2_2.hpp"
	
	const size_t number_of_processes = 16;
	const size_t process_id = 2;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	
			
	const int dim = 16384;
	
	

	
	struct Negate_shift_partitions_horizontally_matrix_functor{
		
		Negate_shift_partitions_horizontally_matrix_functor(){
		}
		
		~Negate_shift_partitions_horizontally_matrix_functor() {}
		
		auto operator()(int a){
			return -((a));
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
	struct Negate_shift_partitions_vertically_matrix_functor{
		
		Negate_shift_partitions_vertically_matrix_functor(){
		}
		
		~Negate_shift_partitions_vertically_matrix_functor() {}
		
		auto operator()(int a){
			return -((a));
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
	struct DotProduct_map_local_index_in_place_matrix_functor{
		
		DotProduct_map_local_index_in_place_matrix_functor(const mkt::DMatrix<float>& _as, const mkt::DMatrix<float>& _bs) : as(_as), bs(_bs){
		}
		
		~DotProduct_map_local_index_in_place_matrix_functor() {}
		
		auto operator()(int i, int j, float Cij){
			float sum = 0.0f;
			for(int k = 0; ((k) < 4096); k++){
				sum += (as.get_data_local((i), (k)) * bs.get_data_local((k), (j)));
			}
			Cij += (sum);
			return (Cij);
		}
	
		void init(int gpu){
			as.init(gpu);
			bs.init(gpu);
		}
		
		void set_id(int gang, int worker, int vector){
			_gang = gang;
			_worker = worker;
			_vector = vector;
		}
		
		
		mkt::DeviceMatrix<float> as;
		mkt::DeviceMatrix<float> bs;
		
		
		int _gang;
		int _worker;
		int _vector;
	};
	struct MinusOne_shift_partitions_horizontally_matrix_functor{
		
		MinusOne_shift_partitions_horizontally_matrix_functor(){
		}
		
		~MinusOne_shift_partitions_horizontally_matrix_functor() {}
		
		auto operator()(int a){
			return -(1);
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
	struct MinusOne_shift_partitions_vertically_matrix_functor{
		
		MinusOne_shift_partitions_vertically_matrix_functor(){
		}
		
		~MinusOne_shift_partitions_vertically_matrix_functor() {}
		
		auto operator()(int a){
			return -(1);
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
	struct Identity_shift_partitions_horizontally_matrix_functor{
		
		Identity_shift_partitions_horizontally_matrix_functor(){
		}
		
		~Identity_shift_partitions_horizontally_matrix_functor() {}
		
		auto operator()(int a){
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
	struct Identity_shift_partitions_vertically_matrix_functor{
		
		Identity_shift_partitions_vertically_matrix_functor(){
		}
		
		~Identity_shift_partitions_vertically_matrix_functor() {}
		
		auto operator()(int a){
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
	struct Square_map_in_place_matrix_functor{
		
		Square_map_in_place_matrix_functor(){
		}
		
		~Square_map_in_place_matrix_functor() {}
		
		auto operator()(float a){
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
	float mkt::reduce_plus<float>(mkt::DMatrix<float>& a){
		float local_result = 0.0f;
		float global_result = 0.0f;
		
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			#pragma omp parallel for reduction(+:local_result)
			for(int gpu = 0; gpu < 2; ++gpu){
				acc_set_device_num(gpu, acc_device_not_host);
				float* devptr = a.get_device_pointer(gpu);
				const int gpu_elements = a.get_size_gpu();
				float gpu_result = 0.0f;
				
				#pragma acc parallel loop deviceptr(devptr) present_or_copy(gpu_result) reduction(+:gpu_result) async(0)
				for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
					#pragma acc cache(gpu_result)
					gpu_result = gpu_result + devptr[counter];
				}
				acc_wait(0);
				local_result = local_result + gpu_result;
			}
		}else if(a.get_device_distribution() == mkt::Distribution::COPY){
			acc_set_device_num(0, acc_device_not_host);
			float* devptr = a.get_device_pointer(0);
			const int gpu_elements = a.get_size_gpu();
			
			#pragma acc parallel loop deviceptr(devptr) present_or_copy(local_result) reduction(+:local_result) async(0)
			for(unsigned int counter = 0; counter < gpu_elements; ++counter) {
				#pragma acc cache(local_result)
				local_result = local_result + devptr[counter];
			}
			acc_wait(0);
		}
		
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
		
		
		
		
		mkt::wait_all();
	
		mkt::DMatrix<float> as(2, 16384, 16384, 4096, 4096, 268435456, 16777216, 1.0f, 4, 4, 0, 2, 0, 8192, mkt::DIST, mkt::DIST);
		mkt::DMatrix<float> bs(2, 16384, 16384, 4096, 4096, 268435456, 16777216, 0.001f, 4, 4, 0, 2, 0, 8192, mkt::DIST, mkt::COPY);
		mkt::DMatrix<float> cs(2, 16384, 16384, 4096, 4096, 268435456, 16777216, 0.0f, 4, 4, 0, 2, 0, 8192, mkt::DIST, mkt::DIST);
		
		Negate_shift_partitions_horizontally_matrix_functor negate_shift_partitions_horizontally_matrix_functor{};
		Negate_shift_partitions_vertically_matrix_functor negate_shift_partitions_vertically_matrix_functor{};
		DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{as, bs};
		MinusOne_shift_partitions_horizontally_matrix_functor minusOne_shift_partitions_horizontally_matrix_functor{};
		MinusOne_shift_partitions_vertically_matrix_functor minusOne_shift_partitions_vertically_matrix_functor{};
		Identity_shift_partitions_horizontally_matrix_functor identity_shift_partitions_horizontally_matrix_functor{};
		Identity_shift_partitions_vertically_matrix_functor identity_shift_partitions_vertically_matrix_functor{};
		Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(4096, 4096, 16384, MPI_FLOAT, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(float) * 4096, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
			MPI_Datatype bs_partition_type;
			MPI_Type_vector(4096, 4096, 16384, MPI_FLOAT, &bs_partition_type);
			MPI_Type_create_resized(bs_partition_type, 0, sizeof(float) * 4096, &bs_partition_type_resized);
			MPI_Type_free(&bs_partition_type);
			MPI_Type_commit(&bs_partition_type_resized);
			MPI_Datatype cs_partition_type;
			MPI_Type_vector(4096, 4096, 16384, MPI_FLOAT, &cs_partition_type);
			MPI_Type_create_resized(cs_partition_type, 0, sizeof(float) * 4096, &cs_partition_type_resized);
			MPI_Type_free(&cs_partition_type);
			MPI_Type_commit(&cs_partition_type_resized);
		
			
		
		
		for(int gpu = 0; gpu < 2; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		mkt::shift_partitions_horizontally<float, Negate_shift_partitions_horizontally_matrix_functor>(as, negate_shift_partitions_horizontally_matrix_functor);
		mkt::shift_partitions_vertically<float, Negate_shift_partitions_vertically_matrix_functor>(bs, negate_shift_partitions_vertically_matrix_functor);
		for(int i = 0; ((i) < 4); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
			mkt::shift_partitions_horizontally<float, MinusOne_shift_partitions_horizontally_matrix_functor>(as, minusOne_shift_partitions_horizontally_matrix_functor);
			mkt::shift_partitions_vertically<float, MinusOne_shift_partitions_vertically_matrix_functor>(bs, minusOne_shift_partitions_vertically_matrix_functor);
		}
		mkt::shift_partitions_horizontally<float, Identity_shift_partitions_horizontally_matrix_functor>(as, identity_shift_partitions_horizontally_matrix_functor);
		mkt::shift_partitions_vertically<float, Identity_shift_partitions_vertically_matrix_functor>(bs, identity_shift_partitions_vertically_matrix_functor);
		for(int gpu = 0; gpu < 2; ++gpu){
			acc_set_device_num(gpu, acc_device_not_host);
			acc_wait_all();
		}
		mkt::map_in_place<float, Square_map_in_place_matrix_functor>(cs, square_map_in_place_matrix_functor);
		double fn = 0.0;
		fn = mkt::reduce_plus<float>(cs);
		fn = std::sqrt((fn));
		
		mkt::wait_all();
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
