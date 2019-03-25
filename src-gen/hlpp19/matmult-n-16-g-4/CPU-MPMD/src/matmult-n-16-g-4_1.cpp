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
	#include "../include/matmult-n-16-g-4_1.hpp"
	
	const size_t number_of_processes = 16;
	const size_t process_id = 1;
	int mpi_rank = -1;
	int mpi_world_size = 0;
	
	

	
	const int dim = 8192;
	mkt::DMatrix<float> as(1, 8192, 8192, 2048, 2048, 67108864, 4194304, 1.0f, 4, 4, 0, 1, 0, 2048, mkt::DIST);
	mkt::DMatrix<float> bs(1, 8192, 8192, 2048, 2048, 67108864, 4194304, 0.001f, 4, 4, 0, 1, 0, 2048, mkt::DIST);
	mkt::DMatrix<float> cs(1, 8192, 8192, 2048, 2048, 67108864, 4194304, 0.0f, 4, 4, 0, 1, 0, 2048, mkt::DIST);
	
	

	
	struct Negate_shift_partitions_horizontally_matrix_functor{
		auto operator()(int a) const{
			return -((a));
		}
		
	};
	struct Negate_shift_partitions_vertically_matrix_functor{
		auto operator()(int a) const{
			return -((a));
		}
		
	};
	struct DotProduct_map_local_index_in_place_matrix_functor{
		auto operator()(int i, int j, float& Cij) const{
			for(int k = 0; ((k) < 2048); k++){
				Cij += (as[(i) * 2048 + (k)] * bs[(k) * 2048 + (j)]);
			}
		}
		
	};
	struct MinusOne_shift_partitions_horizontally_matrix_functor{
		auto operator()(int a) const{
			return -(1);
		}
		
	};
	struct MinusOne_shift_partitions_vertically_matrix_functor{
		auto operator()(int a) const{
			return -(1);
		}
		
	};
	struct Identity_shift_partitions_horizontally_matrix_functor{
		auto operator()(int a) const{
			return (a);
		}
		
	};
	struct Identity_shift_partitions_vertically_matrix_functor{
		auto operator()(int a) const{
			return (a);
		}
		
	};
	struct Square_map_in_place_matrix_functor{
		auto operator()(float& a) const{
			a = ((a) * (a));
		}
		
	};
	
	
	
	
	template<>
	void mkt::shift_partitions_horizontally<float, Negate_shift_partitions_horizontally_matrix_functor>(mkt::DMatrix<float>& m, const Negate_shift_partitions_horizontally_matrix_functor& f){
		int steps = f(m.get_partition_x_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		
		int target = ((((m.get_partition_y_pos() + steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
		int source = ((((m.get_partition_y_pos() - steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
			
		if(target != mpi_rank){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.begin());
		}
	}
	template<>
	void mkt::shift_partitions_horizontally<float, MinusOne_shift_partitions_horizontally_matrix_functor>(mkt::DMatrix<float>& m, const MinusOne_shift_partitions_horizontally_matrix_functor& f){
		int steps = f(m.get_partition_x_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		
		int target = ((((m.get_partition_y_pos() + steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
		int source = ((((m.get_partition_y_pos() - steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
			
		if(target != mpi_rank){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.begin());
		}
	}
	template<>
	void mkt::shift_partitions_horizontally<float, Identity_shift_partitions_horizontally_matrix_functor>(mkt::DMatrix<float>& m, const Identity_shift_partitions_horizontally_matrix_functor& f){
		int steps = f(m.get_partition_x_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		
		int target = ((((m.get_partition_y_pos() + steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
		int source = ((((m.get_partition_y_pos() - steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
			
		if(target != mpi_rank){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.begin());
		}
	}
	
	template<>
	void mkt::shift_partitions_vertically<float, Negate_shift_partitions_vertically_matrix_functor>(mkt::DMatrix<float>& m, const Negate_shift_partitions_vertically_matrix_functor& f){
		int steps = f(m.get_partition_y_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		int partitions_in_column = m.get_partitions_in_column();
		
		int target = ((((m.get_partition_x_pos() + steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		int source = ((((m.get_partition_x_pos() - steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		
		
		if(target != mpi_rank){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.get_data());
		}
	}
	template<>
	void mkt::shift_partitions_vertically<float, MinusOne_shift_partitions_vertically_matrix_functor>(mkt::DMatrix<float>& m, const MinusOne_shift_partitions_vertically_matrix_functor& f){
		int steps = f(m.get_partition_y_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		int partitions_in_column = m.get_partitions_in_column();
		
		int target = ((((m.get_partition_x_pos() + steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		int source = ((((m.get_partition_x_pos() - steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		
		
		if(target != mpi_rank){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.get_data());
		}
	}
	template<>
	void mkt::shift_partitions_vertically<float, Identity_shift_partitions_vertically_matrix_functor>(mkt::DMatrix<float>& m, const Identity_shift_partitions_vertically_matrix_functor& f){
		int steps = f(m.get_partition_y_pos());
		
		int partitions_in_row = m.get_partitions_in_row();
		int partitions_in_column = m.get_partitions_in_column();
		
		int target = ((((m.get_partition_x_pos() + steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		int source = ((((m.get_partition_x_pos() - steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
		
		
		if(target != mpi_rank){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto buffer = std::make_unique<std::vector<float>>(m.get_size_local());
	
			int tag_rec = ((source + mpi_rank) * (source + mpi_rank + 1)) / 2 + mpi_rank;
			int tag_send = ((mpi_rank + target) * (mpi_rank + target + 1)) / 2 + target;
			
			MPI_Irecv(buffer->data(), m.get_size_local(), MPI_FLOAT, source, tag_rec, MPI_COMM_WORLD, &requests[1]);
			MPI_Isend(m.get_data(), m.get_size_local(), MPI_FLOAT, target, tag_send, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(buffer->begin(), buffer->end(), m.get_data());
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
		
		
		
				Negate_shift_partitions_horizontally_matrix_functor negate_shift_partitions_horizontally_matrix_functor{};
				Negate_shift_partitions_vertically_matrix_functor negate_shift_partitions_vertically_matrix_functor{};
				DotProduct_map_local_index_in_place_matrix_functor dotProduct_map_local_index_in_place_matrix_functor{};
				MinusOne_shift_partitions_horizontally_matrix_functor minusOne_shift_partitions_horizontally_matrix_functor{};
				MinusOne_shift_partitions_vertically_matrix_functor minusOne_shift_partitions_vertically_matrix_functor{};
				Identity_shift_partitions_horizontally_matrix_functor identity_shift_partitions_horizontally_matrix_functor{};
				Identity_shift_partitions_vertically_matrix_functor identity_shift_partitions_vertically_matrix_functor{};
				Square_map_in_place_matrix_functor square_map_in_place_matrix_functor{};
		
		
		
				
			
			
			MPI_Datatype as_partition_type;
			MPI_Type_vector(2048, 2048, 8192, MPI_FLOAT, &as_partition_type);
			MPI_Type_create_resized(as_partition_type, 0, sizeof(float) * 2048, &as_partition_type_resized);
			MPI_Type_free(&as_partition_type);
			MPI_Type_commit(&as_partition_type_resized);
			MPI_Datatype bs_partition_type;
			MPI_Type_vector(2048, 2048, 8192, MPI_FLOAT, &bs_partition_type);
			MPI_Type_create_resized(bs_partition_type, 0, sizeof(float) * 2048, &bs_partition_type_resized);
			MPI_Type_free(&bs_partition_type);
			MPI_Type_commit(&bs_partition_type_resized);
			MPI_Datatype cs_partition_type;
			MPI_Type_vector(2048, 2048, 8192, MPI_FLOAT, &cs_partition_type);
			MPI_Type_create_resized(cs_partition_type, 0, sizeof(float) * 2048, &cs_partition_type_resized);
			MPI_Type_free(&cs_partition_type);
			MPI_Type_commit(&cs_partition_type_resized);
		
			
		
		
		mkt::shift_partitions_horizontally<float, Negate_shift_partitions_horizontally_matrix_functor>(as, negate_shift_partitions_horizontally_matrix_functor);
		mkt::shift_partitions_vertically<float, Negate_shift_partitions_vertically_matrix_functor>(bs, negate_shift_partitions_vertically_matrix_functor);
		for(int i = 0; ((i) < 4); ++i){
			mkt::map_local_index_in_place<float, DotProduct_map_local_index_in_place_matrix_functor>(cs, dotProduct_map_local_index_in_place_matrix_functor);
			mkt::shift_partitions_horizontally<float, MinusOne_shift_partitions_horizontally_matrix_functor>(as, minusOne_shift_partitions_horizontally_matrix_functor);
			mkt::shift_partitions_vertically<float, MinusOne_shift_partitions_vertically_matrix_functor>(bs, minusOne_shift_partitions_vertically_matrix_functor);
		}
		mkt::shift_partitions_horizontally<float, Identity_shift_partitions_horizontally_matrix_functor>(as, identity_shift_partitions_horizontally_matrix_functor);
		mkt::shift_partitions_vertically<float, Identity_shift_partitions_vertically_matrix_functor>(bs, identity_shift_partitions_vertically_matrix_functor);
		mkt::map_in_place<float, Square_map_in_place_matrix_functor>(cs, square_map_in_place_matrix_functor);
		double fn = 0.0;
		// TODO: SkeletonGenerator.generateSkeletonExpression: default case
		fn = std::sqrt((fn));
		
		
		MPI_Finalize();
		return EXIT_SUCCESS;
		}
