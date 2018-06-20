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
#include "../include/matmult_float-valgrind-n-4-c-24_3.hpp"

const size_t number_of_processes = 4;
const size_t process_id = 3;
int mpi_rank = -1;
int mpi_world_size = 0;

size_t tmp_size_t = 0;


std::vector<float> as(16384, 1.0f);
std::vector<float> bs(16384, 0.001f);
std::vector<float> cs(16384, 0.0f);




struct InitA_functor{
	auto operator()(int a, int b, float x) const{
		return ((static_cast<float>((a)) * 4) + (b));
	}
};
struct InitB_functor{
	auto operator()(int a, int b, float x) const{
		return ((static_cast<float>(16) + ((a) * 4)) + (b));
	}
};
struct Negate_functor{
	auto operator()(int a) const{
		return -((a));
	}
};
struct Identity_functor{
	auto operator()(int a) const{
		return (a);
	}
};
struct MinusOne_functor{
	auto operator()(int a) const{
		return -(1);
	}
};
struct DotProduct_functor{
	auto operator()(int i, int j, float Cij) const{
		float sum = (Cij);
		for(int k = 0; ((k) < 128); k++){
			sum += ((as)[(i) * 128 + (k)] * (bs)[(k) * 128 + (j)]);
		}
		return (sum);
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
	
	
	
	InitA_functor initA_functor{};
	InitB_functor initB_functor{};
	Negate_functor negate_functor{};
	Identity_functor identity_functor{};
	MinusOne_functor minusOne_functor{};
	DotProduct_functor dotProduct_functor{};
	
	
	
	
	
	
	MPI_Datatype as_partition_type, as_partition_type_resized;
	MPI_Type_vector(128, 128, 256, MPI_FLOAT, &as_partition_type);
	MPI_Type_create_resized(as_partition_type, 0, sizeof(float) * 128, &as_partition_type_resized);
	MPI_Type_free(&as_partition_type);
	MPI_Type_commit(&as_partition_type_resized);
	MPI_Datatype bs_partition_type, bs_partition_type_resized;
	MPI_Type_vector(128, 128, 256, MPI_FLOAT, &bs_partition_type);
	MPI_Type_create_resized(bs_partition_type, 0, sizeof(float) * 128, &bs_partition_type_resized);
	MPI_Type_free(&bs_partition_type);
	MPI_Type_commit(&bs_partition_type_resized);
	MPI_Datatype cs_partition_type, cs_partition_type_resized;
	MPI_Type_vector(128, 128, 256, MPI_FLOAT, &cs_partition_type);
	MPI_Type_create_resized(cs_partition_type, 0, sizeof(float) * 128, &cs_partition_type_resized);
	MPI_Type_free(&cs_partition_type);
	MPI_Type_commit(&cs_partition_type_resized);

	
	int shift_source = 3;
	int shift_target = 3;
	int shift_steps = 0;
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	row_offset = 128;
	col_offset = 128;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 128; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 128; ++counter_cols){
			size_t counter = counter_rows * 128 + counter_cols;
			as[counter] = initA_functor(row_offset + counter_rows, col_offset + counter_cols, as[counter]);
		}
	}
	row_offset = 128;
	col_offset = 128;
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 128; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 128; ++counter_cols){
			size_t counter = counter_rows * 128 + counter_cols;
			bs[counter] = initB_functor(row_offset + counter_rows, col_offset + counter_cols, bs[counter]);
		}
	}
	shift_steps = negate_functor(1);
	
	shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 2;
	shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 2;
	
	if(shift_target != 3){
		MPI_Request requests[2];
		MPI_Status statuses[2];
		auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16384);
		tmp_size_t = 16384 * sizeof(float);
		int tag_36 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
		MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_FLOAT, shift_source, tag_36, MPI_COMM_WORLD, &requests[1]);
		int tag_37 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
		MPI_Isend(as.data(), 16384, MPI_FLOAT, shift_target, tag_37, MPI_COMM_WORLD, &requests[0]);
		MPI_Waitall(2, requests, statuses);
		
		std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
	}
	shift_steps = negate_functor(1);
	shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
	if(shift_target != 3){
		MPI_Request requests[2];
		MPI_Status statuses[2];
		auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16384);
		tmp_size_t = 16384 * sizeof(float);
		int tag_38 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
		MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_FLOAT, shift_source, tag_38, MPI_COMM_WORLD, &requests[1]);
		int tag_39 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
		MPI_Isend(bs.data(), 16384, MPI_FLOAT, shift_target, tag_39, MPI_COMM_WORLD, &requests[0]);
		MPI_Waitall(2, requests, statuses);
		
		std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
	}
	for(int i = 0; ((i) < 2); ++i){
		#pragma omp parallel for 
		for(size_t counter_rows = 0; counter_rows < 128; ++counter_rows){
			#pragma omp simd
			for(size_t counter_cols = 0; counter_cols < 128; ++counter_cols){
				size_t counter = counter_rows * 128 + counter_cols;
				cs[counter] = dotProduct_functor(counter_rows, counter_cols, cs[counter]);
			}
		}
		shift_steps = minusOne_functor(1);
		
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 2;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 2;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16384);
			tmp_size_t = 16384 * sizeof(float);
			int tag_40 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_FLOAT, shift_source, tag_40, MPI_COMM_WORLD, &requests[1]);
			int tag_41 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), 16384, MPI_FLOAT, shift_target, tag_41, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		shift_steps = minusOne_functor(1);
		shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16384);
			tmp_size_t = 16384 * sizeof(float);
			int tag_42 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_FLOAT, shift_source, tag_42, MPI_COMM_WORLD, &requests[1]);
			int tag_43 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), 16384, MPI_FLOAT, shift_target, tag_43, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
	}
	shift_steps = identity_functor(1);
	
	shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) + 2;
	shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) + 2;
	
	if(shift_target != 3){
		MPI_Request requests[2];
		MPI_Status statuses[2];
		auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16384);
		tmp_size_t = 16384 * sizeof(float);
		int tag_44 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
		MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_FLOAT, shift_source, tag_44, MPI_COMM_WORLD, &requests[1]);
		int tag_45 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
		MPI_Isend(as.data(), 16384, MPI_FLOAT, shift_target, tag_45, MPI_COMM_WORLD, &requests[0]);
		MPI_Waitall(2, requests, statuses);
		
		std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
	}
	shift_steps = identity_functor(1);
	shift_target = ((((1 + shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	shift_source = ((((1 - shift_steps) % 2) + 2 ) % 2) * 2 + 1;
	
	if(shift_target != 3){
		MPI_Request requests[2];
		MPI_Status statuses[2];
		auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16384);
		tmp_size_t = 16384 * sizeof(float);
		int tag_46 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
		MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_FLOAT, shift_source, tag_46, MPI_COMM_WORLD, &requests[1]);
		int tag_47 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
		MPI_Isend(bs.data(), 16384, MPI_FLOAT, shift_target, tag_47, MPI_COMM_WORLD, &requests[0]);
		MPI_Waitall(2, requests, statuses);
		
		std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
	}
	
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}