#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-16-c-12.hpp"

const size_t number_of_processes = 16;
int process_id = -1;
size_t tmp_size_t = 0;


std::vector<float> as(16777216);
std::vector<float> bs(16777216);
std::vector<float> cs(16777216);


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	int mpi_world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
	
	if(mpi_world_size != number_of_processes){
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	
	if(process_id == 0){
	printf("Run Matmult_float-n-16-c-12\n\n");			
	}
	
	
	
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16777216; ++counter){
		as[counter] = 1.0f;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16777216; ++counter){
		bs[counter] = 0.001f;
	}
	
	#pragma omp parallel for simd
	for(size_t counter = 0; counter  < 16777216; ++counter){
		cs[counter] = 0.0f;
	}
	
	size_t row_offset = 0;size_t col_offset = 0;
	
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 4096;
		break;
	}
	case 2: {
		row_offset = 0;
		col_offset = 8192;
		break;
	}
	case 3: {
		row_offset = 0;
		col_offset = 12288;
		break;
	}
	case 4: {
		row_offset = 4096;
		col_offset = 0;
		break;
	}
	case 5: {
		row_offset = 4096;
		col_offset = 4096;
		break;
	}
	case 6: {
		row_offset = 4096;
		col_offset = 8192;
		break;
	}
	case 7: {
		row_offset = 4096;
		col_offset = 12288;
		break;
	}
	case 8: {
		row_offset = 8192;
		col_offset = 0;
		break;
	}
	case 9: {
		row_offset = 8192;
		col_offset = 4096;
		break;
	}
	case 10: {
		row_offset = 8192;
		col_offset = 8192;
		break;
	}
	case 11: {
		row_offset = 8192;
		col_offset = 12288;
		break;
	}
	case 12: {
		row_offset = 12288;
		col_offset = 0;
		break;
	}
	case 13: {
		row_offset = 12288;
		col_offset = 4096;
		break;
	}
	case 14: {
		row_offset = 12288;
		col_offset = 8192;
		break;
	}
	case 15: {
		row_offset = 12288;
		col_offset = 12288;
		break;
	}
	}		
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4096; ++counter_cols){
			
			as[counter_rows * 4096 + counter_cols] = ((static_cast<float>(((row_offset + counter_rows))) * 4) + ((col_offset + counter_cols)));
		}
	}
	switch(process_id){
	case 0: {
		row_offset = 0;
		col_offset = 0;
		break;
	}
	case 1: {
		row_offset = 0;
		col_offset = 4096;
		break;
	}
	case 2: {
		row_offset = 0;
		col_offset = 8192;
		break;
	}
	case 3: {
		row_offset = 0;
		col_offset = 12288;
		break;
	}
	case 4: {
		row_offset = 4096;
		col_offset = 0;
		break;
	}
	case 5: {
		row_offset = 4096;
		col_offset = 4096;
		break;
	}
	case 6: {
		row_offset = 4096;
		col_offset = 8192;
		break;
	}
	case 7: {
		row_offset = 4096;
		col_offset = 12288;
		break;
	}
	case 8: {
		row_offset = 8192;
		col_offset = 0;
		break;
	}
	case 9: {
		row_offset = 8192;
		col_offset = 4096;
		break;
	}
	case 10: {
		row_offset = 8192;
		col_offset = 8192;
		break;
	}
	case 11: {
		row_offset = 8192;
		col_offset = 12288;
		break;
	}
	case 12: {
		row_offset = 12288;
		col_offset = 0;
		break;
	}
	case 13: {
		row_offset = 12288;
		col_offset = 4096;
		break;
	}
	case 14: {
		row_offset = 12288;
		col_offset = 8192;
		break;
	}
	case 15: {
		row_offset = 12288;
		col_offset = 12288;
		break;
	}
	}		
	#pragma omp parallel for 
	for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
		#pragma omp simd
		for(size_t counter_cols = 0; counter_cols < 4096; ++counter_cols){
			
			bs[counter_rows * 4096 + counter_cols] = ((static_cast<float>(16) + (((row_offset + counter_rows)) * 4)) + ((col_offset + counter_cols)));
		}
	}
	std::chrono::high_resolution_clock::time_point timer_start = std::chrono::high_resolution_clock::now();
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = -((0));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_624 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_624, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_625 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_625, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = -((0));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_626 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_626, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_627 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_627, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = -((0));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_628 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_628, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_629 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_629, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = -((0));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_630 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_630, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_631 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_631, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = -((1));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_632 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_632, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_633 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_633, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = -((1));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_634 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_634, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_635 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_635, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = -((1));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_636 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_636, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_637 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_637, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = -((1));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_638 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_638, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_639 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_639, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = -((2));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_640 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_640, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_641 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_641, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = -((2));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_642 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_642, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_643 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_643, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = -((2));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_644 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_644, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_645 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_645, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = -((2));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_646 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_646, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_647 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_647, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = -((3));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_648 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_648, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_649 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_649, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = -((3));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_650 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_650, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_651 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_651, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = -((3));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_652 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_652, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_653 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_653, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = -((3));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_654 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_654, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_655 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_655, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	}
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = -((0));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_656 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_656, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_657 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_657, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = -((1));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_658 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_658, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_659 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_659, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = -((2));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_660 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_660, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_661 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_661, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = -((3));
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_662 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_662, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_663 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_663, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = -((0));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_664 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_664, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_665 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_665, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = -((1));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_666 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_666, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_667 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_667, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = -((2));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_668 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_668, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_669 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_669, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = -((3));
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_670 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_670, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_671 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_671, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = -((0));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_672 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_672, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_673 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_673, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = -((1));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_674 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_674, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_675 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_675, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = -((2));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_676 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_676, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_677 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_677, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = -((3));
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_678 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_678, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_679 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_679, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = -((0));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_680 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_680, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_681 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_681, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = -((1));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_682 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_682, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_683 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_683, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = -((2));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_684 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_684, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_685 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_685, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = -((3));
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_686 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_686, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_687 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_687, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	}
	for(int i = 0; ((i) < 4); ++i){
		#pragma omp parallel for 
		for(size_t counter_rows = 0; counter_rows < 4096; ++counter_rows){
			#pragma omp simd
			for(size_t counter_cols = 0; counter_cols < 4096; ++counter_cols){
				
				float sum = (cs[counter_rows * 4096 + counter_cols]);
				for(int k = 0; ((k) < 4096); k++){
					sum += ((as)[(counter_rows) * 4096 + (k)] * (bs)[(k) * 4096 + (counter_cols)]);
				}
				cs[counter_rows * 4096 + counter_cols] = (sum);
			}
		}
		switch(process_id){
		case 0:	{
			int shift_source = 0;
			int shift_target = 0;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 0){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_688 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_688, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_689 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_689, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 1:	{
			int shift_source = 1;
			int shift_target = 1;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 1){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_690 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_690, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_691 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_691, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 2:	{
			int shift_source = 2;
			int shift_target = 2;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 2){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_692 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_692, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_693 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_693, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 3:	{
			int shift_source = 3;
			int shift_target = 3;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 0;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 0;
			
			if(shift_target != 3){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_694 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_694, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_695 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_695, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 4:	{
			int shift_source = 4;
			int shift_target = 4;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 4){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_696 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_696, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_697 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_697, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 5:	{
			int shift_source = 5;
			int shift_target = 5;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 5){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_698 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_698, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_699 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_699, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 6:	{
			int shift_source = 6;
			int shift_target = 6;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 6){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_700 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_700, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_701 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_701, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 7:	{
			int shift_source = 7;
			int shift_target = 7;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 4;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 4;
			
			if(shift_target != 7){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_702 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_702, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_703 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_703, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 8:	{
			int shift_source = 8;
			int shift_target = 8;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 8){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_704 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_704, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_705 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_705, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 9:	{
			int shift_source = 9;
			int shift_target = 9;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 9){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_706 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_706, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_707 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_707, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 10:	{
			int shift_source = 10;
			int shift_target = 10;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 10){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_708 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_708, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_709 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_709, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 11:	{
			int shift_source = 11;
			int shift_target = 11;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 8;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 8;
			
			if(shift_target != 11){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_710 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_710, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_711 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_711, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 12:	{
			int shift_source = 12;
			int shift_target = 12;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 12){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_712 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_712, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_713 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_713, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 13:	{
			int shift_source = 13;
			int shift_target = 13;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 13){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_714 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_714, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_715 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_715, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 14:	{
			int shift_source = 14;
			int shift_target = 14;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 14){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_716 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_716, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_717 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_717, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		case 15:	{
			int shift_source = 15;
			int shift_target = 15;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 12;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 12;
			
			if(shift_target != 15){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_718 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_718, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_719 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_719, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
			}
			break;
		}
		}
		switch(process_id){
		case 0:	{
			int shift_source = 0;
			int shift_target = 0;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 0){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_720 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_720, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_721 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_721, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 1:	{
			int shift_source = 1;
			int shift_target = 1;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 1){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_722 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_722, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_723 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_723, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 2:	{
			int shift_source = 2;
			int shift_target = 2;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 2){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_724 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_724, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_725 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_725, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 3:	{
			int shift_source = 3;
			int shift_target = 3;
			int 
			shift_steps = -(1);
			
			shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 3){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_726 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_726, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_727 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_727, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 4:	{
			int shift_source = 4;
			int shift_target = 4;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 4){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_728 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_728, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_729 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_729, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 5:	{
			int shift_source = 5;
			int shift_target = 5;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 5){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_730 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_730, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_731 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_731, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 6:	{
			int shift_source = 6;
			int shift_target = 6;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 6){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_732 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_732, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_733 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_733, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 7:	{
			int shift_source = 7;
			int shift_target = 7;
			int 
			shift_steps = -(1);
			
			shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 7){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_734 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_734, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_735 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_735, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 8:	{
			int shift_source = 8;
			int shift_target = 8;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 8){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_736 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_736, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_737 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_737, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 9:	{
			int shift_source = 9;
			int shift_target = 9;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 9){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_738 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_738, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_739 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_739, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 10:	{
			int shift_source = 10;
			int shift_target = 10;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 10){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_740 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_740, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_741 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_741, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 11:	{
			int shift_source = 11;
			int shift_target = 11;
			int 
			shift_steps = -(1);
			
			shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 11){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_742 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_742, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_743 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_743, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 12:	{
			int shift_source = 12;
			int shift_target = 12;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		
			if(shift_target != 12){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_744 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_744, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_745 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_745, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 13:	{
			int shift_source = 13;
			int shift_target = 13;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		
			if(shift_target != 13){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_746 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_746, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_747 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_747, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 14:	{
			int shift_source = 14;
			int shift_target = 14;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		
			if(shift_target != 14){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_748 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_748, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_749 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_749, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		case 15:	{
			int shift_source = 15;
			int shift_target = 15;
			int 
			shift_steps = -(1);
			
			shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
			shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		
			if(shift_target != 15){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_750 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_750, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_751 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_751, MPI_COMM_WORLD, &requests[0]);
				MPI_Waitall(2, requests, statuses);
				
				std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
			}
			break;
		}
		}
	}
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = (0);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_752 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_752, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_753 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_753, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = (0);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_754 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_754, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_755 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_755, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = (0);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_756 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_756, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_757 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_757, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = (0);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 0;
		
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_758 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_758, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_759 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_759, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = (1);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_760 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_760, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_761 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_761, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = (1);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_762 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_762, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_763 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_763, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = (1);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_764 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_764, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_765 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_765, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = (1);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 4;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 4;
		
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_766 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_766, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_767 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_767, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = (2);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_768 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_768, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_769 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_769, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = (2);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_770 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_770, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_771 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_771, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = (2);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_772 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_772, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_773 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_773, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = (2);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 8;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 8;
		
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_774 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_774, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_775 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_775, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = (3);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_776 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_776, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_777 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_777, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = (3);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_778 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_778, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_779 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_779, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = (3);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_780 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_780, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_781 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_781, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = (3);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) + 12;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) + 12;
		
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_782 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_782, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_783 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_783, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), as.begin());
		}
		break;
	}
	}
	switch(process_id){
	case 0:	{
		int shift_source = 0;
		int shift_target = 0;
		int 
		shift_steps = (0);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 0){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_784 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_784, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_785 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_785, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 1:	{
		int shift_source = 1;
		int shift_target = 1;
		int 
		shift_steps = (1);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 1){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_786 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_786, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_787 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_787, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 2:	{
		int shift_source = 2;
		int shift_target = 2;
		int 
		shift_steps = (2);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 2){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_788 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_788, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_789 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_789, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 3:	{
		int shift_source = 3;
		int shift_target = 3;
		int 
		shift_steps = (3);
		
		shift_target = ((((0 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((0 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 3){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_790 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_790, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_791 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_791, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 4:	{
		int shift_source = 4;
		int shift_target = 4;
		int 
		shift_steps = (0);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 4){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_792 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_792, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_793 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_793, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 5:	{
		int shift_source = 5;
		int shift_target = 5;
		int 
		shift_steps = (1);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 5){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_794 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_794, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_795 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_795, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 6:	{
		int shift_source = 6;
		int shift_target = 6;
		int 
		shift_steps = (2);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 6){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_796 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_796, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_797 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_797, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 7:	{
		int shift_source = 7;
		int shift_target = 7;
		int 
		shift_steps = (3);
		
		shift_target = ((((1 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((1 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 7){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_798 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_798, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_799 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_799, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 8:	{
		int shift_source = 8;
		int shift_target = 8;
		int 
		shift_steps = (0);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 8){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_800 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_800, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_801 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_801, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 9:	{
		int shift_source = 9;
		int shift_target = 9;
		int 
		shift_steps = (1);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 9){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_802 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_802, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_803 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_803, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 10:	{
		int shift_source = 10;
		int shift_target = 10;
		int 
		shift_steps = (2);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 10){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_804 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_804, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_805 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_805, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 11:	{
		int shift_source = 11;
		int shift_target = 11;
		int 
		shift_steps = (3);
		
		shift_target = ((((2 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((2 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 11){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_806 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_806, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_807 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_807, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 12:	{
		int shift_source = 12;
		int shift_target = 12;
		int 
		shift_steps = (0);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 0;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 0;
	
		if(shift_target != 12){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_808 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_808, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_809 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_809, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 13:	{
		int shift_source = 13;
		int shift_target = 13;
		int 
		shift_steps = (1);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 1;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 1;
	
		if(shift_target != 13){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_810 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_810, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_811 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_811, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 14:	{
		int shift_source = 14;
		int shift_target = 14;
		int 
		shift_steps = (2);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 2;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 2;
	
		if(shift_target != 14){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_812 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_812, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_813 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_813, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	case 15:	{
		int shift_source = 15;
		int shift_target = 15;
		int 
		shift_steps = (3);
		
		shift_target = ((((3 + shift_steps) % 4) + 4 ) % 4) * 4 + 3;
		shift_source = ((((3 - shift_steps) % 4) + 4 ) % 4) * 4 + 3;
	
		if(shift_target != 15){
			MPI_Request requests[2];
			MPI_Status statuses[2];
			auto tmp_shift_buffer = std::make_unique<std::vector<float>>(16777216);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_814 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_814, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_815 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_815, MPI_COMM_WORLD, &requests[0]);
			MPI_Waitall(2, requests, statuses);
			
			std::move(tmp_shift_buffer->begin(), tmp_shift_buffer->end(), bs.begin());		
		}
		break;
	}
	}
	std::chrono::high_resolution_clock::time_point timer_end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(timer_end - timer_start).count();
	
	if(process_id == 0){
	printf("Execution time: %.5fs\n", seconds);
	printf("Threads: %i\n", omp_get_max_threads());
	printf("Processes: %i\n", mpi_world_size);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
