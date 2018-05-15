#include <mpi.h>

#include <omp.h>
#include <array>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <memory>
#include "../include/matmult_float-n-16-c-24.hpp"

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
	printf("Run Matmult_float-n-16-c-24\n\n");			
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
			int tag_1008 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1008, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1009 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1009, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1010 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1010, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1011 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1011, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1012 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1012, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1013 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1013, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1014 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1014, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1015 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1015, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1016 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1016, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1017 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1017, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1018 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1018, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1019 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1019, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1020 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1020, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1021 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1021, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1022 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1022, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1023 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1023, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1024 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1024, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1025 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1025, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1026 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1026, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1027 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1027, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1028 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1028, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1029 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1029, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1030 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1030, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1031 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1031, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1032 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1032, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1033 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1033, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1034 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1034, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1035 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1035, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1036 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1036, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1037 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1037, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1038 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1038, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1039 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1039, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1040 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1040, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1041 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1041, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1042 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1042, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1043 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1043, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1044 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1044, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1045 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1045, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1046 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1046, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1047 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1047, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1048 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1048, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1049 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1049, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1050 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1050, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1051 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1051, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1052 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1052, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1053 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1053, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1054 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1054, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1055 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1055, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1056 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1056, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1057 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1057, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1058 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1058, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1059 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1059, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1060 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1060, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1061 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1061, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1062 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1062, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1063 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1063, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1064 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1064, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1065 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1065, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1066 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1066, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1067 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1067, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1068 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1068, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1069 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1069, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1070 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1070, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1071 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1071, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1072 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1072, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1073 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1073, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1074 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1074, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1075 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1075, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1076 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1076, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1077 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1077, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1078 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1078, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1079 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1079, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1080 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1080, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1081 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1081, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1082 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1082, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1083 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1083, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1084 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1084, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1085 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1085, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1086 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1086, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1087 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1087, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1088 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1088, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1089 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1089, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1090 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1090, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1091 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1091, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1092 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1092, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1093 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1093, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1094 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1094, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1095 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1095, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1096 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1096, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1097 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1097, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1098 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1098, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1099 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1099, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1100 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1100, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1101 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1101, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1102 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1102, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1103 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1103, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1104 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1104, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1105 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1105, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1106 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1106, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1107 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1107, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1108 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1108, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1109 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1109, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1110 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1110, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1111 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1111, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1112 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1112, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1113 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1113, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1114 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1114, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1115 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1115, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1116 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1116, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1117 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1117, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1118 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1118, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1119 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1119, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1120 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1120, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1121 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1121, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1122 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1122, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1123 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1123, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1124 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1124, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1125 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1125, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1126 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1126, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1127 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1127, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1128 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1128, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1129 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1129, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1130 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1130, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1131 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1131, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1132 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1132, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1133 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1133, MPI_COMM_WORLD, &requests[0]);
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
				int tag_1134 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
				MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1134, MPI_COMM_WORLD, &requests[1]);
				tmp_size_t = 16777216 * sizeof(float);
				int tag_1135 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
				MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1135, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1136 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1136, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1137 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1137, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1138 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1138, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1139 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1139, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1140 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1140, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1141 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1141, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1142 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1142, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1143 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1143, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1144 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1144, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1145 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1145, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1146 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1146, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1147 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1147, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1148 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1148, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1149 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1149, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1150 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1150, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1151 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1151, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1152 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1152, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1153 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1153, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1154 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1154, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1155 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1155, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1156 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1156, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1157 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1157, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1158 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1158, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1159 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1159, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1160 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1160, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1161 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1161, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1162 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1162, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1163 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1163, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1164 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1164, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1165 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1165, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1166 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1166, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1167 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(as.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1167, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1168 = ((shift_source + 0) * (shift_source + 0 + 1)) / 2 + 0;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1168, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1169 = ((0 + shift_target) * (0 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1169, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1170 = ((shift_source + 1) * (shift_source + 1 + 1)) / 2 + 1;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1170, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1171 = ((1 + shift_target) * (1 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1171, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1172 = ((shift_source + 2) * (shift_source + 2 + 1)) / 2 + 2;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1172, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1173 = ((2 + shift_target) * (2 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1173, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1174 = ((shift_source + 3) * (shift_source + 3 + 1)) / 2 + 3;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1174, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1175 = ((3 + shift_target) * (3 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1175, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1176 = ((shift_source + 4) * (shift_source + 4 + 1)) / 2 + 4;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1176, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1177 = ((4 + shift_target) * (4 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1177, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1178 = ((shift_source + 5) * (shift_source + 5 + 1)) / 2 + 5;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1178, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1179 = ((5 + shift_target) * (5 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1179, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1180 = ((shift_source + 6) * (shift_source + 6 + 1)) / 2 + 6;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1180, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1181 = ((6 + shift_target) * (6 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1181, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1182 = ((shift_source + 7) * (shift_source + 7 + 1)) / 2 + 7;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1182, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1183 = ((7 + shift_target) * (7 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1183, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1184 = ((shift_source + 8) * (shift_source + 8 + 1)) / 2 + 8;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1184, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1185 = ((8 + shift_target) * (8 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1185, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1186 = ((shift_source + 9) * (shift_source + 9 + 1)) / 2 + 9;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1186, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1187 = ((9 + shift_target) * (9 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1187, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1188 = ((shift_source + 10) * (shift_source + 10 + 1)) / 2 + 10;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1188, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1189 = ((10 + shift_target) * (10 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1189, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1190 = ((shift_source + 11) * (shift_source + 11 + 1)) / 2 + 11;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1190, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1191 = ((11 + shift_target) * (11 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1191, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1192 = ((shift_source + 12) * (shift_source + 12 + 1)) / 2 + 12;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1192, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1193 = ((12 + shift_target) * (12 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1193, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1194 = ((shift_source + 13) * (shift_source + 13 + 1)) / 2 + 13;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1194, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1195 = ((13 + shift_target) * (13 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1195, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1196 = ((shift_source + 14) * (shift_source + 14 + 1)) / 2 + 14;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1196, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1197 = ((14 + shift_target) * (14 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1197, MPI_COMM_WORLD, &requests[0]);
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
			int tag_1198 = ((shift_source + 15) * (shift_source + 15 + 1)) / 2 + 15;
			MPI_Irecv(tmp_shift_buffer->data(), tmp_size_t, MPI_BYTE, shift_source, tag_1198, MPI_COMM_WORLD, &requests[1]);
			tmp_size_t = 16777216 * sizeof(float);
			int tag_1199 = ((15 + shift_target) * (15 + shift_target + 1)) / 2 + shift_target;
			MPI_Isend(bs.data(), tmp_size_t, MPI_BYTE, shift_target, tag_1199, MPI_COMM_WORLD, &requests[0]);
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
